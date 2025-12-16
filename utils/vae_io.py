# utils/vae_io.py

import os, io, yaml, torch
from typing import Tuple, Union, Optional
from PIL import Image
from torchvision import transforms
from models import build_encoder, build_decoder



def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["vae"] if "vae" in cfg else cfg


def _to_pil(img_or_tensor: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(img_or_tensor, Image.Image):
        return img_or_tensor.convert("L")
    if img_or_tensor.dim() == 3 and img_or_tensor.size(0) in {1, 3}:
        return transforms.ToPILImage()(img_or_tensor.cpu()).convert("L")
    raise TypeError("Input must be PIL.Image or 3-D torch.Tensor.")


def load_models(
    config_path: str,
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, dict, torch.device]:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(config_path)

    encoder = build_encoder(
        name=cfg["encoder"],
        img_size=cfg["img_size"],
        latent_channels=cfg.get("latent_channels", 4),
    ).to(device).eval()

    decoder = build_decoder(
        name=cfg["decoder"],
        img_size=cfg["img_size"],
        latent_channels=cfg.get("latent_channels", 4),
    ).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"], strict=False)
    decoder.load_state_dict(ckpt["decoder"], strict=False)

    return encoder, decoder, cfg, device


def encode(
    img: Union[Image.Image, torch.Tensor],
    encoder: torch.nn.Module,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:

    pil = _to_pil(img)

    preprocess = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),          # → [1,H,W], float32 in [0,1]
    ])

    x = preprocess(pil).unsqueeze(0).to(device)  # [1,1,H,W]
    with torch.no_grad():
        mu, _ = encoder(x)
    return mu.squeeze(0).cpu()                   # [C,H,W]


def decode(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    cfg: dict,
    device: torch.device,
) -> Image.Image:

    if latent.dim() != 3:
        raise ValueError("Latent must have shape [C,H,W].")
    z = latent.unsqueeze(0).to(device)           # [1,C,H,W]
    with torch.no_grad():
        recon = decoder(z).squeeze(0).cpu()      # [1,H,W]
    # recon = (recon + 1) / 2             # [-1, 1] --> [0, 1]
    recon = torch.clamp(recon, 0, 1)
    return transforms.ToPILImage()(recon)

if __name__ == "__main__":
    import argparse, json, os, io, lmdb, pickle, random, binascii
    import numpy as np
    import torch
    from torchvision import transforms
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser("Single-key VAE sanity checks: latent->decode vs img->encode->decode")
    parser.add_argument("--config")
    parser.add_argument("--ckpt", required=True, help="VAE checkpoint (.pth)")
    parser.add_argument("--latent_lmdb")
    parser.add_argument("--image_lmdb")
    parser.add_argument("--keys_json")
    parser.add_argument("--pick-source", choices=["latent", "image", "json", "common"], default="latent",
                        )
    parser.add_argument("--key", default=None,
                        )
    parser.add_argument("--key-hex", default=None,
                        )
    parser.add_argument("--outdir", default="vae_onekey_vis")
    parser.add_argument("--latent-shape", nargs=3, type=int, default=[4,16,16], metavar=("C","H","W"))
    parser.add_argument("--latent-dtype", default="float32", choices=["float16","float32","float64"])
    parser.add_argument("--stats-yaml", default=None, help="per-channel mean/std yaml (for optional latent denorm)")
    parser.add_argument("--denorm-latent", action="store_true", help="apply z = z*std + mean before decode (requires --stats-yaml)")
    parser.add_argument("--use-train-normalize", action="store_true", help="apply Normalize((0.5,),(0.5,)) before encoder (if training used it)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)


    encoder, decoder, cfg, device = load_models(args.config, args.ckpt, None)
    encoder.eval(); decoder.eval()


    C, H, W = args.latent_shape
    np_dtype = np.dtype(args.latent_dtype)

    def robust_load_latent(raw_bytes: bytes) -> np.ndarray:
        expect = C * H * W * np_dtype.itemsize
        if len(raw_bytes) == expect:
            return np.frombuffer(raw_bytes, dtype=np_dtype).reshape(C, H, W)
        obj = None
        try:
            obj = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=False)
        except Exception:
            obj = None
        if obj is None:
            try:
                obj = pickle.loads(raw_bytes)
            except Exception as e:
                raise RuntimeError(f"Cannot decode latent: {e}")
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        elif isinstance(obj, np.ndarray):
            arr = obj
        else:
            raise RuntimeError(f"Unsupported latent type: {type(obj)}")
        if arr.size != C * H * W:
            raise ValueError(f"Latent size {arr.shape} ≠ expected {(C, H, W)}")
        return arr.astype(np_dtype, copy=False).reshape(C, H, W)

    def list_keys(env) -> list:
        with env.begin(buffers=True) as txn:
            return [bytes(k) for k, _ in txn.cursor()]

    def pretty_key(kb: bytes) -> str:
        try:
            return kb.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes:{binascii.hexlify(kb).decode()[:32]}...>"


    env_lat = lmdb.open(args.latent_lmdb, readonly=True, lock=False, readahead=False)
    env_img = lmdb.open(args.image_lmdb,  readonly=True, lock=False, readahead=False)

    if args.key is not None and args.key_hex is not None:
        raise ValueError("Use only one of --key or --key-hex, not both.")

    if args.key_hex is not None:
        key_bytes = binascii.unhexlify(args.key_hex.strip())
        print(f"[Key] Using HEX key: {args.key_hex}  ({pretty_key(key_bytes)})")
    elif args.key is not None:

        try:
            key_bytes = args.key.encode("utf-8")
            print(f"[Key] Using UTF-8 key: {args.key}")
        except Exception:
            try:
                key_bytes = binascii.unhexlify(args.key.strip())
                print(f"[Key] Using key (auto HEX): {args.key}")
            except Exception as e:
                raise ValueError(f"--key neither valid UTF-8 nor HEX: {e}")
    else:

        if args.pick_source == "latent":
            keys_lat = list_keys(env_lat)
            if not keys_lat:
                raise RuntimeError("No entries in latent LMDB.")
            key_bytes = random.choice(keys_lat)
            print(f"[Key] Picked from latent LMDB: {pretty_key(key_bytes)}")
        elif args.pick_source == "image":
            keys_img = list_keys(env_img)
            if not keys_img:
                raise RuntimeError("No entries in image LMDB.")
            key_bytes = random.choice(keys_img)
            print(f"[Key] Picked from image LMDB: {pretty_key(key_bytes)}")
        elif args.pick_source == "json":
            with open(args.keys_json, "r", encoding="utf-8") as f:
                keys = json.load(f)
            if not keys:
                raise RuntimeError(f"No keys in {args.keys_json}")
            key_bytes = random.choice(keys).encode("utf-8", errors="ignore")
            print(f"[Key] Picked from JSON: {pretty_key(key_bytes)}")
        else:  # "common"
            keys_lat = set(list_keys(env_lat))
            keys_img = set(list_keys(env_img))
            common = list(keys_lat & keys_img)
            if not common:
                raise RuntimeError("No common keys between image and latent LMDBs.")
            key_bytes = random.choice(common)
            print(f"[Key] Picked from COMMON: {pretty_key(key_bytes)}")


    mean = std = None
    if args.denorm_latent:
        if not args.stats_yaml:
            raise ValueError("--denorm-latent requires --stats-yaml")
        with open(args.stats_yaml, "r") as f:
            stats = yaml.safe_load(f)
        mean = torch.tensor(stats["mean"]).view(C, 1, 1)
        std  = torch.tensor(stats["std"]).view(C, 1, 1)
        print(f"[Stats] mean={stats['mean']}, std={stats['std']}")


    with env_lat.begin(buffers=True) as txn:
        raw = txn.get(key_bytes)
        if raw is None:
            raise KeyError(f"Key not found in latent LMDB: {pretty_key(key_bytes)}")
        arr = robust_load_latent(bytes(raw))         # (C,H,W) numpy
        z = torch.from_numpy(arr).float()            # float32 for decoder
        if args.denorm_latent:
            z = (z * std) + mean                     # [C,H,W]
        img_latent_decode = decode(z, decoder, cfg, device)  # PIL
        path_latent = os.path.join(args.outdir, "A_latent_decode.png")
        os.makedirs(args.outdir, exist_ok=True)
        img_latent_decode.save(path_latent)
        print(f"[Saved] {path_latent}")


    with env_img.begin(buffers=True) as txn:
        raw = txn.get(key_bytes)
        if raw is None:
            raise KeyError(f"Key not found in image LMDB: {pretty_key(key_bytes)}")
        pil = Image.open(io.BytesIO(bytes(raw))).convert("L")

    img_size = cfg["img_size"]
    if args.use_train_normalize:
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),                  # [0,1]
            transforms.Normalize((0.5,), (0.5,)),   # → [-1,1]
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),                  # [0,1]
        ])

    x = preprocess(pil).unsqueeze(0).to(device)     # [1,1,H,W]
    with torch.no_grad():
        enc_out = encoder(x)
        mu = enc_out[0] if (isinstance(enc_out, (tuple, list)) and len(enc_out) >= 1) else enc_out
        recon = decoder(mu).squeeze(0).cpu().clamp(0, 1)  # [1,H,W]

    orig = transforms.ToTensor()(pil.resize((img_size, img_size)))  # [1,H,W], [0,1]
    cat = torch.cat([orig, recon], dim=-1)  
    path_pair  = os.path.join(args.outdir, "B_img_encode_decode_pair.png")
    path_orig  = os.path.join(args.outdir, "B_orig.png")
    path_recon = os.path.join(args.outdir, "B_recon.png")
    save_image(cat, path_pair)
    transforms.ToPILImage()(orig ).save(path_orig)
    transforms.ToPILImage()(recon).save(path_recon)
    print(f"[Saved] {path_pair}")
    print(f"[Saved] {path_orig}")
    print(f"[Saved] {path_recon}")

  