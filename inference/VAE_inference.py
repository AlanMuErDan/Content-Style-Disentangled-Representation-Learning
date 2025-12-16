# inference/VAE_inference.py

import sys
import os
import io
import pickle
import yaml
import lmdb
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.font_dataset import get_all_lmdb_keys
from models import build_encoder
from utils.save_ckpt import build_state_dict  # kept for compatibility (unused but imported previously)


CONFIG_PATH     = CONFIG_PATH
CKPT_PATH       = CKPT_PATH
SRC_LMDB_PATH   = SRC_LMDB_PATH
OUT_LMDB_PATH   = OUT_LMDB_PATH
OUT_PT_PATH     = OUT_PT_PATH
CHAR_LIST_PATH  = CHAR_LIST_PATH
CACHE_PATH      = CACHE_PATH

BATCH_SIZE      = 64
OUTPUT_MODE     = "pt"   # "lmdb" or "pt" (can be overridden by --mode)


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg['vae'] if 'vae' in cfg else cfg

def load_checkpoint(encoder, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    return encoder

def to_bytes(k):
    return k if isinstance(k, bytes) else k.encode("utf-8")

def decode_lmdb_image(value_bytes):
    buffer = io.BytesIO(value_bytes)
    image = Image.open(buffer).convert("L")
    return image

def load_char_list(char_file_path):
    chars = []
    seen = set()
    with open(char_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            ch = line.strip()
            if not ch:
                continue
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    return chars  # ordered list

def parse_key_bytes(key_b):
    if b'+' not in key_b:
        return None, None
    font_b, char_b = key_b.split(b'+', 1)
    try:
        font = font_b.decode('utf-8')
        char = char_b.decode('utf-8')
        return font, char
    except Exception:
        return None, None

def build_ordered_tuples(all_keys_bytes, char_list):
    char_set = set(char_list)
    fonts = set()
    key_map = {}  # (font, char) -> key_bytes

    for kb in all_keys_bytes:
        font, ch = parse_key_bytes(kb)
        if font is None or ch is None:
            continue
        if ch in char_set:
            fonts.add(font)
            key_map[(font, ch)] = kb  

    fonts_sorted = sorted(list(fonts))
    ordered = []
    missing = []

    for font in fonts_sorted:
        for ch in char_list:
            kb = key_map.get((font, ch), None)
            if kb is None:
                missing.append((font, ch))
            else:
                ordered.append((font, ch, kb))

    return fonts_sorted, ordered, missing


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0,1] -> [-1,1]
    ])

def encode_in_batches(encoder, env_src, ordered_tuples, device, img_size, batch_size):

    transform = build_transform(img_size)
    latents = []
    total = len(ordered_tuples)
    idx_global = 0

    with env_src.begin() as txn_src:
        for i in tqdm(range(0, total, batch_size), desc="Encoding"):
            batch = ordered_tuples[i:i + batch_size]
            imgs = []
            valid_meta = []  
            for (font, ch, kb) in batch:
                value = txn_src.get(kb)
                if value is None:
                    print(f"[WARN] Missing LMDB entry: font={font}, char={ch}")
                    continue
                img = decode_lmdb_image(value)
                imgs.append(transform(img))
                valid_meta.append((font, ch, kb))

            if not imgs:
                continue

            imgs_tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                mu, logvar = encoder(imgs_tensor)
                z = mu  # shape: [B, C, H, W]

            z = z.cpu()
            for j, (font, ch, kb) in enumerate(valid_meta):
                z_item = z[j]  # (C,H,W)
                latents.append(z_item)
                print(f"[{idx_global:07d}] font={font} char={ch}")
                idx_global += 1

    return latents  # list of (C,H,W) tensors on CPU

def save_latents_to_lmdb(encoder, ordered_tuples, src_lmdb_path, tgt_lmdb_path, device, img_size=128, batch_size=64):
    env_src = lmdb.open(src_lmdb_path, readonly=True, lock=False, readahead=False)
    env_tgt = lmdb.open(tgt_lmdb_path, map_size=1024**4)

    transform = build_transform(img_size)

    with env_src.begin() as txn_src, env_tgt.begin(write=True) as txn_tgt:
        idx_global = 0
        total = len(ordered_tuples)
        for i in tqdm(range(0, total, batch_size), desc="Encoding", mininterval=2.0, miniters=1000, smoothing=0.05, disable=not sys.stderr.isatty()):
            batch = ordered_tuples[i:i + batch_size]
            imgs, metas = [], []
            for (font, ch, kb) in batch:
                value = txn_src.get(kb)
                if value is None:
                    print(f"[WARN] Missing LMDB entry: font={font}, char={ch}")
                    continue
                img = decode_lmdb_image(value)
                imgs.append(transform(img))
                metas.append((font, ch, kb))

            if not imgs:
                continue

            imgs_tensor = torch.stack(imgs).to(device)
            with torch.no_grad():
                mu, logvar = encoder(imgs_tensor)
                z = mu  # [B, C, H, W]

            z = z.cpu()
            for (font, ch, kb), latent in zip(metas, z):
                txn_tgt.put(kb, pickle.dumps(latent.numpy()))
                # print(f"[{idx_global:07d}] font={font} char={ch}")
                idx_global += 1

def save_latents_to_pt(encoder, ordered_tuples, src_lmdb_path, out_pt_path, device, img_size=128, batch_size=64):
    env_src = lmdb.open(src_lmdb_path, readonly=True, lock=False, readahead=False)


    latents_list = encode_in_batches(
        encoder=encoder,
        env_src=env_src,
        ordered_tuples=ordered_tuples,
        device=device,
        img_size=img_size,
        batch_size=batch_size
    )

    if len(latents_list) == 0:
        raise RuntimeError("No latents encoded. Please check LMDB source and character list.")

    C, H, W = latents_list[0].shape  # (C,H,W)
    N = len(latents_list)
    out_tensor = torch.empty((N, H, W, C), dtype=latents_list[0].dtype)

    for i, zchw in enumerate(latents_list):
        if tuple(zchw.shape) != (C, H, W):
            raise RuntimeError(f"Shape mismatch at index {i}: expected {(C,H,W)}, got {tuple(zchw.shape)}")
        out_tensor[i] = zchw.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)


    torch.save({"latents": out_tensor}, out_pt_path)
    print(f"Saved PT tensor to: {out_pt_path} | shape={tuple(out_tensor.shape)}")

def save_latents_to_pt_sharded(
    encoder, ordered_tuples, src_lmdb_path, out_pt_dir, device,
    img_size=128, batch_size=64, shard_size=50000, dtype="float16",
    meta_path=None, log_every=2000
):

    import numpy as np
    os.makedirs(out_pt_dir, exist_ok=True)

    env_src = lmdb.open(src_lmdb_path, readonly=True, lock=False, readahead=False)
    transform = build_transform(img_size)


    torch_dtype = dict(float16=torch.float16, float32=torch.float32)["float16" if dtype == "float16" else "float32"]
    np_dtype    = dict(float16=np.float16,  float32=np.float32)["float16" if dtype == "float16" else "float32"]

    shard_idx, in_shard_count = 0, 0
    buf_np = None 
    H = W = C = None

    def flush_shard():
        nonlocal shard_idx, in_shard_count, buf_np
        if in_shard_count == 0:
            return
  
        arr = buf_np[:in_shard_count]                               # (m,H,W,C)
        tens = torch.from_numpy(arr)                                # zero-copy 到 torch
        out_path = os.path.join(out_pt_dir, f"latents_{shard_idx:04d}.pt")
        torch.save({"latents": tens}, out_path)
        print(f"[Shard {shard_idx:04d}] saved: shape={tuple(tens.shape)} -> {out_path}")
        shard_idx += 1
        in_shard_count = 0

    with env_src.begin() as txn_src:
        total = len(ordered_tuples)
        idx_global = 0

        for i in tqdm(range(0, total, batch_size), desc="Encoding", mininterval=2.0, miniters=500, smoothing=0.1):
            batch = ordered_tuples[i:i + batch_size]
            imgs, metas = [], []
            for (font, ch, kb) in batch:
                val = txn_src.get(kb)
                if val is None:
                    continue
                img = decode_lmdb_image(val)
                imgs.append(transform(img))
                metas.append((font, ch, kb))
            if not imgs:
                continue

            x = torch.stack(imgs, 0).to(device)
            with torch.no_grad():
                mu, logvar = encoder(x)     # [B,C,H,W]
                z = mu

          
            z_hwC = z.permute(0, 2, 3, 1).contiguous()              # (B,H,W,C)
            if torch_dtype == torch.float16:
                z_hwC = z_hwC.to(torch.float16)
            z_np = z_hwC.cpu().numpy()                              # (B,H,W,C) np.float16/32

            B, h, w, c = z_np.shape
            if buf_np is None:
                H, W, C = h, w, c
                buf_np = np.empty((shard_size, H, W, C), dtype=np_dtype)

   
            start = 0
            while start < B:
                room = shard_size - in_shard_count
                take = min(room, B - start)
                buf_np[in_shard_count:in_shard_count + take] = z_np[start:start + take]
                in_shard_count += take
                start += take
                if in_shard_count == shard_size:
                    flush_shard()

            idx_global += len(metas)
            if log_every and (idx_global % log_every == 0):
                tqdm.write(f"[{idx_global}] encoded")

          
            del z, z_hwC, z_np, x, imgs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

   
    flush_shard()

   
    if meta_path:
        fonts_sorted = sorted({f for (f, _, _) in ordered_tuples})
        chars_in_order = load_char_list(CHAR_LIST_PATH)
        torch.save({"fonts_sorted": fonts_sorted, "chars": chars_in_order}, meta_path)
        print(f"[Meta] saved to {meta_path}")

def merge_pt_shards_to_single_pt(shard_dir, out_pt_path, dtype="float16",
                                 tmp_dir=None, flush_every_gb=4):

    import os, glob, numpy as np, torch, tqdm, io

    shard_files = sorted(glob.glob(os.path.join(shard_dir, "latents_*.pt")))
    assert shard_files, f"No shards found in {shard_dir}"


    total = 0
    H = W = C = None
    for f in tqdm.tqdm(shard_files, desc="Scanning shards", unit="file"):
        t = torch.load(f, map_location="cpu")
        lat = t["latents"]                       # (m,H,W,C)
        if H is None:
            _, H, W, C = lat.shape
        else:
            assert lat.shape[1:] == (H,W,C), f"shape mismatch in {f}"
        total += lat.shape[0]

    np_dtype = np.float16 if dtype == "float16" else np.float32

  
    if tmp_dir is None:
        tmp_dir = shard_dir
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_bin = os.path.join(tmp_dir, "__concat_tmp__.bin")

    big = np.memmap(tmp_bin, mode="w+", dtype=np_dtype, shape=(total, H, W, C))
   
    fh = open(tmp_bin, "r+")

    off = 0
    bytes_since_flush = 0
    flush_bytes = int(flush_every_gb * (1024**3))

    for f in tqdm.tqdm(shard_files, desc="Merging shards", unit="file"):
        t = torch.load(f, map_location="cpu")
        arr_t = t["latents"]                     # torch.Tensor (m,H,W,C), CPU
      
        if dtype == "float16" and arr_t.dtype != torch.float16:
            arr_t = arr_t.to(torch.float16)
        elif dtype == "float32" and arr_t.dtype != torch.float32:
            arr_t = arr_t.to(torch.float32)

        arr = arr_t.numpy()                      # (m,H,W,C) -> numpy 
        m = arr.shape[0]
        big[off:off+m] = arr
        off += m

        bytes_since_flush += arr.nbytes
        if bytes_since_flush >= flush_bytes:
            big.flush()                          
            try:
                os.fsync(fh.fileno())           
            except Exception:
                pass
            bytes_since_flush = 0


    big.flush()
    try:
        os.fsync(fh.fileno())
    except Exception:
        pass
    fh.close()


    big_t = torch.from_numpy(big)               # zero-copy wrapper
    torch.save({"latents": big_t}, out_pt_path)

    
    del big, big_t
    try:
        os.remove(tmp_bin)
    except OSError:
        pass

    print(f"[Merged] {len(shard_files)} shards -> {out_pt_path} | shape={(total,H,W,C)}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=OUTPUT_MODE, choices=["lmdb", "pt"], help="output mode")
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--ckpt", type=str, default=CKPT_PATH)
    parser.add_argument("--src_lmdb", type=str, default=SRC_LMDB_PATH)
    parser.add_argument("--out_lmdb", type=str, default=OUT_LMDB_PATH)
    parser.add_argument("--out_pt", type=str, default=OUT_PT_PATH)
    parser.add_argument("--chars", type=str, default=CHAR_LIST_PATH)
    parser.add_argument("--cache", type=str, default=CACHE_PATH)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    latent_channels = config.get("latent_channels", config.get("latent_chanels", 4))
    encoder = build_encoder(
        name=config["encoder"],
        img_size=config["img_size"],
        latent_channels=latent_channels
    ).to(device)

    encoder = load_checkpoint(encoder, args.ckpt, device)
    print("Encoder loaded from:", args.ckpt)

    char_list = load_char_list(args.chars)
    print(f"Loaded {len(char_list)} characters (ordered) from: {args.chars}")

    #  (font, char, key_bytes) 
    all_keys = get_all_lmdb_keys(args.src_lmdb, args.cache)
    fonts_sorted, ordered_tuples, missing = build_ordered_tuples(all_keys, char_list)

    print(f"Discovered {len(fonts_sorted)} fonts.")
    if missing:
        print(f"[WARN] Missing {len(missing)} (font,char) pairs in LMDB (these will be skipped).")

    print(f"Total valid pairs to encode: {len(ordered_tuples)} (order: font-major, then char order from file)")

    if args.mode == "lmdb":
        save_latents_to_lmdb(
            encoder=encoder,
            ordered_tuples=ordered_tuples,
            src_lmdb_path=args.src_lmdb,
            tgt_lmdb_path=args.out_lmdb,
            device=device,
            img_size=config["img_size"],
            batch_size=args.batch_size
        )
        print("All latents saved to LMDB:", args.out_lmdb)

    elif args.mode == "pt":
        out_dir = os.path.splitext(args.out_pt)[0] + "_shards"   # /.../font_latents_v2_shards/
        meta_p  = os.path.join(out_dir, "meta.pt")
        save_latents_to_pt_sharded(
            encoder=encoder,
            ordered_tuples=ordered_tuples,
            src_lmdb_path=args.src_lmdb,
            out_pt_dir=out_dir,
            device=device,
            img_size=config["img_size"],
            batch_size=args.batch_size,
            shard_size=50000,              
            dtype="float16",               
            meta_path=meta_p,
            log_every=5000
        )
        print("All latents saved to shard dir:", out_dir)
    
    # === NEW: final summary ===
    print(f"SUMMARY: total_chars={len(char_list)}, total_fonts={len(fonts_sorted)}, "
          f"encoded_pairs={len(ordered_tuples)}, missing_pairs={len(missing)}")

if __name__ == "__main__":
    main()