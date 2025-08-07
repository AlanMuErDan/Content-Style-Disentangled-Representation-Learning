# inference/VAE_inference.py

import sys
import os
import torch
import lmdb
import yaml
import io
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.font_dataset import get_all_lmdb_keys
from models import build_encoder
from utils.save_ckpt import build_state_dict



CONFIG_PATH   = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/configs/config.yaml"
CKPT_PATH     = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/checkpoints/20250723_121344_super_vae_16*16*4_seed10086/best_ckpt.pth"
SRC_LMDB_PATH = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/lmdb_data"
OUT_LMDB_PATH = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/lmdb_latent"
CHAR_LIST_PATH = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/intersection_chars.txt"
CACHE_PATH = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/lmdb_keys.json"
BATCH_SIZE    = 64



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
    with open(char_file_path, 'r', encoding='utf-8') as f:
        chars = set(line.strip() for line in f if line.strip())
    return chars



def save_latents_to_lmdb(encoder, keys, src_lmdb_path, tgt_lmdb_path, device, img_size=128, batch_size=64):
    env_src = lmdb.open(src_lmdb_path, readonly=True, lock=False, readahead=False)
    env_tgt = lmdb.open(tgt_lmdb_path, map_size=1024**4)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    with env_src.begin() as txn_src, env_tgt.begin(write=True) as txn_tgt:
        for i in tqdm(range(0, len(keys), batch_size), desc="Encoding"):
            batch_keys = keys[i:i + batch_size]
            imgs = []
            for key in batch_keys:
                value = txn_src.get(to_bytes(key))
                if value is None:
                    continue  # Skip missing entries
                img = decode_lmdb_image(value)
                imgs.append(transform(img))
            if not imgs:
                continue
            imgs_tensor = torch.stack(imgs).to(device)

            with torch.no_grad():
                mu, logvar = encoder(imgs_tensor)
                z = mu  # Use mu as the latent representation

            for key, latent in zip(batch_keys, z):
                txn_tgt.put(to_bytes(key), pickle.dumps(latent.cpu().numpy()))



def main():
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder = build_encoder(
        name=config["encoder"],
        img_size=config["img_size"],
        latent_channels=config.get("latent_chanels", 4)
    ).to(device)

    encoder = load_checkpoint(encoder, CKPT_PATH, device)
    print("Encoder loaded from:", CKPT_PATH)

    allowed_chars = load_char_list(CHAR_LIST_PATH)
    print(f"Loaded {len(allowed_chars)} allowed characters from: {CHAR_LIST_PATH}")

    all_keys = get_all_lmdb_keys(SRC_LMDB_PATH, CACHE_PATH)
    filtered_keys = [
        k.decode("utf-8") for k in all_keys 
        if b'+' in k and k.split(b'+')[1].decode("utf-8") in allowed_chars
    ]
    print(f"Filtered {len(filtered_keys)} valid keys from {len(all_keys)} total keys.")

    save_latents_to_lmdb(
        encoder=encoder,
        keys=filtered_keys,
        src_lmdb_path=SRC_LMDB_PATH,
        tgt_lmdb_path=OUT_LMDB_PATH,
        device=device,
        img_size=config["img_size"],
        batch_size=BATCH_SIZE
    )

    print("All latents saved to:", OUT_LMDB_PATH)


if __name__ == "__main__":
    main()