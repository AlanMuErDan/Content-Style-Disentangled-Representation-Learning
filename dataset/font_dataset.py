# /dataset/font_dataset.py

import os
import io
import lmdb
import glob
import random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
import json 
from typing import Tuple, Dict, List, Optional
import pickle
import yaml
from pathlib import Path
import math 
import lmdb



# open LMDB dataset safely
def _open_lmdb(path: str) -> lmdb.Environment:
    return lmdb.open(
        path,
        readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=2048
    )

def latent_worker_init(worker_id: int):
    FourWayFontPairLatentLMDBDataset._ENV = None



def get_all_fonts(lmdb_path, cache_path="font_list.json"):
    if os.path.exists(cache_path):
        print(f"Loading cached font list from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    print(f"Opened LMDB at {lmdb_path}")
    font_names = set()
    with env.begin() as txn:
        for key, _ in tqdm(txn.cursor(), desc="Reading keys", unit="key"):
            try:
                font = key.decode().split("+")[0]
                font_names.add(font)
            except Exception:
                continue
    env.close()
    font_list = sorted(list(font_names))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(font_list, f, ensure_ascii=False, indent=2)

    return font_list



def get_all_lmdb_keys(lmdb_path, cache_path="lmdb_keys.json", max_keys=None):
    if os.path.exists(cache_path):
        print(f"Loading cached keys from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            keys = json.load(f)
        return [key.encode("utf-8") for key in keys]

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    print(f"Scanning keys from {lmdb_path}")
    keys = []
    with env.begin() as txn:
        for i, (key, _) in enumerate(tqdm(txn.cursor(), desc="Reading LMDB keys", unit="key")):
            keys.append(key)
            if max_keys is not None and len(keys) >= max_keys:
                break

    str_keys = [k.decode("utf-8") for k in keys]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(str_keys, f, ensure_ascii=False, indent=2)

    return keys



def augment_image(img: Image.Image) -> Image.Image:
    """
    randomly scales and shifting 
    """
    scale = random.uniform(0.5, 1.0)  # randonly scale between 0.5 and 1.0 
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    scaled = img.resize((new_w, new_h), resample=Image.BILINEAR) 

    half_w_floor = new_w // 2
    half_h_floor = new_h // 2
    half_w_ceil = new_w - half_w_floor
    half_h_ceil = new_h - half_h_floor
    cx_min = half_w_floor
    cx_max = img.width - half_w_ceil
    cy_min = half_h_floor
    cy_max = img.height - half_h_ceil
    center_x = random.randint(cx_min, cx_max) # pick a random center x
    center_y = random.randint(cy_min, cy_max) # pick a random center y 

    left = center_x - half_w_floor
    top = center_y - half_h_floor

    background = Image.new('L', (img.width, img.height), color=255) # a white background 
    background.paste(scaled, (left, top)) # paste the scaled image onto the background 

    return background



class SingleFontLMDBDataset(Dataset):
    """
    Used for stage 1 VAE training
    """
    def __init__(self, lmdb_path, img_size=128, keys_subset=None, augment_prob=0.5):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.augment_prob = augment_prob

        if keys_subset is not None: # used for train-test split 
            self.keys = keys_subset
        else:
            with self.env.begin() as txn: 
                self.keys = [key for key, _ in txn.cursor()]
            
        if len(self.keys) == 0:
            raise RuntimeError(f"No data found in LMDB: {lmdb_path}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # [0, 1] pixels --> [-1, 1] for VAE training, as TanH gives us [-1, 1] 
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        try:
            with self.env.begin() as txn:
                key = self.keys[idx]
                buffer = txn.get(key)
                img = Image.open(io.BytesIO(buffer)).convert("L")  # convert to grey image
                if random.random() < self.augment_prob:
                    img = augment_image(img)
                return self.transform(img)
        except (UnidentifiedImageError, OSError, TypeError):
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)



class FourWayFontPairLatentLMDBDataset(Dataset):
    """
    Used for stage 2 disentangle DDPM training
    """
    def __init__(self,
                 lmdb_path: str,
                 latent_shape: Tuple[int, int, int] = (4, 16, 16),
                 max_retry: int = 1000,
                 pair_num: int = 1000,
                 stats_yaml: str = None):

        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.latent_shape = latent_shape
        self.max_retry = max_retry       # max number of trial to get the paired data 
        self.pair_num = pair_num         # total number of pairs in the dataset 
        self.data = {}                   # font: {char: bytes_key}
        with self.env.begin() as txn:
            for k_bytes, _ in tqdm(txn.cursor(), desc="Scanning latent LMDB keys", unit="key"):
                try:
                    k_str = k_bytes.decode("utf-8")
                    font, char = k_str.split("+", 1)
                    self.data.setdefault(font, {})[char] = k_bytes
                except Exception:
                    continue
        self.fonts = list(self.data.keys())
        self.common_chars = list(
            set.intersection(*(set(chars.keys()) for chars in self.data.values()))
        )

        print(f"Found {len(self.fonts)} fonts and {len(self.common_chars)} common chars.")
        assert len(self.fonts) >= 2 and len(self.common_chars) >= 2, \
            "Too few fonts or shared characters in latent LMDB"

        if stats_yaml:   
            """
            mean: [-0.5021770477, 0.9270092909, 0.6116915592, 0.4392606947]
            std:  [0.2178721980, 0.1884942846, 0.1611157692, 0.6110689706]
            """
            with open(stats_yaml, "r") as f:
                stats = yaml.safe_load(f)
            self.mean = torch.tensor(stats["mean"]).view(-1,1,1)
            self.std  = torch.tensor(stats["std"]).view(-1,1,1)
        else:
            self.mean = self.std = None

    def __len__(self):
        return int(self.pair_num)  # decide as you wish 

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            for _ in range(self.max_retry):
                try:
                    f_a, f_b = random.sample(self.fonts, 2)        # sampling 2 fonts 
                    c_a, c_b = random.sample(self.common_chars, 2) # sampling 2 characters 

                    k_fa_ca = self.data[f_a][c_a]
                    k_fa_cb = self.data[f_a][c_b]
                    k_fb_ca = self.data[f_b][c_a]
                    k_fb_cb = self.data[f_b][c_b]

                    z_fa_ca = self._load_latent(txn, k_fa_ca)
                    z_fa_cb = self._load_latent(txn, k_fa_cb)
                    z_fb_ca = self._load_latent(txn, k_fb_ca)
                    z_fb_cb = self._load_latent(txn, k_fb_cb)

                    return {
                        "F_A+C_A": z_fa_ca,
                        "F_A+C_B": z_fa_cb,
                        "F_B+C_A": z_fb_ca,
                        "F_B+C_B": z_fb_cb,
                        "font_a": f_a, "font_b": f_b,
                        "char_a": c_a, "char_b": c_b,
                    }

                except Exception as e:
                    print(f"latent load fail: {e}, retry…")
                    continue

            raise RuntimeError("Unable to sample a valid 4-way latent pair after 100 attempts.")

    def _load_latent(self, txn, k_bytes):
        buf = txn.get(k_bytes)
        arr = pickle.loads(buf)         # to numpy array 
        latent = torch.from_numpy(arr)  # to tensor 

        if not isinstance(latent, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(latent)}")

        if latent.shape != self.latent_shape:
            raise ValueError(f"Expected shape {self.latent_shape}, got {latent.shape}")

        if self.mean is not None:
            latent = (latent - self.mean) / self.std     # normalization 

        return latent
    
    def denorm(self, z: torch.Tensor) -> torch.Tensor:
        if self.mean is None:         
            return z
        return z * self.std.to(z.device) + self.mean.to(z.device)




def split_fonts(json_path: str,
                train_ratio: float = 0.9,
                seed: int = 42) -> Tuple[set, set]:
    """used for train/valid split in terms of fonts"""
    with open(json_path, "r") as f:
        fonts = json.load(f)
    random.Random(seed).shuffle(fonts)
    n_train = int(len(fonts) * train_ratio)
    train_fonts = set(fonts[:n_train])
    valid_fonts = set(fonts[n_train:])
    return train_fonts, valid_fonts

def filter_dataset_fonts(ds: FourWayFontPairLatentLMDBDataset, allowed_fonts: set) -> None:
    ds.fonts = [f for f in ds.fonts if f in allowed_fonts]
    assert len(ds.fonts) > 1, "Filtered dataset not enough fonts"

    ds.common_chars = list(
        set.intersection(*(set(ds.data[f].keys()) for f in ds.fonts))
    )
    assert len(ds.common_chars) >= 2, "not enough characters after filtering"


if __name__ == "__main__":
    lmdb_path = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_latents_temp.lmdb"
    CONFIG_PATH = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/configs/config.yaml"
    CKPT_PATH   = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/checkpoints/vae_best_ckpt.pth"

    dataset = FourWayFontPairLatentLMDBDataset(lmdb_path)

    print(f"Dataset size (mock): {len(dataset)}")

    sample = dataset[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} → {v.shape}")
        else:
            print(f"{k} → {v}")
    
    from utils.vae_io import load_models, decode as vae_decode

    encoder, decoder, cfg, device = load_models(CONFIG_PATH, CKPT_PATH)
    sample = dataset[0]                      # sample one latent 
    latent  = sample["F_A+C_A"]              
    recon_img = vae_decode(latent, decoder, cfg, device)  # decode 
    recon_img.save("recon_check.png")                  

    print("Decoded image saved to recon_check.png")