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
import time 



# open LMDB dataset safely
def _open_lmdb(path: str) -> lmdb.Environment:
    return lmdb.open(
        path,
        readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=2048
    )



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



class FourWayFontPairLatentPTDataset(Dataset):
    def __init__(self,
                 pt_path: str,
                 chars_path: str,
                 fonts_json: str,
                 latent_shape: Tuple[int, int, int] = (4, 16, 16),
                 max_retry: int = 1000,
                 pair_num: int = 1000,
                 stats_yaml: Optional[str] = None):
        super().__init__()

        blob = torch.load(pt_path, map_location="cpu") # load latents tensor to CPU 
        if isinstance(blob, dict) and "latents" in blob:
            latents = blob["latents"]
        else:
            latents = blob  # allow raw tensor
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(latents)}")
        if latents.dim() != 4:
            raise ValueError(f"Expected latents.dim()==4, got shape {tuple(latents.shape)}")

        self.latents_hwc = latents.contiguous()  # (N, H, W, C)
        N, H, W, C = self.latents_hwc.shape

        # load chars 
        self.chars = [] 
        seen = set()
        with open(chars_path, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.strip()
                if ch and (ch not in seen):
                    seen.add(ch)
                    self.chars.append(ch)
        self.m = len(self.chars)
        if self.m <= 1:
            raise ValueError(f"Need at least 2 chars, got m={self.m}")

        # load fonts 
        with open(fonts_json, "r", encoding="utf-8") as f:
            self.fonts = json.load(f)
        self.n = len(self.fonts)
        if self.n <= 1:
            raise ValueError(f"Need at least 2 fonts, got n={self.n}")

        # sanity check 
        if N != self.m * self.n:
            raise ValueError(f"PT size mismatch: N={N} but m*n={self.m*self.n} (m={self.m}, n={self.n})")

        # stats for normalization 
        if stats_yaml:
            with open(stats_yaml, "r") as f:
                stats = yaml.safe_load(f)
            self.mean = torch.tensor(stats["mean"]).view(-1, 1, 1)  # (C,1,1)
            self.std  = torch.tensor(stats["std"]).view(-1, 1, 1)
        else:
            self.mean = self.std = None

        self.latent_shape = latent_shape
        if (latent_shape[1], latent_shape[2], latent_shape[0]) != (H, W, C):
            if (H, W, C) != (latent_shape[1], latent_shape[2], latent_shape[0]):
                raise ValueError(
                    f"Shape mismatch: loaded HWC={(H,W,C)} incompatible with expected CHW={latent_shape}"
                )

        self.data: Dict[str, Dict[str, int]] = {}
        for fi, font in enumerate(self.fonts):
            inner = {}
            for ci, ch in enumerate(self.chars):
                inner[ch] = self._flat_index(fi, ci)
            self.data[font] = inner

        self.common_chars = list(self.chars)  
        self._active_font_indices = list(range(self.n))

        self.max_retry = max_retry
        self.pair_num = pair_num

        print(f"[PTDataset] Loaded latents from {pt_path} with shape {tuple(self.latents_hwc.shape)}")
        print(f"[PTDataset] m(chars)={self.m}, n(fonts)={self.n} | total={self.m*self.n}")

    def __len__(self):
        return int(self.pair_num)

    def __getitem__(self, idx):
        if len(self._active_font_indices) < 2 or len(self.common_chars) < 2:
            raise RuntimeError("Not enough fonts or chars to sample a 4-way pair.")

        for _ in range(self.max_retry):
            fi_a, fi_b = random.sample(self._active_font_indices, 2)
            ci_a, ci_b = random.sample(range(self.m), 2)

            # compute flat indices
            idx_fa_ca = self._flat_index(fi_a, ci_a)
            idx_fa_cb = self._flat_index(fi_a, ci_b)
            idx_fb_ca = self._flat_index(fi_b, ci_a)
            idx_fb_cb = self._flat_index(fi_b, ci_b)

            # slice and convert to (C,H,W)
            z_fa_ca = self._get_chw(idx_fa_ca)
            z_fa_cb = self._get_chw(idx_fa_cb)
            z_fb_ca = self._get_chw(idx_fb_ca)
            z_fb_cb = self._get_chw(idx_fb_cb)

            # normalization 
            if self.mean is not None:
                z_fa_ca = (z_fa_ca - self.mean) / self.std
                z_fa_cb = (z_fa_cb - self.mean) / self.std
                z_fb_ca = (z_fb_ca - self.mean) / self.std
                z_fb_cb = (z_fb_cb - self.mean) / self.std

            return {
                "F_A+C_A": z_fa_ca,
                "F_A+C_B": z_fa_cb,
                "F_B+C_A": z_fb_ca,
                "F_B+C_B": z_fb_cb,
                "font_a": self.fonts[fi_a],
                "font_b": self.fonts[fi_b],
                "char_a": self.chars[ci_a],
                "char_b": self.chars[ci_b],
            }

        raise RuntimeError("Unable to sample a valid 4-way latent pair after max_retry attempts.")

    def _flat_index(self, font_idx: int, char_idx: int) -> int:
        return font_idx * self.m + char_idx

    def _get_chw(self, flat_idx: int) -> torch.Tensor:
        z_hwc = self.latents_hwc[flat_idx]  # (H,W,C)
        z_chw = z_hwc.permute(2, 0, 1).contiguous()  # (C,H,W)
        # sanity check
        if tuple(z_chw.shape) != self.latent_shape:
            z_chw = z_chw.view(*self.latent_shape)
        return z_chw

    def denorm(self, z: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return z
        return z * self.std.to(z.device) + self.mean.to(z.device)

    @property
    def fonts_count(self) -> int:
        return len(self._active_font_indices)

    def apply_font_filter(self, allowed_fonts: set):
        """Optionally restrict the active font subset to allowed_fonts."""
        self._active_font_indices = [i for i, f in enumerate(self.fonts) if f in allowed_fonts]
        assert len(self._active_font_indices) > 1, "Filtered dataset not enough fonts"



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

def filter_dataset_fonts(ds, allowed_fonts: set) -> None:
    # PT path (preferred, keeps internal _active_font_indices in sync)
    if hasattr(ds, "apply_font_filter") and callable(getattr(ds, "apply_font_filter")):
        ds.apply_font_filter(allowed_fonts)
        return

    # LMDB path (original behavior)
    ds.fonts = [f for f in ds.fonts if f in allowed_fonts]
    assert len(ds.fonts) > 1, "Filtered dataset not enough fonts"

    ds.common_chars = list(
        set.intersection(*(set(ds.data[f].keys()) for f in ds.fonts))
    )
    assert len(ds.common_chars) >= 2, "not enough characters after filtering"