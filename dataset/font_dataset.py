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


class FontPairDataset(Dataset):
    def __init__(self, root_dir, img_size=128, augment_prob=0.0):
        self.root_dir = root_dir
        self.augment_prob = augment_prob
        self.styles = os.listdir(root_dir)
        self.data = {}  # style -> {char: path}

        for style in self.styles:
            files = glob.glob(os.path.join(root_dir, style, "*.png"))
            char_map = {}
            for f in files:
                basename = os.path.basename(f)
                if "+" in basename:
                    _, char = basename.split("+", 1)
                    char = os.path.splitext(char)[0]
                    char_map[char] = f
            self.data[style] = char_map

        self.common_chars = list(
            set.intersection(*(set(self.data[s].keys()) for s in self.styles))
        )
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        while True:
            try:
                styleA, styleB = random.sample(self.styles, 2)
                charA, charB = random.sample(self.common_chars, 2)

                imgA = Image.open(self.data[styleA][charA]).convert("L")
                imgB = Image.open(self.data[styleB][charB]).convert("L")
                imgAB = Image.open(self.data[styleB][charA]).convert("L")
                imgBA = Image.open(self.data[styleA][charB]).convert("L")

                if random.random() < self.augment_prob:
                    imgA = augment_image(imgA)
                if random.random() < self.augment_prob:
                    imgB = augment_image(imgB)

                return self.transform(imgA), self.transform(imgB), self.transform(imgAB), self.transform(imgBA)
            except (UnidentifiedImageError, FileNotFoundError, OSError, KeyError):
                continue


class SingleFontLMDBDataset(Dataset):
    def __init__(self, lmdb_path, img_size=128, keys_subset=None, augment_prob=0.5):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.augment_prob = augment_prob

        if keys_subset is not None:
            self.keys = keys_subset
        else:
            self.keys = [key for key, _ in txn.cursor()]

        if len(self.keys) == 0:
            raise RuntimeError(f"No data found in LMDB: {lmdb_path}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        try:
            with self.env.begin() as txn:
                key = self.keys[idx]
                buffer = txn.get(key)
                img = Image.open(io.BytesIO(buffer)).convert("L")
                if random.random() < self.augment_prob:
                    img = augment_image(img)
                return self.transform(img)
        except (UnidentifiedImageError, OSError, TypeError):
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)


class FontPairLMDBDataset(Dataset):
    def __init__(self, lmdb_path, img_size=128, augment_prob=0.0):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.augment_prob = augment_prob
        with self.env.begin() as txn:
            self.keys = [key for key, _ in txn.cursor()]

        self.data = {}  # font -> {char: key}
        for key in self.keys:
            try:
                font, char = key.decode().split("+", 1)
                if font not in self.data:
                    self.data[font] = {}
                self.data[font][char] = key
            except Exception:
                continue

        self.fonts = list(self.data.keys())
        self.common_chars = list(
            set.intersection(*(set(chars.keys()) for chars in self.data.values()))
        )

        if len(self.common_chars) < 2 or len(self.fonts) < 2:
            raise ValueError("Insufficient fonts or common characters in LMDB.")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            while True:
                try:
                    styleA, styleB = random.sample(self.fonts, 2)
                    charA, charB = random.sample(self.common_chars, 2)

                    imgA = self._get_image(txn, self.data[styleA][charA])
                    imgB = self._get_image(txn, self.data[styleB][charB])
                    imgAB = self._get_image(txn, self.data[styleB][charA])
                    imgBA = self._get_image(txn, self.data[styleA][charB])

                    if random.random() < self.augment_prob:
                        imgA = augment_image(imgA)
                    if random.random() < self.augment_prob:
                        imgB = augment_image(imgB)

                    return self.transform(imgA), self.transform(imgB), self.transform(imgAB), self.transform(imgBA)
                except Exception:
                    continue

    def _get_image(self, txn, key):
        buffer = txn.get(key)
        img = Image.open(io.BytesIO(buffer)).convert("L")
        return img


import lmdb

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
    """
    从 LMDB 提取所有 keys（或前 max_keys 个），并保存到缓存文件。
    """
    if os.path.exists(cache_path):
        print(f"[Cached] Loading keys from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            keys = json.load(f)
        return [key.encode("utf-8") for key in keys]

    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    print(f"[Loading] Scanning keys from {lmdb_path}")
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




if __name__ == "__main__":
    # Example usage
    font_list = get_all_fonts("/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_data.lmdb")
    train_fonts = font_list[:-50]
    val_fonts = font_list[-50:]