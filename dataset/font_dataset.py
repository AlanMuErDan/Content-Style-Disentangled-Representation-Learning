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
from typing import Tuple
import pickle


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


class FourWayFontPairLatentLMDBDataset(Dataset):
    """
    用于加载 VAE latent 的 Dataset，用于 DDPM 阶段。

    key 格式: "font+char"
    value 是 torch.save 的 latent tensor (shape = [C, H, W])
    """
    def __init__(self,
                 lmdb_path: str,
                 latent_shape: Tuple[int, int, int] = (4, 16, 16),
                 max_retry: int = 1000,
                 pair_num: int = 1000):

        self.env = lmdb.open(lmdb_path, readonly=True,
                             lock=False, readahead=False)
        self.latent_shape = latent_shape
        self.max_retry = max_retry
        self.pair_num = pair_num

        # ---------- 1. 扫描全部 key ----------
        self.data = {}  # font -> {char: bytes_key}
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

        print(f"[LatentDataset] Found {len(self.fonts)} fonts and {len(self.common_chars)} common chars.")
        assert len(self.fonts) >= 2 and len(self.common_chars) >= 2, \
            "Too few fonts or shared characters in latent LMDB"

    def __len__(self):
        return int(self.pair_num)  # 抽样式数据集，无需设置精确长度

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            for _ in range(self.max_retry):
                try:
                    f_a, f_b = random.sample(self.fonts, 2)
                    c_a, c_b = random.sample(self.common_chars, 2)

                    # print(f"[LatentDataset] Sampling: {f_a}+{c_a}, {f_a}+{c_b}, {f_b}+{c_a}, {f_b}+{c_b}")

                    k_fa_ca = self.data[f_a][c_a]
                    k_fa_cb = self.data[f_a][c_b]
                    k_fb_ca = self.data[f_b][c_a]
                    k_fb_cb = self.data[f_b][c_b]

                    # print(f"[LatentDataset] Keys: {k_fa_ca}, {k_fa_cb}, {k_fb_ca}, {k_fb_cb}")

                    z_fa_ca = self._load_latent(txn, k_fa_ca)
                    z_fa_cb = self._load_latent(txn, k_fa_cb)
                    z_fb_ca = self._load_latent(txn, k_fb_ca)
                    z_fb_cb = self._load_latent(txn, k_fb_cb)

                    # print(f"[LatentDataset] Loaded latent shapes: "
                    #       f"{z_fa_ca.shape}, {z_fa_cb.shape}, {z_fb_ca.shape}, {z_fb_cb.shape}")

                    return {
                        "F_A+C_A": z_fa_ca,
                        "F_A+C_B": z_fa_cb,
                        "F_B+C_A": z_fb_ca,
                        "F_B+C_B": z_fb_cb,
                        "font_a": f_a, "font_b": f_b,
                        "char_a": c_a, "char_b": c_b,
                    }

                except Exception as e:
                    print(f"[WARN] latent load fail: {e}, retry…")
                    continue

            raise RuntimeError("Unable to sample a valid 4-way latent pair after 100 attempts.")

    def _load_latent(self, txn, k_bytes):
        buf = txn.get(k_bytes)
        arr = pickle.loads(buf)  # 变成 numpy
        latent = torch.from_numpy(arr)  # 转回 tensor

        if not isinstance(latent, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(latent)}")

        if latent.shape != self.latent_shape:
            raise ValueError(f"Expected shape {self.latent_shape}, got {latent.shape}")
        
        # print(f"[LatentDataset] Loaded latent shape: {latent.shape}")

        return latent


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
    lmdb_path = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_latents_temp.lmdb"
    dataset = FourWayFontPairLatentLMDBDataset(lmdb_path)

    print(f"Dataset size (mock): {len(dataset)}")

    sample = dataset[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} → {v.shape}")
        else:
            print(f"{k} → {v}")
    
    # ===============================
    #  Decode sanity-check (example)
    # ===============================
    from utils.vae_io import load_models, decode as vae_decode

    # ----- 1. 配置路径 -----
    CONFIG_PATH = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/configs/config.yaml"
    CKPT_PATH   = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/checkpoints/vae_best_ckpt.pth"

    # ----- 2. 载入模型（eval 模式）-----
    encoder, decoder, cfg, device = load_models(CONFIG_PATH, CKPT_PATH)

    # ----- 3. 随机取一个 latent -----
    sample = dataset[0]                      # ← 已在上面创建过 FourWayFontPairLatentLMDBDataset
    latent  = sample["F_A+C_A"]              # 任选一条，例如 F_A+C_A

    # ----- 4. 解码并保存 / 可视化 -----
    recon_img = vae_decode(latent, decoder, cfg, device)  # PIL.Image
    recon_img.save("recon_check.png")                     # 或 recon_img.show()

    print("Decoded image saved to recon_check.png")