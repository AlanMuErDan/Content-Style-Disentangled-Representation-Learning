# dataset/font_dataset.py

import os
import io
# import lmdb
import glob
import random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch


def augment_image(img: Image.Image) -> Image.Image:
    """
    对单张灰度图做随机缩放和平移，填充白底。
    """
    scale = random.uniform(0.5, 1.0)
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
    center_x = random.randint(cx_min, cx_max)
    center_y = random.randint(cy_min, cy_max)

    left = center_x - half_w_floor
    top = center_y - half_h_floor

    background = Image.new('L', (img.width, img.height), color=255)
    background.paste(scaled, (left, top))

    return background

class FontPairDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.styles = os.listdir(root_dir)  # a list of styles 
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
        styleA, styleB = random.sample(self.styles, 2)
        charA, charB = random.sample(self.common_chars, 2)

        imgA = self.transform(Image.open(self.data[styleA][charA]).convert("L"))
        imgB = self.transform(Image.open(self.data[styleB][charB]).convert("L"))

        # ground truth for cross-style characters
        img_gt_crossAB = self.transform(Image.open(self.data[styleB][charA]).convert("L"))
        img_gt_crossBA = self.transform(Image.open(self.data[styleA][charB]).convert("L"))

        return imgA, imgB, img_gt_crossAB, img_gt_crossBA


class SingleFontDataset(Dataset):
    def __init__(self, root_dir, img_size=128, augment_prob=1.0):
        self.augment_prob = augment_prob
        self.img_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".png"):
                    self.img_paths.append(os.path.join(root, file))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No PNG files found in {root_dir}")

    def __len__(self):
        return len(self.img_paths)

    # def __getitem__(self, idx):
    #     try:
    #         img = Image.open(self.img_paths[idx]).convert("L")

    #         if random.random() < self.augment_prob:
    #             imgA = augment_image(imgA)
    #         if random.random() < self.augment_prob:
    #             imgB = augment_image(imgB)

    #         return self.transform(img)
    #     except (UnidentifiedImageError, FileNotFoundError, OSError):
    #         new_idx = (idx + 1) % len(self.img_paths)
    #         return self.__getitem__(new_idx)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert("L")

            if random.random() < self.augment_prob:
                img = augment_image(img)

            return self.transform(img)
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            new_idx = (idx + 1) % len(self.img_paths)
            return self.__getitem__(new_idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()

    # 初始化不带增强的数据集
    dataset_plain = SingleFontDataset(
        lmdb_path=args.lmdb_path,
        img_size=args.img_size,
        augment_prob=0.0
    )

    # 初始化带增强的数据集
    dataset_aug = SingleFontDataset(
        lmdb_path=args.lmdb_path,
        img_size=args.img_size,
        augment_prob=0.5
    )