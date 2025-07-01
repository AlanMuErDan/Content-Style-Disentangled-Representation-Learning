# dataset/font_dataset.py

import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FontPairDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.styles = os.listdir(root_dir) # a list of styles 
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