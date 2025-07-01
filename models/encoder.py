# models/encoder.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34

class ResNet18Encoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change input channels to 1 
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze()
        return self.out(x)



class ResNet34Encoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base = resnet34(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) # change input channels to 1
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze()
        return self.out(x)



class UNetEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.ReLU(),   # 128 → 64
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), # 64 → 32
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),# 32 → 16
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),# 16 → 8
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU() # 8 → 4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.down(x)             # [B, 512, 4, 4]
        x = self.pool(x).squeeze()   # [B, 512]
        return self.fc(x)



def build_encoder(name="resnet18", output_dim=256):
    if name == "resnet18":
        return ResNet18Encoder(output_dim)
    elif name == "resnet34":
        return ResNet34Encoder(output_dim)
    elif name == "unet":
        return UNetEncoder(output_dim)
    else:
        raise NotImplementedError(f"Unknown encoder: {name}")