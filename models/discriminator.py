# models/discriminator.py
import torch
import torch.nn as nn

class SimplePatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),           # → 32x32
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(64, 128, 4, 2, 1),                   # → 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(128, 256, 4, 1, 1),                  # ↓ 16x16 → 12x12
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(256, 1, 3, 1, 1)                     # ↓ 12x12 → 10x10
        )

    def forward(self, x):
        return self.net(x)

def build_discriminator(config):
    in_channels = config.get("in_channels", 1)
    return SimplePatchDiscriminator(in_channels=in_channels)
