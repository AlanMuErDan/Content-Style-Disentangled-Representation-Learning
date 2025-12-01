import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleRefinerUNet(nn.Module):
    """
    Lightweight 16x16x4 → 16x16x4 U-Net for latent refinement
    """
    def __init__(self, in_ch=4, base_ch=64, out_ch=4):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.down = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.up = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 3, padding=1)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = F.relu(self.down(x1))
        x3 = self.mid(x2)
        x4 = F.relu(self.up(x3))
        x_cat = torch.cat([x4, x1], dim=1)
        out = self.dec(x_cat)
        return x + out  # residual refinement
