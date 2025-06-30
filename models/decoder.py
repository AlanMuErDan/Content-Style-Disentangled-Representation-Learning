# models/decoder.py

import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=512, img_size=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),  # 4 → 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # 8 → 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # 16 → 32
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),    # 32 → 64
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Tanh()      # 64 → 128
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.upsample(x)


class UNetDecoder(nn.Module):
    def __init__(self, latent_dim=512, img_size=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(),  # 8 → 16
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),  # 16 → 32
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),  # 32 → 64
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Tanh()    # 64 → 128
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 8, 8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x


def build_decoder(name="simple", latent_dim=512, img_size=128):
    if name == "simple":
        return SimpleDecoder(latent_dim=latent_dim, img_size=img_size)
    elif name == "unet":
        return UNetDecoder(latent_dim=latent_dim, img_size=img_size)
    elif name == "ddpm":
        from .ddpm_decoder import DDPMDecoder
        return DDPMDecoder(latent_dim=latent_dim, img_size=img_size)
    else:
        raise NotImplementedError(f"Decoder '{name}' is not supported.")
    