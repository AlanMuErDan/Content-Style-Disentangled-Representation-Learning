import math
from typing import Optional
import torch
import torch.nn as nn
from diffusers.models.vae import Decoder as DiffusersDecoder



class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, img_size: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),  # 4  → 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # 8  → 16
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(),  # 16 → 32
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(),  # 32 → 64
            nn.ConvTranspose2d(32,  1,   4, 2, 1), nn.Tanh(),  # 64 → 128
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.upsample(x)



class UNetDecoder(nn.Module):
    def __init__(self, latent_dim: int = 512, img_size: int = 128):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU())  # 8  → 16
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU())  # 16 → 32
        self.up3 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU())  # 32 → 64
        self.up4 = nn.Sequential(nn.ConvTranspose2d(16, 1,  4, 2, 1), nn.Tanh())  # 64 → 128

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 8, 8)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x



class FeatureMapDecoder(DiffusersDecoder):
    def __init__(self,
                 latent_channels: int = 8,
                 img_size: int = 128,
                 out_channels: int = 1,
                 layers_per_block: int = 2):
        n_up = int(math.log2(img_size // 8))
        if 8 * 2 ** n_up != img_size:
            raise ValueError("img_size must be a power-of-two multiple of 8 (e.g. 32, 64, 128)")

        up_block_types = ("UpDecoderBlock2D",) * n_up
        block_out_channels = (64,) * (n_up + 1)

        super().__init__(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=8,
            act_fn="silu",
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = super().forward(z)  # [B, 1, 128, 128], unbounded output
        return torch.tanh(x)    



def build_decoder(name: str = "diff_decoder",
                  latent_dim: Optional[int] = 512,
                  latent_channels: Optional[int] = 8,
                  img_size: int = 128):
    name = name.lower()

    if name == "simple":
        if latent_dim is None:
            raise ValueError("'simple' decoder expects vector latents; latent_dim cannot be None.")
        return SimpleDecoder(latent_dim=latent_dim, img_size=img_size)

    if name == "unet":
        if latent_dim is None:
            raise ValueError("'unet' decoder expects vector latents; latent_dim cannot be None.")
        return UNetDecoder(latent_dim=latent_dim, img_size=img_size)

    if name in {"diff_decoder", "featuremap", "vae_decoder"}:  
        return FeatureMapDecoder(latent_channels=latent_channels, img_size=img_size)

    raise NotImplementedError(f"Decoder '{name}' is not supported.")