# models/encoder.py

import math
from typing import Optional
import torch
import torch.nn as nn
from torchvision.models import resnet18
from diffusers.models.vae import Encoder as DiffusersEncoder



def _patch_relu(module: nn.Module) -> None:
    """Make every nn.ReLU *out-of-place* to avoid gradient side-effects."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            _patch_relu(child)



class ResNet18Encoder(nn.Module):
    def __init__(self, output_dim: int = 256):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _patch_relu(base)
        self.features = nn.Sequential(*list(base.children())[:-2])  
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.proj     = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.proj(x)
        


# class FeatureMapEncoder(DiffusersEncoder):   # 8x8 latent version 
#     def __init__(self,
#                  in_channels: int = 1,
#                  img_size: int = 128,
#                  latent_channels: int = 16,
#                  layers_per_block: int = 2):
#         n_down = int(math.log2(img_size // 8))
#         if 8 * 2 ** n_down != img_size:
#             raise ValueError("img_size must be 8 × 2^n (e.g. 32, 64, 128)")

#         down_block_types    = ("DownEncoderBlock2D",) * n_down
#         block_out_channels  = (64,) * (n_down + 1)

#         super().__init__(
#             in_channels=in_channels,
#             out_channels=latent_channels,    
#             down_block_types=down_block_types,
#             block_out_channels=block_out_channels,
#             layers_per_block=layers_per_block,
#             norm_num_groups=8,
#             act_fn="silu",
#         )
#         self.latent_channels = latent_channels

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         sample = super().forward(x)  
#         mu, logvar = torch.chunk(sample, 2, dim=1)
#         return mu, logvar



class FeatureMapEncoder(DiffusersEncoder): # 16x16 latent version 
    def __init__(self,
                 in_channels: int = 1,
                 img_size: int = 128,
                 latent_channels: int = 4,  
                 layers_per_block: int = 2):
        
        target_resolution = 16
        n_down = int(math.log2(img_size // target_resolution))
        if target_resolution * 2 ** n_down != img_size:
            raise ValueError("img_size must be target_resolution × 2^n")

        down_block_types    = ("DownEncoderBlock2D",) * n_down
        block_out_channels  = (64,) * (n_down + 1)

        super().__init__(
            in_channels=in_channels,
            out_channels=latent_channels,    
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=8,
            act_fn="silu",
        )
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample = super().forward(x)  
        mu, logvar = torch.chunk(sample, 2, dim=1)
        return mu, logvar



def build_encoder(name: str = "diff_encoder",
                  img_size: int = 128,
                  output_dim: Optional[int] = 256,
                  latent_channels: int = 16):
                  
    name = name.lower()

    if name in {"diff_encoder", "vae_encoder"}:
        return FeatureMapEncoder(img_size=img_size, latent_channels=latent_channels)

    raise NotImplementedError(f"Unknown encoder: {name}")