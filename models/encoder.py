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


class CNNContentStyleEncoder(nn.Module):
    def __init__(self, content_out_dim=1024, style_out_dim=1024):
        super().__init__()
        
        # Content encoder (保留 spatial 信息)
        self.content_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 4, kernel_size=1),  # 回到 4 通道, 尺寸不变
            nn.Tanh(),  # optional, 限制范围 [-1, 1]
        )
        
        # Style encoder (downsample)
        self.style_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        # Projection heads
        self.fc_content = nn.Linear(4 * 16 * 16, content_out_dim)
        self.fc_style = nn.Linear(16 * 8 * 8, style_out_dim)

    def forward(self, x):
        """
        x: (B, 4, 16, 16)
        """
        # content
        c_feat = self.content_encoder(x)     # (B, 4, 16, 16)
        c_flat = c_feat.view(x.size(0), -1)  # (B, 1024)
        c_vec = self.fc_content(c_flat)

        # style
        s_feat = self.style_encoder(x)       # (B, 16, 8, 8)
        s_flat = s_feat.view(x.size(0), -1)  # (B, 1024)
        s_vec = self.fc_style(s_flat)

        return torch.cat([c_vec, s_vec], dim=1)  # (B, 2048)


class CNNDisentangleEncoder(nn.Module):
    """
    Lightweight CNN encoder that keeps the 16x16 spatial grid for both content and style.
    It first builds a shared feature map, then produces two heads with shallow pooling to
    emphasize local (content) vs slightly more global (style) cues.
    """
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        depth: int = 3,
        content_channels: int = 4,
        style_channels: int = 4,
    ):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(depth):
            out_ch = hidden_channels
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.SiLU())
            ch = out_ch
        self.backbone = nn.Sequential(*layers)

        self.content_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, content_channels, kernel_size=1),
        )

        self.style_head = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, style_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_channels, 16, 16)
        Returns:
            content_map: (B, content_channels, 16, 16)
            style_map:   (B, style_channels,   16, 16)
        """
        feat = self.backbone(x)
        content_map = self.content_head(feat)
        style_map = self.style_head(feat)
        return content_map, style_map


def build_encoder(name: str = "diff_encoder",
                  img_size: int = 128,
                  output_dim: Optional[int] = 256,
                  latent_channels: int = 16):
                  
    name = name.lower()

    if name in {"diff_encoder", "vae_encoder"}:
        return FeatureMapEncoder(img_size=img_size, latent_channels=latent_channels)

    raise NotImplementedError(f"Unknown encoder: {name}")
