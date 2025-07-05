import math
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18
from diffusers.models.vae import Encoder as DiffusersEncoder

# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------

def _patch_relu(module: nn.Module) -> None:
    """Make every nn.ReLU *out-of-place* to avoid gradient side-effects."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            _patch_relu(child)

# -----------------------------------------------------------------------------
#  Baseline ResNet encoder (vector latents)
# -----------------------------------------------------------------------------

class ResNet18Encoder(nn.Module):
    """ResNet-18 trunk → 1×1 global pool → linear projection (vector latent)."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        _patch_relu(base)
        self.features = nn.Sequential(*list(base.children())[:-2])  # until last conv
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.proj     = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.proj(x)

# -----------------------------------------------------------------------------
#  Diffusers VAE encoder (spatial latents)
# -----------------------------------------------------------------------------

class FeatureMapEncoder(DiffusersEncoder):
    """Stable-Diffusion VAE encoder that downsamples to [B, latent_channels, 8, 8].

    The base class produces 2×latent_channels (for μ, logσ²).  Here we simply
    return the first half, equivalent to μ, so downstream code gets the exact
    channel count expected by the decoder.
    """

    def __init__(self,
                 in_channels: int = 1,
                 img_size: int = 128,
                 latent_channels: int = 8,
                 layers_per_block: int = 2):
        # 128 → 8 requires 4 downs (factor 2 each)
        n_down = int(math.log2(img_size // 8))
        if 8 * 2 ** n_down != img_size:
            raise ValueError("img_size must be 8 × 2^n (e.g. 32, 64, 128)")

        down_block_types    = ("DownEncoderBlock2D",) * n_down
        block_out_channels  = (64,) * (n_down + 1)

        super().__init__(
            in_channels=in_channels,
            out_channels=latent_channels,    # diffusers will internally ×2
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=8,
            act_fn="silu",
        )
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample = super().forward(x)  # shape: [B, 2*latent_channels, 8, 8]
        mu, logvar = torch.chunk(sample, 2, dim=1)
        return mu, logvar

# -----------------------------------------------------------------------------
#  Factory
# -----------------------------------------------------------------------------

def build_encoder(name: str = "diff_encoder",
                  img_size: int = 128,
                  output_dim: Optional[int] = 256,
                  latent_channels: int = 8):
    """Create encoder by name.

    * ``resnet18`` → vector latent encoder
    * ``diff_encoder`` / ``vae_encoder`` → spatial latent encoder compatible with diffusers Decoder
    """
    name = name.lower()

    if name == "resnet18":
        return ResNet18Encoder(output_dim=output_dim)

    if name in {"diff_encoder", "vae_encoder"}:
        return FeatureMapEncoder(img_size=img_size, latent_channels=latent_channels)

    raise NotImplementedError(f"Unknown encoder: {name}")
