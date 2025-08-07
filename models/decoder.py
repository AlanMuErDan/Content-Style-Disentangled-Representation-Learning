# models/decoder.py

import math
from typing import Optional
import torch
import torch.nn as nn
from diffusers.models.vae import Decoder as DiffusersDecoder



# class FeatureMapDecoder(DiffusersDecoder):  # latent 8x8 version 
#     def __init__(self,
#                  latent_channels: int = 16,
#                  img_size: int = 128,
#                  out_channels: int = 1,
#                  layers_per_block: int = 2):
#         n_up = int(math.log2(img_size // 8))
#         if 8 * 2 ** n_up != img_size:
#             raise ValueError("img_size must be a power-of-two multiple of 8 (e.g. 32, 64, 128)")

#         up_block_types = ("UpDecoderBlock2D",) * n_up
#         block_out_channels = (64,) * (n_up + 1)

#         super().__init__(
#             in_channels=latent_channels,
#             out_channels=out_channels,
#             up_block_types=up_block_types,
#             block_out_channels=block_out_channels,
#             layers_per_block=layers_per_block,
#             norm_num_groups=8,
#             act_fn="silu",
#         )
    
#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         x = super().forward(z) 
#         return torch.tanh(x)    




class FeatureMapDecoder(DiffusersDecoder): # 16 x 16 latent version 
    def __init__(self,
                 latent_channels: int = 4,
                 img_size: int = 128,
                 out_channels: int = 1,
                 layers_per_block: int = 2):
        target_resolution = 16  
        n_up = int(math.log2(img_size // target_resolution))

        if target_resolution * 2 ** n_up != img_size:
            raise ValueError("img_size must be target_resolution × 2^n (e.g. 16→128)")

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
        x = super().forward(z)  
        return torch.tanh(x)  



def build_decoder(name: str = "diff_decoder",
                  latent_dim: Optional[int] = 512,
                  latent_channels: Optional[int] = 8,
                  img_size: int = 128):
    name = name.lower()

    if name in {"diff_decoder", "featuremap", "vae_decoder"}:  
        return FeatureMapDecoder(latent_channels=latent_channels, img_size=img_size)

    raise NotImplementedError(f"Decoder '{name}' is not supported.")