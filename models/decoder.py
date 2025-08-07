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


# 16*16*4
class FeatureMapDecoder(DiffusersDecoder):
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
        x = super().forward(z)  # [B, 1, 128, 128], unbounded output
        return torch.tanh(x)    



def build_decoder(name: str = "diff_decoder",
                  latent_dim: Optional[int] = 512,
                  latent_channels: Optional[int] = 4,
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

import os
import lmdb
import random
import pickle
import torch
from torchvision.utils import save_image
from tqdm import tqdm

def main():
    # 配置路径
    lmdb_path = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/lmdb_test"
    output_dir = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/decoder_output"
    checkpoint_path = "/scratch/rw3239/Content-Style-Disentangled-Representation-Learning/checkpoints/20250723_121344_super_vae_16*16*4_seed10086/best_ckpt.pth"

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打开 LMDB 环境
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    keys = []
    with env.begin() as txn:
        with txn.cursor() as cursor:
            for key, _ in cursor:
                keys.append(key)

    # 抽取 2 个样本
    sample_keys = random.sample(keys, 2)

    # 读取 latent
    latents = []
    with env.begin() as txn:
        for key in sample_keys:
            byte_data = txn.get(key)
            latent_array = pickle.loads(byte_data)
            latent_tensor = torch.tensor(latent_array).to(device)
            latents.append(latent_tensor)

    latent_batch = torch.stack(latents).to(device)  # [2, 4, 16, 16]

    # 初始化 decoder 并加载权重
    decoder = FeatureMapDecoder(latent_channels=4, img_size=128).to(device)

    # 加载 checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "ema_decoder" in ckpt:
        print("Using EMA decoder weights.")
        decoder.load_state_dict(ckpt["ema_decoder"])
    elif "decoder" in ckpt:
        print("Using standard decoder weights.")
        decoder.load_state_dict(ckpt["decoder"])
    else:
        raise KeyError("Decoder weights not found in checkpoint.")

    decoder.eval()

    # 解码
    with torch.no_grad():
        decoded_imgs = decoder(latent_batch)  # [2, 1, 128, 128]

    # 保存图像
    for i, img in enumerate(decoded_imgs):
        print(f"[DEBUG] img.min={img.min().item():.3f}, img.max={img.max().item():.3f}")
        save_path = os.path.join(output_dir, f"decoded_{i}.png")
        save_image(img, save_path)

    print(f"✅ Decoded images saved to: {output_dir}")

if __name__ == "__main__":
    main()


# # 16*8*8
# class FeatureMapDecoder(DiffusersDecoder):
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
#         x = super().forward(z)  # [B, 1, 128, 128], unbounded output
#         return torch.tanh(x)    



# def build_decoder(name: str = "diff_decoder",
#                   latent_dim: Optional[int] = 512,
#                   latent_channels: Optional[int] = 16,
#                   img_size: int = 128):
#     name = name.lower()

#     if name == "simple":
#         if latent_dim is None:
#             raise ValueError("'simple' decoder expects vector latents; latent_dim cannot be None.")
#         return SimpleDecoder(latent_dim=latent_dim, img_size=img_size)

#     if name == "unet":
#         if latent_dim is None:
#             raise ValueError("'unet' decoder expects vector latents; latent_dim cannot be None.")
#         return UNetDecoder(latent_dim=latent_dim, img_size=img_size)

#     if name in {"diff_decoder", "featuremap", "vae_decoder"}:  
#         return FeatureMapDecoder(latent_channels=latent_channels, img_size=img_size)

#     raise NotImplementedError(f"Decoder '{name}' is not supported.")