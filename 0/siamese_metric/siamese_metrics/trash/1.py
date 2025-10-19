# -*- coding: utf-8 -*-
# siamese_latent_pipeline.py
import os
import random
from dataclasses import dataclass
from typing import Any, Tuple, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

# -----------------------------
# 0) HF .pt loader
# -----------------------------
def load_latents_from_hub(
    repo_id: str = "YuanhengLi/Font-Latent-Full-PT",
    filename: str = "font_latents_v2.pt",
    token: Optional[str] = os.getenv("HF_TOKEN"),
    map_location: str = "cpu",
    local_path: Optional[str] = None,
):
    """
    Load latents from HuggingFace Hub or local file.
    Args:
        local_path: If provided, load from local file instead of Hub
    """
    if local_path and os.path.exists(local_path):
        print(f"[INFO] Loading from local file: {local_path}")
        obj = torch.load(local_path, map_location=map_location)
        print(f"[INFO] Loaded from local file")
        _brief(obj)
        return obj
    
    try:
        pt_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        obj = torch.load(pt_path, map_location=map_location)
        print(f"[INFO] Loaded from {repo_id}/{filename}")
        _brief(obj)
        return obj
    except Exception as e:
        print(f"[ERROR] Failed to load from Hub: {e}")
        print("[INFO] Please provide a local_path or check repo_id/filename")
        raise

def _brief(obj: Any):
    print("[INFO] Type:", type(obj))
    if isinstance(obj, torch.Tensor):
        print("[INFO] Tensor shape:", tuple(obj.shape))
    elif isinstance(obj, (list, tuple)):
        print("[INFO] Top-level len:", len(obj))
        if len(obj) and isinstance(obj[0], (list, tuple, torch.Tensor)):
            try:
                if isinstance(obj[0], torch.Tensor):
                    print("[INFO] obj[0] shape:", tuple(obj[0].shape))
                elif isinstance(obj[0], (list, tuple)) and len(obj[0]):
                    inner = obj[0][0]
                    if isinstance(inner, torch.Tensor):
                        print("[INFO] obj[0][0] shape:", tuple(inner.shape))
            except Exception as e:
                print("[WARN] brief fail:", e)
    elif isinstance(obj, dict):
        print("[INFO] Dict keys:", list(obj.keys())[:10])

# -----------------------------
# 1) Generic latent accessor
#    Expect access by (content_i, style_p)
# -----------------------------
class LatentAccessor:
    """
    Try to support:
      A) torch.Tensor [num_styles, num_contents, ...]
      B) list[list[Tensor]]  style-major
      C) dict with key 'latents' or 'data' pointing to A/B
    Default layout is style-major; set layout='content_style' if yours is [num_contents, num_styles, ...].
    """
    def __init__(self, raw, layout: Literal["style_content","content_style"]="style_content"):
        if isinstance(raw, dict):
            if "latents" in raw:
                raw = raw["latents"]
            elif "data" in raw:
                raw = raw["data"]
        self.raw = raw
        self.layout = layout
        self._check()

    def _check(self):
        if isinstance(self.raw, torch.Tensor):
            if self.raw.ndim < 2:
                raise ValueError("Tensor latents must be at least 2D: [styles, contents, ...] or [contents, styles, ...].")
        elif isinstance(self.raw, (list, tuple)):
            if not len(self.raw) or not isinstance(self.raw[0], (list, tuple, torch.Tensor)):
                raise ValueError("List latents must be nested as list[list[tensor]] or list[tensor].")
        else:
            raise ValueError(f"Unsupported latent type: {type(self.raw)}")

    def get(self, content_i: int, style_p: int) -> torch.Tensor:
        r = self.raw
        # style-major by default
        if isinstance(r, torch.Tensor):
            if self.layout == "style_content":
                return r[style_p, content_i]
            else:
                return r[content_i, style_p]
        elif isinstance(r, (list, tuple)):
            # style-major list[list[tensor]] or list[tensor]
            if isinstance(r[0], (list, tuple)):
                return r[style_p][content_i]
            else:
                # if it's list[tensor] we can't disambiguate; raise
                raise ValueError("Latents look like list[tensor]; need 2D (styles x contents).")
        else:
            raise RuntimeError("Unreachable")

# -----------------------------
# 2) Decoder (frozen)
#    Uses the actual VAE decoder implementations
# -----------------------------
import math
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
                 latent_channels: int = 4,
                 img_size: int = 128,
                 out_channels: int = 1,
                 layers_per_block: int = 2):
        target_resolution = 16  # encoder 输出的 spatial size
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

def load_frozen_decoder(decoder_name: str = "diff_decoder", **kwargs) -> nn.Module:
    """
    Load and freeze a VAE decoder.
    Expected: module(latent[B,...]) -> image[B,C,H,W], C in {1,3}, range [-1,1] (tanh output).
    """
    try:
        dec = build_decoder(name=decoder_name, **kwargs)
        print(f"[INFO] Loaded decoder: {decoder_name}")
    except Exception as e:
        print(f"[WARN] Failed to load {decoder_name}: {e}")
        # Fallback: identity "decoder" if your latents ARE already images [C,H,W]
        class IdentityDec(nn.Module):
            def forward(self, z):
                if z.ndim == 3:  # [C,H,W] -> [1,C,H,W]
                    return z.unsqueeze(0)
                elif z.ndim == 4:
                    return z
                else:
                    raise ValueError("Identity decoder expects image-like latent [C,H,W] or [B,C,H,W].")
        print("[WARN] Using Identity decoder fallback.")
        dec = IdentityDec()

    dec.eval()
    for p in dec.parameters():  # freeze
        p.requires_grad_(False)
    return dec

@torch.no_grad()
def decode_to_image(decoder: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    z: [latent_dim] or [B, latent_dim] or image-like feature map.
    Return: img [C,H,W] in [0,1].
    """
    if z.ndim == 1:
        z = z.unsqueeze(0)
    elif z.ndim == 3:  # feature map [C,H,W] -> [1,C,H,W]
        z = z.unsqueeze(0)
    
    img = decoder(z)  # [B,C,H,W]
    if img.ndim != 4:
        raise ValueError("Decoder must output [B,C,H,W].")
    img = img[0]  # [C,H,W]
    
    # Convert from tanh output [-1,1] to [0,1]
    if img.min() < -0.9 or img.max() > 0.9:  # likely tanh output
        img = (img + 1.0) / 2.0
        img = torch.clamp(img, 0.0, 1.0)
    # If already in [0,1] range, keep as is
    elif img.min() >= 0 and img.max() <= 1:
        pass
    else:
        # General normalization fallback
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    return img  # [C,H,W]

# -----------------------------
# 3) Build four images from (i,j,p,q)
# -----------------------------
@dataclass
class FourImages:
    ci_sp: torch.Tensor
    ci_sq: torch.Tensor
    cj_sp: torch.Tensor
    cj_sq: torch.Tensor

def build_four_images(accessor: LatentAccessor, decoder: nn.Module,
                      i: int, j: int, p: int, q: int, device="cpu") -> FourImages:
    with torch.no_grad():
        ci_sp = decode_to_image(decoder, accessor.get(i, p)).to(device)
        ci_sq = decode_to_image(decoder, accessor.get(i, q)).to(device)
        cj_sp = decode_to_image(decoder, accessor.get(j, p)).to(device)
        cj_sq = decode_to_image(decoder, accessor.get(j, q)).to(device)
    return FourImages(ci_sp, ci_sq, cj_sp, cj_sq)

# -----------------------------
# 4) Dataset that yields pairs for a given task
# -----------------------------
class FourWayPairDataset(Dataset):
    """
    Produce pairs for either 'content' or 'style' task.
    For 'content':
      positive (label=1): (ci,sp) vs (ci,sq)   # same content, diff style
      negative (label=0): (ci,sp) vs (cj,sp)   # diff content, same style
    For 'style':
      positive (label=1): (ci,sp) vs (cj,sp)   # same style, diff content
      negative (label=0): (ci,sp) vs (ci,sq)   # diff style, same content
    """
    def __init__(self, imgs: FourImages, task: Literal["content","style"]="content",
                 length: int = 2048, augment: bool = True):
        self.imgs = imgs
        self.task = task
        self.length = length
        self.augment = augment

    def __len__(self): return self.length

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment: return x
        # simple on-the-fly augmentations (geometric / noise)
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal flip
        if random.random() < 0.2:
            x = x + torch.randn_like(x) * 0.02
            x = torch.clamp(x, 0.0, 1.0)
        return x

    def __getitem__(self, idx):
        # 50/50 positive/negative
        pos = (random.random() < 0.5)
        if self.task == "content":
            if pos:
                x1, x2, y = self.imgs.ci_sp, self.imgs.ci_sq, 1.0
            else:
                x1, x2, y = self.imgs.ci_sp, self.imgs.cj_sp, 0.0
        else:  # style
            if pos:
                x1, x2, y = self.imgs.ci_sp, self.imgs.cj_sp, 1.0
            else:
                x1, x2, y = self.imgs.ci_sp, self.imgs.ci_sq, 0.0

        x1 = self._aug(x1)
        x2 = self._aug(x2)
        return x1, x2, torch.tensor([y], dtype=torch.float32)

# -----------------------------
# 5) Siamese model
# -----------------------------
class TinyEncoder(nn.Module):
    """A light CNN encoder for 64x64 or 128x128 grayscale images."""
    def __init__(self, in_ch=1, emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(inplace=True),  # 1/2
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),     # 1/4
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),    # 1/8
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),   # 1/16
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x):
        if x.ndim == 3:  # [C,H,W] -> [B=1,C,H,W]
            x = x.unsqueeze(0)
        if x.size(1) not in (1,3):
            # if decoder returned [C,H,W] with C not 1/3, collapse to 1ch
            x = x.mean(dim=1, keepdim=True)
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class SiameseJudge(nn.Module):
    def __init__(self, in_ch=1, emb_dim=256, mlp_hidden=256):
        super().__init__()
        self.encoder = TinyEncoder(in_ch=in_ch, emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden), nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x1, x2):
        v1 = self.encoder(x1)  # [B, D]
        v2 = self.encoder(x2)
        diff = torch.abs(v1 - v2)
        logit = self.head(diff)
        return logit, v1, v2

# -----------------------------
# 6) Train / evaluate helpers
# -----------------------------
def train_once(
    model: nn.Module,
    loader: DataLoader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    lr=1e-3, epochs=5
):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        model.train()
        tot_loss = 0.0
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logit, _, _ = model(x1, x2)
            loss = crit(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x1.size(0)
        avg = tot_loss / len(loader.dataset)
        print(f"[Epoch {ep}] loss={avg:.4f}")
    return model

@torch.no_grad()
def score_pair(model: nn.Module, img1: torch.Tensor, img2: torch.Tensor, device=None) -> float:
    device = device or next(model.parameters()).device
    model.eval()
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)
    logit, _, _ = model(img1, img2)
    prob = torch.sigmoid(logit).item()
    return prob  # in [0,1]

# -----------------------------
# 7) Glue everything: one-call runner
# -----------------------------
def run_training(
    i: int, j: int, p: int, q: int,
    task: Literal["content","style"]="content",
    layout: Literal["style_content","content_style"]="style_content",
    repo_id="YuanhengLi/Font-Latent-Full-PT",
    filename="font_latents_v2.pt",
    local_path: Optional[str] = None,     # path to local .pt file
    decoder_name: str = "diff_decoder",     # decoder type to use
    decoder_kwargs: dict = {},              # additional decoder args
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # load latents and adapter
    raw = load_latents_from_hub(
        repo_id=repo_id, 
        filename=filename, 
        map_location="cpu",
        local_path=local_path
    )
    accessor = LatentAccessor(raw, layout=layout)

    # load frozen decoder & decode four images
    decoder = load_frozen_decoder(decoder_name=decoder_name, **decoder_kwargs)
    decoder.to(device if next(decoder.parameters(), torch.tensor(0)).is_cuda else "cpu")  # no-op for Identity
    four = build_four_images(accessor, decoder, i, j, p, q, device="cpu")

    # quick infer input channels from one image
    C = four.ci_sp.shape[0]
    print(f"[INFO] Decoded image shape: {tuple(four.ci_sp.shape)} (C={C})")

    # dataset / loader
    ds = FourWayPairDataset(four, task=task, length=4096, augment=True)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4 if os.name != "nt" else 0)

    # model
    model = SiameseJudge(in_ch=C, emb_dim=256, mlp_hidden=256)
    model = train_once(model, dl, device=device, lr=1e-3, epochs=5)

    # sanity check: score positives/negatives
    if task == "content":
        pos = score_pair(model, four.ci_sp, four.ci_sq, device=device)  # same content
        neg = score_pair(model, four.ci_sp, four.cj_sp, device=device)  # diff content
    else:  # style
        pos = score_pair(model, four.ci_sp, four.cj_sp, device=device)  # same style
        neg = score_pair(model, four.ci_sp, four.ci_sq, device=device)  # diff style

    print(f"[CHECK] pos-prob={pos:.3f}  neg-prob={neg:.3f}  (threshold 0.5)")
    return model, four

def create_dummy_latents(num_styles=10, num_contents=20, latent_channels=4, spatial_size=16):
    """Create dummy latents for testing when real data is not available."""
    print("[INFO] Creating dummy latents for testing...")
    # Create random latents in format [styles, contents, channels, height, width]
    latents = torch.randn(num_styles, num_contents, latent_channels, spatial_size, spatial_size)
    return {"latents": latents}

if __name__ == "__main__":
    # Example with local data or dummy data
    i, j, p, q = 2, 5, 1, 3  # Use smaller indices for dummy data
    
    # First try to find local data files
    possible_paths = [
        "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/font_latents_v2.pt",
        "/scratch/gz2199/font_latents_v2.pt",
        "/scratch/gz2199/Content-Style-Disentangled-Representation-Learning/dataset/font_latents.pt"
    ]
    
    local_path = None
    for path in possible_paths:
        if os.path.exists(path):
            local_path = path
            print(f"[INFO] Found local data at: {path}")
            break
    
    if local_path:
        model, four = run_training(i, j, p, q, task="content", local_path=local_path)
    else:
        print("[WARN] No local data found, creating dummy data for testing...")
        # Create and save dummy data
        dummy_data = create_dummy_latents(num_styles=8, num_contents=10)
        dummy_path = "/tmp/dummy_font_latents.pt"
        torch.save(dummy_data, dummy_path)
        
        model, four = run_training(i, j, p, q, task="content", local_path=dummy_path)
