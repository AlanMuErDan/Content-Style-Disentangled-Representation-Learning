# utils/vae_io.py
# -------------------------------------------------
#  Unified Encode / Decode utilities for VAE models
#  (Python 3.9-compatible type annotations)
# -------------------------------------------------
import os, io, yaml, torch
from typing import Tuple, Union, Optional
from PIL import Image
from torchvision import transforms

# ------------------------------------------------------------------
#  Local deps – make sure the project root is on PYTHONPATH
# ------------------------------------------------------------------
from models import build_encoder, build_decoder


# -------------------------------------------------
#  Internal helpers
# -------------------------------------------------
def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["vae"] if "vae" in cfg else cfg


def _to_pil(img_or_tensor: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(img_or_tensor, Image.Image):
        return img_or_tensor.convert("L")
    if img_or_tensor.dim() == 3 and img_or_tensor.size(0) in {1, 3}:
        return transforms.ToPILImage()(img_or_tensor.cpu()).convert("L")
    raise TypeError("Input must be PIL.Image or 3-D torch.Tensor.")


# -------------------------------------------------
#  Public API
# -------------------------------------------------
def load_models(
    config_path: str,
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, dict, torch.device]:
    """
    Build encoder / decoder from YAML config and load weights.

    Returns:
        encoder  – torch.nn.Module (eval mode)
        decoder  – torch.nn.Module (eval mode)
        cfg      – dict (the 'vae' section of config)
        device   – torch.device actually used
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_config(config_path)

    encoder = build_encoder(
        name=cfg["encoder"],
        img_size=cfg["img_size"],
        latent_channels=cfg.get("latent_channels", 4),
    ).to(device).eval()

    decoder = build_decoder(
        name=cfg["decoder"],
        img_size=cfg["img_size"],
        latent_channels=cfg.get("latent_channels", 4),
    ).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    return encoder, decoder, cfg, device


def encode(
    img: Union[Image.Image, torch.Tensor],
    encoder: torch.nn.Module,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode a single image (PIL or Tensor CHW) → latent Tensor [C,H,W].

    Returns latent mean μ (no sampling).
    """
    pil = _to_pil(img)

    preprocess = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),          # → [1,H,W], float32 in [0,1]
    ])

    x = preprocess(pil).unsqueeze(0).to(device)  # [1,1,H,W]
    with torch.no_grad():
        mu, _ = encoder(x)
    return mu.squeeze(0).cpu()                   # [C,H,W]


def decode(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    cfg: dict,
    device: torch.device,
) -> Image.Image:
    """
    Decode a latent Tensor [C,H,W] → reconstructed PIL.Image (grayscale).
    """
    if latent.dim() != 3:
        raise ValueError("Latent must have shape [C,H,W].")
    z = latent.unsqueeze(0).to(device)           # [1,C,H,W]
    with torch.no_grad():
        recon = decoder(z).squeeze(0).cpu()      # [1,H,W]
    recon = torch.clamp(recon, 0, 1)
    return transforms.ToPILImage()(recon)