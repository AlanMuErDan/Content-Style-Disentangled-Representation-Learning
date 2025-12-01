#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference helpers for Siamese content/style similarity."""

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from collections.abc import Mapping

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONTENT_CKPT = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/checkpoints/content_bset_ckpt_vgg.pth"
DEFAULT_STYLE_CKPT = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/checkpoints/style_best_ckpt_vgg.pth"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "siamese_samples"

# Reuse model definitions
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.siamese_CNN import SiameseJudge
from dataset.siamese_dataloading import ImprovedLatentAccessor, latent_to_tensor
from trainer.train_siamese_metrics import load_vae_decoder, DATA_PATHS

TensorPair = Tuple[torch.Tensor, torch.Tensor]


def load_model(checkpoint_path, task: str, device: torch.device, encoder_type: str = "enhanced") -> SiameseJudge:
    checkpoint_path = Path(checkpoint_path)
    """Load siamese model weights for the given task."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping):
        # Training checkpoints often wrap the actual weights—unwrap the relevant section when present.
        for key in ("model_state", "state_dict", "model", "net"):
            maybe_state = state.get(key)
            if isinstance(maybe_state, Mapping):
                state = maybe_state
                break

    if not isinstance(state, Mapping):
        raise RuntimeError(f"Unexpected checkpoint structure in {checkpoint_path}")

    # Remove a leading 'module.' if weights were saved from DataParallel.
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

    def _build(enc_type: str) -> SiameseJudge:
        return SiameseJudge(in_ch=1, emb_dim=512, mlp_hidden=512, task=task, encoder_type=enc_type)

    def _infer_encoder_type_from_state(state_dict: Mapping) -> Optional[str]:
        if any(k.startswith("encoder.fc.") for k in state_dict):
            return "vgg"
        if any(k.startswith("encoder.classifier.") for k in state_dict):
            return "enhanced"
        return None

    candidates = []
    if encoder_type:
        candidates.append(encoder_type)
    inferred = _infer_encoder_type_from_state(state)
    if inferred and inferred not in candidates:
        candidates.append(inferred)
    for fallback in ("vgg", "enhanced"):
        if fallback not in candidates:
            candidates.append(fallback)

    def _prepare_lazy_modules(model: SiameseJudge, state_dict: Mapping) -> None:
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            return
        if getattr(encoder, "fc", None) is None and hasattr(encoder, "_build_fc"):
            fc_weight = state_dict.get("encoder.fc.0.weight")
            if fc_weight is not None:
                encoder._build_fc(fc_weight.shape[1], torch.device(fc_weight.device))

    last_err: RuntimeError | None = None
    for candidate in candidates:
        try:
            model = _build(candidate)
            _prepare_lazy_modules(model, state)
            model.load_state_dict(state)
            if candidate != encoder_type:
                warnings.warn(
                    f"Checkpoint encoder appears to be '{candidate}', "
                    f"but '{encoder_type}' was requested. Using '{candidate}'.",
                    RuntimeWarning,
                )
            model.to(device).eval()
            return model
        except RuntimeError as err:
            last_err = err

    raise last_err if last_err is not None else RuntimeError("Failed to load model weights.")


@torch.no_grad()
def score_pair(model: SiameseJudge, img_a: torch.Tensor, img_b: torch.Tensor, device: torch.device) -> float:
    """Return similarity probability in [0,1] for a tensor pair."""
    if img_a.dim() == 3:
        img_a = img_a.unsqueeze(0)
    if img_b.dim() == 3:
        img_b = img_b.unsqueeze(0)
    logits = model(img_a.to(device), img_b.to(device))
    return torch.sigmoid(logits).item()


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert PIL image (128x128 grayscale) to CHW tensor in [0,1]."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return transform(img)


def create_pair_from_latents(
    accessor: ImprovedLatentAccessor,
    decoder: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Create a tensor pair according to mode: 'same-c-s', 'same-c-diff-s', 'diff-c-same-s', 'diff-c-diff-s'."""
    num_contents = accessor.num_characters
    num_styles = accessor.num_styles_per_char

    content_i = torch.randint(0, num_contents, (1,)).item()
    content_j = torch.randint(0, num_contents, (1,)).item()
    while content_j == content_i:
        content_j = torch.randint(0, num_contents, (1,)).item()

    style_p = torch.randint(0, num_styles, (1,)).item()
    style_q = torch.randint(0, num_styles, (1,)).item()
    while style_q == style_p:
        style_q = torch.randint(0, num_styles, (1,)).item()

    if mode == "same-c-s":
        idx_a = (content_i, style_p)
        idx_b = (content_i, style_p)
    elif mode == "same-c-diff-s":
        idx_a = (content_i, style_p)
        idx_b = (content_i, style_q)
    elif mode == "diff-c-same-s":
        idx_a = (content_i, style_p)
        idx_b = (content_j, style_p)
    elif mode == "diff-c-diff-s":
        idx_a = (content_i, style_p)
        idx_b = (content_j, style_q)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    tensor_a = latent_to_tensor(decoder, accessor.get(*idx_a), cfg, device)
    tensor_b = latent_to_tensor(decoder, accessor.get(*idx_b), cfg, device)

    meta = {
        "mode": mode,
        "content_a": idx_a[0],
        "style_a": idx_a[1],
        "content_b": idx_b[0],
        "style_b": idx_b[1],
    }

    return tensor_a, tensor_b, meta


@torch.no_grad()
def compute_content_style_scores(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    content_model: SiameseJudge,
    style_model: SiameseJudge,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (content_score, style_score) in [0,1]."""
    content_score = score_pair(content_model, img_a, img_b, device)
    style_score = score_pair(style_model, img_a, img_b, device)
    return content_score, style_score


def save_pair_outputs(
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    meta: dict,
    scores: dict,
    root_dir: Path,
) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    index = scores.get("index", "sample")
    to_pil = transforms.ToPILImage()
    to_pil(img_a).save(root_dir / f"{index}_img_a.png")
    to_pil(img_b).save(root_dir / f"{index}_img_b.png")
    with open(root_dir / f"{index}_meta.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "scores": scores}, f, indent=2)


def run_random_demo(
    num_samples: int,
    output_dir: Path,
    device: torch.device,
    encoder_type: str = "enhanced",
) -> None:
    modes = ["same-c-s", "same-c-diff-s", "diff-c-same-s", "diff-c-diff-s"]

    decoder, cfg, _ = load_vae_decoder(
        config_path=DATA_PATHS["vae_config"],
        ckpt_path=DATA_PATHS["vae_ckpt"],
        device=device.type,
    )

    accessor = ImprovedLatentAccessor(
        pt_path=DATA_PATHS["pt_path"],
        chars_path=DATA_PATHS["chars_path"],
        fonts_json=DATA_PATHS["fonts_json"],
        device="cpu",
        latent_shape=(4, 16, 16),
    )

    encoder_type = "vgg"
    content_model = load_model(DEFAULT_CONTENT_CKPT, "content", device, encoder_type)
    style_model = load_model(DEFAULT_STYLE_CKPT, "style", device, encoder_type)

    for idx in range(num_samples):
        mode = modes[idx % len(modes)]
        tensor_a, tensor_b, meta = create_pair_from_latents(accessor, decoder, cfg, device, mode)
        content_score, style_score = compute_content_style_scores(
            tensor_a, tensor_b, content_model, style_model, device
        )
        scores = {
            "index": f"sample_{idx:04d}",
            "content_score": content_score,
            "style_score": style_score,
        }
        save_pair_outputs(tensor_a, tensor_b, meta, scores, output_dir)
        print(
            f"[{idx+1}/{num_samples}] mode={mode} content_score={content_score:.3f} style_score={style_score:.3f}"
        )


def main():
    parser = argparse.ArgumentParser("Siamese content/style scoring demo")
    parser.add_argument("--img_a", type=str, default=None, help="Path to first image (optional)")
    parser.add_argument("--img_b", type=str, default=None, help="Path to second image (optional)")
    parser.add_argument(
        "--content-ckpt",
        type=str,
        default=str(DEFAULT_CONTENT_CKPT),
        help="Content model checkpoint path",
    )
    parser.add_argument(
        "--style-ckpt",
        type=str,
        default=str(DEFAULT_STYLE_CKPT),
        help="Style model checkpoint path",
    )
    parser.add_argument("--encoder-type", type=str, default="enhanced", choices=["enhanced", "vgg"], help="Encoder type")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of random pairs to generate when inputs not given")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    content_model = load_model(Path(args.content_ckpt), "content", device, args.encoder_type)
    style_model = load_model(Path(args.style_ckpt), "style", device, args.encoder_type)

    if args.img_a and args.img_b:
        img_a = preprocess_image(Image.open(args.img_a))
        img_b = preprocess_image(Image.open(args.img_b))
        content_score, style_score = compute_content_style_scores(img_a, img_b, content_model, style_model, device)
        print(f"content_score={content_score:.4f} style_score={style_score:.4f}")
    else:
        run_random_demo(
            num_samples=args.num_samples,
            output_dir=DEFAULT_OUTPUT_DIR,
            device=device,
            encoder_type=args.encoder_type,
        )


if __name__ == "__main__":
    main()
