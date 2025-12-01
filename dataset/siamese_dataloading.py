import os
import math
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, Literal, Sequence, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.vae_io import decode as vae_decode
from .font_dataset import FourWayFontPairLatentPTDataset

TO_TENSOR = transforms.ToTensor()


@torch.no_grad()
def latent_to_tensor(decoder: nn.Module, latent: torch.Tensor, cfg: dict, device: torch.device) -> torch.Tensor:
    """Decode latent tensor to CHW float image tensor in [0,1]."""
    target_dtype = next(decoder.parameters()).dtype
    if latent.dtype != target_dtype:
        latent = latent.to(dtype=target_dtype)
    img = vae_decode(latent, decoder, cfg, device)
    tensor = TO_TENSOR(img)
    return tensor


class ImprovedLatentAccessor:
    """Latent accessor with automatic dataset structure discovery."""
    def __init__(self,
                 pt_path: str,
                 chars_path: Optional[str] = None,
                 fonts_json: Optional[str] = None,
                 device='cpu',
                 latent_shape: Tuple[int, int, int] = (4, 16, 16)):
        self.device = device
        self.pt_path = pt_path

        if chars_path and fonts_json and os.path.exists(chars_path) and os.path.exists(fonts_json):
            print("[info] using FourWayFontPairLatentPTDataset")
            try:
                self.dataset = FourWayFontPairLatentPTDataset(
                    pt_path=pt_path,
                    chars_path=chars_path,
                    fonts_json=fonts_json,
                    latent_shape=latent_shape,
                    pair_num=1
                )

                self.fonts = self.dataset.fonts
                self.chars = self.dataset.chars
                self.common_chars = self.dataset.common_chars
                self.num_fonts = self.dataset.n
                self.num_characters = self.dataset.m

                self.num_styles_per_char = self.num_fonts

                print(f"[accessor] dataset summary:")
                print(f"  - fonts: {self.num_fonts}")
                print(f"  - chars: {self.num_characters}")
                print(f"  - styles per char: {self.num_styles_per_char}")
                print(f"  - total combinations: {self.num_fonts * self.num_characters}")
                print(f"  - indexing: font_idx * {self.num_characters} + char_idx")

                self.font_to_idx = {font: i for i, font in enumerate(self.fonts)}
                self.char_to_idx = {char: i for i, char in enumerate(self.chars)}
                self.fallback_mode = False
                return
            except Exception as e:
                print(f"[warn] FourWayFontPairLatentPTDataset init failed: {e}")
                print("[info] falling back to heuristic accessor")
        else:
            missing_files = []
            if not chars_path:
                missing_files.append("chars_path")
            elif not os.path.exists(chars_path):
                missing_files.append(f"chars_path({chars_path})")
            if not fonts_json:
                missing_files.append("fonts_json")
            elif not os.path.exists(fonts_json):
                missing_files.append(f"fonts_json({fonts_json})")

            print(f"[info] missing files: {', '.join(missing_files)}")
            print("[info] using fallback layout 2056 fonts x 4574 characters")

        self._create_from_old_accessor(pt_path, latent_shape, device)

    def _create_from_old_accessor(self, pt_path: str, latent_shape: Tuple[int, int, int], device: str):
        """Fallback based on legacy LatentAccessor"""
        print("[info] fallback layout assumption 2056 x 4574")

        blob = torch.load(pt_path, map_location="cpu")
        if isinstance(blob, dict) and "latents" in blob:
            latents = blob["latents"]
        else:
            latents = blob

        if isinstance(latents, torch.Tensor):
            if latents.dim() == 4:  # (N, H, W, C)
                self.total_samples = latents.shape[0]
                self.latents_hwc = latents
            else:
                self.total_samples = latents.shape[0]
                self.raw_tensor = latents

            self.num_characters = 312
            self.num_fonts = 128

            expected_total = self.num_fonts * self.num_characters
            if self.total_samples != expected_total:
                print(f"[warn] latent count mismatch")
                print(f"  - actual samples: {self.total_samples}")
                print(f"  - expected samples: {expected_total} (2056×4574)")
                print(f"  - adjusting layout to match data")

                if self.total_samples % 4574 == 0:
                    self.num_fonts = self.total_samples // 4574
                    print(f"  - adjusted to: {self.num_fonts} fonts x 4574 chars")
                elif self.total_samples % 2056 == 0:
                    self.num_characters = self.total_samples // 2056
                    print(f"  - adjusted to: 2056 fonts x {self.num_characters} chars")
                else:
                    sqrt_total = int(math.sqrt(self.total_samples))
                    for chars in [4574, 4000, 3500, sqrt_total]:
                        if self.total_samples % chars == 0:
                            self.num_characters = chars
                            self.num_fonts = self.total_samples // chars
                            break
                    print(f"  - final layout: {self.num_fonts} fonts x {self.num_characters} chars")

            print(f"[info] fallback layout: {self.num_fonts} fonts x {self.num_characters} chars")
            print(f"[info] index layout: font_idx * {self.num_characters} + char_idx")

            self.num_styles_per_char = self.num_fonts

            self.fonts = [f"font_{i:04d}" for i in range(self.num_fonts)]
            self.chars = [f"char_{i:04d}" for i in range(self.num_characters)]
            self.common_chars = self.chars

            self.font_to_idx = {font: i for i, font in enumerate(self.fonts)}
            self.char_to_idx = {char: i for i, char in enumerate(self.chars)}

            self.dataset = None
            self.fallback_mode = True
        else:
            raise RuntimeError("Unsupported latent format")

    def get_by_indices(self, font_idx: int, char_idx: int) -> torch.Tensor:
        """Fetch latent by indices"""
        if font_idx >= self.num_fonts:
            raise IndexError(f"font_idx {font_idx} >= num_fonts {self.num_fonts}")
        if char_idx >= self.num_characters:
            raise IndexError(f"char_idx {char_idx} >= num_characters {self.num_characters}")

        if hasattr(self, 'fallback_mode') and self.fallback_mode:
            idx = font_idx * self.num_characters + char_idx
            if hasattr(self, 'latents_hwc'):
                z_hwc = self.latents_hwc[idx]
                latent = z_hwc.permute(2, 0, 1).contiguous()
            else:
                latent = self.raw_tensor[idx]
            return latent.to(self.device)
        else:
            flat_idx = self.dataset._flat_index(font_idx, char_idx)
            latent = self.dataset._get_chw(flat_idx)
            return latent.to(self.device)

    def get_by_names(self, font_name: str, char_name: str) -> torch.Tensor:
        """Fetch latent by names"""
        if font_name not in self.font_to_idx:
            raise KeyError(f"Font '{font_name}' not found")
        if char_name not in self.char_to_idx:
            raise KeyError(f"Char '{char_name}' not found")

        font_idx = self.font_to_idx[font_name]
        char_idx = self.char_to_idx[char_name]

        return self.get_by_indices(font_idx, char_idx)

    def get(self, content_i: int, style_p: int) -> torch.Tensor:
        """Compatibility helper"""
        return self.get_by_indices(style_p % self.num_fonts, content_i % self.num_characters)


class FullDatasetPairDataset(Dataset):
    """Dataset that decodes latent pairs on the fly."""

    def __init__(
        self,
        accessor: ImprovedLatentAccessor,
        decoder: nn.Module,
        cfg: dict,
        device_used: torch.device,
        task: Literal["content", "style"] = "content",
        num_styles: int = 100,
        num_contents: int = 1000,
        length: int = 50000,
        content_indices: Optional[Sequence[int]] = None,
        style_indices: Optional[Sequence[int]] = None,
        pair_specs: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        self.accessor = accessor
        self.decoder = decoder
        self.cfg = cfg
        self.device_used = device_used
        self.task = task

        self.pair_specs = None
        if pair_specs is not None:
            self.pair_specs = [
                {
                    "anchor_content_idx": int(spec["anchor_content_idx"]),
                    "anchor_style_idx": int(spec["anchor_style_idx"]),
                    "other_content_idx": int(spec["other_content_idx"]),
                    "other_style_idx": int(spec["other_style_idx"]),
                    "is_positive": bool(spec.get("is_positive", spec.get("label", 0))),
                }
                for spec in pair_specs
            ]

            content_values = set()
            style_values = set()
            for spec in self.pair_specs:
                content_values.add(spec["anchor_content_idx"])
                content_values.add(spec["other_content_idx"])
                style_values.add(spec["anchor_style_idx"])
                style_values.add(spec["other_style_idx"])

            self.content_pool = sorted(content_values)
            self.style_pool = sorted(style_values)
            self.num_contents = len(self.content_pool)
            self.num_styles = len(self.style_pool)
            self.length = len(self.pair_specs)
        else:
            max_contents = min(num_contents, self.accessor.num_characters)
            max_styles = min(num_styles, self.accessor.num_styles_per_char)

            if content_indices is not None:
                self.content_pool = [
                    int(idx) for idx in content_indices if 0 <= int(idx) < self.accessor.num_characters
                ]
            else:
                self.content_pool = list(range(max_contents))

            if len(self.content_pool) < 2:
                raise ValueError(
                    f"Need at least two content indices to sample pairs, got {len(self.content_pool)}."
                )

            if style_indices is not None:
                self.style_pool = [
                    int(idx) for idx in style_indices if 0 <= int(idx) < self.accessor.num_styles_per_char
                ]
            else:
                self.style_pool = list(range(max_styles))

            if len(self.style_pool) < 2:
                raise ValueError(
                    f"Need at least two style indices to sample pairs, got {len(self.style_pool)}."
                )

            self.num_styles = len(self.style_pool)
            self.num_contents = len(self.content_pool)
            self.length = length

        try:
            sample = self.accessor.get(0, 0)
            print(f"[dataset] latent sample shape={tuple(sample.shape)}")
        except Exception as exc:
            print(f"[dataset] failed to read latent sample: {exc}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        if self.pair_specs is not None:
            spec = self.pair_specs[idx % len(self.pair_specs)]
            content_i = spec["anchor_content_idx"]
            content_j = spec["other_content_idx"]
            style_p = spec["anchor_style_idx"]
            style_q = spec["other_style_idx"]
            is_positive = bool(spec["is_positive"])
        else:
            content_i = random.choice(self.content_pool)
            content_j = random.choice(self.content_pool)
            while content_j == content_i:
                content_j = random.choice(self.content_pool)

            style_p = random.choice(self.style_pool)
            style_q = random.choice(self.style_pool)
            while style_q == style_p:
                style_q = random.choice(self.style_pool)

            is_positive = random.random() < 0.5

        z1 = self.accessor.get(content_i, style_p)
        x1 = latent_to_tensor(self.decoder, z1, self.cfg, self.device_used)

        if self.task == "content":
            if is_positive:
                z2 = self.accessor.get(content_i, style_q)
                target = 1.0
            else:
                z2 = self.accessor.get(content_j, style_q)
                target = 0.0
        else:
            if is_positive:
                z2 = self.accessor.get(content_j, style_p)
                target = 1.0
            else:
                z2 = self.accessor.get(content_j, style_q)
                target = 0.0

        x2 = latent_to_tensor(self.decoder, z2, self.cfg, self.device_used)

        def safe_name(seq, idx, prefix):
            if hasattr(self.accessor, seq):
                arr = getattr(self.accessor, seq)
                if arr and idx < len(arr):
                    return arr[idx]
            return f"{prefix}_{idx}"

        metadata = {
            "anchor_content_idx": content_i,
            "anchor_content_name": safe_name("chars", content_i, "char"),
            "other_content_idx": content_j,
            "other_content_name": safe_name("chars", content_j, "char"),
            "anchor_style_idx": style_p,
            "anchor_style_name": safe_name("fonts", style_p, "font"),
            "other_style_idx": style_q,
            "other_style_name": safe_name("fonts", style_q, "font"),
            "is_positive": bool(target == 1.0),
            "task": self.task,
        }

        return (
            x1.contiguous(),
            x2.contiguous(),
            torch.tensor(target, dtype=torch.float32),
            metadata,
        )
