# test_normalization_fourway_ptdataset.py
# -------------------------------------------------------
# Usage:
#   python test_normalization_fourway_ptdataset.py
#
# What it does:
#   - Loads true latents from your .pt file (HWC ordering)
#   - Computes per-channel mean/std across (N,H,W)
#   - Injects those stats into FourWayFontPairLatentPTDataset
#   - Samples many items and verifies:
#       * All tensors are (C,H,W) == (4,16,16)
#       * Channel means ~ 0 and stds ~ 1 after normalization
#       * ds.denorm(normalized) == original (numerically)
# -------------------------------------------------------

import os
import json
import yaml
import math
import random
import argparse
from typing import Tuple
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader

# Import your dataset class
from font_dataset import FourWayFontPairLatentPTDataset

# ---------- Paths (edit if needed) ----------
PT_PATH     = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_latents_v2_temp.pt"
CHARS_PATH  = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/intersection_chars_temp.txt"
FONTS_JSON  = "/scratch/yl10337/Content-Style-Disentangled-Representation-Learning/font_list.json"

# Expected latent shape: (C,H,W) and stored source shape: (H,W,C)
EXPECTED_CHW = (4, 16, 16)
EXPECTED_HWC = (16, 16, 4)

# ---------- Helper: compute per-channel stats from true latents ----------
def load_latents_hwc(pt_path: str) -> torch.Tensor:
    blob = torch.load(pt_path, map_location="cpu")
    if isinstance(blob, dict) and "latents" in blob:
        latents = blob["latents"]
    else:
        latents = blob
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor in PT, got {type(latents)}")
    if latents.dim() != 4:
        raise ValueError(f"Expected latents.dim()==4, got {tuple(latents.shape)}")
    return latents.contiguous()  # (N, H, W, C)

@torch.no_grad()
def compute_channel_stats_streaming(latents_hwc: torch.Tensor, batch_size: int = 10000):
    """
    Efficient per-channel mean/std for huge latent tensors.
    Uses running sums instead of loading everything in memory at once.
    """
    N, H, W, C = latents_hwc.shape
    total_count = N * H * W

    sum_c = torch.zeros(C, dtype=torch.float64)
    sumsq_c = torch.zeros(C, dtype=torch.float64)

    for i in tqdm(range(0, N, batch_size), desc="Computing stats"):
        batch = latents_hwc[i : i + batch_size]   # (B,H,W,C)
        flat = batch.view(-1, C).double()         # (B*H*W, C)

        sum_c += flat.sum(dim=0)
        sumsq_c += (flat ** 2).sum(dim=0)

    mean_c = sum_c / total_count
    var_c = (sumsq_c / total_count) - mean_c**2
    std_c = torch.sqrt(torch.clamp(var_c, min=1e-12))

    return mean_c.float(), std_c.float()

# ---------- Tests ----------
@torch.no_grad()
def test_shapes_and_normalization(num_samples: int = 512,
                                  batch_size: int = 32,
                                  seed: int = 123):
    torch.manual_seed(seed)
    random.seed(seed)

    # 1) Load true latents just to compute ground-truth stats
    latents_hwc = load_latents_hwc(PT_PATH)
    N, H, W, C = latents_hwc.shape
    print(f"[INFO] Loaded true latents: {latents_hwc.shape} (expect N,16,16,4)")

    mean_c, std_c = compute_channel_stats_streaming(latents_hwc, batch_size=5000)
    print(f"[INFO] Per-channel mean (HWC order C=last): {mean_c.tolist()}")
    print(f"[INFO] Per-channel std  (HWC order C=last): {std_c.tolist()}")

    # 2) Build dataset WITHOUT stats_yaml (we will inject mean/std tensors)
    ds = FourWayFontPairLatentPTDataset(
        pt_path=PT_PATH,
        chars_path=CHARS_PATH,
        fonts_json=FONTS_JSON,
        latent_shape=EXPECTED_CHW,
        pair_num=max(num_samples, 64),  # ensure enough items available
        stats_yaml=None,                # important: we inject stats below
    )

    # 3) Inject stats (as if coming from stats_yaml)
    # after computing stats
    latent_dtype = latents_hwc.dtype           # likely torch.float16
    ds.mean = mean_c.to(latent_dtype).view(-1,1,1)
    ds.std  = std_c.to(latent_dtype).view(-1,1,1)

    # 4) Sample a bunch of items and check:
    #    - All tensors are (4,16,16)
    #    - Normalized channel means ~ 0, stds ~ 1
    # We'll aggregate many tensors to compute empirical statistics.
    collected = []  # will store many (C,H,W) tensors
    meta = []       # store the index mapping to re-fetch originals
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    with torch.no_grad():
        for i, item in enumerate(loader):
            if len(collected) >= num_samples:
                break

            for key in ("F_A+C_A", "F_A+C_B", "F_B+C_A", "F_B+C_B"):
                z = item[key].squeeze(0)  # (C,H,W)
                assert tuple(z.shape) == EXPECTED_CHW, f"Bad shape {tuple(z.shape)} for key {key}, expected {EXPECTED_CHW}"
                collected.append(z.cpu())

            # For invertibility check, keep the discrete choices for this sample.
            # We can recover originals by recomputing flat indices from the returned metadata.
            fa, fb = item["font_a"][0], item["font_b"][0]
            ca, cb = item["char_a"][0], item["char_b"][0]
            meta.append((fa, fb, ca, cb))

    # Stack to (M, C, H, W)
    Z = torch.stack(collected, dim=0)  # (M, C, H, W)
    M = Z.shape[0]
    print(f"[INFO] Collected {M} normalized tensors of shape {EXPECTED_CHW}")

    # Channel-wise mean/std across all collected pixels
    Z_flat = Z.permute(0, 2, 3, 1).reshape(-1, Z.shape[1])  # (M*H*W, C)
    emp_mean = Z_flat.mean(dim=0)             # (C,)
    emp_std  = Z_flat.std(dim=0, unbiased=False)

    print(f"[CHECK] Empirical mean per channel (should be ~0): {emp_mean.tolist()}")
    print(f"[CHECK] Empirical std  per channel (should be ~1): {emp_std.tolist()}")

    # Reasonable tolerances; you can tighten if you like.
    assert torch.all(emp_mean.abs() < 0.05), \
        f"Empirical means not ~0: {emp_mean}"
    assert torch.all((emp_std > 0.90) & (emp_std < 1.10)), \
        f"Empirical stds not ~1: {emp_std}"

    print("[PASS] Normalization statistics look correct (≈0 mean, ≈1 std).")

    # 5) Invertibility test: denorm(normalized) == original (CHW), numerically.
    # We'll test on a handful of fresh items for clarity.
    for _ in range(8):
        sample = ds[0]  # triggers a fresh random pick internally
        fa, fb = sample["font_a"], sample["font_b"]
        ca, cb = sample["char_a"], sample["char_b"]

        # Helper to reconstruct original (CHW) for a given (font,char)
        def original_chw(font_str: str, char_str: str):
            fi = ds.fonts.index(font_str)
            ci = ds.chars.index(char_str)
            flat_idx = ds._flat_index(fi, ci)
            z_hwc = ds.latents_hwc[flat_idx]          # (H,W,C)
            z_chw = z_hwc.permute(2, 0, 1).contiguous()
            return z_chw

        pairs = [
            ("F_A+C_A", fa, ca),
            ("F_A+C_B", fa, cb),
            ("F_B+C_A", fb, ca),
            ("F_B+C_B", fb, cb),
        ]
        for key, fstr, cstr in pairs:
            z_norm = sample[key]              # (C,H,W), normalized
            z_rec32  = (z_norm.float() * ds.std.float()) + ds.mean.float()
            z_orig32 = original_chw(fstr, cstr).float()
            if not torch.allclose(z_rec32, z_orig32, rtol=1e-6, atol=1e-6):
                mae = (z_rec32 - z_orig32).abs().max().item()
                raise AssertionError(f"denorm(normalized) != original; max|err|={mae}")

    print("[PASS] denorm(normalized) == original for multiple random pairs.")

    print("\nAll normalization tests passed ✅")
    print("— Shapes are correct (4,16,16)")
    print("— Normalization produces ~0 mean and ~1 std per channel")
    print("— denorm() inverts normalization accurately")

if __name__ == "__main__":
    test_shapes_and_normalization(
        num_samples=512,   # increase for tighter empirical stats if desired
        batch_size=32,
        seed=123
    )