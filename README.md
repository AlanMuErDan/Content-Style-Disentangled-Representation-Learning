```
# Content–Style Disentangled Representation Learning for Chinese Font Generation

A general, architecture-agnostic representation-learning framework for disentangling Chinese font content and style in latent space. The system supports reconstruction, style transfer, interpolation, and synthesis of unseen font–character combinations.

---

## 1. Overview

This repository implements a two-stage generative pipeline:

1) **Stage 1 – VAE**: Compress 128×128 grayscale glyphs into compact latents (4×16×16).  
2) **Stage 2 – Disentanglement**: Learn separate content (Zc) and style (Zs) latent spaces using cross-pairing, MCL adversarial loss, SCCR/CCR contrastive regularization, and either DDPM or Flow Matching denoising.  
The design is architecture-agnostic and works with MAR (MLP AdaLN), UNet (LDM-style), and Patch/Transformer UNet-Pro.

---

## 2. Pipeline (High-Level)

    Stage 1: VAE                                  Stage 2: Disentanglement

┌──────────────────────┐   ┌──────────────────────────────────────────┐
│ 128×128 glyph image  │ → │ VAE → latent z                           │
└──────────────────────┘   │ Cross-pairing (4-way)                     │
                           │ Adversarial (MCL)                         │
                           │ Contrastive (SCCR / CCR)                  │
                           │ Diffusion / Flow Matching denoising       │
                           └──────────────────────────────────────────┘
                                        │
                                        ▼
                           Synthesis: combine Zc(A) + Zs(B) → glyph

---

## 3. QuickStart

```bash
# 1) Train VAE (config.yaml with train_stage: VAE)
python main_train.py

# 2) Extract latents (pt or lmdb)
python inference/VAE_inference.py --mode pt

# 3) Train disentanglement model (e.g., MAR + Flow Matching)
python main_train.py   # set train_stage: disentangle_mar / disentangle_sd / disentangle_sd_pro

# 4) Generate new glyphs (pseudo)
python - <<'PY'
# load disentangle model + VAE decoder, build cond (Zc+Zs), run scheduler.p_sample_loop, then decode
PY
```

---

## 4. Core Ideas

- Two-stage model: VAE compression → disentangled generative modeling.  
- Latent space: VAE maps 128×128 → 4×16×16.  
- Cross-pairing: enforce Zc/Zs swappability across fonts and characters (F_A+C_A, F_A+C_B, F_B+C_A, F_B+C_B).  
- Disentanglement losses:  
  - MCL adversarial classifiers strip unwanted content/style.  
  - SCCR/CCR contrastive refinement for style/content separation.  
- Noise models: DDPM or Flow Matching.  
- Backbones available:  
  - MAR (MLP AdaLN) – fast, stable, generalizes well (recommended default).  
  - UNet – highest reconstruction fidelity (slower/heavier).  
  - UNet-Pro – more capacity for complex styles (patch/transformer).  
- Architecture-agnostic: identical training strategy across MLP/CNN/Transformer.

---

## 5. Repository Structure

```
├── configs/                 # YAML configs (config*.yaml for VAE, MAR, SD, SD-Pro, regression)
├── trainer/                 # Training loops and entry points
├── dataset/                 # LMDB/PT dataset logic, splits, latent stats
├── models/                  # VAE, MLP AdaLN, UNet/UNet-Pro, diffusion/FM schedulers, Siamese
├── inference/               # VAE latent extraction, shard merge
├── utils/                   # Metrics, LR schedule, visualization, logging, siamese scores
├── font_gen.py              # Render glyphs from .ttf to PNG/LMDB
├── checkpoints/             # Pretrained weights (VAE, Siamese, MAR/DDPM, SD/SD-Pro)
└── final_codebooks/         # Content/style codebooks
```

---

## 6. Environment

- Python ≥ 3.9, PyTorch ≥ 1.12 (CUDA recommended)  
- Dependencies: lmdb, Pillow, fontTools, tqdm, lpips, wandb, pyyaml

Install:

```bash
pip install torch torchvision lmdb Pillow fonttools tqdm lpips wandb pyyaml
```

---

## 7. Data Preparation

1) **Fonts & charset**  
```
fonts/
  FontA.ttf
  FontB.ttf
char_list.txt   # one character per line
```

2) **Render glyph dataset (PNG or LMDB)**  
```bash
python font_gen.py \
  --fonts fonts/ \
  --char_list char_list.txt \
  --output data/ \
  --use_lmdb --lmdb_path font_data.lmdb
```

3) **(Optional) latent statistics**  
```
python dataset/compute_latent_stats.py --lmdb font_latents_*.lmdb --shape 4 16 16
```

---

## 8. Stage 1 — VAE Training

- Config: `configs/config.yaml` (`train_stage: VAE`, data paths, latent_channels=4, GAN/EMA/LPIPS, LR schedule).  
- Train: `python main_train.py`  
- Outputs: `checkpoints/vae_best_ckpt.pth`, wandb logs (if enabled).

---

## 9. Latent Extraction

```bash
python inference/VAE_inference.py --mode pt   # or --mode lmdb
```

Outputs: `font_latents_*.pt` or `font_latents_*.lmdb` (used directly in Stage 2). Paths are configurable at the top of the script.

---

## 10. Stage 2 — Disentanglement Training

- Entrypoints: `main_train.py` (config.yaml) or `main_train1/2/3.py` (config_1/2/3.yaml).  
- Algorithms:  
  - `disentangle_mar` – MLP AdaLN + DDPM/FM.  
  - `disentangle_sd` – Light UNet (LDM-style).  
  - `disentangle_sd_pro` – Patch/Transformer UNet.  
  - `disentangle_regression` – baseline MLP denoising/regression.  
- Important flags:  
  - `algo.type`: `ddpm` | `flow_matching`  
  - `contrastive.enable`, `lambda`, style/content weights, `tau`, `detach`  
  - `adv.enable` for MCL adversarial classifiers  
  - `train.cfg` for classifier-free guidance (drop prob & scale)  
  - `vis.enable` to decode samples with VAE  
  - `wandb.enable` for logging  
- Example:  
```bash
python main_train.py   # set train_stage: disentangle_mar / disentangle_sd / disentangle_sd_pro
```

---

## 11. Sampling & Synthesis

Pseudo-steps to generate a glyph:

1) Obtain Zc (content) and Zs (style) from the encoder or codebook.  
2) Concatenate to form the conditioning vector.  
3) Run `scheduler.p_sample_loop` (DDPM or FM) with CFG if enabled.  
4) Decode 4×16×16 latent with `vae_decode` to 128×128 glyph.

---

## 12. Evaluation

- **Pixel metrics**: PSNR, SSIM, L1/L2 reconstruction (`utils/evaluate_basic.py`).  
- **Content/Style Siamese metrics**: `utils/siamese_scores.py`; checkpoints in `checkpoints/content_*`, `checkpoints/style_*`.

---

## 13. Results

- FM consistently outperforms DDPM.  
- UNet achieves best reconstruction but trains slower.  
- MAR balances efficiency and generalization and is the default choice.  
- Siamese metrics are more reliable for disentanglement evaluation.

Example qualitative swap (ASCII sketch):
```
Content A     Style B      Output (A in B-style)
███████       ██░░██       ██░░███
██░░██   +    ██████   →   ███░███
█░░░██        ██░░██       █░░░███
```

Example quantitative snapshot (illustrative):
```
Model        PSNR↑   SSIM↑   C-Siam↑   S-Siam↑
UNet + FM    22.7    0.93     0.91      0.87
MAR + FM     21.3    0.91     0.89      0.88
DDPM + MAR   19.2    0.88     0.84      0.83
```

---

## 14. Troubleshooting & Pitfalls

- LMDB access bottleneck: use `readahead=False` (already set) and cache `lmdb_keys.json`.  
- Broken font cmap: verify supported char counts from `font_gen.py` logs.  
- Latent shape mismatch: default 4×16×16; update dataset + VAE + denoiser if changed.  
- DDPM instability: warm up with uniform timesteps or switch to Flow Matching.  
- Cross-pairing: keep the four paired views aligned; do not reshuffle independently.  
- Siamese metrics: backbone must match checkpoint (vgg/enhanced) or scores are unreliable.

---

## 15. Pretrained Models

- `checkpoints/vae_best_ckpt.pth`  
- `checkpoints/content_best_ckpt_vgg_full.pth`, `checkpoints/style_best_ckpt_vgg_full.pth`  
- `checkpoints/ddpm_disentangle/epoch_0499.pth`  
- `checkpoints/sd_unet/epoch_*.pth` (0–999)  
- `checkpoints/sd_unet_pro/epoch_*.pth` (0–999)  
- `final_codebooks/codebook_*`

---

## 16. Citation

```
@misc{content_style_disentangle_2025,
  title  = {Content–Style Disentangled Representation Learning for Chinese Font Generation},
  author = {Guodong Zheng and Ruilin Wu and Yuanheng Li},
  year   = {2025}
}
```

---

## 17. License

MIT License.
```
