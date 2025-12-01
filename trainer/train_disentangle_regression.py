# trainer/train_disentangle_regression.py
# ------------------------------------------------------------
# Train a disentangle REGRESSION model (no diffusion).
# - Optional MLP encoder to produce (content, style) = (1024, 1024) from a 1024-d latent
# - Concatenate two halves (c_A, s_B) or (c_B, s_A) -> 2048-d condition
# - A ConcatRegressor MLP maps 2048-d condition -> 1024-d target latent
# - Keeps: W&B logging, EMA, LR scheduler, VAE-based visualization, checkpoints
# ------------------------------------------------------------

import os
import json
import math
import random
import yaml
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from torchvision import transforms

# === Your project imports ===
from dataset.font_dataset import FourWayFontPairLatentPTDataset, split_fonts, filter_dataset_fonts
from models.mlp import build_residual_mlp
from utils.logger import init_wandb, log_losses
from utils.vae_io import load_models, decode as vae_decode
from utils.lr_scheduler import (
    Scheduler_LinearWarmup,
    Scheduler_LinearWarmup_CosineDecay,
)
from utils.ema import LitEma
from utils.siamese_scores import (
    load_model as load_siamese_model,
    score_pair as siamese_score_pair,
    DEFAULT_CONTENT_CKPT,
    DEFAULT_STYLE_CKPT,
)


# ---------------------------
# Config
# ---------------------------
def load_config(cfg_path: str) -> dict:
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Data
# ---------------------------
def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, callable]:
    font_json = cfg["dataset"]["font_json"]

    train_fonts, valid_fonts = split_fonts(
        font_json,
        train_ratio=cfg["dataset"].get("train_ratio", 0.9),
        seed=cfg.get("seed", 1234),
    )

    print(f"Train fonts: {len(train_fonts)}, Valid fonts: {len(valid_fonts)}")

    latent_size = cfg["dataset"].get("latent_size", 16)
    latent_channels = cfg["dataset"].get("latent_channels", 4)
    latent_shape = (latent_channels, latent_size, latent_size)

    ds_train = FourWayFontPairLatentPTDataset(
        pt_path=cfg["dataset"].get("pt_path"),
        chars_path=cfg["dataset"].get("chars_path"),
        fonts_json=cfg["dataset"].get("font_json"),
        latent_shape=latent_shape,
        pair_num=int(cfg["dataset"].get("pair_num_train", 100000)),
        stats_yaml=cfg["dataset"].get("stats_yaml", None),
    )

    ds_valid = FourWayFontPairLatentPTDataset(
        pt_path=cfg["dataset"].get("pt_path"),
        chars_path=cfg["dataset"].get("chars_path"),
        fonts_json=cfg["dataset"].get("font_json"),
        latent_shape=latent_shape,
        pair_num=int(cfg["dataset"].get("pair_num_valid", 1000)),
        stats_yaml=cfg["dataset"].get("stats_yaml", None),
    )

    filter_dataset_fonts(ds_train, train_fonts)
    filter_dataset_fonts(ds_valid, valid_fonts)

    print(f"Train dataset: {len(ds_train)} samples, Valid dataset: {len(ds_valid)} samples")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    return dl_train, dl_valid, ds_train.denorm


# ---------------------------
# Models
# ---------------------------
def build_content_style_encoder(cfg: dict) -> nn.Module:
    """
    A residual MLP mapping 1024 -> 2048. We use first 1024 as content, last 1024 as style.
    """
    return build_residual_mlp(
        input_dim=cfg.get("input_dim", 1024),
        hidden_dim=cfg.get("hidden_dim", 2048),  # output dim = hidden_dim in your impl
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.1),
        use_layernorm=cfg.get("layernorm", True),
    )


class ConcatRegressor(nn.Module):
    """
    Regress target latent (1024) from condition vector (2048).
    backbone: residual MLP (2048 -> 2048)
    head: linear (2048 -> 1024)
    """
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        out_dim: int = 1024,
    ):
        super().__init__()
        assert input_dim == hidden_dim, \
            "For simplicity we use residual MLP with hidden_dim==input_dim; adjust if needed."
        self.backbone = build_residual_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,    # your impl returns hidden_dim features
            num_layers=num_layers,
            dropout=dropout,
            use_layernorm=use_layernorm,
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        h = self.backbone(cond)
        y = self.head(h)
        return y


# ---------------------------
# Train / Eval
# ---------------------------
def train_disentangle_regression_loop(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = cfg.get("seed", 1234)
    torch.manual_seed(seed)
    random.seed(seed)

    # VAE for visualization
    vae_cfg_path = cfg["vis"]["vae_config"]
    vae_ckpt_path = cfg["vis"]["vae_ckpt"]
    encoder_vae, decoder_vae, vae_cfg, _ = load_models(vae_cfg_path, vae_ckpt_path, device)
    print("VAE decoder ready.")

    # Data
    dl_train, dl_valid, denorm = build_dataloaders(cfg)

    # Encoder switch
    enc_cfg = cfg.get("encoder", {})
    use_encoder = bool(enc_cfg.get("enable", True))
    if use_encoder:
        encoder = build_content_style_encoder(enc_cfg).to(device)
        print(f"Encoder: {encoder}")
    else:
        encoder = None
        print("Encoder: DISABLED (concat raw latents as condition).")

    # Regressor (decoder)
    reg_cfg = cfg.get("regressor", {})
    regressor = ConcatRegressor(
        input_dim=int(reg_cfg.get("input_dim", 2048)),
        hidden_dim=int(reg_cfg.get("hidden_dim", 2048)),
        num_layers=int(reg_cfg.get("num_layers", 4)),
        dropout=float(reg_cfg.get("dropout", 0.1)),
        use_layernorm=bool(reg_cfg.get("layernorm", True)),
        out_dim=int(reg_cfg.get("out_dim", 1024)),
    ).to(device)
    print(f"Regressor: {regressor}")

    # Optimizer
    if use_encoder:
        params = list(encoder.parameters()) + list(regressor.parameters())
    else:
        params = list(regressor.parameters())

    opt = optim.AdamW(
        params,
        lr=float(cfg["train"]["lr"]),
        betas=(0.9, 0.999),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    # LR Scheduler
    scheduler_cfg = cfg["train"].get("scheduler", {})
    sched_type = str(scheduler_cfg.get("type", "none")).lower()
    base_lr = float(cfg["train"]["lr"])
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))

    step_per_epoch = len(dl_train)
    total_steps = step_per_epoch * int(cfg["train"]["epochs"])
    warmup_steps = warmup_epochs * step_per_epoch
    min_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

    if sched_type == "linear-warmup":
        lr_lambda = Scheduler_LinearWarmup(warmup_steps)
    elif sched_type == "linear-warmup_cosine-decay":
        lr_lambda = Scheduler_LinearWarmup_CosineDecay(warmup_steps, total_steps, min_ratio)
    else:  # "none"
        lr_lambda = lambda step: 1.0
    lr_scheduler = LambdaLR(opt, lr_lambda)

    # EMA
    ema_cfg = cfg["train"].get("ema", {})
    use_ema = bool(ema_cfg.get("enable", False))
    ema_decay = float(ema_cfg.get("decay", 0.9999))

    ema_encoder = LitEma(encoder, decay=ema_decay) if (use_ema and use_encoder) else None
    ema_regressor = LitEma(regressor, decay=ema_decay) if use_ema else None

    # W&B
    if cfg.get("wandb", {}).get("enable", False):
        init_wandb(cfg)

    siamese_cfg = cfg.get("siamese", {})
    siamese_models = None
    if siamese_cfg.get("enable", False):
        siamese_device = torch.device(siamese_cfg.get("device", str(device)))
        content_ckpt = siamese_cfg.get("content_ckpt") or DEFAULT_CONTENT_CKPT
        style_ckpt = siamese_cfg.get("style_ckpt") or DEFAULT_STYLE_CKPT
        encoder_type = siamese_cfg.get("encoder_type", "enhanced")

        content_model = load_siamese_model(content_ckpt, "content", siamese_device, encoder_type)
        style_model = load_siamese_model(style_ckpt, "style", siamese_device, encoder_type)

        siamese_models = {
            "content": content_model,
            "style": style_model,
            "device": siamese_device,
            "log_table": bool(siamese_cfg.get("log_table", False)),
        }
        print("Loaded Siamese evaluators for content/style scoring.")

    # Train
    step = 0
    for epoch in range(int(cfg["train"]["epochs"])):
        print(f"\n=== Epoch {epoch + 1}/{cfg['train']['epochs']} ===")
        if use_encoder and encoder is not None:
            encoder.train()
        regressor.train()

        pbar = tqdm(dl_train, desc=f"Epoch {epoch + 1}/{cfg['train']['epochs']}")
        for batch in pbar:
            # Flatten to (B, 1024)
            x_fa_ca = batch["F_A+C_A"].to(device).reshape(batch["F_A+C_A"].size(0), -1)  # c_A + s_A latent
            x_fa_cb = batch["F_A+C_B"].to(device).reshape(batch["F_A+C_B"].size(0), -1)  # target for (c_B, s_A)
            x_fb_ca = batch["F_B+C_A"].to(device).reshape(batch["F_B+C_A"].size(0), -1)  # target for (c_A, s_B)
            x_fb_cb = batch["F_B+C_B"].to(device).reshape(batch["F_B+C_B"].size(0), -1)  # c_B + s_B latent

            # Build 2048-d condition vectors
            if use_encoder and encoder is not None:
                z_fa_ca = encoder(x_fa_ca)  # (B, 2048)
                z_fb_cb = encoder(x_fb_cb)  # (B, 2048)
                content_fa, style_fa = z_fa_ca[:, :1024], z_fa_ca[:, 1024:]
                content_fb, style_fb = z_fb_cb[:, :1024], z_fb_cb[:, 1024:]
                cond_cA_sB = torch.cat([content_fa, style_fb], dim=1)  # -> predict x_fb_ca
                cond_cB_sA = torch.cat([content_fb, style_fa], dim=1)  # -> predict x_fa_cb
            else:
                # Directly concatenate raw latents (each 1024)
                cond_cA_sB = torch.cat([x_fa_ca, x_fb_cb], dim=1)
                cond_cB_sA = torch.cat([x_fb_cb, x_fa_ca], dim=1)

            # Forward (two directions)
            y_pred_1 = regressor(cond_cA_sB)  # predict fb_ca (c_A + s_B)
            y_pred_2 = regressor(cond_cB_sA)  # predict fa_cb (c_B + s_A)

            # Targets
            y_true_1 = x_fb_ca
            y_true_2 = x_fa_cb

            # Loss
            loss_1 = F.mse_loss(y_pred_1, y_true_1)
            loss_2 = F.mse_loss(y_pred_2, y_true_2)
            loss = 0.5 * (loss_1 + loss_2)

            # Optimize
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ((list(encoder.parameters()) if (use_encoder and encoder is not None) else []) + list(regressor.parameters())),
                max_norm=1.0,
            )
            opt.step()
            lr_scheduler.step()

            # EMA
            if ema_regressor is not None:
                ema_regressor(regressor)
            if ema_encoder is not None:
                ema_encoder(encoder)

            # Logs
            if cfg.get("wandb", {}).get("enable", False) and step % int(cfg["wandb"].get("log_interval", 50)) == 0:
                log_losses(step=step, loss_dict={
                    "train/loss": loss.item(),
                    "train/loss_fb_ca": loss_1.item(),
                    "train/loss_fa_cb": loss_2.item(),
                })
                try:
                    lr_val = lr_scheduler.get_last_lr()[0]
                    import wandb
                    wandb.log({"lr": lr_val}, step=step)
                except Exception:
                    pass

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            step += 1

        # Visualization on train set
        if cfg.get("wandb", {}).get("enable", False) and cfg.get("vis", {}).get("enable", False):
            # swap EMA weights into place
            if ema_encoder is not None:
                ema_encoder.store(encoder.parameters())
                ema_encoder.copy_to(encoder)
            if ema_regressor is not None:
                ema_regressor.store(regressor.parameters())
                ema_regressor.copy_to(regressor)

            train_batch = next(iter(dl_train))
            log_sample_images(
                step, train_batch, encoder, regressor,
                decoder_vae, vae_cfg, denorm, device,
                name="train/quad_grid", use_encoder=use_encoder,
                siamese_models=siamese_models,
            )

            # restore
            if ema_encoder is not None:
                ema_encoder.restore(encoder.parameters())
            if ema_regressor is not None:
                ema_regressor.restore(regressor.parameters())

        # Eval
        if (epoch + 1) % int(cfg["eval"]["interval"]) == 0:
            # swap EMA for eval
            if ema_encoder is not None:
                ema_encoder.store(encoder.parameters())
                ema_encoder.copy_to(encoder)
            if ema_regressor is not None:
                ema_regressor.store(regressor.parameters())
                ema_regressor.copy_to(regressor)

            val_loss, vis_batch = evaluate(dl_valid, encoder, regressor, device, use_encoder=use_encoder)

            # restore
            if ema_encoder is not None:
                ema_encoder.restore(encoder.parameters())
            if ema_regressor is not None:
                ema_regressor.restore(regressor.parameters())

            if cfg.get("wandb", {}).get("enable", False):
                log_losses(step=step, loss_dict={"valid/loss": val_loss})

                if cfg.get("vis", {}).get("enable", False):
                    # swap EMA again for vis
                    if ema_encoder is not None:
                        ema_encoder.store(encoder.parameters())
                        ema_encoder.copy_to(encoder)
                    if ema_regressor is not None:
                        ema_regressor.store(regressor.parameters())
                        ema_regressor.copy_to(regressor)

                    log_sample_images(
                        step, vis_batch, encoder, regressor,
                        decoder_vae, vae_cfg, denorm, device,
                        name="val/quad_grid", use_encoder=use_encoder,
                        siamese_models=siamese_models,
                    )

                    # restore
                    if ema_encoder is not None:
                        ema_encoder.restore(encoder.parameters())
                    if ema_regressor is not None:
                        ema_regressor.restore(regressor.parameters())

        # Save
        if (epoch + 1) % int(cfg["train"]["save_interval"]) == 0:
            save_ckpt(
                cfg, epoch, encoder, regressor, opt,
                ema_encoder=ema_encoder if use_ema else None,
                ema_regressor=ema_regressor if use_ema else None,
                lr_scheduler=lr_scheduler,
            )


@torch.no_grad()
def evaluate(loader, encoder, regressor, device, use_encoder: bool = True):
    if use_encoder and encoder is not None:
        encoder.eval()
    regressor.eval()

    losses, sample_batch = [], None

    for batch in loader:
        fa_ca = batch["F_A+C_A"].to(device).view(-1, 1024)
        fa_cb = batch["F_A+C_B"].to(device).view(-1, 1024)
        fb_ca = batch["F_B+C_A"].to(device).view(-1, 1024)
        fb_cb = batch["F_B+C_B"].to(device).view(-1, 1024)

        if use_encoder and encoder is not None:
            z_a, z_b = encoder(fa_ca), encoder(fb_cb)
            cond1 = torch.cat([z_a[:, :1024], z_b[:, 1024:]], 1)  # -> predict fb_ca
            cond2 = torch.cat([z_b[:, :1024], z_a[:, 1024:]], 1)  # -> predict fa_cb
        else:
            cond1 = torch.cat([fa_ca, fb_cb], 1)
            cond2 = torch.cat([fb_cb, fa_ca], 1)

        y_hat1 = regressor(cond1)
        y_hat2 = regressor(cond2)

        loss = 0.5 * (F.mse_loss(y_hat1, fb_ca) + F.mse_loss(y_hat2, fa_cb))
        losses.append(loss.item())

        if sample_batch is None:
            sample_batch = batch

    return sum(losses) / max(1, len(losses)), sample_batch


@torch.no_grad()
def log_sample_images(
    step, batch, encoder, regressor,
    decoder, vae_cfg, denorm, device,
    name: str, use_encoder: bool = True, num_show: int = 4,
    siamese_models: dict = None,
):
    import wandb
    to_tensor = transforms.ToTensor()

    idx = random.sample(range(batch["F_A+C_A"].size(0)),
                        k=min(num_show, batch["F_A+C_A"].size(0)))

    fa_ca = batch["F_A+C_A"][idx].to(device).view(-1, 1024)
    fb_cb = batch["F_B+C_B"][idx].to(device).view(-1, 1024)
    fa_cb = batch["F_A+C_B"][idx].to(device).view(-1, 1024)  # GT for (c_B, s_A)
    fb_ca = batch["F_B+C_A"][idx].to(device).view(-1, 1024)  # GT for (c_A, s_B)

    if use_encoder and encoder is not None:
        z_a = encoder(fa_ca)
        z_b = encoder(fb_cb)
        cond1 = torch.cat([z_a[:, :1024], z_b[:, 1024:]], dim=1)  # for fb_ca
        cond2 = torch.cat([z_b[:, :1024], z_a[:, 1024:]], dim=1)  # for fa_cb
    else:
        cond1 = torch.cat([fa_ca, fb_cb], dim=1)
        cond2 = torch.cat([fb_cb, fa_ca], dim=1)

    lat_fb_ca = regressor(cond1)  # predicted
    lat_fa_cb = regressor(cond2)  # predicted

    def reshape_and_decode(flat_latents):
        latents = flat_latents.view(-1, 4, 16, 16)
        decoded_imgs = []
        for lat in latents:
            lat = denorm(lat)
            img = vae_decode(lat, decoder, vae_cfg, device)  # PIL
            decoded_imgs.append(to_tensor(img))
        return decoded_imgs  # list of [1, H, W]

    img_gt_fa_cb = reshape_and_decode(fa_cb)
    img_gt_fb_ca = reshape_and_decode(fb_ca)
    img_fa_ca = reshape_and_decode(fa_ca)
    img_fb_cb = reshape_and_decode(fb_cb)
    img_gen_fa_cb = reshape_and_decode(lat_fa_cb)   # generated
    img_gen_fb_ca = reshape_and_decode(lat_fb_ca)   # generated

    # Row per example: [GT_fa_cb | GT_fb_ca | Ref_fa_ca | Ref_fb_cb | Gen_fa_cb | Gen_fb_ca]
    rows = []
    for i in range(len(idx)):
        row = torch.cat([
            img_gt_fa_cb[i],
            img_gt_fb_ca[i],
            img_fa_ca[i],
            img_fb_cb[i],
            img_gen_fa_cb[i],
            img_gen_fb_ca[i],
        ], dim=2)  # concat width-wise (C, H, 6*W)
        rows.append(row)

    grid = make_grid(torch.stack(rows), nrow=1, padding=2)
    wandb.log({name: wandb.Image(grid)}, step=step)

    if not siamese_models or wandb.run is None:
        return

    img_fa_ca = torch.stack(img_fa_ca)
    img_fb_cb = torch.stack(img_fb_cb)
    img_gen_fa_cb = torch.stack(img_gen_fa_cb)
    img_gen_fb_ca = torch.stack(img_gen_fb_ca)

    content_model = siamese_models["content"]
    style_model = siamese_models["style"]
    siamese_device = siamese_models.get("device", device)

    def batch_scores(preds, content_refs, style_refs):
        content_scores = []
        style_scores = []
        for pred, c_ref, s_ref in zip(preds, content_refs, style_refs):
            content_scores.append(siamese_score_pair(content_model, pred, c_ref, siamese_device))
            style_scores.append(siamese_score_pair(style_model, pred, s_ref, siamese_device))
        return content_scores, style_scores

    fb_content_scores, fb_style_scores = batch_scores(
        img_gen_fb_ca, img_fa_ca, img_fb_cb
    )
    fa_content_scores, fa_style_scores = batch_scores(
        img_gen_fa_cb, img_fb_cb, img_fa_ca
    )

    def avg(values):
        return float(sum(values) / max(len(values), 1)) if values else 0.0

    wandb.log(
        {
            f"{name}/fb_ca_content_score": avg(fb_content_scores),
            f"{name}/fb_ca_style_score": avg(fb_style_scores),
            f"{name}/fa_cb_content_score": avg(fa_content_scores),
            f"{name}/fa_cb_style_score": avg(fa_style_scores),
        },
        step=step,
    )


def save_ckpt(
    cfg: dict,
    epoch: int,
    encoder: nn.Module,
    regressor: nn.Module,
    optimizer: optim.Optimizer,
    ema_encoder=None,
    ema_regressor=None,
    lr_scheduler=None,
) -> None:
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
    ckpt = {
        "epoch": epoch,
        "regressor": regressor.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if encoder is not None:
        ckpt["encoder"] = encoder.state_dict()
    if ema_encoder is not None:
        ckpt["ema_encoder"] = ema_encoder.state_dict()
    if ema_regressor is not None:
        ckpt["ema_regressor"] = ema_regressor.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(ckpt, ckpt_path)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # You can still use the old ddpm config; the "denoiser" section will be ignored here.
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_loop(cfg)
