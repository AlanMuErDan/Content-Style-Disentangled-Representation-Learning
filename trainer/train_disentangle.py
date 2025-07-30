# trainer/train_disentangle.py
# -------------------------------------------------------------
#  Content–Style Disentanglement  +  DDPM Denoising (MLP-based)
# -------------------------------------------------------------
#  Author : ChatGPT  (2025-07-27)
#  Requires : PyTorch ≥ 2.1, tqdm, pyyaml
# -------------------------------------------------------------
import os
import json
import math
import random
from pathlib import Path
from typing import Tuple
import wandb 

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# -----------------------------
#  Local modules (existing)
# -----------------------------
from dataset.font_dataset import FourWayFontPairLatentLMDBDataset
from models.mlp import build_residual_mlp, SimpleMLPAdaLN          # ← 已在 models/mlp.py 中实现
from utils.logger import init_wandb, log_losses, log_images        # ← 如无需要，可屏蔽

# ---------------------------------
#  New imports for image logging
# ---------------------------------
from utils.vae_io import load_models, decode as vae_decode
import random as pyrand           # 避免与 torch.random 混淆
from torchvision.utils import make_grid
from torchvision import transforms
# -------------------------------------------------------------


# =============================================================
# 1. Diffusion helpers
# =============================================================
def make_beta_schedule(timesteps: int = 1000,
                       beta_start: float = 1e-4,
                       beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class DDPMNoiseScheduler:
    """Pre-computes sqrt(ᾱ_t) 等常量，便于训练快速取用。"""
    def __init__(self, timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 2e-2,
                 device: torch.device = torch.device("cpu")) -> None:
        self.timesteps = timesteps
        self.betas = make_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion：得到 x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise


# =============================================================
# 2. Content / Style Encoder  (Residual MLP)
# =============================================================
def build_content_style_encoder(cfg: dict) -> nn.Module:
    """
    返回一个把 1024-D latent → 2048-D 向量的编码器。
    “上半 1024 = content, 下半 1024 = style”
    """
    return build_residual_mlp(
        input_dim=cfg.get("input_dim", 1024),
        hidden_dim=cfg.get("hidden_dim", 2048),
        num_layers=cfg.get("num_layers", 4),
    )


# =============================================================
# 3. 配置加载 & 训练主循环
# =============================================================
def load_config(cfg_path: str) -> dict:
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def split_fonts(json_path: str,
                train_ratio: float = 0.9,
                seed: int = 42) -> Tuple[set, set]:
    """根据字体 list（json）划分  Train / Valid."""
    with open(json_path, "r") as f:
        fonts = json.load(f)
    random.Random(seed).shuffle(fonts)
    n_train = int(len(fonts) * train_ratio)
    train_fonts = set(fonts[:n_train])
    valid_fonts = set(fonts[n_train:])
    return train_fonts, valid_fonts


def filter_dataset_fonts(ds: FourWayFontPairLatentLMDBDataset, allowed_fonts: set) -> None:
    ds.fonts = [f for f in ds.fonts if f in allowed_fonts]
    assert len(ds.fonts) > 1, "Filtered dataset 没有足够字体可用"

    # 重新计算 train/valid 子集的公共字符
    ds.common_chars = list(
        set.intersection(*(set(ds.data[f].keys()) for f in ds.fonts))
    )
    assert len(ds.common_chars) >= 2, "过滤后字体的公共字符不足 2 个"


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    lmdb_path = cfg["dataset"]["lmdb_path"]
    font_json = cfg["dataset"]["font_json"]

    train_fonts, valid_fonts = split_fonts(
        font_json,
        train_ratio=cfg["dataset"].get("train_ratio", 0.9),
        seed=cfg.get("seed", 1234),
    )

    print(f"Train fonts: {len(train_fonts)}, Valid fonts: {len(valid_fonts)}")

    # Dataset（共享 env，减少 IO）
    latent_size = cfg["dataset"].get("latent_size", 16)
    latent_channels = cfg["dataset"].get("latent_channels", 4)
    latent_shape = (latent_channels, latent_size, latent_size)

    ds_train = FourWayFontPairLatentLMDBDataset(
        lmdb_path=lmdb_path,
        latent_shape=latent_shape,
        pair_num=10000
    )
    print(f"Train dataset loaded: {len(ds_train)} samples")

    ds_valid = FourWayFontPairLatentLMDBDataset(
        lmdb_path=lmdb_path,
        pair_num=1000
    )
    print(f"Validation dataset loaded: {len(ds_valid)} samples")
    filter_dataset_fonts(ds_train, train_fonts)
    filter_dataset_fonts(ds_valid, valid_fonts)

    print(f"Filtered train dataset: {len(ds_train)} samples, "
          f"Filtered valid dataset: {len(ds_valid)} samples")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    print(f"Train DataLoader created with batch size {cfg['train']['batch_size']}")


    dl_valid = DataLoader(
        ds_valid,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    print(f"Validation DataLoader created with batch size {cfg['eval']['batch_size']}")

    
    return dl_train, dl_valid


def predict_x0(x_t, eps_pred, t_idx, scheduler):
    """
    根据  ε̂  反推出  x̂_0 ；x_t / eps_pred shape = (B, 1024)
    """
    dev = x_t.device
    sqrt_alpha_bar_t = scheduler.sqrt_alphas_cumprod[t_idx].to(dev).view(-1, 1)
    sqrt_one_minus   = scheduler.sqrt_one_minus_alphas_cumprod[t_idx].to(dev).view(-1, 1)
    return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar_t

def log_sample_images(step, batch, eps_pred_1, eps_pred_2,
                      t_idx, scheduler, decoder, vae_cfg, device):
    """
    随机取 4 组 → 6 张图：原始 AA / BB，GT AB / BA，Pred AB / BA
    """
    # ----------- 从 batch 随机抽 4 个索引 -----------
    B = batch["F_A+C_A"].size(0)
    sel = pyrand.sample(range(B), k=min(4, B))

    imgs = []  # 将拼成一张 grid 上传
    for i in sel:
        # --- 1) 原始 self-recon （content+style 保持一致）---
        fa_ca = batch["F_A+C_A"][i]
        fb_cb = batch["F_B+C_B"][i]

        # --- 2) GT cross-pair latent ---
        fa_cb_gt = batch["F_A+C_B"][i]
        fb_ca_gt = batch["F_B+C_A"][i]

        # --- 3) 预测的 cross-pair latent ---
        x_t_fb_ca = batch["x_t_fb_ca"][i]     # 见 evaluate() 修改
        x_t_fa_cb = batch["x_t_fa_cb"][i]

        # 使用训练时得到的 eps_pred（已经是对应 index 预测的）
        fa_cb_pred = predict_x0(
            x_t_fa_cb, eps_pred_2[i], t_idx[i], scheduler
        ).reshape(fa_cb_gt.shape)
        fb_ca_pred = predict_x0(
            x_t_fb_ca, eps_pred_1[i], t_idx[i], scheduler
        ).reshape(fb_ca_gt.shape)

        # ---- Decode 6 latent → PIL ----
        for lat in (fa_ca, fb_cb, fa_cb_gt, fb_ca_gt, fa_cb_pred, fb_ca_pred):
            pil = vae_decode(lat.cpu(), decoder, vae_cfg, device)
            imgs.append(pil)

    # --------- 拼 grid & 上传 ---------
    grid = make_grid(
        torch.stack([transforms.ToTensor()(p) for p in imgs]), nrow=6, padding=2
    )

    wandb.log({"val/quad_grid": wandb.Image(grid)}, step=step)


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = cfg.get("seed", 1234)
    torch.manual_seed(seed)
    random.seed(seed)

    # ----------  VAE decoder for visualisation ----------
    vae_cfg_path = cfg["vis"]["vae_config"]
    vae_ckpt_path = cfg["vis"]["vae_ckpt"]
    encoder_vae, decoder_vae, vae_cfg, _ = load_models(
        vae_cfg_path, vae_ckpt_path, device
    )
    print("[Vis] VAE decoder ready.")

    # ---------------  Dataset / Dataloader  ---------------
    dl_train, dl_valid = build_dataloaders(cfg)

    # ---------------  Models ---------------
    encoder = build_content_style_encoder(cfg["encoder"]).to(device)
    print("---------------  Encoder ---------------")
    print(f"Encoder: {encoder}")

    denoiser = SimpleMLPAdaLN(
        in_channels=1024,                     # x_t shape
        model_channels=cfg["denoiser"]["model_channels"],
        out_channels=1024,                    # predict ε
        z_channels=2048,                      # condition vector
        num_res_blocks=cfg["denoiser"].get("num_res_blocks", 4),
        grad_checkpointing=cfg["denoiser"].get("grad_ckpt", False),
    ).to(device)
    print("---------------  Denoiser ---------------")
    print(f"Denoiser: {denoiser}")

    # ---------------  Optimizer & Scheduler ---------------
    opt = optim.AdamW(
        list(encoder.parameters()) + list(denoiser.parameters()),
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    # ---------------  Diffusion Schedule ---------------
    scheduler = DDPMNoiseScheduler(
        timesteps   = cfg["denoiser"].get("timesteps", 1000),
        beta_start  = float(cfg["denoiser"].get("beta_start", 1e-4)),
        beta_end    = float(cfg["denoiser"].get("beta_end",  2e-2)),
        device=device,
    )

    # ---------------  Logging ---------------
    if cfg.get("wandb", {}).get("enable", False):
        init_wandb(cfg)

    # ======================================================
    #                >>>  TRAIN LOOP  <<<
    # ======================================================
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{cfg['train']['epochs']} ===")
        encoder.train(); denoiser.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            # --------------------------------------------
            # 1. 取四元组 latent，并 flatten 为 (B, 1024)
            # --------------------------------------------
            x_fa_ca = batch["F_A+C_A"].to(device).reshape(batch["F_A+C_A"].size(0), -1)
            x_fa_cb = batch["F_A+C_B"].to(device).reshape(batch["F_A+C_B"].size(0), -1)
            x_fb_ca = batch["F_B+C_A"].to(device).reshape(batch["F_B+C_A"].size(0), -1)
            x_fb_cb = batch["F_B+C_B"].to(device).reshape(batch["F_B+C_B"].size(0), -1)

            # print(f"Batch size: {x_fa_ca.size(0)}")


            # --------------------------------------------
            # 2. 编码  (content | style)
            # --------------------------------------------
            z_fa_ca = encoder(x_fa_ca)   # (B, 2048)
            z_fb_cb = encoder(x_fb_cb)   # (B, 2048)

            # print(f"Encoded shapes: z_fa_ca: {z_fa_ca.shape}, z_fb_cb: {z_fb_cb.shape}")

            content_fa = z_fa_ca[:, :1024]    # c_A
            style_fb  = z_fb_cb[:, 1024:]     # s_B

            content_fb = z_fb_cb[:, :1024]    # c_B
            style_fa   = z_fa_ca[:, 1024:]    # s_A

            # 条件向量
            cond_cA_sB = torch.cat([content_fa, style_fb], dim=1)   # 用于 denoise x_fb_ca
            cond_cB_sA = torch.cat([content_fb, style_fa], dim=1)   # 用于 denoise x_fa_cb

            # print(f"The shape of condition cA_sB: {cond_cA_sB.shape}, cB_sA: {cond_cB_sA.shape}")

            # --------------------------------------------
            # 3. 前向扩散 (q_sample)
            # --------------------------------------------
            B = x_fb_ca.size(0)
            t = scheduler.sample_timesteps(B, device)        # 随机 t∈[0,T)
            noise = torch.randn_like(x_fb_ca)
            x_t_fb_ca = scheduler.q_sample(x_fb_ca, t, noise)

            noise2 = torch.randn_like(x_fa_cb)
            x_t_fa_cb = scheduler.q_sample(x_fa_cb, t, noise2)  # 共享同一 t，可简化；也可单独采样

            # --------------------------------------------
            # 4. 噪声预测  ε_θ(x_t, t, cond)
            # --------------------------------------------
            eps_pred_1 = denoiser(x_t_fb_ca, t, cond_cA_sB)
            eps_pred_2 = denoiser(x_t_fa_cb, t, cond_cB_sA)

            loss_1 = F.mse_loss(eps_pred_1, noise)
            loss_2 = F.mse_loss(eps_pred_2, noise2)
            loss = 0.5 * (loss_1 + loss_2)

            # print(f"Loss 1: {loss_1.item():.4f}, Loss 2: {loss_2.item():.4f}, Total Loss: {loss.item():.4f}")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) +
                                           list(denoiser.parameters()),
                                           max_norm=1.0)
            opt.step()

            # 记录
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if cfg.get("wandb", {}).get("enable", False) and step % cfg["wandb"].get("log_interval", 50) == 0:
                log_losses(step=step, loss_dict={"train/loss": loss.item()})
            step += 1

        # ==================================================
        #             Validation  (optional)
        # ==================================================
        if (epoch + 1) % cfg["eval"]["interval"] == 0:
            val_loss, vis_batch, vis_e1, vis_e2, vis_t = evaluate(
                 dl_valid, encoder, denoiser, scheduler, device
            )
            if cfg.get("wandb", {}).get("enable", False):
                log_losses(step=step, loss_dict={"valid/loss": val_loss})
                # ---- 可视化，开关由 cfg 控制 ----
                if cfg["vis"].get("enable", True):
                    log_sample_images(
                        step, vis_batch, vis_e1, vis_e2,
                        vis_t, scheduler, decoder_vae, vae_cfg, device
                    )

        # Save ckpt
        if (epoch + 1) % cfg["train"]["save_interval"] == 0:
            save_ckpt(cfg, epoch, encoder, denoiser, opt)


@torch.no_grad()
def evaluate(dataloader: DataLoader,
             encoder: nn.Module,
             denoiser: nn.Module,
             scheduler: DDPMNoiseScheduler,
             device: torch.device) -> float:
    encoder.eval(); denoiser.eval()
    sample_batch = None
    sample_eps1 = sample_eps2 = sample_t = None
    losses = []
    for batch in dataloader:
        x_fa_ca = batch["F_A+C_A"].to(device).reshape(batch["F_A+C_A"].size(0), -1)
        x_fa_cb = batch["F_A+C_B"].to(device).reshape(batch["F_A+C_B"].size(0), -1)
        x_fb_ca = batch["F_B+C_A"].to(device).reshape(batch["F_B+C_A"].size(0), -1)
        x_fb_cb = batch["F_B+C_B"].to(device).reshape(batch["F_B+C_B"].size(0), -1)

        z_fa_ca = encoder(x_fa_ca)
        z_fb_cb = encoder(x_fb_cb)

        content_fa, style_fb = z_fa_ca[:, :1024], z_fb_cb[:, 1024:]
        content_fb, style_fa = z_fb_cb[:, :1024], z_fa_ca[:, 1024:]

        cond_cA_sB = torch.cat([content_fa, style_fb], dim=1)
        cond_cB_sA = torch.cat([content_fb, style_fa], dim=1)

        B = x_fb_ca.size(0)
        t = scheduler.sample_timesteps(B, device)
        noise = torch.randn_like(x_fb_ca)
        noise2 = torch.randn_like(x_fa_cb)
        x_t_fb_ca = scheduler.q_sample(x_fb_ca, t, noise)
        x_t_fa_cb = scheduler.q_sample(x_fa_cb, t, noise2)

        eps_pred_1 = denoiser(x_t_fb_ca, t, cond_cA_sB)
        eps_pred_2 = denoiser(x_t_fa_cb, t, cond_cB_sA)

        loss_1 = F.mse_loss(eps_pred_1, noise, reduction="mean")
        loss_2 = F.mse_loss(eps_pred_2, noise2, reduction="mean")
        losses.append(0.5 * (loss_1 + loss_2).item())

        if sample_batch is None:               # 只保存第一批用于可视化
            sample_batch = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in batch.items()
                }
            sample_batch["x_t_fb_ca"] = x_t_fb_ca.cpu()
            sample_batch["x_t_fa_cb"] = x_t_fa_cb.cpu()
            sample_eps1 = eps_pred_1.detach().cpu()
            sample_eps2 = eps_pred_2.detach().cpu()
            sample_t = t.cpu()

    avg_loss = sum(losses) / len(losses)        
    return avg_loss, sample_batch, sample_eps1, sample_eps2, sample_t


def save_ckpt(cfg: dict, epoch: int,
              encoder: nn.Module,
              denoiser: nn.Module,
              optimizer: optim.Optimizer) -> None:
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
    ckpt = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "denoiser": denoiser.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_path)


# =============================================================
# 4. CLI
# =============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ddpm_disentangle.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)