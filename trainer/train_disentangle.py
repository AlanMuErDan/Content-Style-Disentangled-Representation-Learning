# trainer/train_disentangle.py

import os
import json
import math
import random
import wandb 
import yaml
import math 
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

from dataset.font_dataset import FourWayFontPairLatentPTDataset, split_fonts, filter_dataset_fonts

from models.mlp import build_residual_mlp, SimpleMLPAdaLN 
from models.DDPM import GaussianDiffusion as DDPMNoiseScheduler

from utils.logger import init_wandb, log_losses
from utils.vae_io import load_models, decode as vae_decode
from utils.lr_scheduler import (
    Scheduler_LinearWarmup,
    Scheduler_LinearWarmup_CosineDecay,
)
from utils.ema import LitEma

def load_config(cfg_path: str) -> dict:
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)



def build_content_style_encoder(cfg: dict) -> nn.Module:
    return build_residual_mlp(
        input_dim=cfg.get("input_dim", 1024),
        hidden_dim=cfg.get("hidden_dim", 2048),
        num_layers=cfg.get("num_layers", 4),
    )



def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
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

    # ds_train = FourWayFontPairLatentLMDBDataset(
    #     lmdb_path=lmdb_path,
    #     latent_shape=latent_shape,
    #     pair_num=10000,
    #     stats_yaml=cfg["dataset"].get("stats_yaml", None)
    # )

    ds_train = FourWayFontPairLatentPTDataset(
        pt_path = cfg["dataset"].get("pt_path"),
        chars_path = cfg["dataset"].get("chars_path"),
        fonts_json = cfg["dataset"].get("font_json"),
        latent_shape = latent_shape,
        pair_num=100000,
        stats_yaml=cfg["dataset"].get("stats_yaml", None)
    )

    # ds_valid = FourWayFontPairLatentLMDBDataset(
    #     lmdb_path=lmdb_path,
    #     latent_shape=latent_shape,
    #     pair_num=1000,
    #     stats_yaml=cfg["dataset"].get("stats_yaml", None)
    # )

    ds_valid = FourWayFontPairLatentPTDataset(
        pt_path = cfg["dataset"].get("pt_path"),
        chars_path = cfg["dataset"].get("chars_path"),
        fonts_json = cfg["dataset"].get("font_json"),
        latent_shape = latent_shape,
        pair_num=1000,
        stats_yaml=cfg["dataset"].get("stats_yaml", None)
    )

    filter_dataset_fonts(ds_train, train_fonts)
    filter_dataset_fonts(ds_valid, valid_fonts)

    print(f"Train dataset: {len(ds_train)} samples, "
          f"Valid dataset: {len(ds_valid)} samples")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True, 
        drop_last=False,
        persistent_workers=True
    )
    
    return dl_train, dl_valid, ds_train.denorm


def train_disentangle_loop(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = cfg.get("seed", 1234)
    torch.manual_seed(seed)
    random.seed(seed)

    vae_cfg_path = cfg["vis"]["vae_config"]
    vae_ckpt_path = cfg["vis"]["vae_ckpt"]
    encoder_vae, decoder_vae, vae_cfg, _ = load_models(
        vae_cfg_path, vae_ckpt_path, device
    )
    print("VAE decoder ready.")

    dl_train, dl_valid, denorm = build_dataloaders(cfg)

    encoder_cfg = cfg["encoder"]
    encoder = build_residual_mlp(
        input_dim=encoder_cfg.get("input_dim", 1024),
        hidden_dim=encoder_cfg.get("hidden_dim", 2048),
        num_layers=encoder_cfg.get("num_layers", 4),
        dropout=encoder_cfg.get("dropout", 0.1),
        use_layernorm=encoder_cfg.get("layernorm", True)
    ).to(device)
    print(f"Encoder: {encoder}")

    denoiser = SimpleMLPAdaLN(
        in_channels=1024,                     # x_t shape
        model_channels=cfg["denoiser"]["model_channels"],
        out_channels=1024,                    # predict ε
        z_channels=2048,                      # condition vector
        num_res_blocks=cfg["denoiser"].get("num_res_blocks", 4),
        grad_checkpointing=cfg["denoiser"].get("grad_ckpt", False),
    ).to(device)
    print(f"Denoiser: {denoiser}")

    opt = optim.AdamW(
        list(encoder.parameters()) + list(denoiser.parameters()),
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    # LR Scheduler
    scheduler_cfg  = cfg["train"].get("scheduler", {})
    sched_type     = str(scheduler_cfg.get("type", "none")).lower()
    base_lr        = float(cfg["train"]["lr"])
    min_lr         = float(scheduler_cfg.get("min_lr", 0.0))
    warmup_epochs  = int(scheduler_cfg.get("warmup_epochs", 0))

    step_per_epoch = len(dl_train)
    total_steps    = step_per_epoch * cfg["train"]["epochs"]
    warmup_steps   = warmup_epochs * step_per_epoch
    min_ratio      = (min_lr / base_lr) if base_lr > 0 else 0.0

    if sched_type == "linear-warmup":
        lr_lambda = Scheduler_LinearWarmup(warmup_steps)
    elif sched_type == "linear-warmup_cosine-decay":
        lr_lambda = Scheduler_LinearWarmup_CosineDecay(warmup_steps, total_steps, min_ratio)
    else:  # "none"
        lr_lambda = lambda step: 1.0

    lr_scheduler = LambdaLR(opt, lr_lambda)

    # EMA
    ema_cfg     = cfg["train"].get("ema", {})
    use_ema     = bool(ema_cfg.get("enable", False))
    ema_decay   = float(ema_cfg.get("decay", 0.9999))

    if use_ema:
        ema_encoder  = LitEma(encoder,  decay=ema_decay)
        ema_denoiser = LitEma(denoiser, decay=ema_decay)
    else:
        ema_encoder = None
        ema_denoiser = None 

    sampler_cfg = cfg["denoiser"].get("timestep_sampler", {})
    scheduler = DDPMNoiseScheduler(
        timesteps   = cfg["denoiser"].get("timesteps", 1000),
        beta_start  = float(cfg["denoiser"].get("beta_start", 1e-4)),
        beta_end    = float(cfg["denoiser"].get("beta_end",  2e-2)),
        beta_schedule = cfg["denoiser"].get("beta_schedule", "linear"),
        device=device,
        t_sampler       = sampler_cfg.get("type", "uniform"),
        t_log_mean      = float(sampler_cfg.get("log_mean", -0.5)),
        t_log_sigma     = float(sampler_cfg.get("log_sigma", 1.0)),
        t_mix_uniform_p = float(sampler_cfg.get("mix_uniform_p", 0.05)),
        t_clip_quantile = float(sampler_cfg.get("clip_quantile", 0.999)),
    ).to(device)

    if cfg.get("wandb", {}).get("enable", False):
        init_wandb(cfg)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{cfg['train']['epochs']} ===")
        encoder.train(); denoiser.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            x_fa_ca = batch["F_A+C_A"].to(device).reshape(batch["F_A+C_A"].size(0), -1)
            x_fa_cb = batch["F_A+C_B"].to(device).reshape(batch["F_A+C_B"].size(0), -1)
            x_fb_ca = batch["F_B+C_A"].to(device).reshape(batch["F_B+C_A"].size(0), -1)
            x_fb_cb = batch["F_B+C_B"].to(device).reshape(batch["F_B+C_B"].size(0), -1)

            z_fa_ca = encoder(x_fa_ca)   # (B, 2048)
            z_fb_cb = encoder(x_fb_cb)   # (B, 2048)

            content_fa = z_fa_ca[:, :1024]    # c_A
            style_fb  = z_fb_cb[:, 1024:]     # s_B

            content_fb = z_fb_cb[:, :1024]    # c_B
            style_fa   = z_fa_ca[:, 1024:]    # s_A

            cond_cA_sB = torch.cat([content_fa, style_fb], dim=1)   # used for denoise x_fb_ca
            cond_cB_sA = torch.cat([content_fb, style_fa], dim=1)   # used for denoise x_fa_cb

            B = x_fb_ca.size(0)
            t = scheduler.sample_timesteps(B)        # sample t∈[0,T)

            noise = torch.randn_like(x_fb_ca)
            x_t_fb_ca = scheduler.q_sample(x_fb_ca, t, noise)

            noise2 = torch.randn_like(x_fa_cb)
            x_t_fa_cb = scheduler.q_sample(x_fa_cb, t, noise2)  

            eps_pred_1 = denoiser(x_t_fb_ca, t, cond_cA_sB)
            eps_pred_2 = denoiser(x_t_fa_cb, t, cond_cB_sA)

            loss_1 = F.mse_loss(eps_pred_1, noise)
            loss_2 = F.mse_loss(eps_pred_2, noise2)

            loss = 0.5 * (loss_1 + loss_2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(denoiser.parameters()),
                max_norm=1.0
            )
            opt.step()
            lr_scheduler.step()
            if use_ema:
                ema_encoder(encoder)
                ema_denoiser(denoiser)
            
            if cfg.get("wandb", {}).get("enable", False) and step % cfg["wandb"].get("log_interval", 50) == 0:
                log_losses(step=step, loss_dict={"train/loss": loss.item()})
                try:
                    wandb.log({"lr": lr_scheduler.get_last_lr()[0]}, step=step)
                except Exception:
                    pass

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            step += 1
        
        if cfg["wandb"]["enable"] and cfg["vis"]["enable"]:  # Log one batch from training set
            if use_ema:
                ema_encoder.store(encoder.parameters())
                ema_denoiser.store(denoiser.parameters())
                ema_encoder.copy_to(encoder)
                ema_denoiser.copy_to(denoiser)

            train_batch = next(iter(dl_train))
            log_sample_images(step, train_batch,
                            encoder, denoiser, scheduler,
                            decoder_vae, vae_cfg, denorm,
                            device, name="train/quad_grid")
            
            if use_ema:
                ema_encoder.restore(encoder.parameters())
                ema_denoiser.restore(denoiser.parameters())

        if (epoch + 1) % cfg["eval"]["interval"] == 0:
            if use_ema:
                ema_encoder.store(encoder.parameters())
                ema_denoiser.store(denoiser.parameters())
                ema_encoder.copy_to(encoder)
                ema_denoiser.copy_to(denoiser)

            val_loss, vis_batch = evaluate(dl_valid, encoder, denoiser, scheduler, device)

            if use_ema:
                ema_encoder.restore(encoder.parameters())
                ema_denoiser.restore(denoiser.parameters())

            if cfg["wandb"]["enable"]:
                log_losses(step=step, loss_dict={"valid/loss": val_loss})

                if cfg["vis"]["enable"]:
                    if use_ema:
                        ema_encoder.store(encoder.parameters())
                        ema_denoiser.store(denoiser.parameters())
                        ema_encoder.copy_to(encoder)
                        ema_denoiser.copy_to(denoiser)

                    log_sample_images(step, vis_batch,
                                    encoder, denoiser, scheduler,
                                    decoder_vae, vae_cfg, denorm,
                                    device, name="val/quad_grid")
                    
                    if use_ema:
                        ema_encoder.restore(encoder.parameters())
                        ema_denoiser.restore(denoiser.parameters())
            

        if (epoch + 1) % cfg["train"]["save_interval"] == 0:
            save_ckpt(cfg, epoch, encoder, denoiser, opt,
                    ema_encoder=ema_encoder if use_ema else None,
                    ema_denoiser=ema_denoiser if use_ema else None,
                    lr_scheduler=lr_scheduler)



@torch.no_grad()
def evaluate(loader, encoder, denoiser, scheduler, device):
    encoder.eval(); denoiser.eval()
    losses, sample_batch = [], None

    for batch in loader:
        fa_ca = batch["F_A+C_A"].to(device).view(-1, 1024)
        fa_cb = batch["F_A+C_B"].to(device).view(-1, 1024)
        fb_ca = batch["F_B+C_A"].to(device).view(-1, 1024)
        fb_cb = batch["F_B+C_B"].to(device).view(-1, 1024)

        z_a, z_b = encoder(fa_ca), encoder(fb_cb)
        cond1 = torch.cat([z_a[:, :1024], z_b[:, 1024:]], 1)
        cond2 = torch.cat([z_b[:, :1024], z_a[:, 1024:]], 1)

        t   = scheduler.sample_timesteps(fa_ca.size(0))
        eps1, eps2 = torch.randn_like(fb_ca), torch.randn_like(fa_cb)
        x_t1 = scheduler.q_sample(fb_ca, t, eps1)
        x_t2 = scheduler.q_sample(fa_cb, t, eps2)

        eps_hat1 = denoiser(x_t1, t, cond1)
        eps_hat2 = denoiser(x_t2, t, cond2)
        losses.append(0.5 * (F.mse_loss(eps_hat1, eps1) +
                             F.mse_loss(eps_hat2, eps2)).item())

        if sample_batch is None:                      # use the 1st batch to do visualization 
            sample_batch = batch

    return sum(losses) / len(losses), sample_batch



@torch.no_grad()
def log_sample_images(step, batch, encoder, denoiser, scheduler,
                      decoder, vae_cfg, denorm, device, name, num_show=4):
    to_tensor = transforms.ToTensor()

    idx = random.sample(range(batch["F_A+C_A"].size(0)),
                        k=min(num_show, batch["F_A+C_A"].size(0)))

    fa_ca = batch["F_A+C_A"][idx].to(device).view(-1, 1024)
    fb_cb = batch["F_B+C_B"][idx].to(device).view(-1, 1024)
    fa_cb = batch["F_A+C_B"][idx].to(device).view(-1, 1024)
    fb_ca = batch["F_B+C_A"][idx].to(device).view(-1, 1024)

    z_a = encoder(fa_ca)
    z_b = encoder(fb_cb)
    cond1 = torch.cat([z_a[:, :1024], z_b[:, 1024:]], dim=1)  # for fb_ca
    cond2 = torch.cat([z_b[:, :1024], z_a[:, 1024:]], dim=1)  # for fa_cb

    lat_fb_ca = scheduler.p_sample_loop(denoiser, (len(idx), 1024), cond1, device)
    lat_fa_cb = scheduler.p_sample_loop(denoiser, (len(idx), 1024), cond2, device)

    def reshape_and_decode(flat_latents):
        latents = flat_latents.view(-1, 4, 16, 16)
        decoded_imgs = []
        for lat in latents:
            lat = denorm(lat)
            img = vae_decode(lat, decoder, vae_cfg, device)  # returns PIL
            decoded_imgs.append(to_tensor(img))
        return decoded_imgs  # list of [1, H, W]

    img_gt_fa_cb = reshape_and_decode(fa_cb)
    img_gt_fb_ca = reshape_and_decode(fb_ca)
    img_fa_ca = reshape_and_decode(fa_ca)
    img_fb_cb = reshape_and_decode(fb_cb)
    img_gen_fa_cb = reshape_and_decode(lat_fa_cb) # generated 
    img_gen_fb_ca = reshape_and_decode(lat_fb_ca) # generated 

    # Concatenate image quads: [GT_fa_cb | GT_fb_ca | Ref_fa_ca | Ref_fb_cb | Gen_fa_cb | Gen_fb_ca]
    imgs = []
    for i in range(len(idx)):
        row = torch.cat([
            img_gt_fa_cb[i],
            img_gt_fb_ca[i],
            img_fa_ca[i],
            img_fb_cb[i],
            img_gen_fa_cb[i],
            img_gen_fb_ca[i]
        ], dim=2)  # concat width-wise (C, H, 6*W)
        imgs.append(row)
    grid = make_grid(torch.stack(imgs), nrow=1, padding=2)

    wandb.log({name: wandb.Image(grid)}, step=step)

def save_ckpt(cfg: dict, epoch: int,
              encoder: nn.Module,
              denoiser: nn.Module,
              optimizer: optim.Optimizer,
              ema_encoder=None,
              ema_denoiser=None,
              lr_scheduler=None) -> None:
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
    ckpt = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "denoiser": denoiser.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if ema_encoder is not None:
        ckpt["ema_encoder"] = ema_encoder.state_dict()
    if ema_denoiser is not None:
        ckpt["ema_denoiser"] = ema_denoiser.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(ckpt, ckpt_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ddpm_disentangle.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)