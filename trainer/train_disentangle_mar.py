# trainer/train_disentangle.py

import os
import json
import math
import random
import wandb
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

from dataset.font_dataset import FourWayFontPairLatentPTDataset, split_fonts, filter_dataset_fonts

from models.mlp import build_residual_mlp, SimpleMLPAdaLN
from models.DDPM import GaussianDiffusion as DDPMNoiseScheduler
from models.flow_matching import FlowMatching, LinearRFPath, FM_CFG_Wrapper

from utils.logger import init_wandb, log_losses
from utils.vae_io import load_models, decode as vae_decode
from utils.lr_scheduler import (
    Scheduler_LinearWarmup,
    Scheduler_LinearWarmup_CosineDecay,
)
from utils.ema import LitEma
from utils.evaluate_basic import compute_psnr, compute_ssim, compute_metrics




class CFGDenoiser:
    """
    Wraps a denoiser(x, t, c) to use classifier-free guidance at sampling time.
    Assumes denoiser implements forward_with_cfg(x, t, c, cfg_scale) and supports both eps-only and VLB.
    """
    def __init__(self, denoiser, cfg_scale: float):
        self.denoiser = denoiser
        self.cfg_scale = cfg_scale

    @torch.no_grad()
    def __call__(self, x, t, c):
        # Build unconditional condition (zeros). You can swap to a learned null vector if you prefer.
        c_uncond = torch.zeros_like(c)

        # Duplicate batch: [cond; uncond] with the same x,t shapes duplicated
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        c_in = torch.cat([c, c_uncond], dim=0)

        out = self.denoiser.forward_with_cfg(x_in, t_in, c_in, self.cfg_scale)
        # forward_with_cfg returns combined output; keep the first half as the guided eps
        B = x.shape[0]
        out = out[:B]  # if VLB, this still returns [B, 2*C]; if eps-only, [B, C]
        # If your scheduler expects just eps (and not concatenated [eps, rest]), slice accordingly:
        if getattr(self.denoiser, "vlb_mode", False):
            out = out[:, :self.denoiser.in_channels]  # take eps part
        return out



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


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, callable]:  # FIX: typing 返回了 denorm
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
        pair_num=100000,
        stats_yaml=cfg["dataset"].get("stats_yaml", None),
    )

    ds_valid = FourWayFontPairLatentPTDataset(
        pt_path=cfg["dataset"].get("pt_path"),
        chars_path=cfg["dataset"].get("chars_path"),
        fonts_json=cfg["dataset"].get("font_json"),
        latent_shape=latent_shape,
        pair_num=1000,
        stats_yaml=cfg["dataset"].get("stats_yaml", None),
    )

    filter_dataset_fonts(ds_train, train_fonts)
    filter_dataset_fonts(ds_valid, valid_fonts)

    print(f"Train dataset: {len(ds_train)} samples, Valid dataset: {len(ds_valid)} samples")

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    return dl_train, dl_valid, ds_train.denorm


def train_disentangle_loop_mar(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    seed = cfg.get("seed", 1234)
    torch.manual_seed(seed)
    random.seed(seed)

    vae_cfg_path = cfg["vis"]["vae_config"]
    vae_ckpt_path = cfg["vis"]["vae_ckpt"]
    encoder_vae, decoder_vae, vae_cfg, _ = load_models(vae_cfg_path, vae_ckpt_path, device)
    print("VAE decoder ready.")

    enc_cfg = cfg.get("encoder", {})
    use_encoder = bool(enc_cfg.get("enable", True))  # skip MLP encoder or not

    cfg_train = cfg.get("train", {}).get("cfg", {})
    cfg_enable = bool(cfg_train.get("enable", False))
    p_uncond = float(cfg_train.get("p_uncond", 0.1))
    cfg_scale = float(cfg_train.get("scale", 3.0))  # used for sampling/vis

    if use_encoder:
        encoder_cfg = cfg["encoder"]
        encoder = build_residual_mlp(
            input_dim=encoder_cfg.get("input_dim", 1024),
            hidden_dim=encoder_cfg.get("hidden_dim", 2048),
            num_layers=encoder_cfg.get("num_layers", 4),
            dropout=encoder_cfg.get("dropout", 0.1),
            use_layernorm=encoder_cfg.get("layernorm", True),
        ).to(device)
        print(f"Encoder: {encoder}")
    else:
        encoder = None
        print("Encoder: DISABLED (concat raw latents as condition).")

    dl_train, dl_valid, denorm = build_dataloaders(cfg)

    denoiser = SimpleMLPAdaLN(
        in_channels=1024,                     # x_t shape
        model_channels=cfg["denoiser"]["model_channels"],
        out_channels=1024,                    # predict ε
        z_channels=2048,                      # condition vector
        num_res_blocks=cfg["denoiser"].get("num_res_blocks", 4),
        grad_checkpointing=cfg["denoiser"].get("grad_ckpt", False),
    ).to(device)
    print(f"Denoiser: {denoiser}")

    if use_encoder:
        opt = optim.AdamW(
            list(encoder.parameters()) + list(denoiser.parameters()),
            lr=cfg["train"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=cfg["train"].get("weight_decay", 0.0),
        )
    else:
        opt = optim.AdamW(
            denoiser.parameters(),
            lr=cfg["train"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=cfg["train"].get("weight_decay", 0.0),
        )

    # LR Scheduler
    scheduler_cfg = cfg["train"].get("scheduler", {})
    sched_type = str(scheduler_cfg.get("type", "none")).lower()
    base_lr = float(cfg["train"]["lr"])
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))

    step_per_epoch = len(dl_train)
    total_steps = step_per_epoch * cfg["train"]["epochs"]
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
    if use_ema:
        print(f"Using EMA with decay {ema_cfg.get('decay', 0.9999)}")
    ema_decay = float(ema_cfg.get("decay", 0.9999))

    ema_encoder = LitEma(encoder, decay=ema_decay) if (use_ema and use_encoder) else None
    ema_denoiser = LitEma(denoiser, decay=ema_decay) if use_ema else None

    algo_type = cfg.get("algo", {}).get("type", "ddpm").lower()

    sampler_cfg = cfg["denoiser"].get("timestep_sampler", {})
    print(sampler_cfg)

    if algo_type == "ddpm":
        # 原 DDPM 构造保持不变
        scheduler = DDPMNoiseScheduler(
            timesteps=cfg["denoiser"].get("timesteps", 1000),
            beta_start=float(cfg["denoiser"].get("beta_start", 1e-4)),
            beta_end=float(cfg["denoiser"].get("beta_end", 2e-2)),
            beta_schedule=cfg["denoiser"].get("beta_schedule", "linear"),
            device=device,
            t_sampler=sampler_cfg.get("type", "uniform"),
            t_log_mean=float(sampler_cfg.get("log_mean", -0.5)),
            t_log_sigma=float(sampler_cfg.get("log_sigma", 1.0)),
            t_mix_uniform_p=float(sampler_cfg.get("mix_uniform_p", 0.05)),
            t_clip_quantile=float(sampler_cfg.get("clip_quantile", 0.999)),
        ).to(device)
    else:
        fm_cfg = cfg.get("flow_matching", {})
        path_cfg = fm_cfg.get("path", {})
        path = LinearRFPath(
            t_sampler=path_cfg.get("t_sampler", "uniform"),
            ln_mu=float(path_cfg.get("ln_mu", -0.5)),
            ln_sigma=float(path_cfg.get("ln_sigma", 1.0)),
            mix_unif_p=float(path_cfg.get("mix_unif_p", 0.05)),
            clip_q=float(path_cfg.get("clip_q", 0.999)),
        )
        scheduler = FlowMatching(
            path=path,
            t_epsilon=float(fm_cfg.get("t_epsilon", 1e-5)),
            ode_solver=str(fm_cfg.get("ode_solver", "heun")),
            ode_steps=int(fm_cfg.get("ode_steps", cfg["sample"].get("steps", 50))),
        ).to(device)

    if cfg.get("wandb", {}).get("enable", False):
        init_wandb(cfg)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        print(f"\n=== Epoch {epoch + 1}/{cfg['train']['epochs']} ===")
        if use_encoder and encoder is not None:
            encoder.train()
        denoiser.train()
        
        # Change sampling algorithm 
        if step < sampler_cfg.get("warmup", 10000):
            if algo_type == "ddpm":
                scheduler.t_sampler = "uniform"
            else:
                scheduler.path.t_sampler = "uniform"
            print("Uniform Sampling t ... ")
        else:
            if algo_type == "ddpm":
                scheduler.t_sampler = sampler_cfg.get("type", "uniform")
            else:
                scheduler.path.t_sampler = sampler_cfg.get("type", "uniform")
            print(f"{sampler_cfg.get('type','uniform').capitalize()} Sampling t ... ")

        pbar = tqdm(dl_train, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            x_fa_ca = batch["F_A+C_A"].to(device).reshape(batch["F_A+C_A"].size(0), -1)
            x_fa_cb = batch["F_A+C_B"].to(device).reshape(batch["F_A+C_B"].size(0), -1)
            x_fb_ca = batch["F_B+C_A"].to(device).reshape(batch["F_B+C_A"].size(0), -1)
            x_fb_cb = batch["F_B+C_B"].to(device).reshape(batch["F_B+C_B"].size(0), -1)

            if use_encoder and encoder is not None:
                z_fa_ca = encoder(x_fa_ca)   # (B, 2048)
                z_fb_cb = encoder(x_fb_cb)   # (B, 2048)

                content_fa = z_fa_ca[:, :1024]  # c_A
                style_fb = z_fb_cb[:, 1024:]    # s_B

                content_fb = z_fb_cb[:, :1024]  # c_B
                style_fa = z_fa_ca[:, 1024:]    # s_A

                cond_cA_sB = torch.cat([content_fa, style_fb], dim=1)  # (B, 2048)
                cond_cB_sA = torch.cat([content_fb, style_fa], dim=1)  # (B, 2048)
            else:
                cond_cA_sB = torch.cat([x_fa_ca, x_fb_cb], dim=1)  # denoise x_fb_ca
                cond_cB_sA = torch.cat([x_fb_cb, x_fa_ca], dim=1)  # denoise x_fa_cb

            B = x_fb_ca.size(0)
            t = scheduler.sample_timesteps(B, device=x_fb_ca.device)  # sample t∈[0,T)

            noise = torch.randn_like(x_fb_ca)
            x_t_fb_ca = scheduler.q_sample(x_fb_ca, t, noise)

            noise2 = torch.randn_like(x_fa_cb)
            x_t_fa_cb = scheduler.q_sample(x_fa_cb, t, noise2)

            # CFG
            if cfg_enable and p_uncond > 0.0:
                # Bernoulli mask per sample
                drop_mask = (torch.rand(B, device=device) < p_uncond).float().unsqueeze(1)  # (B,1)
                # Zero-out condition where masked (simple and effective)
                cond_cA_sB = cond_cA_sB * (1.0 - drop_mask)
                cond_cB_sA = cond_cB_sA * (1.0 - drop_mask)

            if algo_type == "ddpm":
                eps_pred_1 = denoiser(x_t_fb_ca, t, cond_cA_sB)
                eps_pred_2 = denoiser(x_t_fa_cb, t, cond_cB_sA)
                loss_1 = F.mse_loss(eps_pred_1, noise)
                loss_2 = F.mse_loss(eps_pred_2, noise2)
                loss = 0.5 * (loss_1 + loss_2)
            else:
                v_target_1 = x_fb_ca - noise
                v_target_2 = x_fa_cb - noise2 
                v_pred_1 = denoiser(x_t_fb_ca, t, cond_cA_sB)
                v_pred_2 = denoiser(x_t_fa_cb, t, cond_cB_sA)
                loss_1 = F.mse_loss(v_pred_1, v_target_1)
                loss_2 = F.mse_loss(v_pred_2, v_target_2)
                loss = 0.5 * (loss_1 + loss_2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ((list(encoder.parameters()) if (use_encoder and encoder is not None) else []) + list(denoiser.parameters())),
                max_norm=1.0,
            )
            opt.step()
            lr_scheduler.step()

            # GUARD: EMA 调用需判空
            if use_ema:
                if ema_encoder is not None and encoder is not None:
                    ema_encoder(encoder)
                if ema_denoiser is not None:
                    ema_denoiser(denoiser)

            if cfg.get("wandb", {}).get("enable", False) and step % cfg["wandb"].get("log_interval", 50) == 0:
                log_losses(step=step, loss_dict={"train/loss": loss.item()})
                try:
                    wandb.log({"lr": lr_scheduler.get_last_lr()[0]}, step=step)
                except Exception:
                    pass

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            step += 1

        # 可视化（训练集）
        if cfg["wandb"]["enable"] and cfg["vis"]["enable"]:
            if use_ema:
                if ema_encoder is not None and encoder is not None:
                    ema_encoder.store(encoder.parameters())
                    ema_encoder.copy_to(encoder)
                if ema_denoiser is not None:
                    ema_denoiser.store(denoiser.parameters())
                    ema_denoiser.copy_to(denoiser)

            train_batch = next(iter(dl_train))
            log_sample_images(
                step, train_batch,  # FIX: 传 use_encoder
                encoder, denoiser, scheduler,
                decoder_vae, vae_cfg, denorm,
                device, name="train/quad_grid", use_encoder=use_encoder, cfg=cfg, algo_type=algo_type
            )

            if use_ema:
                if ema_encoder is not None and encoder is not None:
                    ema_encoder.restore(encoder.parameters())
                if ema_denoiser is not None:
                    ema_denoiser.restore(denoiser.parameters())

        # 验证
        if (epoch + 1) % cfg["eval"]["interval"] == 0:
            if use_ema:
                if ema_encoder is not None and encoder is not None:
                    ema_encoder.store(encoder.parameters())
                    ema_encoder.copy_to(encoder)
                if ema_denoiser is not None:
                    ema_denoiser.store(denoiser.parameters())
                    ema_denoiser.copy_to(denoiser)

            val_loss, vis_batch = evaluate(dl_valid, encoder, denoiser, scheduler, device, use_encoder, algo_type)

            if use_ema:
                if ema_encoder is not None and encoder is not None:
                    ema_encoder.restore(encoder.parameters())
                if ema_denoiser is not None:
                    ema_denoiser.restore(denoiser.parameters())

            if cfg["wandb"]["enable"]:
                log_losses(step=step, loss_dict={"valid/loss": val_loss})

                if cfg["vis"]["enable"]:
                    if use_ema:
                        if ema_encoder is not None and encoder is not None:
                            ema_encoder.store(encoder.parameters())
                            ema_encoder.copy_to(encoder)
                        if ema_denoiser is not None:
                            ema_denoiser.store(denoiser.parameters())
                            ema_denoiser.copy_to(denoiser)

                    log_sample_images(
                        step, vis_batch,  # FIX: 传 use_encoder
                        encoder, denoiser, scheduler,
                        decoder_vae, vae_cfg, denorm,
                        device, name="val/quad_grid", use_encoder=use_encoder, cfg=cfg, algo_type=algo_type
                    )

                    if use_ema:
                        if ema_encoder is not None and encoder is not None:
                            ema_encoder.restore(encoder.parameters())
                        if ema_denoiser is not None:
                            ema_denoiser.restore(denoiser.parameters())

        # 存 ckpt
        if (epoch + 1) % cfg["train"]["save_interval"] == 0:
            save_ckpt(
                cfg, epoch, encoder, denoiser, opt,
                ema_encoder=ema_encoder if use_ema else None,
                ema_denoiser=ema_denoiser if use_ema else None,
                lr_scheduler=lr_scheduler,
            )


@torch.no_grad()
def evaluate(loader, encoder, denoiser, scheduler, device, use_encoder: bool, algo_type):
    if use_encoder and encoder is not None:
        encoder.eval()
    denoiser.eval()
    losses, sample_batch = [], None

    pbar = tqdm(loader, desc="Eval")
    for batch in pbar:
        fa_ca = batch["F_A+C_A"].to(device).view(-1, 1024)
        fa_cb = batch["F_A+C_B"].to(device).view(-1, 1024)
        fb_ca = batch["F_B+C_A"].to(device).view(-1, 1024)
        fb_cb = batch["F_B+C_B"].to(device).view(-1, 1024)

        if use_encoder and encoder is not None:
            z_a, z_b = encoder(fa_ca), encoder(fb_cb)
            cond1 = torch.cat([z_a[:, :1024], z_b[:, 1024:]], 1)  # for fb_ca
            cond2 = torch.cat([z_b[:, :1024], z_a[:, 1024:]], 1)  # for fa_cb
        else:
            cond1 = torch.cat([fa_ca, fb_cb], 1)  # for fb_ca
            cond2 = torch.cat([fb_cb, fa_ca], 1)  # for fa_cb

        B = fa_ca.size(0)

        # (optional) mirror train-time dropout
        p_uncond_eval = 0.1
        if p_uncond_eval > 0.0:
            drop_mask = (torch.rand(B, device=device) < p_uncond_eval).float().unsqueeze(1)
            cond1 = cond1 * (1.0 - drop_mask)
            cond2 = cond2 * (1.0 - drop_mask)

        t = scheduler.sample_timesteps(B, device=fb_ca.device)
        eps1, eps2 = torch.randn_like(fb_ca), torch.randn_like(fa_cb)
        x_t1 = scheduler.q_sample(fb_ca, t, eps1)
        x_t2 = scheduler.q_sample(fa_cb, t, eps2)

        if algo_type == "ddpm":
            eps_hat1 = denoiser(x_t1, t, cond1)
            eps_hat2 = denoiser(x_t2, t, cond2)
            losses.append(0.5 * (F.mse_loss(eps_hat1, eps1) + F.mse_loss(eps_hat2, eps2)).item())
        else:
            v_tgt1 = fb_ca - eps1 
            v_tgt2 = fa_cb - eps2 
            v_hat1 = denoiser(x_t1, t, cond1)
            v_hat2 = denoiser(x_t2, t, cond2)
            losses.append(0.5 * (F.mse_loss(v_hat1, v_tgt1) + F.mse_loss(v_hat2, v_tgt2)).item())

    if sample_batch is None:
        sample_batch = batch

    return sum(losses) / len(losses), sample_batch


@torch.no_grad()
def log_sample_images(
    step, batch, encoder, denoiser, scheduler,
    decoder, vae_cfg, denorm, device, name, num_show=4, use_encoder: bool = True, cfg: dict = None, algo_type: str = "ddpm"  
):
    to_tensor = transforms.ToTensor()
    idx = random.sample(range(batch["F_A+C_A"].size(0)), k=min(num_show, batch["F_A+C_A"].size(0)))

    fa_ca = batch["F_A+C_A"][idx].to(device).view(-1, 1024)
    fb_cb = batch["F_B+C_B"][idx].to(device).view(-1, 1024)
    fa_cb = batch["F_A+C_B"][idx].to(device).view(-1, 1024)
    fb_ca = batch["F_B+C_A"][idx].to(device).view(-1, 1024)

    if use_encoder and encoder is not None:
        z_a = encoder(fa_ca)
        z_b = encoder(fb_cb)
        cond1 = torch.cat([z_a[:, :1024], z_b[:, 1024:]], dim=1)  # for fb_ca
        cond2 = torch.cat([z_b[:, :1024], z_a[:, 1024:]], dim=1)  # for fa_cb
    else:
        cond1 = torch.cat([fa_ca, fb_cb], dim=1)  # for fb_ca
        cond2 = torch.cat([fb_cb, fa_ca], dim=1)  # for fa_cb
    
    # CFG
    use_cfg = False
    cfg_scale_local = 3.0
    if cfg is not None:
        use_cfg = bool(cfg.get("train", {}).get("cfg", {}).get("enable", False))
        cfg_scale_local = float(cfg.get("train", {}).get("cfg", {}).get("scale", 3.0))
    
    if use_cfg and algo_type == "flow_matching":
        guided_denoiser = FM_CFG_Wrapper(denoiser, cfg_scale_local)
    elif use_cfg:  # ddpm
        guided_denoiser = CFGDenoiser(denoiser, cfg_scale_local)
    else:
        guided_denoiser = denoiser

    lat_fb_ca = scheduler.p_sample_loop(guided_denoiser, (len(idx), 1024), cond1, device)
    lat_fa_cb = scheduler.p_sample_loop(guided_denoiser, (len(idx), 1024), cond2, device)

    def reshape_and_decode(flat_latents):
        latents = flat_latents.view(-1, 4, 16, 16)
        decoded_imgs = []
        for lat in latents:
            lat = denorm(lat)
            img = vae_decode(lat, decoder, vae_cfg, device)  # returns PIL
            t = to_tensor(img) # original
            # t = (t >= 0.5).float()  # binary
            decoded_imgs.append(t)
        return decoded_imgs

    img_gt_fa_cb = reshape_and_decode(fa_cb)
    img_gt_fb_ca = reshape_and_decode(fb_ca)
    img_fa_ca = reshape_and_decode(fa_ca)
    img_fb_cb = reshape_and_decode(fb_cb)
    img_gen_fa_cb = reshape_and_decode(lat_fa_cb)
    img_gen_fb_ca = reshape_and_decode(lat_fb_ca)

    imgs = []
    for i in range(len(idx)):
        row = torch.cat([
            img_gt_fa_cb[i],
            img_gt_fb_ca[i],
            img_fa_ca[i],
            img_fb_cb[i],
            img_gen_fa_cb[i],
            img_gen_fb_ca[i]
        ], dim=2)
        imgs.append(row)
    grid = make_grid(torch.stack(imgs), nrow=1, padding=2)
    wandb.log({name: wandb.Image(grid)}, step=step)

    img_gt_fa_cb = torch.stack(img_gt_fa_cb)
    img_gt_fb_ca = torch.stack(img_gt_fb_ca)
    img_gen_fa_cb = torch.stack(img_gen_fa_cb)
    img_gen_fb_ca = torch.stack(img_gen_fb_ca)

    if cfg.get("wandb", {}).get("enable", False):
        metrics_fa_cb = compute_metrics(img_gen_fa_cb, img_gt_fa_cb)
        metrics_fb_ca = compute_metrics(img_gen_fb_ca, img_gt_fb_ca)
        averaged_metrics = {f"{name}/metric_{k}": (metrics_fa_cb[k] + metrics_fb_ca[k]) / 2 for k in metrics_fa_cb}
        wandb.log(averaged_metrics, step=step)


def save_ckpt(
    cfg: dict, epoch: int,
    encoder: nn.Module,
    denoiser: nn.Module,
    optimizer: optim.Optimizer,
    ema_encoder=None,
    ema_denoiser=None,
    lr_scheduler=None
) -> None:
    ckpt_dir = Path(cfg["train"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
    ckpt = {
        "epoch": epoch,
        "denoiser": denoiser.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # GUARD: 仅在存在 encoder 时保存
    if encoder is not None:
        ckpt["encoder"] = encoder.state_dict()
    if ema_encoder is not None:
        ckpt["ema_encoder"] = ema_encoder.state_dict()
    if ema_denoiser is not None:
        ckpt["ema_denoiser"] = ema_denoiser.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(ckpt, ckpt_path)

