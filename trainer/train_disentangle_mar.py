# trainer/train_disentangle_mar.py

import os
import json
import math
import random
import wandb
import yaml
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid
from torchvision import transforms

from dataset.font_dataset import (
    FourWayFontPairLatentPTDataset,
    split_fonts,
    split_chars,
    filter_dataset_fonts,
    sample_holdout_quads,
)
 
from models.mlp import build_residual_mlp, SimpleMLPAdaLN
from models.encoder import CNNContentStyleEncoder
from models.DDPM import GaussianDiffusion as DDPMNoiseScheduler
from models.flow_matching import FlowMatching, LinearRFPath, FM_CFG_Wrapper
from models.refiner import SimpleRefinerUNet

from utils.logger import init_wandb, log_losses
from utils.vae_io import load_models, decode as vae_decode
from utils.lr_scheduler import (
    Scheduler_LinearWarmup,
    Scheduler_LinearWarmup_CosineDecay,
)
from utils.ema import LitEma
from utils.evaluate_basic import compute_psnr, compute_ssim, compute_metrics
from utils.siamese_scores import (
    load_model as load_siamese_model,
    score_pair as siamese_score_pair,
    DEFAULT_CONTENT_CKPT,
    DEFAULT_STYLE_CKPT,
)
from utils.info_NCE import info_nce_pairwise


def seed_worker(worker_id):
    """Deterministic worker seeding to align DataLoader randomness across runs."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CFGDenoiser:
    def __init__(self, denoiser, cfg_scale: float):
        self.denoiser = denoiser
        self.cfg_scale = cfg_scale

    @torch.no_grad()
    def __call__(self, x, t, c):
        c_uncond = torch.zeros_like(c)

        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        c_in = torch.cat([c, c_uncond], dim=0)

        out = self.denoiser.forward_with_cfg(x_in, t_in, c_in, self.cfg_scale)
        B = x.shape[0]
        out = out[:B]  
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


def build_dataloaders(cfg: dict):
    """Build dataloaders according to configured split strategy."""
    font_json = cfg["dataset"]["font_json"]
    chars_path = cfg["dataset"]["chars_path"]
    split_mode = cfg["dataset"].get("train_test_split", "quad-split").lower()
    seed = cfg.get("seed", 1234)
    g = torch.Generator().manual_seed(seed)

    split_mode = cfg["dataset"].get("train_test_split", "quad-split").lower()

    # split fonts and chars 
    if split_mode == "soft-train-test-split":
        # use all fonts/chars for training; validation combos handled via holdout
        with open(font_json, "r", encoding="utf-8") as f:
            all_fonts = json.load(f)
        with open(chars_path, "r", encoding="utf-8") as f:
            all_chars = [line.strip() for line in f if line.strip()]
        train_fonts = valid_fonts = set(all_fonts)
        train_chars = valid_chars = set(all_chars)
    else:
        train_fonts, valid_fonts = split_fonts(
            font_json,
            train_ratio=cfg["dataset"].get("train_ratio_font", 0.9),
            seed=cfg.get("seed", 1234),
        )
        train_chars, valid_chars = split_chars(
            chars_path,
            train_ratio=cfg["dataset"].get("train_ratio_char", 0.9),
            seed=cfg.get("seed", 1234),
        )

    print(f"Fonts split: train={len(train_fonts)}, valid={len(valid_fonts)}")
    print(f"Chars split: train={len(train_chars)}, valid={len(valid_chars)}")

    latent_size = cfg["dataset"].get("latent_size", 16)
    latent_channels = cfg["dataset"].get("latent_channels", 4)
    latent_shape = (latent_channels, latent_size, latent_size)

    def make_ds(pair_num):
        return FourWayFontPairLatentPTDataset(
            pt_path=cfg["dataset"]["pt_path"],
            chars_path=cfg["dataset"]["chars_path"],
            fonts_json=cfg["dataset"]["font_json"],
            latent_shape=latent_shape,
            pair_num=pair_num,
            stats_yaml=cfg["dataset"].get("stats_yaml", None),
        )

    ds_scsf = make_ds(500000)
    ds_scuf = None
    ds_ucsf = None
    ds_ucuf = None
    ds_holdout = None

    # 1. SCSF: seen char + seen font (train)
    filter_dataset_fonts(ds_scsf, train_fonts)
    ds_scsf.apply_char_filter(train_chars)

    if split_mode != "soft-train-test-split":
        ds_scuf = make_ds(2000)
        ds_ucsf = make_ds(2000)
        ds_ucuf = make_ds(2000)

        # 2. SCUF: seen char + unseen font
        filter_dataset_fonts(ds_scuf, valid_fonts)
        ds_scuf.apply_char_filter(train_chars)

        # 3. UCSF: unseen char + seen font
        filter_dataset_fonts(ds_ucsf, train_fonts)
        ds_ucsf.apply_char_filter(valid_chars)

        # 4. UCUF: unseen char + unseen font
        filter_dataset_fonts(ds_ucuf, valid_fonts)
        ds_ucuf.apply_char_filter(valid_chars)
    else:
        soft_ratio = cfg["dataset"].get("soft_holdout_ratio", 0.05)
        soft_count = cfg["dataset"].get("soft_holdout_count")
        soft_seed = cfg["dataset"].get("soft_holdout_seed", cfg.get("seed", 1234))
        soft_pair_num = cfg["dataset"].get("soft_holdout_pair_num", 2000)

        holdout_set = sample_holdout_quads(
            ds_scsf,
            count=soft_count,
            ratio=soft_ratio,
            seed=soft_seed,
        )
        print(f"[split] soft-train-test-split holdout combos: {len(holdout_set)}")

        if soft_pair_num and soft_pair_num > 0 and len(holdout_set) > 0:
            ds_holdout = make_ds(soft_pair_num)
            filter_dataset_fonts(ds_holdout, train_fonts)
            ds_holdout.apply_char_filter(train_chars)
            ds_holdout.set_holdout_quads(holdout_set, mode="exclude")

    if split_mode == "soft-train-test-split":
        print(
            f"Dataset summary: SCSF={len(ds_scsf)}"
            + (f", HOLDOUT={len(ds_holdout)}" if ds_holdout is not None else "")
        )
    else:
        print(
            f"Dataset summary: "
            f"SCSF={len(ds_scsf)}, SCUF={len(ds_scuf)}, UCSF={len(ds_ucsf)}, UCUF={len(ds_ucuf)}"
            + (f", HOLDOUT={len(ds_holdout)}" if ds_holdout is not None else "")
        )

    num_workers = cfg.get("num_workers", 4)
    dl_scsf = DataLoader(
        ds_scsf,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loaders = {}
    if ds_scuf is not None:
        dl_scuf = DataLoader(
            ds_scuf,
            batch_size=cfg["eval"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        dl_ucsf = DataLoader(
            ds_ucsf,
            batch_size=cfg["eval"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        dl_ucuf = DataLoader(
            ds_ucuf,
            batch_size=cfg["eval"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        val_loaders.update(
            {
                "SCUF": dl_scuf,
                "UCSF": dl_ucsf,
                "UCUF": dl_ucuf,
            }
        )

    if ds_holdout is not None:
        dl_holdout = DataLoader(
            ds_holdout,
            batch_size=cfg["eval"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loaders["SCSF_Holdout"] = dl_holdout

    return dl_scsf, val_loaders, ds_scsf.denorm



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

    dl_scsf, val_loaders, denorm = build_dataloaders(cfg)


    denoiser = SimpleMLPAdaLN(
        in_channels=1024,                     # x_t shape
        model_channels=cfg["denoiser"]["model_channels"],
        out_channels=1024,                    # predict ε
        z_channels=2048,                      # condition vector
        num_res_blocks=cfg["denoiser"].get("num_res_blocks", 4),
        grad_checkpointing=cfg["denoiser"].get("grad_ckpt", False),
    ).to(device)
    print(f"Denoiser: {denoiser}")

    # Adversarial disentanglement classifiers 
    adv_cfg = cfg.get("adv", {})
    use_adv = bool(adv_cfg.get("enable", False))
    lambda_adv = float(adv_cfg.get("lambda", 0.1))
    if use_adv:
        print(f"[ADV] Enabling adversarial disentanglement loss (λ={lambda_adv})")

        # calculate num of fonts and chars
        with open(cfg["dataset"]["font_json"], "r", encoding="utf-8") as f:
            num_fonts = len(json.load(f))
        with open(cfg["dataset"]["chars_path"], "r", encoding="utf-8") as f:
            num_chars = sum(1 for _ in f if _.strip())

        adv_hidden = int(adv_cfg.get("hidden_dim", 512))
        Cls_s = nn.Sequential(
            nn.Linear(1024, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, num_fonts)
        ).to(device)

        Cls_c = nn.Sequential(
            nn.Linear(1024, adv_hidden),
            nn.ReLU(),
            nn.Linear(adv_hidden, num_chars)
        ).to(device)

        ce_loss = nn.CrossEntropyLoss()
        def entropy_loss(logits):
            probs = torch.softmax(logits, dim=1)
            return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
    else:
        Cls_s = Cls_c = ce_loss = entropy_loss = None

    opt_params = list(denoiser.parameters())
    if use_encoder and encoder is not None:
        opt_params += list(encoder.parameters())
    if use_adv:
        opt_params += list(Cls_s.parameters()) + list(Cls_c.parameters())

    opt = optim.AdamW(
        opt_params,
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

    step_per_epoch = len(dl_scsf)
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

    siamese_cfg = cfg.get("siamese", {})
    siamese_models = None
    if siamese_cfg.get("enable", False):
        siamese_device = torch.device(siamese_cfg.get("device", str(device)))
        content_ckpt = siamese_cfg.get("content_ckpt") or DEFAULT_CONTENT_CKPT
        style_ckpt = siamese_cfg.get("style_ckpt") or DEFAULT_STYLE_CKPT
        content_encoder_type = siamese_cfg.get("content_encoder_type", "enhanced")
        style_encoder_type = siamese_cfg.get("style_encoder_type", "vgg")

        content_model = load_siamese_model(content_ckpt, "content", siamese_device, content_encoder_type)
        style_model = load_siamese_model(style_ckpt, "style", siamese_device, style_encoder_type)

        siamese_models = {
            "content": content_model,
            "style": style_model,
            "device": siamese_device,
            "log_table": bool(siamese_cfg.get("log_table", False)),
        }
        print("Loaded Siamese evaluators for content/style scoring.")

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

        pbar = tqdm(dl_scsf, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
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
            # t = scheduler.sample_timesteps(B)  # sample t∈[0,T) -- DDPM
            t = scheduler.sample_timesteps(B, device=x_fb_ca.device)  # sample t∈[0,T). -- Flow Matching

            noise = torch.randn_like(x_fb_ca)
            x_t_fb_ca = scheduler.q_sample(x_fb_ca, t, noise)

            noise2 = torch.randn_like(x_fa_cb)
            x_t_fa_cb = scheduler.q_sample(x_fa_cb, t, noise2)

            # CFG
            if cfg_enable and p_uncond > 0.0:
                drop_mask = (torch.rand(B, device=device) < p_uncond).float().unsqueeze(1)  # (B,1)
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
            
                        # === Adversarial disentanglement loss ===
            if use_adv and use_encoder and encoder is not None:
                
                z_fa_ca = encoder(x_fa_ca)   # (B, 2048)
                z_fa_cb = encoder(x_fa_cb)   # (B, 2048)
                z_fb_ca = encoder(x_fb_ca)   # (B, 2048)
                z_fb_cb = encoder(x_fb_cb)   # (B, 2048)

                # content/style 
                c_fa_ca, s_fa_ca = z_fa_ca[:, :1024], z_fa_ca[:, 1024:]
                c_fa_cb, s_fa_cb = z_fa_cb[:, :1024], z_fa_cb[:, 1024:]
                c_fb_ca, s_fb_ca = z_fb_ca[:, :1024], z_fb_ca[:, 1024:]
                c_fb_cb, s_fb_cb = z_fb_cb[:, :1024], z_fb_cb[:, 1024:]

                # adversarial 
                y_font_a = batch["font_id_a"].to(device)
                y_font_b = batch["font_id_b"].to(device)
                y_char_a = batch["char_id_a"].to(device)
                y_char_b = batch["char_id_b"].to(device)

                logit_s_on_s = Cls_s(s_fb_cb)     # s_B identifies font_B
                logit_c_on_c = Cls_c(c_fa_ca)     # c_A identifies char_A
                logit_c_on_s = Cls_c(s_fb_cb)     # s_B can't identifies char_B
                logit_s_on_c = Cls_s(c_fa_ca)     # c_A can't identifies font_B

                adv_loss_s = ce_loss(logit_s_on_s, y_font_b) - entropy_loss(logit_c_on_s)
                adv_loss_c = ce_loss(logit_c_on_c, y_char_a) - entropy_loss(logit_s_on_c)
                adv_loss = 0.5 * (adv_loss_s + adv_loss_c)
            else:
                
                if use_encoder and encoder is not None:
                    z_fa_ca = encoder(x_fa_ca)
                    z_fa_cb = encoder(x_fa_cb)
                    z_fb_ca = encoder(x_fb_ca)
                    z_fb_cb = encoder(x_fb_cb)

                    c_fa_ca, s_fa_ca = z_fa_ca[:, :1024], z_fa_ca[:, 1024:]
                    c_fa_cb, s_fa_cb = z_fa_cb[:, :1024], z_fa_cb[:, 1024:]
                    c_fb_ca, s_fb_ca = z_fb_ca[:, :1024], z_fb_ca[:, 1024:]
                    c_fb_cb, s_fb_cb = z_fb_cb[:, :1024], z_fb_cb[:, 1024:]
                adv_loss = torch.tensor(0.0, device=device)

            # Contrastive Regularization (SCR + CCR) 
            ctr_cfg = cfg.get("contrastive", {})
            use_ctr = bool(ctr_cfg.get("enable", False)) and (use_encoder and encoder is not None)
            if use_ctr:
                tau = float(ctr_cfg.get("tau", 0.07))
                do_norm = bool(ctr_cfg.get("normalize", True))
                lam_ctr = float(ctr_cfg.get("lambda", 0.05))
                w_style = float(ctr_cfg.get("weight", {}).get("style", 1.0))
                w_content = float(ctr_cfg.get("weight", {}).get("content", 1.0))
                do_detach = bool(ctr_cfg.get("detach", True))

                
                if do_detach:
                    c_fa_ca = c_fa_ca.detach(); c_fa_cb = c_fa_cb.detach()
                    c_fb_ca = c_fb_ca.detach(); c_fb_cb = c_fb_cb.detach()
                    s_fa_ca = s_fa_ca.detach(); s_fa_cb = s_fa_cb.detach()
                    s_fb_ca = s_fb_ca.detach(); s_fb_cb = s_fb_cb.detach()

                # SCR
                style_loss_fa = info_nce_pairwise(
                    anchor=s_fa_ca, positive=s_fa_cb,
                    negatives=torch.cat([s_fb_ca, s_fb_cb], dim=0),
                    tau=tau, do_norm=do_norm
                )
                style_loss_fb = info_nce_pairwise(
                    anchor=s_fb_cb, positive=s_fb_ca,
                    negatives=torch.cat([s_fa_ca, s_fa_cb], dim=0),
                    tau=tau, do_norm=do_norm
                )
                scr_loss = 0.5 * (style_loss_fa + style_loss_fb)

                # CCR
                content_loss_ca = info_nce_pairwise(
                    anchor=c_fa_ca, positive=c_fb_ca,
                    negatives=torch.cat([c_fa_cb, c_fb_cb], dim=0),
                    tau=tau, do_norm=do_norm
                )
                content_loss_cb = info_nce_pairwise(
                    anchor=c_fa_cb, positive=c_fb_cb,
                    negatives=torch.cat([c_fa_ca, c_fb_ca], dim=0),
                    tau=tau, do_norm=do_norm
                )
                ccr_loss = 0.5 * (content_loss_ca + content_loss_cb)

                contrastive_loss = w_style * scr_loss + w_content * ccr_loss
            else:
                scr_loss = torch.tensor(0.0, device=device)
                ccr_loss = torch.tensor(0.0, device=device)
                contrastive_loss = torch.tensor(0.0, device=device)

            
            loss = loss + lambda_adv * adv_loss + (lam_ctr * contrastive_loss if use_ctr else 0.0)

            # log
            if cfg.get("wandb", {}).get("enable", False) and step % cfg["wandb"].get("log_interval", 50) == 0:
                log_dict = {
                    "train/loss": loss.item(),
                }
                if use_adv:
                    log_dict.update({
                        "train/adv_loss": adv_loss.item(),
                    })
                if use_ctr:
                    log_dict.update({
                        "train/scr_loss": scr_loss.item(),
                        "train/ccr_loss": ccr_loss.item(),
                        "train/contrastive_loss": contrastive_loss.item(),
                    })
                wandb.log(log_dict, step=step)

            if step % 500 == 0 and use_encoder and encoder is not None:
                # L2 norm
                c_norm = lambda x: F.normalize(x, dim=1)
                s_norm = lambda x: F.normalize(x, dim=1)

                # cosine sim 
                cos_content = F.cosine_similarity(c_norm(c_fa_ca), c_norm(c_fb_ca), dim=1).mean()
                cos_style = F.cosine_similarity(s_norm(s_fa_ca), s_norm(s_fa_cb), dim=1).mean()

                if cfg.get("wandb", {}).get("enable", False):
                    wandb.log({
                        "train/cosine_content": cos_content.item(),
                        "train/cosine_style": cos_style.item(),
                    }, step=step)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ((list(encoder.parameters()) if (use_encoder and encoder is not None) else []) + list(denoiser.parameters())),
                max_norm=1.0,
            )
            opt.step()
            lr_scheduler.step()

            # EMA
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

        # Visualization
        if cfg["wandb"]["enable"] and cfg["vis"]["enable"]:
            if use_ema:
                if ema_encoder is not None and encoder is not None:
                    ema_encoder.store(encoder.parameters())
                    ema_encoder.copy_to(encoder)
                if ema_denoiser is not None:
                    ema_denoiser.store(denoiser.parameters())
                    ema_denoiser.copy_to(denoiser)

            train_batch = next(iter(dl_scsf))
            log_sample_images(
                step,
                train_batch,
                encoder,
                denoiser,
                scheduler,
                decoder_vae,
                vae_cfg,
                denorm,
                device,
                name="train/quad_grid",
                use_encoder=use_encoder,
                cfg={**cfg, "current_epoch": epoch},
                algo_type=algo_type,
                siamese_models=siamese_models,
            )

            if use_ema:
                if ema_encoder is not None and encoder is not None:
                    ema_encoder.restore(encoder.parameters())
                if ema_denoiser is not None:
                    ema_denoiser.restore(denoiser.parameters())

        # Validation
        if (epoch + 1) % cfg["eval"]["interval"] == 0:
            for split_name, loader in val_loaders.items():
                if use_ema:
                    if ema_encoder is not None and encoder is not None:
                        ema_encoder.store(encoder.parameters())
                        ema_encoder.copy_to(encoder)
                    if ema_denoiser is not None:
                        ema_denoiser.store(denoiser.parameters())
                        ema_denoiser.copy_to(denoiser)

                val_loss, vis_batch = evaluate(loader, encoder, denoiser, scheduler, device, use_encoder, algo_type)

                if cfg["wandb"]["enable"]:
                    log_losses(step=step, loss_dict={f"valid/{split_name}_loss": val_loss})

                if cfg["vis"]["enable"]:
                    log_sample_images(
                        step,
                        vis_batch,
                        encoder,
                        denoiser,
                        scheduler,
                        decoder_vae,
                        vae_cfg,
                        denorm,
                        device,
                        name=f"val/{split_name}/quad_grid",
                        use_encoder=use_encoder,
                        cfg={**cfg, "current_epoch": epoch},
                        algo_type=algo_type,
                        siamese_models=siamese_models,
                    )

                if use_ema:
                    if ema_encoder is not None and encoder is not None:
                        ema_encoder.restore(encoder.parameters())
                    if ema_denoiser is not None:
                        ema_denoiser.restore(denoiser.parameters())


        # save ckpt
        if (epoch + 1) % cfg["train"]["save_interval"] == 0:
            save_ckpt(
                cfg, epoch, encoder, denoiser, opt,
                ema_encoder=ema_encoder if use_ema else None,
                ema_denoiser=ema_denoiser if use_ema else None,
                lr_scheduler=lr_scheduler,
            )
        
        variance_cfg = cfg.get("analyze", {})
        use_variance_eval = bool(variance_cfg.get("enable", False))
        save_interval = int(variance_cfg.get("interval", 50))
        save_dir = Path(variance_cfg.get("save_dir", "analysis"))

        if use_variance_eval and ((epoch + 1) % save_interval == 0):
            encoder.eval()
            print(f"[Analyze] Computing codebook at epoch {epoch+1}...")

            # load latent PT
            pt_path = cfg["dataset"]["pt_path"]
            blob = torch.load(pt_path, map_location=device)
            latents = blob["latents"] if isinstance(blob, dict) and "latents" in blob else blob
            latents = latents.to(device)
            N, H, W, C = latents.shape
            latents = latents.permute(0, 3, 1, 2).contiguous().view(N, -1)

            # encoder
            with torch.no_grad():
                z_all = []
                for i in tqdm(range(0, N, 512), desc="Encode PT"):
                    z_batch = latents[i:i+512]
                    z_batch = z_batch.to(device=device, dtype=next(encoder.parameters()).dtype)  
                    z = encoder(z_batch)
                    z_all.append(z)
                z_all = torch.cat(z_all, dim=0)  # (N, 2048)

            c_all, s_all = z_all[:, :1024], z_all[:, 1024:]

            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(c_all.cpu(), save_dir / f"epoch_{epoch+1:04d}_C_codebook.pt")
            torch.save(s_all.cpu(), save_dir / f"epoch_{epoch+1:04d}_S_codebook.pt")

            # intra-class variance 
            with open(cfg["dataset"]["font_json"], "r", encoding="utf-8") as f:
                fonts = json.load(f)
            with open(cfg["dataset"]["chars_path"], "r", encoding="utf-8") as f:
                chars = [line.strip() for line in f if line.strip()]

            m, n = len(chars), len(fonts)

            # c variance
            c_vars = []
            for ci in range(m):
                subset = c_all[ci::m]  # same content indexing 
                c_vars.append(subset.var(dim=0).mean().item())
            mean_c_var = sum(c_vars) / len(c_vars)

            # s variance
            s_vars = []
            for fi in range(n):
                subset = s_all[fi*m : (fi+1)*m]  # same font indexing 
                s_vars.append(subset.var(dim=0).mean().item())
            mean_s_var = sum(s_vars) / len(s_vars)

            print(f"[Analyze] mean_content_var={mean_c_var:.6f}, mean_style_var={mean_s_var:.6f}")


            if cfg.get("wandb", {}).get("enable", False):
                wandb.log({
                    "analyze/mean_content_var": mean_c_var,
                    "analyze/mean_style_var": mean_s_var,
                }, step=step)



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

        # t = scheduler.sample_timesteps(B)  # sample t∈[0,T) -- DDPM
        t = scheduler.sample_timesteps(B, device=fb_ca.device)  # sample t∈[0,T). -- Flow Matching
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
    step,
    batch,
    encoder,
    denoiser,
    scheduler,
    decoder,
    vae_cfg,
    denorm,
    device,
    name,
    num_show=16,
    num_display=4,
    use_encoder: bool = True,
    cfg: dict = None,
    algo_type: str = "ddpm",
    siamese_models: dict = None,
):
    """
    Generate sample grids, optionally refine latents using SimpleRefinerUNet.
    Adds:
      1. Refine-before/after comparison logging
      2. Delayed refiner start (refiner.start_epoch)
    """
    to_tensor = transforms.ToTensor()
    # Use deterministic sampling + noise for cross-experiment alignment
    # cpu_rng_state = torch.get_rng_state()
    # cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    # py_rng_state = random.getstate()
    # base_seed = cfg.get("seed", 1234) if cfg else 1234
    # torch.manual_seed(base_seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(base_seed)
    # random.seed(base_seed)
    try:
        B_all = batch["F_A+C_A"].size(0)
        idx = list(range(min(num_show, B_all)))

        fa_ca = batch["F_A+C_A"][idx].to(device).view(-1, 1024)
        fb_cb = batch["F_B+C_B"][idx].to(device).view(-1, 1024)
        fa_cb = batch["F_A+C_B"][idx].to(device).view(-1, 1024)
        fb_ca = batch["F_B+C_A"][idx].to(device).view(-1, 1024)

        # build condition vectors 
        if use_encoder and encoder is not None:
            z_a = encoder(fa_ca)
            z_b = encoder(fb_cb)
            cond1 = torch.cat([z_a[:, :1024], z_b[:, 1024:]], dim=1)  # for fb_ca
            cond2 = torch.cat([z_b[:, :1024], z_a[:, 1024:]], dim=1)  # for fa_cb
        else:
            cond1 = torch.cat([fa_ca, fb_cb], dim=1)
            cond2 = torch.cat([fb_cb, fa_ca], dim=1)

        # CFG handling 
        use_cfg = bool(cfg.get("train", {}).get("cfg", {}).get("enable", False)) if cfg else False
        cfg_scale_local = float(cfg.get("train", {}).get("cfg", {}).get("scale", 3.0)) if cfg else 3.0

        if use_cfg and algo_type == "flow_matching":
            guided_denoiser = FM_CFG_Wrapper(denoiser, cfg_scale_local)
        elif use_cfg:
            guided_denoiser = CFGDenoiser(denoiser, cfg_scale_local)
        else:
            guided_denoiser = denoiser

        #  Generate latent samples 
        lat_fb_ca = scheduler.p_sample_loop(guided_denoiser, (len(idx), 1024), cond1, device)
        lat_fa_cb = scheduler.p_sample_loop(guided_denoiser, (len(idx), 1024), cond2, device)

        #  Optional Refiner 
        ref_cfg = cfg.get("refiner", {}) if cfg else {}
        use_refiner = bool(ref_cfg.get("enable", False))
        start_epoch = int(ref_cfg.get("start_epoch", 0))
        current_epoch = int(cfg.get("current_epoch", 0))  
        refiner_applied = False

       
        if use_refiner and (current_epoch >= start_epoch):
            from models.refiner import SimpleRefinerUNet
            ckpt_dir = Path(ref_cfg.get("ckpt_dir", "checkpoints/refiner"))
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "latest.pth"

            refiner = SimpleRefinerUNet(in_ch=4, base_ch=64, out_ch=4).to(device)
            opt_ref = optim.AdamW(refiner.parameters(), lr=float(ref_cfg.get("lr", 1e-4)))
            lambda_l1 = float(ref_cfg.get("lambda_l1", 1.0))
            train_steps = int(ref_cfg.get("train_steps", 100))
            batch_ref = int(ref_cfg.get("batch_size", 16))

            # load checkpoint if exists
            if ckpt_path.exists():
                refiner.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"[Refiner] Loaded from {ckpt_path}")

            # refiner
            if "train" in name:
                refiner.train()
                print(f"[Refiner] Training {train_steps} steps ...")
                opt_ref.zero_grad()
                input_latents = torch.cat([lat_fa_cb, lat_fb_ca], dim=0).detach().view(-1, 4, 16, 16)
                target_latents = torch.cat([fa_cb, fb_ca], dim=0).detach().view(-1, 4, 16, 16)

                with torch.enable_grad():
                    for i in range(train_steps):
                        idx_rand = torch.randint(0, input_latents.size(0), (min(batch_ref, input_latents.size(0)),))
                        inp = input_latents[idx_rand]
                        tgt = target_latents[idx_rand]
                        out = refiner(inp)
                        loss_ref = F.l1_loss(out, tgt) * lambda_l1
                        opt_ref.zero_grad()
                        loss_ref.backward()
                        opt_ref.step()
                        if (i + 1) % 20 == 0:
                            print(f"[Refiner] Step {i+1}/{train_steps}, L1={loss_ref.item():.5f}")
                            if cfg.get("wandb", {}).get("enable", False):
                                wandb.log({f"{name}/refiner_L1": loss_ref.item()}, step=step)
                torch.save(refiner.state_dict(), ckpt_path)
                print(f"[Refiner] Saved to {ckpt_path}")
            else:
                refiner.eval()

            
            lat_fa_cb_raw = lat_fa_cb.clone()
            lat_fb_ca_raw = lat_fb_ca.clone()

           
            with torch.no_grad():
                lat_fa_cb = refiner(lat_fa_cb.view(-1, 4, 16, 16)).view(-1, 1024)
                lat_fb_ca = refiner(lat_fb_ca.view(-1, 4, 16, 16)).view(-1, 1024)
            refiner_applied = True
            print("[Refiner] Applied for inference.")
        else:
            refiner = None
            refiner_applied = False

        # Decode Function
        def reshape_and_decode(flat_latents):
            latents = flat_latents.view(-1, 4, 16, 16)
            decoded_imgs = []
            for lat in latents:
                lat = denorm(lat)
                img = vae_decode(lat, decoder, vae_cfg, device)
                decoded_imgs.append(to_tensor(img))
            return decoded_imgs
        
        #  Decode all sets 
        img_gt_fa_cb = reshape_and_decode(fa_cb)
        img_gt_fb_ca = reshape_and_decode(fb_ca)
        img_fa_ca = reshape_and_decode(fa_ca)
        img_fb_cb = reshape_and_decode(fb_cb)
        img_gen_fa_cb = reshape_and_decode(lat_fa_cb)
        img_gen_fb_ca = reshape_and_decode(lat_fb_ca)
        
        # Optional: decode before-refine version
        if refiner_applied:
            img_gen_fa_cb_raw = reshape_and_decode(lat_fa_cb_raw)
            img_gen_fb_ca_raw = reshape_and_decode(lat_fb_ca_raw)
        else:
            img_gen_fa_cb_raw = img_gen_fa_cb
            img_gen_fb_ca_raw = img_gen_fb_ca
            
        # Grid Visualization (main)
        imgs = []
        for i in range(min(num_display, len(idx))):
            row = torch.cat([
            img_gt_fa_cb[i],
            img_gt_fb_ca[i],
            img_fa_ca[i],
            img_fb_cb[i],
            img_gen_fa_cb[i],   # after refiner
            img_gen_fb_ca[i]
            ], dim=2)
            imgs.append(row)
        grid = make_grid(torch.stack(imgs), nrow=1, padding=2)
        wandb.log({name: wandb.Image(grid)}, step=step)
        
        #  Refiner Before/After Comparison 
        if refiner_applied:
            imgs_compare = []
            for i in range(min(num_display, len(idx))):
                row = torch.cat([
                img_gen_fa_cb_raw[i],   # before
                img_gen_fa_cb[i],       # after
                img_gt_fa_cb[i],        # GT
                ], dim=2)
                imgs_compare.append(row)
            grid_ref = make_grid(torch.stack(imgs_compare), nrow=1, padding=2)
            wandb.log({f"{name}/refiner_compare": wandb.Image(grid_ref)}, step=step)
            
        # Metric Evaluation 
        wandb_enabled = cfg.get("wandb", {}).get("enable", False) if cfg else False
        if not wandb_enabled:
            return
        
        img_gen_fa_cb = torch.stack(img_gen_fa_cb)
        img_gen_fb_ca = torch.stack(img_gen_fb_ca)
        img_gt_fa_cb = torch.stack(img_gt_fa_cb)
        img_gt_fb_ca = torch.stack(img_gt_fb_ca)
        
        metrics_fa_cb = compute_metrics(img_gen_fa_cb, img_gt_fa_cb)
        metrics_fb_ca = compute_metrics(img_gen_fb_ca, img_gt_fb_ca)
        logs = {f"{name}/metric_{k}": (metrics_fa_cb[k] + metrics_fb_ca[k]) / 2 for k in metrics_fa_cb}
        
        if siamese_models:
            content_model = siamese_models["content"]
            style_model = siamese_models["style"]
            siamese_device = siamese_models.get("device", device)
        
        def batch_scores(preds, content_refs, style_refs):
            content_scores, style_scores = [], []
            for pred, c_ref, s_ref in zip(preds, content_refs, style_refs):
                content_scores.append(siamese_score_pair(content_model, pred, c_ref, siamese_device))
                style_scores.append(siamese_score_pair(style_model, pred, s_ref, siamese_device))
            return content_scores, style_scores
        
        fb_content_scores, fb_style_scores = batch_scores(img_gen_fb_ca, img_fa_ca, img_fb_cb)
        fa_content_scores, fa_style_scores = batch_scores(img_gen_fa_cb, img_fb_cb, img_fa_ca)
        
        def avg(values): return float(sum(values) / max(len(values), 1)) if values else 0.0

        logs.update({
            f"{name}/fb_ca_content_score": avg(fb_content_scores),
            f"{name}/fb_ca_style_score": avg(fb_style_scores),
            f"{name}/fa_cb_content_score": avg(fa_content_scores),
            f"{name}/fa_cb_style_score": avg(fa_style_scores),
            })
            
        wandb.log(logs, step=step)

    finally:
        pass
        # torch.set_rng_state(cpu_rng_state)
        # if cuda_rng_state is not None:
        #     torch.cuda.set_rng_state_all(cuda_rng_state)
        # random.setstate(py_rng_state)

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
    # GUARD
    if encoder is not None:
        ckpt["encoder"] = encoder.state_dict()
    if ema_encoder is not None:
        ckpt["ema_encoder"] = ema_encoder.state_dict()
    if ema_denoiser is not None:
        ckpt["ema_denoiser"] = ema_denoiser.state_dict()
    if lr_scheduler is not None:
        ckpt["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(ckpt, ckpt_path)
