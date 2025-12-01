# trainer/train_disentangle_sd_pro.py

import random
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb

from dataset.font_dataset import (
    FourWayFontPairLatentPTDataset,
    filter_dataset_fonts,
    split_chars,
    split_fonts,
    sample_holdout_quads,
)
from models.encoder import CNNDisentangleEncoder
from models.DDPM import GaussianDiffusion as DDPMNoiseScheduler
from models.flow_matching import FlowMatching, LinearRFPath
from models.unet import CSPatchUNetDenoiser, SimplePatchUNetDenoiser
from utils.ema import LitEma
from utils.evaluate_basic import compute_metrics
from utils.siamese_scores import (
    load_model as load_siamese_model,
    score_pair as siamese_score_pair,
    DEFAULT_CONTENT_CKPT,
    DEFAULT_STYLE_CKPT,
)
from utils.lr_scheduler import (
    Scheduler_LinearWarmup,
    Scheduler_LinearWarmup_CosineDecay,
)
from utils.logger import init_wandb, log_losses
from utils.vae_io import decode as vae_decode, load_models


def seed_worker(worker_id):
    """Deterministic worker seeding to align DataLoader randomness across runs."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CFGPatchDenoiser:
    """Wrapper that delegates to denoiser.forward_with_cfg for CFG sampling."""

    def __init__(self, denoiser, cfg_scale: float):
        self.denoiser = denoiser
        self.cfg_scale = cfg_scale

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor):
        return self.denoiser.forward_with_cfg(x, t, tokens, self.cfg_scale)


def build_dataloaders(cfg: dict):
    font_json = cfg["dataset"]["font_json"]
    chars_path = cfg["dataset"]["chars_path"]
    split_mode = cfg["dataset"].get("train_test_split", "quad-split").lower()
    seed = cfg.get("seed", 1234)
    g = torch.Generator().manual_seed(seed)

    split_mode = cfg["dataset"].get("train_test_split", "quad-split").lower()

    if split_mode == "soft-train-test-split":
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
    ds_holdout = None
    ds_scuf = ds_ucsf = ds_ucuf = None

    filter_dataset_fonts(ds_scsf, train_fonts)
    ds_scsf.apply_char_filter(train_chars)

    if split_mode != "soft-train-test-split":
        ds_scuf = make_ds(2000)
        ds_ucsf = make_ds(2000)
        ds_ucuf = make_ds(2000)

        filter_dataset_fonts(ds_scuf, valid_fonts)
        ds_scuf.apply_char_filter(train_chars)

        filter_dataset_fonts(ds_ucsf, train_fonts)
        ds_ucsf.apply_char_filter(valid_chars)

        filter_dataset_fonts(ds_ucuf, valid_fonts)
        ds_ucuf.apply_char_filter(valid_chars)

    if split_mode == "soft-train-test-split":
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
    if split_mode != "soft-train-test-split":
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
        val_loaders = {
            "SCUF": dl_scuf,
            "UCSF": dl_ucsf,
            "UCUF": dl_ucuf,
        }

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


def combine_tokens(denoiser, content_a: torch.Tensor, style_b: torch.Tensor) -> torch.Tensor:
    return denoiser.build_tokens(content_a, style_b)


def sample_timesteps(noise_sched, batch_size: int, device, algo_type: str):
    if algo_type == "ddpm":
        return noise_sched.sample_timesteps(batch_size)
    return noise_sched.sample_timesteps(batch_size, device=device)


def make_scheduler(cfg: dict, device):
    algo_type = cfg.get("algo", {}).get("type", "flow_matching").lower()
    sampler_cfg = cfg["denoiser"].get("timestep_sampler", {})
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
            ode_steps=int(fm_cfg.get("ode_steps", cfg.get("sample", {}).get("steps", 50))),
        ).to(device)
    return scheduler


def train_disentangle_loop_sd_pro(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 1234))
    random.seed(cfg.get("seed", 1234))

    encoder_vae, decoder_vae, vae_cfg, _ = load_models(cfg["vis"]["vae_config"], cfg["vis"]["vae_ckpt"], device)

    enc_cfg = cfg.get("encoder", {})
    encoder = CNNDisentangleEncoder(
        in_channels=enc_cfg.get("in_channels", 4),
        hidden_channels=enc_cfg.get("hidden_channels", 64),
        depth=enc_cfg.get("depth", 3),
        content_channels=enc_cfg.get("content_channels", 4),
        style_channels=enc_cfg.get("style_channels", 4),
    ).to(device)

    dl_scsf, val_loaders, denorm = build_dataloaders(cfg)

    den_cfg = cfg.get("denoiser", {})
    arch = str(den_cfg.get("arch", "transformer_patch")).lower()
    sample_size = cfg["dataset"].get("latent_size", 16)
    latent_channels = cfg["dataset"].get("latent_channels", 4)
    patch_size = den_cfg.get("patch_size", enc_cfg.get("patch_size", 2))
    if arch == "simple_patch":
        denoiser = SimplePatchUNetDenoiser(
            sample_size=sample_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            base_channels=den_cfg.get("base_channels", 160),
            channel_mult=tuple(den_cfg.get("channel_mults", [1, 2])),
            num_heads=den_cfg.get("num_heads", 4),
            ctx_dim=den_cfg.get("ctx_dim", 96),
            content_channels=enc_cfg.get("content_channels", 4),
            style_channels=enc_cfg.get("style_channels", 4),
            patch_size=patch_size,
            use_learned_null=den_cfg.get("use_learned_null", True),
        ).to(device)
    else:
        denoiser = CSPatchUNetDenoiser(
            sample_size=sample_size,
            in_channels=latent_channels,
            out_channels=latent_channels,
            base_channels=den_cfg.get("base_channels", 256),
            channel_mult=tuple(den_cfg.get("channel_mults", [1, 2, 2])),
            num_res_blocks=den_cfg.get("num_res_blocks", 2),
            num_heads=den_cfg.get("num_heads", 8),
            ctx_dim=den_cfg.get("ctx_dim", 128),
            content_channels=enc_cfg.get("content_channels", 4),
            style_channels=enc_cfg.get("style_channels", 4),
            patch_size=patch_size,
            use_learned_null=den_cfg.get("use_learned_null", True),
        ).to(device)

    params = list(encoder.parameters()) + list(denoiser.parameters())
    opt = optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    scheduler_cfg = cfg["train"].get("scheduler", {})
    base_lr = float(cfg["train"]["lr"])
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
    step_per_epoch = len(dl_scsf)
    total_steps = step_per_epoch * cfg["train"]["epochs"]
    warmup_steps = warmup_epochs * step_per_epoch
    min_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

    sched_type = str(scheduler_cfg.get("type", "none")).lower()
    if sched_type == "linear-warmup":
        lr_lambda = Scheduler_LinearWarmup(warmup_steps)
    elif sched_type == "linear-warmup_cosine-decay":
        lr_lambda = Scheduler_LinearWarmup_CosineDecay(warmup_steps, total_steps, min_ratio)
    else:
        lr_lambda = lambda step: 1.0
    lr_scheduler = LambdaLR(opt, lr_lambda)

    ema_cfg = cfg["train"].get("ema", {})
    use_ema = bool(ema_cfg.get("enable", False))
    ema_decay = float(ema_cfg.get("decay", 0.9999))
    ema_encoder = LitEma(encoder, decay=ema_decay) if use_ema else None
    ema_denoiser = LitEma(denoiser, decay=ema_decay) if use_ema else None

    algo_type = cfg.get("algo", {}).get("type", "flow_matching").lower()
    noise_sched = make_scheduler(cfg, device)
    sampler_cfg = cfg["denoiser"].get("timestep_sampler", {})

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

    cfg_train = cfg.get("train", {}).get("cfg", {})
    cfg_enable = bool(cfg_train.get("enable", False))
    p_uncond = float(cfg_train.get("p_uncond", 0.0))
    cfg_scale = float(cfg_train.get("scale", 3.0))

    step = 0
    printed_debug = False
    debug_tokens_saved = None
    debug_input_latent = None
    for epoch in range(cfg["train"]["epochs"]):
        encoder.train()
        denoiser.train()

        if step < sampler_cfg.get("warmup", 10000):
            if algo_type == "ddpm":
                noise_sched.t_sampler = "uniform"
            else:
                noise_sched.path.t_sampler = "uniform"
        else:
            if algo_type == "ddpm":
                noise_sched.t_sampler = sampler_cfg.get("type", "uniform")
            else:
                noise_sched.path.t_sampler = sampler_cfg.get("type", "uniform")

        pbar = tqdm(dl_scsf, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            x_fa_ca = batch["F_A+C_A"].to(device)
            x_fa_cb = batch["F_A+C_B"].to(device)
            x_fb_ca = batch["F_B+C_A"].to(device)
            x_fb_cb = batch["F_B+C_B"].to(device)

            content_fa, style_fa = encoder(x_fa_ca)
            content_fb, style_fb = encoder(x_fb_cb)

            cond_tokens_cA_sB = combine_tokens(denoiser, content_fa, style_fb)
            cond_tokens_cB_sA = combine_tokens(denoiser, content_fb, style_fa)

            if not printed_debug:
                debug_content = content_fa[:1].detach()
                debug_style = style_fb[:1].detach()
                debug_tokens = cond_tokens_cA_sB[:1].detach()
                print("[SANITY] Encoder outputs -> content:", tuple(debug_content.shape), "style:", tuple(debug_style.shape))
                print("[SANITY] Patch tokens shape:", tuple(debug_tokens.shape))
                debug_tokens_saved = debug_tokens
                debug_input_latent = x_fb_ca[:1].detach()

            if cfg_enable and p_uncond > 0.0:
                drop = (torch.rand(x_fa_ca.size(0), device=device) < p_uncond).float().view(-1, 1, 1)
                cond_tokens_cA_sB = cond_tokens_cA_sB * (1.0 - drop)
                cond_tokens_cB_sA = cond_tokens_cB_sA * (1.0 - drop)

            B = x_fb_ca.size(0)
            t = sample_timesteps(noise_sched, B, x_fb_ca.device, algo_type)

            if not printed_debug:
                debug_t = t[:1].detach()
                try:
                    if hasattr(denoiser, "debug_shapes"):
                        denoiser.debug_shapes(debug_input_latent, debug_t, debug_tokens_saved)
                    else:
                        out_dbg = denoiser(debug_input_latent, debug_t, debug_tokens_saved)
                        print("[SANITY] Denoiser output:", tuple(out_dbg.shape))
                except Exception as err:
                    print(f"[SANITY] Denoiser debug failed: {err}")
                printed_debug = True

            eps1 = torch.randn_like(x_fb_ca)
            eps2 = torch.randn_like(x_fa_cb)
            x_t1 = noise_sched.q_sample(x_fb_ca.view(B, -1), t, eps1.view(B, -1)).view_as(x_fb_ca)
            x_t2 = noise_sched.q_sample(x_fa_cb.view(B, -1), t, eps2.view(B, -1)).view_as(x_fa_cb)

            if algo_type == "ddpm":
                eps_hat1 = denoiser(x_t1, t, cond_tokens_cA_sB)
                eps_hat2 = denoiser(x_t2, t, cond_tokens_cB_sA)
                loss = 0.5 * (F.mse_loss(eps_hat1, eps1) + F.mse_loss(eps_hat2, eps2))
            else:
                v_target1 = x_fb_ca - eps1
                v_target2 = x_fa_cb - eps2
                v_pred1 = denoiser(x_t1, t, cond_tokens_cA_sB)
                v_pred2 = denoiser(x_t2, t, cond_tokens_cB_sA)
                loss = 0.5 * (F.mse_loss(v_pred1, v_target1) + F.mse_loss(v_pred2, v_target2))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
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

        if cfg["wandb"]["enable"] and cfg["vis"]["enable"]:
            if use_ema:
                ema_encoder.store(encoder.parameters())
                ema_encoder.copy_to(encoder)
                ema_denoiser.store(denoiser.parameters())
                ema_denoiser.copy_to(denoiser)

            train_batch = next(iter(dl_scsf))
            log_sample_images(
                cfg,
                step,
                train_batch,
                encoder,
                denoiser,
                noise_sched,
                decoder_vae,
                vae_cfg,
                denorm,
                device,
                name="train/quad_grid",
                cfg_scale=cfg_scale,
                algo_type=algo_type,
                siamese_models=siamese_models,
            )

            if use_ema:
                ema_encoder.restore(encoder.parameters())
                ema_denoiser.restore(denoiser.parameters())

        if (epoch + 1) % cfg["eval"]["interval"] == 0:
            if use_ema:
                ema_encoder.store(encoder.parameters())
                ema_encoder.copy_to(encoder)
                ema_denoiser.store(denoiser.parameters())
                ema_denoiser.copy_to(denoiser)

            for split_name, loader in val_loaders.items():
                val_loss, vis_batch = evaluate(
                    loader,
                    encoder,
                    denoiser,
                    noise_sched,
                    device,
                    algo_type,
                )

                if cfg["wandb"]["enable"]:
                    log_losses(step=step, loss_dict={f"valid/{split_name}_loss": val_loss})

                if cfg["vis"]["enable"]:
                    log_sample_images(
                        cfg,
                        step,
                        vis_batch,
                        encoder,
                        denoiser,
                        noise_sched,
                        decoder_vae,
                        vae_cfg,
                        denorm,
                        device,
                        name=f"val/{split_name}/quad_grid",
                        cfg_scale=cfg_scale,
                        algo_type=algo_type,
                        siamese_models=siamese_models,
                    )

            if use_ema:
                ema_encoder.restore(encoder.parameters())
                ema_denoiser.restore(denoiser.parameters())

        if (epoch + 1) % cfg["train"]["save_interval"] == 0:
            save_ckpt(
                cfg,
                epoch,
                encoder,
                denoiser,
                opt,
                ema_encoder=ema_encoder if use_ema else None,
                ema_denoiser=ema_denoiser if use_ema else None,
                lr_scheduler=lr_scheduler,
            )


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    encoder: CNNDisentangleEncoder,
    denoiser,
    scheduler,
    device,
    algo_type: str,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    encoder.eval()
    denoiser.eval()
    losses, sample_batch = [], None

    pbar = tqdm(loader, desc="Eval")
    for batch in pbar:
        fa_ca = batch["F_A+C_A"].to(device)
        fa_cb = batch["F_A+C_B"].to(device)
        fb_ca = batch["F_B+C_A"].to(device)
        fb_cb = batch["F_B+C_B"].to(device)

        content_fa, style_fa = encoder(fa_ca)
        content_fb, style_fb = encoder(fb_cb)

        cond1 = combine_tokens(denoiser, content_fa, style_fb)
        cond2 = combine_tokens(denoiser, content_fb, style_fa)

        B = fa_ca.size(0)
        t = sample_timesteps(scheduler, B, fa_ca.device, algo_type)

        eps1 = torch.randn_like(fb_ca)
        eps2 = torch.randn_like(fa_cb)
        x_t1 = scheduler.q_sample(fb_ca.view(B, -1), t, eps1.view(B, -1)).view_as(fb_ca)
        x_t2 = scheduler.q_sample(fa_cb.view(B, -1), t, eps2.view(B, -1)).view_as(fa_cb)

        if algo_type == "ddpm":
            eps_hat1 = denoiser(x_t1, t, cond1)
            eps_hat2 = denoiser(x_t2, t, cond2)
            batch_loss = 0.5 * (F.mse_loss(eps_hat1, eps1) + F.mse_loss(eps_hat2, eps2)).item()
        else:
            v_tgt1 = fb_ca - eps1
            v_tgt2 = fa_cb - eps2
            v_hat1 = denoiser(x_t1, t, cond1)
            v_hat2 = denoiser(x_t2, t, cond2)
            batch_loss = 0.5 * (F.mse_loss(v_hat1, v_tgt1) + F.mse_loss(v_hat2, v_tgt2)).item()

        losses.append(batch_loss)
        sample_batch = batch

    return sum(losses) / len(losses), sample_batch


@torch.no_grad()
def log_sample_images(
    cfg: dict,
    step: int,
    batch: Dict[str, torch.Tensor],
    encoder: CNNDisentangleEncoder,
    denoiser,
    scheduler,
    decoder,
    vae_cfg,
    denorm,
    device,
    name: str,
    cfg_scale: float,
    algo_type: str,
    num_show: int = 16,
    num_display: int = 4,
    siamese_models: dict = None,
):
    to_tensor = transforms.ToTensor()
    B = batch["F_A+C_A"].size(0)
    idx = list(range(min(num_show, B)))

    fa_ca = batch["F_A+C_A"][idx].to(device)
    fb_cb = batch["F_B+C_B"][idx].to(device)
    fa_cb = batch["F_A+C_B"][idx].to(device)
    fb_ca = batch["F_B+C_A"][idx].to(device)

    content_fa, style_fa = encoder(fa_ca)
    content_fb, style_fb = encoder(fb_cb)

    cond1 = combine_tokens(denoiser, content_fa, style_fb)
    cond2 = combine_tokens(denoiser, content_fb, style_fa)

    if cfg.get("train", {}).get("cfg", {}).get("enable", False):
        guided = CFGPatchDenoiser(denoiser, cfg_scale)
    else:
        guided = denoiser

    lat_fb_ca = scheduler.p_sample_loop(guided, fb_ca.shape, cond1, device)
    lat_fa_cb = scheduler.p_sample_loop(guided, fa_cb.shape, cond2, device)

    def reshape_and_decode(latents_flat: torch.Tensor):
        decoded_imgs = []
        for lat in latents_flat:
            lat = denorm(lat)
            img = vae_decode(lat, decoder, vae_cfg, device)
            decoded_imgs.append(to_tensor(img))
        return decoded_imgs

    img_gt_fa_cb = reshape_and_decode(fa_cb)
    img_gt_fb_ca = reshape_and_decode(fb_ca)
    img_fa_ca = reshape_and_decode(fa_ca)
    img_fb_cb = reshape_and_decode(fb_cb)
    img_gen_fa_cb = reshape_and_decode(lat_fa_cb)
    img_gen_fb_ca = reshape_and_decode(lat_fb_ca)

    imgs = []
    for i in range(min(num_display, len(idx))):
        row = torch.cat(
            [
                img_gt_fa_cb[i],
                img_gt_fb_ca[i],
                img_fa_ca[i],
                img_fb_cb[i],
                img_gen_fa_cb[i],
                img_gen_fb_ca[i],
            ],
            dim=2,
        )
        imgs.append(row)

    grid = make_grid(torch.stack(imgs), nrow=1, padding=2)
    wandb.log({name: wandb.Image(grid)}, step=step)

    wandb_enabled = cfg.get("wandb", {}).get("enable", False)
    if not wandb_enabled:
        return

    img_fa_ca = torch.stack(img_fa_ca)
    img_fb_cb = torch.stack(img_fb_cb)
    img_gt_fa_cb = torch.stack(img_gt_fa_cb)
    img_gt_fb_ca = torch.stack(img_gt_fb_ca)
    img_gen_fa_cb = torch.stack(img_gen_fa_cb)
    img_gen_fb_ca = torch.stack(img_gen_fb_ca)

    logs = {}
    metrics_fa_cb = compute_metrics(img_gen_fa_cb, img_gt_fa_cb)
    metrics_fb_ca = compute_metrics(img_gen_fb_ca, img_gt_fb_ca)
    logs.update(
        {f"{name}/metric_{k}": (metrics_fa_cb[k] + metrics_fb_ca[k]) / 2 for k in metrics_fa_cb}
    )

    if siamese_models:
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

        logs.update(
            {
                f"{name}/fb_ca_content_score": avg(fb_content_scores),
                f"{name}/fb_ca_style_score": avg(fb_style_scores),
                f"{name}/fa_cb_content_score": avg(fa_content_scores),
                f"{name}/fa_cb_style_score": avg(fa_style_scores),
            }
        )

    if logs:
        wandb.log(logs, step=step)


def save_ckpt(
    cfg: dict,
    epoch: int,
    encoder: nn.Module,
    denoiser: nn.Module,
    optimizer: optim.Optimizer,
    ema_encoder=None,
    ema_denoiser=None,
    lr_scheduler=None,
) -> None:
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
