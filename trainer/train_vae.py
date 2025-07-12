# trainer/train_vae.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import yaml
import wandb
import lpips

from models import build_encoder, build_decoder, build_quantizer
from models.discriminator import build_discriminator
from utils.losses import reconstruction_loss, kl_penalty, lpips_loss_fn
from utils.logger import init_wandb, log_single_image, log_loss, log_epoch, log_losses, log_latents
from utils.ema import LitEma
from utils.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from dataset.font_dataset import SingleFontDataset

def run_validation(encoder, decoder, val_loader, lpips_model, device, global_step):
    encoder.eval(); decoder.eval()
    total_l2, total_lpips = 0.0, 0.0
    n_samples = 0
    with torch.no_grad():
        for i, val_img in enumerate(val_loader):
            val_img = val_img.to(device)
            mu, logvar = encoder(val_img)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            rec = decoder(z)

            l2 = reconstruction_loss(rec, val_img).item()
            lp = lpips_loss_fn(lpips_model, [rec], [val_img]).item()

            total_l2 += l2 * val_img.size(0)
            total_lpips += lp * val_img.size(0)
            n_samples += val_img.size(0)

            if i == 0:
                log_single_image(val_img, rec, split="val")

    avg_l2 = total_l2 / n_samples
    avg_lpips = total_lpips / n_samples
    log_losses(step=global_step, loss_dict={"val_recon_l2": avg_l2, "val_lpips": avg_lpips})
    print(f"[Validation] L2: {avg_l2:.4f}, LPIPS: {avg_lpips:.4f}")


def train_vae_loop(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    use_gan = config.get("use_gan", False)
    use_ema = config.get("use_ema", False)
    print(f"Using GAN: {use_gan}, EMA: {use_ema}")

    # Model
    encoder = build_encoder(config["encoder"], output_dim=None).to(device)
    decoder = build_decoder(config["decoder"], latent_dim=None, img_size=config["img_size"]).to(device)
    vq = build_quantizer(config).get("content")
    if vq:
        vq = vq.to(device)
    if use_gan:
        discriminator = build_discriminator(config).to(device)
    # EMA initialization
    if use_ema:
        ema_encoder = LitEma(encoder)
        ema_decoder = LitEma(decoder)
        ema_vq = LitEma(vq) if vq else None

    start_epoch = 0
    if config.get("resume_ckpt") and os.path.exists(config["resume_ckpt"]):
        print(f"Resuming from checkpoint: {config['resume_ckpt']}")
        ckpt = torch.load(config["resume_ckpt"], map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        if vq and ckpt.get("vq") is not None:
            vq.load_state_dict(ckpt["vq"])
        start_epoch = ckpt.get("epoch", 0) + 1
    else:
        print("No checkpoint found or resume_ckpt not specified.")

    # Optimizer
    gen_modules = [encoder, decoder] + ([vq] if vq else [])
    gen_optim = torch.optim.Adam([p for m in gen_modules for p in m.parameters()], lr=config["lr"])
    if use_gan:
        disc_optim = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])

    # Dataset & wandb
    train_dataset = SingleFontDataset(config["train_data_root"], img_size=config["img_size"])
    print(f"Found {len(train_dataset)} images in dataset.")
    dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)

    val_dataset = SingleFontDataset(config["val_data_root"], img_size=config["img_size"])
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)

    init_wandb(config)

    # LR Scheduler
    step_per_epoch = len(dataloader)
    total_steps = step_per_epoch * config["epochs"]
    warmup_steps = config.get("warmup_epochs", 0) * step_per_epoch
    min_lr = config.get("min_lr", 0.0)
    base_lr = config["lr"]
    scheduler_type = config.get("lr_scheduler", "None")

    if scheduler_type == "linear-warmup":
        lr_lambda = Scheduler_LinearWarmup(warmup_steps)
    elif scheduler_type == "linear-warmup_cosine-decay":
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        lr_lambda = Scheduler_LinearWarmup_CosineDecay(warmup_steps, total_steps, min_ratio)
    else:
        lr_lambda = lambda step: 1.0  # constant lr

    gen_scheduler = LambdaLR(gen_optim, lr_lambda)
    if use_gan:
        disc_scheduler = LambdaLR(disc_optim, lr_lambda)

    global_step = 0
    val_interval = config.get("val_interval", 1000)
    train_log_interval = config.get("train_log_interval", 100)

    for epoch in range(start_epoch, start_epoch + config["epochs"]):
        encoder.train(); decoder.train()
        if use_gan:
            discriminator.train()
        losses = []

        for i, img in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            global_step += 1
            img = img.to(device)

            mu, logvar = encoder(img)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            # print(f"Encoder output shape: {z.shape}")
            vq_loss = 0.0
            if vq:
                z, loss_vq = vq(z)
                vq_loss += loss_vq

            if use_gan:
                with torch.no_grad():
                    fake_detach = decoder(z).detach()
                real_pred = discriminator(img)
                fake_pred = discriminator(fake_detach)
                d_loss_gan = torch.relu(1.0 - real_pred).mean() + torch.relu(1.0 + fake_pred).mean()

                # LeCam 正则项
                lecam_reg = ((real_pred - fake_pred) ** 2).mean()
                lambda_lecam = config.get("lecam_weight", 0.1)

                # 合并
                d_loss = d_loss_gan + lambda_lecam * lecam_reg
                
                disc_optim.zero_grad()
                d_loss.backward()

                # # Gradient clipping for GAN
                # if config.get("gradient_clip_val", 0.0) > 0:
                #     nn_utils.clip_grad_norm_(
                #         discriminator.parameters(),
                #         max_norm=config["gradient_clip_val"],
                #         norm_type=2
                #     )

                disc_optim.step()
                disc_scheduler.step()
            else:
                d_loss = torch.tensor(0.0, device=device)

            rec = decoder(z)
            # print(f"Decoder output shape: {rec.shape}")
            recon_l2 = reconstruction_loss(rec, img)
            lpips_l = lpips_loss_fn(lpips_model, [rec], [img])
            kl_l = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1) / mu.numel()

            if use_gan:
                g_fake = discriminator(rec)
                g_loss = -g_fake.mean()
            else:
                g_loss = torch.tensor(0.0, device=device)

            # KL Annealing
            if config.get("kl_anneal", {}).get("enabled", False):
                start = config["kl_anneal"].get("start_epoch", 0)
                end = config["kl_anneal"].get("end_epoch", 100)
                if epoch < start:
                    kl_weight = 0.0
                elif epoch >= end:
                    kl_weight = config["kl_weight"]
                else:
                    alpha = (epoch - start) / (end - start)
                    kl_weight = alpha * config["kl_weight"]
            else:
                kl_weight = config["kl_weight"]

            total_loss = (
                recon_l2 +
                config["lpips_weight"] * lpips_l +
                kl_weight * kl_l +
                config["vq_loss_weight"] * vq_loss +
                config["gan_weight"] * g_loss
            )

            gen_optim.zero_grad()
            total_loss.backward()

            if config.get("gradient_clip_val", 0.0) > 0:
                nn_utils.clip_grad_norm_(
                    [p for m in gen_modules for p in m.parameters() if p.requires_grad],
                    max_norm=config["gradient_clip_val"],
                    norm_type=2
                )

            gen_optim.step()
            gen_scheduler.step()

            # EMA update
            if use_ema:
                ema_encoder(encoder)
                ema_decoder(decoder)
                if ema_vq:
                    ema_vq(vq)

            losses.append(total_loss.item())

            if i % train_log_interval == 0:
                log_single_image(img, rec, split="train")
                log_loss(step=global_step, loss=total_loss.item())
                log_losses(step=global_step, loss_dict={
                    "recon_l2": recon_l2.item(),
                    "kl": kl_l.item(),
                    "lpips": lpips_l.item(),
                    "vq": vq_loss.item() if vq else 0.0,
                    "g_gan": g_loss.item(),
                    "d_gan": d_loss.item()
                })
                log_latents(global_step, z=z)
                wandb.log({
                    "lr": gen_scheduler.get_last_lr()[0],
                    "kl_weight": kl_weight
                }, step=global_step)

            if global_step % val_interval == 0:
                run_validation(encoder, decoder, val_loader, lpips_model, device, global_step)

        avg = sum(losses) / len(losses)
        print(f"[Epoch {epoch}] Avg Loss: {avg:.4f}")
        log_epoch(epoch, avg)

        # Save checkpoint
        if (epoch + 1) % config.get("checkpoint_interval", 1) == 0:
            os.makedirs("checkpoints", exist_ok=True)
            if use_ema:
                # Save ema parameters
                torch.save({
                    "encoder": {k: v.clone() for k, v in ema_encoder.named_buffers() if k in ema_encoder.m_name2s_name.values()},
                    "decoder": {k: v.clone() for k, v in ema_decoder.named_buffers() if k in ema_decoder.m_name2s_name.values()},
                    "vq": {k: v.clone() for k, v in ema_vq.named_buffers() if k in ema_vq.m_name2s_name.values()} if ema_vq else None,
                    "epoch": epoch
                }, f"checkpoints/vae_ema_epoch_{epoch}.pth")
            else:
                torch.save({
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "vq": vq.state_dict() if vq else None,
                    "epoch": epoch
                }, f"checkpoints/vae_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")