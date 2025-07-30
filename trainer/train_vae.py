# trainer/train_vae.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.utils as nn_utils 
from torch.optim.lr_scheduler import LambdaLR 
import random 
import numpy as np 

import wandb

from tqdm import tqdm
import yaml
import lpips

from models import build_encoder, build_decoder, build_quantizer
from models.discriminator import build_discriminator
from utils.losses import reconstruction_loss, kl_penalty, lpips_loss_fn, GANLossModule
from utils.logger import init_wandb, log_single_image, log_loss, log_epoch, log_losses, log_latents
from utils.ema import LitEma
from utils.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from utils.save_ckpt import create_experiment_dir, CheckpointManager, build_state_dict
# from dataset.font_dataset import SingleFontDataset
from dataset.font_dataset import SingleFontLMDBDataset, get_all_fonts, get_all_lmdb_keys

def seed_everything(seed: int = 10086):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def run_validation(encoder, decoder, val_loader, lpips_model, device, global_step):
    encoder.eval(); decoder.eval()
    total_l2, total_lpips = 0.0, 0.0
    n_samples = 0
    random_batch = None

    with torch.no_grad():
        for i, val_img in tqdm(enumerate(val_loader)):
            val_img = val_img.to(device)
            mu, logvar = encoder(val_img)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            rec = decoder(z)

            l2 = reconstruction_loss(rec, val_img).item()
            lp = lpips_loss_fn(lpips_model, [rec], [val_img]).item()

            total_l2 += l2 * val_img.size(0)
            total_lpips += lp * val_img.size(0)
            n_samples += val_img.size(0)

            if random_batch is None and random.random() < 0.05:
                random_batch = (val_img.clone(), rec.clone())

        if random_batch is None:
            random_batch = (val_img, rec)
        log_single_image(random_batch[0], random_batch[1], split="val")

    avg_l2 = total_l2 / n_samples
    avg_lpips = total_lpips / n_samples
    log_losses(step=global_step, loss_dict={"val_recon_l2": avg_l2, "val_lpips": avg_lpips})
    print(f"[Validation] L2: {avg_l2:.4f}, LPIPS: {avg_lpips:.4f}")
    return avg_l2, avg_lpips

def train_vae_loop(config):
    seed_everything(config.get("seed", 10086))
    g = torch.Generator()
    g.manual_seed(config.get("seed", 10086))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    use_gan = config.get("use_gan", False)
    use_ema = config.get("use_ema", False)
    print(f"Using GAN: {use_gan}, EMA: {use_ema}")

    encoder = build_encoder(config["encoder"], output_dim=None).to(device)
    decoder = build_decoder(
        name=config["decoder"],
        latent_dim=None,
        latent_channels=config.get("latent_channels", 16),  # 保持和 encoder 一致
        img_size=config["img_size"]
    ).to(device)
    vq = build_quantizer(config).get("content")
    if vq:
        vq = vq.to(device)
    if use_gan:
        discriminator = build_discriminator(config).to(device)
        gan_loss = GANLossModule(discriminator, gan_type=config.get("gan_type", "hinge"),
                                 lecam_weight=config.get("lecam_weight", 0.1))
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

    gen_modules = [encoder, decoder] + ([vq] if vq else [])
    gen_optim = torch.optim.Adam([p for m in gen_modules for p in m.parameters()], lr=config["lr"])
    if use_gan:
        disc_optim = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])

    # train_dataset = SingleFontDataset(config["train_data_root"], img_size=config["img_size"])
    # print(f"Found {len(train_dataset)} images in dataset.")
    # dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)

    # val_dataset = SingleFontDataset(config["val_data_root"], img_size=config["img_size"])
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)

    # font_list = get_all_fonts(config["train_data_root"])  # or save它成json以复用

    all_keys = get_all_lmdb_keys(
        config["train_data_root"],
        cache_path=config.get("key_cache_path", "lmdb_keys.json")
    )
    print(f"Total {len(all_keys)} LMDB keys loaded.")


    m = config.get("train_sample_count", 10000)
    n = config.get("val_sample_count", 2000)
    assert m + n <= len(all_keys), "Not enough total samples."

    print(f"Using {m} training keys and {n} validation keys.")

    random.shuffle(all_keys)
    train_keys = all_keys[:m]
    val_keys = all_keys[m:m+n]

    train_dataset = SingleFontLMDBDataset(
        config["train_data_root"],
        img_size=config["img_size"],
        keys_subset=train_keys,
        augment_prob=config.get("augment_prob", 0.5)
    )

    val_dataset = SingleFontLMDBDataset(
        config["val_data_root"],
        img_size=config["img_size"],
        keys_subset=val_keys,
        augment_prob=0.5  # 验证禁用增强
    )

    print(f"Found {len(train_dataset)} images in dataset.")
    dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1, worker_init_fn=seed_worker, generator=g)
    print("Using LMDB dataset for training.")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1, worker_init_fn=seed_worker, generator=g)

    init_wandb(config)

    # LR Scheduler
    step_per_epoch = len(dataloader)
    total_steps = step_per_epoch * config["epochs"]
    warmup_steps = config.get("warmup_epochs", 0) * step_per_epoch
    min_lr = config.get("min_lr", 0.0)
    base_lr = config["lr"]
    scheduler_type = config.get("lr_scheduler", "None")

    global_step = 0

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

    val_interval = config.get("val_interval", 1000)
    train_log_interval = config.get("train_log_interval", 100)

    exp_dir = create_experiment_dir(config)  
    ckpt_mgr = CheckpointManager(
        exp_dir=exp_dir,
        monitor="val_recon_l2",
        mode="min"
    )

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
            # eps = torch.randn_like(std.size(), device=std.device)
            eps = torch.randn_like(std)
            z = mu + eps * std
            vq_loss = 0.0
            if vq:
                z, loss_vq = vq(z)
                vq_loss += loss_vq

            if use_gan:
                d_loss = gan_loss(real_img=img, fake_img=decoder(z), mode=1)
                disc_optim.zero_grad()
                d_loss.backward()
                disc_optim.step()
                disc_scheduler.step(global_step)
            else:
                d_loss = torch.tensor(0.0, device=device)

            print("decoder.conv_in.weight.shape:", decoder.conv_in.weight.shape)
            print("z.shape before decoder:", z.shape)
            rec = decoder(z)
            print(f"Reconstruction shape: {rec.shape}, Original shape: {img.shape}")
            recon_l2 = reconstruction_loss(rec, img)
            lpips_l = lpips_loss_fn(lpips_model, [rec], [img])
            kl_l = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1) / mu.numel()

            if use_gan:
                g_loss = gan_loss(real_img=None, fake_img=rec, mode=0)
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
            gen_scheduler.step(global_step) 

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
                if use_ema:
                    ema_encoder.store(encoder.parameters())
                    ema_decoder.store(decoder.parameters())
                    if ema_vq:
                        ema_vq.store(vq.parameters())

                    ema_encoder.copy_to(encoder)
                    ema_decoder.copy_to(decoder)
                    if ema_vq:
                        ema_vq.copy_to(vq)

                val_l2, val_lpips = run_validation(encoder, decoder, val_loader, lpips_model, device, global_step)

                if use_ema:
                    ema_encoder.restore(encoder.parameters())
                    ema_decoder.restore(decoder.parameters())
                    if ema_vq:
                        ema_vq.restore(vq.parameters())
                
                step_state = build_state_dict(
                    encoder=encoder,
                    decoder=decoder,
                    vq=vq,
                    gen_optim=gen_optim,
                    gen_scheduler=gen_scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    ema_encoder=ema_encoder if use_ema else None,
                    ema_decoder=ema_decoder if use_ema else None,
                    ema_vq=ema_vq if (use_ema and vq) else None,
                    extra={
                        "val_recon_l2": val_l2,
                        "val_lpips": val_lpips,
                        "from_step_validation": True
                    }
                )
                ckpt_mgr.maybe_update_best(metric_value=val_l2, state=step_state)

        avg = sum(losses) / len(losses)
        print(f"[Epoch {epoch}] Avg Loss: {avg:.4f}")
        log_epoch(epoch, avg)

        # Save checkpoint
        epoch_state = build_state_dict(
            encoder=encoder,
            decoder=decoder,
            vq=vq,
            gen_optim=gen_optim,
            gen_scheduler=gen_scheduler,
            epoch=epoch,
            global_step=global_step,
            ema_encoder=ema_encoder if use_ema else None,
            ema_decoder=ema_decoder if use_ema else None,
            ema_vq=ema_vq if (use_ema and vq) else None,
            extra={"train_avg_loss": avg}
        )

        ckpt_mgr.save_last(epoch_state)

        # # Save checkpoint
        # if (epoch + 1) % config.get("checkpoint_interval", 1) == 0:
        #     os.makedirs("checkpoints", exist_ok=True)
        #     if use_ema:
        #         # Save ema parameters
        #         torch.save({
        #             "encoder": {k: v.clone() for k, v in ema_encoder.named_buffers() if k in ema_encoder.m_name2s_name.values()},
        #             "decoder": {k: v.clone() for k, v in ema_decoder.named_buffers() if k in ema_decoder.m_name2s_name.values()},
        #             "vq": {k: v.clone() for k, v in ema_vq.named_buffers() if k in ema_vq.m_name2s_name.values()} if ema_vq else None,
        #             "epoch": epoch,
        #             "global_step": global_step,
        #             "gen_optim": gen_optim.state_dict(),
        #             "gen_scheduler": gen_scheduler.state_dict()
        #         }, f"checkpoints/vae_ema_epoch_{epoch}.pth")
        #     else:
        #         torch.save({
        #             "encoder": encoder.state_dict(),
        #             "decoder": decoder.state_dict(),
        #             "vq": vq.state_dict() if vq else None,
        #             "epoch": epoch,
        #             "global_step": global_step,
        #             "gen_optim": gen_optim.state_dict(),
        #             "gen_scheduler": gen_scheduler.state_dict()
        #         }, f"checkpoints/vae_epoch_{epoch}.pth")
        #     print(f"Checkpoint saved at epoch {epoch}")