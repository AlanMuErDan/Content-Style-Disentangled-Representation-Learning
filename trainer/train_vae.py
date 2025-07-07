# VAE_train.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml
import lpips

from models import build_encoder, build_decoder, build_quantizer
from models.discriminator import build_discriminator
from utils.losses import reconstruction_loss, kl_penalty, lpips_loss_fn
from utils.logger import init_wandb, log_single_image, log_loss, log_epoch, log_losses, log_latents
from dataset.font_dataset import SingleFontDataset

def train_vae_loop(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    use_gan = config.get("use_gan", False)
    print(f"Using GAN: {use_gan}")

    # Model
    encoder = build_encoder(config["encoder"], output_dim=None).to(device)
    decoder = build_decoder(config["decoder"], latent_dim=None, img_size=config["img_size"]).to(device)
    vq = build_quantizer(config).get("content")
    if vq:
        vq = vq.to(device)
    if use_gan:
        discriminator = build_discriminator(config).to(device)

    # Optimizer
    gen_modules = [encoder, decoder] + ([vq] if vq else [])
    gen_optim = torch.optim.Adam([p for m in gen_modules for p in m.parameters()], lr=config["lr"])
    if use_gan:
        disc_optim = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])

    # Dataset & wandb
    dataset = SingleFontDataset(config["data_root"], img_size=config["img_size"])
    print(f"Found {len(dataset)} images in dataset.")
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)
    init_wandb(config)

    for epoch in range(config["epochs"]):
        encoder.train(); decoder.train()
        if use_gan:
            discriminator.train()
        losses = []

        for i, img in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            img = img.to(device)

            mu, logvar = encoder(img)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            print(f"Encoder output shape: {z.shape}")
            vq_loss = 0.0
            if vq:
                z, loss_vq = vq(z)
                vq_loss += loss_vq

            if use_gan:
                with torch.no_grad():
                    fake_detach = decoder(z).detach()
                real_pred = discriminator(img)
                fake_pred = discriminator(fake_detach)
                d_loss = torch.relu(1.0 - real_pred).mean() + torch.relu(1.0 + fake_pred).mean()
                disc_optim.zero_grad()
                d_loss.backward()
                disc_optim.step()
            else:
                d_loss = torch.tensor(0.0, device=device)

            rec = decoder(z)
            print(f"Decoder output shape: {rec.shape}")
            recon_l2 = reconstruction_loss(rec, img)
            lpips_l = lpips_loss_fn(lpips_model, [rec], [img])
            kl_l = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1) / mu.numel()

            if use_gan:
                g_fake = discriminator(rec)
                g_loss = -g_fake.mean()
            else:
                g_loss = torch.tensor(0.0, device=device)

            total_loss = (
                recon_l2 +
                config["lpips_weight"] * lpips_l +
                config["kl_weight"] * kl_l +
                config["vq_loss_weight"] * vq_loss +
                config["gan_weight"] * g_loss
            )

            gen_optim.zero_grad()
            total_loss.backward()
            gen_optim.step()

            losses.append(total_loss.item())

            if i % 100 == 0:
                log_single_image(img, rec)
                log_loss(step=epoch * 1000 + i, loss=total_loss.item())
                log_losses(step=epoch * 1000 + i, loss_dict={
                    "recon_l2": recon_l2.item(),
                    "kl": kl_l.item(),
                    "lpips": lpips_l.item(),
                    "vq": vq_loss.item() if vq else 0.0,
                    "g_gan": g_loss.item(),
                    "d_gan": d_loss.item()
                })
                log_latents(epoch * 1000 + i, z=z)

        avg = sum(losses) / len(losses)
        print(f"[Epoch {epoch}] Avg Loss: {avg:.4f}")
        log_epoch(epoch, avg)

        # Save checkpoint
        if (epoch + 1) % config.get("checkpoint_interval", 1) == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vq": vq.state_dict() if vq else None,
                "epoch": epoch
            }, f"checkpoints/vae_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}") 