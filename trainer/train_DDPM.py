# trainer/train_DDPM.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import init_wandb, log_images, log_loss, log_epoch
from models import build_encoder, build_decoder, build_mlp
import torch.nn.functional as F
import yaml

# Load config
config_path = "configs/config.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

start_epoch = 0

def sample_ddpm_images(decoder, latents, device):
    decoder.eval()
    with torch.no_grad():
        return decoder.sample(latents, steps=100)

def train_stage2_ddpm(config, dataset, freeze_ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model components
    encoder = build_encoder(config['encoder'], output_dim=config['latent_dim']).to(device)
    mlp_s = build_mlp(config['latent_dim'], config['latent_dim']).to(device)
    mlp_c = build_mlp(config['latent_dim'], config['latent_dim']).to(device)
    decoder = build_decoder("ddpm", latent_dim=2 * config['latent_dim'], img_size=config['img_size']).to(device)

    # Load and freeze pretrained encoder and mlps
    ckpt = torch.load(freeze_ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    mlp_s.load_state_dict(ckpt['mlp_s'])
    mlp_c.load_state_dict(ckpt['mlp_c'])

    encoder.eval()
    mlp_s.eval()
    mlp_c.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in mlp_s.parameters():
        p.requires_grad = False
    for p in mlp_c.parameters():
        p.requires_grad = False

    # Only optimize decoder
    optim = torch.optim.Adam(decoder.parameters(), lr=config['lr'])

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    init_wandb(config)

    for epoch in range(config['epochs']):
        decoder.train()
        losses = []

        for i, (imgA, _, _, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config['epochs']}"):
            imgA = imgA.to(device)
            with torch.no_grad():
                z = encoder(imgA)
                zs, zc = mlp_s(z), mlp_c(z)
                z_latent = torch.cat([zs, zc], dim=1)

            noise_pred, noise = decoder(imgA, z_latent)
            loss = F.mse_loss(noise_pred, noise)

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

            if i % 100 == 0:
                with torch.no_grad():
                    samples = sample_ddpm_images(decoder, z_latent, device)
                log_images(imgA, samples, samples, imgA, imgA, samples, samples, imgA)  # simplified logging
                log_loss(step=epoch * 1000 + i, loss=loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        log_epoch(epoch, avg_loss)

        torch.save({"decoder": decoder.state_dict(), "epoch": epoch}, f"checkpoints/ddpm/ddpm_epoch_{epoch}.pth")
