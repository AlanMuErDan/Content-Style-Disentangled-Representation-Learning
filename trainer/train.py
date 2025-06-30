# trainer/train.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.losses import reconstruction_loss
from utils.logger import init_wandb, log_images, log_loss, log_epoch
from models import build_encoder, build_decoder, build_quantizer, build_mlp
import random 
import torch.nn.functional as F

# read configuration file
import yaml
config_path = "configs/config.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

start_epoch = 0


def sample_ddpm_images(decoder, latents, device):
    decoder.eval()
    with torch.no_grad():
        samples = decoder.sample(latents, steps=100)
    return samples


def train_loop(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model components
    encoder = build_encoder(config['encoder'], output_dim=config['latent_dim']).to(device)
    decoder = build_decoder(config['decoder'], latent_dim=2 * config['latent_dim'], img_size=config['img_size']).to(device)
    print(f"Using {config['encoder']} encoder and {config['decoder']} decoder.")
    mlp_s = build_mlp(config['latent_dim'], config['latent_dim']).to(device)
    mlp_c = build_mlp(config['latent_dim'], config['latent_dim']).to(device)
    vq_dict = build_quantizer(config)  # {'content': VQ or None, 'style': VQ or None}
    if vq_dict.get("content"):
        vq_dict["content"] = vq_dict["content"].to(device)
    if vq_dict.get("style"):
        vq_dict["style"] = vq_dict["style"].to(device)

    # resume checkpoint
    resume_ckpt_path = config.get("resume_ckpt", "")
    if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        mlp_s.load_state_dict(ckpt["mlp_s"])
        mlp_c.load_state_dict(ckpt["mlp_c"])
        if "vq_content" in ckpt and vq_dict.get("content"):
            vq_dict["content"].load_state_dict(ckpt["vq_content"])
        if "vq_style" in ckpt and vq_dict.get("style"):
            vq_dict["style"].load_state_dict(ckpt["vq_style"])
        global start_epoch
        start_epoch = ckpt.get("epoch", 0) + 1

    # optimizer
    modules = [encoder, decoder, mlp_s, mlp_c]
    if vq_dict.get("content"):
        modules.append(vq_dict["content"])
    if vq_dict.get("style"):
        modules.append(vq_dict["style"])
    optim = torch.optim.Adam(
        [p for m in modules for p in m.parameters()],
        lr=config['lr']
    )

    # dataloader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    init_wandb(config)

    use_ddpm = config['decoder'] == 'ddpm'

    for epoch in range(start_epoch, config['epochs']):
        encoder.train(); decoder.train(); mlp_s.train(); mlp_c.train()
        losses = []

        for i, (imgA, imgB, gt_crossAB, gt_crossBA) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config['epochs']}"):
            imgA, imgB = imgA.to(device), imgB.to(device)
            gt_crossAB, gt_crossBA = gt_crossAB.to(device), gt_crossBA.to(device)

            # Encode
            zA = encoder(imgA)
            zB = encoder(imgB)
            zsA, zcA = mlp_s(zA), mlp_c(zA)
            zsB, zcB = mlp_s(zB), mlp_c(zB)

            # VQ
            if vq_dict.get("content"):
                zcA, _ = vq_dict["content"](zcA)
                zcB, _ = vq_dict["content"](zcB)
            if vq_dict.get("style"):
                zsA, _ = vq_dict["style"](zsA)
                zsB, _ = vq_dict["style"](zsB)

            if use_ddpm:
                # pick one of four tasks at each step (random or round robin)
                choice = random.choice(["recA", "recB", "crossAB", "crossBA"])

                if choice == "recA":
                    noise_pred, noise = decoder(imgA, torch.cat([zsA, zcA], dim=1))
                elif choice == "recB":
                    noise_pred, noise = decoder(imgB, torch.cat([zsB, zcB], dim=1))
                elif choice == "crossAB":
                    noise_pred, noise = decoder(gt_crossAB, torch.cat([zsB, zcA], dim=1))
                else:
                    noise_pred, noise = decoder(gt_crossBA, torch.cat([zsA, zcB], dim=1))

                loss = F.mse_loss(noise_pred, noise)
            else:
                recA = decoder(torch.cat([zsA, zcA], dim=1))
                recB = decoder(torch.cat([zsB, zcB], dim=1))
                crossAB = decoder(torch.cat([zcA, zsB], dim=1))
                crossBA = decoder(torch.cat([zcB, zsA], dim=1))

                loss = (
                    reconstruction_loss(recA, imgA) +
                    reconstruction_loss(recB, imgB) +
                    reconstruction_loss(crossAB, gt_crossAB) +
                    reconstruction_loss(crossBA, gt_crossBA)
                ) / 4

            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())

            if i % 100 == 0:
                if use_ddpm:
                    with torch.no_grad():
                        sampleA = sample_ddpm_images(decoder, torch.cat([zsA, zcA], dim=1), device)
                        sampleB = sample_ddpm_images(decoder, torch.cat([zsB, zcB], dim=1), device)
                        sampleAB = sample_ddpm_images(decoder, torch.cat([zsB, zcA], dim=1), device)
                        sampleBA = sample_ddpm_images(decoder, torch.cat([zsA, zcB], dim=1), device)
                    log_images(imgA, sampleA, sampleBA, gt_crossBA, imgB, sampleB, sampleAB, gt_crossAB)
                else:
                    log_images(imgA, recA, crossBA, gt_crossBA, imgB, recB, crossAB, gt_crossAB)
                log_loss(step=epoch * 1000 + i, loss=loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        log_epoch(epoch, avg_loss)

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "mlp_s": mlp_s.state_dict(),
            "mlp_c": mlp_c.state_dict(),
            "epoch": epoch
        }
        if vq_dict.get("content"):
            ckpt["vq_content"] = vq_dict["content"].state_dict()
        if vq_dict.get("style"):
            ckpt["vq_style"] = vq_dict["style"].state_dict()

        torch.save(ckpt, f"checkpoints/ckpt_epoch_{epoch}.pth")
