# trainer/train_disentangle.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.losses import reconstruction_loss, kl_penalty, lpips_loss_fn
from utils.logger import init_wandb, log_images, log_loss, log_epoch, log_losses, log_latents
from models import build_encoder, build_decoder, build_quantizer
from models.discriminator import build_discriminator
import torch.nn.functional as F
import yaml
import lpips

# 读取配置
config_path = "configs/config.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

start_epoch = 0
lpips_model = lpips.LPIPS(net='vgg').cuda()

def train_loop(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # 配置开关
    use_gan = config.get("use_gan", False)
    print(f"Using GAN: {use_gan}")

    # 模型组件
    encoder_c = build_encoder(config['encoder_c'], output_dim=config['latent_dim']).to(device)
    encoder_s = build_encoder(config['encoder_s'], output_dim=config['latent_dim']).to(device)
    decoder = build_decoder(config['decoder'], latent_dim=2 * config['latent_dim'], img_size=config['img_size']).to(device)
    if use_gan:
        discriminator = build_discriminator(config).to(device)

    print(f"Using device: {device}")
    print(f"Encoder C: {config['encoder_c']}, Encoder S: {config['encoder_s']}, Decoder: {config['decoder']}")

    # VQ模块
    vq_dict = build_quantizer(config)
    if vq_dict.get("content"):
        vq_dict["content"] = vq_dict["content"].to(device)
    if vq_dict.get("style"):
        vq_dict["style"] = vq_dict["style"].to(device)

    # 加载权重
    resume_ckpt_path = config.get("resume_ckpt", "")
    if resume_ckpt_path and os.path.isfile(resume_ckpt_path):
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        encoder_c.load_state_dict(ckpt["encoder_c"])
        encoder_s.load_state_dict(ckpt["encoder_s"])
        decoder.load_state_dict(ckpt["decoder"])
        if "vq_content" in ckpt and vq_dict.get("content"):
            vq_dict["content"].load_state_dict(ckpt["vq_content"])
        if "vq_style" in ckpt and vq_dict.get("style"):
            vq_dict["style"].load_state_dict(ckpt["vq_style"])
        global start_epoch
        start_epoch = ckpt.get("epoch", 0) + 1

    # 优化器
    gen_modules = [encoder_c, encoder_s, decoder]
    if vq_dict.get("content"):
        gen_modules.append(vq_dict["content"])
    if vq_dict.get("style"):
        gen_modules.append(vq_dict["style"])
    gen_optim = torch.optim.Adam([p for m in gen_modules for p in m.parameters()], lr=config['lr'])
    if use_gan:
        disc_optim = torch.optim.Adam(discriminator.parameters(), lr=config['lr'])

    # dataloader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    init_wandb(config)

    for epoch in range(start_epoch, config['epochs']):
        encoder_c.train(); encoder_s.train(); decoder.train()
        if use_gan:
            discriminator.train()
        losses = []

        for i, (imgA, imgB, gt_crossAB, gt_crossBA) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config['epochs']}"):
            imgA, imgB = imgA.to(device), imgB.to(device)
            gt_crossAB, gt_crossBA = gt_crossAB.to(device), gt_crossBA.to(device)

            # 编码
            zcA = encoder_c(imgA)
            zcB = encoder_c(imgB)
            zsA = encoder_s(imgA)
            zsB = encoder_s(imgB)

            # VQ量化
            vq_loss = 0.0
            if vq_dict.get("content"):
                zcA, loss_contentA = vq_dict["content"](zcA)
                zcB, loss_contentB = vq_dict["content"](zcB)
                vq_loss += (loss_contentA + loss_contentB) / 2
            if vq_dict.get("style"):
                zsA, loss_styleA = vq_dict["style"](zsA)
                zsB, loss_styleB = vq_dict["style"](zsB)
                vq_loss += (loss_styleA + loss_styleB) / 2

            # 判别器训练（可选）
            if use_gan:
                recA_detach = decoder(torch.cat([zsA, zcA], dim=1)).detach()
                recB_detach = decoder(torch.cat([zsB, zcB], dim=1)).detach()
                real = torch.cat([imgA, imgB], dim=0)
                fake = torch.cat([recA_detach, recB_detach], dim=0)
                real_pred = discriminator(real)
                fake_pred = discriminator(fake)
                d_loss = F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()

                disc_optim.zero_grad()
                d_loss.backward()
                disc_optim.step()
            else:
                d_loss = torch.tensor(0.0, device=device)

            # 生成器阶段
            recA = decoder(torch.cat([zsA, zcA], dim=1))
            recB = decoder(torch.cat([zsB, zcB], dim=1))
            crossAB = decoder(torch.cat([zcA, zsB], dim=1))
            crossBA = decoder(torch.cat([zcB, zsA], dim=1))

            recon_loss = (
                reconstruction_loss(recA, imgA) +
                reconstruction_loss(recB, imgB) +
                reconstruction_loss(crossAB, gt_crossAB) +
                reconstruction_loss(crossBA, gt_crossBA)
            ) / 4

            lpips_loss = lpips_loss_fn(lpips_model, [recA, recB, crossAB, crossBA], [imgA, imgB, gt_crossAB, gt_crossBA])
            kl_loss = kl_penalty(zcA, zcB, zsA, zsB)

            if use_gan:
                fake = torch.cat([recA, recB], dim=0)
                fake_pred = discriminator(fake)
                g_gan_loss = -fake_pred.mean()
            else:
                g_gan_loss = torch.tensor(0.0, device=device)

            total_loss = recon_loss \
                + config.get("vq_loss_weight", 1.0) * vq_loss \
                + config.get("kl_weight", 0.1) * kl_loss \
                + config.get("lpips_weight", 0.8) * lpips_loss \
                + config.get("gan_weight", 0.1) * g_gan_loss

            gen_optim.zero_grad()
            total_loss.backward()
            gen_optim.step()

            losses.append(total_loss.item())

            if i % 100 == 0:
                log_images(imgA, recA, crossBA, gt_crossBA, imgB, recB, crossAB, gt_crossAB)
                log_loss(step=epoch * 1000 + i, loss=total_loss.item())
                log_losses(step=epoch * 1000 + i, loss_dict={
                    "kl_loss": kl_loss.item(),
                    "lpips_loss": lpips_loss.item(),
                    "recon_l2": recon_loss.item(),
                    "gan_gen_loss": g_gan_loss.item(),
                    "gan_disc_loss": d_loss.item()
                })
                log_latents(epoch * 1000 + i, zcA=zcA, zsA=zsA)

        avg_loss = sum(losses) / len(losses)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
        log_epoch(epoch, avg_loss)

        os.makedirs("checkpoints", exist_ok=True)
        ckpt = {
            "encoder_c": encoder_c.state_dict(),
            "encoder_s": encoder_s.state_dict(),
            "decoder": decoder.state_dict(),
            "epoch": epoch
        }
        if vq_dict.get("content"):
            ckpt["vq_content"] = vq_dict["content"].state_dict()
        if vq_dict.get("style"):
            ckpt["vq_style"] = vq_dict["style"].state_dict()

        torch.save(ckpt, f"checkpoints/ckpt_epoch_{epoch}.pth")