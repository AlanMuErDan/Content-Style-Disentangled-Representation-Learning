# utils/losses.py

import torch
import torch.nn.functional as F
import torch.nn as nn

# reconstruction loss 
def reconstruction_loss(pred, target):
    return F.l1_loss(pred, target)

# KL penalty 
def kl_penalty(*latents):
    """Compute KL loss as L2 norm over multiple latent vectors."""
    return sum([z.pow(2).mean() for z in latents]) / len(latents)

# LPIPS loss 
def lpips_loss_fn(lpips_model, preds, targets):
    loss = 0
    for p, t in zip(preds, targets):
        d = lpips_model(p, t)
        if torch.isnan(d).any():
            print("Warning: LPIPS returned NaN, skipping this pair.")
            continue
        loss += d.mean()
    return loss / len(preds)

# GAN loss module 
class GANLossModule(nn.Module):
    def __init__(self, discriminator, gan_type="hinge", lecam_weight=0.1):
        super().__init__()
        self.disc = discriminator
        self.gan_type = gan_type
        self.lambda_lecam = lecam_weight

    def forward(self, real_img, fake_img, mode):
        """
        mode = 1: discriminator update
        mode = 0: generator update
        """
        if mode == 1:
            with torch.no_grad():
                fake_img = fake_img.detach()
            real_pred = self.disc(real_img)
            fake_pred = self.disc(fake_img)

            print(f"GAN Disc: Real pred shape: {real_pred.shape}, Fake pred shape: {fake_pred.shape}")

            if self.gan_type == "hinge":
                loss_gan = torch.relu(1.0 - real_pred.view(-1)).mean() + torch.relu(1.0 + fake_pred.view(-1)).mean()
            elif self.gan_type == "bce":
                loss_gan = nn.functional.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) + \
                           nn.functional.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            else:
                raise NotImplementedError

            lecam_reg = ((real_pred - fake_pred) ** 2).view(-1).mean()
            return loss_gan + self.lambda_lecam * lecam_reg

        else:
            fake_pred = self.disc(fake_img)

            if self.gan_type == "hinge":
                loss = -fake_pred.view(-1).mean()
            elif self.gan_type == "bce":
                loss = nn.functional.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
            else:
                raise NotImplementedError

            return loss
