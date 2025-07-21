# utils/gan_loss.py

import torch
import torch.nn as nn

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
