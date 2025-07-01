# models/quantizer.py

# Updated VQ module with commitment loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class VQ(nn.Module):
    def __init__(self, codebook_size=512, code_dim=256, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.codebook.weight.data.uniform_(-1 / code_dim, 1 / code_dim) # Initialize codebook weights

    def forward(self, z):
        z_flat = z.view(z.size(0), -1)
        dist = (z_flat.unsqueeze(1) - self.codebook.weight).pow(2).sum(-1)
        indices = dist.argmin(1)   # [B]
        z_q = self.codebook(indices)  # [B, code_dim]

        # VQ losses
        commitment_loss = F.mse_loss(z_q.detach(), z_flat) # backprop update encoder 
        codebook_loss = F.mse_loss(z_q, z_flat.detach()) # backprop update codebook
        loss = self.beta * commitment_loss + codebook_loss

        z_q = z_flat + (z_q - z_flat).detach() # STE

        return z_q, loss


def build_quantizer(config):
    quantizers = {}

    if config.get("vq_content", False):
        quantizers["content"] = VQ(
            codebook_size=config.get("vq_content_codebook_size", 512),
            code_dim=config.get("latent_dim", 256),
            beta=config.get("vq_beta", 0.25)
        )
        print("Using VQ for content quantization with codebook size:", 
              config.get("vq_content_codebook_size", 512))

    if config.get("vq_style", False):
        quantizers["style"] = VQ(
            codebook_size=config.get("vq_style_codebook_size", 512),
            code_dim=config.get("latent_dim", 256),
            beta=config.get("vq_beta", 0.25)
        )
        print("Using VQ for style quantization with codebook size:", 
              config.get("vq_style_codebook_size", 256))

    return quantizers