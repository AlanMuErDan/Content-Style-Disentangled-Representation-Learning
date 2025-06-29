import torch.nn as nn

class VQ(nn.Module):
    def __init__(self, codebook_size=512, code_dim=256):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(self, z):
        z_flat = z.view(z.size(0), -1)
        dist = (z_flat.unsqueeze(1) - self.codebook.weight).pow(2).sum(-1)
        indices = dist.argmin(1)
        z_q = self.codebook(indices)
        return z_q, indices


def build_quantizer(config):
    quantizers = {}

    if config.get("vq_content", False):
        quantizers["content"] = VQ(
            codebook_size=config.get("vq_content_codebook_size", 512),
            code_dim=config.get("latent_dim", 256)
        )

    if config.get("vq_style", False):
        quantizers["style"] = VQ(
            codebook_size=config.get("vq_style_codebook_size", 512),
            code_dim=config.get("latent_dim", 256)
        )

    return quantizers  