# models/ddpm_decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def sinusoidal_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResidualBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.cond_proj = nn.Linear(cond_emb_dim, channels)

    def forward(self, x, t_emb, z_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_out = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        z_out = self.cond_proj(z_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_out + z_out

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return x + h


# noise predictor: UNet 
class UNet(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, base_channels=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, base_channels), nn.ReLU(),
            nn.Linear(base_channels, base_channels)
        )
        self.latent_proj = nn.Linear(latent_dim, base_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + base_channels, base_channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), nn.ReLU()
        )

        self.middle = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1)  # output noise
        )

    def forward(self, x_t, t, z):
        t = t.view(-1, 1).float() / 1000 
        t_embed = self.time_embed(t).view(-1, self.latent_proj.out_features, 1, 1)
        z_embed = self.latent_proj(z).view(-1, self.latent_proj.out_features, 1, 1)
        z_map = z_embed.expand(-1, self.latent_proj.out_features, x_t.size(2), x_t.size(3))
        x = torch.cat([x_t, z_map], dim=1)

        h = self.encoder(x)
        h = self.middle(h + t_embed)
        out = self.decoder(h)
        return out

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(latent_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.res1 = ResidualBlock(base_channels * 2, time_emb_dim, time_emb_dim)

        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
        self.res2 = ResidualBlock(base_channels * 4, time_emb_dim, time_emb_dim)

        self.mid = ResidualBlock(base_channels * 4, time_emb_dim, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.res3 = ResidualBlock(base_channels * 2, time_emb_dim, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.res4 = ResidualBlock(base_channels, time_emb_dim, time_emb_dim)

        self.final = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t, z):
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_mlp[0].in_features))
        z_emb = self.cond_mlp(z)

        x = self.init_conv(x)
        x = self.down1(x)
        x = self.res1(x, t_emb, z_emb)
        x = self.down2(x)
        x = self.res2(x, t_emb, z_emb)

        x = self.mid(x, t_emb, z_emb)

        x = self.up1(x)
        x = self.res3(x, t_emb, z_emb)
        x = self.up2(x)
        x = self.res4(x, t_emb, z_emb)

        return self.final(x)

class NoiseScheduler(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        beta = torch.linspace(beta_start, beta_end, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x_start, t, noise):
        a_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return (a_bar.sqrt() * x_start + (1 - a_bar).sqrt() * noise)

class DDPMDecoder(nn.Module):
    def __init__(self, latent_dim=512, img_size=128, enhanced=False):
        super().__init__()
        if enhanced:
            self.noise_predictor = EnhancedUNet(in_channels=1, latent_dim=latent_dim)
        else:
            self.noise_predictor = UNet(in_channels=1, latent_dim=latent_dim)
        self.schedule = NoiseScheduler()

    def forward(self, x_start, z):
        B = x_start.size(0)
        t = torch.randint(0, self.schedule.beta.size(0), (B,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.schedule.q_sample(x_start, t, noise)
        noise_pred = self.noise_predictor(x_t, t, z)
        return noise_pred, noise

    def sample(self, z, steps=100):
        B = z.size(0)
        x = torch.randn((B, 1, 128, 128), device=z.device)
        for t in reversed(range(steps)):
            t_tensor = torch.full((B,), t, device=z.device, dtype=torch.long)
            noise_pred = self.noise_predictor(x, t_tensor, z)
            beta_t = self.schedule.beta[t].view(1, 1, 1, 1)
            alpha_t = self.schedule.alpha[t].view(1, 1, 1, 1)
            alpha_bar_t = self.schedule.alpha_bar[t].view(1, 1, 1, 1)

            x = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_bar_t).sqrt()) * noise_pred)
            if t > 0:
                x += beta_t.sqrt() * torch.randn_like(x)
        return x
