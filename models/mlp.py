# models/mlp.py

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint



class ResidualMLP(nn.Module):
    """
    used for disentangle DDPM encoder
    """
    def __init__(self, input_dim=1024, hidden_dim=2048, num_layers=4):
        super().__init__()
        print(f"[ResidualMLP] Creating with input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        self.layers = nn.ModuleList()
        self.gates = nn.ParameterList()

        for i in range(num_layers):
            print(f"  - Layer {i}: Linear({hidden_dim}, {hidden_dim})")
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.gates.append(nn.Parameter(torch.tensor(1.0)))

        self.project_residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        x = self.project_residual(x)
        for i, (layer, gate) in enumerate(zip(self.layers, self.gates)):
            out = layer(x)
            if i < len(self.layers) - 1:
                out = torch.relu(out)
            assert out.shape == x.shape, f"Residual shape mismatch at layer {i}: x={x.shape}, out={out.shape}"
            x = x + gate * out
        return x



def build_residual_mlp(input_dim=1024, hidden_dim=2048, num_layers=4):
    return ResidualMLP(input_dim, hidden_dim, num_layers)

def build_mlp(input_dim, hidden_dim, num_layers):
    layers = []
    in_dim = input_dim
    for i in range(num_layers - 1):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    return nn.Sequential(*layers)

"""
below are used from:
https://github.com/LTH14/mar 
"""

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels)
        )

    def forward(self, x, y):
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift, scale)
        h = self.mlp(h)
        return x + gate * h



class FinalLayer(nn.Module):
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, grad_checkpointing=False):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.res_blocks = nn.ModuleList([
            ResBlock(model_channels) for _ in range(num_res_blocks)
        ])

        self.final_layer = FinalLayer(model_channels, out_channels)
        self.grad_checkpointing = grad_checkpointing

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c
        for block in self.res_blocks:
            if self.grad_checkpointing:
                x = checkpoint(block, x, y)
            else:
                x = block(x, y)
        return self.final_layer(x, y)


if __name__ == "__main__":
    print("--- Residual MLP ---")
    model1 = build_residual_mlp(input_dim=1024, hidden_dim=2048, num_layers=4)
    x1 = torch.randn(8, 1024)
    out1 = model1(x1)
    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {out1.shape}")

    print("\n--- SimpleMLPAdaLN (Kaiming) ---")
    model2 = SimpleMLPAdaLN(
        in_channels=1024,      # x = flattened latent from 16x16x4
        model_channels=2048,   # internal width
        out_channels=1024,     # output = predicted noise, same shape as x
        z_channels=2048,       # condition vector dimension (e.g. from style+content MLP)
        num_res_blocks=4
    )
    x2 = torch.randn(8, 1024)     # noisy latent
    t = torch.randint(0, 1000, (8,))  # timestep
    c = torch.randn(8, 2048)      # condition
    out2 = model2(x2, t, c)
    print(f"Input shape: {x2.shape}, Condition shape: {c.shape}, Output shape: {out2.shape}")