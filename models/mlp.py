# models/mlp.py

import torch
import torch.nn as nn
import math
from torch.utils.checkpoint import checkpoint



class ResidualMLP(nn.Module):
    """
    used for disentangle DDPM encoder
    """
    def __init__(self, input_dim=1024, hidden_dim=2048, num_layers=4, dropout=0.1, use_layernorm=True):
        super().__init__()
        print(f"[ResidualMLP] Creating with input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, "
              f"dropout={dropout}, layernorm={use_layernorm}")
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layernorm else None
        self.gates = nn.ParameterList()
        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.gates.append(nn.Parameter(torch.tensor(1.0)))
            if use_layernorm:
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.project_residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        x = self.project_residual(x)
        for i, (layer, gate) in enumerate(zip(self.layers, self.gates)):
            out = layer(x)
            if i < len(self.layers) - 1:
                if self.use_layernorm:
                    out = self.norms[i](out)
                out = torch.relu(out)
                out = self.dropout(out)
            assert out.shape == x.shape, f"Residual shape mismatch at layer {i}: x={x.shape}, out={out.shape}"
            x = x + gate * out
        return x



def build_residual_mlp(input_dim=1024, hidden_dim=2048, num_layers=4,
                       dropout=0.1, use_layernorm=True):
    return ResidualMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_layernorm=use_layernorm,
    )

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


# -------------------------
# SimpleMLPAdaLN with MAR zero-init + CFG-ready
# -------------------------
class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, grad_checkpointing=False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.eps_only = (out_channels == in_channels)
        self.vlb_mode = (out_channels == in_channels * 2)
        assert self.eps_only or self.vlb_mode, \
            f"out_channels must be in_channels (eps-only) or 2*in_channels (VLB). Got {out_channels}."

        self.input_proj = nn.Linear(in_channels, model_channels, bias=True)
        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels, bias=True)

        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)
        self.grad_checkpointing = grad_checkpointing

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _blocks_forward(self, x, y):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)
        return x

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c
        x = self._blocks_forward(x, y)
        return self.final_layer(x, y)

    @torch.no_grad()
    def forward_with_cfg(self, x, t, c, cfg_scale: float):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        out = self.forward(combined, t, c)  # [B, out_channels]

        if self.eps_only:
            eps = out
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([guided_eps, guided_eps], dim=0)
            return eps

        # VLB / learned-sigma mode
        eps, rest = out[:, :self.in_channels], out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([guided_eps, guided_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
    print("--- Residual MLP ---")
    model1 = build_residual_mlp(input_dim=1024, hidden_dim=2048, num_layers=4)
    x1 = torch.randn(8, 1024)
    out1 = model1(x1)
    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {out1.shape}")

    print("\n--- SimpleMLPAdaLN (eps-only) ---")
    model2 = SimpleMLPAdaLN(
        in_channels=1024,      # 16x16x4 flattened latent
        model_channels=2048,
        out_channels=1024,     # eps-only
        z_channels=2048,       # style+content
        num_res_blocks=4
    )
    x2 = torch.randn(8, 1024)
    t = torch.randint(0, 1000, (8,))
    c = torch.randn(8, 2048)
    out2 = model2(x2, t, c)
    print(f"[eps-only] Output shape: {out2.shape}")

    print("\n--- SimpleMLPAdaLN (VLB / learned-sigma) ---")
    model3 = SimpleMLPAdaLN(
        in_channels=1024,
        model_channels=2048,
        out_channels=2048,     # 2 * in_channels
        z_channels=2048,
        num_res_blocks=4
    )
    out3 = model3(x2, t, c)
    print(f"[VLB] Output shape: {out3.shape}")

    print("\n--- CFG check (VLB) ---")
    # Make even batch for CFG (duplicate-batch trick)
    x_cfg = torch.randn(10, 1024)
    t_cfg = torch.randint(0, 1000, (10,))
    c_cfg = torch.randn(10, 2048)
    out_cfg = model3.forward_with_cfg(x_cfg, t_cfg, c_cfg, cfg_scale=3.0)
    print(f"[CFG VLB] Output shape: {out_cfg.shape}")