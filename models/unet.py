# models/unet.py 
# Content-Style conditional UNet, implemented on top of the "official" LDM UNetModel
# you pasted (same internals: ResBlock, SpatialTransformer, FiLM-like scale_shift).
# This keeps the API compatible with your trainer: forward(x, t, c) and forward_with_cfg(...).

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# --- Use the same official LDM utilities & modules you pasted ---
from .unet_utils import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .unet_utils import SpatialTransformer


# ---------------------------------------------------------------------
# Minimal copies of helper blocks (kept identical to the "official" style)
# ---------------------------------------------------------------------
class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        if self.dims == 3:
            x = nn.functional.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block with FiLM-like conditioning (scale-shift norm) if enabled.
    Matches the logic of the LDM block you pasted.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=True,  # enable FiLM-style conditioning (recommended)
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h); x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


# ---------------------------------------------------------------------
# Official-logic UNetModel with SpatialTransformer cross-attention
# ---------------------------------------------------------------------
class UNetModel(nn.Module):
    """
    A faithful reimplementation of the official LDM UNet (shortened comments).
    Attention is injected via SpatialTransformer at chosen downsample rates.
    """
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 2),
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        num_heads: int = 8,
        num_head_channels: int = -1,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        legacy: bool = True,
    ):
        super().__init__()
        assert not use_spatial_transformer or context_dim is not None, "context_dim required for cross-attn"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Input stage
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(
            conv_nd(dims, in_channels, model_channels, 3, padding=1)
        )])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1  # downsample rate
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=mult * model_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    # Head width logic matches official
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ) if use_spatial_transformer else nn.Identity()
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                        use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True,
                    ) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # Middle (bottleneck)
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                if use_spatial_transformer else nn.Identity(),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
        )

        # Output stage
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(
                    ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult, dims=dims,
                    use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm
                )]
                ch = model_channels * mult
                ds_here = ds
                if ds_here in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim)
                        if use_spatial_transformer else nn.Identity()
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True,
                        ) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, context=None):
        # Standard LDM forward with optional cross-attn context
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)


# ---------------------------------------------------------------------
# Content/Style conditioning: tokens for cross-attention
# ---------------------------------------------------------------------
@dataclass
class CSCondCfg:
    content_dim: int = 1024
    style_dim: int = 1024
    ctx_dim: int = 768
    n_content_tokens: int = 1
    n_style_tokens: int = 1
    use_learned_null: bool = True


class ContentStyleToTokens(nn.Module):
    def __init__(self, cfg: CSCondCfg):
        super().__init__()
        self.cfg = cfg
        self.proj_c = nn.Linear(cfg.content_dim, cfg.n_content_tokens * cfg.ctx_dim)
        self.proj_s = nn.Linear(cfg.style_dim,   cfg.n_style_tokens   * cfg.ctx_dim)
        nn.init.normal_(self.proj_c.weight, std=0.02); nn.init.zeros_(self.proj_c.bias)
        nn.init.normal_(self.proj_s.weight, std=0.02); nn.init.zeros_(self.proj_s.bias)
        self.null_embed = nn.Parameter(
            torch.zeros(1, (cfg.n_content_tokens + cfg.n_style_tokens), cfg.ctx_dim)
        ) if cfg.use_learned_null else None

    def to_tokens(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B = content.size(0)
        c_tok = self.proj_c(content).view(B, self.cfg.n_content_tokens, self.cfg.ctx_dim)
        s_tok = self.proj_s(style).view(B, self.cfg.n_style_tokens,   self.cfg.ctx_dim)
        return torch.cat([c_tok, s_tok], dim=1)

    def null_tokens(self, B: int, device) -> torch.Tensor:
        if self.null_embed is not None:
            return self.null_embed.expand(B, -1, -1).to(device)
        return torch.zeros(B, self.cfg.n_content_tokens + self.cfg.n_style_tokens, self.cfg.ctx_dim, device=device)


# ---------------------------------------------------------------------
# Wrapper that matches your trainer API
# ---------------------------------------------------------------------
class CSUNetDenoiser(nn.Module):
    """
    Wrapper around the official-style UNetModel:
      - operates on (B,4,16,16) latents,
      - conditions via cross-attn on content/style tokens,
      - forward(x,t,c) and forward_with_cfg(x,t,c,w).
    """
    def __init__(
        self,
        sample_size: int = 16,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 256,
        channel_mult: Tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        num_heads: int = 8,
        cs_cfg: Optional[CSCondCfg] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cs_cfg = cs_cfg or CSCondCfg()

        # Attention at all spatial scales for 16x16: ds in {1,2,4} -> sizes 16,8,4
        attention_resolutions = (1, 2, 4)

        self.unet = UNetModel(
            image_size=sample_size,
            in_channels=in_channels,
            model_channels=base_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=0.0,
            channel_mult=channel_mult,
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            num_heads=num_heads,
            num_head_channels=-1,            # use num_heads path
            use_scale_shift_norm=True,       # FiLM-like conditioning
            resblock_updown=False,
            use_spatial_transformer=True,    # enable cross-attn
            transformer_depth=1,
            context_dim=self.cs_cfg.ctx_dim, # width of context tokens
            legacy=True,
        )

        self.cs_tokens = ContentStyleToTokens(self.cs_cfg)

    def _split_c(self, c: torch.Tensor):
        return c[:, :self.cs_cfg.content_dim], c[:, self.cs_cfg.content_dim:]

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        content, style = self._split_c(c)
        ctx = self.cs_tokens.to_tokens(content, style)  # (B, N_ctx, ctx_dim)
        return self.unet(x, timesteps=t, context=ctx)   # eps, shape (B,4,16,16)

    @torch.no_grad()
    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        B = x.size(0)
        content, style = self._split_c(c)
        ctx_cond = self.cs_tokens.to_tokens(content, style)
        ctx_null = self.cs_tokens.null_tokens(B, x.device)
        eps_uncond = self.unet(x, timesteps=t, context=ctx_null)
        eps_cond   = self.unet(x, timesteps=t, context=ctx_cond)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)