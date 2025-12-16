# models/unet.py 

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from .mlp import TimestepEmbedder


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


####### Self-designed part 


@dataclass
class CSCondCfg:
    content_dim: int = 1024
    style_dim: int = 1024
    ctx_dim: int = 1024
    n_content_tokens: int = 1
    n_style_tokens: int = 1
    use_learned_null: bool = True


class ContentStyleToTokens(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.n_content_tokens == 1 and cfg.ctx_dim == cfg.content_dim:
            self.proj_c = nn.Identity()
        else:
            self.proj_c = nn.Linear(cfg.content_dim, cfg.n_content_tokens * cfg.ctx_dim)

        if cfg.n_style_tokens == 1 and cfg.ctx_dim == cfg.style_dim:
            self.proj_s = nn.Identity()
        else:
            self.proj_s = nn.Linear(cfg.style_dim, cfg.n_style_tokens * cfg.ctx_dim)

        self.null_embed = nn.Parameter(torch.zeros(
            1, cfg.n_content_tokens + cfg.n_style_tokens, cfg.ctx_dim
        )) if cfg.use_learned_null else None

    def to_tokens(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        B = content.size(0)
        c_tok = self.proj_c(content).view(B, self.cfg.n_content_tokens, self.cfg.ctx_dim)
        s_tok = self.proj_s(style).view(B, self.cfg.n_style_tokens,   self.cfg.ctx_dim)
        return torch.cat([c_tok, s_tok], dim=1)

    def null_tokens(self, B: int, device) -> torch.Tensor:
        if self.null_embed is not None:
            return self.null_embed.expand(B, -1, -1).to(device)
        return torch.zeros(B, self.cfg.n_content_tokens + self.cfg.n_style_tokens, self.cfg.ctx_dim, device=device)


class PatchTokenAdapter(nn.Module):
    """
    Shared helper that converts spatial content/style maps into patch tokens and
    provides null-context embeddings for classifier-free guidance.
    """
    def __init__(
        self,
        sample_size: int,
        content_channels: int,
        style_channels: int,
        patch_size: int,
        ctx_dim: int,
        use_learned_null: bool,
    ):
        super().__init__()
        assert sample_size % patch_size == 0, "sample_size must be divisible by patch_size"
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.content_channels = content_channels
        self.style_channels = style_channels
        self.ctx_dim = ctx_dim

        tokens_per_field = (sample_size // patch_size) ** 2
        self.total_tokens = tokens_per_field * 2

        self.content_proj = nn.Linear(content_channels * patch_size * patch_size, ctx_dim)
        self.style_proj = nn.Linear(style_channels * patch_size * patch_size, ctx_dim)

        if use_learned_null:
            self.null_tokens = nn.Parameter(torch.zeros(1, self.total_tokens, ctx_dim))
        else:
            self.register_buffer(
                "null_tokens", torch.zeros(1, self.total_tokens, ctx_dim), persistent=False
            )

    def _patchify(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        p = self.patch_size
        feat = feat.view(B, C, H // p, p, W // p, p)
        feat = feat.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C * p * p)
        return feat

    def build_tokens(self, content_map: torch.Tensor, style_map: torch.Tensor) -> torch.Tensor:
        content_tokens = self.content_proj(self._patchify(content_map))
        style_tokens = self.style_proj(self._patchify(style_map))
        return torch.cat([content_tokens, style_tokens], dim=1)

    def get_null_tokens(self, batch_size: int, device) -> torch.Tensor:
        return self.null_tokens.expand(batch_size, -1, -1).to(device)


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
        attention_resolutions: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cs_cfg = cs_cfg or CSCondCfg()

        # If not specified, only keep the lowest-resolution attention to cut compute.
        if attention_resolutions is None:
            # Downsample rate after the deepest block is 2 ** (len(channel_mult) - 1)
            lowest_ds = 2 ** max(len(channel_mult) - 1, 0)
            attention_resolutions = (lowest_ds,)

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

    @torch.no_grad()
    def debug_shapes(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> None:
        content, style = self._split_c(c)
        ctx = self.cs_tokens.to_tokens(content, style)
        print("[SANITY] CSUNetDenoiser")
        print("  input:", tuple(x.shape))
        print("  content split:", tuple(content.shape))
        print("  style split:", tuple(style.shape))
        print("  ctx tokens:", tuple(ctx.shape))

        unet = self.unet
        hs = []
        t_emb = timestep_embedding(t, unet.model_channels, repeat_only=False)
        emb = unet.time_embed(t_emb)
        print("  time embedding:", tuple(emb.shape))

        h = x
        for idx, module in enumerate(unet.input_blocks):
            h = module(h, emb, ctx)
            hs.append(h)
            print(f"  input_block[{idx}]:", tuple(h.shape))

        h = unet.middle_block(h, emb, ctx)
        print("  middle_block:", tuple(h.shape))

        for idx, module in enumerate(unet.output_blocks):
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            print(f"  output_block[{idx}] concat:", tuple(h.shape))
            h = module(h, emb, ctx)
            print(f"  output_block[{idx}] out:", tuple(h.shape))

        out = unet.out(h)
        print("  final output:", tuple(out.shape))



class CSPatchUNetDenoiser(nn.Module):
    """
    UNet wrapper that consumes patch tokens produced from spatial content/style maps.
    The interface expects the trainer to pass pre-built tokens, but exposes helpers to
    convert 16x16 feature maps into (B, N_tokens, ctx_dim) sequences.
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
        ctx_dim: int = 128,
        content_channels: int = 4,
        style_channels: int = 4,
        patch_size: int = 2,
        use_learned_null: bool = True,
    ):
        super().__init__()
        assert sample_size % patch_size == 0, "sample_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.ctx_dim = ctx_dim
        self.content_channels = content_channels
        self.style_channels = style_channels

        self.n_content_tokens = (sample_size // patch_size) ** 2
        self.n_style_tokens = self.n_content_tokens
        self.total_tokens = self.n_content_tokens + self.n_style_tokens

        self.unet = UNetModel(
            image_size=sample_size,
            in_channels=in_channels,
            model_channels=base_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=(1, 2, 4),
            dropout=0.0,
            channel_mult=channel_mult,
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            num_heads=num_heads,
            num_head_channels=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=ctx_dim,
            legacy=True,
        )

        token_dim_c = content_channels * patch_size * patch_size
        token_dim_s = style_channels * patch_size * patch_size
        self.content_proj = nn.Linear(token_dim_c, ctx_dim)
        self.style_proj = nn.Linear(token_dim_s, ctx_dim)

        if use_learned_null:
            self.null_tokens = nn.Parameter(torch.zeros(1, self.total_tokens, ctx_dim))
        else:
            self.register_buffer("null_tokens", torch.zeros(1, self.total_tokens, ctx_dim), persistent=False)

    def _patchify(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        p = self.patch_size
        feat = feat.view(B, C, H // p, p, W // p, p)
        feat = feat.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C * p * p)
        return feat

    def build_tokens(self, content_map: torch.Tensor, style_map: torch.Tensor) -> torch.Tensor:
        content_tokens = self.content_proj(self._patchify(content_map))
        style_tokens = self.style_proj(self._patchify(style_map))
        return torch.cat([content_tokens, style_tokens], dim=1)

    def get_null_tokens(self, batch_size: int, device) -> torch.Tensor:
        return self.null_tokens.expand(batch_size, -1, -1).to(device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.unet(x, timesteps=t, context=tokens)

    @torch.no_grad()
    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        B = x.size(0)
        tokens_null = self.get_null_tokens(B, x.device)
        eps_uncond = self.unet(x, timesteps=t, context=tokens_null)
        eps_cond = self.unet(x, timesteps=t, context=tokens)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def debug_shapes(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> None:
        print("[SANITY] CSPatchUNetDenoiser")
        print("  input:", tuple(x.shape))
        print("  tokens:", tuple(tokens.shape))
        unet = self.unet
        hs = []
        t_emb = timestep_embedding(t, unet.model_channels, repeat_only=False)
        emb = unet.time_embed(t_emb)
        print("  time embedding:", tuple(emb.shape))

        h = x
        for idx, module in enumerate(unet.input_blocks):
            h = module(h, emb, tokens)
            hs.append(h)
            print(f"  input_block[{idx}]:", tuple(h.shape))

        h = unet.middle_block(h, emb, tokens)
        print("  middle_block:", tuple(h.shape))

        for idx, module in enumerate(unet.output_blocks):
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            print(f"  output_block[{idx}] concat:", tuple(h.shape))
            h = module(h, emb, tokens)
            print(f"  output_block[{idx}] out:", tuple(h.shape))

        out = unet.out(h)
        print("  final output:", tuple(out.shape))


def _make_group_norm(channels: int) -> nn.GroupNorm:
    num_groups = min(32, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = _make_group_norm(in_channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = _make_group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        temb = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + temb
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SimplePatchUNetDenoiser(nn.Module):
    """
    Lightweight UNet with a single down/up pair and cross-attention only in the bottleneck.
    Designed for small 16x16 latents.
    """
    def __init__(
        self,
        sample_size: int = 16,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 160,
        channel_mult: Tuple[int, ...] = (1, 2),
        num_heads: int = 4,
        ctx_dim: int = 96,
        content_channels: int = 4,
        style_channels: int = 4,
        patch_size: int = 4,
        use_learned_null: bool = True,
    ):
        super().__init__()
        assert len(channel_mult) >= 2, "channel_mult must provide at least two entries"

        mid_mult = channel_mult[-1]
        mid_channels = base_channels * mid_mult
        assert mid_channels % num_heads == 0, "mid_channels must be divisible by num_heads"

        time_dim = base_channels * 4
        self.time_embed = TimestepEmbedder(time_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.in_block = TimeResBlock(base_channels, base_channels, time_dim)

        self.downsample = nn.Conv2d(base_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.down_block = TimeResBlock(mid_channels, mid_channels, time_dim)
        self.mid_block1 = TimeResBlock(mid_channels, mid_channels, time_dim)
        self.cross_attn = SpatialTransformer(
            mid_channels,
            num_heads,
            mid_channels // num_heads,
            depth=1,
            context_dim=ctx_dim,
        )
        self.mid_block2 = TimeResBlock(mid_channels, mid_channels, time_dim)

        self.up_proj = nn.Conv2d(mid_channels, base_channels, kernel_size=3, padding=1)
        self.up_block = TimeResBlock(base_channels + base_channels, base_channels, time_dim)
        self.final_norm = _make_group_norm(base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        self.token_adapter = PatchTokenAdapter(
            sample_size=sample_size,
            content_channels=content_channels,
            style_channels=style_channels,
            patch_size=patch_size,
            ctx_dim=ctx_dim,
            use_learned_null=use_learned_null,
        )

    def build_tokens(self, content_map: torch.Tensor, style_map: torch.Tensor) -> torch.Tensor:
        return self.token_adapter.build_tokens(content_map, style_map)

    def get_null_tokens(self, batch_size: int, device) -> torch.Tensor:
        return self.token_adapter.get_null_tokens(batch_size, device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        h = self.input_proj(x)
        h = self.in_block(h, t_emb)
        skip = h

        h = self.downsample(h)
        h = self.down_block(h, t_emb)
        h = self.mid_block1(h, t_emb)
        h = self.cross_attn(h, context=tokens)
        h = self.mid_block2(h, t_emb)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up_proj(h)
        h = torch.cat([h, skip], dim=1)
        h = self.up_block(h, t_emb)

        h = self.final_conv(self.final_act(self.final_norm(h)))
        return h

    @torch.no_grad()
    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        B = x.size(0)
        tokens_null = self.get_null_tokens(B, x.device)
        eps_uncond = self.forward(x, t, tokens_null)
        eps_cond = self.forward(x, t, tokens)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    @torch.no_grad()
    def debug_shapes(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> None:
        print("[SANITY] SimplePatchUNetDenoiser")
        print("  input:", tuple(x.shape))
        print("  tokens:", tuple(tokens.shape))
        t_emb = self.time_embed(t)
        print("  time_embed:", tuple(t_emb.shape))
        t_emb = self.time_mlp(t_emb)
        h = self.input_proj(x)
        print("  after input_proj:", tuple(h.shape))
        h = self.in_block(h, t_emb)
        print("  after in_block:", tuple(h.shape))
        skip = h
        h = self.downsample(h)
        print("  after downsample:", tuple(h.shape))
        h = self.down_block(h, t_emb)
        print("  after down_block:", tuple(h.shape))
        h = self.mid_block1(h, t_emb)
        print("  mid_block1:", tuple(h.shape))
        h = self.cross_attn(h, context=tokens)
        print("  after cross_attn:", tuple(h.shape))
        h = self.mid_block2(h, t_emb)
        print("  mid_block2:", tuple(h.shape))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        print("  after upsample:", tuple(h.shape))
        h = self.up_proj(h)
        print("  after up_proj:", tuple(h.shape))
        h = torch.cat([h, skip], dim=1)
        print("  concat skip:", tuple(h.shape))
        h = self.up_block(h, t_emb)
        print("  after up_block:", tuple(h.shape))
        h = self.final_conv(self.final_act(self.final_norm(h)))
        print("  output:", tuple(h.shape))


class LightCSUNetDenoiser(nn.Module):
    """
    Lightweight single-down/up UNet that uses content/style tokens (not patch tokens).
    Designed to mimic SimplePatchUNet speed while staying compatible with CSCondCfg.
    """
    def __init__(
        self,
        sample_size: int = 16,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 160,
        num_heads: int = 4,
        cs_cfg: Optional[CSCondCfg] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cs_cfg = cs_cfg or CSCondCfg()

        mid_channels = base_channels * 2
        assert mid_channels % num_heads == 0, "mid_channels must be divisible by num_heads"

        time_dim = base_channels * 4
        self.time_embed = TimestepEmbedder(time_dim)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.cs_tokens = ContentStyleToTokens(self.cs_cfg)

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.in_block = TimeResBlock(base_channels, base_channels, time_dim)

        self.downsample = nn.Conv2d(base_channels, mid_channels, kernel_size=3, stride=2, padding=1)
        self.down_block = TimeResBlock(mid_channels, mid_channels, time_dim)
        self.mid_block1 = TimeResBlock(mid_channels, mid_channels, time_dim)
        self.cross_attn = SpatialTransformer(
            mid_channels,
            num_heads,
            mid_channels // num_heads,
            depth=1,
            context_dim=self.cs_cfg.ctx_dim,
        )
        self.mid_block2 = TimeResBlock(mid_channels, mid_channels, time_dim)

        self.up_proj = nn.Conv2d(mid_channels, base_channels, kernel_size=3, padding=1)
        self.up_block = TimeResBlock(base_channels + base_channels, base_channels, time_dim)
        self.final_norm = _make_group_norm(base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def _split_c(self, c: torch.Tensor):
        return c[:, :self.cs_cfg.content_dim], c[:, self.cs_cfg.content_dim:]

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        content, style = self._split_c(c)
        tokens = self.cs_tokens.to_tokens(content, style)
        return self._forward_with_tokens(x, t, tokens)

    def _forward_with_tokens(self, x: torch.Tensor, t: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        h = self.input_proj(x)
        h = self.in_block(h, t_emb)
        skip = h

        h = self.downsample(h)
        h = self.down_block(h, t_emb)
        h = self.mid_block1(h, t_emb)
        h = self.cross_attn(h, context=tokens)
        h = self.mid_block2(h, t_emb)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up_proj(h)
        h = torch.cat([h, skip], dim=1)
        h = self.up_block(h, t_emb)

        return self.final_conv(self.final_act(self.final_norm(h)))

    @torch.no_grad()
    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        B = x.size(0)
        content, style = self._split_c(c)
        tokens = self.cs_tokens.to_tokens(content, style)
        tokens_null = self.cs_tokens.null_tokens(B, x.device)
        eps_uncond = self._forward_with_tokens(x, t, tokens_null)
        eps_cond = self._forward_with_tokens(x, t, tokens)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


