import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List, Optional


# =============================
# Sinusoidal Position Embeddings
# =============================
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timesteps."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# =============================
# Basic Conv Block
# =============================
class Block(nn.Module):
    """Conv + GroupNorm + SiLU block."""
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        g = min(groups, out_channels)
        while out_channels % g != 0 and g > 1:
            g -= 1
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(g, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# =============================
# Residual Block with time embedding
# =============================
class ResidualBlock(nn.Module):
    """Residual block with optional time conditioning."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        groups: int = 8,
    ):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, groups)
        self.block2 = Block(out_channels, out_channels, groups)
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_channels * 2)
            )

    def forward(
        self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.block1(x)
        if t_emb is not None and self.time_mlp is not None:
            t = self.time_mlp(t_emb)[:, :, None, None]
            scale, shift = t.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        h = self.block2(h)
        return h + self.res_conv(x)


# =============================
# Window Partition / Reverse
# =============================
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    """Reverse window partition."""
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, C, H, W)


# =============================
# Window Cross-Attention (with condition)
# =============================
class WindowCrossAttention(nn.Module):
    """Window-based cross-attention between query (x) and key/value (cond)."""
    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        num_heads: int = 4,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        assert dim % num_heads == 0

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size

        # Interpolate cond to match x size if needed
        if cond.shape[2:] != (H, W):
            cond = F.interpolate(cond, size=(H, W), mode="bilinear", align_corners=False)

        # Pad to make divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            cond = F.pad(cond, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[2], x.shape[3]

        x_win = window_partition(x, ws)
        cond_win = window_partition(cond, ws)

        q = self.norm_q(x_win)
        kv = self.norm_kv(cond_win)
        attn_out, _ = self.attn(q, kv, kv)

        x_win = x_win + attn_out
        x_win = x_win + self.mlp(self.norm(x_win))

        x = window_reverse(x_win, ws, Hp, Wp, B)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]

        return x


# =============================
# MRI Encoder (for MR only, but now supports multi-channel cond)
# =============================
class CondEncoder(nn.Module):
    """Encoder for conditioning input (mr + optional mask)."""
    def __init__(
        self,
        in_channels: int = 1,  # mr (默认1通道)
        base_channels: int = 32,
        num_levels: int = 3,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels),
        )
        self.levels = nn.ModuleList()
        ch = base_channels
        for _ in range(num_levels):
            level = nn.Sequential(
                nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                ResidualBlock(ch * 2, ch * 2),
                ResidualBlock(ch * 2, ch * 2),
            )
            self.levels.append(level)
            ch *= 2
        self.bottleneck = ResidualBlock(ch, ch)

    def forward(self, cond: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        x = self.initial(cond)
        feats.append(x)
        cur = x
        for level in self.levels:
            cur = level(cur)
            feats.append(cur)
        feats.append(self.bottleneck(cur))
        return feats


# =============================
# LowFreqUNet (eps-pred)
# =============================
class LowFreqUNet(nn.Module):
    """
    U-Net for low-frequency epsilon prediction.
    Input:
        x:    (B, 1, H, W)     noisy image
        t:    (B,)             timestep
        cond: (B, 2, H, W)     [mr, mask]  # 必须是2通道
    Output:
        predicted noise (B, 1, H, W)
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        has_attention: Tuple[bool, ...] = (True, True, True),
        num_heads: int = 4,
        window_sizes: Tuple[int, ...] = (8, 8, 8),
        mid_window_size: int = 4,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.num_levels = len(has_attention)
        self.use_checkpoint = use_checkpoint

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial conv for noisy image
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.cond_encoder = CondEncoder(
            in_channels=2, base_channels=base_channels, num_levels=self.num_levels
        )

        # Down blocks
        ch = base_channels
        self.down_res_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(self.num_levels):
            blocks = nn.ModuleList(
                [ResidualBlock(ch, ch, time_emb_dim) for _ in range(num_res_blocks)]
            )
            self.down_res_blocks.append(blocks)
            self.down_attn.append(
                WindowCrossAttention(
                    ch, window_size=window_sizes[i], num_heads=num_heads
                )
                if has_attention[i]
                else nn.Identity()
            )
            self.down_samples.append(
                nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1)
            )
            ch *= 2

        # Middle block
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim)
        self.mid_attn = WindowCrossAttention(
            ch, window_size=mid_window_size, num_heads=num_heads
        )
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim)

        # Up blocks
        self.up_res_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            out_ch = base_channels * (2 ** level)
            blocks = nn.ModuleList()
            blocks.append(ResidualBlock(out_ch * 2, out_ch, time_emb_dim))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim))
            self.up_res_blocks.append(blocks)
            self.up_attn.append(
                WindowCrossAttention(
                    out_ch, window_size=window_sizes[level], num_heads=num_heads
                )
                if has_attention[level]
                else nn.Identity()
            )
            self.up_samples.append(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            ch = out_ch

        # Final conv
        self.final_conv = nn.Sequential(
            Block(ch, ch),
            nn.Conv2d(ch, out_channels, 1),
        )

    def _checkpoint(self, module, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        cond must be (B, 1, H, W) -> [mr]
        """

        t_emb = self.time_mlp(t)
        cond_feats = self.cond_encoder(cond)  

        h = self.init_conv(x)
        skips = []

        # Down path
        for level in range(self.num_levels):
            for block in self.down_res_blocks[level]:
                h = self._checkpoint(block, h, t_emb)
            attn_mod = self.down_attn[level]
            if isinstance(attn_mod, WindowCrossAttention):
                h = self._checkpoint(attn_mod, h, cond_feats[level])
            else:
                h = attn_mod(h)
            skips.append(h)
            h = self.down_samples[level](h)

        # Middle
        h = self._checkpoint(self.mid_block1, h, t_emb)
        h = self._checkpoint(self.mid_attn, h, cond_feats[-1])
        h = self._checkpoint(self.mid_block2, h, t_emb)

        # Up path
        cond_feats_up = cond_feats[self.num_levels - 1 :: -1]  # reversed
        for stage in range(self.num_levels):
            h = self.up_samples[stage](h)
            skip = skips.pop()
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            for block in self.up_res_blocks[stage]:
                h = self._checkpoint(block, h, t_emb)
            attn_mod = self.up_attn[stage]
            if isinstance(attn_mod, WindowCrossAttention):
                h = self._checkpoint(attn_mod, h, cond_feats_up[stage])
            else:
                h = attn_mod(h)

        return self.final_conv(h)