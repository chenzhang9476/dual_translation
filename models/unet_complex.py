import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================
# Sinusoidal Time Embedding
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ============================================================
# Basic Blocks
# ============================================================
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8):
        super().__init__()
        self.block1 = Block(in_ch, out_ch, groups=groups)
        self.block2 = Block(out_ch, out_ch, groups=groups)
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))

    def forward(self, x, t_emb=None):
        h = self.block1(x)
        if self.time_mlp is not None and t_emb is not None:
            t = self.time_mlp(t_emb)[:, :, None, None]
            scale, shift = t.chunk(2, dim=1)
            h = h * (scale + 1) + shift
        h = self.block2(h)
        return h + self.res(x)


# ============================================================
# Window utils
# ============================================================
def window_partition(x: torch.Tensor, ws: int):
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, ws * ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int, B: int):
    C = windows.shape[-1]
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, C, H, W)


# ============================================================
# Window Cross Attention (FIXED: cond_proj is registered in __init__)
# ============================================================
class WindowCrossAttention(nn.Module):
    """
    关键修复点：
    - cond_proj 不再在 forward 里动态创建
    - 在 __init__ 里确定 cond_channels，并注册 cond_proj (Conv 或 Identity)
    """
    def __init__(self, dim, cond_channels, window_size=4, num_heads=2, mlp_ratio=2):
        super().__init__()
        self.dim = int(dim)
        self.cond_channels = int(cond_channels)
        self.window_size = int(window_size)

        self.norm_q = nn.LayerNorm(self.dim)
        self.norm_kv = nn.LayerNorm(self.dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(self.dim * mlp_ratio, self.dim),
        )

        # ✅ 提前注册：保证 state_dict 稳定
        if self.cond_channels == self.dim:
            self.cond_proj = nn.Identity()
        else:
            self.cond_proj = nn.Conv2d(self.cond_channels, self.dim, kernel_size=1, bias=False)

    def forward(self, x, cond):
        """
        x    : [B, dim, H, W]
        cond : [B, cond_channels, Hc, Wc]  (Hc/Wc 可以不同，会插值到 H/W)
        """
        B, C, H, W = x.shape
        ws = self.window_size
        assert C == self.dim, f"x channel {C} != dim {self.dim}"

        if cond.shape[2:] != (H, W):
            cond = F.interpolate(cond, size=(H, W), mode="bilinear", align_corners=False)

        # ✅ 更安全：强制检查 cond 通道是否符合初始化时的 cond_channels
        if cond.shape[1] != self.cond_channels:
            raise RuntimeError(
                f"[WindowCrossAttention] cond channels mismatch: got {cond.shape[1]}, "
                f"expected {self.cond_channels}. "
                f"请确认 cond_feats 的通道数是否和构建时一致。"
            )

        cond = self.cond_proj(cond)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            cond = F.pad(cond, (0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[2:]
        x_win = window_partition(x, ws)
        c_win = window_partition(cond, ws)

        q = self.norm_q(x_win)
        kv = self.norm_kv(c_win)
        attn_out, _ = self.attn(q, kv, kv)

        x_win = x_win + attn_out
        x_win = x_win + self.mlp(self.norm2(x_win))

        x = window_reverse(x_win, ws, Hp, Wp, B)
        return x[:, :, :H, :W]


# ============================================================
# MRI Encoder (MR only)
# ============================================================
class MRIEncoder(nn.Module):
    def __init__(self, base=8, levels=2, in_channels=1):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.GroupNorm(max(1, min(4, base)), base),
            nn.SiLU(),
        )
        self.levels = nn.ModuleList()
        ch = base
        for _ in range(levels):
            self.levels.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                    nn.GroupNorm(max(1, min(4, ch * 2)), ch * 2),
                    nn.SiLU(),
                )
            )
            ch *= 2
        self.bottleneck = ResidualBlock(ch, ch)

        self.out_channels_list = [base]  # f0
        tmp = base
        for _ in range(levels):
            tmp *= 2
            self.out_channels_list.append(tmp)  # f1,f2,...
        self.out_channels_list.append(tmp)  # bottleneck 通道

    def forward(self, x):
        feats = []
        h = self.initial(x)   # f0
        feats.append(h)
        for l in self.levels:
            h = l(h)          # f1,f2,f3...
            feats.append(h)
        feats.append(self.bottleneck(h))  # bottleneck
        return feats


# ============================================================
# High Frequency UNet (确定性版本，只预测x0)
# ============================================================
class HighFreqUNet(nn.Module):
    """
    确定性高频UNet：只预测x0
    Output: [B, 3, H, W]

    关键修复：
    - WindowCrossAttention 传入固定的 cond_channels，cond_proj 在 __init__ 注册
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=16,
        time_emb_dim=64,
        window_sizes=(4, 4, 4),
        num_heads=2,
        use_checkpoint=True,
        cond_in_channels=2,   # 你原来 MRIEncoder in_channels=2，这里显式化
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.out_channels = out_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Input
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # MRI Encoder (3 levels now)
        cond_base = max(4, base_channels // 2)   # 例如 base_channels=64 -> cond_base=32
        self.cond_encoder = MRIEncoder(
            base=cond_base,
            levels=3,
            in_channels=cond_in_channels,
        )

        # cond_feats 通道：
        # cond_feats[0] = cond_base
        # cond_feats[-1] = cond_base * 2^3 = cond_base*8
        cond_ch_f0 = cond_base
        cond_ch_bottleneck = cond_base * (2 ** 3)

        # Encoder
        self.enc1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.attn1 = WindowCrossAttention(
            dim=base_channels,
            cond_channels=cond_ch_f0,
            window_size=window_sizes[0],
            num_heads=num_heads
        )

        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1)
        self.enc2 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.enc3 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1)

        # Bottleneck
        self.mid1 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.mid_attn1 = WindowCrossAttention(
            dim=base_channels * 8,
            cond_channels=cond_ch_bottleneck,
            window_size=window_sizes[2],
            num_heads=num_heads
        )

        self.mid2 = ResidualBlock(base_channels * 8, base_channels * 8, time_emb_dim)
        self.mid_attn2 = WindowCrossAttention(
            dim=base_channels * 8,
            cond_channels=cond_ch_bottleneck,
            window_size=window_sizes[2],
            num_heads=num_heads
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1)
        self.dec3 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.dec2 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.dec1 = ResidualBlock(base_channels, base_channels, time_emb_dim)

        # Output
        self.final = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def _ckpt(self, module, *args):
        if self.use_checkpoint and self.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x, t, cond_mr):
        """
        x       : [B,3,H,W] noisy HF
        t       : [B]
        cond_mr : [B,cond_in_channels,H,W]  (注意：这里应与 cond_in_channels 对齐，默认 2)
        """
        t_emb = self.time_mlp(t)

        cond_feats = self.cond_encoder(cond_mr)  # [f0, f1, f2, f3, bottleneck]

        h = self.init_conv(x)

        # Encoder
        h = self._ckpt(self.enc1, h, t_emb)
        h = self._ckpt(self.attn1, h, cond_feats[0])

        h = self.down1(h)
        h = self._ckpt(self.enc2, h, t_emb)

        h = self.down2(h)
        h = self._ckpt(self.enc3, h, t_emb)

        h = self.down3(h)

        # Bottleneck
        h = self._ckpt(self.mid1, h, t_emb)
        h = self._ckpt(self.mid_attn1, h, cond_feats[-1])

        h = self._ckpt(self.mid2, h, t_emb)
        h = self._ckpt(self.mid_attn2, h, cond_feats[-1])

        # Decoder
        h = self.up3(h)
        h = self._ckpt(self.dec3, h, t_emb)

        h = self.up2(h)
        h = self._ckpt(self.dec2, h, t_emb)

        h = self.up1(h)
        h = self._ckpt(self.dec1, h, t_emb)

        return self.final(h)