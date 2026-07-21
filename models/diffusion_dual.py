import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import matplotlib.pyplot as plt
import numpy as np
import os

class LowFreqDiffusion(nn.Module):
    """
    Low-frequency diffusion (eps-pred)
    """
    def __init__(self, unet, timesteps=120, beta_start=1e-4, beta_end=5e-3):
        super().__init__()
        self.unet = unet
        self.timesteps = int(timesteps)

        beta = torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_cumprod = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod",
            torch.sqrt(1.0 - alpha_cumprod)
        )

    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        return (
            self.sqrt_alpha_cumprod[t].view(B, 1, 1, 1) * x0
            + self.sqrt_one_minus_alpha_cumprod[t].view(B, 1, 1, 1) * noise
        )

    def forward(self, x0, mr, mask=None, t=None):
        """
        x0: 已经是 mask-aware normalized 的 low (mask=0 -> 0)
        """
        B = x0.shape[0]
        device = x0.device

        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=device)

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        #cond = mr * mask if mask is not None else mr
        mask = (mask > 0.5).float()
        mask = mask[:, :1, ...]
        cond = torch.cat([mr, mask], dim=1)   # [B,2,H,W]

        #change here
        xt = xt * mask
        
        pred_eps = self.unet(xt, t, cond)

        if mask is not None:
            mask = (mask > 0.5).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = mask[:, :1, ...]
            loss = F.mse_loss(pred_eps * mask, noise * mask)

        #loss = F.l1_loss(pred_eps, noise) + 0.1 * loss_fg

        return loss, pred_eps, noise

    @torch.no_grad()
    def sample(self, mr, mask=None, progress_bar=True):
        device = mr.device
        B, _, H, W = mr.shape

        # ---------- normalize mask ----------
        if mask is None:
            mask = torch.ones((B, 1, H, W), device=device)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float()
            mask = mask[:, :1, ...]

        cond = torch.cat([mr, mask], dim=1)

        # ---------- init noise ----------
        x = torch.randn((B, 1, H, W), device=device)

        it = range(self.timesteps - 1, -1, -1)
        if progress_bar:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="LowFreq DDPM", total=self.timesteps)
            except Exception:
                pass

        for t in it:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            eps = self.unet(x, t_batch, cond)

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps
            )

            if t > 0:
                beta_t = self.beta[t]
                alpha_bar_prev = self.alpha_cumprod[t - 1]
                sigma = torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
                )
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean
            # ---------- enforce ROI ----------
            x = x * mask
        return x


# ============================================================
# High-Frequency Diffusion (修改为x0-pred版本)
class HighFreqDiffusion(nn.Module):
    """
    High-frequency diffusion (x0-pred, with uncertainty via Monte Carlo sampling)
    - 修改：从预测eps改为预测x0（原始），损失为MSE(pred_x0, x0) + 可选梯度损失。
    - 采样：使用x0-pred的DDPM/DDIM逆过程。
    """
    def __init__(
        self,
        unet,
        timesteps=50,
        schedule="cosine",
        device="cuda",
    ):
        super().__init__()
        self.unet = unet
        self.timesteps = int(timesteps)
        self.device = device
        betas = self._get_betas(schedule, self.timesteps).to(torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        alphas_cumprod_prev = torch.cat(
            [alphas_cumprod[:1], alphas_cumprod[:-1]], dim=0
        )
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
    
    def _get_betas(self, schedule: str, timesteps: int) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(1e-4, 2e-2, timesteps)
        if schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 1e-4, 0.9999)
        raise NotImplementedError(schedule)
    
    def q_sample(self, x0, t, noise):
        B = x0.shape[0]
        return (
            self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1) * x0
            + self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1) * noise
        )
    
    def gradient_loss(self, pred_x0, gt_x0, mask):
        # 修改：使用从eps推导的pred_x0计算梯度损失
        gx_pred = pred_x0[:, :, :, 1:] - pred_x0[:, :, :, :-1]
        gx_gt = gt_x0[:, :, :, 1:] - gt_x0[:, :, :, :-1]
        mx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        gy_pred = pred_x0[:, :, 1:, :] - pred_x0[:, :, :-1, :]
        gy_gt = gt_x0[:, :, 1:, :] - gt_x0[:, :, :-1, :]
        my = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        loss_gx = (torch.abs(gx_pred - gx_gt) * mx).mean()
        loss_gy = (torch.abs(gy_pred - gy_gt) * my).mean()
        return loss_gx + loss_gy
    
    def forward(self, x0, mr, mask=None, t=None):
        B = x0.shape[0]
        device = x0.device
        if t is None:
            t = torch.randint(0, self.timesteps, (B,), device=device)
        # ---------- Make sure mask exists and is [B,1,H,W] ----------
        if mask is None:
            mask = torch.ones((B, 1, x0.shape[2], x0.shape[3]), device=device)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float()
        mask_rep = mask.repeat(1, x0.shape[1], 1, 1)  # [B,3,H,W]
        cond = torch.cat([mr, mask], dim=1)
        # ---------- Noise ----------
        noise = torch.randn_like(x0) * mask_rep
        xt = self.q_sample(x0, t, noise)
        # ---------- Predict x0 ----------
        pred_x0 = self.unet(xt, t, cond)
        # ---------- Uncertainty proxy ----------
        with torch.no_grad():
            uncertainty = (pred_x0 - x0).abs()
        # ---------- Weight (stabilized) ----------
        weight = 1.0 / (uncertainty + 1e-3)
        weight = torch.clamp(weight, max=10.0)  # 防止爆炸
        # ---------- Weighted MSE ----------
        denom = mask_rep.sum().clamp_min(1.0)
        loss_mse = (weight * (pred_x0 - x0).pow(2) * mask_rep).sum() / denom
        # ---------- Gradient loss ----------
        loss_grad = self.gradient_loss(pred_x0, x0, mask_rep)
        # ---------- Total ----------
        loss = loss_mse + 0.1 * loss_grad
        return loss, pred_x0, x0
    
    @torch.no_grad()
    def sample(
        self,
        mr,
        mask=None,
        progress_bar=True,
        ddim_eta=0.0,
        generator=None,
    ):
        device = mr.device
        B, _, H, W = mr.shape
        # ---------- normalize mask to [B,1,H,W] ----------
        if mask is None:
            mask = torch.ones((B, 1, H, W), device=device)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float()
        cond = torch.cat([mr, mask], dim=1)
        # ---------- init noise (ROI only) ----------
        if generator is not None:
            with torch.random.fork_rng():
                torch.manual_seed(int(generator.initial_seed()))
                x_t = torch.randn((B, 3, H, W), device=device)  # 3 channels
        else:
            x_t = torch.randn((B, 3, H, W), device=device)  # 3 channels
        x_t = x_t * mask.repeat(1, 3, 1, 1)  # Repeat mask
        it = range(self.timesteps - 1, -1, -1)
        if progress_bar:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="HighFreq DDIM (x0-pred)", total=self.timesteps)
            except Exception:
                pass
        for t in it:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            # 修改：预测x0
            pred_x0 = self.unet(x_t, t_batch, cond)
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod_prev[t]
            sigma_t = ddim_eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            direction = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * (
                (x_t - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1 - alpha_bar_t)
            )
            mean = torch.sqrt(alpha_bar_prev) * pred_x0 + direction
            if t > 0:
                if generator is not None:
                    with torch.random.fork_rng():
                        torch.manual_seed(int(generator.initial_seed()))
                        z = torch.randn_like(x_t)
                else:
                    z = torch.randn_like(x_t)
                z = z * mask.repeat(1, 3, 1, 1)  # Repeat mask
                x_t = mean + sigma_t * z
            else:
                x_t = mean
            # ---------- enforce ROI ----------
            x_t = x_t * mask.repeat(1, 3, 1, 1)
        return x_t
    
    @torch.no_grad()
    def estimate_uncertainty(
            self,
            mr,
            mask=None,
            n_samples=10,
            progress_bar=False,
            ddim_eta=0.0,
            return_all_samples=False,
    ):
        device = mr.device
        samples = []
        for i in range(n_samples):
            seed = torch.randint(0, 1_000_000, (1,)).item() + i
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            sample_i = self.sample(
                mr=mr,
                mask=mask,
                progress_bar=progress_bar and (i == 0),
                ddim_eta=ddim_eta,
                generator=generator,
            )
            samples.append(sample_i)
        samples_tensor = torch.stack(samples, dim=1)  # [B, n, 3, H, W]
        mean = samples_tensor.mean(dim=1)
        variance = samples_tensor.var(dim=1, unbiased=False)
        logvar = torch.log(variance + 1e-8)
        if return_all_samples:
            return mean, variance, logvar, samples_tensor
        else:
            return mean, variance, logvar


# ============================================================
# DualDiffusion Scheduler with uncertainty analysis
# - ✅ Low normalize: remove DC + std norm, store mean/std
# - ✅ Low schedule gentler by default (constructor defaults updated, but interface unchanged)
# ============================================================
class DualDiffusion(nn.Module):
    def __init__(
        self,
        low_freq_unet,
        high_freq_unet,
        wavelet="haar",
        low_timesteps=120,
        high_timesteps=100,
        beta_start=1e-4,
        beta_end=5e-3,
        low_weight=0.7,
        high_weight=0.3,
        device="cuda",
        # normalization scales (kept for future use if you want)
        high_scale=8.00,
        high_inv_scale=0.125,
        # NEW: EMA stats for low-frequency denormalization
        low_stats_ema_momentum=0.99,
        low_stats_eps=1e-6,
    ):
        super().__init__()
        self.device = device
        self.wavelet = wavelet
        self.low_weight = float(low_weight)
        self.high_weight = float(high_weight)
        self.low_freq_diffusion = LowFreqDiffusion(
            unet=low_freq_unet,
            timesteps=low_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        ).to(device)
        self.high_freq_diffusion = HighFreqDiffusion(
            unet=high_freq_unet,
            timesteps=high_timesteps,
            schedule="cosine",
            device=device
        ).to(device)
        # --------------------------------------------
        # FIX: Use global EMA mean/std to denormalize low
        # shape fixed -> never batch-mismatch
        # --------------------------------------------
        self.low_stats_ema_momentum = float(low_stats_ema_momentum)
        self.low_stats_eps = float(low_stats_eps)
        self.high_scale = float(high_scale)
        self.high_inv_scale = float(high_inv_scale)
        self.register_buffer("low_mean_ema", torch.zeros(1, 1, 1, 1))
        self.register_buffer("low_std_ema", torch.ones(1, 1, 1, 1))
    
    # ---------------- low freq normalization ----------------
    def normalize_low(self, approx, mask, eps=1e-6):
        m = mask.float()
        if m.ndim == 3:
            m = m.unsqueeze(1)
        m = m[:, :1, ...]
        
        denom = m.sum(dim=[2, 3], keepdim=True).clamp_min(1.0)

        mean = (approx * m).sum(dim=[2, 3], keepdim=True) / denom
        var = ((approx - mean) ** 2 * m).sum(dim=[2, 3], keepdim=True) / denom
        std = torch.sqrt(var + eps)

        approx_norm = (approx - mean) / std
        approx_norm = approx_norm * m  # mask 外 = 0

        return approx_norm, mean, std
    
    def _update_low_stats_ema(self, mean, std):
        """
        mean/std: [B,1,1,1] or [B,1,H?,W?] (we expect keepdim [B,1,1,1])
        We update EMA as [1,1,1,1].
        """
        # mean/std are per-sample with keepdim -> average over batch to get global stats
        mean_g = mean.mean(dim=0, keepdim=True).detach()  # [1,1,1,1]
        std_g = std.mean(dim=0, keepdim=True).detach()  # [1,1,1,1]
        m = self.low_stats_ema_momentum
        self.low_mean_ema.mul_(m).add_(mean_g * (1.0 - m))
        self.low_std_ema.mul_(m).add_(std_g * (1.0 - m))
    
    def denormalize_low(self, approx_norm):
        """
        FIXED: use EMA stats, shape [1,1,1,1] so it broadcasts to any batch size.
        """
        std = self.low_std_ema.clamp_min(self.low_stats_eps)
        return approx_norm * std + self.low_mean_ema
    
    # ---------------- high freq normalization ----------------
    def normalize_high(self, x, mask, eps=1e-6):
        # x: [B,3,H,W], mask: [B,1,H,W]
        # ROI outside -> 0
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float()
        x = x * mask.repeat(1, 3, 1, 1)
        return x * self.high_scale
    
    def denormalize_high(self, x_norm):
        return x_norm * self.high_inv_scale
    
    # ---------------- wavelet decomposition ----------------
    def wavelet_dec(self, x, mask):
        """
        x: [B,1,H,W] (ct)
        mask: [B,1,H,W]
        return:
          approx_norm: [B,1,H,W]
          high_norm: [B,3,H,W]
        and update EMA stats for low-frequency denormalization.
        """
        B, _, _, _ = x.shape
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float()
        approx_list, lh_list, hl_list, hh_list = [], [], [], []
        x_cpu = x.detach().cpu()
        for b in range(B):
            approx, (LH, HL, HH) = pywt.swt2(x_cpu[b, 0].numpy(), self.wavelet, level=1)[0]
            approx_list.append(approx)
            lh_list.append(LH)
            hl_list.append(HL)
            hh_list.append(HH)
        approx = torch.from_numpy(np.stack(approx_list)).unsqueeze(1).to(x.device)  # [B,1,H,W]
        high = torch.cat([
            torch.from_numpy(np.stack(lh_list)).unsqueeze(1).to(x.device),
            torch.from_numpy(np.stack(hl_list)).unsqueeze(1).to(x.device),
            torch.from_numpy(np.stack(hh_list)).unsqueeze(1).to(x.device)
        ], dim=1)  # [B,3,H,W]

        #print(f'Wavelet Dec Approx min: {approx.min()}, max: {approx.max()}, mean: {approx.mean():.4f}, std: {approx.std():.4f}')
        #print(f'Wavelet Dec High min: {high.min()}, max: {high.max()}, mean: {high.mean():.4f}, std: {high.std():.4f}')
        approx_norm, low_mean, low_std = self.normalize_low(approx, mask, eps=self.low_stats_eps)
        # update EMA stats ONLY in training mode to avoid messing stats during val sampling
        if self.training:
            self._update_low_stats_ema(low_mean, low_std)
        high_norm = self.normalize_high(high, mask)  # [B,3,H,W]
        #print(f'Wavelet Dec Norm Approx min: {approx_norm.min()}, max: {approx_norm.max()}, mean: {approx_norm.mean():.4f}, std: {approx_norm.std():.4f}')
        #print(f'Wavelet Dec Norm High min: {high_norm.min()}, max: {high_norm.max()}, mean: {high_norm.mean():.4f}, std: {high_norm.std():.4f}')
        #visualize_low_high(approx_norm, high_norm, mask,'./debug', title_prefix='after_norm_')
        return approx_norm, high_norm
    
    def wavelet_rec(self, approx_norm, high_norm, mode="full"):
        """
        approx_norm: [B,1,H,W] (normalized domain)
        high_norm: [B,3,H,W]
        reconstruct ct: [B,1,H,W]
        """
        high = self.denormalize_high(high_norm)  # [B,3,H,W]
        approx = self.denormalize_low(approx_norm)  # [B,1,H,W] using EMA stats
        if mode == "low_lh": #1
            high[:, 1:] = 0.0   # keep LH, zero HL & HH
        elif mode == "low_lh_hl": #2
            high[:, 2] = 0.0    # zero HH only
        elif mode == "full":
            pass
        rec = []
        for b in range(approx.shape[0]):
            coeffs = (
                approx[b, 0].detach().cpu().numpy(),
                (
                    high[b, 0].detach().cpu().numpy(),  # LH
                    high[b, 1].detach().cpu().numpy(),  # HL
                    high[b, 2].detach().cpu().numpy(),  # HH
                )
            )
            rec.append(pywt.iswt2([coeffs], self.wavelet))
        return torch.from_numpy(np.stack(rec)).unsqueeze(1).to(approx.device)
    
    def wavelet_dec_raw(self, x, mask):
        """用于 GT 可视化（raw系数，不做归一化）"""
        B, _, _, _ = x.shape
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float()
        approx_list, lh_list, hl_list, hh_list = [], [], [], []
        x_cpu = x.detach().cpu()
        for b in range(B):
            approx, (LH, HL, HH) = pywt.swt2(x_cpu[b, 0].numpy(), self.wavelet, level=1)[0]
            approx_list.append(approx)
            lh_list.append(LH)
            hl_list.append(HL)
            hh_list.append(HH)
        approx = torch.from_numpy(np.stack(approx_list)).unsqueeze(1).to(x.device)
        high = torch.cat([
            torch.from_numpy(np.stack(lh_list)).unsqueeze(1).to(x.device),
            torch.from_numpy(np.stack(hl_list)).unsqueeze(1).to(x.device),
            torch.from_numpy(np.stack(hh_list)).unsqueeze(1).to(x.device)
        ], dim=1)  # [B,3,H,W]
        return approx, high
    
    # ---------------- training forward ----------------
    def forward(self, x0, mr, mask=None):
        B, _, H, W = x0.shape
        if mask is None:
            mask = torch.ones((B, 1, H, W), device=x0.device)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float()
        #visualize_inputs(x0, mr, mask)
        approx_n, high_n = self.wavelet_dec(x0, mask)
        low_loss, _, _ = self.low_freq_diffusion(approx_n, mr, mask)
        high_loss, _, _ = self.high_freq_diffusion(high_n, mr, mask)
        total_loss = self.low_weight * low_loss + self.high_weight * high_loss
        return {
            "total": total_loss,
            "low": low_loss,
            "high": high_loss,
        }
    
    # ---------------- sampling ----------------
    @torch.no_grad()
    def sample(self, mr, mask=None, progress_bar=False, ddim_eta=0.0):
        B, _, H, W = mr.shape
        device = mr.device
        if mask is None:
            mask = torch.ones((B, 1, H, W), device=device)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float()
        approx_n = self.low_freq_diffusion.sample(
            mr, mask=mask, progress_bar=progress_bar
        )
        high_n = self.high_freq_diffusion.sample(
            mr=mr, mask=mask, progress_bar=progress_bar, ddim_eta=ddim_eta
        )
        rec = self.wavelet_rec(approx_n, high_n)
        if rec.shape[2:] != (H, W):
            rec = F.interpolate(rec, size=(H, W), mode="bilinear", align_corners=False)
        rec = torch.clamp(rec, -1, 1)
        # enforce semantic background
        background = -torch.ones_like(rec)
        rec = mask * rec + (1.0 - mask) * background
        return rec
    
    @torch.no_grad()
    def sample_with_uncertainty_analysis(
        self,
        mr,
        mask=None,
        n_samples=10,
        progress_bar=False,
        ddim_eta=0.0,
        return_all_samples=False,
    ):
        """
        Uncertainty ONLY comes from high-frequency diffusion.
        Low-frequency is deterministic backbone.
        """
        B, _, H, W = mr.shape
        device = mr.device
        if mask is None:
            mask = torch.ones((B, 1, H, W), device=device)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float()
        approx_pred = self.low_freq_diffusion.sample(
            mr, mask=mask, progress_bar=progress_bar
        )
        hf_mean, hf_variance, hf_logvar, hf_samples = (
            self.high_freq_diffusion.estimate_uncertainty(
                mr=mr,
                mask=mask,
                n_samples=n_samples,
                progress_bar=progress_bar,
                ddim_eta=ddim_eta,
                return_all_samples=True,
            )
        )
        ct_samples = []
        for i in range(n_samples):
            rec_i = self.wavelet_rec(approx_pred, hf_samples[:, i])
            if rec_i.shape[2:] != (H, W):
                rec_i = F.interpolate(
                    rec_i, size=(H, W), mode="bilinear", align_corners=False
                )
            rec_i = torch.clamp(rec_i, -1, 1)
            background = -torch.ones_like(rec_i)
            rec_i = mask * rec_i + (1.0 - mask) * background
            ct_samples.append(rec_i)
        ct_samples_tensor = torch.stack(ct_samples, dim=1)  # [B, n, 1, H, W]
        ct_mean = ct_samples_tensor.mean(dim=1)
        ct_variance = ct_samples_tensor.var(dim=1, unbiased=False)
        ct_logvar = torch.log(ct_variance + 1e-8)
        result = {
            "ct_mean": ct_mean,
            "ct_variance": ct_variance,
            "ct_logvar": ct_logvar,
            "hf_mean": hf_mean,
            "hf_variance": hf_variance,
            "hf_logvar": hf_logvar,
            "lf": approx_pred,
        }
        if return_all_samples:
            result["ct_samples"] = ct_samples_tensor
            result["hf_samples"] = hf_samples
        return result
    
    @torch.no_grad()
    def generate_sample_row(
        self,
        mr,
        ct,
        mask,
        device=None,
        n_uncertainty_samples=5,
    ):
        """
        One-row visualization data for debugging / paper figures
        """
        self.eval()
        if device is None:
            device = mr.device
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float()
        # GT decomposition (raw)
        gt_low, gt_high = self.wavelet_dec_raw(ct, mask)
        # deterministic predictions
        approx_pred = self.low_freq_diffusion.sample(
            mr, mask=mask, progress_bar=False
        )
        high_pred = self.high_freq_diffusion.sample(
            mr=mr,
            mask=mask,
            progress_bar=False,
            ddim_eta=0.0,
        )
        # deterministic reconstruction
        rec_det = self.wavelet_rec(approx_pred, high_pred)
        rec_det = rec_det[..., :ct.shape[2], :ct.shape[3]]
        rec_det = torch.clamp(rec_det, -1, 1)
        background = -torch.ones_like(rec_det)
        rec_det = mask * rec_det + (1.0 - mask) * background
        # uncertainty analysis
        uncertainty_result = self.sample_with_uncertainty_analysis(
            mr=mr,
            mask=mask,
            n_samples=n_uncertainty_samples,
            progress_bar=False,
            ddim_eta=0.0,
            return_all_samples=False,
        )
        return {
            "mri": mr.detach().cpu(),
            "ct_gt": ct.detach().cpu(),
            "mask": mask.detach().cpu(),
            "gt_low": gt_low.detach().cpu(),
            "gt_high": gt_high.detach().cpu(),  # [B,3,H,W]
            "pred_low": approx_pred.detach().cpu(),
            "pred_high_det": high_pred.detach().cpu(),  # [B,3,H,W]
            "pred_ct_det": rec_det.detach().cpu(),
            "pred_ct_mean": uncertainty_result["ct_mean"].detach().cpu(),
            "pred_ct_std": torch.sqrt(uncertainty_result["ct_variance"]).detach().cpu(),
            "pred_hf_mean": uncertainty_result["hf_mean"].detach().cpu(),
            "pred_hf_std": torch.sqrt(uncertainty_result["hf_variance"]).detach().cpu(),
        }
    
    @torch.no_grad()
    def sample_with_uncertainty_from_lf(
        self,
        mr,
        approx_pred,
        mask=None,
        n_samples=10,
        progress_bar=False,
        ddim_eta=0.0,
    ):
        """
        FAST inference:
        - Low-frequency approx_pred computed ONCE outside
        - Uncertainty ONLY from high-frequency diffusion
        """
        B, _, H, W = mr.shape
        device = mr.device
        if mask is None:
            mask = torch.ones((B, 1, H, W), device=device)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = (mask > 0.5).float()
        hf_mean, hf_variance, hf_logvar, hf_samples = (
            self.high_freq_diffusion.estimate_uncertainty(
                mr=mr,
                mask=mask,
                n_samples=n_samples,
                progress_bar=progress_bar,
                ddim_eta=ddim_eta,
                return_all_samples=True,
            )
        )
        ct_samples = []
        for i in range(n_samples):
            rec_i = self.wavelet_rec(approx_pred, hf_samples[:, i])
            if rec_i.shape[2:] != (H, W):
                rec_i = F.interpolate(
                    rec_i, size=(H, W),
                    mode="bilinear", align_corners=False
                )
            rec_i = torch.clamp(rec_i, -1, 1)
            background = -torch.ones_like(rec_i)
            rec_i = mask * rec_i + (1.0 - mask) * background
            ct_samples.append(rec_i)
        ct_samples_tensor = torch.stack(ct_samples, dim=1)  # [B,n,1,H,W]
        return {
            "ct_mean": ct_samples_tensor.mean(dim=1),
            "ct_variance": ct_samples_tensor.var(dim=1, unbiased=False),
            "hf_mean": hf_mean,
            "hf_variance": hf_variance,
            "lf": approx_pred,
        }

# ============================================================
# Visualization utility for low/high freq components
# ============================================================
def visualize_low_high(x0_low, x0_high, mask, save_path=None, title_prefix=""):
    assert x0_low.dim() == 4 and x0_high.dim() == 4
    assert x0_low.size(1) == 1, f"Expected low C=1, got {x0_low.size(1)}"
    assert x0_high.size(1) == 3, f"Expected high C=3, got {x0_high.size(1)}"
    # ---- 取第一个样本 ----
    low = x0_low[0, 0].detach().cpu().numpy()
    lh = x0_high[0, 0].detach().cpu().numpy()
    hl = x0_high[0, 1].detach().cpu().numpy()
    hh = x0_high[0, 2].detach().cpu().numpy()
    # ---- 处理 mask ----
    if mask.dim() == 4:
        mask_img = mask[0, 0]
    elif mask.dim() == 3:
        mask_img = mask[0]
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    mask_img = mask_img.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 5, 1)
    plt.imshow(mask_img, cmap='gray')
    plt.title(f"{title_prefix}Mask")
    plt.colorbar(fraction=0.046)
    plt.axis('off')
    plt.subplot(1, 5, 2)
    plt.imshow(low, cmap='gray')
    plt.title(f"{title_prefix}Low (LL)")
    plt.colorbar(fraction=0.046)
    plt.axis('off')
    plt.subplot(1, 5, 3)
    plt.imshow(lh, cmap='gray')
    plt.title(f"{title_prefix}High (LH)")
    plt.colorbar(fraction=0.046)
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.imshow(hl, cmap='gray')
    plt.title(f"{title_prefix}High (HL)")
    plt.colorbar(fraction=0.046)
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.imshow(hh, cmap='gray')
    plt.title(f"{title_prefix}High (HH)")
    plt.colorbar(fraction=0.046)
    plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def visualize_inputs(x0, mr, mask, step=None, save_dir="./debug_inputs"):
    print("Saving visualized inputs...")

    os.makedirs(save_dir, exist_ok=True)

    """
    x0, mr, mask: [B, 1, H, W]
    """
    # 只看 batch 里的第 0 张
    x0_np = x0[0, 0].detach().cpu().numpy()
    mr_np = mr[0, 0].detach().cpu().numpy()
    mask_np = mask[0, 0].detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(x0_np, cmap="gray")
    plt.title("x0")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mr_np, cmap="gray")
    plt.title("mr")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_np, cmap="gray")
    plt.title("mask")
    plt.axis("off")

    if step is not None:
        plt.suptitle(f"Step {step}")

    plt.tight_layout()

    # ---------- 保存 ----------
    filename = f"inputs_step_{step}.png" if step is not None else "inputs.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")