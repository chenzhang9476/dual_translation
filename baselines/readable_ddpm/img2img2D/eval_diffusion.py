from __future__ import annotations
import os, glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from diffusion import Diffusion
from loader import load_dataset


# =====================
# metrics
# =====================
def mae(pred, gt):
    return np.mean(np.abs(pred - gt))


def psnr(pred, gt):
    dr = gt.max() - gt.min()
    return peak_signal_noise_ratio(gt, pred, data_range=dr)


def ssim(pred, gt):
    dr = gt.max() - gt.min()
    return structural_similarity(gt, pred, data_range=dr)


# =====================
# auto find experiment
# =====================
def auto_find_exp():
    exps = [e for e in os.listdir("logs_diffusion") if not e.startswith(".")]
    assert len(exps) == 1, f"❌ 发现多个实验：{exps}"
    return exps[0]


def auto_find_ckpt(exp):
    ckpts = glob.glob(f"logs_diffusion/{exp}/version_0/checkpoints/epoch=99-step=57000-train_All=0.0180_latest.ckpt")
    assert len(ckpts) > 0, "❌ 找不到 checkpoint"
    return sorted(ckpts, key=os.path.getmtime)[-1]


# =====================
# main
# =====================
@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🔥 自动找实验 & ckpt
    exp_name = auto_find_exp()
    ckpt_path = auto_find_ckpt(exp_name)

    print(f"[EXP ] {exp_name}")
    print(f"[CKPT] {ckpt_path}")

    # 🔥 从 checkpoint 恢复 opt（关键）
    ckpt = torch.load(ckpt_path, map_location="cpu")
    opt = ckpt["hyper_parameters"]["opt"]
    opt.new = False

    # dataset (test)
    dataset_test, in_channel = load_dataset.getDataset(opt, train=False)
    loader = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # model
    model = Diffusion(opt=opt, in_channel=in_channel)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    PSNR, SSIM, MAE = [], [], []

    for batch in tqdm(loader, desc="Inference"):
        target = batch["target"].to(device)
        cond = batch["condition"].to(device)

        pred, _ = model.forward_ddim(
            batch_size=1,
            intermediate=list(range(0, opt.timesteps, 10)),
            x_conditional=cond,
            eta=0.0,
            w=0,
            progressbar=False,
        )

        pred = pred.cpu().numpy()[0, 0]
        gt = model.denormalize(target).cpu().numpy()[0, 0]

        PSNR.append(psnr(pred, gt))
        SSIM.append(ssim(pred, gt))
        MAE.append(mae(pred, gt))

    print("\n========== RESULT ==========")
    print(f"PSNR: {np.mean(PSNR):.4f} ± {np.std(PSNR):.4f}")
    print(f"SSIM: {np.mean(SSIM):.4f} ± {np.std(SSIM):.4f}")
    print(f"MAE : {np.mean(MAE):.4f} ± {np.std(MAE):.4f}")


if __name__ == "__main__":
    main()
