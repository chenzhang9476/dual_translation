import os
import argparse
import json
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from core.factory import DualTranslationFactory

# ============================================================
# Utils
# ============================================================

def to_2d(x, i: int, c: int = 0):
    """x: torch.Tensor or np.ndarray. [B,C,H,W] -> [H,W] numpy"""
    if isinstance(x, torch.Tensor):
        if x.ndim == 4:
            return x[i, c].detach().cpu().numpy().astype(np.float32)
        else:
            return x[i].detach().cpu().numpy().astype(np.float32)

    # numpy
    x = np.asarray(x)
    if x.ndim == 4:
        return x[i, c].astype(np.float32)
    else:
        return x[i].astype(np.float32)



def fill_background_2d(x: torch.Tensor, mask: torch.Tensor, bg_value: float):
    """
    x, mask: [B,1,H,W], mask in {0,1}
    Fill background with bg_value instead of zeroing.
    """
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    return x * mask + bg_value * (1.0 - mask)


def compute_metrics_2d(gt_hu: np.ndarray, pred_hu: np.ndarray, data_range=2000.0):
    """
    Metrics in HU space.
    Background already filled with -1000 HU.
    Metrics computed on full image.
    """
    gt = np.squeeze(gt_hu).astype(np.float64)
    pred = np.squeeze(pred_hu).astype(np.float64)

    psnr_val = psnr(gt, pred, data_range=data_range)

    h, w = gt.shape
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        ssim_val = np.nan
    else:
        ssim_val = ssim(gt, pred, data_range=data_range)

    mae_val = float(np.mean(np.abs(gt - pred)))
    return psnr_val, ssim_val, mae_val


# ============================================================
# Config
# ============================================================

def get_config():
    parser = argparse.ArgumentParser("DualDiffusion 2D inference (background-aware)")

    parser.add_argument("--modality", type=str, default="HN", choices=["HN", "AB", "TH"])
    parser.add_argument("--task", type=str, default="MR", choices=["MR", "CBCT"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/inference_2d")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_mc_samples", type=int, default=10)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--save_vis", action="store_true", default=False)

    # diffusion params (must match training)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--low_timesteps", type=int, default=800)
    parser.add_argument("--high_timesteps", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--low_weight", type=float, default=0.7)
    parser.add_argument("--high_weight", type=float, default=0.3)
    parser.add_argument("--high_scale", type=float, default=4.0)
    parser.add_argument("--high_inv_scale", type=float, default=1 / 4.0)

    # background / HU
    parser.add_argument("--bg_value", type=float, default=-1.0)
    parser.add_argument("--hu_scale", type=float, default=1000.0)

    args = parser.parse_args()
    if args.data_root is None:
        task_number = 1 if args.task == "MR" else 2
        args.data_root = f"./dataset/synthRAD2025_Task{task_number}_Train_process"
    if args.checkpoint is None:
        args.checkpoint = f"./checkpoints/{args.task}/best_model_{args.modality}.pth"
    args.checkpoint = args.checkpoint.format(
        task=args.task,
        modality=args.modality,
    )
    args.data_root = os.path.join(args.data_root, args.modality)
    args.output_dir = os.path.join(args.output_dir, args.task, args.modality)
    return args


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    cfg = get_config()

    # ---------- Distributed ----------
    use_dist = torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_dist:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Output ----------
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "pred_ct"), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "vis"), exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
            json.dump(vars(cfg), f, indent=2)

    if use_dist:
        dist.barrier()

   # -------------------------
    # Load test split
    # -------------------------
    split_path = os.path.join(os.path.dirname(cfg.data_root), f"synthrad_split_{cfg.task}_{cfg.modality}.json")
    with open(split_path, "r") as f:
        split = json.load(f)
    test_cases = split["test_cases"]
    if rank == 0:
        print(f"Loaded {len(test_cases)} test cases for modality {cfg.modality} for task {cfg.task}")
    # -------------------------
    # Dataset
    # -------------------------
    dataset = DualTranslationFactory.build_dataset(
        modality=cfg.modality,
        root=cfg.data_root,
        case_list=test_cases,
    )

    sampler = DistributedSampler(dataset, world_size, rank, shuffle=False) if use_dist else None
    loader = DataLoader(dataset, cfg.batch_size, sampler=sampler,
                        shuffle=False, num_workers=cfg.num_workers)

    # ---------- Model ----------
    model = DualTranslationFactory.load_model(cfg, device)
    if use_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    model.eval()

    metrics = []
    iterable = tqdm(loader, desc="Inference") if rank == 0 else loader

    # ---------- Inference ----------
    for batch in iterable:
        batch_psnr, batch_ssim, batch_mae = [], [], []

        mr = batch["image"].to(device)
        ct_gt = batch["ct"].to(device)
        mask = batch.get("mask", torch.ones_like(ct_gt)).to(device)
        mask = (mask > 0.5).float()

        patient_ids = batch["patient_id"]
        slice_idxs = batch["slice_idx"]

        m = model.module if hasattr(model, "module") else model
        out = m.sample_with_uncertainty_analysis(
            mr=mr,
            mask=mask,
            n_samples=cfg.n_mc_samples,
            ddim_eta=cfg.ddim_eta,
            progress_bar=False,
            return_all_samples=False,
        )

        ct_pred = fill_background_2d(out["ct_mean"], mask, cfg.bg_value)
        ct_gt   = fill_background_2d(ct_gt, mask, cfg.bg_value)

        ct_pred_hu = ct_pred.cpu().numpy() * cfg.hu_scale
        ct_gt_hu   = ct_gt.cpu().numpy() * cfg.hu_scale

        for i in range(ct_pred.size(0)):
            pid = patient_ids[i]
            sid = int(slice_idxs[i])

            pred_np = to_2d(ct_pred_hu, i)
            gt_np   = to_2d(ct_gt_hu, i)

            p, s, mval = compute_metrics_2d(gt_np, pred_np)

            batch_psnr.append(p)
            batch_ssim.append(s)
            batch_mae.append(mval)

            metrics.append({
                "patient_id": pid,
                "slice_idx": sid,
                "PSNR": p,
                "SSIM": s,
                "MAE": mval
            })

            nib.save(
                nib.Nifti1Image(pred_np, np.eye(4)),
                os.path.join(cfg.output_dir, "pred_ct", f"{pid}_slice{sid:04d}.nii.gz")
            )

        if rank == 0:
            iterable.set_postfix({
                "PSNR": f"{np.mean(batch_psnr):.2f}",
                "SSIM": f"{np.mean(batch_ssim):.3f}",
                "MAE": f"{np.mean(batch_mae):.1f}",
            })

    if use_dist:
        dist.barrier()

    if rank == 0:
        df = pd.DataFrame(metrics)
        csv = os.path.join(cfg.output_dir, f"metrics_2d_{cfg.modality}.csv")
        df.to_csv(csv, index=False)
        print(f"\nSaved metrics to {csv}")
        print(f"PSNR: {df.PSNR.mean():.2f} ± {df.PSNR.std():.2f}")
        print(f"SSIM: {df.SSIM.mean():.4f} ± {df.SSIM.std():.4f}")
        print(f"MAE : {df.MAE.mean():.2f} ± {df.MAE.std():.2f}")

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
