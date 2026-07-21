import os
import json
import argparse
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter, median_filter
from core.factory import DualTranslationFactory
# ===============================
# Utils
# ===============================
def to_2d(x, i: int):
    # x: [B,1,H,W]
    return x[i, 0].detach().cpu().numpy().astype(np.float32)
def fill_background(x: torch.Tensor, mask: torch.Tensor, bg_value: float):
    """
    x, mask: [B,1,H,W], mask in {0,1}
    set outside-mask to bg_value instead of 0.
    """
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    return x * mask + bg_value * (1.0 - mask)
def fill_background_np(x: np.ndarray, mask: np.ndarray, bg_value: float):
    """
    x, mask: [D,H,W], mask in {0,1}
    """
    return x * mask + bg_value * (1.0 - mask)
def compute_metrics_3d(gt, pred, mask, bg_value=-1.0):
    """
    gt, pred, mask: [D,H,W]
    IMPORTANT: Use background fill (e.g., -1) instead of zeroing.
    """
    gt_eval = fill_background_np(gt, mask, bg_value)
    pred_eval = fill_background_np(pred, mask, bg_value)
    # Denormalize to HU space
    scale = 1000.0
    gt_hu = gt_eval * scale
    pred_hu = pred_eval * scale
    data_range = 2000.0  # from -1000 to 1000
    psnr_val = float(psnr(gt_hu, pred_hu, data_range=data_range))
    # slice-avg SSIM
    ssim_list = []
    for z in range(gt_hu.shape[0]):
        if np.sum(mask[z]) > 0:
            ssim_list.append(float(ssim(gt_hu[z], pred_hu[z], data_range=data_range)))
    ssim_val = float(np.mean(ssim_list)) if len(ssim_list) > 0 else np.nan
    mae_val = float(np.mean(np.abs(gt_hu - pred_hu)))
    return psnr_val, ssim_val, mae_val
def postprocess_volume(pred_3d, mask_3d, cfg):
    """
    Volumetric post-processing to reduce slice discontinuity.
    Do NOT introduce gray background: keep outside-mask = bg_value.
    """
    if cfg.post_smooth == "none":
        return pred_3d
    # Only smooth the in-mask region by smoothing full volume then restoring background
    if cfg.post_smooth == "gaussian":
        smoothed = gaussian_filter(pred_3d, sigma=cfg.gaussian_sigma)
    elif cfg.post_smooth == "median":
        smoothed = median_filter(pred_3d, size=cfg.median_size)
    else:
        raise ValueError(cfg.post_smooth)
    # restore background to bg_value
    smoothed = fill_background_np(smoothed, mask_3d, cfg.bg_value)
    return smoothed
# ===============================
# Config
# ===============================
def get_config():
    parser = argparse.ArgumentParser("DualDiffusion inference → 3D reconstruction")

    parser.add_argument("--modality", type=str, default="HN",
                        choices=["HN", "AB", "TH"])
    parser.add_argument("--task", type=str, default="MR",
                        choices=["MR", "CBCT"])

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/{task}/best_model_{modality}.pth")
    parser.add_argument("--output_dir", type=str,
                        default="./results/inference_3d")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_mc_samples", type=int, default=10)
    parser.add_argument("--ddim_eta", type=float, default=0.0)

    # diffusion params
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--low_timesteps", type=int, default=800)
    parser.add_argument("--high_timesteps", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--low_weight", type=float, default=0.7)
    parser.add_argument("--high_weight", type=float, default=0.3)
    parser.add_argument("--high_scale", type=float, default=4.0)
    parser.add_argument("--high_inv_scale", type=float, default=0.25)

    # background
    parser.add_argument("--bg_value", type=float, default=-1.0)

    # post-processing
    parser.add_argument("--post_smooth", type=str, default="median",
                        choices=["none", "gaussian", "median"])
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument("--median_size", type=int, default=3)

    args = parser.parse_args()

    if args.data_root is None:
        task_number = 1 if args.task == "MR" else 2
        args.data_root = f"./dataset/synthRAD2025_Task{task_number}_Train_process"
    args.data_root = os.path.join(args.data_root, args.modality)
    args.output_dir = os.path.join(args.output_dir, args.task, args.modality)
    args.checkpoint = args.checkpoint.format(task=args.task, modality=args.modality)


    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isfile(args.checkpoint), f"Checkpoint not found: {args.checkpoint}"

    return args
# ===============================
# Main
# ===============================
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
    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "volumes"), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, "volumes_gt"), exist_ok=True)
        with open(os.path.join(cfg.output_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(vars(cfg), f, indent=2)
    if use_dist:
        dist.barrier()
    # ---------- Load split ----------
    split_path = os.path.join(os.path.dirname(cfg.data_root), f"synthrad_split_{cfg.task}_{cfg.modality}.json")
    with open(split_path, "r") as f:
        split = json.load(f)
    test_cases = split["test_cases"]
    # ---------- Dataset ----------
    dataset = DualTranslationFactory.build_dataset(
        modality=cfg.modality,
        root=cfg.data_root,
        case_list=test_cases,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if use_dist else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    # ---------- Model ----------
    model = DualTranslationFactory.load_model(cfg, device)
    if use_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    model.eval()
    # ---------- Local buffers ----------
    pred_local = defaultdict(dict)
    gt_local = defaultdict(dict)
    mask_local = defaultdict(dict)
    iterable = tqdm(loader, desc="Inference") if rank == 0 else loader
    for batch in iterable:
        mr = batch["image"].to(device)
        ct_gt = batch["ct"].to(device)
        mask = batch.get("mask", torch.ones_like(ct_gt)).to(device)
        # ensure binary mask
        mask = (mask > 0.5).float()
        patient_ids = batch["patient_id"]
        slice_idxs = batch["slice_idx"]
        m = model.module if hasattr(model, "module") else model
        out = m.sample_with_uncertainty_analysis(
            mr=mr,
            mask=mask,
            n_samples=cfg.n_mc_samples,
            progress_bar=False,
            ddim_eta=cfg.ddim_eta,
            return_all_samples=False,
        )
        ct_pred = out["ct_mean"]
        # ✅ background-aware fill (instead of *mask)
        ct_pred = fill_background(ct_pred, mask, cfg.bg_value)
        ct_gt = fill_background(ct_gt, mask, cfg.bg_value)
        for i in range(mr.size(0)):
            pid = patient_ids[i]
            sid = int(slice_idxs[i])
            pred_local[pid][sid] = to_2d(ct_pred, i)
            gt_local[pid][sid] = to_2d(ct_gt, i)
            mask_local[pid][sid] = to_2d(mask, i)
    if use_dist:
        dist.barrier()
    # ---------- Gather ----------
    if use_dist:
        pred_all = [None] * world_size
        gt_all = [None] * world_size
        mask_all = [None] * world_size
        dist.all_gather_object(pred_all, pred_local)
        dist.all_gather_object(gt_all, gt_local)
        dist.all_gather_object(mask_all, mask_local)
        if rank == 0:
            pred_merged = defaultdict(dict)
            gt_merged = defaultdict(dict)
            mask_merged = defaultdict(dict)
            for d in pred_all:
                for pid in d:
                    pred_merged[pid].update(d[pid])
            for d in gt_all:
                for pid in d:
                    gt_merged[pid].update(d[pid])
            for d in mask_all:
                for pid in d:
                    mask_merged[pid].update(d[pid])
            pred_local, gt_local, mask_local = pred_merged, gt_merged, mask_merged
    # ---------- Rank0: reconstruct & metrics ----------
    if rank == 0:
        rows = []
        for pid in sorted(pred_local.keys()):
            z_ids = sorted(pred_local[pid].keys())
            pred_3d = np.stack([pred_local[pid][z] for z in z_ids], 0).astype(np.float32)
            gt_3d = np.stack([gt_local[pid][z] for z in z_ids], 0).astype(np.float32)
            mask_3d = np.stack([mask_local[pid][z] for z in z_ids], 0).astype(np.float32)
            # ✅ ensure background = bg_value (not 0)
            pred_3d = fill_background_np(pred_3d, mask_3d, cfg.bg_value)
            gt_3d = fill_background_np(gt_3d, mask_3d, cfg.bg_value)
            # ===== SynthRAD smoothing quick fix =====
            pred_3d = postprocess_volume(pred_3d, mask_3d, cfg)
            p, s, m = compute_metrics_3d(gt_3d, pred_3d, mask_3d, bg_value=cfg.bg_value)
            rows.append({
                "patient_id": pid,
                "num_slices": len(z_ids),
                "PSNR_3D": p,
                "SSIM_3D": s,
                "MAE_3D": m
            })
            nib.save(
                nib.Nifti1Image(pred_3d, np.eye(4)),
                os.path.join(cfg.output_dir, "volumes", f"{pid}_pred.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(gt_3d, np.eye(4)),
                os.path.join(cfg.output_dir, "volumes_gt", f"{pid}_gt.nii.gz")
            )
        df = pd.DataFrame(rows)
        csv_path = os.path.join(cfg.output_dir, f"metrics_3d_{cfg.modality}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved metrics to {csv_path}")
        print(f"PSNR: {df.PSNR_3D.mean():.3f} ± {df.PSNR_3D.std():.3f}")
        print(f"SSIM: {df.SSIM_3D.mean():.4f} ± {df.SSIM_3D.std():.4f}")
        print(f"MAE : {df.MAE_3D.mean():.4f} ± {df.MAE_3D.std():.4f}")
    if use_dist:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
