"""
General-purpose test script for image-to-image translation (HU-aware).

This version extends the original CycleGAN test.py with:
- HU denormalization
- MAE / PSNR / SSIM computed in HU domain
- Per-sample (slice-level) metric saving to CSV
- Case-level metric aggregation (mean ± std)

The modification is minimal and does NOT break the original framework.
"""

import os
from pathlib import Path
import torch
import numpy as np
import csv
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


# =========================
# HU settings
# =========================
HU_MIN = -1024.0
HU_MAX = 2000.0


# =========================
# Utility functions
# =========================
def tensor_to_numpy(x):
    """
    x: torch.Tensor [1,1,H,W] or [1,3,H,W], range [-1,1]
    return: numpy [H,W]
    """
    x = x.detach().cpu().float().numpy()
    x = x.squeeze()
    return x


def denormalize_to_hu(x, hu_min=HU_MIN, hu_max=HU_MAX):
    """
    x: numpy array, range [-1,1]
    return: numpy array, range [hu_min, hu_max]
    """
    x = (x + 1.0) / 2.0
    x = x * (hu_max - hu_min) + hu_min
    return x


def compute_hu_metrics(fake, real):
    """
    fake, real: numpy [H,W], range [-1,1]
    Metrics are computed in HU domain.
    """
    fake_hu = denormalize_to_hu(fake)
    real_hu = denormalize_to_hu(real)

    mae = np.mean(np.abs(fake_hu - real_hu))

    psnr = peak_signal_noise_ratio(
        real_hu,
        fake_hu,
        data_range=(HU_MAX - HU_MIN)
    )

    ssim = structural_similarity(
        real_hu,
        fake_hu,
        data_range=(HU_MAX - HU_MIN)
    )

    return psnr, ssim, mae


# =========================
# Main
# =========================
if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- official test behavior ----
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # ---- create dataset & model ----
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # ---- website ----
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"
    if opt.load_iter > 0:
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")

    print(f"creating web directory {web_dir}")
    webpage = html.HTML(
        web_dir,
        f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}"
    )

    if opt.eval:
        model.eval()

    # =========================
    # Metric accumulators
    # =========================
    psnr_list = []
    ssim_list = []
    mae_list  = []

    records = []   # slice-level records

    # case-level container
    case_metrics = defaultdict(lambda: {
        "psnr": [],
        "ssim": [],
        "mae":  []
    })

    # =========================
    # Inference loop
    # =========================
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        # ----- HU metrics -----
        if 'fake_B' in visuals and 'real_B' in visuals:
            fake = tensor_to_numpy(visuals['fake_B'])
            real = tensor_to_numpy(visuals['real_B'])

            psnr, ssim, mae = compute_hu_metrics(fake, real)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mae_list.append(mae)

            img_name = img_path[0] if isinstance(img_path, list) else img_path

            # extract case id: e.g. 1HNA082_81 -> 1HNA082
            case_id = os.path.basename(img_name).split("_")[0]

            # slice-level record
            records.append({
                "index": i,
                "case_id": case_id,
                "image_path": img_name,
                "PSNR_HU": psnr,
                "SSIM_HU": ssim,
                "MAE_HU": mae,
            })

            # case-level accumulation
            case_metrics[case_id]["psnr"].append(psnr)
            case_metrics[case_id]["ssim"].append(ssim)
            case_metrics[case_id]["mae"].append(mae)

        if i % 5 == 0:
            print(f"processing ({i:04d})-th image... {img_path}")

        save_images(
            webpage,
            visuals,
            img_path,
            aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize
        )

    # ---- save HTML ----
    webpage.save()

    # =========================
    # Save slice-level CSV
    # =========================
    slice_csv_path = web_dir / "metrics_hu_slice_level.csv"
    with open(slice_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "case_id", "image_path", "PSNR_HU", "SSIM_HU", "MAE_HU"]
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"Slice-level HU metrics saved to: {slice_csv_path}")

    # =========================
    # Case-level statistics
    # =========================
    print("\n" + "=" * 80)
    print("Case-level Evaluation (HU domain)")
    print(f"HU range: [{HU_MIN}, {HU_MAX}]\n")
    print("| Case ID | #Slices | PSNR (mean±std) | SSIM (mean±std) | MAE (mean±std) |")
    print("|--------|---------|------------------|------------------|----------------|")

    for case_id, m in sorted(case_metrics.items()):
        psnr_arr = np.array(m["psnr"])
        ssim_arr = np.array(m["ssim"])
        mae_arr  = np.array(m["mae"])

        print(
            f"| {case_id} "
            f"| {len(psnr_arr):3d} "
            f"| {psnr_arr.mean():.3f} ± {psnr_arr.std():.3f} "
            f"| {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f} "
            f"| {mae_arr.mean():.2f} ± {mae_arr.std():.2f} |"
        )

    print("=" * 80)

    # =========================
    # Save case-level CSV
    # =========================
    case_csv_path = web_dir / "metrics_hu_case_level.csv"
    with open(case_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "num_slices",
                "PSNR_mean", "PSNR_std",
                "SSIM_mean", "SSIM_std",
                "MAE_mean",  "MAE_std",
            ]
        )
        writer.writeheader()

        for case_id, m in sorted(case_metrics.items()):
            psnr_arr = np.array(m["psnr"])
            ssim_arr = np.array(m["ssim"])
            mae_arr  = np.array(m["mae"])

            writer.writerow({
                "case_id": case_id,
                "num_slices": len(psnr_arr),
                "PSNR_mean": psnr_arr.mean(),
                "PSNR_std":  psnr_arr.std(),
                "SSIM_mean": ssim_arr.mean(),
                "SSIM_std":  ssim_arr.std(),
                "MAE_mean":  mae_arr.mean(),
                "MAE_std":   mae_arr.std(),
            })

    print(f"Case-level HU metrics saved to: {case_csv_path}")
