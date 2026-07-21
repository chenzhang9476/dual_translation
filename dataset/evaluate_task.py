import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = "./synthRAD2025_Task2_Train_processed/HN"
JSON_PATH = os.path.join(os.path.dirname(DATA_ROOT), "slice_index_HN.json")
SAVE_DIR = "./visualization_results"
NUM_SAMPLES = 30   # 随机可视化多少张

os.makedirs(SAVE_DIR, exist_ok=True)


def load_npy(path):
    return np.load(path)


def visualize(ct, mr, mask, save_path, title=""):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(ct, cmap="gray")
    plt.title("CT")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mr, cmap="gray")
    plt.title("MR")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    with open(JSON_PATH, "r") as f:
        records = json.load(f)

    print(f"Total slices: {len(records)}")

    samples = random.sample(records, NUM_SAMPLES)

    for i, rec in enumerate(samples):
        ct_path = os.path.join(DATA_ROOT, rec["ct_path"])
        mr_path = os.path.join(DATA_ROOT, rec["cbct_path"])
        mask_path = os.path.join(DATA_ROOT, rec["mask_path"])

        ct = load_npy(ct_path)
        mr = load_npy(mr_path)
        mask = load_npy(mask_path)

        save_name = f"{rec['patient_id']}_slice{rec['slice_idx']:03d}.png"
        save_path = os.path.join(SAVE_DIR, save_name)

        title = f"{rec['patient_id']} | Slice {rec['slice_idx']}"

        visualize(ct, mr, mask, save_path, title)
        print(f'ct.min: {ct.min()}, ct.max: {ct.max()}, ct.mean: {ct.mean():.4f}, ct.std: {ct.std():.4f}')
        print(f'mr.min: {mr.min()}, mr.max: {mr.max()}, mr.mean: {mr.mean():.4f}, mr.std: {mr.std():.4f}')
        print(f"[{i+1}/{NUM_SAMPLES}] Saved {save_name}")

    print("\nVisualization finished!")


if __name__ == "__main__":
    main()
