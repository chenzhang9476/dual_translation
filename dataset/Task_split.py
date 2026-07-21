import os
import json
import SimpleITK as sitk
import numpy as np

# =========================
# Path configuration
# =========================
INPUT_ROOT = r"./synthRAD2025_Task2_Train/Task2"
OUTPUT_ROOT = r"./synthRAD2025_Task2_Train_processed"

MODALITIES = ["ct", "cbct", "mr", "mask"]   # Supported modalities: CT / CBCT / MRI

# HU range for CT / CBCT
HU_MIN = -1024
HU_MAX = 2000

# =========================
# Utility functions
# =========================
def load_mha(path):
    """Load .mha volume and return numpy array with shape [D, H, W]."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # [D, H, W]
    return arr.astype(np.float32)

def normalize_ct_cbct(volume):
    """CT / CBCT normalization: clip to [-1024, 2000] HU."""
    volume = np.clip(volume, HU_MIN, HU_MAX)
    return volume

def normalize_mri(volume):
    """MRI normalization: clip to 1st–99th percentile, keep original scale."""
    p1 = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)

    if p99 - p1 < 1e-6:
        return np.zeros_like(volume)

    volume = np.clip(volume, p1, p99)
    return volume

def save_slices(volume, out_dir, prefix):
    """Save a 3D volume as individual 2D slice .npy files."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(volume.shape[0]):
        fname = f"{prefix}_{i:03d}.npy"
        np.save(os.path.join(out_dir, fname), volume[i])

# =========================
# Main processing pipeline
# =========================
all_index = {
    "AB": [],
    "HN": [],
    "TH": []
}

for organ in ["AB", "HN", "TH"]:
    organ_in = os.path.join(INPUT_ROOT, organ)
    organ_out = os.path.join(OUTPUT_ROOT, organ)

    if not os.path.exists(organ_in):
        print(f"Skip {organ}, directory not found.")
        continue

    print(f"\nProcessing {organ}...")

    for case in sorted(os.listdir(organ_in)):
        case_dir = os.path.join(organ_in, case)
        if not os.path.isdir(case_dir):
            continue

        print(f"  Case: {case}")
        out_case_dir = os.path.join(organ_out, case)

        volumes = {}

        for mod in MODALITIES:
            mha_path = os.path.join(case_dir, f"{mod}.mha")
            if not os.path.exists(mha_path):
                print(f"   Missing {mod}.mha")
                continue

            vol = load_mha(mha_path)

            if mod in ["ct", "cbct"]:
                vol = normalize_ct_cbct(vol)

            elif mod == "mr":
                vol = normalize_mri(vol)

            elif mod == "mask":
                vol = (vol > 0).astype(np.uint8)

            volumes[mod] = vol
            save_slices(vol, out_case_dir, mod)

        # Generate slice-level index
        if "ct" in volumes and "cbct" in volumes and "mask" in volumes:
            num_slices = volumes["ct"].shape[0]

            for i in range(num_slices):
                entry = {
                    "patient_id": case,
                    "slice_idx": i,
                    "ct_path": f"{case}/ct_{i:03d}.npy",
                    "cbct_path": f"{case}/cbct_{i:03d}.npy",
                    "mask_path": f"{case}/mask_{i:03d}.npy"
                }

                if "mr" in volumes:
                    entry["mr_path"] = f"{case}/mr_{i:03d}.npy"

                all_index[organ].append(entry)

# =========================
# Save JSON index files
# =========================
for organ in ["AB", "HN", "TH"]:
    json_path = os.path.join(OUTPUT_ROOT, f"slice_index_{organ}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_index[organ], f, indent=2)
    print(f"Saved: {json_path}")

print("\n✅ Done! CT/CBCT and MRI normalization applied.")
