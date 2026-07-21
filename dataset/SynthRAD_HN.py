import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SynthRAD_HN_Dataset(Dataset):
    def __init__(
        self,
        root="./synthRAD_Task2_Train_processed/HN",
        index_json="slice_index_HN.json",
        case_list=None,
        target_size=(256, 256),
        min_fg_ratio=0.10,   # ✅ Foreground ratio threshold (10%)
    ):
        super().__init__()
        self.root = root
        self.target_size = target_size
        self.min_fg_ratio = min_fg_ratio
        index_path = os.path.join(os.path.dirname(root), index_json)
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"[Dataset ERROR] index file not found: {index_path}\n"
                f"→ Either generate slice_index.json OR use a JSON-free Dataset."
            )

        with open(index_path, "r") as f:
            records = json.load(f)

        # ---------------- case-level filtering ----------------
        if case_list is not None:
            case_set = set(case_list)
            records = [r for r in records if r.get("patient_id") in case_set]

        # ---------------- filter invalid slices ----------------
        valid_records = [
            r for r in records
            if self._has_any_valid_modality(r) and self._is_valid_mask(r)
        ]

        print(
            f"[Dataset] Filtered {len(records) - len(valid_records)} invalid slices "
            f"(missing image, empty mask, or fg < {self.min_fg_ratio*100:.1f}%)"
        )

        self.records = valid_records

    # ==========================================================
    # Utilities
    # ==========================================================
    def _has_any_valid_modality(self, r):
        for key in ["mr_path", "cbct_path", "ct_path"]:
            p = r.get(key)
            if p is not None:
                full = os.path.join(self.root, p)
                if os.path.exists(full):
                    return True
        return False

    def _is_valid_mask(self, r):
        p = r.get("mask_path")
        if p is None:
            return False

        full = os.path.join(self.root, p)
        if not os.path.exists(full):
            return False

        arr = np.load(full)

        # 1) Skip if mask is completely empty
        if not np.any(arr > 0):
            return False

        # 2) Foreground ratio filtering (skip if < threshold)
        fg_ratio = np.sum(arr > 0) / arr.size
        if fg_ratio < self.min_fg_ratio:
            return False

        return True

    def safe_load(self, path):
        if path is None:
            return None

        full_path = os.path.join(self.root, path)
        if not os.path.exists(full_path):
            return None

        arr = np.load(full_path).astype(np.float32)

        # ---------------- automatically determine normalization ----------------
        vmin, vmax = arr.min(), arr.max()

        if vmax > 1.0 or vmin < -1.0:
            # Raw intensity → normalize to [-1, 1]
            arr = (arr - vmin) / (vmax - vmin + 1e-6)
            arr = arr * 2 - 1

        # Otherwise assume already in [-1, 1] or [0, 1]

        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return self.resize(tensor)

    def load_mask(self, path):
        if path is None:
            return None

        full_path = os.path.join(self.root, path)
        if not os.path.exists(full_path):
            return None

        arr = np.load(full_path)
        arr = (arr > 0).astype(np.uint8)   # enforce binary mask

        tensor = torch.from_numpy(arr).unsqueeze(0).float()  # [1,H,W]
        return self.resize_mask(tensor)

    def resize_mask(self, x):
        x = x.unsqueeze(0)  # [1,1,H,W]
        x = F.interpolate(
            x,
            size=self.target_size,
            mode="nearest",   # 🚨 critical: preserve label integrity
        )
        return x.squeeze(0)  # [1,H,W]

    def resize(self, x):
        # x: [1,H,W] → [1,1,H,W]
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(0)  # [1,H,W]

    def select_input_image(self, r):
        for key, name in [
            ("mr_path", "mr"),
            ("cbct_path", "cbct"),
            ("ct_path", "ct"),
        ]:
            img = self.safe_load(r.get(key))
            if img is not None:
                return img, name

        raise RuntimeError(
            f"[Dataset ERROR] No valid input modality for "
            f"{r['patient_id']} slice {r['slice_idx']}"
        )

    # ==========================================================
    # Dataset Interface
    # ==========================================================
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        image, image_type = self.select_input_image(r)
        ct = self.safe_load(r.get("ct_path"))
        mask = self.load_mask(r.get("mask_path"))
        
        return {
            "image": image,        # [1,H,W]
            "image_type": image_type,
            "ct": ct,              # [1,H,W] or None
            "mask": mask,          # always valid (foreground ≥ threshold)
            "patient_id": r["patient_id"],
            "slice_idx": r["slice_idx"],
        }

    # ==================================================
    # For generating dataset splits
    # ==================================================
    def get_all_case_ids(self):
        """Return all unique patient IDs (case-level)."""
        return sorted({r["patient_id"] for r in self.records})
