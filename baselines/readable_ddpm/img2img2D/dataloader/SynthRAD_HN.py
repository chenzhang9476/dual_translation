import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SynthRAD_HN_Dataset(Dataset):
    """
    Final stable version for Readable-Conditional-Diffusion (img2img2D)

    - Safe with Wrapper_Dataset (no RuntimeError in __getitem__)
    - Condition modality mapping handled internally (MRI -> mr)
    - Slice-level filtering guarantees all indices are valid
    """

    MODALITY_MAP = {
        "MRI": "mr",
        "MR": "mr",
        "Mri": "mr",   # 防御（可选）
        "mri": "mr",   # 防御（可选）
        "CBCT": "cbct",
        "cbct": "cbct",
    }

    def __init__(
        self,
        root="comparison/Readable-Conditional-Denoising-Diffusion-main/Readable-Conditional-Denoising-Diffusion-main/img2img2D/dataloader/HN",
        index_json="slice_index_HN.json",
        case_list=None,
        target_size=(256, 256),
        min_fg_ratio=0.10,
        condition_modalities=None,   # e.g. ["MRI"]
        train=True,                  # required by Wrapper
    ):
        super().__init__()
        self.train = train
        self.root = root
        self.target_size = target_size
        self.min_fg_ratio = min_fg_ratio

        # -------- normalize condition modality names --------
        raw_modalities = condition_modalities or ["mr", "cbct"]
        self.condition_modalities = [
            self.MODALITY_MAP[m.upper()] for m in raw_modalities
        ]

        print("[Dataset] condition_modalities =", self.condition_modalities)

        index_path = os.path.join(os.path.dirname(root), index_json)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"[Dataset ERROR] index file not found: {index_path}")

        with open(index_path, "r") as f:
            records = json.load(f)

        # -------- case-level filtering --------
        if case_list is not None:
            case_set = set(case_list)
            records = [r for r in records if r.get("patient_id") in case_set]

        # -------- slice-level filtering (CRITICAL) --------
        valid_records = [
            r for r in records
            if self._has_valid_condition(r) and self._is_valid_mask(r)
        ]

        print(
            f"[Dataset] Filtered {len(records) - len(valid_records)} invalid slices "
            f"(no valid condition or fg < {self.min_fg_ratio*100:.1f}%)"
        )

        if len(valid_records) == 0:
            raise RuntimeError(
                "[Dataset ERROR] No valid slices left after filtering. "
                "Check condition_types / modality mapping."
            )

        self.records = valid_records

    # ==========================================================
    # Filtering utils
    # ==========================================================
    def _has_valid_condition(self, r):
        for mod in self.condition_modalities:
            key = f"{mod}_path"
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
        if not np.any(arr > 0):
            return False

        fg_ratio = np.sum(arr > 0) / arr.size
        return fg_ratio >= self.min_fg_ratio

    # ==========================================================
    # IO utils
    # ==========================================================
    def safe_load(self, path):
        if path is None:
            return None

        full_path = os.path.join(self.root, path)
        if not os.path.exists(full_path):
            return None

        arr = np.load(full_path).astype(np.float32)

        # auto normalize to [-1, 1]
        vmin, vmax = arr.min(), arr.max()
        if vmax > 1.0 or vmin < -1.0:
            arr = (arr - vmin) / (vmax - vmin + 1e-6)
            arr = arr * 2 - 1

        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return self.resize(x)

    def load_mask(self, path):
        if path is None:
            return None

        full = os.path.join(self.root, path)
        if not os.path.exists(full):
            return None

        arr = (np.load(full) > 0).astype(np.uint8)
        x = torch.from_numpy(arr).unsqueeze(0).float()
        return self.resize_mask(x)

    # ==========================================================
    # Resize
    # ==========================================================
    def resize(self, x):
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(0)

    def resize_mask(self, x):
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=self.target_size,
            mode="nearest",
        )
        return x.squeeze(0)

    # ==========================================================
    # Dataset interface
    # ==========================================================
    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        # -------- condition (MR) --------
        condition = None
        for mod in self.condition_modalities:
            key = f"{mod}_path"
            condition = self.safe_load(r.get(key))
            if condition is not None:
                break

        # 防御：理论上 init 已过滤，不应发生
        if condition is None:
            return self.__getitem__((idx + 1) % len(self))

        # -------- target (CT) --------
        target = self.safe_load(r.get("ct_path"))
        if target is None:
            return self.__getitem__((idx + 1) % len(self))

        # ✅ 关键：返回 tuple，不要返回 dict
        return target, condition

    # ==========================================================
    # Case-level helper
    # ==========================================================
    def get_all_case_ids(self):
        return sorted({r["patient_id"] for r in self.records})
