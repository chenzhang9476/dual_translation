import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SynthRAD_TH_Dataset(Dataset):
    def __init__(
        self,
        root="./synthRAD_Task1_Train_process/TH",
        index_json="slice_index_TH.json",
        case_list=None,
        target_size=(256, 256),
        min_fg_ratio=0.10,   # ✅ 前景比例阈值（10%）
    ):
        super().__init__()
        self.root = root
        self.target_size = target_size
        self.min_fg_ratio = min_fg_ratio

        index_path = os.path.join(os.path.dirname(root), index_json)
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"[Dataset ERROR] index file not found: {index_path}\n"
                f"→ Either generate slices_index.json OR use a JSON-free Dataset."
            )

        with open(index_path, "r") as f:
            records = json.load(f)

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
            f"(no image, empty mask, or fg < {self.min_fg_ratio*100:.1f}%)"
        )

        self.records = valid_records

    # ==========================================================
    # Utils
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

        if not np.any(arr > 0):
            return False

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

        # ---------------- 自动判断是否需要归一化 ----------------
        vmin, vmax = arr.min(), arr.max()

        if vmax > 1.0 or vmin < -1.0:
            # 说明是原始强度 → 归一化到 [-1, 1]
            arr = (arr - vmin) / (vmax - vmin + 1e-6)
            arr = arr * 2 - 1

        # 否则认为已经是 [-1,1] 或 [0,1]，不再处理

        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return self.resize(tensor)

    def resize(self, x):
        x = x.unsqueeze(0)
        x = F.interpolate(
            x,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        return x.squeeze(0)

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

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        image, image_type = self.select_input_image(r)
        ct = self.safe_load(r.get("ct_path"))
        mask = self.safe_load(r.get("mask_path"))

        return {
            "image": image,
            "image_type": image_type,
            "ct": ct,
            "mask": mask,
            "patient_id": r["patient_id"],
            "slice_idx": r["slice_idx"],
        }

    def get_all_case_ids(self):
        return sorted({r["patient_id"] for r in self.records})
