# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import glob
import random
import os
import numpy as np #V
import torch as torch #V
import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, task_id,name,debug, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        if task_id==1:
            task_target="mr"
        elif task_id==2:
            task_target="cbct"

        root=f"/img/synthRAD2025_Task{task_id}_Train_process/"
        self.min_fg_ratio=0.10
        root_img=root+"HN/"
        # 読み込んだデータを表示

        files={"A":[],"B":[],"M":[]}

        dir_img=root_img+f"{name}/"
        num_img = len([f for f in os.listdir(dir_img) if f.endswith(".npy")])//3

        name_A=[dir_img+f"ct_{str(j).zfill(3)}.npy" for j in range(num_img)]
        name_B=[dir_img+f"{task_target}_{str(j).zfill(3)}.npy" for j in range(num_img)]

        name_M=[dir_img+f"mask_{str(j).zfill(3)}.npy" for j in range(num_img)]
        
        files["A"]+=name_A
        files["B"]+=name_B
        files["M"]+=name_M

        #dict of lists → list of dicts
        files = [
            {"A": a, "B": b, "M": m}
            for a, b, m in zip(files["A"], files["B"], files["M"])
        ]

        self.files_A=[]
        self.files_B=[]
        self.files_M=[]

        for i in range(len(files)):
            A=files[i]["A"]
            B=files[i]["B"]
            M=files[i]["M"]
            # if (mode=="test")or(self._has_any_valid_modality(A,B,M) and self._is_valid_mask(M)):
            if (self._has_any_valid_modality(A,B,M) and self._is_valid_mask(M)):
                self.files_A.append(A)
                self.files_B.append(B)
                self.files_M.append(M)
                
        self.files_A=sorted(self.files_A)
        self.files_B=sorted(self.files_B)
        self.files_M=sorted(self.files_M)
        
        if debug:
            self.files_A=self.files_A[:2]
            self.files_B=self.files_B[:2]
            self.files_M=self.files_M[:2]

        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

        self.target_size = (256, 256)

    def _has_any_valid_modality(self, a,b,m):
        if ((os.path.exists(a))&(os.path.exists(b)))&(os.path.exists(m)):
            return True
        return False

    def _is_valid_mask(self, m):
        if not os.path.exists(m):
            return False

        arr = np.load(m)

        # 1) Skip if mask is completely empty
        if not np.any(arr > 0):
            return False

        # 2) Foreground ratio filtering (skip if < threshold)
        fg_ratio = np.sum(arr > 0) / arr.size
        if fg_ratio < self.min_fg_ratio:
            return False

        return True

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

    def safe_load(self, full_path):
        if full_path is None:
            return None

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

    def __getitem__(self, index):
        item_A = self.safe_load(self.files_A[index % self.len_A])
        item_B = self.safe_load(self.files_B[index % self.len_B])
        item_M = self.safe_load(self.files_M[index % self.len_A])

        return {
            "pth":self.files_A[index % self.len_A],
            "M": item_M,
            "A": item_A,
            "B": item_B,

        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

