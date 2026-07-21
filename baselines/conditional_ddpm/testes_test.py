import os
import torch
import numpy as np
from PIL import Image

# ----------------------------
# 設定
# ----------------------------
pt_path = "test/task1/result_epoch110.pt"
save_dir = "test/task1/png"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 辞書ロード
# ----------------------------
data_dict = torch.load(pt_path, map_location="cpu")

print(f"Loaded {len(data_dict)} images")

# ----------------------------
# PNG保存
# ----------------------------
for key, img in data_dict.items():

    # img: (1,H,W) → (H,W)
    img = np.squeeze(img)

    # -------- 正規化 → 0-255 --------
    # [-1,1] 想定
    img = (img + 1) / 2          # → [0,1]
    # eps = 1e-8
    # img = (img - img.min()) / (img.max() - img.min() + eps)
    print(img.max(),img.min())
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    # -------- 保存 --------
    print(key)
    save_path = os.path.join(save_dir, f'{key.split(".")[0].split("/")[-1]}.png')
    Image.fromarray(img).save(save_path)

print("PNG conversion done.")
