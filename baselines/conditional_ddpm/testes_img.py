import numpy as np
import matplotlib.pyplot as plt

# ===== ここに表示したい npy パスを書く =====
npy_path = "/synthrad2025/conditional_DDPM/test/task1/1HNC040/ct_011.npy"
npy_path = "/img/synthRAD2025_Task1_Train_process/HN/1HNC040/ct_011.npy"
# ===========================================

# 読み込み
img = np.load(npy_path)
print(img.shape)
print("shape:", img.shape)
print("dtype:", img.dtype)
print("min:", img.min(), "max:", img.max())

# もし3D (H,W,1) などなら squeeze
img = np.squeeze(img)

# min-max 正規化
img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

# 表示
plt.figure()
plt.imshow(img_norm, cmap="gray")
plt.colorbar()
plt.title(npy_path)
plt.axis("off")
plt.savefig("output.png", bbox_inches="tight", pad_inches=0)
print("保存完了 → output.png")