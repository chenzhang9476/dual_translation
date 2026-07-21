# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import os
from typing import Dict
import time
import datetime
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets #V
from torchvision.utils import save_image
from torch.autograd import Variable #V
import json
import numpy as np
from Diffusion_condition import GaussianDiffusionSampler_cond
from Model_condition import UNet
from datasets_test import * #V

from tqdm import tqdm

import random

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # マルチGPUの場合
    # 決定論的演算を有効にする（速度は少し落ちる可能性があります）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# プログラムの最初の方で実行
set_seed(42)

debug=False
dataset_name=2

split_id=0
split_n=1

out_name=f"task{dataset_name}"

bg_value=-1
scale=1000

batch_size=512
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3
beta_1 = 1e-4
beta_T = 0.02
grad_clip = 1
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

# Create sample directories
# os.makedirs("test/%s" % out_name, exist_ok=True)


root=f"/img/synthRAD2025_Task{dataset_name}_Train_process/"
root_img=root+"HN/"
pth_json=root+"split.json"
# JSONファイルを開いて読み込む
with open(pth_json, 'r') as file:
    name_img = json.load(file)[f"test_cases"]
name_img_len=len(name_img)
name_img_block=(name_img_len//split_n)+1
name_img=name_img[name_img_block*split_id:name_img_block*(split_id+1)]

print(root)
def test():

    psnr_val_list=[]
    ssim_val_list=[]

    for name in tqdm(name_img):

        # print("seed",int(name[-3:]))
        set_seed(int(name[-3:]))

        save_dir=f"/synthrad2025/conditional_DDPM/test_300/{out_name}/{name}/"
        num_saved=len(os.listdir(save_dir))

        root_img=root+"HN/"
        dir_img=root_img+f"{name}/"
        num_img = len([f for f in os.listdir(dir_img) if f.endswith(".npy")])//3

        test_dataloader = DataLoader(
            ImageDataset(dataset_name,name,debug=debug, transforms_=False, unaligned=True, mode="test"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
        )
        with torch.no_grad():
            for ii, batch1 in enumerate(test_dataloader):
                ii = ii + 1
                # print(ii)
                ct = batch1["A"].numpy().astype(np.float32)
                ct = ct.squeeze(1)
                ct=np.clip(ct, -1, 1)
                


                
                # cbct = Variable(batch1["B"].type(Tensor)) #condition

                pths = batch1["pth"]
                # “inverse the normalized image to HU value”
                ct_pred=[]
                for pth in pths:
                    pth_list=pth.split('/')
                    
                    pred=np.load(save_dir+f"/{pth_list[-1]}")
                    # print(pred.shape)
                    ct_pred.append(pred)
                ct_pred=np.stack(ct_pred)
                ct_pred=np.clip(ct_pred, -1, 1)

                ct_pred=ct_pred*scale
                ct=ct*scale
                psnr_val = float(psnr(ct, ct_pred, data_range=2000))
                ssim_val = float(ssim(ct, ct_pred, data_range=2000))
                psnr_val_list.append(psnr_val)
                ssim_val_list.append(ssim_val)
    print("PSNR",np.mean(psnr_val_list),np.std(psnr_val_list))
    print("SSIM",np.mean(ssim_val_list),np.std(ssim_val_list))

# sampler = GaussianDiffusionSampler_cond(net_model,beta_1,beta_T,T).to(device)
test()
    
    
