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

split_id=3
split_n=4
epoch=300

out_name=f"task{dataset_name}"


batch_size=64
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3
beta_1 = 1e-4
beta_T = 0.02
grad_clip = 1
save_weight_dir = "./Checkpoints/%s"%out_name
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")

# Create sample directories
os.makedirs("test/%s" % out_name, exist_ok=True)


root=f"/img/synthRAD2025_Task{dataset_name}_Train_process/"
root_img=root+"HN/"
pth_json=root+"split.json"
# JSONファイルを開いて読み込む
with open(pth_json, 'r') as file:
    name_img = json.load(file)[f"test_cases"]
name_img_len=len(name_img)
name_img_block=(name_img_len//split_n)+1
name_img=name_img[name_img_block*split_id:name_img_block*(split_id+1)]

def test():
    sampler = GaussianDiffusionSampler_cond(net_model,beta_1,beta_T,T).to(device)
    
    for name in tqdm(name_img):




        print("seed",int(name[-3:]))
        set_seed(int(name[-3:]))

        save_dir=f"test_300/{out_name}/{name}/"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        num_saved=len(os.listdir(save_dir))

        root_img=root+"HN/"
        dir_img=root_img+f"{name}/"
        num_img = len([f for f in os.listdir(dir_img) if f.endswith(".npy")])//3
        if num_saved>=num_img:
            continue




        test_dataloader = DataLoader(
            ImageDataset(dataset_name,name,debug=debug, transforms_=False, unaligned=True, mode="test"),
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
        )
        with torch.no_grad():
            for ii, batch1 in enumerate(test_dataloader):
                ii = ii + 1
                print(ii)
                # ct = Variable(batch1["A"].type(Tensor))
                cbct = Variable(batch1["B"].type(Tensor)) #condition

                B_size = cbct.size(0)
                pth = batch1["pth"]
                noisyImage = torch.randn(size=[B_size, 1, 256, 256], device=device)
                x_in = torch.cat((noisyImage,cbct),1)
                x_out = sampler(x_in)
                
                # “inverse the normalized image to HU value”
                
                fake = x_out[:,0,:,:]

                fake = fake.cpu().numpy().astype(np.float32)
                for i in range(B_size):
                    print(save_dir+"/"+pth[i].split("/")[-1])
                    np.save(save_dir+"/"+pth[i].split("/")[-1], fake[i])




# def test():
#     result_dict = {}

#     img_save = torch.Tensor()
#     sampler = GaussianDiffusionSampler_cond(net_model,beta_1,beta_T,T).to(device)
#     with torch.no_grad():
#         for ii, batch1 in enumerate(tqdm(test_dataloader)):
#             ii = ii + 1
#             print(ii)
#             ct = Variable(batch1["A"].type(Tensor))
#             cbct = Variable(batch1["B"].type(Tensor)) #condition

#             B_size = cbct.size(0)
#             pth = batch1["pth"]
#             noisyImage = torch.randn(size=[B_size, 1, 256, 256], device=device)
#             x_in = torch.cat((noisyImage,cbct),1)
#             x_out = sampler(x_in)
            
#             # “inverse the normalized image to HU value”
            
#             fake = x_out[:,0,:,:]

#             fake = fake.cpu().numpy().astype(np.float32)
#             for i in range(B_size):
#                 result_dict[pth[i]] = fake[i]
#         save_path = f"test/{out_name}/result_epoch{epoch}.pt"
#         torch.save(result_dict, save_path)

#             # fake = torch.unsqueeze(fake,1)

#             # # img = torch.cat((cbct,fake,ct),3)
#             # img = fake
#             # img = img.cpu()
#             # img_save = torch.cat((img_save,img),1)
#             # img_tst = img_save.numpy()
#             # img_tst = img_tst.astype(np.uint16)
#             # img_tst.tofile("test/%s/%s.raw" % (out_name, f"{epoch}"))

net_model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
ckpt = torch.load(os.path.join(save_weight_dir, f"ckpt_{epoch}_.pt"), map_location=device)
net_model.load_state_dict(ckpt)
print("model load weight done.")
net_model.eval()
# sampler = GaussianDiffusionSampler_cond(net_model,beta_1,beta_T,T).to(device)
test()
    
    
