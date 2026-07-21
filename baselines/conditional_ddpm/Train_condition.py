# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import os
from typing import Dict
import time
import datetime
import sys
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets #V
from torchvision.utils import save_image
from torch.autograd import Variable #V

from Diffusion_condition import GaussianDiffusionTrainer_cond
from Model_condition import UNet
from datasets import * #V

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
out_name=f"task{dataset_name}"
batch_size = 16#original:2
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3
lr = 1e-4
n_epochs = 1000#original:1000
beta_1 = 1e-4
beta_T = 0.02
grad_clip = 1
save_weight_dir = "./Checkpoints/%s"%out_name

Tensor = torch.cuda.FloatTensor
device = torch.device("cuda")

os.makedirs("%s" % save_weight_dir, exist_ok=True)
train_dataloader = DataLoader(
    ImageDataset(dataset_name,debug=debug, transforms_=False, unaligned=True),
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
)
#success
net_model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout).to(device)
optimizer = torch.optim.AdamW(net_model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
trainer = GaussianDiffusionTrainer_cond(net_model, beta_1, beta_T, T).to(device)

prev_time = time.time()
for epoch in range(n_epochs):
    print(f"Epoch:{epoch} start")
    epoch = epoch + 1
    loss_save = 0
    for i, batch in (enumerate(tqdm(train_dataloader))):
        i = i + 1
        optimizer.zero_grad()
        ct = Variable(batch["A"].type(Tensor))
        cbct = Variable(batch["B"].type(Tensor)) #condition
        x_0 = torch.cat((ct,cbct),1)
        loss = trainer(x_0)
        loss_save = loss_save + loss/65536
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_model.parameters(), grad_clip)
        optimizer.step()
    loss_save = loss_save / i

    time_duration = datetime.timedelta(seconds=(time.time() - prev_time))
    epoch_left = n_epochs - epoch
    time_left = datetime.timedelta(seconds=epoch_left * (time.time() - prev_time))
    prev_time = time.time()
    if epoch > 99 and epoch % 10 == 0:
        torch.save(net_model.state_dict(), os.path.join(save_weight_dir, 'ckpt_' + str(epoch) + "_.pt"))
    sys.stdout.write(
        "\r[Epoch %d/%d] [ETA: %s] [EpochDuration: %s] [MSELoss: %s]"
        % (
            epoch,
            n_epochs,
            time_left,
            time_duration,
            loss_save.item(),
        )
    )
    
