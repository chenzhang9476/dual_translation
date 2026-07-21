# =========================
# file: train_dual_simple.py
# Dual diffusion training script (DDP compatible) - Deterministic version
# =========================
import os
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.factory import DualTranslationFactory
from models.diffusion_dual import visualize_low_high
# ============================================================
# Config
# ============================================================
def get_config():
    parser = argparse.ArgumentParser(description="Dual Frequency Diffusion Training (Deterministic)")
    # training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--modality", type=str, required=True, choices=["HN", "AB", "TH"])
    parser.add_argument("--task", type=str, default="MR", choices=["MR", "CBCT"])
    # diffusion
    parser.add_argument("--low_timesteps", type=int, default=800)
    parser.add_argument("--high_timesteps", type=int, default=50)
    parser.add_argument("--low_weight", type=float, default=0.7)
    parser.add_argument("--high_weight", type=float, default=0.3)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--high_scale", type=float, default=4.0)
    parser.add_argument("--high_inv_scale", type=float, default=1/4.0)
    # unets
    parser.add_argument("--low_base_channels", type=int, default=64)
    parser.add_argument("--high_base_channels", type=int, default=64)
    # wavelet
    parser.add_argument("--wavelet", type=str, default="haar")
    # paths
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--sample_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    # ddp
    parser.add_argument("--world_size", type=int, default=max(1, torch.cuda.device_count()))
    parser.add_argument("--ddp_port", type=str, default="12356")
    # misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    # pretrained/resume
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)

    # uncertainty analysis settings (for validation only)
    parser.add_argument("--uncertainty_samples", type=int, default=5,
                       help="Number of MC samples for uncertainty analysis in validation")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                       help="DDIM eta parameter for stochastic sampling (0 for deterministic)")
    args = parser.parse_args()
    if args.data_root is None:
        if args.task == "MR":
            args.data_root = f"./dataset/synthRAD2025_Task1_Train_process/{args.modality}"
        else:
            args.data_root = f"./dataset/synthRAD2025_Task2_Train_process/{args.modality}"
    if args.save_dir is None:
        args.save_dir = f"./checkpoints/{args.task}"
    if args.sample_dir is None:
        args.sample_dir = f"./results/samples/{args.task}/{args.modality}"
    if args.log_dir is None:
        args.log_dir = f"./results/logs/{args.task}/{args.modality}"
    for dir_path in [args.save_dir, args.sample_dir, args.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    config_path = os.path.join(args.log_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    return args
# ============================================================
# Data augmentation
# ============================================================
class DataAugmentation:
    @staticmethod
    def random_flip(mr, ct, mask, p=0.5):
        if random.random() < p:
            return mr.flip(-1), ct.flip(-1), mask.flip(-1)
        return mr, ct, mask
    @staticmethod
    def random_rotate(mr, ct, mask, p=0.3):
        if random.random() < p:
            k = random.randint(1, 3)
            return mr.rot90(k, [-2, -1]), ct.rot90(k, [-2, -1]), mask.rot90(k, [-2, -1])
        return mr, ct, mask
    @staticmethod
    def apply(mr, ct, mask):
        mr, ct, mask = DataAugmentation.random_flip(mr, ct, mask)
        mr, ct, mask = DataAugmentation.random_rotate(mr, ct, mask)
        return mr, ct, mask
# ============================================================
# Trainer
# ============================================================
class DualDiffusionTrainer:
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self._set_seed(config.seed + rank)
        self._init_models()
        self._init_optimizers()
        if config.pretrained is not None:
            self.load_pretrained(config.pretrained)
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def _init_models(self):
        self.model = DualTranslationFactory.build_model(self.config, self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=False)
    def _init_optimizers(self):
        params = list(self.model.module.parameters() if hasattr(self.model, "module") else self.model.parameters())
        self.optimizer = optim.AdamW(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay, betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-6)
        self.accumulation_counter = 0
    def _get_lr_scale(self):
        if self.global_step < self.config.warmup_steps:
            return min(1.0, float(self.global_step) / float(max(1, self.config.warmup_steps)))
        return 1.0
    def load_pretrained(self, pretrained_path):
        if pretrained_path is None or (not os.path.exists(pretrained_path)):
            print(f"[Rank {self.rank}] Pretrained not found: {pretrained_path}")
            return
        print(f"[Rank {self.rank}] Loading pretrained weights from {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        if hasattr(self.model, "module"):
            model_dict = self.model.module.state_dict()
        else:
            model_dict = self.model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_dict, strict=False)
        else:
            self.model.load_state_dict(model_dict, strict=False)
        print(f"[Rank {self.rank}] Loaded {len(filtered)}/{len(model_dict)} params")
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        if self.rank != 0:
            return
        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "config": vars(self.config),
        }
        ckpt_path = os.path.join(self.config.save_dir, f"checkpoint_epoch{epoch+1}_{self.config.modality}.pth")
        torch.save(ckpt, ckpt_path)
        latest_path = os.path.join(self.config.save_dir, f"latest_model_{self.config.modality}.pth")
        torch.save(ckpt, latest_path)
        if is_best:
            best_path = os.path.join(self.config.save_dir, f"best_model_{self.config.modality}.pth")
            torch.save(ckpt, best_path)
        print(f"[Rank {self.rank}] Saved checkpoint: {ckpt_path}")
    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"[Rank {self.rank}] Checkpoint not found: {checkpoint_path}")
            return False
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        pretrained = ckpt["model_state_dict"]
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained)
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_dict, strict=False)
        else:
            self.model.load_state_dict(model_dict, strict=False)
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[Rank {self.rank}] Optimizer load fail: {e}")
        try:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[Rank {self.rank}] Scheduler load fail: {e}")
        self.epoch = ckpt.get("epoch", 0) + 1
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[Rank {self.rank}] Loaded checkpoint: epoch={ckpt.get('epoch', 0)}")
        return True

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_low = 0.0
        epoch_high = 0.0
        num_batches = 0
        pbar = train_loader
        if self.rank == 0:
            try:
                from tqdm import tqdm
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            except Exception:
                pass
        for batch in pbar:
            mr = batch["image"].to(self.device, non_blocking=True)
            ct = batch["ct"].to(self.device, non_blocking=True)
            mask = batch.get("mask", None)
            if mask is None:
                mask = torch.ones_like(ct)
            mask = mask.to(self.device, non_blocking=True)
            if random.random() < 0.5:
                mr, ct, mask = DataAugmentation.apply(mr, ct, mask)
            # 简化：现在只返回损失
            loss_dict = self.model(ct, mr, mask)
           
            total_loss = loss_dict["total"]
            low_loss = loss_dict["low"]
            high_loss = loss_dict["high"]
            loss = total_loss / self.config.accumulation_steps
            loss.backward()
            self.accumulation_counter += 1
            if self.accumulation_counter == self.config.accumulation_steps:
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                if self.global_step < self.config.warmup_steps:
                    lr_scale = self._get_lr_scale()
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr_scale * self.config.lr
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.accumulation_counter = 0
                self.global_step += 1
            epoch_loss += float(total_loss.item())
            epoch_low += float(low_loss.item())
            epoch_high += float(high_loss.item())
            num_batches += 1
            if self.rank == 0 and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    low=f"{low_loss.item():.4f}",
                    high=f"{high_loss.item():.4f}",
                    lr=self.optimizer.param_groups[0]["lr"],
                )
        if num_batches > 0:
            epoch_loss /= num_batches
            epoch_low /= num_batches
            epoch_high /= num_batches
        if self.world_size > 1:
            losses = torch.tensor([epoch_loss, epoch_low, epoch_high], device=self.device)
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            epoch_loss, epoch_low, epoch_high = (losses / self.world_size).tolist()
        return epoch_loss, epoch_low, epoch_high
        
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0.0
        val_low = 0.0
        val_high = 0.0
        num_batches = 0
        sample_batch = None
        for batch_idx, batch in enumerate(val_loader):
            mr = batch["image"].to(self.device, non_blocking=True)
            ct = batch["ct"].to(self.device, non_blocking=True)
            mask = batch.get("mask", None)
            if mask is None:
                mask = torch.ones_like(ct)
            mask = mask.to(self.device, non_blocking=True)
            loss_dict = self.model(ct, mr, mask)
            total_loss = loss_dict["total"]
            low_loss = loss_dict["low"]
            high_loss = loss_dict["high"]
            val_loss += float(total_loss.item())
            val_low += float(low_loss.item())
            val_high += float(high_loss.item())
            num_batches += 1
            if batch_idx == 0 and self.rank == 0:
                sample_batch = (
                    mr[: self.config.num_samples],
                    ct[: self.config.num_samples],
                    mask[: self.config.num_samples],
                )
        if num_batches > 0:
            val_loss /= num_batches
            val_low /= num_batches
            val_high /= num_batches
        if self.world_size > 1:
            losses = torch.tensor([val_loss, val_low, val_high], device=self.device)
            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            val_loss, val_low, val_high = (losses / self.world_size).tolist()
        if self.rank == 0 and sample_batch is not None:
            self._save_sample_visualization(*sample_batch, epoch)
        return val_loss, val_low, val_high

    def _save_sample_visualization(self, mr, ct, mask, epoch):
        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model
        num_samples = min(self.config.num_samples, mr.size(0))
        # ---------- deterministic + GT ----------
        det = model.generate_sample_row(
            mr, ct, mask,
            device=self.device,
            n_uncertainty_samples=self.config.uncertainty_samples,
        )
        # ---------- uncertainty ----------
        unc = model.sample_with_uncertainty_analysis(
            mr,
            mask=mask,
            n_samples=self.config.uncertainty_samples,
            progress_bar=False,
            ddim_eta=self.config.ddim_eta,
            return_all_samples=False,
        )
        # ---------- layout: 18 columns (高频分开显示3通道) ----------
        fig, axes = plt.subplots(
            num_samples,
            15,
            figsize=(72, 4 * num_samples),
            dpi=100,
        )
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        def _to_gray(x, channel=0):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if x.ndim == 4: # [B, C, H, W]
                x = x[0, channel if x.shape[1] > 1 else 0]
            elif x.ndim == 3: # [C, H, W]
                x = x[channel if x.shape[0] > 1 else 0]
            return x
        for i in range(num_samples):
            # ---- 0 MRI ----
            mri_img = _to_gray(det["mri"][i:i+1])
            #print(f'MRI min: {mri_img.min()}, max: {mri_img.max()}')
            axes[i, 0].imshow(mri_img, cmap="gray", vmin=-1, vmax=1)
            axes[i, 0].set_title("MRI")
            axes[i, 0].axis("off")
            # ---- 1 mask ----
            mask_img = _to_gray(det["mask"][i:i+1])
            #print(f'Mask min: {mask_img.min()}, max: {mask_img.max()}')
            axes[i, 1].imshow(mask_img, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Mask")
            axes[i, 1].axis("off")
            # ---- 2 GT Low ----
            gt_low_img = _to_gray(det["gt_low"][i:i+1])
            #print(f'GT Low min: {gt_low_img.min()}, max: {gt_low_img.max()}')
            axes[i, 2].imshow(gt_low_img, cmap="gray", vmin=-1, vmax=1)
            axes[i, 2].set_title("GT Low")
            axes[i, 2].axis("off")
            # ---- 3-5 GT High LH, HL, HH ----
            for ch, title_suffix in zip([0,1,2], ["LH", "HL", "HH"]):
                img = _to_gray(det["gt_high"][i:i+1], channel=ch)
                abs_max = np.abs(img).max()
                vmin, vmax = (-abs_max, abs_max) if abs_max > 0 else (-1, 1)
                #print(f'GT High {title_suffix} min: {img.min()}, max: {img.max()}, abs_max: {abs_max}')
                axes[i, 3+ch].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                axes[i, 3+ch].set_title(f"GT High {title_suffix}")
                axes[i, 3+ch].axis("off")
            # ---- 6 Pred Low ----
            pred_low_img = _to_gray(det["pred_low"][i:i+1])
            #print(f'Pred Low min: {pred_low_img.min()}, max: {pred_low_img.max()}')
            axes[i, 6].imshow(pred_low_img, cmap="gray", vmin=-1, vmax=1)
            axes[i, 6].set_title("Pred Low")
            axes[i, 6].axis("off")
            # ---- 7-9 Pred High (det) LH, HL, HH ----
            for ch, title_suffix in zip([0,1,2], ["LH", "HL", "HH"]):
                img = _to_gray(det["pred_high_det"][i:i+1], channel=ch)
                abs_max = np.abs(img).max()
                vmin, vmax = (-abs_max, abs_max) if abs_max > 0 else (-1, 1)
                #print(f'Pred High Det {title_suffix} min: {img.min()}, max: {img.max()}, abs_max: {abs_max}')
                axes[i, 7+ch].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                axes[i, 7+ch].set_title(f"Pred High Det {title_suffix}")
                axes[i, 7+ch].axis("off")
            # ---- 10 CT Mean ----
            ct_mean_img = _to_gray(unc["ct_mean"][i:i+1])
            #print(f'CT MC Mean min: {ct_mean_img.min()}, max: {ct_mean_img.max()}')
            axes[i, 10].imshow(ct_mean_img, cmap="gray", vmin=-1, vmax=1)
            axes[i, 10].set_title("CT MC Mean")
            axes[i, 10].axis("off")
            # ---- 11 CT Std ----
            ct_variance = unc["ct_variance"][i:i+1]
            ct_std_img = _to_gray(torch.sqrt(ct_variance))
            #print(f'CT MC Std min: {ct_std_img.min()}, max: {ct_std_img.max()}')
            axes[i, 11].imshow(ct_std_img, cmap="hot", vmin=0, vmax=ct_std_img.max())
            axes[i, 11].set_title("CT MC Std")
            axes[i, 11].axis("off")
            # ---- 12-14 HF Mean LH, HL, HH ----
            for ch, title_suffix in zip([0,1,2], ["LH", "HL", "HH"]):
                img = _to_gray(unc["hf_mean"][i:i+1], channel=ch)
                abs_max = np.abs(img).max()
                vmin, vmax = (-abs_max, abs_max) if abs_max > 0 else (-1, 1)
                #print(f'HF MC Mean {title_suffix} min: {img.min()}, max: {img.max()}, abs_max: {abs_max}')
                axes[i, 12+ch].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
                axes[i, 12+ch].set_title(f"HF MC Mean {title_suffix}")
                axes[i, 12+ch].axis("off")
        plt.tight_layout(pad=3.0)
        plt.suptitle(
            f"Epoch {epoch+1} | DualDiffusion | MC={self.config.uncertainty_samples}",
            fontsize=16,
            y=1.02,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            self.config.sample_dir,
            f"samples_epoch{epoch+1}_{timestamp}.png"
        )
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[Rank {self.rank}] Saved samples: {save_path}")
  
    def train(self, train_loader, val_loader):
        start_epoch = 0
        if self.config.resume:
            if self.load_checkpoint(self.config.resume):
                start_epoch = self.epoch
        for epoch in range(start_epoch, self.config.epochs):
            self.epoch = epoch
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            tr_loss, tr_low, tr_high = self.train_epoch(train_loader, epoch)
            if self.rank == 0:
                print(f"\nEpoch {epoch+1}/{self.config.epochs}")
                print(f"Train: total={tr_loss:.6f} low={tr_low:.6f} high={tr_high:.6f}")
                print(f"LR: {self.optimizer.param_groups[0]['lr']:.6e}")
            val_loss = float("inf")
            if (epoch + 1) % self.config.eval_interval == 0 or epoch == (self.config.epochs - 1):
                val_loss, val_low, val_high = self.validate(val_loader, epoch)
                if self.rank == 0:
                    print(f"Val: total={val_loss:.6f} low={val_low:.6f} high={val_high:.6f}")
                    print(f"Best Val: {self.best_val_loss:.6f}")
                if self.rank == 0 and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)
            if (epoch + 1) % self.config.save_interval == 0 and self.rank == 0:
                self.save_checkpoint(epoch, val_loss, is_best=False)
            self.scheduler.step()
        if self.rank == 0:
            self.save_checkpoint(self.config.epochs - 1, self.best_val_loss, is_best=False)
# ============================================================
# DDP setup/cleanup
# ============================================================
def setup_ddp(rank, world_size, config):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", config.ddp_port)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    print(f"[Rank {rank}] DDP ready (world_size={world_size})", flush=True)
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
# ============================================================
# Worker
# ============================================================
def main_worker(rank, world_size, config):
    if world_size > 1:
        setup_ddp(rank, world_size, config)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    #print(f"[Rank {rank}] Loading dataset: {config.data_root}", flush=True)
    split_path = os.path.join(os.path.dirname(config.data_root), f"synthrad_split_{config.task}_{config.modality}.json")
    with open(split_path, "r") as f:
        split = json.load(f)
    train_set = DualTranslationFactory.build_dataset(
        modality=config.modality,
        root=config.data_root,
        case_list=split["train_cases"],
    )
    val_set = DualTranslationFactory.build_dataset(
        modality=config.modality,
        root=config.data_root,
        case_list=split["test_cases"],
    )
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=min(config.num_workers, 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    train_len = len(train_set)
    val_len = len(val_set)
    N = train_len + val_len
    print(f"[Rank {rank}] Dataset: total={N}, train={train_len}, test={val_len}", flush=True)
    trainer = DualDiffusionTrainer(config, rank, world_size)
    try:
        trainer.train(train_loader, val_loader)
    finally:
        if world_size > 1:
            cleanup_ddp()
# ============================================================
# Main
# ============================================================
def main():
    config = get_config()
    # torchrun mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"[torchrun] rank={rank} world_size={world_size}", flush=True)
        main_worker(rank, world_size, config)
        return
    # single / mp.spawn
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = config.ddp_port
    if config.world_size == 1:
        print("Single GPU training", flush=True)
        main_worker(0, 1, config)
    else:
        print(f"DDP training with {config.world_size} GPUs", flush=True)
        mp.spawn(main_worker, args=(config.world_size, config), nprocs=config.world_size, join=True)
if __name__ == "__main__":
    main()
