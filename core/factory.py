"""Centralized dataset and model construction for training and inference."""

from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from dataset.SynthRAD_AB import SynthRAD_AB_Dataset
from dataset.SynthRAD_HN import SynthRAD_HN_Dataset
from dataset.SynthRAD_TH import SynthRAD_TH_Dataset
from models import DualDiffusion, HighFreqUNet, LowFreqUNet


class DualTranslationFactory:
    """Build datasets and DualDiffusion models from a shared configuration."""

    _DATASETS = {
        "ab": SynthRAD_AB_Dataset,
        "hn": SynthRAD_HN_Dataset,
        "th": SynthRAD_TH_Dataset,
    }

    @classmethod
    def build_dataset(
        cls,
        modality: str,
        root: str,
        case_list: Optional[Iterable[str]] = None,
        **dataset_kwargs: Any,
    ):
        key = modality.lower()
        if key not in cls._DATASETS:
            supported = ", ".join(sorted(cls._DATASETS))
            raise ValueError(f"Unknown modality '{modality}'. Expected one of: {supported}")
        return cls._DATASETS[key](
            root=str(Path(root)),
            case_list=case_list,
            **dataset_kwargs,
        )

    @staticmethod
    def build_model(config: Any, device: torch.device) -> DualDiffusion:
        low_unet = LowFreqUNet(
            in_channels=1,
            out_channels=1,
            base_channels=getattr(config, "low_base_channels", 64),
            time_emb_dim=256,
        ).to(device)
        high_unet = HighFreqUNet(
            in_channels=3,
            out_channels=3,
            base_channels=getattr(config, "high_base_channels", 64),
            time_emb_dim=128,
        ).to(device)

        return DualDiffusion(
            low_freq_unet=low_unet,
            high_freq_unet=high_unet,
            wavelet=config.wavelet,
            low_timesteps=config.low_timesteps,
            high_timesteps=config.high_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            low_weight=config.low_weight,
            high_weight=config.high_weight,
            high_scale=config.high_scale,
            high_inv_scale=config.high_inv_scale,
            device=device,
        ).to(device)

    @classmethod
    def load_model(
        cls,
        config: Any,
        device: torch.device,
        checkpoint_path: Optional[str] = None,
        strict: bool = True,
    ) -> DualDiffusion:
        path = Path(checkpoint_path or config.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model = cls.build_model(config, device)
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=strict)
        return model
