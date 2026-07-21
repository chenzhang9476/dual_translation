# DualTranslation

DualTranslation is a dual-frequency diffusion pipeline for medical image
translation on SynthRAD data. This repository contains the maintained training,
2D inference, 3D reconstruction, preprocessing, and baseline source code.

> This is a source-only repository. Patient data, model checkpoints, experiment
> outputs, and logs are intentionally excluded from Git. Keep those artifacts in
> external storage and place them locally only when running an experiment.

## Main entry points

| File | Purpose |
|---|---|
| `train_dual.py` | Train the dual-frequency diffusion model for one task and anatomical region. Supports one GPU, `torchrun`, or local multi-GPU spawning. |
| `inference.py` | Run slice-wise 2D inference and PSNR/SSIM/MAE evaluation. |
| `inference_3d.py` | Reconstruct volumes from slice predictions and report 3D metrics. |
| `core/factory.py` | Build datasets and models consistently for all entry points. |
| `tools/verify_release.py` | Validate the local code, data, and checkpoint layout. |
| `MIGRATION_GUIDE.md` | Set up a clone on a new workstation or server. |

## Repository layout

```text
dual_translation/
  baselines/          # Source-only comparison methods
  config/             # Docker and dependency configuration
  core/               # Shared model and dataset construction
  dataset/            # Dataset loaders and split utilities
  models/             # DualDiffusion and UNet implementations
  tools/              # Validation and SynthRAD preprocessing tools
  train_dual.py
  inference.py
  inference_3d.py
```

## Environment

Linux with an NVIDIA GPU is recommended. The pinned stack uses PyTorch 2.4.1
and CUDA 12.1 in the provided Docker image.

Native Python setup:

```bash
git clone https://github.com/chenzhang9476/dual_translation.git
cd dual_translation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/verify_release.py
```

Docker setup:

```bash
docker compose -f config/docker-compose.gpu.yml build
docker compose -f config/docker-compose.gpu.yml run --rm dual-translation
```

## Data layout

Obtain SynthRAD data through its official distribution and preprocessing
requirements. Place raw and processed data under `dataset/` as follows:

```text
dataset/
  synthRAD2025_Task1_Train/
  synthRAD2025_Task1_Train_process/
    AB/
    HN/
    TH/
    slice_index_*.json
    synthrad_split_*.json
  synthRAD2025_Task2_Train/
  synthRAD2025_Task2_Train_process/
    AB/
    HN/
    TH/
    slice_index_*.json
    synthrad_split_*.json
```

Task mapping used by the code:

- `MR` uses `synthRAD2025_Task1_Train_process`.
- `CBCT` uses `synthRAD2025_Task2_Train_process`.
- Modalities are `HN`, `AB`, and `TH`.

Validate the full local layout after adding data:

```bash
python tools/verify_release.py --require-data
```

## Training

Default data, checkpoint, sample, and log paths are inferred from `--task` and
`--modality`.

```bash
# MR-to-CT, head and neck
python train_dual.py --task MR --modality HN

# CBCT-to-CT, head and neck
python train_dual.py --task CBCT --modality HN

# Distributed training on two GPUs
torchrun --standalone --nproc_per_node=2 train_dual.py \
  --task MR \
  --modality AB \
  --world_size 2
```

Only task/modality combinations with valid processed data and split JSON files
can run.

## Inference

Model weights are not committed to this repository. Train a model or copy the
backed-up checkpoints into:

```text
checkpoints/MR/best_model_AB.pth
checkpoints/MR/best_model_HN.pth
checkpoints/MR/best_model_TH.pth
checkpoints/CBCT/best_model_HN.pth
```

Then run:

```bash
python tools/verify_release.py --require-checkpoints
python inference.py --task MR --modality HN --save_vis
python inference_3d.py --task MR --modality HN
```

Generated files are written below `results/` and remain untracked.

## Baselines

The `baselines/` directory contains source snapshots of comparison methods.
Each baseline keeps its own README and upstream license. Check those files for
method-specific setup and attribution requirements.

## Reproducibility notes

- Do not commit patient data, model weights, generated images, or logs.
- Keep split and slice-index JSON files beside the corresponding processed task
  directory.
- Use explicit `--data_root`, `--checkpoint`, and `--output_dir` arguments when
  running outside the default layout.
- See `MIGRATION_GUIDE.md` for a complete new-server checklist.

## License

No project-level license has been declared. Third-party code under `baselines/`
is governed by the license included with each baseline.
