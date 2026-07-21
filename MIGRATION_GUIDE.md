# New Server Setup Guide

This repository contains source code only. Data, checkpoints, and generated
results must be restored separately from approved storage.

## 1. Clone the repository

```bash
mkdir -p /workspace/projects
cd /workspace/projects
git clone https://github.com/chenzhang9476/dual_translation.git
cd dual_translation
```

## 2. Create the environment

Docker workflow:

```bash
docker compose -f config/docker-compose.gpu.yml build
docker compose -f config/docker-compose.gpu.yml run --rm dual-translation
```

Native Python workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm CUDA and imports:

```bash
python -c "import torch, monai, nibabel; print(torch.__version__, torch.cuda.is_available())"
python -m compileall -q core dataset models train_dual.py inference.py inference_3d.py
```

## 3. Restore data

Place SynthRAD data under `dataset/` without adding another nested
`dual_translation/dual_translation` level. The final tree should contain:

```text
dataset/
  synthRAD2025_Task1_Train/
  synthRAD2025_Task1_Train_process/
  synthRAD2025_Task2_Train/
  synthRAD2025_Task2_Train_process/
```

Processed task directories must retain the `AB`, `HN`, and `TH` subdirectories
and their `slice_index_*.json` and `synthrad_split_*.json` files.

Path conventions:

- Inference `--data_root` expects the processed task root.
- Training `--data_root` expects the final modality directory.

## 4. Restore checkpoints when needed

Copy the backed-up checkpoint directory into the repository root so that, for
example, MR/HN resolves to:

```text
checkpoints/MR/best_model_HN.pth
```

Checkpoints are deliberately ignored by Git because each file is much larger
than GitHub's normal file limit.

## 5. Verify the installation

Code only:

```bash
python tools/verify_release.py
```

Code, data, and checkpoints:

```bash
python tools/verify_release.py --require-data --require-checkpoints
```

## 6. Run a smoke test

Inference with a restored checkpoint:

```bash
python inference_3d.py \
  --task MR \
  --modality HN \
  --checkpoint checkpoints/MR/best_model_HN.pth
```

Training:

```bash
python train_dual.py --task MR --modality HN
```

Generated checkpoints, results, logs, medical-image files, and local
environments are covered by `.gitignore` and should stay outside version
control.
