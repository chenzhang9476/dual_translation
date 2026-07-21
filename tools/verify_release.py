#!/usr/bin/env python3
"""Validate a migrated DualTranslation release layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CODE_PATHS = [
    "train_dual.py",
    "inference.py",
    "inference_3d.py",
    "core/factory.py",
    "models/diffusion_dual.py",
    "models/unet_simple.py",
    "models/unet_complex.py",
    "dataset/SynthRAD_HN.py",
    "dataset/SynthRAD_AB.py",
    "dataset/SynthRAD_TH.py",
]

CHECKPOINT_PATHS = [
    "checkpoints/MR/best_model_AB.pth",
    "checkpoints/MR/best_model_HN.pth",
    "checkpoints/MR/best_model_TH.pth",
    "checkpoints/CBCT/best_model_HN.pth",
]

DATA_PATHS = [
    "dataset/synthRAD2025_Task1_Train",
    "dataset/synthRAD2025_Task1_Train_process",
    "dataset/synthRAD2025_Task2_Train",
    "dataset/synthRAD2025_Task2_Train_process",
]


def check_paths(paths: list[str], label: str) -> list[str]:
    missing = [path for path in paths if not (PROJECT_ROOT / path).exists()]
    status = "OK" if not missing else "MISSING"
    print(f"[{status}] {label}: {len(paths) - len(missing)}/{len(paths)} present")
    for path in missing:
        print(f"  - {path}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()

    failures = check_paths(CODE_PATHS, "core code")
    if args.require_checkpoints:
        failures.extend(check_paths(CHECKPOINT_PATHS, "best checkpoints"))
    if args.require_data:
        failures.extend(check_paths(DATA_PATHS, "raw and processed datasets"))

    if failures:
        print("\nRelease verification failed.")
        return 1

    print("\nRelease layout is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
