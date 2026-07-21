# ============================================================
# file: build_split.py
# Build case-level train/test split for SynthRAD (task + organ)
# ============================================================
import os
import json
import argparse
import random
from pathlib import Path

from SynthRAD_HN import SynthRAD_HN_Dataset
from SynthRAD_TH import SynthRAD_TH_Dataset
from SynthRAD_AB import SynthRAD_AB_Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build case-level train/test split (0.8 / 0.2) for SynthRAD"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of SynthRAD dataset (e.g. ./synthRAD2025_Task1_Train_process/HN)"
    )
    parser.add_argument(
        "--index_json",
        type=str,
        required=True,
        help="Index json file name (e.g. slice_index_HN.json)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., CBCT, MR)"
    )
    parser.add_argument(
        "--organs",
        type=str,
        required=True,
        help="Organ name (e.g., HN, TH, AB)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load dataset by task + organ
    # ------------------------------------------------------------
    if args.organs == 'HN':
        dataset = SynthRAD_HN_Dataset(
            root=args.data_root,
            index_json=args.index_json,
            case_list=None
        )
    elif args.organs == 'TH':
        dataset = SynthRAD_TH_Dataset(
            root=args.data_root,
            index_json=args.index_json,
            case_list=None
        )
    elif args.organs == 'AB':
        dataset = SynthRAD_AB_Dataset(
            root=args.data_root,
            index_json=args.index_json,
            case_list=None
        )
    else:
        raise ValueError(f"Unsupported organ: {args.organs}")

    # ------------------------------------------------------------
    # Collect unique case ids
    # ------------------------------------------------------------
    case_ids = dataset.get_all_case_ids()
    if len(case_ids) == 0:
        raise RuntimeError("No patient_id found in dataset!")

    print(f"Total cases for {args.task}-{args.organs}: {len(case_ids)}")

    # ------------------------------------------------------------
    # Shuffle & split
    # ------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(case_ids)

    n_train = int(len(case_ids) * args.train_ratio)
    train_cases = case_ids[:n_train]
    test_cases = case_ids[n_train:]

    # ------------------------------------------------------------
    # Save split file (task + organ specific)
    # ------------------------------------------------------------
    split = {
        "task": args.task,
        "organ": args.organs,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "num_train_cases": len(train_cases),
        "num_test_cases": len(test_cases),
        "train_cases": train_cases,
        "test_cases": test_cases,
    }

    save_name = f"synthrad_split_{args.task}_{args.organs}.json"
    save_path = Path(os.path.dirname(args.data_root)) / save_name

    with open(save_path, "w") as f:
        json.dump(split, f, indent=2)

    # ------------------------------------------------------------
    # Log
    # ------------------------------------------------------------
    print(f"\n✅ Split file saved to: {save_path}")
    print(f"Train cases ({len(train_cases)}):")
    print(train_cases[:10], "..." if len(train_cases) > 10 else "")
    print(f"Test cases  ({len(test_cases)}):")
    print(test_cases[:10], "..." if len(test_cases) > 10 else "")


if __name__ == "__main__":
    main()
