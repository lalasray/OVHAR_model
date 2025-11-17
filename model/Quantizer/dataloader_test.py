"""
Quick smoke test for OVHARDataset/loader to verify windowing and labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from imu_OVHAR import OVHARDataset


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset = script_dir.parent.parent / "dataset"
    parser = argparse.ArgumentParser(description="Test OVHARDataset windowing and labels.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset)
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-files", type=int, help="Limit number of Sensor*.csv files to load.")
    parser.add_argument("--batches", type=int, default=2, help="How many batches to iterate/print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = OVHARDataset(
        root=args.dataset_root,
        seq_length=args.seq_length,
        stride=args.stride,
        max_files=args.max_files,
    )
    print(f"Total windows: {len(ds)}")
    print(f"Label vocab size: {len(ds.id_to_label)} (unknown label id: {ds.unknown_label})")
    if ds.id_to_label:
        for lid, text in enumerate(ds.id_to_label):
            print(f"{lid}: {text}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    label_counts = Counter(ds.window_labels)
    print("Label distribution (top 10):", label_counts.most_common(10))

    for i, (batch, labels) in enumerate(loader):
        print(f"Batch {i}: data shape {batch.shape}, labels shape {labels.shape}")
        uniques = labels.unique()
        print(f"  Unique labels in batch: {uniques.tolist()}")
        if i + 1 >= args.batches:
            break


if __name__ == "__main__":
    main()
