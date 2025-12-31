#!/usr/bin/env python3
"""Analyze the CATH dataset to understand filtering/pruning."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


def iter_chain_set_jsonl(path: Path):
    """Iterate over samples in chain_set.jsonl."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def has_nan_coords(sample: dict) -> bool:
    """Check if sample has any NaN in CA coordinates."""
    ca_coords = np.array(sample["coords"]["CA"], dtype=np.float32)
    return np.isnan(ca_coords).any()


def main():
    data_path = Path("data/chain_set.jsonl")
    splits_path = Path("data/chain_set_splits.json")

    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    train_ids = set(splits["train"])
    val_ids = set(splits["validation"])
    test_ids = set(splits["test"])
    all_split_ids = train_ids | val_ids | test_ids

    print(f"Split file counts:")
    print(f"  Train:      {len(train_ids):,}")
    print(f"  Validation: {len(val_ids):,}")
    print(f"  Test:       {len(test_ids):,}")
    print(f"  Total:      {len(all_split_ids):,}")
    print()

    # Analyze dataset
    lengths = []
    has_nans = []
    in_train = []
    in_val = []
    in_test = []
    not_in_splits = []

    print("Scanning dataset...")
    for i, sample in enumerate(iter_chain_set_jsonl(data_path)):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,}...")

        name = sample["name"]
        seq_len = len(sample["seq"])
        nan_flag = has_nan_coords(sample)

        lengths.append(seq_len)
        has_nans.append(nan_flag)

        if name in train_ids:
            in_train.append((seq_len, nan_flag))
        elif name in val_ids:
            in_val.append((seq_len, nan_flag))
        elif name in test_ids:
            in_test.append((seq_len, nan_flag))
        else:
            not_in_splits.append(name)

    total = len(lengths)
    lengths = np.array(lengths)
    has_nans = np.array(has_nans)

    print(f"\n{'='*60}")
    print("OVERALL DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total chains in jsonl:     {total:,}")
    print(f"Chains with NaN coords:    {has_nans.sum():,} ({100*has_nans.mean():.1f}%)")
    print(f"Chains NOT in any split:   {len(not_in_splits):,}")
    print()

    print(f"Length distribution (all chains):")
    print(f"  Min:    {lengths.min()}")
    print(f"  Max:    {lengths.max()}")
    print(f"  Mean:   {lengths.mean():.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print()

    # Length buckets
    buckets = [
        (0, 40, "< 40 (too short)"),
        (40, 100, "40-99"),
        (100, 200, "100-199"),
        (200, 256, "200-255"),
        (256, 300, "256-299"),
        (300, 400, "300-399"),
        (400, 512, "400-511"),
        (512, 10000, ">= 512 (too long)"),
    ]

    print("Length distribution by bucket:")
    for lo, hi, label in buckets:
        count = ((lengths >= lo) & (lengths < hi)).sum()
        pct = 100 * count / total
        print(f"  {label:20s}: {count:6,} ({pct:5.1f}%)")
    print()

    # Analyze each split
    def analyze_split(split_data, split_name, min_len=40, max_len=256):
        """Analyze filtering for a split."""
        total_in_split = len(split_data)
        if total_in_split == 0:
            print(f"{split_name}: No chains found in dataset")
            return

        lens = np.array([x[0] for x in split_data])
        nans = np.array([x[1] for x in split_data])

        in_range = (lens >= min_len) & (lens < max_len)
        no_nans = ~nans
        passes_all = in_range & no_nans

        print(f"{split_name} (len {min_len}-{max_len}):")
        print(f"  In dataset:           {total_in_split:,}")
        print(f"  Pass length filter:   {in_range.sum():,} ({100*in_range.mean():.1f}%)")
        print(f"  Have no NaNs:         {no_nans.sum():,} ({100*no_nans.mean():.1f}%)")
        print(f"  Pass both filters:    {passes_all.sum():,} ({100*passes_all.mean():.1f}%)")

        # Breakdown of failures
        too_short = lens < min_len
        too_long = lens >= max_len
        print(f"  Rejected - too short: {too_short.sum():,}")
        print(f"  Rejected - too long:  {too_long.sum():,}")
        print(f"  Rejected - has NaNs:  {(in_range & nans).sum():,} (in length range but has NaNs)")
        print()

    print(f"{'='*60}")
    print("SPLIT-BY-SPLIT ANALYSIS")
    print(f"{'='*60}")
    analyze_split(in_train, "Train", 40, 256)
    analyze_split(in_val, "Validation", 40, 256)
    analyze_split(in_test, "Test", 40, 256)

    # Check for missing IDs
    found_train = len(in_train)
    found_val = len(in_val)
    found_test = len(in_test)

    print(f"{'='*60}")
    print("SPLIT ID MATCHING")
    print(f"{'='*60}")
    print(f"Train IDs in splits file:      {len(train_ids):,}")
    print(f"Train IDs found in dataset:    {found_train:,}")
    print(f"Train IDs missing from dataset: {len(train_ids) - found_train:,}")
    print()
    print(f"Val IDs in splits file:        {len(val_ids):,}")
    print(f"Val IDs found in dataset:      {found_val:,}")
    print(f"Val IDs missing from dataset:  {len(val_ids) - found_val:,}")
    print()
    print(f"Test IDs in splits file:       {len(test_ids):,}")
    print(f"Test IDs found in dataset:     {found_test:,}")
    print(f"Test IDs missing from dataset: {len(test_ids) - found_test:,}")


if __name__ == "__main__":
    main()
