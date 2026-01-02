#!/usr/bin/env python3
"""
Quick check of dataset size for different data loading parameters.

Usage:
    uv run python scripts/check_data_size.py --max_len 256 --min_len 64 --stride 64
    uv run python scripts/check_data_size.py --max_len 128 --min_len 40  # segments mode
"""

import argparse
from pathlib import Path

from src.data_cath import (
    load_chain_segments_by_ids,
    load_chain_windows_by_ids,
    load_splits,
)


def main():
    parser = argparse.ArgumentParser(description="Check dataset size with different params")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--min_len", type=int, default=40, help="Min sequence length")
    parser.add_argument("--stride", type=int, default=64, help="Stride for sliding windows")
    parser.add_argument(
        "--use_sliding_windows", action="store_true",
        help="Use sliding windows (default: segments mode)"
    )
    parser.add_argument(
        "--keep_longest_only", action="store_true", default=True,
        help="Keep only longest segment per chain"
    )
    parser.add_argument(
        "--no_keep_longest_only", action="store_false", dest="keep_longest_only",
        help="Keep all valid segments per chain"
    )
    args = parser.parse_args()

    data_path = Path("data/chain_set.jsonl")
    splits_path = Path("data/chain_set_splits.json")

    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return

    print("=" * 60)
    print("Dataset Size Check")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  max_len:           {args.max_len}")
    print(f"  min_len:           {args.min_len}")
    print(f"  stride:            {args.stride}")
    print(f"  sliding_windows:   {args.use_sliding_windows}")
    print(f"  keep_longest_only: {args.keep_longest_only}")

    # Load splits
    splits = load_splits(splits_path)
    print(f"\nSplit sizes (chain IDs):")
    print(f"  train: {len(splits['train'])}")
    print(f"  val:   {len(splits['validation'])}")
    print(f"  test:  {len(splits['test'])}")

    print("\n" + "-" * 60)

    if args.use_sliding_windows:
        print(f"Loading with SLIDING WINDOWS (window={args.max_len}, stride={args.stride})...")

        print("\nTrain set:")
        train_chains = load_chain_windows_by_ids(
            data_path,
            chain_ids=splits["train"],
            window_size=args.max_len,
            stride=args.stride,
            min_len=args.min_len,
            keep_longest_only=args.keep_longest_only,
            verbose=True,
        )

        print("\nValidation set:")
        val_chains = load_chain_windows_by_ids(
            data_path,
            chain_ids=splits["validation"],
            window_size=args.max_len,
            stride=args.stride,
            min_len=args.min_len,
            keep_longest_only=args.keep_longest_only,
            verbose=True,
        )

        data_type = "windows"
    else:
        print(f"Loading with SEGMENTS (len {args.min_len}-{args.max_len})...")

        print("\nTrain set:")
        train_chains = load_chain_segments_by_ids(
            data_path,
            chain_ids=splits["train"],
            min_len=args.min_len,
            max_len=args.max_len,
            keep_longest_only=args.keep_longest_only,
            verbose=True,
        )

        print("\nValidation set:")
        val_chains = load_chain_segments_by_ids(
            data_path,
            chain_ids=splits["validation"],
            min_len=args.min_len,
            max_len=args.max_len,
            keep_longest_only=args.keep_longest_only,
            verbose=True,
        )

        data_type = "segments"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Train: {len(train_chains):,} {data_type}")
    print(f"  Val:   {len(val_chains):,} {data_type}")
    print(f"  Total: {len(train_chains) + len(val_chains):,} {data_type}")

    # Length distribution
    if train_chains:
        train_lens = [len(c["coords"]["CA"]) for c in train_chains]
        print(f"\nTrain length distribution:")
        print(f"  min:  {min(train_lens)}")
        print(f"  max:  {max(train_lens)}")
        print(f"  mean: {sum(train_lens) / len(train_lens):.1f}")


if __name__ == "__main__":
    main()
