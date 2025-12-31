#!/usr/bin/env python3
"""Analyze potential data recovery using contiguous resolved segment extraction.

Instead of rejecting chains with any NaN, extract contiguous segments where
CA coordinates are valid. This recovers data from:
1. Chains with missing loops/termini (common in crystallography)
2. Long chains that contain multiple valid segments within length limits
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def iter_chain_set_jsonl(path: Path):
    """Iterate over samples in chain_set.jsonl."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def get_ca_mask(sample: dict) -> np.ndarray:
    """Get boolean mask for valid (non-NaN) CA coordinates."""
    ca_coords = np.array(sample["coords"]["CA"], dtype=np.float32)
    return np.isfinite(ca_coords).all(axis=-1)  # (L,)


def find_contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in a boolean mask.

    Returns:
        List of (start, end) tuples (end is exclusive)
    """
    segments = []
    in_segment = False
    start = 0

    for i, valid in enumerate(mask):
        if valid and not in_segment:
            # Start of new segment
            in_segment = True
            start = i
        elif not valid and in_segment:
            # End of segment
            in_segment = False
            segments.append((start, i))

    # Handle segment that extends to end
    if in_segment:
        segments.append((start, len(mask)))

    return segments


def main():
    data_path = Path("data/chain_set.jsonl")
    splits_path = Path("data/chain_set_splits.json")

    min_len = 40
    max_len = 256

    # Load splits
    with open(splits_path) as f:
        splits = json.load(f)

    train_ids = set(splits["train"])
    val_ids = set(splits["validation"])
    test_ids = set(splits["test"])

    # Track statistics
    stats = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }

    # Track segment details
    all_segments = {"train": [], "val": [], "test": []}

    # For comparison with current approach
    current_approach = {"train": 0, "val": 0, "test": 0}

    print("Scanning dataset for contiguous resolved segments...")
    print(f"Length criteria: {min_len} <= len < {max_len}")
    print()

    for i, sample in enumerate(iter_chain_set_jsonl(data_path)):
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,}...")

        name = sample["name"]

        # Determine split
        if name in train_ids:
            split = "train"
        elif name in val_ids:
            split = "val"
        elif name in test_ids:
            split = "test"
        else:
            continue  # Not in any split

        seq_len = len(sample["seq"])
        ca_mask = get_ca_mask(sample)

        stats[split]["total_chains"] += 1

        # Current approach: reject if any NaN or wrong length
        has_no_nans = ca_mask.all()
        in_length_range = min_len <= seq_len < max_len
        if has_no_nans and in_length_range:
            current_approach[split] += 1

        # New approach: find contiguous segments
        segments = find_contiguous_segments(ca_mask)

        # Filter segments by length
        valid_segments = [(s, e) for s, e in segments if min_len <= (e - s) < max_len]

        if valid_segments:
            stats[split]["chains_with_valid_segments"] += 1
            stats[split]["total_valid_segments"] += len(valid_segments)

            # Track segment lengths
            for start, end in valid_segments:
                seg_len = end - start
                all_segments[split].append(seg_len)


        # Detailed breakdown
        if not valid_segments:
            if not segments:
                stats[split]["no_valid_residues"] += 1
            else:
                longest = max(e - s for s, e in segments)
                if longest < min_len:
                    stats[split]["all_segments_too_short"] += 1
                else:
                    stats[split]["all_segments_too_long"] += 1

    print()
    print("=" * 70)
    print("COMPARISON: CURRENT vs SEGMENT EXTRACTION")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        s = stats[split]
        total = s["total_chains"]
        current = current_approach[split]

        # Policy A: Keep only longest valid segment per chain
        chains_with_segments = s["chains_with_valid_segments"]

        # Policy B: Keep all valid segments
        total_segments = s["total_valid_segments"]

        print(f"\n{split.upper()}:")
        print(f"  Total chains in split:              {total:,}")
        print(f"  Current approach (no NaN + length): {current:,} ({100*current/total:.1f}%)")
        print(f"  ")
        print(f"  Segment extraction:")
        print(f"    Policy A (longest per chain):     {chains_with_segments:,} ({100*chains_with_segments/total:.1f}%)")
        print(f"    Policy B (all valid segments):    {total_segments:,}")
        print(f"  ")
        print(f"  Improvement over current:")
        print(f"    Policy A: +{chains_with_segments - current:,} chains ({100*(chains_with_segments/current - 1):.1f}% more)" if current > 0 else "    Policy A: N/A")
        print(f"    Policy B: +{total_segments - current:,} samples ({100*(total_segments/current - 1):.1f}% more)" if current > 0 else "    Policy B: N/A")

    print()
    print("=" * 70)
    print("SEGMENT LENGTH DISTRIBUTION")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        segs = all_segments[split]
        if segs:
            segs = np.array(segs)
            print(f"\n{split.upper()} valid segments ({len(segs):,} total):")
            print(f"  Min:    {segs.min()}")
            print(f"  Max:    {segs.max()}")
            print(f"  Mean:   {segs.mean():.1f}")
            print(f"  Median: {np.median(segs):.1f}")

            # Length buckets
            buckets = [(40, 80), (80, 120), (120, 160), (160, 200), (200, 256)]
            print(f"  Distribution:")
            for lo, hi in buckets:
                count = ((segs >= lo) & (segs < hi)).sum()
                print(f"    {lo:3d}-{hi-1:3d}: {count:5,} ({100*count/len(segs):5.1f}%)")

    print()
    print("=" * 70)
    print("WHY CHAINS FAIL (no valid segments)")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        s = stats[split]
        total = s["total_chains"]
        no_segs = total - s["chains_with_valid_segments"]
        print(f"\n{split.upper()}: {no_segs:,} chains with no valid segments")
        print(f"  No valid residues at all:    {s.get('no_valid_residues', 0):,}")
        print(f"  All segments too short (<{min_len}): {s.get('all_segments_too_short', 0):,}")
        print(f"  All segments too long (>={max_len}): {s.get('all_segments_too_long', 0):,}")


if __name__ == "__main__":
    main()
