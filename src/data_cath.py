"""CATH dataset loading utilities for protein structure data.

Usage:
    uv run python -m src.data_cath                     # default path
    uv run python -m src.data_cath data/chain_set.jsonl  # explicit path
"""

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def iter_chain_set_jsonl(path: Path) -> Iterator[dict]:
    """Stream records from a JSONL file one at a time.

    Args:
        path: Path to chain_set.jsonl file

    Yields:
        dict: Parsed JSON record for each protein chain
    """
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    """Load records from a JSONL file.

    Args:
        path: Path to chain_set.jsonl file
        limit: Maximum number of records to load (None for all)

    Returns:
        List of parsed JSON records
    """
    records = []
    for i, record in enumerate(iter_chain_set_jsonl(path)):
        if limit is not None and i >= limit:
            break
        records.append(record)
    return records


def process_coords(sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract coordinates and create a validity mask.

    Args:
        sample: Dict with 'seq' and 'coords' keys

    Returns:
        xyz: (L, 4, 3) tensor of backbone atom coordinates (N, CA, C, O)
        mask: (L,) boolean tensor, True where CA atom is resolved
    """
    coords_dict = sample["coords"]
    L = len(sample["seq"])

    # Stack coordinates: (L, 4, 3)
    xyz = np.full((L, 4, 3), np.nan, dtype=np.float32)
    for i, atom_name in enumerate(["N", "CA", "C", "O"]):
        if atom_name in coords_dict:
            xyz[:, i, :] = coords_dict[atom_name]

    # Mask based on CA atom (alpha carbon)
    ca_coords = xyz[:, 1, :]  # (L, 3)
    mask = np.isfinite(ca_coords).all(axis=-1)  # (L,)

    return torch.from_numpy(xyz), torch.from_numpy(mask)


def get_ca_coords(sample: dict) -> np.ndarray:
    """Extract CA (alpha carbon) coordinates from a sample.

    Args:
        sample: Dict with 'coords' key containing atom coordinates

    Returns:
        (L, 3) array of CA coordinates
    """
    return np.array(sample["coords"]["CA"], dtype=np.float32)


def find_contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in a boolean mask.

    Args:
        mask: 1D boolean array

    Returns:
        List of (start, end) tuples where end is exclusive
    """
    segments = []
    in_segment = False
    start = 0

    for i, valid in enumerate(mask):
        if valid and not in_segment:
            in_segment = True
            start = i
        elif not valid and in_segment:
            in_segment = False
            segments.append((start, i))

    if in_segment:
        segments.append((start, len(mask)))

    return segments


def extract_sliding_windows(
    segment: dict,
    window_size: int = 128,
    stride: int = 64,
    min_len: int = 40,
) -> list[dict]:
    """Extract overlapping windows from a segment.

    For segments <= window_size, returns the segment as a single window.
    For longer segments, extracts overlapping windows with the given stride,
    plus a final window aligned to the end to ensure full coverage.

    Example: For a 256-residue segment with window_size=128, stride=64:
        → [0:128], [64:192], [128:256] = 3 windows

    Args:
        segment: Dict with protein segment data
        window_size: Size of each window
        stride: Step between window starts
        min_len: Minimum length for a valid window

    Returns:
        List of window sample dicts
    """
    L = len(segment["seq"])
    parent_name = segment.get("parent_name", segment["name"])

    # Short segment: return as-is (if meets min_len)
    if L <= window_size:
        if L >= min_len:
            return [segment]
        return []

    windows = []
    start = 0
    window_idx = 0
    last_end = 0

    # Extract windows with stride
    while start + window_size <= L:
        end = start + window_size
        window_sample = {
            "name": f"{segment['name']}_win{window_idx}",
            "parent_name": parent_name,
            "seq": segment["seq"][start:end],
            "window_start": start,
            "window_end": end,
            "coords": {},
        }
        for atom_name, coords in segment["coords"].items():
            window_sample["coords"][atom_name] = coords[start:end]

        windows.append(window_sample)
        last_end = end
        start += stride
        window_idx += 1

    # Add final window aligned to end if there's uncovered sequence
    if last_end < L:
        final_start = L - window_size
        window_sample = {
            "name": f"{segment['name']}_win{window_idx}",
            "parent_name": parent_name,
            "seq": segment["seq"][final_start:L],
            "window_start": final_start,
            "window_end": L,
            "coords": {},
        }
        for atom_name, coords in segment["coords"].items():
            window_sample["coords"][atom_name] = coords[final_start:L]
        windows.append(window_sample)

    return windows


def extract_segments_from_chain(
    sample: dict,
    min_len: int = 40,
    max_len: int = 512,
    keep_longest_only: bool = True,
) -> list[dict]:
    """Extract contiguous resolved segments from a chain.

    Instead of rejecting chains with NaN coordinates, extract the valid
    contiguous segments. This recovers data from chains with missing loops
    or termini, and can yield multiple segments from long chains.

    Args:
        sample: Dict with protein chain data
        min_len: Minimum segment length to keep
        max_len: Maximum segment length to keep
        keep_longest_only: If True, return only the longest valid segment.
                          If False, return all valid segments.

    Returns:
        List of sample dicts, each representing a valid segment.
        Each segment dict has:
        - 'name': original chain name with segment suffix (e.g., '1abc.A_seg0')
        - 'parent_name': original chain name (for split tracking)
        - 'seq': subsequence for this segment
        - 'coords': dict with sliced coordinate arrays
        - 'segment_start': start index in original chain
        - 'segment_end': end index in original chain (exclusive)
    """
    ca_coords = get_ca_coords(sample)
    ca_mask = np.isfinite(ca_coords).all(axis=-1)

    segments = find_contiguous_segments(ca_mask)

    # Filter by length
    valid_segments = [(s, e) for s, e in segments if min_len <= (e - s) < max_len]

    if not valid_segments:
        return []

    if keep_longest_only:
        # Keep only the longest segment
        valid_segments = [max(valid_segments, key=lambda x: x[1] - x[0])]

    results = []
    for seg_idx, (start, end) in enumerate(valid_segments):
        # Create new sample dict for this segment
        seg_sample = {
            "name": f"{sample['name']}_seg{seg_idx}",
            "parent_name": sample["name"],
            "seq": sample["seq"][start:end],
            "segment_start": start,
            "segment_end": end,
            "coords": {},
        }

        # Slice all coordinate arrays
        for atom_name, coords in sample["coords"].items():
            seg_sample["coords"][atom_name] = coords[start:end]

        results.append(seg_sample)

    return results


def select_chain(
    sample: dict,
    min_len: int = 40,
    max_len: int = 512,
    require_no_nans: bool = True,
) -> bool:
    """Check if a chain meets selection criteria.

    Args:
        sample: Dict with protein chain data
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        require_no_nans: If True, reject chains with any NaN coordinates

    Returns:
        True if chain passes all criteria
    """
    seq_len = len(sample["seq"])

    # Length check
    if seq_len < min_len or seq_len > max_len:
        return False

    # NaN check
    if require_no_nans:
        _, mask = process_coords(sample)
        if not mask.all():
            return False

    return True


def get_one_chain(
    path: Path,
    min_len: int = 40,
    max_len: int = 512,
    require_no_nans: bool = True,
) -> tuple[str, str, np.ndarray] | None:
    """Get the first chain that meets selection criteria.

    Args:
        path: Path to chain_set.jsonl file
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        require_no_nans: If True, reject chains with any NaN coordinates

    Returns:
        Tuple of (name, seq, ca_coords) or None if no chain found
        ca_coords has shape (L, 3)
    """
    for sample in iter_chain_set_jsonl(path):
        if select_chain(sample, min_len, max_len, require_no_nans):
            name = sample["name"]
            seq = sample["seq"]
            ca_coords = get_ca_coords(sample)
            return name, seq, ca_coords
    return None


def filter_chains(
    path: Path,
    min_len: int = 40,
    max_len: int = 512,
    require_no_nans: bool = True,
    limit: int | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Filter dataset for chains meeting selection criteria.

    Args:
        path: Path to chain_set.jsonl file
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        require_no_nans: If True, reject chains with any NaN coordinates
        limit: Maximum number of chains to return (None for all)
        verbose: Print progress updates

    Returns:
        List of sample dicts passing all criteria
    """
    filtered = []
    total = 0

    for sample in iter_chain_set_jsonl(path):
        total += 1

        if select_chain(sample, min_len, max_len, require_no_nans):
            filtered.append(sample)

            if limit is not None and len(filtered) >= limit:
                break

        if verbose and total % 1000 == 0:
            print(f"Processed {total} records... (Found {len(filtered)} matching)", end="\r")

    if verbose:
        print(f"\nTotal scanned: {total}, Kept: {len(filtered)} ({100*len(filtered)/total:.1f}%)")

    return filtered


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Load pre-defined train/validation/test splits.

    Args:
        splits_path: Path to chain_set_splits.json

    Returns:
        Dict with 'train', 'validation', 'test' keys mapping to chain ID lists
    """
    with open(splits_path) as f:
        splits = json.load(f)
    return {
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"],
    }


def load_chains_by_ids(
    data_path: Path,
    chain_ids: set[str],
    min_len: int = 40,
    max_len: int = 512,
    require_no_nans: bool = True,
    limit: int | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Load chains matching specific IDs from the dataset.

    Args:
        data_path: Path to chain_set.jsonl file
        chain_ids: Set of chain IDs to load (e.g. {'1abc.A', '2def.B'})
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        require_no_nans: If True, reject chains with any NaN coordinates
        limit: Maximum number of chains to return (None for all)
        verbose: Print progress updates

    Returns:
        List of sample dicts matching IDs and passing criteria
    """
    filtered = []
    total = 0
    chain_ids = set(chain_ids)  # Ensure it's a set for O(1) lookup

    for sample in iter_chain_set_jsonl(data_path):
        total += 1

        # Check if this chain is in our target set
        if sample["name"] not in chain_ids:
            continue

        # Apply length/NaN filters
        if select_chain(sample, min_len, max_len, require_no_nans):
            filtered.append(sample)

            if limit is not None and len(filtered) >= limit:
                break

        if verbose and total % 5000 == 0:
            print(f"Scanned {total}... (Found {len(filtered)} matching)", end="\r")

    if verbose:
        print(f"\nScanned: {total}, Matched IDs & criteria: {len(filtered)}")

    return filtered


def load_chain_segments_by_ids(
    data_path: Path,
    chain_ids: set[str],
    min_len: int = 40,
    max_len: int = 512,
    keep_longest_only: bool = True,
    limit: int | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Load contiguous resolved segments from chains matching specific IDs.

    Instead of rejecting chains with NaN coordinates, extracts valid contiguous
    segments. This recovers ~4x more training data from the CATH dataset.

    Args:
        data_path: Path to chain_set.jsonl file
        chain_ids: Set of chain IDs to load (e.g. {'1abc.A', '2def.B'})
        min_len: Minimum segment length
        max_len: Maximum segment length
        keep_longest_only: If True, keep only longest valid segment per chain.
                          If False, keep all valid segments (more data).
        limit: Maximum number of segments to return (None for all)
        verbose: Print progress updates

    Returns:
        List of segment sample dicts. Each has 'parent_name' for split tracking.
    """
    segments = []
    total = 0
    chains_with_segments = 0
    chain_ids = set(chain_ids)

    for sample in iter_chain_set_jsonl(data_path):
        total += 1

        if sample["name"] not in chain_ids:
            continue

        # Extract valid segments from this chain
        chain_segments = extract_segments_from_chain(
            sample,
            min_len=min_len,
            max_len=max_len,
            keep_longest_only=keep_longest_only,
        )

        if chain_segments:
            chains_with_segments += 1
            segments.extend(chain_segments)

            if limit is not None and len(segments) >= limit:
                segments = segments[:limit]
                break

        if verbose and total % 5000 == 0:
            print(f"Scanned {total}... (Found {len(segments)} segments from {chains_with_segments} chains)")

    if verbose:
        print(f"\nScanned: {total}, Chains with segments: {chains_with_segments}, Total segments: {len(segments)}")

    return segments


def load_chain_windows_by_ids(
    data_path: Path,
    chain_ids: set[str],
    window_size: int = 128,
    stride: int = 64,
    min_len: int = 40,
    keep_longest_only: bool = True,
    limit: int | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Load sliding windows from chains, recovering data from long chains.

    This function:
    1. Extracts contiguous resolved segments from each chain (handles NaN gaps)
    2. For segments > window_size: extracts overlapping windows with stride
    3. For segments <= window_size: keeps as single sample (if >= min_len)

    This recovers training data from long chains that would otherwise be
    rejected by max_len filtering.

    Example: A 300-residue clean chain with window_size=128, stride=64:
        → [0:128], [64:192], [128:256], [172:300] = 4 training samples

    Args:
        data_path: Path to chain_set.jsonl file
        chain_ids: Set of chain IDs to load
        window_size: Size of each window (should match model's max_len)
        stride: Step between window starts (e.g., 64 = 50% overlap)
        min_len: Minimum length for a valid window
        keep_longest_only: If True, only process longest segment per chain
                          (maintains split integrity). If False, process all
                          segments (more data but potential split leakage).
        limit: Maximum number of windows to return (None for all)
        verbose: Print progress updates

    Returns:
        List of window sample dicts. Each has 'parent_name' for split tracking.
    """
    windows = []
    total = 0
    chains_processed = 0
    chain_ids = set(chain_ids)

    for sample in iter_chain_set_jsonl(data_path):
        total += 1

        if sample["name"] not in chain_ids:
            continue

        # Extract contiguous resolved segments (no max_len filter - we'll window them)
        # Use a high max_len to accept long segments that we'll then window
        chain_segments = extract_segments_from_chain(
            sample,
            min_len=min_len,
            max_len=10000,  # Accept very long segments
            keep_longest_only=keep_longest_only,
        )

        if not chain_segments:
            continue

        chains_processed += 1

        # Apply sliding windows to each segment
        for segment in chain_segments:
            segment_windows = extract_sliding_windows(
                segment,
                window_size=window_size,
                stride=stride,
                min_len=min_len,
            )
            windows.extend(segment_windows)

            if limit is not None and len(windows) >= limit:
                windows = windows[:limit]
                break

        if limit is not None and len(windows) >= limit:
            break

        if verbose and total % 5000 == 0:
            print(f"Scanned {total}... ({len(windows)} windows from {chains_processed} chains)")

    if verbose:
        print(f"\nScanned: {total}, Chains processed: {chains_processed}, Total windows: {len(windows)}")

    return windows


def main():
    """Demo: load one chain and print stats."""
    import sys

    # Default path or take from command line
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path("data/chain_set.jsonl")

    if not path.exists():
        print(f"Error: {path} not found")
        print("Usage: uv run python -m src.data_cath [path/to/chain_set.jsonl]")
        sys.exit(1)

    print(f"Loading from {path}...")

    # Get one chain
    result = get_one_chain(path, min_len=40, max_len=150)
    if result is None:
        print("No chains found matching criteria")
        sys.exit(1)

    name, seq, ca_coords = result

    # Print stats
    print(f"\n{'='*50}")
    print(f"Name:      {name}")
    print(f"Length:    {len(seq)} residues")
    print(f"Sequence:  {seq[:50]}{'...' if len(seq) > 50 else ''}")
    print(f"\nCA coords shape: {ca_coords.shape}")
    print(f"CA coords dtype: {ca_coords.dtype}")
    print(f"Coord range:")
    print(f"  X: [{ca_coords[:, 0].min():.1f}, {ca_coords[:, 0].max():.1f}]")
    print(f"  Y: [{ca_coords[:, 1].min():.1f}, {ca_coords[:, 1].max():.1f}]")
    print(f"  Z: [{ca_coords[:, 2].min():.1f}, {ca_coords[:, 2].max():.1f}]")

    # CA-CA distances (bond lengths)
    ca_dists = np.linalg.norm(np.diff(ca_coords, axis=0), axis=1)
    print(f"\nCA-CA distances:")
    print(f"  Mean: {ca_dists.mean():.2f} Å")
    print(f"  Std:  {ca_dists.std():.2f} Å")
    print(f"  Range: [{ca_dists.min():.2f}, {ca_dists.max():.2f}] Å")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
