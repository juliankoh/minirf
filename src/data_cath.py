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
