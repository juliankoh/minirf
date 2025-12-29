# %% [markdown]
# # Visualize Data
#
# Load and explore a segment of the protein structure dataset.

# %%
import json
from pathlib import Path

import numpy as np
import torch

# %%
DATA_PATH = Path("../data/chain_set.jsonl")

# %%
def load_jsonl(path: Path, limit: int = 10) -> list[dict]:
    """Load first `limit` records from a JSONL file."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            records.append(json.loads(line))
    return records

# %%
# Load a small segment of the dataset
data = load_jsonl(DATA_PATH, limit=5)
print(f"Loaded {len(data)} records")

# %%
# Inspect the first record
sample = data[0]
print(f"Keys: {list(sample.keys())}")
print(f"Sequence length: {len(sample['seq'])}")
print(f"Sequence (first 50 chars): {sample['seq'][:50]}...")

# %%
# Inspect coordinate structure
coords = sample["coords"]
print(f"Coordinate keys: {list(coords.keys())}")
for atom, positions in coords.items():
    print(f"  {atom}: {len(positions)} positions, first 3: {positions[:3]}")

# %% [markdown]
# ## Handling Missing Coordinates
#
# Many residues have NaN coordinates due to unresolved regions in the crystal structure.
# We need to create masks to handle this during training.

# %%
def process_coords(sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract coordinates and create a validity mask.

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

# %%
# Process the sample
xyz, mask = process_coords(sample)

print(f"Coordinates shape: {xyz.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Valid residues: {mask.sum().item()} / {len(mask)} ({100 * mask.float().mean():.1f}%)")
print(f"First 10 mask values: {mask[:10].tolist()}")

# %%
# Check coverage across multiple samples
for i, s in enumerate(data):
    _, m = process_coords(s)
    coverage = 100 * m.float().mean()
    print(f"Sample {i}: {s['name']:20s} | {m.sum():3d}/{len(m):3d} resolved ({coverage:.1f}%)")
