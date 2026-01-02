"""Compare multiple model checkpoints by sampling with consistent seeds.

Usage:
    uv run python scripts/compare_checkpoints.py \
        runs/20251230_224500/model.pt \
        runs/20251229_182613/model.pt \
        --num_seeds 100 \
        --length 64

Compares distributions (median/P95) of:
- Bond lengths
- Radius of gyration
- Clashes
- NN RMSD (memorization check against validation set)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.data_cath import load_chains_by_ids, load_splits
from src.diffusion import DiffusionSchedule
from src.geom import (
    align_to_principal_axes,
    ca_bond_lengths,
    center,
    compute_clashes,
    radius_of_gyration,
    rmsd,
)
from src.model import DiffusionTransformer
from src.sampler import DiffusionSampler


def load_model(checkpoint_path: Path, device: torch.device) -> DiffusionTransformer:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})

    model = DiffusionTransformer(
        d_model=args.get("d_model", 128),
        num_layers=args.get("num_layers", 4),
        num_heads=args.get("num_heads", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_reference_set(
    data_path: Path,
    splits_path: Path,
    max_len: int = 128,
    num_refs: int = 100,
    scale_factor: float = 10.0,
) -> np.ndarray:
    """Load reference structures from validation set for NN RMSD."""
    splits = load_splits(splits_path)
    chain_ids = splits["validation"]

    chains = load_chains_by_ids(
        data_path,
        chain_ids=chain_ids,
        min_len=40,
        max_len=max_len,
        verbose=False,
    )

    if len(chains) > num_refs:
        chains = sorted(chains, key=lambda c: c["name"])[:num_refs]

    ref_coords = []
    for chain in chains:
        ca_coords = np.array(chain["coords"]["CA"])
        ca_centered, _ = center(ca_coords)
        ca_aligned = align_to_principal_axes(ca_centered)
        ref_coords.append(ca_aligned)  # Keep in Angstroms

    return ref_coords


def compute_nn_rmsd(sample: np.ndarray, ref_coords: list[np.ndarray]) -> float:
    """Compute nearest neighbor RMSD between sample and reference set."""
    min_rmsd = float("inf")

    for ref in ref_coords:
        # Compare overlapping N-terminus region
        cmp_len = min(len(sample), len(ref))
        if cmp_len < 10:
            continue

        sample_crop = sample[:cmp_len]
        ref_crop = ref[:cmp_len]

        d = rmsd(sample_crop, ref_crop, align=True)
        min_rmsd = min(min_rmsd, d)

    return min_rmsd if min_rmsd < float("inf") else np.nan


def sample_with_seed(
    sampler: DiffusionSampler,
    shape: tuple,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Sample with fixed seed for reproducibility."""
    torch.manual_seed(seed)
    samples = sampler.sample(
        shape,
        verbose=False,
        device=device,
        add_noise=True,
        use_self_cond=True,
    )
    return samples.cpu().numpy()


def compute_metrics(
    samples_np: np.ndarray,
    ref_coords: list[np.ndarray],
    scale_factor: float = 10.0,
) -> dict[str, list[float]]:
    """Compute metrics for a batch of samples."""
    # Scale to Angstroms
    samples_angstrom = samples_np * scale_factor

    bond_means = []
    rg_values = []
    clash_counts = []
    nn_rmsds = []

    for sample in samples_angstrom:
        # Bond lengths
        bonds = ca_bond_lengths(sample)
        bond_means.append(bonds.mean())

        # Radius of gyration
        rg_values.append(radius_of_gyration(sample))

        # Clashes
        clash_counts.append(compute_clashes(sample))

        # NN RMSD
        nn_rmsds.append(compute_nn_rmsd(sample, ref_coords))

    return {
        "bond": bond_means,
        "rg": rg_values,
        "clashes": clash_counts,
        "nn_rmsd": nn_rmsds,
    }


def print_comparison_table(
    checkpoint_names: list[str],
    all_metrics: list[dict[str, list[float]]],
) -> None:
    """Print formatted comparison table."""
    metrics_order = ["bond", "rg", "clashes", "nn_rmsd"]
    metric_labels = {
        "bond": "Bond Length (A)",
        "rg": "Rg (A)",
        "clashes": "Clashes",
        "nn_rmsd": "NN RMSD (A)",
    }

    # Calculate column widths
    name_width = max(len(name) for name in checkpoint_names) + 2
    col_width = 22

    # Header
    print("\n" + "=" * (name_width + col_width * 4 + 3))
    print("CHECKPOINT COMPARISON")
    print("=" * (name_width + col_width * 4 + 3))

    for metric_key in metrics_order:
        print(f"\n{metric_labels[metric_key]}")
        print("-" * (name_width + col_width * 2 + 1))
        print(f"{'Checkpoint':<{name_width}} {'Median':>{col_width}} {'P95':>{col_width}}")
        print("-" * (name_width + col_width * 2 + 1))

        for name, metrics in zip(checkpoint_names, all_metrics):
            values = np.array(metrics[metric_key])
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                median = np.median(valid_values)
                p95 = np.percentile(valid_values, 95)
                print(f"{name:<{name_width}} {median:>{col_width}.3f} {p95:>{col_width}.3f}")
            else:
                print(f"{name:<{name_width}} {'N/A':>{col_width}} {'N/A':>{col_width}}")

    print("\n" + "=" * (name_width + col_width * 4 + 3))


def main():
    parser = argparse.ArgumentParser(
        description="Compare model checkpoints by sampling with consistent seeds"
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=str,
        help="Paths to model checkpoint files",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=100,
        help="Number of random seeds to sample (default: 100)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=64,
        help="Length of generated sequences (default: 64)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Samples per seed (default: 10)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/chain_set.jsonl",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default="data/chain_set_splits.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed_start",
        type=int,
        default=0,
        help="Starting seed value (default: 0)",
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Validate checkpoints
    checkpoint_paths = [Path(p) for p in args.checkpoints]
    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Use parent directory name as checkpoint identifier
    checkpoint_names = [p.parent.name for p in checkpoint_paths]
    print(f"\nComparing {len(checkpoint_paths)} checkpoints:")
    for name, path in zip(checkpoint_names, checkpoint_paths):
        print(f"  - {name}: {path}")

    # Load models
    print("\nLoading models...")
    models = []
    for path in tqdm(checkpoint_paths, desc="Loading"):
        models.append(load_model(path, device))

    # Create schedule and samplers
    schedule = DiffusionSchedule(T=1000).to(device)
    samplers = [DiffusionSampler(model, schedule) for model in models]

    # Load reference set for NN RMSD
    print("\nLoading reference set for NN RMSD...")
    ref_coords = load_reference_set(
        Path(args.data_path),
        Path(args.splits_path),
        max_len=args.length,
        num_refs=100,
    )
    print(f"  Loaded {len(ref_coords)} reference structures")

    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    shape = (args.batch_size, args.length, 3)

    print(f"\nSampling with {args.num_seeds} seeds, {args.batch_size} samples each...")
    print(f"  Shape: {shape}")
    print(f"  Total samples per checkpoint: {args.num_seeds * args.batch_size}")

    # Collect metrics for each checkpoint
    all_metrics = [{
        "bond": [],
        "rg": [],
        "clashes": [],
        "nn_rmsd": [],
    } for _ in models]

    for seed in tqdm(seeds, desc="Sampling"):
        for i, sampler in enumerate(samplers):
            # Sample with this seed
            samples_np = sample_with_seed(sampler, shape, seed, device)

            # Compute metrics
            metrics = compute_metrics(samples_np, ref_coords)

            # Accumulate
            for key in all_metrics[i]:
                all_metrics[i][key].extend(metrics[key])

    # Print comparison
    print_comparison_table(checkpoint_names, all_metrics)

    # Also print raw statistics
    print("\nDetailed Statistics:")
    print("-" * 80)
    for name, metrics in zip(checkpoint_names, all_metrics):
        print(f"\n{name}:")
        for key in ["bond", "rg", "clashes", "nn_rmsd"]:
            values = np.array(metrics[key])
            valid = values[~np.isnan(values)]
            if len(valid) > 0:
                print(f"  {key:>10}: median={np.median(valid):.3f}, "
                      f"mean={np.mean(valid):.3f}, std={np.std(valid):.3f}, "
                      f"P5={np.percentile(valid, 5):.3f}, P95={np.percentile(valid, 95):.3f}")


if __name__ == "__main__":
    main()
