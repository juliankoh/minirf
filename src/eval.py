"""Comprehensive evaluation suite for Diffusion Transformers.

Usage:
    # Run eval on a specific checkpoint
    uv run python -m src.eval --model_path runs/20231229_120000/model.pt

    # Run on a different device
    uv run python -m src.eval --model_path runs/latest/model.pt --device cuda

    # Use test set instead of validation set
    uv run python -m src.eval --model_path runs/latest/model.pt --split test

Evaluation Strategy
-------------------
This module implements a "scorecard" approach with three categories:

1. Denoiser Metrics:
   - MSE and x0-prediction RMSD at specific timesteps
   - Tests if the model can predict noise accurately

2. Reconstruction Metrics:
   - Noise real data -> Denoise -> Compare to original
   - Tests if the sampler can recover structure (the "missing link" metric)

3. Generation Quality:
   - Bond length validity, clash detection, radius of gyration
   - Diversity (pairwise RMSD) and memorization checks
   - Tests if generated structures are protein-like
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data_cath import load_chains_by_ids, load_splits
from .diffusion import DiffusionSchedule
from .geom import (
    align_to_principal_axes,
    batch_rmsd_pairwise,
    ca_bond_lengths,
    center,
    compute_clashes,
    radius_of_gyration,
    rmsd,
)
from .model import DiffusionTransformer
from .sampler import DiffusionSampler


class Evaluator:
    """Comprehensive evaluation for diffusion models on protein structures."""

    def __init__(
        self,
        model: DiffusionTransformer,
        schedule: DiffusionSchedule,
        device: torch.device,
        scale_factor: float = 10.0,
    ):
        self.model = model
        self.schedule = schedule
        self.device = device
        self.scale_factor = scale_factor
        self.sampler = DiffusionSampler(model, schedule)

    @torch.no_grad()
    def run_full_eval(
        self,
        anchor_batch: tuple[torch.Tensor, torch.Tensor],
        sample_batch_size: int = 32,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Run the full scorecard evaluation."""
        self.model.eval()
        metrics = {}

        x0, mask = anchor_batch
        x0 = x0.to(self.device)
        mask = mask.to(self.device)

        if verbose:
            print(f"  1. Evaluating Denoiser (on {len(x0)} anchor chains)...")
        denoiser_stats = self._evaluate_denoiser(x0, mask)
        metrics.update(denoiser_stats)

        if verbose:
            print("  2. Evaluating Reconstruction (Sampler Sanity)...")
        recon_stats = self._evaluate_reconstruction(x0, mask)
        metrics.update(recon_stats)

        if verbose:
            print(
                f"  3. Evaluating Unconditional Generation ({sample_batch_size} samples)..."
            )

        # Determine generation length (median of anchor batch)
        lengths = mask.sum(dim=1)
        modal_length = int(lengths.float().median().item())

        gen_stats = self._evaluate_generation(
            n_samples=sample_batch_size,
            length=modal_length,
            ref_batch=x0,
            ref_mask=mask,
        )
        metrics.update(gen_stats)

        return metrics

    def _evaluate_denoiser(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        timesteps: list[int] = [100, 500, 900],
    ) -> dict[str, float]:
        """Check supervised loss and single-step x0 prediction accuracy."""
        stats = {}

        for t_val in timesteps:
            B = x0.shape[0]
            t = torch.full((B,), t_val, device=self.device, dtype=torch.long)

            # Forward diff
            x_t, noise = self.schedule.q_sample(x0, t)

            # Predict
            eps_pred = self.model(x_t, t, mask=mask)

            # MSE (masked)
            mask_3d = mask.unsqueeze(-1).expand_as(eps_pred)
            mse = F.mse_loss(eps_pred[mask_3d], noise[mask_3d]).item()
            stats[f"denoise/mse_t{t_val}"] = mse

            # x0 Prediction RMSD
            sqrt_alpha_bar = self.schedule.sqrt_alpha_bars[t].view(B, 1, 1)
            sqrt_one_minus = self.schedule.sqrt_one_minus_alpha_bars[t].view(B, 1, 1)

            x0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar

            # Compute RMSD for each item in batch
            rmsds = []
            x0_np = x0.cpu().numpy() * self.scale_factor
            x0_pred_np = x0_pred.cpu().numpy() * self.scale_factor
            mask_np = mask.cpu().numpy()

            for i in range(B):
                valid = mask_np[i].astype(bool)
                if valid.sum() < 3:
                    continue
                val = rmsd(x0_pred_np[i, valid], x0_np[i, valid], align=True)
                rmsds.append(val)

            if rmsds:
                stats[f"denoise/x0_rmsd_t{t_val}"] = np.mean(rmsds)

        return stats

    def _evaluate_reconstruction(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        start_timesteps: list[int] = [500, 900],
    ) -> dict[str, float]:
        """Noise real data and try to rebuild it. Tests sampler mechanics."""
        stats = {}

        for t_start in start_timesteps:
            # 1. Noise to t_start
            t_tensor = torch.full(
                (x0.shape[0],), t_start, device=self.device, dtype=torch.long
            )
            x_t, _ = self.schedule.q_sample(x0, t_tensor)

            # 2. Denoise back to 0 (pass mask so model knows which positions are real)
            x_recon = self.sampler.sample_from(x_t, start_t=t_start, verbose=False, mask=mask)

            # 3. Measure RMSD to ground truth
            rmsds = []
            x0_np = x0.cpu().numpy() * self.scale_factor
            recon_np = x_recon.cpu().numpy() * self.scale_factor
            mask_np = mask.cpu().numpy()

            for i in range(len(x0)):
                valid = mask_np[i].astype(bool)
                if valid.sum() < 3:
                    continue
                val = rmsd(recon_np[i, valid], x0_np[i, valid], align=True)
                rmsds.append(val)

            if rmsds:
                stats[f"recon/rmsd_t{t_start}"] = np.median(rmsds)

        return stats

    def _evaluate_generation(
        self,
        n_samples: int,
        length: int,
        ref_batch: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Unconditional generation stats: Physics validity and Diversity."""
        stats = {}

        # Generate batch
        shape = (n_samples, length, 3)
        samples = self.sampler.sample(shape, verbose=False, device=self.device)
        samples_np = samples.cpu().numpy() * self.scale_factor

        # 1. Geometry Checks
        bond_lengths = []
        clashes = []
        rgs = []

        for i in range(n_samples):
            bonds = ca_bond_lengths(samples_np[i])
            bond_lengths.append(bonds.mean())
            clashes.append(compute_clashes(samples_np[i]))
            rgs.append(radius_of_gyration(samples_np[i]))

        stats["gen/bond_len_mean"] = np.mean(bond_lengths)
        stats["gen/clashes_mean"] = np.mean(clashes)
        stats["gen/rg_mean"] = np.mean(rgs)

        # Bond length validity (% in 3.6 - 4.0 range)
        all_bonds = np.concatenate([ca_bond_lengths(s) for s in samples_np])
        valid_bonds = ((all_bonds > 3.6) & (all_bonds < 4.0)).mean()
        stats["gen/valid_bond_pct"] = valid_bonds

        # 2. Diversity (Pairwise RMSD)
        stats["gen/diversity_rmsd"] = batch_rmsd_pairwise(samples_np)

        # 3. Memorization Check (Nearest Neighbor to anchor batch)
        ref_np = ref_batch.cpu().numpy() * self.scale_factor
        ref_mask_np = ref_mask.cpu().numpy()
        min_rmsds = []

        for i in range(n_samples):
            dists = []
            for j in range(len(ref_np)):
                # Extract valid coordinates from reference
                valid_mask = ref_mask_np[j].astype(bool)
                ref_valid_coords = ref_np[j, valid_mask]

                n_valid = len(ref_valid_coords)
                if n_valid < 3:
                    continue

                # CRITICAL FIX: Ensure shapes match for RMSD
                # Truncate both to the minimum length to compare overlapping N-terminus
                cmp_len = min(length, n_valid)

                # We need a reasonable overlap to calculate RMSD
                if cmp_len < 10:
                    continue

                sample_crop = samples_np[i, :cmp_len]
                ref_crop = ref_valid_coords[:cmp_len]

                d = rmsd(sample_crop, ref_crop, align=True)
                dists.append(d)

            if dists:
                min_rmsds.append(min(dists))

        if min_rmsds:
            stats["gen/memorization_nn_rmsd"] = np.median(min_rmsds)

        return stats


def print_eval_report(metrics: dict[str, float], step: int) -> None:
    """Print a formatted evaluation report."""
    print(f"\n{'─' * 50}")
    print(f"Eval @ Step {step}")
    print(f"{'─' * 50}")

    print("Denoiser:")
    print(f"  MSE t=100: {metrics.get('denoise/mse_t100', 0):.4f}  |  t=500: {metrics.get('denoise/mse_t500', 0):.4f}  |  t=900: {metrics.get('denoise/mse_t900', 0):.4f}")
    print(f"  x0 RMSD t=100: {metrics.get('denoise/x0_rmsd_t100', 0):.2f} A  |  t=500: {metrics.get('denoise/x0_rmsd_t500', 0):.2f} A")

    print("Reconstruction:")
    print(f"  RMSD from t=500: {metrics.get('recon/rmsd_t500', 0):.2f} A  |  t=900: {metrics.get('recon/rmsd_t900', 0):.2f} A")

    print("Generation:")
    print(f"  Bond: {metrics.get('gen/bond_len_mean', 0):.2f} A (valid: {metrics.get('gen/valid_bond_pct', 0):.0%})  |  Clashes: {metrics.get('gen/clashes_mean', 0):.1f}  |  Rg: {metrics.get('gen/rg_mean', 0):.1f} A")
    print(f"  Diversity: {metrics.get('gen/diversity_rmsd', 0):.1f} A  |  NN RMSD: {metrics.get('gen/memorization_nn_rmsd', 0):.1f} A")
    print(f"{'─' * 50}\n")


def load_anchor_set(
    data_path: Path,
    splits_path: Path,
    split: str = "validation",
    max_len: int = 128,
    num_anchors: int = 32,
    scale_factor: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a consistent set of chains from a specific split."""
    print(f"Loading {split} anchor set...")
    splits = load_splits(splits_path)
    chain_ids = splits[split]

    chains = load_chains_by_ids(
        data_path,
        chain_ids=chain_ids,
        min_len=40,
        max_len=max_len,
        verbose=False,
    )

    if len(chains) == 0:
        raise ValueError(f"No chains found in {split} split matching length criteria")

    if len(chains) > num_anchors:
        chains = sorted(chains, key=lambda c: c["name"])[:num_anchors]

    print(f"  Using {len(chains)} anchor chains")

    batch_coords = []
    batch_mask = []

    for chain in chains:
        ca_coords = np.array(chain["coords"]["CA"])
        ca_centered, _ = center(ca_coords)
        ca_aligned = align_to_principal_axes(ca_centered)
        coords = ca_aligned / scale_factor

        L = len(coords)
        padded = np.zeros((max_len, 3))
        padded[:L] = coords
        mask = np.zeros(max_len, dtype=bool)
        mask[:L] = True

        batch_coords.append(padded)
        batch_mask.append(mask)

    return (
        torch.from_numpy(np.stack(batch_coords)).float(),
        torch.from_numpy(np.stack(batch_mask)).bool(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Diffusion Transformer checkpoint"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model.pt checkpoint"
    )
    parser.add_argument("--data_path", type=str, default="data/chain_set.jsonl")
    parser.add_argument("--splits_path", type=str, default="data/chain_set_splits.json")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--num_anchors", type=int, default=32)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_args = checkpoint.get("args", {})
    d_model = model_args.get("d_model", 128)
    num_layers = model_args.get("num_layers", 4)
    num_heads = model_args.get("num_heads", 4)
    max_len = model_args.get("max_len", 128)
    scale_factor = 10.0

    print(f"  Model config: d_model={d_model}, layers={num_layers}, heads={num_heads}")

    model = DiffusionTransformer(
        d_model=d_model, num_layers=num_layers, num_heads=num_heads
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    schedule = DiffusionSchedule(T=1000).to(device)

    anchor_set = load_anchor_set(
        data_path=Path(args.data_path),
        splits_path=Path(args.splits_path),
        split=args.split,
        max_len=max_len,
        num_anchors=args.num_anchors,
        scale_factor=scale_factor,
    )

    print("\nRunning evaluation...")
    evaluator = Evaluator(model, schedule, device, scale_factor=scale_factor)
    metrics = evaluator.run_full_eval(
        anchor_batch=anchor_set,
        sample_batch_size=args.num_samples,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {model_path.name}")
    print(
        f"Split: {args.split} | Anchors: {args.num_anchors} | Samples: {args.num_samples}"
    )
    print("=" * 60)

    print("\n1. Denoiser Quality (Lower is better)")
    print(f"   MSE (t=100):          {metrics.get('denoise/mse_t100', 0):.4f}")
    print(f"   MSE (t=500):          {metrics.get('denoise/mse_t500', 0):.4f}")
    print(f"   MSE (t=900):          {metrics.get('denoise/mse_t900', 0):.4f}")
    print(f"   x0 RMSD (t=100):      {metrics.get('denoise/x0_rmsd_t100', 0):.2f} A")
    print(f"   x0 RMSD (t=500):      {metrics.get('denoise/x0_rmsd_t500', 0):.2f} A")

    print("\n2. Reconstruction (Sampler Sanity)")
    print(f"   RMSD from t=500:      {metrics.get('recon/rmsd_t500', 0):.2f} A")
    print(f"   RMSD from t=900:      {metrics.get('recon/rmsd_t900', 0):.2f} A")

    print(f"\n3. Unconditional Generation ({args.num_samples} samples)")
    print(
        f"   Mean Bond Length:     {metrics.get('gen/bond_len_mean', 0):.2f} A  (Target: 3.8)"
    )
    print(
        f"   Valid Bonds (3.6-4A): {metrics.get('gen/valid_bond_pct', 0):.1%}  (Target: >90%)"
    )
    print(
        f"   Clashes per sample:   {metrics.get('gen/clashes_mean', 0):.2f}  (Target: <1.0)"
    )
    print(f"   Radius of Gyration:   {metrics.get('gen/rg_mean', 0):.2f} A")
    print(
        f"   Diversity (RMSD):     {metrics.get('gen/diversity_rmsd', 0):.2f} A  (Higher = better)"
    )
    if "gen/memorization_nn_rmsd" in metrics:
        print(
            f"   Memorization (NN):    {metrics['gen/memorization_nn_rmsd']:.2f} A  (Too low = overfitting)"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
