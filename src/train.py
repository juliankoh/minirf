"""Training loop for the Diffusion Transformer.

Usage:
    # Overfit on single chain (sanity check)
    uv run python -m src.train --overfit --steps 2000

    # Train on dataset (100 chains, complete domains 40-128 residues)
    uv run python -m src.train --steps 10000 --num_chains 100

    # Larger training run
    uv run python -m src.train --steps 50000 --num_chains 500 --batch_size 8

Training Objective (ε-prediction):
    Given noisy coordinates x_t at timestep t, predict the noise ε that was added.
    Loss = MSE(ε_pred, ε)

    Why ε-prediction instead of x0-prediction?
    - At high noise levels (t→T), x_t ≈ ε, so predicting ε is easy
    - At low noise (t→0), x_t ≈ x0, so ε = (x_t - x0)/σ is also learnable
    - This makes learning more uniform across timesteps
    - x0-prediction struggles at high t where x_t has little signal about x0

Data Strategy (Complete Domains):
    We train on complete protein domains (not random crops) so the model
    learns what a whole protein looks like - beginning, middle, and end.
    Chains are filtered to 40-128 residues and padded to max_len.
    This is crucial for learning proper protein structure.

Canonical Orientation:
    All proteins are aligned to their principal axes (via SVD) during loading.
    This gives a consistent orientation (longest axis along X) so the model
    learns to generate proteins in a standard pose.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data_cath import get_one_chain, load_chain_segments_by_ids, load_chain_windows_by_ids, load_splits
from .diffusion import DiffusionSchedule
from .eval import Evaluator, print_eval_report
from .geom import align_to_principal_axes, ca_bond_lengths, center, rmsd
from .model import DiffusionTransformer
from .sampler import DiffusionSampler


class ProteinDataset:
    """Simple dataset for protein CA coordinates.

    Loads complete protein domains and pads them to a fixed length.
    No cropping - the model learns what complete proteins look like.
    """

    def __init__(
        self,
        chains: list[dict],
        max_len: int = 128,
        scale_factor: float = 10.0,
    ):
        """Initialize dataset.

        Args:
            chains: List of chain dicts with 'coords' key containing CA coords
            max_len: Maximum length to pad sequences to
            scale_factor: Coordinate scaling factor
        """
        self.chains = chains
        self.max_len = max_len
        self.scale_factor = scale_factor

        # Preprocess all chains: center, align to principal axes, and scale
        self.coords_list = []
        for chain in chains:
            ca_coords = np.array(chain["coords"]["CA"])
            ca_centered, _ = center(ca_coords)
            ca_aligned = align_to_principal_axes(ca_centered)
            self.coords_list.append(ca_aligned / scale_factor)

    def __len__(self) -> int:
        return len(self.chains)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of complete domains, padded to max_len."""
        batch_coords = []
        batch_mask = []

        for _ in range(batch_size):
            idx = rng.integers(0, len(self.coords_list))
            coords = self.coords_list[idx]
            L = len(coords)

            # Pad to max_len (no cropping - chains are pre-filtered to fit)
            padded = np.zeros((self.max_len, 3))
            padded[:L] = coords
            mask = np.zeros(self.max_len, dtype=bool)
            mask[:L] = True

            batch_coords.append(padded)
            batch_mask.append(mask)

        return (
            torch.from_numpy(np.stack(batch_coords)).float(),
            torch.from_numpy(np.stack(batch_mask)).bool(),
        )


def train_step(
    model: DiffusionTransformer,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scale_factor: float = 10.0,
    bond_loss_weight: float = 0.1,
    self_cond_prob: float = 0.5,
) -> dict:
    """Single training step with ε-prediction, bond loss, and self-conditioning.

    Self-conditioning: 50% of the time, run the model forward once to get an
    initial x0 estimate, then feed it back (with stopped gradients) as a
    conditioning signal for the second pass. This dramatically improves
    coherence and performance (per RFdiffusion paper).

    Args:
        model: The diffusion transformer (predicts ε)
        schedule: Diffusion schedule for q_sample
        x0: (B, L, 3) clean coordinates (scaled, canonically oriented)
        optimizer: Optimizer
        device: Device to run on (cuda/mps/cpu)
        scale_factor: Coordinate scaling factor (for bond length target)
        bond_loss_weight: Weight for auxiliary bond-length loss (λ)
        self_cond_prob: Probability of using self-conditioning (default 0.5)

    Returns:
        Dictionary with loss and metrics
    """
    model.train()

    # UNPACK MASKED COORDINATES
    if isinstance(x0, tuple):
        x0, mask = x0
    else:
        mask = None  # For single chain overfit mode

    B = x0.shape[0]

    # Move to device
    x0 = x0.to(device)
    if mask is not None:
        mask = mask.to(device)

    # Sample random timesteps
    t = torch.randint(0, schedule.T, (B,), device=device)

    # Forward diffusion: add noise
    # x_t = sqrt(α̅_t) * x0 + sqrt(1-α̅_t) * ε
    x_t, noise = schedule.q_sample(x0, t)

    # SELF-CONDITIONING: 50% of the time, do a first pass and use its output
    x0_self_cond = None
    used_self_cond = False
    if torch.rand(1).item() < self_cond_prob:
        with torch.no_grad():
            # First pass: get initial estimate without self-conditioning
            eps_pred_initial = model(x_t, t, mask=mask, x0_self_cond=None)
            # Convert eps prediction to x0 prediction
            sqrt_ab = schedule.sqrt_alpha_bars[t].view(B, 1, 1)
            sqrt_omb = schedule.sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
            x0_self_cond = (x_t - sqrt_omb * eps_pred_initial) / sqrt_ab
            # x0_self_cond is detached (no gradients) due to torch.no_grad()
        used_self_cond = True

    # Model prediction: predict ε (with optional self-conditioning)
    eps_pred = model(x_t, t, mask=mask, x0_self_cond=x0_self_cond)

    # MASKED LOSS CALCULATION: MSE(ε_pred, ε)
    if mask is not None:
        # Expand mask for coordinates: (B, L) -> (B, L, 3)
        mask_3d = mask.unsqueeze(-1).expand_as(eps_pred)
        # Only compute error on real atoms
        eps_loss = F.mse_loss(eps_pred[mask_3d], noise[mask_3d])
    else:
        eps_loss = F.mse_loss(eps_pred, noise)

    # AUXILIARY BOND-LENGTH LOSS
    # Compute implied x0 from the predicted noise
    sqrt_ab = schedule.sqrt_alpha_bars[t].view(B, 1, 1)
    sqrt_omb = schedule.sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
    x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab

    # --- bond loss (stable version, gated by timestep) ---
    bond_vecs = x0_pred[:, 1:] - x0_pred[:, :-1]  # (B, L-1, 3)
    bond_lens = torch.linalg.norm(bond_vecs, dim=-1)  # (B, L-1)
    target = 3.8 / scale_factor

    # Only apply bond loss for t < 200 (x0 predictions are garbage at high t)
    bond_t_max = 200
    t_gate = (t < bond_t_max).float().view(B, 1)  # (B, 1) broadcasts over (B, L-1)

    bond_sq_err = (bond_lens - target) ** 2  # (B, L-1)

    if mask is not None:
        valid_bonds = mask[:, 1:] & mask[:, :-1]  # (B, L-1)
        bond_sq_err = bond_sq_err * valid_bonds.float()
        denom = (valid_bonds.float() * t_gate).sum().clamp(min=1.0)
        bond_loss = (bond_sq_err * t_gate).sum() / denom
    else:
        denom = t_gate.sum().clamp(min=1.0) * bond_lens.shape[1]
        bond_loss = (bond_sq_err * t_gate).sum() / denom

    # Combined loss
    loss = eps_loss + bond_loss_weight * bond_loss

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {
        "loss": loss.item(),
        "eps_loss": eps_loss.item(),
        "bond_loss": bond_loss.item() if isinstance(bond_loss, torch.Tensor) else bond_loss,
        "t_mean": t.float().mean().item(),
        "self_cond": float(used_self_cond),
    }


@torch.no_grad()
def eval_step(
    model: DiffusionTransformer,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    t: int = 500,
    scale_factor: float = 10.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Evaluate model at a fixed timestep (ε-prediction).

    Args:
        model: The diffusion transformer (predicts ε)
        schedule: Diffusion schedule
        x0: (1, L, 3) clean coordinates (scaled)
        t: Timestep to evaluate at
        scale_factor: For converting back to Angstroms
        device: Device to run on

    Returns:
        Dictionary with RMSD and loss (ε MSE)
    """
    model.eval()
    x0 = x0.to(device)

    t_tensor = torch.tensor([t], device=device)
    x_t, noise = schedule.q_sample(x0, t_tensor)

    # Model predicts ε
    eps_pred = model(x_t, t_tensor)

    # MSE loss on ε
    loss = F.mse_loss(eps_pred, noise).item()

    # Convert ε_pred to x0_pred for RMSD
    # x_t = sqrt(α̅_t) * x0 + sqrt(1-α̅_t) * ε
    # => x0 = (x_t - sqrt(1-α̅_t) * ε) / sqrt(α̅_t)
    sqrt_alpha_bar = schedule.sqrt_alpha_bars[t]
    sqrt_one_minus_alpha_bar = schedule.sqrt_one_minus_alpha_bars[t]
    x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / sqrt_alpha_bar

    # RMSD in Angstroms - move to CPU for numpy
    x0_np = x0.squeeze(0).cpu().numpy() * scale_factor
    x0_pred_np = x0_pred.squeeze(0).cpu().numpy() * scale_factor
    rmsd_val = rmsd(x0_pred_np, x0_np, align=True)

    return {
        "loss": loss,
        "rmsd": rmsd_val,
    }


def plot_loss_curve(
    losses: list[float],
    total_steps: int,
    smoothing_window: int = 100,
    val_losses: list[float] | None = None,
    save_path: str = "loss_curve.png",
) -> None:
    """Plot and save the training loss curve.

    Args:
        losses: List of per-step training losses
        total_steps: Total training steps
        smoothing_window: Window size for smoothed loss
        val_losses: Optional list of validation losses (sampled less frequently)
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw train loss (semi-transparent)
    steps = np.arange(1, len(losses) + 1)
    ax.plot(steps, losses, alpha=0.3, color="blue", linewidth=0.5, label="Train (raw)")

    # Smoothed train loss
    if len(losses) >= smoothing_window:
        smoothed = np.convolve(
            losses, np.ones(smoothing_window) / smoothing_window, mode="valid"
        )
        smooth_steps = np.arange(smoothing_window, len(losses) + 1)
        ax.plot(
            smooth_steps,
            smoothed,
            color="blue",
            linewidth=2,
            label="Train (smoothed)",
        )

    # Val loss if provided
    if val_losses and len(val_losses) > 0:
        # Val losses are sampled every log_every steps
        val_steps = np.linspace(smoothing_window, len(losses), len(val_losses))
        ax.plot(
            val_steps,
            val_losses,
            color="orange",
            linewidth=2,
            marker="o",
            markersize=3,
            label="Val",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss Curve ({total_steps} steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale if loss varies a lot
    if len(losses) > 0 and max(losses) / (min(losses) + 1e-8) > 10:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nLoss curve saved to: {save_path}")
    plt.close()


@torch.no_grad()
def eval_batch_loss(
    model: DiffusionTransformer,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    mask: torch.Tensor | None,
    t: int = 100,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute ε MSE loss on a batch at fixed timestep.

    Args:
        model: The diffusion transformer (predicts ε)
        schedule: Diffusion schedule
        x0: (B, L, 3) clean coordinates (scaled)
        mask: (B, L) boolean mask for valid atoms
        t: Timestep to evaluate at
        device: Device to run on

    Returns:
        MSE loss value (on ε prediction)
    """
    model.eval()
    x0 = x0.to(device)
    if mask is not None:
        mask = mask.to(device)

    B = x0.shape[0]
    t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
    x_t, noise = schedule.q_sample(x0, t_tensor)
    eps_pred = model(x_t, t_tensor, mask=mask)

    if mask is not None:
        mask_3d = mask.unsqueeze(-1).expand_as(eps_pred)
        return F.mse_loss(eps_pred[mask_3d], noise[mask_3d]).item()
    else:
        return F.mse_loss(eps_pred, noise).item()


def load_single_chain(
    path: Path, scale_factor: float = 10.0
) -> tuple[torch.Tensor, str]:
    """Load and preprocess a single chain.

    Applies same preprocessing as ProteinDataset: center, align, scale.

    Args:
        path: Path to chain_set.jsonl
        scale_factor: Coordinate scaling factor

    Returns:
        x0: (1, L, 3) scaled, centered, aligned coordinates
        name: Chain name
    """
    result = get_one_chain(path)
    if result is None:
        raise ValueError("No chain found in dataset")

    name, seq, ca_coords = result
    # Same preprocessing as ProteinDataset: center -> align -> scale
    ca_centered, _ = center(ca_coords)
    ca_aligned = align_to_principal_axes(ca_centered)
    x0 = torch.from_numpy(ca_aligned).float().unsqueeze(0) / scale_factor

    return x0, name


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    args: argparse.Namespace,
    val_loss: float | None = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        step: Current training step
        args: Training arguments
        val_loss: Optional validation loss at this checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "args": vars(args),
    }
    if val_loss is not None:
        checkpoint["val_loss"] = val_loss
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Transformer")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument(
        "--overfit", action="store_true", help="Overfit on single chain"
    )
    parser.add_argument(
        "--num_chains", type=int, default=100, help="Number of chains to load"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Max sequence length (complete domains only)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--bond_loss_weight",
        type=float,
        default=0.1,
        help="Weight for auxiliary bond-length loss (λ)",
    )
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument(
        "--eval_every", type=int, default=10000, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=2000,
        help="Run reconstruction test every N steps (0 to disable)",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs",
        help="Directory to save run outputs (model, loss curve)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (0 to disable periodic saves)",
    )
    parser.add_argument(
        "--use_sliding_windows",
        action="store_true",
        help="Use sliding windows to extract crops from long chains (recovers more data)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride for sliding windows (default 64 = 50%% overlap with max_len=128)",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=40,
        help="Minimum segment length to keep",
    )
    parser.add_argument(
        "--keep_longest_only",
        action="store_true",
        default=True,
        help="Keep only the longest valid segment per chain (default: True)",
    )
    parser.add_argument(
        "--no_keep_longest_only",
        action="store_false",
        dest="keep_longest_only",
        help="Keep all valid segments per chain (disables --keep_longest_only)",
    )
    args = parser.parse_args()

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to log file
    log_path = run_dir / "train.log"
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"Run directory: {run_dir}")
    print(f"Logging to: {log_path}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path("data/chain_set.jsonl")
    splits_path = Path("data/chain_set_splits.json")
    scale_factor = 10.0

    print("=" * 60)
    print("Diffusion Transformer Training")
    print("=" * 60)

    # Print all hyperparameters
    print("\nHyperparameters:")
    print(f"  steps:            {args.steps}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  lr:               {args.lr}")
    print(f"  max_len:          {args.max_len}")
    print(f"  d_model:          {args.d_model}")
    print(f"  num_layers:       {args.num_layers}")
    print(f"  num_heads:        {args.num_heads}")
    print(f"  num_chains:       {args.num_chains}")
    print(f"  seed:             {args.seed}")
    print(f"  bond_loss_weight: {args.bond_loss_weight}")
    print(f"  log_every:        {args.log_every}")
    print(f"  eval_every:       {args.eval_every}")
    print(f"  sample_every:     {args.sample_every}")
    print(f"  save_every:       {args.save_every}")
    print(f"  sliding_windows:  {args.use_sliding_windows}")
    print(f"  stride:           {args.stride}")
    print(f"  min_len:          {args.min_len}")
    print(f"  keep_longest:     {args.keep_longest_only}")
    print(f"  overfit:          {args.overfit}")
    print(f"  scale_factor:     {scale_factor}")
    print("  diffusion_T:      1000")
    print(f"  device:           {device}")

    if args.overfit:
        print("\nMode: Single-chain overfitting (sanity check)")
        x0_raw, name = load_single_chain(data_path, scale_factor)
        L = x0_raw.shape[1]
        print(f"Chain: {name}, Length: {L}")

        # Print ground truth bond stats to verify dataset quality
        true_np = x0_raw.squeeze(0).cpu().numpy() * scale_factor
        true_bonds = ca_bond_lengths(true_np)
        print(f"Ground truth bonds: mean={true_bonds.mean():.2f} Å std={true_bonds.std():.2f} Å "
              f"valid%={((true_bonds > 3.6) & (true_bonds < 4.0)).mean() * 100:.1f}%")

        # Pad to max_len and create mask (tests same code path as dataset training)
        if L > args.max_len:
            print(f"Warning: chain length {L} > max_len {args.max_len}, truncating")
            x0_raw = x0_raw[:, :args.max_len]
            L = args.max_len

        x0 = torch.zeros(1, args.max_len, 3)
        x0[:, :L] = x0_raw[:, :L]
        overfit_mask = torch.zeros(1, args.max_len, dtype=torch.bool)
        overfit_mask[:, :L] = True
        print(f"Padded to max_len={args.max_len}, mask covers {L} real atoms")

        train_dataset = None
        val_dataset = None
    else:
        print("\nMode: Dataset training")
        print(f"Using pre-defined splits from {splits_path}")

        # Load pre-defined splits (CATH-stratified)
        splits = load_splits(splits_path)
        print(
            f"Split sizes: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}"
        )

        # Load training data
        if args.use_sliding_windows:
            # Sliding windows: extract overlapping crops from long chains
            print(f"Loading train windows (window={args.max_len}, stride={args.stride})...")
            train_chains = load_chain_windows_by_ids(
                data_path,
                chain_ids=splits["train"],
                window_size=args.max_len,
                stride=args.stride,
                min_len=args.min_len,
                keep_longest_only=args.keep_longest_only,
                limit=args.num_chains if args.num_chains else None,
                verbose=True,
            )

            print(f"Loading validation windows (window={args.max_len}, stride={args.stride})...")
            val_chains = load_chain_windows_by_ids(
                data_path,
                chain_ids=splits["validation"],
                window_size=args.max_len,
                stride=args.stride,
                min_len=args.min_len,
                keep_longest_only=args.keep_longest_only,
                verbose=True,
            )
        else:
            # Original behavior: complete domains only (reject chains > max_len)
            print(f"Loading train segments (len {args.min_len}-{args.max_len})...")
            train_chains = load_chain_segments_by_ids(
                data_path,
                chain_ids=splits["train"],
                min_len=args.min_len,
                max_len=args.max_len,
                keep_longest_only=args.keep_longest_only,
                limit=args.num_chains if args.num_chains else None,
                verbose=True,
            )

            print(f"Loading validation segments (len {args.min_len}-{args.max_len})...")
            val_chains = load_chain_segments_by_ids(
                data_path,
                chain_ids=splits["validation"],
                min_len=args.min_len,
                max_len=args.max_len,
                keep_longest_only=args.keep_longest_only,
                verbose=True,
            )

        # Shuffle train data (for training randomness)
        rng.shuffle(train_chains)

        data_type = "windows" if args.use_sliding_windows else "segments"
        print("After extraction:")
        print(f"  Train: {len(train_chains)} {data_type}")
        print(f"  Val:   {len(val_chains)} {data_type}")

        train_dataset = ProteinDataset(
            train_chains, max_len=args.max_len, scale_factor=scale_factor
        )
        val_dataset = ProteinDataset(
            val_chains, max_len=args.max_len, scale_factor=scale_factor
        )

        # Use first validation chain (padded) for reconstruction test
        val_coords = val_dataset.coords_list[0]
        L_val = len(val_coords)
        val_padded = np.zeros((args.max_len, 3))
        val_padded[:L_val] = val_coords
        val_x0_for_recon = torch.from_numpy(val_padded).float().unsqueeze(0)
        val_recon_mask = torch.zeros(args.max_len, dtype=torch.bool)
        val_recon_mask[:L_val] = True

        # === CREATE FROZEN ANCHOR SET ===
        # Grab a fixed batch from validation for ALL evaluations.
        # This removes "lucky batch" noise from metrics.
        print("Creating frozen evaluation anchor set...")
        n_anchor = min(32, len(val_dataset))
        anchor_indices = np.linspace(0, len(val_dataset) - 1, n_anchor, dtype=int)
        anchor_batch_list = []
        anchor_mask_list = []

        for idx in anchor_indices:
            coords = val_dataset.coords_list[idx]
            L = len(coords)
            padded = np.zeros((args.max_len, 3))
            padded[:L] = coords
            mask = np.zeros(args.max_len, dtype=bool)
            mask[:L] = True

            anchor_batch_list.append(padded)
            anchor_mask_list.append(mask)

        anchor_imgs = torch.from_numpy(np.stack(anchor_batch_list)).float()
        anchor_masks = torch.from_numpy(np.stack(anchor_mask_list)).bool()
        anchor_set = (anchor_imgs, anchor_masks)
        print(
            f"Anchor set: {anchor_imgs.shape[0]} chains, max_len={anchor_imgs.shape[1]}"
        )

    # Create model and move to device
    model = DiffusionTransformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create diffusion schedule and move to device
    schedule = DiffusionSchedule(T=1000)
    schedule = schedule.to(device)

    # Initialize evaluator (only for dataset mode)
    evaluator = None
    if not args.overfit:
        evaluator = Evaluator(model, schedule, device, scale_factor=scale_factor)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    # Initial evaluation
    print("\nInitial evaluation (t=100):")
    if args.overfit:
        x0_dev = x0.to(device)
        eval_result = eval_step(
            model, schedule, x0_dev, t=100, scale_factor=scale_factor, device=device
        )
        print(f"  Loss: {eval_result['loss']:.4f}, RMSD: {eval_result['rmsd']:.2f} Å")
    else:
        # Sample batches for initial eval
        train_batch, train_mask = train_dataset.sample_batch(4, rng)
        val_batch, val_mask = val_dataset.sample_batch(4, rng)
        train_eval = eval_batch_loss(
            model, schedule, train_batch, train_mask, t=100, device=device
        )
        val_eval = eval_batch_loss(
            model, schedule, val_batch, val_mask, t=100, device=device
        )
        print(f"  Train Loss: {train_eval:.4f}, Val Loss: {val_eval:.4f}")

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    if args.overfit:
        print(f"{'Step':>6} {'Loss':>10} {'LR':>12}")
    else:
        print(f"{'Step':>6} {'Loss':>10} {'LR':>12} {'Train':>10} {'Val':>10}")
    print("-" * 60)

    losses = []
    val_losses = []
    best_val_loss = float("inf")

    for step in range(1, args.steps + 1):
        # Get batch
        if args.overfit:
            # Repeat the same chain so each step trains on many timesteps/noise draws
            # Include mask to test same code path as dataset training
            batch_data = (
                x0.repeat(args.batch_size, 1, 1),  # (B, L, 3)
                overfit_mask.repeat(args.batch_size, 1),  # (B, L)
            )
        else:
            batch_data = train_dataset.sample_batch(args.batch_size, rng)

        result = train_step(
            model, schedule, batch_data, optimizer, device=device,
            scale_factor=scale_factor, bond_loss_weight=args.bond_loss_weight
        )
        losses.append(result["loss"])
        scheduler.step()

        if step % args.log_every == 0:
            avg_loss = np.mean(losses[-args.log_every :])
            lr = scheduler.get_last_lr()[0]
            if args.overfit:
                print(f"{step:>6} {avg_loss:>10.4f} {lr:>12.6f}")
            else:
                # Compute train loss on random batch, val loss on frozen anchor set
                train_batch, train_mask = train_dataset.sample_batch(4, rng)
                train_eval = eval_batch_loss(
                    model, schedule, train_batch, train_mask, t=100, device=device
                )
                # Use frozen anchor set for stable val metrics (less noise)
                anchor_x0, anchor_mask = anchor_set
                val_eval = eval_batch_loss(
                    model, schedule, anchor_x0, anchor_mask, t=100, device=device
                )
                val_losses.append(val_eval)
                print(
                    f"{step:>6} {avg_loss:>10.4f} {lr:>12.6f} {train_eval:>10.4f} {val_eval:>10.4f}"
                )

                # Save best checkpoint if val loss improved
                if val_eval < best_val_loss:
                    best_val_loss = val_eval
                    save_checkpoint(
                        run_dir / "best_model.pt",
                        model, optimizer, scheduler, step, args, val_loss=val_eval
                    )
                    print(f"       New best val loss: {val_eval:.4f} (saved)")

        # Periodic checkpoint saving
        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = run_dir / f"checkpoint_step{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step, args)
            print(f"\n       Checkpoint saved: {ckpt_path.name}\n")

        # Run reconstruction test (meaningful metric for denoising ability)
        if args.sample_every > 0 and step % args.sample_every == 0:
            recon_target = x0 if args.overfit else val_x0_for_recon
            recon_target = recon_target.to(device)

            # Use mask in overfit mode too (tests same code path as dataset training)
            if args.overfit:
                recon_mask = overfit_mask.to(device)  # (1, L)
            else:
                recon_mask = val_recon_mask.unsqueeze(0).to(device)  # (1, L)

            sampler = DiffusionSampler(model, schedule)
            t_start = 300
            t_tensor = torch.tensor([t_start], device=device)

            # fixed noise WITHOUT touching global torch RNG
            g = torch.Generator(device="cpu")
            g.manual_seed(0)
            noise = torch.randn(recon_target.shape, generator=g, device="cpu", dtype=recon_target.dtype).to(device)

            x_t, _ = schedule.q_sample(recon_target, t_tensor, noise=noise)

            # Deterministic sampling (add_noise=False) for reproducible evaluation
            recon = sampler.sample_from(x_t, start_t=t_start, mask=recon_mask, verbose=False, add_noise=False)

            recon_np = recon.squeeze(0).detach().cpu().numpy() * scale_factor
            true_np  = recon_target.squeeze(0).detach().cpu().numpy() * scale_factor

            if recon_mask is not None:
                valid = recon_mask.squeeze(0).detach().cpu().numpy().astype(bool)
                recon_rmsd = rmsd(recon_np[valid], true_np[valid], align=True)
                coords_for_bonds = recon_np[valid]
            else:
                recon_rmsd = rmsd(recon_np, true_np, align=True)
                coords_for_bonds = recon_np

            # Bond metrics (cheap early warning for broken local geometry)
            bonds = ca_bond_lengths(coords_for_bonds)
            bond_valid_pct = ((bonds > 3.6) & (bonds < 4.0)).mean() * 100.0

            mode = "Train" if args.overfit else "Val"
            print(f"\n       Reconstruction ({mode}) from t={t_start}: RMSD={recon_rmsd:.2f} Å")
            print(f"       Bonds: mean={bonds.mean():.2f} Å  std={bonds.std():.2f} Å  valid%={bond_valid_pct:.1f}%\n")

        # Run full evaluation scorecard (dataset mode only)
        if evaluator is not None and step % args.eval_every == 0:
            print(f"\nRunning detailed evaluation at step {step}...")
            metrics = evaluator.run_full_eval(
                anchor_batch=anchor_set,
                sample_batch_size=16,  # Small batch for speed during training
            )
            print_eval_report(metrics, step)

    # Final evaluation
    print(f"\n{'=' * 60}")
    print("Final Evaluation")
    print("=" * 60)

    if args.overfit:
        # Single chain eval
        print(f"{'Timestep':>10} {'Loss':>10} {'RMSD (Å)':>10}")
        print("-" * 35)
        for t in [100, 250, 500, 750, 900]:
            eval_result = eval_step(
                model, schedule, x0, t=t, scale_factor=scale_factor, device=device
            )
            print(f"{t:>10} {eval_result['loss']:>10.4f} {eval_result['rmsd']:>10.2f}")
    else:
        # Train vs Val comparison
        print(f"{'Timestep':>10} {'Train Loss':>12} {'Val Loss':>12}")
        print("-" * 40)
        for t in [100, 250, 500, 750, 900]:
            train_batch, train_mask = train_dataset.sample_batch(8, rng)
            val_batch, val_mask = val_dataset.sample_batch(8, rng)
            train_loss = eval_batch_loss(
                model, schedule, train_batch, train_mask, t=t, device=device
            )
            val_loss = eval_batch_loss(
                model, schedule, val_batch, val_mask, t=t, device=device
            )
            print(f"{t:>10} {train_loss:>12.4f} {val_loss:>12.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Training Complete")
    print("=" * 60)
    print(f"Final train loss (avg last 100): {np.mean(losses[-100:]):.4f}")
    print(f"Best train loss: {min(losses):.4f}")
    if not args.overfit and val_losses:
        print(f"Final val loss (avg last 10): {np.mean(val_losses[-10:]):.4f}")
        print(f"Best val loss: {min(val_losses):.4f}")

    # Success criteria
    if args.overfit:
        final_eval = eval_step(
            model, schedule, x0, t=250, scale_factor=scale_factor, device=device
        )
        if final_eval["rmsd"] < 5.0:
            print(
                f"\n✓ Overfitting successful! RMSD at t=250: {final_eval['rmsd']:.2f} Å < 5.0 Å"
            )
        else:
            print(
                f"\n✗ Overfitting incomplete. RMSD at t=250: {final_eval['rmsd']:.2f} Å >= 5.0 Å"
            )
            print("  Try more steps or adjust hyperparameters.")
    else:
        # Check generalization gap
        train_batch, train_mask = train_dataset.sample_batch(8, rng)
        val_batch, val_mask = val_dataset.sample_batch(8, rng)
        final_train = eval_batch_loss(
            model, schedule, train_batch, train_mask, t=100, device=device
        )
        final_val = eval_batch_loss(
            model, schedule, val_batch, val_mask, t=100, device=device
        )
        gap = final_val - final_train

        print(f"\nGeneralization gap (Val - Train): {gap:.4f}")
        if gap < 0.1:
            print("✓ Good generalization! Train and val loss are close.")
        elif gap < 0.3:
            print(
                "~ Moderate gap. Model is learning but may benefit from more data/regularization."
            )
        else:
            print(
                "✗ Large gap. Model may be memorizing. Try more data or regularization."
            )

        # Final comprehensive evaluation
        print("\n" + "=" * 60)
        print("FINAL COMPREHENSIVE EVALUATION")
        print("=" * 60)
        final_metrics = evaluator.run_full_eval(
            anchor_batch=anchor_set,
            sample_batch_size=32,  # Larger batch for final eval
        )
        print_eval_report(final_metrics, args.steps)

    # Plot and save loss curve
    loss_curve_path = run_dir / "loss_curve.png"
    plot_loss_curve(
        losses,
        args.steps,
        args.log_every,
        val_losses if not args.overfit else None,
        save_path=str(loss_curve_path),
    )

    # Save final model (with optimizer/scheduler for potential resumption)
    model_path = run_dir / "model.pt"
    final_val = np.mean(val_losses[-10:]) if (not args.overfit and val_losses) else None
    save_checkpoint(model_path, model, optimizer, scheduler, args.steps, args, val_loss=final_val)

    # Also save a simple dict for backward compatibility
    save_dict = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "final_train_loss": np.mean(losses[-100:]),
    }
    if final_val is not None:
        save_dict["final_val_loss"] = final_val
    torch.save(save_dict, run_dir / "model_simple.pt")

    print(f"Model saved to: {model_path}")
    if not args.overfit:
        print(f"Best model saved to: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
