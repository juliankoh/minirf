"""Training loop for the Diffusion Transformer.

Usage:
    # Overfit on single chain (sanity check)
    uv run python -m src.train --overfit --steps 2000

    # Train on dataset (100 chains, fixed length crops)
    uv run python -m src.train --steps 10000 --num_chains 100

    # Larger training run
    uv run python -m src.train --steps 50000 --num_chains 500 --batch_size 8

Training Objective:
    Given noisy coordinates x_t at timestep t, predict clean coordinates x0.
    Loss = MSE(x0_pred, x0)

Data Augmentation:
    Random rotations are applied to make the model robust to different
    orientations (since we don't have full SE(3) equivariance).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .data_cath import filter_chains, get_one_chain
from .diffusion import DiffusionSchedule
from .geom import apply_random_rotation, center, rmsd
from .model import DiffusionTransformer
from .sampler import DiffusionSampler


class ProteinDataset:
    """Simple dataset for protein CA coordinates.

    Loads chains into memory and provides random sampling with
    fixed-length cropping for batching.
    """

    def __init__(
        self,
        chains: list[dict],
        crop_len: int = 128,
        scale_factor: float = 10.0,
    ):
        """Initialize dataset.

        Args:
            chains: List of chain dicts with 'coords' key containing CA coords
            crop_len: Fixed length to crop/pad sequences to
            scale_factor: Coordinate scaling factor
        """
        self.chains = chains
        self.crop_len = crop_len
        self.scale_factor = scale_factor

        # Preprocess all chains: center and scale
        self.coords_list = []
        for chain in chains:
            ca_coords = np.array(chain["coords"]["CA"])
            ca_centered, _ = center(ca_coords)
            self.coords_list.append(ca_centered / scale_factor)

    def __len__(self) -> int:
        return len(self.chains)

    def sample_batch(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:  # Updated return type
        """Sample a batch of fixed-length coordinate crops."""
        batch_coords = []
        batch_mask = []  # New

        for _ in range(batch_size):
            idx = rng.integers(0, len(self.coords_list))
            coords = self.coords_list[idx]
            L = len(coords)

            if L >= self.crop_len:
                start = rng.integers(0, L - self.crop_len + 1)
                crop = coords[start : start + self.crop_len]
                mask = np.ones(self.crop_len, dtype=bool)  # All real
            else:
                crop = np.zeros((self.crop_len, 3))
                crop[:L] = coords
                mask = np.zeros(self.crop_len, dtype=bool)  # Mixed
                mask[:L] = True

            batch_coords.append(crop)
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
    augment: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    """Single training step.

    Args:
        model: The diffusion transformer
        schedule: Diffusion schedule for q_sample
        x0: (B, L, 3) clean coordinates (scaled)
        optimizer: Optimizer
        device: Device to run on (cuda/mps/cpu)
        augment: Whether to apply random rotation
        rng: Random generator for augmentation

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

    # Random rotation augmentation (per-sample) - done on CPU before moving to device
    if augment:
        x0_aug = x0.clone()
        for i in range(B):
            x0_np = x0[i].cpu().numpy()
            x0_rot, _ = apply_random_rotation(x0_np, rng)
            x0_aug[i] = torch.from_numpy(x0_rot).float()
        x0 = x0_aug

    # Move to device
    x0 = x0.to(device)
    if mask is not None:
        mask = mask.to(device)

    # Sample random timesteps
    t = torch.randint(0, schedule.T, (B,), device=device)

    # Forward diffusion: add noise
    x_t, noise = schedule.q_sample(x0, t)

    # Model prediction: predict x0
    x0_pred = model(x_t, t, mask=mask)

    # MASKED LOSS CALCULATION
    if mask is not None:
        # Expand mask for coordinates: (B, L) -> (B, L, 3)
        mask_3d = mask.unsqueeze(-1).expand_as(x0_pred)
        # Only compute error on real atoms
        loss = F.mse_loss(x0_pred[mask_3d], x0[mask_3d])
    else:
        loss = F.mse_loss(x0_pred, x0)

    # Backprop
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {
        "loss": loss.item(),
        "t_mean": t.float().mean().item(),
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
    """Evaluate model at a fixed timestep.

    Args:
        model: The diffusion transformer
        schedule: Diffusion schedule
        x0: (1, L, 3) clean coordinates (scaled)
        t: Timestep to evaluate at
        scale_factor: For converting back to Angstroms
        device: Device to run on

    Returns:
        Dictionary with RMSD and loss
    """
    model.eval()
    x0 = x0.to(device)

    t_tensor = torch.tensor([t], device=device)
    x_t, noise = schedule.q_sample(x0, t_tensor)

    x0_pred = model(x_t, t_tensor)

    # MSE loss
    loss = F.mse_loss(x0_pred, x0).item()

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
            label=f"Train (smoothed)",
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
    """Compute MSE loss on a batch at fixed timestep.

    Args:
        model: The diffusion transformer
        schedule: Diffusion schedule
        x0: (B, L, 3) clean coordinates (scaled)
        mask: (B, L) boolean mask for valid atoms
        t: Timestep to evaluate at
        device: Device to run on

    Returns:
        MSE loss value
    """
    model.eval()
    x0 = x0.to(device)
    if mask is not None:
        mask = mask.to(device)

    B = x0.shape[0]
    t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
    x_t, _ = schedule.q_sample(x0, t_tensor)
    x0_pred = model(x_t, t_tensor, mask=mask)

    if mask is not None:
        mask_3d = mask.unsqueeze(-1).expand_as(x0_pred)
        return F.mse_loss(x0_pred[mask_3d], x0[mask_3d]).item()
    else:
        return F.mse_loss(x0_pred, x0).item()


def load_single_chain(
    path: Path, scale_factor: float = 10.0
) -> tuple[torch.Tensor, str]:
    """Load and preprocess a single chain.

    Args:
        path: Path to chain_set.jsonl
        scale_factor: Coordinate scaling factor

    Returns:
        x0: (1, L, 3) scaled, centered coordinates
        name: Chain name
    """
    result = get_one_chain(path)
    if result is None:
        raise ValueError("No chain found in dataset")

    name, seq, ca_coords = result
    ca_centered, _ = center(ca_coords)
    x0 = torch.from_numpy(ca_centered).float().unsqueeze(0) / scale_factor

    return x0, name


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
        "--crop_len", type=int, default=128, help="Crop length for batching"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument(
        "--eval_every", type=int, default=500, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--sample_every", type=int, default=2000, help="Run sampler every N steps (0 to disable)"
    )
    parser.add_argument(
        "--save_path", type=str, default="model.pt", help="Path to save trained model"
    )
    args = parser.parse_args()

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
    scale_factor = 10.0

    print("=" * 60)
    print("Diffusion Transformer Training")
    print("=" * 60)

    if args.overfit:
        print("\nMode: Single-chain overfitting (sanity check)")
        x0, name = load_single_chain(data_path, scale_factor)
        print(f"Chain: {name}, Length: {x0.shape[1]}")
        train_dataset = None
        val_dataset = None
    else:
        print(f"\nMode: Dataset training")
        print(f"Loading {args.num_chains} chains (min_len={args.crop_len})...")
        all_chains = filter_chains(
            data_path,
            min_len=args.crop_len,
            max_len=512,
            limit=args.num_chains,
            verbose=False,
        )

        # Shuffle and split 80/20
        rng.shuffle(all_chains)
        split_idx = int(len(all_chains) * 0.8)
        train_chains = all_chains[:split_idx]
        val_chains = all_chains[split_idx:]

        print(f"Loaded {len(all_chains)} chains")
        print(f"  Train: {len(train_chains)} chains")
        print(f"  Val:   {len(val_chains)} chains")

        train_dataset = ProteinDataset(
            train_chains, crop_len=args.crop_len, scale_factor=scale_factor
        )
        val_dataset = ProteinDataset(
            val_chains, crop_len=args.crop_len, scale_factor=scale_factor
        )

        # Use first validation chain for sampler evaluation (true generalization test)
        val_x0_for_sampler = torch.from_numpy(val_dataset.coords_list[0]).float().unsqueeze(0)

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

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    # Initial evaluation
    print(f"\nInitial evaluation (t=100):")
    if args.overfit:
        x0_dev = x0.to(device)
        eval_result = eval_step(model, schedule, x0_dev, t=100, scale_factor=scale_factor, device=device)
        print(f"  Loss: {eval_result['loss']:.4f}, RMSD: {eval_result['rmsd']:.2f} Å")
    else:
        # Sample batches for initial eval
        train_batch, train_mask = train_dataset.sample_batch(4, rng)
        val_batch, val_mask = val_dataset.sample_batch(4, rng)
        train_eval = eval_batch_loss(model, schedule, train_batch, train_mask, t=100, device=device)
        val_eval = eval_batch_loss(model, schedule, val_batch, val_mask, t=100, device=device)
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
    for step in range(1, args.steps + 1):
        # Get batch
        if args.overfit:
            batch_data = x0  # Tuple logic handles this being just tensor
        else:
            batch_data = train_dataset.sample_batch(args.batch_size, rng)

        result = train_step(
            model, schedule, batch_data, optimizer, device=device, augment=True, rng=rng
        )
        losses.append(result["loss"])
        scheduler.step()

        if step % args.log_every == 0:
            avg_loss = np.mean(losses[-args.log_every :])
            lr = scheduler.get_last_lr()[0]
            if args.overfit:
                print(f"{step:>6} {avg_loss:>10.4f} {lr:>12.6f}")
            else:
                # Compute train and val loss
                train_batch, train_mask = train_dataset.sample_batch(4, rng)
                val_batch, val_mask = val_dataset.sample_batch(4, rng)
                train_eval = eval_batch_loss(model, schedule, train_batch, train_mask, t=100, device=device)
                val_eval = eval_batch_loss(model, schedule, val_batch, val_mask, t=100, device=device)
                val_losses.append(val_eval)
                print(f"{step:>6} {avg_loss:>10.4f} {lr:>12.6f} {train_eval:>10.4f} {val_eval:>10.4f}")

        # Run sampler to test generative quality
        if args.sample_every > 0 and step % args.sample_every == 0:
            if args.overfit:
                sampler_target = x0
            else:
                sampler_target = val_x0_for_sampler

            print(f"\n       Running sampler at step {step}...")
            sampler = DiffusionSampler(model, schedule)
            L = sampler_target.shape[1]
            sample = sampler.sample(shape=(1, L, 3), verbose=False, device=device)

            # Compute generative RMSD - move to CPU for numpy
            sample_np = sample.squeeze().cpu().numpy() * scale_factor
            true_np = sampler_target.squeeze().cpu().numpy() * scale_factor
            gen_rmsd = rmsd(sample_np, true_np, align=True)
            mode = "Train" if args.overfit else "Val"
            print(f"       Generative RMSD ({mode}): {gen_rmsd:.2f} Å\n")

    # Final evaluation
    print(f"\n{'=' * 60}")
    print("Final Evaluation")
    print("=" * 60)

    if args.overfit:
        # Single chain eval
        print(f"{'Timestep':>10} {'Loss':>10} {'RMSD (Å)':>10}")
        print("-" * 35)
        for t in [100, 250, 500, 750, 900]:
            eval_result = eval_step(model, schedule, x0, t=t, scale_factor=scale_factor, device=device)
            print(f"{t:>10} {eval_result['loss']:>10.4f} {eval_result['rmsd']:>10.2f}")
    else:
        # Train vs Val comparison
        print(f"{'Timestep':>10} {'Train Loss':>12} {'Val Loss':>12}")
        print("-" * 40)
        for t in [100, 250, 500, 750, 900]:
            train_batch, train_mask = train_dataset.sample_batch(8, rng)
            val_batch, val_mask = val_dataset.sample_batch(8, rng)
            train_loss = eval_batch_loss(model, schedule, train_batch, train_mask, t=t, device=device)
            val_loss = eval_batch_loss(model, schedule, val_batch, val_mask, t=t, device=device)
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
        final_eval = eval_step(model, schedule, x0, t=250, scale_factor=scale_factor, device=device)
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
        final_train = eval_batch_loss(model, schedule, train_batch, train_mask, t=100, device=device)
        final_val = eval_batch_loss(model, schedule, val_batch, val_mask, t=100, device=device)
        gap = final_val - final_train

        print(f"\nGeneralization gap (Val - Train): {gap:.4f}")
        if gap < 0.1:
            print("✓ Good generalization! Train and val loss are close.")
        elif gap < 0.3:
            print("~ Moderate gap. Model is learning but may benefit from more data/regularization.")
        else:
            print("✗ Large gap. Model may be memorizing. Try more data or regularization.")

    # Plot and save loss curve
    plot_loss_curve(losses, args.steps, args.log_every, val_losses if not args.overfit else None)

    # Save model
    save_dict = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "final_train_loss": np.mean(losses[-100:]),
    }
    if not args.overfit and val_losses:
        save_dict["final_val_loss"] = np.mean(val_losses[-10:])

    torch.save(save_dict, args.save_path)
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
