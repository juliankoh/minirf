"""Toy training script for debugging diffusion on synthetic data.

This is a minimal sanity-check script that trains on a synthetic straight-line
chain where we know the "right answer" (bond lengths exactly 3.8 Å).

Usage:
    uv run python -m src.train_toy --steps 1000

Why synthetic data for debugging:
    - You know the correct answer (perfect 3.8 Å bonds)
    - Very short chains (L=16/32) make runs super fast
    - No confounders (weird geometries, missing atoms, etc.)
    - Easy to verify if the model is learning anything

Success criteria:
    - Reconstruction RMSD should drop below 1.0 Å within ~500 steps
    - Bond lengths should stay close to 3.8 Å (mean ≈ 3.8, valid% > 90%)
    - If bonds are garbage but RMSD looks okay, something is wrong
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .diffusion import DiffusionSchedule
from .geom import ca_bond_lengths, rmsd
from .model import DiffusionTransformer
from .sampler import DiffusionSampler


def make_toy_chain(L: int = 32, scale_factor: float = 10.0) -> torch.Tensor:
    """Generate a synthetic straight-line chain.

    Creates a chain of L atoms spaced exactly 3.8 Å apart along the X-axis.
    This is the simplest possible protein-like structure for debugging.

    Args:
        L: Number of atoms (default 32)
        scale_factor: Coordinate scaling factor (default 10.0)

    Returns:
        x0: (1, L, 3) tensor of coordinates (scaled)
    """
    bond = 3.8 / scale_factor  # Scaled bond length
    x0 = torch.zeros(1, L, 3)
    x0[0, :, 0] = torch.arange(L, dtype=torch.float32) * bond
    return x0


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
    """Single training step with ε-prediction, bond loss, and self-conditioning."""
    model.train()

    B = x0.shape[0]
    x0 = x0.to(device)

    # Sample random timesteps
    t = torch.randint(0, schedule.T, (B,), device=device)

    # Forward diffusion: add noise
    x_t, noise = schedule.q_sample(x0, t)

    # Self-conditioning: 50% of the time, do a first pass
    x0_self_cond = None
    used_self_cond = False
    if torch.rand(1).item() < self_cond_prob:
        with torch.no_grad():
            eps_pred_initial = model(x_t, t, mask=None, x0_self_cond=None)
            sqrt_ab = schedule.sqrt_alpha_bars[t].view(B, 1, 1)
            sqrt_omb = schedule.sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
            x0_self_cond = (x_t - sqrt_omb * eps_pred_initial) / sqrt_ab
        used_self_cond = True

    # Model prediction
    eps_pred = model(x_t, t, mask=None, x0_self_cond=x0_self_cond)

    # MSE loss on ε
    eps_loss = F.mse_loss(eps_pred, noise)

    # Auxiliary bond-length loss (only at low/moderate noise)
    # At high t, sqrt_ab is tiny, so x0_pred = (x_t - sqrt_omb*eps)/sqrt_ab
    # amplifies errors massively, making bond loss noisy and unhelpful
    sqrt_ab = schedule.sqrt_alpha_bars[t].view(B, 1, 1)
    sqrt_omb = schedule.sqrt_one_minus_alpha_bars[t].view(B, 1, 1)
    x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab

    bond_vecs = x0_pred[:, 1:] - x0_pred[:, :-1]
    bond_lens = torch.linalg.norm(bond_vecs, dim=-1)  # (B, L-1)
    target = 3.8 / scale_factor

    # Only apply bond loss for t < 200 (low/moderate noise)
    bond_t_max = 200
    bond_mask = (t < bond_t_max).float().view(B, 1)  # (B, 1) broadcasts to (B, L-1)
    bond_sq_err = (bond_lens - target) ** 2  # (B, L-1)
    if bond_mask.sum() > 0:
        bond_loss = (bond_sq_err * bond_mask).sum() / bond_mask.sum().clamp(min=1.0) / bond_lens.shape[1]
    else:
        bond_loss = torch.tensor(0.0, device=device)

    # Combined loss
    loss = eps_loss + bond_loss_weight * bond_loss

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "eps_loss": eps_loss.item(),
        "bond_loss": bond_loss.item(),
        "t_mean": t.float().mean().item(),
        "self_cond": float(used_self_cond),
    }


@torch.no_grad()
def eval_reconstruction(
    model: DiffusionTransformer,
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    device: torch.device,
    scale_factor: float = 10.0,
    t_start: int = 300,
    use_self_cond: bool = False,
) -> dict:
    """Run deterministic reconstruction and measure RMSD + bond lengths.

    Args:
        model: The diffusion transformer
        schedule: Diffusion schedule
        x0: (1, L, 3) clean coordinates (scaled)
        device: Device to run on
        scale_factor: For converting back to Angstroms
        t_start: Timestep to start denoising from
        use_self_cond: Whether to use self-conditioning during sampling

    Returns:
        Dictionary with RMSD and bond statistics
    """
    model.eval()
    x0 = x0.to(device)

    sampler = DiffusionSampler(model, schedule)
    t_tensor = torch.tensor([t_start], device=device)

    # Fixed noise for reproducibility
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    noise = torch.randn(x0.shape, generator=g, device="cpu", dtype=x0.dtype).to(device)

    x_t, _ = schedule.q_sample(x0, t_tensor, noise=noise)

    # Deterministic sampling (self-conditioning off by default for debugging)
    recon = sampler.sample_from(
        x_t, start_t=t_start, mask=None, verbose=False,
        add_noise=False, use_self_cond=use_self_cond
    )

    # Convert to Angstroms
    recon_np = recon.squeeze(0).cpu().numpy() * scale_factor
    true_np = x0.squeeze(0).cpu().numpy() * scale_factor

    # RMSD
    recon_rmsd = rmsd(recon_np, true_np, align=True)

    # Bond lengths
    bonds = ca_bond_lengths(recon_np)
    valid_pct = ((bonds > 3.6) & (bonds < 4.0)).mean()

    return {
        "rmsd": recon_rmsd,
        "bond_mean": bonds.mean(),
        "bond_std": bonds.std(),
        "bond_valid_pct": valid_pct * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Toy training on synthetic chain")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (samples many timesteps per step)")
    parser.add_argument("--chain_len", type=int, default=32, help="Synthetic chain length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension (small for speed)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (small)")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--bond_loss_weight",
        type=float,
        default=1.0,
        help="Weight for bond loss (higher = easier to debug bond wiring)",
    )
    parser.add_argument(
        "--self_cond",
        action="store_true",
        help="Enable self-conditioning (off by default for debugging clarity)",
    )
    parser.add_argument("--log_every", type=int, default=50, help="Log every N steps")
    parser.add_argument("--sample_every", type=int, default=200, help="Run reconstruction test every N steps")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs_toy",
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TOY TRAINING: Synthetic Straight-Line Chain")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Generate synthetic chain
    scale_factor = 10.0
    x0 = make_toy_chain(L=args.chain_len, scale_factor=scale_factor)
    print(f"\nSynthetic chain: L={args.chain_len}")
    print(f"  Bond length (target): 3.8 Å (scaled: {3.8 / scale_factor:.3f})")

    # Verify toy chain
    toy_coords = x0.squeeze(0).numpy() * scale_factor
    toy_bonds = ca_bond_lengths(toy_coords)
    print(f"  Bond length (actual): {toy_bonds.mean():.2f} ± {toy_bonds.std():.4f} Å")

    # Keep single example for evaluation, expand batch for training
    # With batch_size=64 and T=1000, each step samples 64 different timesteps
    # This ensures good coverage across all timesteps
    x0_single = x0  # (1, L, 3) for eval
    x0 = x0.repeat(args.batch_size, 1, 1)  # (B, L, 3) for training
    print(f"  Training batch: {x0.shape} (same chain repeated {args.batch_size}x)")

    # Hyperparameters
    print("\nHyperparameters:")
    print(f"  steps:            {args.steps}")
    print(f"  batch_size:       {args.batch_size}")
    print(f"  lr:               {args.lr}")
    print(f"  d_model:          {args.d_model}")
    print(f"  num_layers:       {args.num_layers}")
    print(f"  num_heads:        {args.num_heads}")
    print(f"  bond_loss_weight: {args.bond_loss_weight}")
    print(f"  self_cond:        {args.self_cond}")
    print(f"  log_every:        {args.log_every}")
    print(f"  sample_every:     {args.sample_every}")

    # Create model
    model = DiffusionTransformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Create diffusion schedule
    schedule = DiffusionSchedule(T=1000)
    schedule = schedule.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    # Initial reconstruction test
    print("\nInitial reconstruction (before training):")
    init_metrics = eval_reconstruction(
        model, schedule, x0_single, device, scale_factor, use_self_cond=args.self_cond
    )
    print(f"  RMSD: {init_metrics['rmsd']:.2f} Å")
    print(f"  Bonds: mean={init_metrics['bond_mean']:.2f} Å, valid%={init_metrics['bond_valid_pct']:.1f}%")

    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    print(f"{'Step':>6} {'Loss':>10} {'Eps':>10} {'Bond':>10} {'LR':>12}")
    print("-" * 60)

    losses = []

    for step in range(1, args.steps + 1):
        result = train_step(
            model, schedule, x0, optimizer, device=device,
            scale_factor=scale_factor, bond_loss_weight=args.bond_loss_weight,
            self_cond_prob=0.5 if args.self_cond else 0.0,
        )
        losses.append(result["loss"])
        scheduler.step()

        if step % args.log_every == 0:
            avg_loss = np.mean(losses[-args.log_every:])
            lr = scheduler.get_last_lr()[0]
            print(
                f"{step:>6} {avg_loss:>10.4f} {result['eps_loss']:>10.4f} "
                f"{result['bond_loss']:>10.4f} {lr:>12.6f}"
            )

        # Reconstruction test with bond length analysis
        if args.sample_every > 0 and step % args.sample_every == 0:
            metrics = eval_reconstruction(
                model, schedule, x0_single, device, scale_factor, use_self_cond=args.self_cond
            )
            print(f"\n       Reconstruction RMSD from t=300: {metrics['rmsd']:.2f} Å")
            print(
                f"       Bonds: mean={metrics['bond_mean']:.2f} Å  "
                f"std={metrics['bond_std']:.2f} Å  "
                f"valid%={metrics['bond_valid_pct']:.1f}%"
            )
            print()

    # Final evaluation
    print("=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_metrics = eval_reconstruction(
        model, schedule, x0_single, device, scale_factor, use_self_cond=args.self_cond
    )
    print(f"Reconstruction RMSD: {final_metrics['rmsd']:.2f} Å")
    print("Bond lengths:")
    print(f"  Mean:    {final_metrics['bond_mean']:.2f} Å (target: 3.80 Å)")
    print(f"  Std:     {final_metrics['bond_std']:.2f} Å (target: ~0.00 Å)")
    print(f"  Valid%:  {final_metrics['bond_valid_pct']:.1f}% (3.6-4.0 Å range)")

    # Success criteria
    print("\n" + "-" * 60)
    if final_metrics["rmsd"] < 1.0:
        print(f"✓ RMSD check PASSED: {final_metrics['rmsd']:.2f} Å < 1.0 Å")
    else:
        print(f"✗ RMSD check FAILED: {final_metrics['rmsd']:.2f} Å >= 1.0 Å")

    if final_metrics["bond_valid_pct"] > 90:
        print(f"✓ Bond check PASSED: {final_metrics['bond_valid_pct']:.1f}% > 90%")
    else:
        print(f"✗ Bond check FAILED: {final_metrics['bond_valid_pct']:.1f}% <= 90%")

    if abs(final_metrics["bond_mean"] - 3.8) < 0.2:
        print(f"✓ Mean bond PASSED: {final_metrics['bond_mean']:.2f} Å ≈ 3.8 Å")
    else:
        print(f"✗ Mean bond FAILED: {final_metrics['bond_mean']:.2f} Å far from 3.8 Å")

    # Save model
    model_path = run_dir / "toy_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "final_metrics": final_metrics,
    }, model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
