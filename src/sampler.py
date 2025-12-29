"""Diffusion sampler for generating protein structures from noise.

Usage:
    uv run python -m src.sampler

The sampler reverses the diffusion process:
    1. Start with pure Gaussian noise x_T
    2. Iteratively denoise: x_T -> x_{T-1} -> ... -> x_0
    3. At each step, model predicts ε (noise), we compute posterior mean, take a step

Why ε-prediction works well:
    - At high t, x_t ≈ ε, so predicting ε is easy (strong correlation)
    - At low t, x_t ≈ x0, so ε = (x_t - x0)/σ is also learnable
    - Makes learning more uniform across timesteps than x0-prediction
"""

import torch
from tqdm import tqdm

from .diffusion import DiffusionSchedule
from .model import DiffusionTransformer


class DiffusionSampler:
    """Generates proteins by reversing the diffusion process.

    Implements standard DDPM posterior sampling for models that predict ε (noise).
    """

    def __init__(self, model: DiffusionTransformer, schedule: DiffusionSchedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        verbose: bool = True,
        device: torch.device | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples from pure noise.

        Args:
            shape: (B, L, 3) desired output shape
            verbose: Show progress bar
            device: Device to run on (defaults to model's device)
            mask: (B, L) boolean mask for valid positions (True=valid, False=padding)
                  If None, all positions are treated as valid.

        Returns:
            (B, L, 3) generated coordinates (scaled)
        """
        self.model.eval()
        B, L, _ = shape

        # Infer device from model if not specified
        if device is None:
            device = next(self.model.parameters()).device

        # 1. Start with pure Gaussian noise
        x_t = torch.randn(shape, device=device)

        # Default mask: all positions valid
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)

        # Reverse timesteps: T-1, T-2, ..., 0
        T = self.schedule.T
        iterator = range(T - 1, -1, -1)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling")

        # Expand mask for coordinate masking: (B, L) -> (B, L, 1)
        mask_3d = mask.unsqueeze(-1)

        for i in iterator:
            t = torch.full((B,), i, dtype=torch.long, device=device)

            # 2. Model predicts ε (noise) from current noisy state x_t
            pred_eps = self.model(x_t, t, mask=mask)

            # 3. Take one step: x_t -> x_{t-1}
            x_t = self.p_sample(x_t, pred_eps, t, step_index=i)

            # 4. Keep padding frozen at zero (prevents pad drift)
            x_t = x_t * mask_3d

        return x_t

    @torch.no_grad()
    def sample_from(
        self,
        x_t: torch.Tensor,
        start_t: int,
        verbose: bool = True,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples starting from a given noisy state.

        This is useful for reconstruction tests: noise a real structure to
        timestep t, then denoise it back. RMSD should decrease with training.

        Args:
            x_t: (B, L, 3) noisy coordinates at timestep start_t
            start_t: Timestep to start denoising from (0 to T-1)
            verbose: Show progress bar
            mask: (B, L) boolean mask for valid positions (True=valid, False=padding)
                  If None, all positions are treated as valid.

        Returns:
            (B, L, 3) generated coordinates (scaled)
        """
        self.model.eval()
        B, L, _ = x_t.shape
        device = x_t.device

        # Default mask: all positions valid
        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=device)
        else:
            mask = mask.to(device)

        # Reverse timesteps: start_t, start_t-1, ..., 0
        iterator = range(start_t, -1, -1)
        if verbose:
            iterator = tqdm(iterator, desc=f"Sampling from t={start_t}")

        # Expand mask for coordinate masking: (B, L) -> (B, L, 1)
        mask_3d = mask.unsqueeze(-1)

        for i in iterator:
            t = torch.full((B,), i, dtype=torch.long, device=device)

            # Model predicts ε (noise) from current noisy state x_t
            pred_eps = self.model(x_t, t, mask=mask)

            # Take one step: x_t -> x_{t-1}
            x_t = self.p_sample(x_t, pred_eps, t, step_index=i)

            # Keep padding frozen at zero (prevents pad drift)
            x_t = x_t * mask_3d

        return x_t

    def p_sample(
        self,
        x_t: torch.Tensor,
        pred_eps: torch.Tensor,
        t: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Calculate x_{t-1} given x_t and predicted ε.

        Uses the DDPM posterior formula:
            x_{t-1} = 1/sqrt(α_t) * (x_t - β_t/sqrt(1-α̅_t) * ε) + σ_t * z

        The correct posterior variance is:
            β̃_t = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
            σ_t = sqrt(β̃_t)

        Using sqrt(β_t) directly injects too much noise and hurts sample quality.

        Args:
            x_t: Current noisy sample
            pred_eps: Model's prediction of the noise ε
            t: Current timestep indices
            step_index: Integer step (for checking if t=0)

        Returns:
            x_{t-1}: Sample at previous timestep
        """
        # Get coefficients for this timestep, reshape for broadcasting
        beta_t = self.schedule.betas[t][:, None, None]
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bars[t][:, None, None]
        sqrt_recip_alpha_t = self.schedule.sqrt_recip_alphas[t][:, None, None]
        alpha_bar_t = self.schedule.alpha_bars[t][:, None, None]

        # α̅_{t-1}, with convention α̅_{-1} = 1 for t=0
        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.schedule.alpha_bars[t_prev][:, None, None]
        alpha_bar_prev = torch.where(
            (t == 0)[:, None, None],
            torch.ones_like(alpha_bar_prev),
            alpha_bar_prev
        )

        # DDPM posterior mean (using predicted ε directly)
        # x_{t-1} = 1/sqrt(α_t) * (x_t - β_t/sqrt(1-α̅_t) * ε)
        mean = sqrt_recip_alpha_t * (
            x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_eps
        )

        # Add noise (except at final step t=0)
        if step_index > 0:
            # Correct posterior variance: β̃_t = β_t * (1 - α̅_{t-1}) / (1 - α̅_t)
            beta_tilde = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            sigma_t = torch.sqrt(beta_tilde)
            return mean + sigma_t * torch.randn_like(x_t)
        else:
            return mean


def main():
    """Test the sampler with reconstruction-from-noise.

    IMPORTANT CONCEPTUAL NOTE:
    --------------------------
    Unconditional sampling (from pure noise) generates *a* random protein,
    NOT a specific target protein. Comparing it to a validation chain gives
    high RMSD (~20-30Å) - this is expected and correct!

    To test if the diffusion model is learning correctly, use reconstruction:
    1. Take a real protein x0
    2. Noise it to timestep t: x_t = q_sample(x0, t)
    3. Denoise it back: x_hat = sample_from(x_t, start_t=t)
    4. Compare x_hat to x0 - this RMSD should improve with training
    """
    from pathlib import Path

    from .geom import rmsd, ca_bond_lengths, radius_of_gyration
    from .pdb_io import ca_to_pdb_str
    from .train import load_single_chain

    scale_factor = 10.0

    # Load ground truth
    x0_true, name = load_single_chain(Path("data/chain_set.jsonl"), scale_factor)
    L = x0_true.shape[1]
    print(f"Target: {name}, Length: {L}")

    # Try to load a trained model, fall back to untrained
    model_path = Path("runs/model.pt")
    schedule = DiffusionSchedule(T=1000)

    # Find most recent model checkpoint
    runs_dir = Path("runs")
    checkpoints = list(runs_dir.glob("*/model.pt")) if runs_dir.exists() else []
    if checkpoints:
        # Sort by modification time, get most recent
        model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"\nLoading trained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        args = checkpoint["args"]
        model = DiffusionTransformer(
            d_model=args.get("d_model", 128),
            num_layers=args.get("num_layers", 4),
            num_heads=args.get("num_heads", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        trained = True
    else:
        print("\nNo trained model found. Using untrained model.")
        print("Train a model first: uv run python -m src.train --steps 5000")
        model = DiffusionTransformer(d_model=128, num_layers=4, num_heads=4)
        trained = False

    sampler = DiffusionSampler(model, schedule)

    # =========================================================================
    # Test 1: Reconstruction from noise (the correct test!)
    # =========================================================================
    print("\n" + "=" * 60)
    print("RECONSTRUCTION TEST (correct way to evaluate)")
    print("=" * 60)
    print("This tests if the model can denoise a noised version of a real chain.")
    print("RMSD should improve with training.\n")

    # Test reconstruction at different noise levels
    test_timesteps = [100, 300, 500, 700, 900]
    print(f"{'t_start':>8} {'alpha_bar':>10} {'RMSD (Å)':>12}")
    print("-" * 35)

    torch.manual_seed(42)
    for t_start in test_timesteps:
        # 1. Noise the real chain to timestep t
        x_t, _ = schedule.q_sample(x0_true, t_start)

        # 2. Denoise it back
        x_hat = sampler.sample_from(x_t, start_t=t_start, verbose=False)

        # 3. Compare to original
        x_hat_np = x_hat.squeeze().numpy() * scale_factor
        x0_np = x0_true.squeeze().numpy() * scale_factor
        recon_rmsd = rmsd(x_hat_np, x0_np, align=True)

        alpha_bar = schedule.alpha_bars[t_start].item()
        print(f"{t_start:>8} {alpha_bar:>10.4f} {recon_rmsd:>12.2f}")

    print("\nExpected behavior:")
    print("  - Untrained: High RMSD at all timesteps (~20-30Å)")
    print("  - Trained: Lower RMSD, especially at lower t (more signal)")
    print("  - t=100 should have lowest RMSD (most signal preserved)")

    # =========================================================================
    # Test 2: Unconditional sampling (for reference)
    # =========================================================================
    print("\n" + "=" * 60)
    print("UNCONDITIONAL SAMPLING (for comparison)")
    print("=" * 60)
    print("This generates a random protein from pure noise.")
    print("High RMSD to target is EXPECTED - it's a different protein!\n")

    generated = sampler.sample(shape=(1, L, 3), verbose=True)

    gen_np = generated.squeeze().numpy() * scale_factor
    true_np = x0_true.squeeze().numpy() * scale_factor

    gen_rmsd = rmsd(gen_np, true_np, align=True)
    gen_rg = radius_of_gyration(gen_np)
    gen_bonds = ca_bond_lengths(gen_np)

    print(f"\nGenerated structure:")
    print(f"  RMSD to target: {gen_rmsd:.2f} Å (high is OK - different protein)")
    print(f"  Radius of gyration: {gen_rg:.2f} Å")
    print(f"  CA-CA bonds: mean={gen_bonds.mean():.2f} Å, std={gen_bonds.std():.2f} Å")

    # Save PDB
    pdb_str = ca_to_pdb_str(gen_np)
    pdb_path = Path("generated_test.pdb")
    pdb_path.write_text(pdb_str)
    print(f"\nSaved generated structure to: {pdb_path}")


if __name__ == "__main__":
    main()
