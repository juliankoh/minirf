"""Diffusion sampler for generating protein structures from noise.

Usage:
    uv run python -m src.sampler

The sampler reverses the diffusion process:
    1. Start with pure Gaussian noise x_T
    2. Iteratively denoise: x_T -> x_{T-1} -> ... -> x_0
    3. At each step, model predicts x_0, we compute implied noise, take a step

Why sampling works when single-step eval fails:
    - At t=500, the model makes a noisy guess (high RMSD)
    - But sampling takes 1000 small steps, allowing course correction
    - Early steps point "roughly right", later steps refine precisely
"""

import torch
from tqdm import tqdm

from .diffusion import DiffusionSchedule
from .model import DiffusionTransformer


class DiffusionSampler:
    """Generates proteins by reversing the diffusion process.

    Implements standard DDPM posterior sampling for models that predict x0.
    """

    def __init__(self, model: DiffusionTransformer, schedule: DiffusionSchedule):
        self.model = model
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        verbose: bool = True,
    ) -> torch.Tensor:
        """Generate samples from pure noise.

        Args:
            shape: (B, L, 3) desired output shape
            verbose: Show progress bar

        Returns:
            (B, L, 3) generated coordinates (scaled)
        """
        self.model.eval()
        B, L, _ = shape

        # 1. Start with pure Gaussian noise
        x_t = torch.randn(shape)

        # Reverse timesteps: T-1, T-2, ..., 0
        T = self.schedule.T
        iterator = range(T - 1, -1, -1)
        if verbose:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = torch.full((B,), i, dtype=torch.long)

            # 2. Predict clean structure x0 from current noisy state x_t
            mask = torch.ones((B, L), dtype=torch.bool)
            pred_x0 = self.model(x_t, t, mask=mask)

            # 3. Take one step: x_t -> x_{t-1}
            x_t = self.p_sample(x_t, pred_x0, t, step_index=i)

        return x_t

    def p_sample(
        self,
        x_t: torch.Tensor,
        pred_x0: torch.Tensor,
        t: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Calculate x_{t-1} given x_t and predicted x_0.

        Uses the DDPM posterior formula:
            x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps) + sigma_t * z

        Since we predict x0 (not eps), we first compute implied eps:
            eps = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1 - alpha_bar_t)

        Args:
            x_t: Current noisy sample
            pred_x0: Model's prediction of clean x0
            t: Current timestep indices
            step_index: Integer step (for checking if t=0)

        Returns:
            x_{t-1}: Sample at previous timestep
        """
        # Get coefficients for this timestep, reshape for broadcasting
        beta_t = self.schedule.betas[t][:, None, None]
        sqrt_alpha_bar_t = self.schedule.sqrt_alpha_bars[t][:, None, None]
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bars[t][:, None, None]
        sqrt_recip_alpha_t = self.schedule.sqrt_recip_alphas[t][:, None, None]

        # 1. Compute implied noise from predicted x0
        # x_t = sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps
        # => eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
        pred_eps = (x_t - sqrt_alpha_bar_t * pred_x0) / sqrt_one_minus_alpha_bar_t

        # 2. DDPM posterior mean
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps)
        mean = sqrt_recip_alpha_t * (
            x_t - beta_t / sqrt_one_minus_alpha_bar_t * pred_eps
        )

        # 3. Add noise (except at final step t=0)
        if step_index > 0:
            noise = torch.randn_like(x_t)
            # Posterior variance = beta_t (simplified)
            return mean + torch.sqrt(beta_t) * noise
        else:
            return mean


def main():
    """Test the sampler with a trained model."""
    from pathlib import Path

    import numpy as np

    from .geom import rmsd, ca_bond_lengths, radius_of_gyration
    from .pdb_io import ca_to_pdb_str
    from .train import load_single_chain

    scale_factor = 10.0

    # Load ground truth for comparison
    x0_true, name = load_single_chain(Path("data/chain_set.jsonl"), scale_factor)
    L = x0_true.shape[1]
    print(f"Target: {name}, Length: {L}")

    # Create model (untrained - just for testing the sampler mechanics)
    model = DiffusionTransformer(d_model=128, num_layers=4, num_heads=4)
    schedule = DiffusionSchedule(T=1000)
    sampler = DiffusionSampler(model, schedule)

    print(f"\nGenerating structure of length {L} from noise...")
    print("(Note: Using untrained model, expect random output)")

    # Generate
    generated = sampler.sample(shape=(1, L, 3), verbose=True)

    # Analyze
    gen_np = generated.squeeze().numpy() * scale_factor
    true_np = x0_true.squeeze().numpy() * scale_factor

    gen_rmsd = rmsd(gen_np, true_np, align=True)
    gen_rg = radius_of_gyration(gen_np)
    gen_bonds = ca_bond_lengths(gen_np)

    print(f"\nGenerated structure analysis:")
    print(f"  RMSD to target: {gen_rmsd:.2f} Å (expected high for untrained)")
    print(f"  Radius of gyration: {gen_rg:.2f} Å")
    print(f"  CA-CA bonds: mean={gen_bonds.mean():.2f} Å, std={gen_bonds.std():.2f} Å")

    # Save PDB
    pdb_str = ca_to_pdb_str(gen_np)
    pdb_path = Path("generated_test.pdb")
    pdb_path.write_text(pdb_str)
    print(f"\nSaved to: {pdb_path}")


if __name__ == "__main__":
    main()
