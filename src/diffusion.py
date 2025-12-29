"""Diffusion utilities for the forward and reverse process.

Usage:
    uv run python -m src.diffusion

Forward Process (Noising)
-------------------------
The forward process gradually adds noise to data over T timesteps.

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

where:
    - x0: original clean data
    - epsilon: random noise ~ N(0, I)
    - alpha_bar_t: cumulative product of (1 - beta_t), the "signal remaining"
    - x_t: noisy data at timestep t

Beta Schedule
-------------
    beta_t: amount of noise added at step t (small, e.g. 0.0001 to 0.02)
    alpha_t = 1 - beta_t: signal kept at step t
    alpha_bar_t = prod(alpha_1, ..., alpha_t): total signal remaining

At t=0: alpha_bar ≈ 1 (all signal, no noise)
At t=T: alpha_bar ≈ 0 (no signal, all noise)
"""

import numpy as np
import torch


def make_beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    kind: str = "linear",
) -> torch.Tensor:
    """Create a noise schedule.

    Args:
        T: Number of diffusion timesteps
        beta_start: Starting noise level
        beta_end: Ending noise level
        kind: Schedule type ('linear' or 'cosine')

    Returns:
        (T,) tensor of beta values
    """
    if kind == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    elif kind == "cosine":
        # Cosine schedule from "Improved DDPM" paper
        steps = torch.linspace(0, T, T + 1)
        alpha_bar = torch.cos((steps / T + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clamp(betas, min=1e-4, max=0.999)
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")

    return betas


class DiffusionSchedule:
    """Precomputed diffusion schedule for efficient sampling."""

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        kind: str = "linear",
    ):
        self.T = T

        # Beta schedule
        self.betas = make_beta_schedule(T, beta_start, beta_end, kind)

        # Alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute useful quantities
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        # For reverse process (will use later)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor | int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to x0 at timestep t.

        Args:
            x0: (B, L, 3) or (L, 3) clean coordinates
            t: (B,) tensor, scalar tensor, or int timestep indices
            noise: Optional pre-generated noise, same shape as x0

        Returns:
            x_t: Noisy coordinates at timestep t
            noise: The noise that was added (useful for training)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # Handle int or scalar tensor t
        if isinstance(t, int):
            t = torch.tensor([t], device=x0.device)
        elif t.dim() == 0:
            t = t.unsqueeze(0)

        # Get coefficients for this timestep
        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]

        # Reshape for broadcasting: (B,) -> (B, 1, 1) for (B, L, 3)
        while sqrt_alpha_bar.dim() < x0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)

        # Forward diffusion equation
        x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        return x_t, noise

    def to(self, device: torch.device) -> "DiffusionSchedule":
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_alphas = self.sqrt_alphas.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        return self


def main():
    """Visualize the diffusion schedule with proper coordinate scaling."""
    from pathlib import Path

    from .data_cath import get_one_chain
    from .geom import ca_bond_lengths, center, radius_of_gyration

    # Load a protein
    result = get_one_chain(Path("data/chain_set.jsonl"))
    if result is None:
        print("No chain found")
        return

    name, seq, ca_coords = result
    print(f"Loaded {name}, {len(seq)} residues")

    # Center coordinates
    ca_centered, _ = center(ca_coords)

    # IMPORTANT: Normalize coordinates for diffusion
    # Protein coordinates are in Angstroms (~15Å range), but diffusion assumes N(0,1)
    # Without scaling, the noise term dominates and the structure "shrinks" to origin
    scale_factor = 10.0
    x0 = torch.from_numpy(ca_centered).float() / scale_factor

    # Create schedule
    schedule = DiffusionSchedule(T=1000, kind="linear")

    print(f"\nDiffusion Schedule (T={schedule.T}):")
    print(f"  beta range: [{schedule.betas[0]:.6f}, {schedule.betas[-1]:.6f}]")
    print(f"  alpha_bar at t=0:   {schedule.alpha_bars[0]:.4f}")
    print(f"  alpha_bar at t=500: {schedule.alpha_bars[500]:.4f}")
    print(f"  alpha_bar at t=999: {schedule.alpha_bars[999]:.6f}")
    print(f"  scale_factor: {scale_factor} (coords normalized to ~N(0,1))")

    # Show noising at different timesteps
    print(f"\nForward diffusion on {name} (scaled by 1/{scale_factor}):")
    print(f"{'t':>6} {'alpha_bar':>10} {'Rg (scaled)':>12} {'Rg (Å)':>10} {'CA-CA (Å)':>12}")
    print("-" * 55)

    timesteps = [0, 100, 250, 500, 750, 900, 999]
    torch.manual_seed(42)

    for t in timesteps:
        t_tensor = torch.tensor([t])
        x_t, _ = schedule.q_sample(x0.unsqueeze(0), t_tensor)
        x_t_scaled = x_t.squeeze(0).numpy()

        # Metrics in scaled space
        rg_scaled = radius_of_gyration(x_t_scaled)

        # Metrics in physical space (Angstroms)
        x_t_physical = x_t_scaled * scale_factor
        rg_physical = radius_of_gyration(x_t_physical)
        bonds_physical = ca_bond_lengths(x_t_physical)
        alpha_bar = schedule.alpha_bars[t].item()

        print(f"{t:>6} {alpha_bar:>10.4f} {rg_scaled:>12.2f} {rg_physical:>10.2f} {bonds_physical.mean():>12.2f}")

    print("\nWith proper scaling:")
    print("  - Rg (scaled) stays ~1.0-1.4 throughout (proper variance)")
    print("  - Rg (Å) stays ~10-14Å (not shrinking to 1.7Å)")
    print("  - Bond lengths diverge from 3.8Å as expected (noise destroys local structure)")


if __name__ == "__main__":
    main()
