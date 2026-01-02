"""Inference script for protein structure diffusion model.

Usage:
    # Generate samples
    python -m src.inference sample --model runs/exp1/model.pt --length 100 --batch_size 8

    # Reconstruct from clean structure
    python -m src.inference reconstruct --model runs/exp1/model.pt --input_x0 ref.pdb --start_t 500

    # Single-step denoising (debug)
    python -m src.inference denoise_step --model runs/exp1/model.pt --input_xt noisy.npy --t 500

See INFERENCE_SPEC.md for full documentation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .diffusion import DiffusionSchedule
from .geom import (
    align_to_principal_axes,
    apply_random_rotation,
    ca_bond_lengths,
    center,
    compute_clashes,
    kabsch_align,
    radius_of_gyration,
    rmsd,
)
from .model import DiffusionTransformer
from .pdb_io import ca_to_pdb_str

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Core Functions
# =============================================================================


def load_checkpoint(path: str | Path) -> dict:
    """Load and validate a checkpoint file.

    Args:
        path: Path to .pt checkpoint file

    Returns:
        Checkpoint dictionary with model_state_dict, args, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Validate required keys
    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing 'model_state_dict' key")

    if "args" not in checkpoint:
        logger.warning("Checkpoint missing 'args' key, using defaults")
        checkpoint["args"] = {}

    return checkpoint


def build_model(
    checkpoint: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> DiffusionTransformer:
    """Instantiate model and load weights from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dict
        device: Device to load model onto
        dtype: Data type for model parameters

    Returns:
        Loaded DiffusionTransformer model
    """
    args = checkpoint.get("args", {})

    # Extract model config with defaults
    d_model = args.get("d_model", 128)
    num_layers = args.get("num_layers", 4)
    num_heads = args.get("num_heads", 4)
    max_len = args.get("max_len", 512)

    # Log model config
    logger.info(
        f"Loading DiffusionTransformer: d_model={d_model}, "
        f"layers={num_layers}, heads={num_heads}, max_len={max_len}"
    )

    # Instantiate model
    model = DiffusionTransformer(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        max_len=max_len,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


def build_schedule(checkpoint: dict, device: torch.device) -> DiffusionSchedule:
    """Create diffusion schedule from checkpoint args.

    Args:
        checkpoint: Loaded checkpoint dict
        device: Device to move schedule tensors to

    Returns:
        DiffusionSchedule instance
    """
    args = checkpoint.get("args", {})

    # Extract schedule config with defaults
    T = args.get("T", 1000)
    # Schedule params not typically stored, use defaults
    beta_start = 1e-4
    beta_end = 0.02
    kind = "cosine"

    schedule = DiffusionSchedule(
        T=T,
        beta_start=beta_start,
        beta_end=beta_end,
        kind=kind,
    )

    return schedule.to(device)


def get_max_training_length(checkpoint: dict) -> int:
    """Extract maximum training length from checkpoint."""
    args = checkpoint.get("args", {})
    return args.get("max_len", 512)


# =============================================================================
# Preprocessing / Postprocessing
# =============================================================================


def preprocess_coords(
    coords: np.ndarray,
    scale_factor: float = 10.0,
    do_center: bool = True,
    do_align: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Preprocess raw coordinates for model input.

    Applies centering, principal axis alignment, and scaling.

    Args:
        coords: (L, 3) or (B, L, 3) coordinates in Angstroms
        scale_factor: Divide coordinates by this value
        do_center: Whether to center coordinates at origin
        do_align: Whether to align to principal axes (first PC along Z)

    Returns:
        Tuple of (preprocessed tensor, metadata dict for inverse transform)
    """
    # Handle batched input
    if coords.ndim == 2:
        coords = coords[np.newaxis, ...]  # (1, L, 3)
        was_unbatched = True
    else:
        was_unbatched = False

    B, L, _ = coords.shape
    processed = []
    metadata = {"was_unbatched": was_unbatched, "scale_factor": scale_factor}

    for i in range(B):
        c = coords[i].copy()

        if do_center:
            c, centroid = center(c)

        if do_align:
            c = align_to_principal_axes(c)

        c = c / scale_factor
        processed.append(c)

    result = np.stack(processed, axis=0)
    return torch.from_numpy(result).float(), metadata


def postprocess_coords(
    coords: torch.Tensor,
    scale_factor: float = 10.0,
    random_rotate: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert model output back to Angstroms.

    Args:
        coords: (B, L, 3) tensor in scaled units
        scale_factor: Multiply coordinates by this value
        random_rotate: Apply random rotation per sample
        rng: Random generator for rotations

    Returns:
        (B, L, 3) numpy array in Angstroms
    """
    coords_np = coords.cpu().numpy() * scale_factor

    if random_rotate:
        if rng is None:
            rng = np.random.default_rng()
        rotated = []
        for i in range(len(coords_np)):
            r, _ = apply_random_rotation(coords_np[i], rng)
            rotated.append(r)
        coords_np = np.stack(rotated, axis=0)

    return coords_np


# =============================================================================
# Input File Handling
# =============================================================================


def load_coords_from_file(path: str | Path) -> np.ndarray:
    """Load coordinates from .npy or .pdb file.

    Args:
        path: Path to coordinate file

    Returns:
        (L, 3) or (B, L, 3) numpy array of coordinates

    Raises:
        ValueError: If file format is not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        coords = np.load(path)
        # Auto-reshape common formats
        coords = _normalize_coord_shape(coords)
        return coords

    elif suffix == ".pdb":
        coords = _extract_ca_from_pdb(path)
        return coords

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .npy or .pdb")


def _normalize_coord_shape(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinate array to (L, 3) or (B, L, 3) shape."""
    # Squeeze extra dimensions
    coords = np.squeeze(coords)

    if coords.ndim == 2:
        # Should be (L, 3)
        if coords.shape[-1] != 3:
            raise ValueError(f"Expected last dim to be 3, got shape {coords.shape}")
        return coords

    elif coords.ndim == 3:
        # Should be (B, L, 3)
        if coords.shape[-1] != 3:
            raise ValueError(f"Expected last dim to be 3, got shape {coords.shape}")
        return coords

    else:
        raise ValueError(f"Cannot interpret shape {coords.shape} as coordinates")


def _extract_ca_from_pdb(path: Path) -> np.ndarray:
    """Extract CA atom coordinates from a PDB file.

    Args:
        path: Path to PDB file

    Returns:
        (L, 3) numpy array of CA coordinates
    """
    coords = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])

    if len(coords) == 0:
        raise ValueError(f"No CA atoms found in {path}")

    return np.array(coords, dtype=np.float32)


# =============================================================================
# Output Functions
# =============================================================================


def save_pdb(
    coords: np.ndarray,
    path: str | Path,
    chain_id: str = "A",
) -> None:
    """Save coordinates as PDB file with ALA residues.

    Args:
        coords: (L, 3) coordinates in Angstroms
        path: Output path
        chain_id: Chain identifier
    """
    # Use ALA for all residues as per spec
    seq = "A" * len(coords)
    pdb_str = ca_to_pdb_str(coords, seq=seq, chain_id=chain_id)
    Path(path).write_text(pdb_str)


def save_trajectory_pdb(
    coords_list: list[np.ndarray],
    path: str | Path,
    timesteps: list[int] | None = None,
) -> None:
    """Save multi-model PDB for trajectory visualization.

    Args:
        coords_list: List of (L, 3) coordinate arrays
        path: Output path
        timesteps: Optional timestep labels for REMARK
    """
    lines = ["HEADER    TRAJECTORY"]

    for i, coords in enumerate(coords_list):
        t_label = timesteps[i] if timesteps else i
        lines.append(f"MODEL     {i + 1}")
        lines.append(f"REMARK    TIMESTEP {t_label}")

        seq = "A" * len(coords)
        for j, (xyz, aa) in enumerate(zip(coords, seq)):
            atom_num = j + 1
            res_num = j + 1
            x, y, z = xyz
            line = (
                f"ATOM  {atom_num:5d}  CA  ALA A{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            lines.append(line)

        lines.append("ENDMDL")

    lines.append("END")
    Path(path).write_text("\n".join(lines))


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        path: Output path
    """
    # Convert numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    metrics = convert(metrics)
    Path(path).write_text(json.dumps(metrics, indent=2))


# =============================================================================
# Quality Metrics
# =============================================================================


def compute_metrics(
    coords: np.ndarray,
    reference: np.ndarray | None = None,
    align_for_rmsd: bool = True,
) -> dict:
    """Compute quality metrics for a single structure.

    Args:
        coords: (L, 3) coordinates in Angstroms
        reference: Optional (L, 3) reference for RMSD
        align_for_rmsd: Use Kabsch alignment before RMSD

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Bond length statistics
    bonds = ca_bond_lengths(coords)
    metrics["bond_length_mean"] = float(bonds.mean())
    metrics["bond_length_std"] = float(bonds.std())
    metrics["bond_valid_pct"] = float(((bonds > 3.6) & (bonds < 4.0)).mean() * 100)

    # Radius of gyration
    metrics["radius_of_gyration"] = float(radius_of_gyration(coords))

    # Clashes
    metrics["clashes"] = int(compute_clashes(coords))

    # RMSD to reference if provided
    if reference is not None:
        if len(coords) != len(reference):
            # Truncate to common length
            min_len = min(len(coords), len(reference))
            coords = coords[:min_len]
            reference = reference[:min_len]
        metrics["rmsd"] = float(rmsd(coords, reference, align=align_for_rmsd))

    return metrics


def compute_batch_metrics(
    coords_batch: np.ndarray,
    reference: np.ndarray | None = None,
    align_for_rmsd: bool = True,
) -> dict:
    """Compute metrics for a batch of structures.

    Args:
        coords_batch: (B, L, 3) coordinates in Angstroms
        reference: Optional (L, 3) reference for RMSD
        align_for_rmsd: Use Kabsch alignment before RMSD

    Returns:
        Dictionary with per-sample metrics and aggregates
    """
    B = len(coords_batch)
    all_metrics = []

    for i in range(B):
        m = compute_metrics(coords_batch[i], reference, align_for_rmsd)
        m["sample_idx"] = i
        all_metrics.append(m)

    # Aggregate statistics
    result = {
        "samples": all_metrics,
        "num_samples": B,
        "bond_length_mean": float(np.mean([m["bond_length_mean"] for m in all_metrics])),
        "bond_valid_pct_mean": float(np.mean([m["bond_valid_pct"] for m in all_metrics])),
        "radius_of_gyration_mean": float(
            np.mean([m["radius_of_gyration"] for m in all_metrics])
        ),
        "clashes_mean": float(np.mean([m["clashes"] for m in all_metrics])),
    }

    if reference is not None:
        result["rmsd_mean"] = float(np.mean([m["rmsd"] for m in all_metrics]))

    return result


# =============================================================================
# Quality Filtering
# =============================================================================


def filter_samples(
    coords: np.ndarray,
    min_bond_pct: float = 70.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter samples by quality threshold.

    Args:
        coords: (B, L, 3) coordinates in Angstroms
        min_bond_pct: Minimum percentage of bonds in [3.6, 4.0] range

    Returns:
        Tuple of (passed_coords, rejected_coords, passed_mask)
    """
    B = len(coords)
    passed_mask = np.zeros(B, dtype=bool)

    for i in range(B):
        bonds = ca_bond_lengths(coords[i])
        valid_pct = ((bonds > 3.6) & (bonds < 4.0)).mean() * 100
        passed_mask[i] = valid_pct >= min_bond_pct

    passed = coords[passed_mask]
    rejected = coords[~passed_mask]

    return passed, rejected, passed_mask


# =============================================================================
# Kabsch Alignment
# =============================================================================


def kabsch_align_coords(
    mobile: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Optimal superposition of mobile onto target.

    Args:
        mobile: (L, 3) coordinates to align
        target: (L, 3) reference coordinates

    Returns:
        (L, 3) aligned coordinates
    """
    aligned, _, _ = kabsch_align(mobile, target)
    return aligned


# =============================================================================
# Task Functions
# =============================================================================


class InferenceSampler:
    """Wrapper around DiffusionSampler with trajectory saving support."""

    def __init__(
        self,
        model: DiffusionTransformer,
        schedule: DiffusionSchedule,
        device: torch.device,
    ):
        self.model = model
        self.schedule = schedule
        self.device = device

    @torch.no_grad()
    def sample(
        self,
        length: int,
        batch_size: int,
        add_noise: bool = True,
        use_self_cond: bool = True,
        save_at: list[int] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]] | None]:
        """Generate samples from pure noise.

        Args:
            length: Number of residues per sample
            batch_size: Number of samples to generate
            add_noise: Use stochastic sampling
            use_self_cond: Use self-conditioning
            save_at: Optional list of timesteps to save trajectory
            seed: Random seed for reproducibility

        Returns:
            Tuple of (final coords, optional list of (timestep, coords) for trajectory)
        """
        self.model.eval()
        T = self.schedule.T

        # Set seed for full trajectory control
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)

        # Start from pure noise
        shape = (batch_size, length, 3)
        x_t = torch.randn(shape, device=self.device)

        # Default mask: all valid
        mask = torch.ones((batch_size, length), dtype=torch.bool, device=self.device)
        mask_3d = mask.unsqueeze(-1)

        # Self-conditioning state
        x0_self_cond = torch.zeros_like(x_t) if use_self_cond else None

        # Trajectory storage
        trajectory = [] if save_at else None
        save_at_set = set(save_at) if save_at else set()

        # Reverse timesteps
        iterator = range(T - 1, -1, -1)
        iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = torch.full((batch_size,), i, dtype=torch.long, device=self.device)

            # Save trajectory if requested
            if i in save_at_set:
                trajectory.append((i, x_t.clone()))

            # Model predicts noise
            eps_pred = self.model(
                x_t, t, mask=mask,
                x0_self_cond=x0_self_cond if use_self_cond else None
            )

            # Update self-conditioning
            if use_self_cond:
                sqrt_alpha_bar = self.schedule.sqrt_alpha_bars[i]
                sqrt_one_minus = self.schedule.sqrt_one_minus_alpha_bars[i]
                x0_self_cond = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar
                x0_self_cond = x0_self_cond * mask_3d

            # Take one step
            x_t = self._p_sample(x_t, eps_pred, t, i, add_noise)
            x_t = x_t * mask_3d

        # Save final state if 0 was requested
        if 0 in save_at_set:
            trajectory.append((0, x_t.clone()))

        return x_t, trajectory

    @torch.no_grad()
    def sample_from(
        self,
        x_t: torch.Tensor,
        start_t: int,
        add_noise: bool = True,
        use_self_cond: bool = True,
        mask: torch.Tensor | None = None,
        save_at: list[int] | None = None,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]] | None]:
        """Generate samples starting from a noisy state.

        Args:
            x_t: (B, L, 3) noisy coordinates at timestep start_t
            start_t: Starting timestep
            add_noise: Use stochastic sampling
            use_self_cond: Use self-conditioning
            mask: Optional (B, L) mask for valid positions
            save_at: Optional list of timesteps to save trajectory
            seed: Random seed

        Returns:
            Tuple of (final coords, optional trajectory list)
        """
        self.model.eval()
        B, L, _ = x_t.shape

        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)

        if mask is None:
            mask = torch.ones((B, L), dtype=torch.bool, device=self.device)
        mask_3d = mask.unsqueeze(-1)

        x0_self_cond = torch.zeros_like(x_t) if use_self_cond else None

        trajectory = [] if save_at else None
        save_at_set = set(save_at) if save_at else set()

        iterator = range(start_t, -1, -1)
        iterator = tqdm(iterator, desc=f"Sampling from t={start_t}")

        for i in iterator:
            t = torch.full((B,), i, dtype=torch.long, device=self.device)

            if i in save_at_set:
                trajectory.append((i, x_t.clone()))

            eps_pred = self.model(
                x_t, t, mask=mask,
                x0_self_cond=x0_self_cond if use_self_cond else None
            )

            if use_self_cond:
                sqrt_alpha_bar = self.schedule.sqrt_alpha_bars[i]
                sqrt_one_minus = self.schedule.sqrt_one_minus_alpha_bars[i]
                x0_self_cond = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_bar
                x0_self_cond = x0_self_cond * mask_3d

            x_t = self._p_sample(x_t, eps_pred, t, i, add_noise)
            x_t = x_t * mask_3d

        if 0 in save_at_set:
            trajectory.append((0, x_t.clone()))

        return x_t, trajectory

    def _p_sample(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: torch.Tensor,
        step_index: int,
        add_noise: bool,
    ) -> torch.Tensor:
        """Take one reverse diffusion step."""
        beta_t = self.schedule.betas[t][:, None, None]
        sqrt_one_minus_alpha_bar_t = self.schedule.sqrt_one_minus_alpha_bars[t][:, None, None]
        sqrt_recip_alpha_t = self.schedule.sqrt_recip_alphas[t][:, None, None]
        alpha_bar_t = self.schedule.alpha_bars[t][:, None, None]

        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.schedule.alpha_bars[t_prev][:, None, None]
        alpha_bar_prev = torch.where(
            (t == 0)[:, None, None],
            torch.ones_like(alpha_bar_prev),
            alpha_bar_prev
        )

        mean = sqrt_recip_alpha_t * (
            x_t - beta_t / sqrt_one_minus_alpha_bar_t * eps_pred
        )

        if step_index > 0 and add_noise:
            beta_tilde = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            sigma_t = torch.sqrt(beta_tilde)
            return mean + sigma_t * torch.randn_like(x_t)
        else:
            return mean


def run_sample(
    model: DiffusionTransformer,
    schedule: DiffusionSchedule,
    device: torch.device,
    length: int,
    batch_size: int,
    scale_factor: float = 10.0,
    add_noise: bool = True,
    use_self_cond: bool = True,
    save_at: list[int] | None = None,
    seed: int | None = None,
    random_rotate: bool = False,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]] | None, dict]:
    """Generate samples unconditionally.

    Returns:
        Tuple of (coords in Angstroms, optional trajectory, metrics)
    """
    sampler = InferenceSampler(model, schedule, device)

    coords, trajectory = sampler.sample(
        length=length,
        batch_size=batch_size,
        add_noise=add_noise,
        use_self_cond=use_self_cond,
        save_at=save_at,
        seed=seed,
    )

    # Compute metrics before rotation
    coords_np = coords.cpu().numpy() * scale_factor
    metrics = compute_batch_metrics(coords_np)

    # Apply random rotation if requested
    if random_rotate:
        rng = np.random.default_rng(seed)
        coords_np = postprocess_coords(coords, scale_factor, random_rotate=True, rng=rng)
    else:
        coords_np = postprocess_coords(coords, scale_factor, random_rotate=False)

    # Convert trajectory
    if trajectory:
        trajectory = [(t, c.cpu().numpy() * scale_factor) for t, c in trajectory]

    return coords_np, trajectory, metrics


def run_reconstruct(
    model: DiffusionTransformer,
    schedule: DiffusionSchedule,
    device: torch.device,
    x0: np.ndarray,
    start_t: int,
    scale_factor: float = 10.0,
    deterministic: bool = False,
    use_self_cond: bool = True,
    shared_noise: bool = False,
    save_at: list[int] | None = None,
    seed: int | None = None,
    align_for_rmsd: bool = True,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]] | None, dict]:
    """Reconstruct from clean coordinates.

    Args:
        x0: (L, 3) or (B, L, 3) clean coordinates (raw, in Angstroms)
        start_t: Timestep to noise to and start denoising from
        deterministic: Use deterministic sampling
        shared_noise: Use same noise for all samples in batch
        align_for_rmsd: Use Kabsch alignment for RMSD

    Returns:
        Tuple of (reconstructed coords, optional trajectory, metrics)
    """
    # Preprocess
    x0_tensor, _ = preprocess_coords(x0, scale_factor=scale_factor)
    x0_tensor = x0_tensor.to(device)
    B, L, _ = x0_tensor.shape

    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

    # Generate noise
    if shared_noise:
        noise = torch.randn(1, L, 3, device=device).expand(B, -1, -1)
    else:
        noise = torch.randn(B, L, 3, device=device)

    # Forward diffusion to start_t
    t_tensor = torch.full((B,), start_t, dtype=torch.long, device=device)
    x_t, _ = schedule.q_sample(x0_tensor, t_tensor, noise=noise)

    # Denoise
    sampler = InferenceSampler(model, schedule, device)
    recon, trajectory = sampler.sample_from(
        x_t=x_t,
        start_t=start_t,
        add_noise=not deterministic,
        use_self_cond=use_self_cond,
        save_at=save_at,
        seed=seed,
    )

    # Convert to Angstroms
    recon_np = recon.cpu().numpy() * scale_factor

    # Compute metrics with reference
    if x0.ndim == 2:
        reference = x0
    else:
        reference = x0[0]  # Use first sample as reference
    metrics = compute_batch_metrics(recon_np, reference, align_for_rmsd)

    # Convert trajectory
    if trajectory:
        trajectory = [(t, c.cpu().numpy() * scale_factor) for t, c in trajectory]

    return recon_np, trajectory, metrics


def run_denoise_step(
    model: DiffusionTransformer,
    schedule: DiffusionSchedule,
    device: torch.device,
    x_t: np.ndarray,
    t: int,
    scale_factor: float = 10.0,
    full_debug: bool = False,
) -> dict:
    """Single-step denoising for debugging.

    Args:
        x_t: (L, 3) or (B, L, 3) noisy coordinates (preprocessed, scaled)
        t: Current timestep
        full_debug: Include additional debug outputs

    Returns:
        Dictionary with eps_pred, x0_pred, and optionally x_prev, snr, etc.
    """
    model.eval()

    # Handle input
    if x_t.ndim == 2:
        x_t = x_t[np.newaxis, ...]

    # Assume input is already preprocessed (scaled)
    x_t_tensor = torch.from_numpy(x_t).float().to(device)
    B, L, _ = x_t_tensor.shape

    t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

    with torch.no_grad():
        eps_pred = model(x_t_tensor, t_tensor)

        # Compute x0_pred
        sqrt_alpha_bar = schedule.sqrt_alpha_bars[t]
        sqrt_one_minus = schedule.sqrt_one_minus_alpha_bars[t]
        x0_pred = (x_t_tensor - sqrt_one_minus * eps_pred) / sqrt_alpha_bar

    result = {
        "eps_pred": eps_pred.cpu().numpy(),
        "x0_pred": x0_pred.cpu().numpy() * scale_factor,
        "timestep": t,
    }

    if full_debug:
        # Compute x_{t-1}
        sampler = InferenceSampler(model, schedule, device)
        x_prev = sampler._p_sample(x_t_tensor, eps_pred, t_tensor, t, add_noise=False)
        result["x_prev"] = x_prev.cpu().numpy() * scale_factor

        # SNR
        alpha_bar = schedule.alpha_bars[t].item()
        snr = alpha_bar / (1 - alpha_bar)
        result["snr"] = snr

        # Bond/clash metrics for x0_pred
        x0_pred_np = result["x0_pred"]
        if x0_pred_np.ndim == 3:
            x0_pred_np = x0_pred_np[0]
        bonds = ca_bond_lengths(x0_pred_np)
        result["x0_pred_bond_mean"] = float(bonds.mean())
        result["x0_pred_bond_valid_pct"] = float(
            ((bonds > 3.6) & (bonds < 4.0)).mean() * 100
        )
        result["x0_pred_clashes"] = int(compute_clashes(x0_pred_np))

    return result


# =============================================================================
# Device / Dtype Detection
# =============================================================================


def get_device(device_str: str) -> torch.device:
    """Get torch device from string, with auto-detection.

    Priority: CUDA > MPS > CPU
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def get_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    """Get torch dtype from string, validating device compatibility."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    dtype = dtype_map[dtype_str]

    # Check compatibility
    if dtype == torch.bfloat16 and device.type == "mps":
        supported = ["float32", "float16"]
        raise ValueError(
            f"dtype bfloat16 is not supported on device 'mps'.\n"
            f"Supported dtypes for MPS: {', '.join(supported)}"
        )

    return dtype


# =============================================================================
# CLI Implementation
# =============================================================================


def setup_logging(verbose: bool, quiet: bool):
    """Configure logging based on CLI flags."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def create_parent_parser() -> argparse.ArgumentParser:
    """Create parent parser with common arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .pt checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: float32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (file or directory). Default: same as checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential output",
    )

    # Preprocessing
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=None,
        help="Coordinate scaling factor (default: from checkpoint or 10.0)",
    )
    parser.add_argument(
        "--already_preprocessed",
        action="store_true",
        help="Input is already scaled/centered/aligned",
    )

    # Postprocessing
    parser.add_argument(
        "--random_rotate_output",
        action="store_true",
        help="Apply random rotation to output",
    )

    # Quality filtering
    parser.add_argument(
        "--min_bond_pct",
        type=float,
        default=70.0,
        help="Minimum %% of bonds in [3.6, 4.0] range (default: 70)",
    )

    return parser


def create_sample_parser(parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Create parser for sample subcommand."""
    parser = argparse.ArgumentParser(
        parents=[parent],
        description="Generate protein backbone structures unconditionally",
    )

    parser.add_argument(
        "--length",
        type=int,
        required=True,
        help="Number of residues to generate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)",
    )
    parser.add_argument(
        "--add_noise",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use stochastic DDPM sampling (default: True)",
    )
    parser.add_argument(
        "--use_self_cond",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable self-conditioning (default: True)",
    )
    parser.add_argument(
        "--save_at",
        type=str,
        default=None,
        help="Comma-separated timesteps for trajectory, e.g. '900,500,100'",
    )

    return parser


def create_reconstruct_parser(parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Create parser for reconstruct subcommand."""
    parser = argparse.ArgumentParser(
        parents=[parent],
        description="Denoise from a clean or pre-noised structure",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_x0",
        type=str,
        help="Path to clean structure (.npy or .pdb)",
    )
    input_group.add_argument(
        "--input_xt",
        type=str,
        help="Path to pre-noised structure (.npy)",
    )

    parser.add_argument(
        "--start_t",
        type=int,
        required=True,
        help="Timestep to start denoising from",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic sampling (no stochastic noise)",
    )
    parser.add_argument(
        "--shared_noise",
        action="store_true",
        help="Use same noise for all samples in batch",
    )
    parser.add_argument(
        "--no_align_rmsd",
        action="store_true",
        help="Compute raw RMSD without Kabsch alignment",
    )
    parser.add_argument(
        "--save_at",
        type=str,
        default=None,
        help="Comma-separated timesteps for trajectory",
    )

    return parser


def create_denoise_step_parser(parent: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Create parser for denoise_step subcommand."""
    parser = argparse.ArgumentParser(
        parents=[parent],
        description="Single-step denoising for debugging",
    )

    parser.add_argument(
        "--input_xt",
        type=str,
        required=True,
        help="Path to noisy structure (.npy)",
    )
    parser.add_argument(
        "--t",
        type=int,
        required=True,
        help="Current timestep",
    )
    parser.add_argument(
        "--full_debug",
        action="store_true",
        help="Output additional intermediate values",
    )

    return parser


def get_output_dir(args, checkpoint_path: Path) -> Path:
    """Determine output directory from args."""
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix:
            # It's a file, use parent directory
            return out_path.parent
        return out_path
    else:
        # Default to checkpoint directory
        return checkpoint_path.parent


def check_output_exists(path: Path, force: bool):
    """Check if output exists and handle according to --force flag."""
    if path.exists() and not force:
        raise FileExistsError(
            f"Output file '{path}' already exists.\n"
            f"Use --force to overwrite."
        )


def cmd_sample(args):
    """Execute sample subcommand."""
    setup_logging(args.verbose, args.quiet)

    # Load checkpoint
    checkpoint_path = Path(args.model)
    checkpoint = load_checkpoint(checkpoint_path)

    # Validate length
    max_len = get_max_training_length(checkpoint)
    if args.length > max_len:
        raise ValueError(
            f"Length {args.length} exceeds maximum training length ({max_len}).\n"
            f"The model was not trained on sequences this long."
        )

    # Setup device and dtype
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Build model and schedule
    model = build_model(checkpoint, device, dtype)
    schedule = build_schedule(checkpoint, device)

    # Get scale factor
    scale_factor = args.scale_factor
    if scale_factor is None:
        scale_factor = checkpoint.get("args", {}).get("scale_factor", 10.0)

    # Parse save_at
    save_at = None
    if args.save_at:
        save_at = [int(t.strip()) for t in args.save_at.split(",")]

    # Generate
    coords, trajectory, metrics = run_sample(
        model=model,
        schedule=schedule,
        device=device,
        length=args.length,
        batch_size=args.batch_size,
        scale_factor=scale_factor,
        add_noise=args.add_noise,
        use_self_cond=args.use_self_cond,
        save_at=save_at,
        seed=args.seed,
        random_rotate=args.random_rotate_output,
    )

    # Quality filtering
    passed, rejected, mask = filter_samples(coords, args.min_bond_pct)
    metrics["num_passed"] = int(mask.sum())
    metrics["num_rejected"] = int((~mask).sum())

    # Setup output directory
    out_dir = get_output_dir(args, checkpoint_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save passed samples
    for i, idx in enumerate(np.where(mask)[0]):
        pdb_path = out_dir / f"sample_{i}.pdb"
        check_output_exists(pdb_path, args.force)
        save_pdb(coords[idx], pdb_path)
        logger.info(f"Saved: {pdb_path}")

    # Save rejected samples
    if len(rejected) > 0:
        rejected_dir = out_dir / "rejected"
        rejected_dir.mkdir(exist_ok=True)
        for i, idx in enumerate(np.where(~mask)[0]):
            pdb_path = rejected_dir / f"rejected_{i}.pdb"
            save_pdb(coords[idx], pdb_path)
        logger.info(f"Saved {len(rejected)} rejected samples to {rejected_dir}")

    # Save trajectory if requested
    if trajectory:
        traj_path = out_dir / "trajectory.pdb"
        check_output_exists(traj_path, args.force)
        # Use first sample for trajectory
        traj_coords = [c[0] for _, c in trajectory]
        traj_timesteps = [t for t, _ in trajectory]
        save_trajectory_pdb(traj_coords, traj_path, traj_timesteps)
        logger.info(f"Saved trajectory: {traj_path}")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    logger.info(f"Saved metrics: {metrics_path}")

    logger.info(
        f"Generated {args.batch_size} samples: "
        f"{metrics['num_passed']} passed, {metrics['num_rejected']} rejected"
    )


def cmd_reconstruct(args):
    """Execute reconstruct subcommand."""
    setup_logging(args.verbose, args.quiet)

    # Load checkpoint
    checkpoint_path = Path(args.model)
    checkpoint = load_checkpoint(checkpoint_path)

    # Setup device and dtype
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Build model and schedule
    model = build_model(checkpoint, device, dtype)
    schedule = build_schedule(checkpoint, device)

    # Get scale factor
    scale_factor = args.scale_factor
    if scale_factor is None:
        scale_factor = checkpoint.get("args", {}).get("scale_factor", 10.0)

    # Load input
    if args.input_x0:
        x0 = load_coords_from_file(args.input_x0)
        logger.info(f"Loaded clean structure: {args.input_x0}, shape={x0.shape}")
    else:
        # input_xt case - load pre-noised
        # For pre-noised, we skip the forward noising step
        # Assume it's already in scaled units
        raise NotImplementedError(
            "--input_xt for pre-noised structures is not yet fully implemented. "
            "Use --input_x0 instead."
        )

    # Parse save_at
    save_at = None
    if args.save_at:
        save_at = [int(t.strip()) for t in args.save_at.split(",")]

    # Reconstruct
    recon, trajectory, metrics = run_reconstruct(
        model=model,
        schedule=schedule,
        device=device,
        x0=x0,
        start_t=args.start_t,
        scale_factor=scale_factor,
        deterministic=args.deterministic,
        use_self_cond=True,
        shared_noise=args.shared_noise,
        save_at=save_at,
        seed=args.seed,
        align_for_rmsd=not args.no_align_rmsd,
    )

    # Setup output
    out_dir = get_output_dir(args, checkpoint_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if args.out and Path(args.out).suffix:
        out_path = Path(args.out)
    else:
        out_path = out_dir / "reconstructed.pdb"

    check_output_exists(out_path, args.force)

    # Save (first sample if batch)
    if recon.ndim == 3:
        save_pdb(recon[0], out_path)
    else:
        save_pdb(recon, out_path)
    logger.info(f"Saved reconstruction: {out_path}")

    # Save trajectory if requested
    if trajectory:
        traj_path = out_dir / "trajectory.pdb"
        check_output_exists(traj_path, args.force)
        traj_coords = [c[0] if c.ndim == 3 else c for _, c in trajectory]
        traj_timesteps = [t for t, _ in trajectory]
        save_trajectory_pdb(traj_coords, traj_path, traj_timesteps)
        logger.info(f"Saved trajectory: {traj_path}")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    logger.info(f"Saved metrics: {metrics_path}")

    if "rmsd_mean" in metrics:
        logger.info(f"Reconstruction RMSD: {metrics['rmsd_mean']:.2f} A")


def cmd_denoise_step(args):
    """Execute denoise_step subcommand."""
    setup_logging(args.verbose, args.quiet)

    # Load checkpoint
    checkpoint_path = Path(args.model)
    checkpoint = load_checkpoint(checkpoint_path)

    # Setup device and dtype
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Build model and schedule
    model = build_model(checkpoint, device, dtype)
    schedule = build_schedule(checkpoint, device)

    # Get scale factor
    scale_factor = args.scale_factor
    if scale_factor is None:
        scale_factor = checkpoint.get("args", {}).get("scale_factor", 10.0)

    # Load input (assume already in scaled units)
    x_t = np.load(args.input_xt)
    x_t = _normalize_coord_shape(x_t)
    logger.info(f"Loaded noisy structure: {args.input_xt}, shape={x_t.shape}")

    # Run single step
    result = run_denoise_step(
        model=model,
        schedule=schedule,
        device=device,
        x_t=x_t,
        t=args.t,
        scale_factor=scale_factor,
        full_debug=args.full_debug,
    )

    # Setup output
    out_dir = get_output_dir(args, checkpoint_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    eps_path = out_dir / "eps_pred.npy"
    check_output_exists(eps_path, args.force)
    np.save(eps_path, result["eps_pred"])
    logger.info(f"Saved eps_pred: {eps_path}")

    x0_npy_path = out_dir / "x0_pred.npy"
    check_output_exists(x0_npy_path, args.force)
    np.save(x0_npy_path, result["x0_pred"])
    logger.info(f"Saved x0_pred (npy): {x0_npy_path}")

    x0_pdb_path = out_dir / "x0_pred.pdb"
    check_output_exists(x0_pdb_path, args.force)
    x0_pred = result["x0_pred"]
    if x0_pred.ndim == 3:
        x0_pred = x0_pred[0]
    save_pdb(x0_pred, x0_pdb_path)
    logger.info(f"Saved x0_pred (pdb): {x0_pdb_path}")

    if args.full_debug:
        x_prev_path = out_dir / "x_prev.npy"
        np.save(x_prev_path, result["x_prev"])
        logger.info(f"Saved x_prev: {x_prev_path}")

        logger.info(f"SNR at t={args.t}: {result['snr']:.4f}")
        logger.info(
            f"x0_pred bonds: mean={result['x0_pred_bond_mean']:.2f} A, "
            f"valid={result['x0_pred_bond_valid_pct']:.1f}%"
        )
        logger.info(f"x0_pred clashes: {result['x0_pred_clashes']}")

    # Save debug info as JSON
    debug_info = {
        "timestep": result["timestep"],
    }
    if args.full_debug:
        debug_info["snr"] = result["snr"]
        debug_info["x0_pred_bond_mean"] = result["x0_pred_bond_mean"]
        debug_info["x0_pred_bond_valid_pct"] = result["x0_pred_bond_valid_pct"]
        debug_info["x0_pred_clashes"] = result["x0_pred_clashes"]

    debug_path = out_dir / "debug_info.json"
    save_metrics(debug_info, debug_path)
    logger.info(f"Saved debug info: {debug_path}")


def main():
    """Main entry point."""
    # Create parent parser
    parent = create_parent_parser()

    # Create main parser with subcommands
    parser = argparse.ArgumentParser(
        description="Inference for protein structure diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add subcommands
    sample_parser = subparsers.add_parser(
        "sample",
        parents=[parent],
        help="Generate samples unconditionally",
    )
    sample_parser.add_argument("--length", type=int, required=True)
    sample_parser.add_argument("--batch_size", type=int, default=1)
    sample_parser.add_argument(
        "--add_noise", type=lambda x: x.lower() == "true", default=True
    )
    sample_parser.add_argument(
        "--use_self_cond", type=lambda x: x.lower() == "true", default=True
    )
    sample_parser.add_argument("--save_at", type=str, default=None)
    sample_parser.set_defaults(func=cmd_sample)

    reconstruct_parser = subparsers.add_parser(
        "reconstruct",
        parents=[parent],
        help="Reconstruct from clean/noisy structure",
    )
    input_group = reconstruct_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_x0", type=str)
    input_group.add_argument("--input_xt", type=str)
    reconstruct_parser.add_argument("--start_t", type=int, required=True)
    reconstruct_parser.add_argument("--deterministic", action="store_true")
    reconstruct_parser.add_argument("--shared_noise", action="store_true")
    reconstruct_parser.add_argument("--no_align_rmsd", action="store_true")
    reconstruct_parser.add_argument("--save_at", type=str, default=None)
    reconstruct_parser.set_defaults(func=cmd_reconstruct)

    denoise_parser = subparsers.add_parser(
        "denoise_step",
        parents=[parent],
        help="Single-step denoising for debugging",
    )
    denoise_parser.add_argument("--input_xt", type=str, required=True)
    denoise_parser.add_argument("--t", type=int, required=True)
    denoise_parser.add_argument("--full_debug", action="store_true")
    denoise_parser.set_defaults(func=cmd_denoise_step)

    # Parse args
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except FileExistsError as e:
        logger.error(str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                f"CUDA out of memory.\n"
                f"Try reducing --batch_size (current: {getattr(args, 'batch_size', 'N/A')})."
            )
        else:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
