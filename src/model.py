"""Diffusion Transformer for protein structure denoising.

Usage:
    uv run python -m src.model

This module implements a Transformer-based coordinate predictor for diffusion
models. The key architectural feature is pairwise distance bias in attention,
which incorporates 3D geometric information.

Architecture:
    1. Input: noisy CA coordinates (B, L, 3) + timestep t
    2. Project coordinates to d_model dimension
    3. Add sinusoidal positional encoding (sequence position)
    4. Add timestep embedding
    5. Process through N transformer blocks with geometric attention
    6. Project back to 3D: predict noise epsilon (B, L, 3)

Prediction Target:
    This model predicts epsilon (noise) rather than x0 (clean coordinates).
    This makes learning more uniform across timesteps - at high t, x_t ≈ ε,
    so predicting ε is easy; at low t, x_t ≈ x0, so ε is also learnable.

The attention mechanism uses RBF-encoded pairwise distances as an additive
bias, allowing the model to reason about 3D structure while maintaining
the efficiency of standard attention.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Generate sinusoidal positional embeddings.

    Uses the standard formula from "Attention is All You Need":
        PE(pos, 2i) = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Args:
        positions: (...) position indices (any shape)
        dim: Embedding dimension (should be even)

    Returns:
        (..., dim) positional embeddings
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=positions.device) / half_dim
    )

    # (..., 1) * (half_dim,) -> (..., half_dim)
    angles = positions.unsqueeze(-1) * freqs

    # Interleave sin and cos
    embeddings = torch.zeros(*positions.shape, dim, device=positions.device)
    embeddings[..., 0::2] = torch.sin(angles)
    embeddings[..., 1::2] = torch.cos(angles)

    return embeddings


class RBFDistanceEmbedding(nn.Module):
    """Embed pairwise distances using radial basis functions.

    Converts continuous distances into a soft histogram representation
    that the model can learn from. RBFs are centered at evenly spaced
    values from 0 to max_dist.

    Args:
        num_rbf: Number of RBF centers
        max_dist: Maximum distance cutoff (in scaled coordinate units)

    Note:
        With scale_factor=10.0, distances are in model units not Angstroms.
        A CA-CA distance of 3.8A becomes 0.38 model units.
        max_dist=2.0 covers ~20 Angstroms physical distance.
    """

    def __init__(self, num_rbf: int = 16, max_dist: float = 2.0) -> None:
        super().__init__()
        self.num_rbf = num_rbf

        # RBF centers evenly spaced from 0 to max_dist
        self.register_buffer("mu", torch.linspace(0, max_dist, num_rbf))

        # Width of each RBF (covers the range evenly)
        sigma = max_dist / num_rbf
        self.register_buffer("sigma", torch.tensor(sigma))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute RBF embeddings for pairwise distances.

        Args:
            distances: (B, L, L) pairwise distance matrix

        Returns:
            (B, L, L, num_rbf) RBF embeddings
        """
        # (B, L, L, 1) - (num_rbf,) -> (B, L, L, num_rbf)
        d = distances.unsqueeze(-1)
        return torch.exp(-((d - self.mu) ** 2) / (2 * self.sigma**2))


class TimestepEmbedding(nn.Module):
    """Embed diffusion timesteps using sinusoidal encoding + MLP.

    Follows the common practice of sinusoidal encoding followed by
    a small MLP to create learnable timestep representations.

    Args:
        d_model: Output dimension
        hidden_dim: Hidden dimension for MLP (default: d_model)
    """

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: (B,) integer timestep indices

        Returns:
            (B, d_model) timestep embeddings
        """
        # Sinusoidal base embedding
        t_embed = sinusoidal_embedding(t.float(), self.d_model)
        # MLP projection
        return self.mlp(t_embed)


class PairwiseBiasedAttention(nn.Module):
    """Multi-head self-attention with geometric distance bias.

    Incorporates 3D structural information by adding a learned bias
    to attention logits based on pairwise distances between residues.

    The attention formula becomes:
        attn = softmax((Q @ K^T) / sqrt(d_k) + distance_bias)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_rbf: Number of RBF centers for distance embedding
        max_dist: Maximum distance for RBF embedding
        dropout: Attention dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_rbf: int = 16,
        max_dist: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        # Q, K, V projections (fused for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Geometric bias components
        self.rbf_embed = RBFDistanceEmbedding(num_rbf, max_dist)
        self.bias_proj = nn.Linear(num_rbf, num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention with geometric bias.

        Args:
            x: (B, L, d_model) input features
            coords: (B, L, 3) 3D coordinates for distance computation
            mask: (B, L) boolean mask, True for valid positions

        Returns:
            (B, L, d_model) output features
        """
        B, L, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3 * d_model)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, D)

        # Attention logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Compute geometric bias from pairwise distances
        dists = torch.cdist(coords, coords)  # (B, L, L)
        rbf = self.rbf_embed(dists)  # (B, L, L, num_rbf)
        geo_bias = self.bias_proj(rbf)  # (B, L, L, num_heads)
        geo_bias = geo_bias.permute(0, 3, 1, 2)  # (B, H, L, L)

        # Add geometric bias
        attn_logits = attn_logits + geo_bias

        # Apply mask if provided
        if mask is not None:
            # mask: (B, L) where True = valid, False = padding
            # Need to mask out attention TO padding positions
            mask_2d = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            attn_logits = attn_logits.masked_fill(~mask_2d, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)  # (B, L, d_model)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and geometric attention.

    Uses pre-LayerNorm architecture (norm before attention/FFN) for
    more stable training.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward hidden dimension (default: 4 * d_model)
        num_rbf: Number of RBF centers
        max_dist: Maximum distance for RBF
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        num_rbf: int = 16,
        max_dist: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PairwiseBiasedAttention(
            d_model, num_heads, num_rbf, max_dist, dropout
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            x: (B, L, d_model) input features
            coords: (B, L, 3) coordinates for distance bias
            mask: (B, L) boolean mask

        Returns:
            (B, L, d_model) output features
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x), coords, mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class DiffusionTransformer(nn.Module):
    """Diffusion Transformer for protein structure denoising.

    Predicts noise epsilon from noisy coordinates x_t and timestep t.
    The noise prediction is used by the sampler to iteratively denoise.

    The model uses geometric attention with pairwise distance bias to
    incorporate 3D structural information.

    Args:
        d_model: Transformer hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward dimension (default: 4 * d_model)
        num_rbf: RBF centers for distance embedding
        max_dist: Maximum distance for RBF (in scaled coords)
        max_len: Maximum sequence length for positional encoding
        dropout: Dropout probability

    Example:
        >>> model = DiffusionTransformer(d_model=256, num_layers=6, num_heads=8)
        >>> x_t = torch.randn(2, 100, 3)  # Batch of 2, length 100
        >>> t = torch.tensor([500, 750])  # Timesteps
        >>> eps_pred = model(x_t, t)  # (2, 100, 3) predicted noise
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int | None = None,
        num_rbf: int = 16,
        max_dist: float = 2.0,
        max_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Input projection: 3D coords -> d_model
        self.input_proj = nn.Linear(3, d_model)

        # Self-conditioning projection: previous x0 estimate -> d_model
        # Initialized to zero so model works without self-conditioning initially
        self.self_cond_proj = nn.Linear(3, d_model)
        nn.init.zeros_(self.self_cond_proj.weight)
        nn.init.zeros_(self.self_cond_proj.bias)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, num_rbf, max_dist, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final norm and output projection
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 3)

        # Initialize output projection to near-zero for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor | int,
        mask: torch.Tensor | None = None,
        x0_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise epsilon from noisy input.

        Args:
            x_t: (B, L, 3) or (L, 3) noisy CA coordinates
            t: (B,) or int timestep indices
            mask: (B, L) boolean mask, True for valid positions
            x0_self_cond: (B, L, 3) previous x0 estimate for self-conditioning.
                          If None, self-conditioning is disabled for this call.

        Returns:
            eps_pred: Same shape as x_t, predicted noise
        """
        # Handle unbatched input
        unbatched = x_t.dim() == 2
        if unbatched:
            x_t = x_t.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            if x0_self_cond is not None:
                x0_self_cond = x0_self_cond.unsqueeze(0)

        B, L, _ = x_t.shape

        # Handle int timestep
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # Expand t to batch size if needed
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)

        # Input projection
        x = self.input_proj(x_t)  # (B, L, d_model)

        # Add self-conditioning if provided
        if x0_self_cond is not None:
            x = x + self.self_cond_proj(x0_self_cond)  # (B, L, d_model)

        # Add positional encoding
        pos_idx = torch.arange(L, device=x_t.device)
        pos_enc = sinusoidal_embedding(pos_idx, self.d_model)  # (L, d_model)
        x = x + pos_enc

        # Add timestep embedding (broadcast over sequence)
        t_emb = self.time_embed(t)  # (B, d_model)
        x = x + t_emb.unsqueeze(1)  # (B, L, d_model)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, coords=x_t, mask=mask)

        # Output projection
        x = self.final_norm(x)
        eps_pred = self.output_proj(x)  # (B, L, 3)

        # Handle unbatched output
        if unbatched:
            eps_pred = eps_pred.squeeze(0)

        return eps_pred


def main():
    """Smoke test: verify forward pass and gradients work."""
    from pathlib import Path

    from .data_cath import get_one_chain
    from .diffusion import DiffusionSchedule
    from .geom import center

    print("=" * 60)
    print("DiffusionTransformer Smoke Test")
    print("=" * 60)

    # Load one chain
    result = get_one_chain(Path("data/chain_set.jsonl"))
    if result is None:
        print("No chain found - using random data")
        x0 = torch.randn(1, 50, 3) * 0.1
        name = "random"
    else:
        name, seq, ca_coords = result
        ca_centered, _ = center(ca_coords)
        x0 = torch.from_numpy(ca_centered).float().unsqueeze(0) / 10.0
        print(f"Loaded {name}, {len(seq)} residues")

    # Create model
    model = DiffusionTransformer(d_model=128, num_layers=4, num_heads=4)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Test forward pass
    schedule = DiffusionSchedule(T=1000)
    t = torch.tensor([500])
    x_t, eps = schedule.q_sample(x0, t)

    print("\nForward pass:")
    print(f"  Input x_t shape:  {x_t.shape}")
    print(f"  Timestep t:       {t.item()}")

    eps_pred = model(x_t, t)
    print(f"  Output eps_pred:  {eps_pred.shape}")

    # Compute loss (MSE to true noise epsilon)
    loss = F.mse_loss(eps_pred, eps)
    print(f"  Initial MSE loss: {loss.item():.4f}")

    # Verify gradients
    print("\nGradient check:")
    loss.backward()
    grad_norms = {}
    for pname, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[pname] = param.grad.norm().item()

    print(f"  Parameters with gradients: {len(grad_norms)}/{num_params}")
    print(f"  Gradient norm range: [{min(grad_norms.values()):.6f}, {max(grad_norms.values()):.4f}]")

    # Test unbatched input
    print("\nUnbatched input test:")
    x_t_unbatched = x_t.squeeze(0)
    eps_pred_unbatched = model(x_t_unbatched, 500)
    print(f"  Input:  {x_t_unbatched.shape}")
    print(f"  Output: {eps_pred_unbatched.shape}")

    # Test with mask
    print("\nMask test:")
    mask = torch.ones(1, x_t.shape[1], dtype=torch.bool)
    mask[0, -10:] = False  # Mask last 10 positions
    eps_pred_masked = model(x_t, t, mask=mask)
    print(f"  Mask shape: {mask.shape} (last 10 positions masked)")
    print(f"  Output:     {eps_pred_masked.shape}")

    print("\n" + "=" * 60)
    print("Smoke test passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
