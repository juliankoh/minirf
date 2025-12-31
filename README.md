# minirf

A minimal protein backbone diffusion model for generating protein structures from noise.

## Overview

This project implements a Denoising Diffusion Probabilistic Model (DDPM) that learns to generate protein backbone structures (CA-only) from the CATH dataset. The model uses a Transformer architecture with geometric attention that incorporates pairwise distance information.

**Key features:**
- Epsilon-prediction diffusion (predicts noise, not coordinates)
- Geometric attention with RBF-encoded pairwise distance bias
- Self-conditioning (feeds previous x0 estimate back to model)
- Auxiliary bond-length loss for valid local geometry
- Complete domain training (40-128 residues, no cropping)
- Cosine noise schedule

## Architecture

The `DiffusionTransformer` takes noisy CA coordinates `x_t` and timestep `t`, then predicts the noise `epsilon`:

```
Input: x_t (B, L, 3) + t (B,)
  -> Linear projection to d_model
  -> Add sinusoidal positional encoding (sequence position)
  -> Add timestep embedding (MLP on sinusoidal)
  -> N Transformer blocks with:
     - Pre-LayerNorm
     - Multi-head attention + geometric distance bias
     - FFN with GELU
  -> Output projection to 3D: epsilon_pred (B, L, 3)
```

The attention mechanism adds a learned bias based on RBF-encoded pairwise distances between residues, allowing the model to reason about 3D structure.

## Installation

```bash
# Clone and install
git clone <repo>
cd minirf

# Install dependencies (requires uv)
uv sync
```

Requires the CATH dataset:
- `data/chain_set.jsonl` - protein chain coordinates
- `data/chain_set_splits.json` - train/val/test splits

## Usage

### Training

```bash
# Sanity check: overfit on a single chain
uv run python -m src.train --overfit --steps 2000

# Train on dataset (100 chains, 40-128 residues)
uv run python -m src.train --steps 10000 --num_chains 100

# Larger run
uv run python -m src.train --steps 50000 --num_chains 500 --batch_size 8
```

Training outputs are saved to `runs/<timestamp>/`:
- `model.pt` - final checkpoint
- `best_model.pt` - best validation loss
- `loss_curve.png` - training curve
- `train.log` - full log

### Debugging with Synthetic Data

```bash
# Train on a straight-line chain (perfect 3.8A bonds)
uv run python -m src.train_toy --steps 1000
```

This is useful for verifying the diffusion pipeline works before training on real proteins.

### Evaluation

```bash
# Evaluate a checkpoint
uv run python -m src.eval --model_path runs/<timestamp>/model.pt

# Use test set
uv run python -m src.eval --model_path runs/latest/model.pt --split test
```

Evaluation metrics:
- **Denoiser quality**: MSE and x0-prediction RMSD at various timesteps
- **Reconstruction**: Noise real data -> denoise -> compare to original
- **Generation**: Bond validity, steric clashes, radius of gyration, diversity

### Sampling

```bash
uv run python -m src.sampler
```

Generates structures from pure noise and runs reconstruction tests.

## Modules

| Module | Description |
|--------|-------------|
| `data_cath.py` | CATH dataset streaming, filtering, segment extraction |
| `geom.py` | Geometry utilities (centering, Kabsch RMSD, bond lengths, dihedrals) |
| `diffusion.py` | Forward diffusion (noise schedule, q_sample) |
| `model.py` | DiffusionTransformer with geometric attention |
| `sampler.py` | Reverse diffusion (DDPM posterior sampling) |
| `train.py` | Training loop with self-conditioning and bond loss |
| `eval.py` | Evaluation suite (denoiser, reconstruction, generation metrics) |
| `train_toy.py` | Debugging on synthetic straight-line chains |
| `pdb_io.py` | PDB file writing for CA traces |
| `viz.py` | py3Dmol visualization helpers |

## Coordinate Scaling

Protein coordinates are in Angstroms (typical range ~30A), but diffusion works best with data near N(0,1). All coordinates are scaled by `scale_factor=10.0`:

```python
x_scaled = x_angstroms / 10.0  # for training
x_angstroms = x_scaled * 10.0  # for evaluation/output
```

## Training Details

**Loss function:**
```
L = MSE(epsilon_pred, epsilon) + lambda * bond_loss
```

The bond loss encourages predicted x0 to have ~3.8A CA-CA distances, but is only applied for t < 200 (at high noise, x0 predictions are unreliable).

**Self-conditioning:** 50% of training steps run the model twice - first to get an initial x0 estimate, then with that estimate as additional input.

**Data preprocessing:**
1. Extract longest contiguous resolved segment from each chain
2. Center at origin
3. Align to principal axes (canonical orientation)
4. Scale by 1/10

## Expected Results

After training:
- Reconstruction RMSD from t=300: < 5A
- Bond lengths: mean ~3.8A, >90% in valid range (3.6-4.0A)
- Generated structures: protein-like traces (not collapsed, not exploded)

## Project Structure

```
minirf/
  data/                  # CATH dataset (not included)
  notebooks/             # Jupyter notebooks for exploration
  runs/                  # Training outputs
  src/
    __init__.py
    data_cath.py
    diffusion.py
    eval.py
    geom.py
    model.py
    pdb_io.py
    sampler.py
    train.py
    train_toy.py
    viz.py
```
