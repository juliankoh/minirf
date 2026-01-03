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

## Dataset

Requires the CATH dataset (Ingraham format):
- `data/chain_set.jsonl` - protein chain coordinates
- `data/chain_set_splits.json` - train/val/test splits

### Format

Each line in `chain_set.jsonl` is a JSON object:

```json
{
  "name": "132l.A",
  "num_chains": 2,
  "CATH": ["1.10.530"],
  "seq": "KVFGRCELAA...GCRL",
  "coords": {
    "N": [[-9.649, 18.097, 49.778], ...],
    "CA": [[-8.887, 17.726, 48.556], ...],
    "C": [[-8.095, 16.434, 48.729], ...],
    "O": [[-7.335, 16.282, 49.666], ...]
  }
}
```

For CA-only training: `x_ca = np.array(data['coords']['CA'])` gives shape `(L, 3)`.

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

Useful for verifying the diffusion pipeline works before training on real proteins.

### Inference

The `inference.py` script provides three subcommands for running trained models:

#### Unconditional Sampling

Generate new protein structures from noise:

```bash
uv run python -m src.inference sample \
  --model runs/<timestamp>/model.pt \
  --length 100 \
  --batch_size 8 \
  --seed 42 \
  --out outputs/generated/
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--length` | required | Number of residues |
| `--batch_size` | 1 | Number of samples to generate |
| `--seed` | None | Random seed for reproducibility |
| `--add_noise` | True | Stochastic DDPM vs deterministic |
| `--use_self_cond` | True | Enable self-conditioning |
| `--save_at` | None | Timesteps for trajectory (e.g., "900,500,100") |

#### Reconstruction

Denoise from a clean structure (adds noise, then denoises):

```bash
uv run python -m src.inference reconstruct \
  --model runs/<timestamp>/model.pt \
  --input_x0 reference.pdb \
  --start_t 500 \
  --deterministic \
  --out outputs/reconstructed/
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--input_x0` | None | Path to clean structure (.npy or .pdb) |
| `--input_xt` | None | Path to pre-noised structure |
| `--start_t` | required | Timestep to start denoising from |
| `--deterministic` | False | Disable stochastic sampling |
| `--shared_noise` | False | Same noise for all samples in batch |

#### Single-Step Debugging

Debug a single denoising step:

```bash
uv run python -m src.inference denoise_step \
  --model runs/<timestamp>/model.pt \
  --input_xt noisy_coords.npy \
  --t 500 \
  --full_debug \
  --out outputs/debug/
```

#### Global Inference Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to `.pt` checkpoint |
| `--device` | auto | Device: `auto`, `cuda`, `mps`, `cpu` |
| `--dtype` | float32 | Data type: `float32`, `float16`, `bfloat16` |
| `--seed` | None | Random seed (controls full trajectory) |
| `--out` | checkpoint dir | Output path |
| `--force` | False | Overwrite existing files |
| `--min_bond_pct` | 70.0 | Quality filter threshold |

Device auto-detection priority: `CUDA > MPS > CPU`

### Evaluation

```bash
# Evaluate a checkpoint
uv run python -m src.eval --model_path runs/<timestamp>/model.pt

# Use test set
uv run python -m src.eval --model_path runs/latest/model.pt --split test
```

## Preprocessing

Input coordinates go through the following pipeline:

1. **Extract** longest contiguous resolved segment from each chain
2. **Center** at origin
3. **Align** to principal axes (first PC along Z-axis)
4. **Scale** by `1/scale_factor` (default: 10.0)

Protein coordinates are in Angstroms (typical range ~30A), but diffusion works best with data near N(0,1).

```python
x_scaled = x_angstroms / 10.0  # for training
x_angstroms = x_scaled * 10.0  # for evaluation/output
```

For inference with `--already_preprocessed`, the input is assumed to be already scaled/centered/aligned.

## Quality Metrics

The following metrics are computed for generated/reconstructed structures:

| Metric | Description |
|--------|-------------|
| Bond length mean | Average CA-CA distance (target: ~3.8 A) |
| Bond length std | Standard deviation of CA-CA distances |
| Valid bond % | Percentage of bonds in [3.6, 4.0] A range |
| Radius of gyration | Compactness measure |
| Clash count | Number of non-bonded atoms < 3.0 A apart |
| RMSD | For reconstruction: vs original with Kabsch alignment |

### Quality Filtering

Samples are filtered based on bond validity. Samples with < 70% valid bonds (configurable via `--min_bond_pct`) are saved to a `rejected/` subdirectory.

## Output Formats

### PDB Files
- Chain ID: `A`
- Residue name: `ALA` (placeholder)
- Atom name: `CA`

For batch generation, each sample is saved as an individual file (e.g., `sample_0.pdb`, `sample_1.pdb`).

### Trajectory Files
When using `--save_at`, trajectories are saved as multi-model PDB files with MODEL/ENDMDL records, viewable as animations in PyMOL.

### NumPy Arrays
Coordinates can also be saved as `.npy` files with shape `(B, L, 3)` or `(L, 3)`.

## Training Details

**Loss function:**
```
L = MSE(epsilon_pred, epsilon) + lambda * bond_loss
```

The bond loss encourages predicted x0 to have ~3.8A CA-CA distances, but is only applied for t < 200 (at high noise, x0 predictions are unreliable).

**Self-conditioning:** 50% of training steps run the model twice - first to get an initial x0 estimate, then with that estimate as additional input.

## Expected Results

After training:
- Reconstruction RMSD from t=300: < 5A
- Bond lengths: mean ~3.8A, >90% in valid range (3.6-4.0A)
- Generated structures: protein-like traces (not collapsed, not exploded)

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
| `inference.py` | CLI for sampling, reconstruction, and debugging |
| `train_toy.py` | Debugging on synthetic straight-line chains |
| `pdb_io.py` | PDB file writing for CA traces |
| `viz.py` | py3Dmol visualization helpers |

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
    inference.py
    model.py
    pdb_io.py
    sampler.py
    train.py
    train_toy.py
    viz.py
```
