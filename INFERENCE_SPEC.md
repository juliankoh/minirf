# Inference.py Specification

This document specifies the design and behavior of `inference.py` for the protein structure diffusion model.

---

## 1. Overview

`inference.py` provides three subcommands for running inference with trained diffusion models:

1. **`sample`** - Unconditional generation of protein backbone structures
2. **`reconstruct`** - Denoise from a given clean structure (or pre-noised structure)
3. **`denoise_step`** - Single-step denoising for debugging

---

## 2. Model Architecture

### Assumptions

- **Model type**: DiffusionTransformer (hardcoded for now; refactor when additional architectures exist)
- **Checkpoint format**: Local `.pt` files only (no URL/HuggingFace Hub support)
- **Architecture detection**: Not needed; assume DiffusionTransformer

### Checkpoint Loading

On load, the script:
1. Loads the checkpoint dict from the `.pt` file
2. Reads `checkpoint["args"]` for model configuration
3. **Always prints a summary** of the model config (d_model, layers, heads, etc.)
4. Instantiates DiffusionTransformer with checkpoint config
5. Loads weights

**Length validation**: If user-specified `--length` exceeds the maximum sequence length from training, **fail with a hard error**. No extrapolation allowed.

---

## 3. CLI Structure

Use **argparse with parent parsers** for clean inheritance. Common arguments are defined once in a parent parser, subcommands add their specific arguments.

### Global Arguments (all subcommands)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | required | Path to `.pt` checkpoint file |
| `--device` | str | `"auto"` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `--dtype` | str | `"float32"` | Data type: `float32`, `float16`, `bfloat16` |
| `--seed` | int | `None` | Random seed for full reproducibility |
| `--out` | str | checkpoint dir | Output path (file or directory) |
| `--force` | flag | `False` | Overwrite existing output files |
| `--verbose` | flag | `False` | Enable verbose logging |
| `--quiet` | flag | `False` | Suppress non-essential output |

### Device Auto-Detection Priority

```
CUDA > MPS > CPU
```

MPS (Apple Silicon) is treated as production-ready with no warning.

### Dtype Compatibility

If requested dtype is unsupported on the device, **fail explicitly** with a helpful error message suggesting supported dtypes for that device.

### Output Path Behavior

- If `--out` is not specified: use same directory as checkpoint
- If `--out` is a directory: write files into it
- If `--out` is a file path and file exists: **fail unless `--force` is specified**

---

## 4. Diffusion Schedule Parameters

These are read from the checkpoint by default. User overrides are **allowed but discouraged**.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--T` | int | from ckpt | Number of diffusion timesteps |
| `--schedule_kind` | str | from ckpt | `"cosine"` or `"linear"` |
| `--beta_start` | float | from ckpt | Starting beta value |
| `--beta_end` | float | from ckpt | Ending beta value |

**Behavior**: If user provides any of these and they differ from checkpoint values:
- **Log a warning** about potential quality degradation
- **Use checkpoint values** (not user overrides)

---

## 5. Preprocessing & Postprocessing

### Preprocessing (for input structures)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scale_factor` | float | from ckpt (10.0) | Coordinate scaling factor |
| `--already_preprocessed` | flag | `False` | Input is already scaled/centered/aligned |

**Default behavior**: Assume raw input coordinates. Apply:
1. Center to origin
2. Align to principal axes (first PC along **Z-axis**)
3. Scale by `1/scale_factor`

### Postprocessing (for output structures)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--random_rotate_output` | flag | `False` | Apply random rotation to output |

**Random rotation**: Applied **per-sample** (independent rotation for each structure in batch).

**Metric computation**: Quality metrics are computed **before** random rotation is applied.

---

## 6. Input File Handling

### Supported Formats

- **`.npy`**: NumPy coordinate arrays
- **`.pdb`**: PDB files (CA atoms extracted automatically)

Format is auto-detected from file extension.

### Shape Handling

Accept and auto-reshape common array shapes:
- `(N, 3)` - single structure
- `(B, N, 3)` - batch of structures
- `(1, N, 3, 1)` and similar - squeeze extra dimensions

### NaN Handling

If input contains NaN values:
- **Treat NaN positions as masked residues**
- Proceed with generation/reconstruction
- Model must support masking for this to work correctly

---

## 7. Output Formats

### Coordinate Output

- **`.pdb`**: ALA residues on chain A (CA-only backbone)
- **`.npy`**: NumPy arrays

### PDB Conventions

- Chain ID: `A`
- Residue name: `ALA` (placeholder)
- Atom name: `CA`

### Variable-Length Batches

**Strip-on-Save strategy**:
1. Internally: use padded arrays `(B, L_max, 3)` with mask for GPU efficiency
2. On save: strip padding using mask
3. Save each protein as **individual `.pdb` file** (e.g., `sample_0.pdb`, `sample_1.pdb`)

---

## 8. Quality Metrics

### Output Format

Metrics are saved to a **JSON file** (`metrics.json`) alongside output coordinates. Nothing printed to stdout by default.

### Computed Metrics

For each sample:
- Bond length mean and std
- Percentage of bonds in [3.6, 4.0] Å range
- Radius of gyration
- Clash count
- For reconstruct mode: RMSD vs original (with Kabsch alignment)

### RMSD Computation

**Default**: Use Kabsch alignment (optimal superposition) before computing RMSD.

Add `--no_align_rmsd` flag to compute raw RMSD without alignment.

---

## 9. Quality Filtering

Post-hoc filtering of low-quality samples based on bond length statistics.

### Filter Criterion

**Bond length percentage in range [3.6, 4.0] Å**

### Threshold

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--min_bond_pct` | float | `70.0` | Minimum % of bonds in valid range |

### Rejected Samples

Samples failing the quality filter are saved to a **`rejected/`** subdirectory within the output directory.

The metrics JSON indicates how many samples were rejected.

---

## 10. `sample` Subcommand

Generate protein backbone structures unconditionally.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--length` | int | required | Number of residues |
| `--batch_size` | int | `1` | Number of samples to generate |
| `--add_noise` | bool | `True` | Stochastic DDPM vs deterministic |
| `--use_self_cond` | bool | `True` | Enable self-conditioning |
| `--save_at` | str | `None` | Comma-separated timesteps for trajectory (e.g., "900,500,100") |

### Self-Conditioning Initialization

First step uses **zeros tensor** as the initial x0 estimate.

### Seed Behavior

`--seed` controls **full trajectory**: initial noise and all stochastic sampling steps. Same seed = exact same output.

### Memory Handling

If OOM occurs, **fail with error** and suggest reducing `--batch_size`.

### Progress Display

Use **tqdm progress bar** for sampling loop.

### Trajectory Output

When `--save_at` is specified:
- Save as **multi-model PDB file** with MODEL/ENDMDL records
- Can be animated in molecular viewers (PyMOL, etc.)
- Filename: `trajectory.pdb`

### Example

```bash
python inference.py sample \
  --model runs/exp1/model.pt \
  --length 100 \
  --batch_size 8 \
  --seed 42 \
  --out outputs/generated/
```

---

## 11. `reconstruct` Subcommand

Denoise from a clean structure (adds noise, then denoises) or from a pre-noised structure.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_x0` | str | None | Path to clean structure (.npy or .pdb) |
| `--input_xt` | str | None | Path to pre-noised structure |
| `--start_t` | int | required | Timestep to start denoising from |
| `--deterministic` | flag | `False` | Alias for `--add_noise False` |
| `--shared_noise` | flag | `False` | Use same noise for all samples in batch |

Must provide exactly one of `--input_x0` or `--input_xt`.

### Noise Behavior

- **Default**: Independent noise per sample in batch
- **With `--shared_noise`**: Same noise realization for all samples (useful for ablations)

### Seed Behavior

Single `--seed` controls both forward noising and reverse sampling.

### Example

```bash
python inference.py reconstruct \
  --model runs/exp1/model.pt \
  --input_x0 reference.pdb \
  --start_t 500 \
  --deterministic \
  --out outputs/reconstructed.pdb
```

---

## 12. `denoise_step` Subcommand

Single-step denoising for debugging and analysis.

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_xt` | str | required | Path to noisy structure |
| `--t` | int | required | Current timestep |
| `--full_debug` | flag | `False` | Output additional intermediate values |

### Output

**Default output**:
- `eps_pred`: Predicted noise (saved as .npy)
- `x0_pred`: Predicted clean structure (saved as .npy and .pdb)

**With `--full_debug`**:
- Also output `x_{t-1}` prediction
- SNR for the timestep
- Bond length violations and clash score for `x0_pred`

### Example

```bash
python inference.py denoise_step \
  --model runs/exp1/model.pt \
  --input_xt noisy_coords.npy \
  --t 500 \
  --full_debug \
  --out outputs/debug/
```

---

## 13. Logging

Use **Python logging module** with configurable levels:

- Default: `INFO` level
- `--verbose`: `DEBUG` level
- `--quiet`: `WARNING` level only

Log format includes timestamp and level.

---

## 14. Internal API Structure

Organize code into testable functions:

### Core Functions

```python
def load_checkpoint(path: str) -> dict:
    """Load checkpoint and validate."""

def build_model(checkpoint: dict, device: str, dtype: torch.dtype) -> nn.Module:
    """Instantiate and load model weights."""

def build_schedule(checkpoint: dict, device: str) -> DiffusionSchedule:
    """Create diffusion schedule from checkpoint args."""

def preprocess_coords(
    coords: np.ndarray,
    scale_factor: float,
    center: bool = True,
    align: bool = True
) -> tuple[torch.Tensor, dict]:
    """Preprocess coordinates. Returns tensor and metadata for inverse."""

def postprocess_coords(
    coords: torch.Tensor,
    scale_factor: float,
    random_rotate: bool = False
) -> np.ndarray:
    """Convert back to Angstroms, optionally rotate."""
```

### Task Functions

```python
def run_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    length: int,
    batch_size: int,
    **kwargs
) -> np.ndarray:
    """Generate samples. Returns (B, L, 3) in Angstroms."""

def run_reconstruct(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x0: np.ndarray,
    start_t: int,
    **kwargs
) -> tuple[np.ndarray, dict]:
    """Reconstruct from x0. Returns coords and metrics."""

def run_denoise_step(
    model: nn.Module,
    schedule: DiffusionSchedule,
    xt: np.ndarray,
    t: int,
    **kwargs
) -> dict:
    """Single step. Returns dict with eps_pred, x0_pred, etc."""
```

### Utility Functions

```python
def save_pdb(coords: np.ndarray, path: str) -> None:
    """Save as PDB with ALA residues."""

def save_trajectory_pdb(coords_list: list[np.ndarray], path: str) -> None:
    """Save multi-model PDB for trajectory visualization."""

def compute_metrics(coords: np.ndarray, reference: np.ndarray = None) -> dict:
    """Compute quality metrics."""

def filter_samples(
    coords: np.ndarray,
    min_bond_pct: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter by quality. Returns (passed, rejected, mask)."""

def kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimal superposition of mobile onto target."""
```

---

## 15. Error Messages

Provide clear, actionable error messages:

```
Error: Length 500 exceeds maximum training length (256).
       The model was not trained on sequences this long.

Error: Output file 'output.pdb' already exists.
       Use --force to overwrite.

Error: dtype bfloat16 is not supported on device 'mps'.
       Supported dtypes for MPS: float32, float16

Error: CUDA out of memory.
       Try reducing --batch_size (current: 64).
       Suggested: 32 or lower based on available memory.
```

---

## 16. Summary of Key Defaults

| Setting | Default Value |
|---------|---------------|
| Device priority | CUDA > MPS > CPU |
| dtype | float32 |
| scale_factor | 10.0 (from checkpoint) |
| Self-cond init | Zeros |
| Seed scope | Full trajectory |
| Quality threshold | 70% bonds in range |
| RMSD alignment | Kabsch (enabled) |
| Principal axis | First PC along Z |
| PDB residue | ALA |
| Output location | Same as checkpoint |
| Overwrite | Disabled (require --force) |

---

## 17. Example Workflows

### Generate 100 samples of length 128

```bash
python inference.py sample \
  --model runs/best/model.pt \
  --length 128 \
  --batch_size 100 \
  --seed 42 \
  --out outputs/gen_128/
```

### Reconstruct with trajectory visualization

```bash
python inference.py reconstruct \
  --model runs/best/model.pt \
  --input_x0 native.pdb \
  --start_t 800 \
  --save_at 800,600,400,200,0 \
  --out outputs/recon/
```

### Debug single denoising step

```bash
python inference.py denoise_step \
  --model runs/best/model.pt \
  --input_xt noisy.npy \
  --t 500 \
  --full_debug \
  --out outputs/debug/
```
