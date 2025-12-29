# Minimal Protein Diffusion Model - Implementation Plan

## Dataset Reference

The Ingraham CATH dataset format:
- `chain_set.jsonl` - one JSON object per line (~493MB)
- `chain_set_splits.json` - train/val/test identifiers

Schema:
```json
{
  "name": "132l.A",
  "num_chains": 2,
  "CATH": ["1.10.530"],
  "seq": "KVFGRCELAA...GCRL",
  "coords": {
    "N": [[-9.649, 18.097, 49.778], [-8.277, 15.485, 47.81], ...],
    "CA": [[-8.887, 17.726, 48.556], [-7.528, 14.242, 47.858], ...],
    "C": [[-8.095, 16.434, 48.729], [-6.505, 14.284, 46.761], ...],
    "O": [[-7.335, 16.282, 49.666], [-6.824, 14.36, 45.6], ...]
  }
}
```

**`coords` is a dict with keys `["N", "CA", "C", "O"]`, each containing a list of `[x, y, z]` per residue**

For CA-only: `x_ca = np.array(data['coords']['CA'])` → shape `(L, 3)`

---

## Step 0: Repo Skeleton + Dependencies

**Goal:** Establish project structure to prevent sprawl.

### Checklist
- [x] Create `requirements.txt` or `pyproject.toml` with dependencies
- [x] Create `src/` directory with `__init__.py`
- [x] Create `notebooks/` directory
- [x] Create `runs/` directory for tensorboard logs
- [x] Verify imports work: `python -c "import torch; print(torch.__version__)"`

### Target Structure
```
protein_diffusion_min/
  data/                    # raw downloads go here
  notebooks/
    00_sanity_load_visualize.ipynb
    01_forward_diffusion_sanity.ipynb
    02_overfit_single_chain.ipynb
    03_sampling_eval.ipynb
  src/
    __init__.py
    data_cath.py           # jsonl streaming + filtering
    geom.py                # center, kabsch, rmsd, ca_dist checks
    pdb_io.py              # coords->PDB string/file
    viz.py                 # py3Dmol helpers (thin wrappers)
    diffusion.py           # schedules + q_sample + p_sample
    model.py               # tiny Transformer eps-predictor
    train.py               # train step + logging
    eval.py                # metrics + sampling driver
  runs/                    # tensorboard logs
  README.md
```

### Acceptance Criteria
- [x] `python -c "import torch; print(torch.__version__)"` works
- [x] Notebook can import `src.*` without path hacks

---

## Step 1: Download + Stream-Read ONE Chain

**Goal:** Load one protein example and get `(L,3)` CA coords in <50 lines.

### Checklist
- [x] Implement `src/data_cath.py`:
  - [x] `iter_chain_set_jsonl(path)` → yields dicts
  - [x] `select_chain(example, min_len=40, max_len=80, require_no_nans=True)` → bool
  - [x] `get_one_chain(path, ...)` → returns `(name, seq, ca_coords)`
- [x] Create `notebooks/00_sanity_load_visualize.ipynb`:
  - [x] Load one chain
  - [x] Print name + length
  - [x] Show CA coordinates stats

### Acceptance Criteria
- [x] Prints something like `Loaded <name> length=57` *(shows "132l.A" etc)*
- [x] `ca_coords.shape == (L, 3)` *(process_coords returns (L,4,3) with mask)*
- [x] `np.isfinite(ca_coords).all() == True` *(filter_perfect_samples ensures this)*

---

## Step 2: Geometry Utilities

**Goal:** Build debugging tools before training anything.

### Checklist
- [x] Implement `src/geom.py`:
  - [x] `center(x) -> x_centered, centroid`
  - [x] `kabsch_align(P, Q) -> P_aligned, R, t`
  - [x] `rmsd(P, Q, align=True)`
  - [x] `ca_bond_lengths(x)` → returns `||x[i+1]-x[i]||` array

### Acceptance Criteria
- [x] **Kabsch test:** Take `x`, apply random rotation + translation → `x2`, verify `rmsd(x, x2, align=True)` is ~`1e-5` to `1e-3` *(got 2.3e-6)*
- [x] **CA-CA bond test:** `ca_bond_lengths(x)` average is ~3.8 Å for real proteins *(got 3.79 Å)*

---

## Step 3: PDB Writer + py3Dmol Viewer

**Goal:** Ability to "see the bug" visually.

### Checklist
- [x] Implement `src/pdb_io.py`:
  - [x] `ca_to_pdb_str(x, name="X", chain_id="A") -> str`
  - [x] `write_pdb(path, pdb_str)`
- [x] Implement `src/viz.py`:
  - [x] `show_pdb(pdb_str)` using py3Dmol
- [x] Update `notebooks/00_sanity_load_visualize.ipynb` to visualize a chain

### Acceptance Criteria
- [x] Can visually see a plausible protein trace (not a tiny dot, not exploded)

---

## Step 4: Forward Diffusion + Visualization

**Goal:** Implement noise schedule and verify noising "looks right".

### Checklist
- [x] Implement in `src/diffusion.py`:
  - [x] `make_beta_schedule(T, beta_start, beta_end, kind="linear")`
  - [x] Precompute `alphas`, `alpha_bars`
  - [x] `q_sample(x0, t, noise)`: `x_t = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps`
- [x] Create `notebooks/01_forward_diffusion_sanity.ipynb`:
  - [x] Pick one chain `x0`
  - [x] Show `x_t` at timesteps: `t=0, T*0.25, T*0.5, T*0.75, T-1`

### Acceptance Criteria
- [x] At small `t`, structure is mostly intact *(t=0: CA-CA=3.79Å)*
- [x] At large `t`, becomes roughly spherical cloud centered near 0 *(t=999: Rg collapses, CA-CA~2.3Å)*

### Practical Tip
Consider coordinate scaling for stability:
- `x0 = x0 / scale` where `scale = 10.0` (Å → "model units")
- When writing PDB, multiply back by `scale`

---

## Step 5: Minimal ε-Predictor Model

**Goal:** Model that takes `(x_t, t, mask)` and predicts `eps_hat` with shape `(L,3)`.

### Checklist
- [ ] Implement `src/model.py` with `EpsTransformer`:
  - [ ] Input projection: `Linear(3 -> d_model)`
  - [ ] **Residue positional encoding** (sinusoidal or learned) - CRITICAL!
  - [ ] Timestep embedding (sinusoidal MLP → `d_model`)
  - [ ] `TransformerEncoder` 3-4 layers
  - [ ] Output projection: `Linear(d_model -> 3)`
- [ ] Use `src_key_padding_mask` (PyTorch: `True` means "ignore")

### Acceptance Criteria
- [ ] One forward pass works: input `(B,L,3)` → output `(B,L,3)`
- [ ] Loss computation runs without shape errors

### Why Positional Encoding Matters
Without it, Transformer is permutation-invariant over residues → weird "unordered point cloud" behavior even if loss decreases.

---

## Step 6: Overfit a Single Chain

**Goal:** Make the model memorize one protein fast. If this fails, don't scale up.

### Checklist
- [ ] Create `notebooks/02_overfit_single_chain.ipynb`:
  - [ ] Load one chain `x0`
  - [ ] Training loop:
    1. [ ] Sample `t ~ Uniform(1..T)`
    2. [ ] Sample `eps ~ N(0, I)`
    3. [ ] Build `x_t = q_sample(x0, t, eps)`
    4. [ ] `eps_hat = model(x_t, t)`
    5. [ ] Loss = `MSE(eps_hat, eps)`
- [ ] Add TensorBoard logging:
  - [ ] Train loss
  - [ ] `x0_recon_rmsd` diagnostic

### Critical Diagnostic
Compute implied `x0_hat`:
```
x0_hat = (x_t - sqrt(1-alpha_bar_t)*eps_hat) / sqrt(alpha_bar_t)
```
Then compute Kabsch RMSD(`x0_hat`, `x0`) occasionally.

### Acceptance Criteria
- [ ] Training loss drops sharply
- [ ] `x0_hat` RMSD trends downward
- [ ] `ca_bond_lengths(x0_hat)` becomes reasonable (not perfect, but not insane)

---

## Step 7: Sampling Loop (Reverse Diffusion) + Evaluation

**Goal:** Start from noise and denoise to a full CA chain.

### Checklist
- [ ] Add to `src/diffusion.py`:
  - [ ] `p_sample(x_t, t, eps_hat)` (DDPM update)
  - [ ] `sample(model, shape=(L,3), T=..., ...)` full loop
- [ ] Create `notebooks/03_sampling_eval.ipynb`:
  - [ ] Sample 5-20 structures
  - [ ] Align to `x0` and compute RMSD distribution
  - [ ] Visualize best / median sample

### Acceptance Criteria
- [ ] Samples are not exploding (NaNs) and not collapsing to a point
- [ ] At least some samples have visibly protein-like continuity
- [ ] Overfit case: at least one sample close to native after alignment

---

## Ticket Summary (Sequential Tasks)

| Ticket | Description | Status |
|--------|-------------|--------|
| TICKET-001 | Create repo skeleton + requirements + importable `src/` package | [x] done |
| TICKET-002 | Implement `data_cath.py` streaming jsonl reader; return CA coords using `coords['CA']` | [x] done |
| TICKET-003 | Implement `geom.py` (center, Kabsch, RMSD, CA-CA distances) + unit tests | [x] done |
| TICKET-004 | Implement `pdb_io.py` + `viz.py` + Notebook 00 that renders one chain | [x] done |
| TICKET-005 | Implement `diffusion.py` schedule + `q_sample` + Notebook 01 visualizing noising | [x] done |
| TICKET-006 | Implement `EpsTransformer` (with residue + time embeddings) + smoke test | [ ] |
| TICKET-007 | Notebook 02: overfit single chain + TensorBoard + `x0_hat` RMSD diagnostic | [ ] |
| TICKET-008 | Implement reverse sampling + Notebook 03 eval: RMSD + geometry checks + viz | [ ] |

---

## Critical Rule

> **"No modeling code until I can load one chain, center it, write a CA-only PDB, visualize it, and compute Kabsch RMSD."**

This tight loop prevents 90% of diffusion-project dead ends.

---

## After Overfit Works: Next Minimal Expansion

Once single-chain overfit works:
- [ ] Train on small subset (e.g., 1k short chains, length 80-120)
- [ ] Add padding + mask properly
- [ ] Evaluate unconditional samples with:
  - [ ] CA-CA bond length stats
  - [ ] Radius of gyration distribution
  - [ ] Nearest-neighbor distance distribution
  - [ ] Visual inspection

**No SE(3), no sidechains yet.**
