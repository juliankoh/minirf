#!/usr/bin/env python3
"""
Analyze all training runs and extract hyperparameters + evaluation metrics to CSV.
"""

import os
import re
import csv
import torch
from pathlib import Path
from datetime import datetime


def parse_hyperparams_from_log(log_path: Path) -> dict:
    """Extract hyperparameters from train.log file."""
    hyperparams = {}

    if not log_path.exists():
        return hyperparams

    with open(log_path, 'r') as f:
        content = f.read()

    # Look for the Hyperparameters section
    hyperparam_match = re.search(r'Hyperparameters:\n((?:\s+\w+:\s+[\w\.\-]+\n)+)', content)
    if hyperparam_match:
        lines = hyperparam_match.group(1).strip().split('\n')
        for line in lines:
            match = re.match(r'\s*(\w+):\s+([\w\.\-]+)', line)
            if match:
                key, value = match.groups()
                # Try to convert to appropriate type
                try:
                    if value.lower() in ('true', 'false'):
                        hyperparams[key] = value.lower() == 'true'
                    elif '.' in value:
                        hyperparams[key] = float(value)
                    else:
                        hyperparams[key] = int(value)
                except ValueError:
                    hyperparams[key] = value

    return hyperparams


def parse_hyperparams_from_checkpoint(checkpoint_path: Path) -> dict:
    """Extract hyperparameters from a PyTorch checkpoint."""
    hyperparams = {}

    if not checkpoint_path.exists():
        return hyperparams

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'args' in checkpoint:
            args = checkpoint['args']
            if hasattr(args, '__dict__'):
                hyperparams = vars(args)
            elif isinstance(args, dict):
                hyperparams = args
    except Exception as e:
        print(f"  Warning: Could not load checkpoint {checkpoint_path}: {e}")

    return hyperparams


def parse_evals_from_log(log_path: Path) -> dict:
    """Extract evaluation metrics from train.log file."""
    evals = {}

    if not log_path.exists():
        return evals

    with open(log_path, 'r') as f:
        content = f.read()

    # Extract final train loss
    final_loss_match = re.search(r'Final train loss \(avg last \d+\):\s+([\d\.]+)', content)
    if final_loss_match:
        evals['final_train_loss'] = float(final_loss_match.group(1))

    # Extract best train loss
    best_loss_match = re.search(r'Best train loss:\s+([\d\.]+)', content)
    if best_loss_match:
        evals['best_train_loss'] = float(best_loss_match.group(1))

    # Extract final evaluation metrics (timestep-based)
    final_eval_match = re.search(
        r'Final Evaluation\n=+\n\s+Timestep\s+Loss\s+RMSD.*?\n-+\n((?:\s+\d+\s+[\d\.]+\s+[\d\.]+\n?)+)',
        content
    )
    if final_eval_match:
        lines = final_eval_match.group(1).strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                timestep = parts[0]
                loss = parts[1]
                rmsd = parts[2]
                evals[f'eval_loss_t{timestep}'] = float(loss)
                evals[f'eval_rmsd_t{timestep}'] = float(rmsd)

    # Extract reconstruction RMSD from periodic samples
    recon_matches = re.findall(r'Reconstruction \(Train\) from t=(\d+): RMSD=([\d\.]+)', content)
    if recon_matches:
        # Take the last one
        last_t, last_rmsd = recon_matches[-1]
        evals['last_recon_rmsd'] = float(last_rmsd)
        evals['last_recon_t'] = int(last_t)

    # Extract bond statistics
    bond_match = re.search(r'Bonds: mean=([\d\.]+).*?std=([\d\.]+).*?valid%=([\d\.]+)', content)
    if bond_match:
        evals['bond_mean'] = float(bond_match.group(1))
        evals['bond_std'] = float(bond_match.group(2))
        evals['bond_valid_pct'] = float(bond_match.group(3))

    # Check if overfitting was successful
    overfit_success = re.search(r'Overfitting successful', content)
    evals['overfit_success'] = overfit_success is not None

    # Extract initial evaluation metrics
    initial_match = re.search(r'Initial evaluation \(t=(\d+)\):\n\s+Loss:\s+([\d\.]+),\s+RMSD:\s+([\d\.]+)', content)
    if initial_match:
        evals['initial_t'] = int(initial_match.group(1))
        evals['initial_loss'] = float(initial_match.group(2))
        evals['initial_rmsd'] = float(initial_match.group(3))

    # Extract total training steps completed
    step_matches = re.findall(r'^\s*(\d+)\s+[\d\.]+\s+[\d\.]+', content, re.MULTILINE)
    if step_matches:
        evals['steps_completed'] = int(step_matches[-1])

    return evals


def analyze_run(run_dir: Path) -> dict:
    """Analyze a single run directory and extract all metadata."""
    result = {
        'run_dir': run_dir.name,
        'run_path': str(run_dir),
        'source': None,
    }

    # Try to parse timestamp from directory name
    try:
        timestamp = datetime.strptime(run_dir.name, '%Y%m%d_%H%M%S')
        result['timestamp'] = timestamp.isoformat()
    except ValueError:
        result['timestamp'] = None

    # Try to get hyperparameters from log first
    log_path = run_dir / 'train.log'
    hyperparams = parse_hyperparams_from_log(log_path)
    if hyperparams:
        result['source'] = 'log'

    # If no hyperparams from log, try checkpoint
    if not hyperparams:
        for ckpt_name in ['model.pt', 'model_simple.pt', 'toy_model.pt']:
            ckpt_path = run_dir / ckpt_name
            if ckpt_path.exists():
                hyperparams = parse_hyperparams_from_checkpoint(ckpt_path)
                if hyperparams:
                    result['source'] = f'checkpoint:{ckpt_name}'
                    break

    # Add hyperparameters with 'hp_' prefix
    for key, value in hyperparams.items():
        result[f'hp_{key}'] = value

    # Get evaluation metrics from log
    evals = parse_evals_from_log(log_path)
    for key, value in evals.items():
        result[f'eval_{key}'] = value

    # Check what files exist
    result['has_log'] = log_path.exists()
    result['has_model'] = (run_dir / 'model.pt').exists()
    result['has_model_simple'] = (run_dir / 'model_simple.pt').exists()
    result['has_toy_model'] = (run_dir / 'toy_model.pt').exists()
    result['has_loss_curve'] = (run_dir / 'loss_curve.png').exists()

    return result


def main():
    project_root = Path(__file__).parent.parent
    runs_dirs = [
        project_root / 'runs',
        project_root / 'runs_toy',
    ]

    all_runs = []

    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            print(f"Skipping {runs_dir} (does not exist)")
            continue

        print(f"Scanning {runs_dir}...")

        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                print(f"  Analyzing {run_dir.name}...")
                run_data = analyze_run(run_dir)
                run_data['runs_type'] = runs_dir.name  # 'runs' or 'runs_toy'
                all_runs.append(run_data)

    if not all_runs:
        print("No runs found!")
        return

    # Collect all unique keys
    all_keys = set()
    for run in all_runs:
        all_keys.update(run.keys())

    # Sort keys: metadata first, then hp_, then eval_
    def key_sort(k):
        if k.startswith('hp_'):
            return (1, k)
        elif k.startswith('eval_'):
            return (2, k)
        elif k.startswith('has_'):
            return (3, k)
        else:
            return (0, k)

    sorted_keys = sorted(all_keys, key=key_sort)

    # Write to CSV
    output_path = project_root / 'runs_analysis.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_keys)
        writer.writeheader()
        for run in all_runs:
            writer.writerow(run)

    print(f"\nAnalysis complete!")
    print(f"  Total runs analyzed: {len(all_runs)}")
    print(f"  Output saved to: {output_path}")

    # Print summary
    print(f"\nColumns in CSV:")
    for key in sorted_keys:
        print(f"  - {key}")


if __name__ == '__main__':
    main()
