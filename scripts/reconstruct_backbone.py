"""Reconstruct N-CA-C-O backbone from CA-only PDB.

This script uses vector geometry to "hallucinate" the N, C, and O atoms based on
the curvature and torsion of the CA trace. It produces the 4-atom-per-residue
format (N, CA, C, O) that ProteinMPNN/LigandMPNN strictly requires.

Usage:
    python scripts/reconstruct_backbone.py input.pdb output.pdb

    # Or with uv:
    uv run python scripts/reconstruct_backbone.py input.pdb output.pdb

Example:
    python scripts/reconstruct_backbone.py outputs/sample_0.pdb outputs/sample_0_full.pdb
"""

import argparse
import sys

import numpy as np


def normalize(v):
    """Normalize vectors along last axis."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)


def reconstruct_backbone_geometry(ca_coords):
    """
    Reconstructs N, C, O atoms from CA coordinates using geometric heuristics.
    Assumes standard trans-peptide geometry.

    Args:
        ca_coords: (L, 3) numpy array of CA coordinates

    Returns:
        (L, 4, 3) numpy array containing [N, CA, C, O] for each residue
    """
    L = len(ca_coords)

    # Placeholders for results
    n_coords = np.zeros((L, 3))
    c_coords = np.zeros((L, 3))
    o_coords = np.zeros((L, 3))

    # 1. Compute vectors between CAs
    # v_next[i] = CA[i+1] - CA[i]
    v_next = np.zeros_like(ca_coords)
    v_next[:-1] = ca_coords[1:] - ca_coords[:-1]

    # v_prev[i] = CA[i] - CA[i-1]
    v_prev = np.zeros_like(ca_coords)
    v_prev[1:] = ca_coords[1:] - ca_coords[:-1]

    # Extrapolate termini vectors
    v_next[-1] = v_next[-2]
    v_prev[0] = v_prev[1]

    # Normalize
    v_next = normalize(v_next)
    v_prev = normalize(v_prev)

    # 2. Construct local coordinate frames
    # This roughly defines the "curvature" vector pointing towards the center of the turn
    # If points are collinear, this cross product is zero (handled by small epsilon in norm)
    curvature = normalize(np.cross(v_prev, v_next))

    # Handle straight lines (rare in proteins, but possible in generated noise)
    # If curvature is 0, pick arbitrary orthogonal vector
    invalid_mask = np.linalg.norm(curvature, axis=-1) < 1e-3
    if np.any(invalid_mask):
        dummy = np.array([1.0, 0.0, 0.0])
        fallback = np.cross(v_next, dummy)
        curvature[invalid_mask] = normalize(fallback)[invalid_mask]

    # Re-orthogonalize to ensure perfect frame
    # We want a vector 'y' perpendicular to v_next and in the plane of curvature
    y_axis = normalize(np.cross(v_next, curvature))

    # 3. Place atoms relative to CA using standard geometric constants (approximate)
    # Vectors are combinations of the trace direction (v_next) and the plane normal (y_axis)

    # N is usually "behind" CA and slightly offset
    # Vector CA->N
    vec_ca_n = -0.48 * v_next + 1.28 * y_axis + 0.0 * curvature  # Heuristic weights
    n_coords = ca_coords + normalize(vec_ca_n) * 1.46  # Scale to bond length 1.46A

    # C is usually "ahead" of CA
    # Vector CA->C
    vec_ca_c = 0.49 * v_next + 1.20 * y_axis  # Heuristic weights
    c_coords = ca_coords + normalize(vec_ca_c) * 1.51  # Scale to bond length 1.51A

    # O is attached to C, pointing outward
    # For simplicity, we define O relative to CA for the "direction" then place it on C
    # This is a simplification; rigorous placement uses the C-CA vector
    vec_c_o = 0.1 * v_next + 1.0 * y_axis
    o_coords = c_coords + normalize(vec_c_o) * 1.23

    # 4. Handle Termini better
    # The first N and last C/O are often messy with this simple logic.
    # We just accept the slight geometric error for MPNN purposes.

    # Stack into (L, 4, 3) -> [N, CA, C, O]
    backbone = np.stack([n_coords, ca_coords, c_coords, o_coords], axis=1)

    return backbone


def read_ca_pdb(path):
    """Reads only CA atoms from PDB."""
    coords = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") and " CA " in line:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue
    return np.array(coords)


def write_full_pdb(backbone, out_path):
    """Writes standard PDB format with N, CA, C, O."""
    L = len(backbone)
    atom_types = ["N", "CA", "C", "O"]

    with open(out_path, "w") as f:
        atom_idx = 1
        for i in range(L):
            res_seq = i + 1
            # Write N, CA, C, O for this residue
            for j, atom_type in enumerate(atom_types):
                x, y, z = backbone[i, j]
                # PDB ATOM format
                # col 1-4: "ATOM"
                # col 7-11: Atom serial number
                # col 13-16: Atom name (padded)
                # col 18-20: Residue name "ALA"
                # col 22: Chain ID "A"
                # col 23-26: Residue seq num
                # col 31-38: X
                # col 39-46: Y
                # col 47-54: Z
                # col 55-60: Occupancy (1.00)
                # col 61-66: Temp factor (0.00)
                # col 77-78: Element symbol

                # Align atom name carefully: " N  ", " CA ", " C  ", " O  "
                name_padded = f" {atom_type:<3}"

                line = (
                    f"ATOM  {atom_idx:>5} {name_padded} ALA A{res_seq:>4}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           {atom_type[0]}  \n"
                )
                f.write(line)
                atom_idx += 1
        f.write("END\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct N-CA-C-O backbone from CA-only PDB"
    )
    parser.add_argument("input", help="Input PDB file (CA only)")
    parser.add_argument("output", help="Output PDB file (Full backbone)")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    try:
        ca_coords = read_ca_pdb(args.input)
        if len(ca_coords) == 0:
            print("Error: No CA atoms found in input file.")
            sys.exit(1)

        print(f"Reconstructing backbone for {len(ca_coords)} residues...")
        full_backbone = reconstruct_backbone_geometry(ca_coords)

        print(f"Writing {args.output}...")
        write_full_pdb(full_backbone, args.output)
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
