"""Geometry utilities for protein structure analysis.

Usage:
    uv run python -m src.geom

Functions Overview
------------------

center(x)
    Moves the protein so its center of mass is at the origin (0,0,0).
    Why: Diffusion models work better when data is centered. Also required
    before Kabsch alignment.

ca_bond_lengths(x)
    Measures distances between consecutive CA atoms.
    Real proteins have CA-CA distances of ~3.8 Angstrom (physical constraint).
    If generated structure has 1 or 10 Angstrom distances, it's broken.

kabsch_align(P, Q)
    Rotates and translates structure P to best overlap with structure Q.
    Uses SVD to find optimal rotation matrix. You can't compare two structures
    directly if one is rotated/translated - Kabsch removes this ambiguity.

rmsd(P, Q, align=True)
    Measures how different two structures are (in Angstroms).
        0-1 A  = Nearly identical
        1-3 A  = Very similar
        3-5 A  = Same fold, some differences
        >10 A  = Completely different
    Primary metric for evaluating if generated structure matches target.

radius_of_gyration(x)
    Measures how spread out/compact the protein is.
    Sanity check that generated structures aren't collapsed to a point or
    exploded outward.

compute_ca_dihedrals(x)
    Measures the twist angle between 4 consecutive CA atoms.
        ~50 deg   = Alpha helix
        ~180 deg  = Beta sheet
        random    = Disordered loop
    RMSD can be fooled by a crumpled wire that roughly overlaps the target.
    Dihedrals check that LOCAL geometry is correct.

Summary Table
-------------
    center              Is the protein at the origin?
    ca_bond_lengths     Are atoms the right distance apart?
    kabsch_align        How do I overlay two structures?
    rmsd                How similar are two structures globally?
    radius_of_gyration  Is the protein the right size?
    compute_ca_dihedrals Does local geometry look like a real protein?
"""

import numpy as np


def center(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center coordinates at the origin.

    Args:
        x: (L, 3) array of coordinates

    Returns:
        x_centered: (L, 3) centered coordinates
        centroid: (3,) original centroid
    """
    centroid = x.mean(axis=0)
    x_centered = x - centroid
    return x_centered, centroid


def ca_bond_lengths(x: np.ndarray) -> np.ndarray:
    """Compute distances between consecutive CA atoms.

    Args:
        x: (L, 3) array of CA coordinates

    Returns:
        (L-1,) array of distances between consecutive atoms
    """
    return np.linalg.norm(np.diff(x, axis=0), axis=1)


def kabsch_align(
    P: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align P onto Q using the Kabsch algorithm.

    Finds the optimal rotation and translation to minimize RMSD.

    Args:
        P: (L, 3) array of coordinates to be aligned
        Q: (L, 3) array of reference coordinates

    Returns:
        P_aligned: (L, 3) aligned coordinates
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    # Center both structures
    P_centered, P_centroid = center(P)
    Q_centered, Q_centroid = center(Q)

    # Compute covariance matrix
    H = P_centered.T @ Q_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Handle reflection case (ensure proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = Q_centroid - R @ P_centroid

    # Apply transformation
    P_aligned = (R @ P.T).T + t

    return P_aligned, R, t


def rmsd(P: np.ndarray, Q: np.ndarray, align: bool = True) -> float:
    """Compute RMSD between two structures.

    Args:
        P: (L, 3) array of coordinates
        Q: (L, 3) array of coordinates
        align: If True, align P onto Q first using Kabsch

    Returns:
        RMSD value in same units as input coordinates
    """
    if align:
        P_aligned, _, _ = kabsch_align(P, Q)
    else:
        P_aligned = P

    diff = P_aligned - Q
    return np.sqrt((diff**2).sum() / len(P))


def radius_of_gyration(x: np.ndarray) -> float:
    """Compute radius of gyration.

    Args:
        x: (L, 3) array of coordinates

    Returns:
        Radius of gyration
    """
    x_centered, _ = center(x)
    return np.sqrt((x_centered**2).sum() / len(x))


def random_rotation_matrix(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random 3D rotation matrix (uniformly distributed).

    Uses QR decomposition of a random matrix to generate uniform rotations.

    Args:
        rng: NumPy random generator (default: creates new one)

    Returns:
        (3, 3) rotation matrix (determinant = 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random matrix -> QR decomposition gives orthogonal matrix
    random_matrix = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(random_matrix)

    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    return Q


def apply_random_rotation(
    x: np.ndarray, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a random rotation to coordinates.

    Args:
        x: (L, 3) array of coordinates
        rng: NumPy random generator

    Returns:
        x_rotated: (L, 3) rotated coordinates
        R: (3, 3) rotation matrix that was applied
    """
    R = random_rotation_matrix(rng)
    x_rotated = (R @ x.T).T
    return x_rotated, R


def compute_ca_dihedrals(ca_coords: np.ndarray) -> np.ndarray:
    """Compute pseudo-dihedral angles between 4 consecutive CA atoms.

    The angle is defined by the planes formed by (i, i+1, i+2) and (i+1, i+2, i+3).
    Useful for validating local protein geometry:
    - Alpha helices cluster around +50°
    - Beta sheets cluster around ±180°

    Args:
        ca_coords: (L, 3) array of CA coordinates

    Returns:
        (L-3,) array of angles in radians, range [-pi, pi]
    """
    # Vectors between consecutive CAs
    v = np.diff(ca_coords, axis=0)  # (L-1, 3)

    # b1 = v[i], b2 = v[i+1], b3 = v[i+2]
    b1 = v[:-2]
    b2 = v[1:-1]
    b3 = v[2:]

    # Normals to planes (a,b,c) and (b,c,d)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize (epsilon to avoid division by zero for collinear atoms)
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True) + 1e-8
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-8

    # Cosine of angle (dot product of normals)
    x = (n1 * n2).sum(axis=1)

    # Sine of angle (requires checking orientation relative to b2)
    m = np.cross(n1, b2)
    m /= np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8
    y = (m * n2).sum(axis=1)

    return np.arctan2(y, x)


def align_to_principal_axes(coords):
    """Align coordinates to principal axes via SVD."""
    # coords: (L, 3) centered
    u, s, vh = np.linalg.svd(coords, full_matrices=False)
    # vh contains the rotation matrix
    return coords @ vh.T


def main():
    """Test geometry utilities."""
    from pathlib import Path

    from .data_cath import get_one_chain

    # Load a protein
    result = get_one_chain(Path("data/chain_set.jsonl"))
    if result is None:
        print("No chain found")
        return

    name, seq, ca_coords = result
    print(f"Loaded {name}, {len(seq)} residues")

    # Test centering
    centered, centroid = center(ca_coords)
    print(f"\nCentering:")
    print(
        f"  Original centroid: [{centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}]"
    )
    print(
        f"  New centroid: [{centered.mean(axis=0)[0]:.2e}, {centered.mean(axis=0)[1]:.2e}, {centered.mean(axis=0)[2]:.2e}]"
    )

    # Test CA bond lengths
    bonds = ca_bond_lengths(ca_coords)
    print(f"\nCA-CA bond lengths:")
    print(f"  Mean: {bonds.mean():.2f} Å")
    print(f"  Std:  {bonds.std():.2f} Å")
    print(f"  Range: [{bonds.min():.2f}, {bonds.max():.2f}] Å")

    # Test Kabsch alignment (apply random rotation + translation, then recover)
    print(f"\nKabsch alignment test:")
    rng = np.random.default_rng(42)

    # Random rotation matrix (via QR decomposition)
    random_matrix = rng.standard_normal((3, 3))
    R_random, _ = np.linalg.qr(random_matrix)
    if np.linalg.det(R_random) < 0:
        R_random[:, 0] *= -1

    # Random translation
    t_random = rng.standard_normal(3) * 50

    # Apply transformation
    ca_transformed = (R_random @ ca_coords.T).T + t_random

    # Recover with Kabsch
    rmsd_before = rmsd(ca_transformed, ca_coords, align=False)
    rmsd_after = rmsd(ca_transformed, ca_coords, align=True)

    print(f"  RMSD before alignment: {rmsd_before:.2f} Å")
    print(f"  RMSD after alignment:  {rmsd_after:.2e} Å (should be ~0)")

    # Radius of gyration
    rg = radius_of_gyration(ca_coords)
    print(f"\nRadius of gyration: {rg:.2f} Å")

    # Test pseudo-dihedrals
    dihedrals = compute_ca_dihedrals(ca_coords)
    dihedrals_deg = np.degrees(dihedrals)

    print(f"\nCA Pseudo-dihedrals (L-3={len(dihedrals)}):")
    print(f"  Mean: {dihedrals_deg.mean():.1f}°")
    print(f"  Std:  {dihedrals_deg.std():.1f}°")

    # Secondary structure content (smell test for generated structures)
    helix_fraction = ((dihedrals_deg > 40) & (dihedrals_deg < 70)).mean()
    sheet_fraction = ((dihedrals_deg > 160) | (dihedrals_deg < -160)).mean()
    print(f"  Helix-like (40° to 70°):   {helix_fraction:.1%}")
    print(f"  Sheet-like (|angle|>160°): {sheet_fraction:.1%}")


if __name__ == "__main__":
    main()
