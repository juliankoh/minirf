"""PDB file I/O utilities for CA-only protein structures."""

from pathlib import Path

import numpy as np


def ca_to_pdb_str(
    coords: np.ndarray,
    seq: str | None = None,
    name: str = "PROT",
    chain_id: str = "A",
) -> str:
    """Convert CA coordinates to a PDB format string.

    Args:
        coords: (L, 3) array of CA atom coordinates
        seq: Optional amino acid sequence (single letter codes)
        name: Structure name for HEADER record
        chain_id: Chain identifier

    Returns:
        PDB format string
    """
    # 3-letter amino acid codes
    aa_map = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
        "X": "UNK",
    }

    L = len(coords)
    if seq is None:
        seq = "G" * L  # Default to glycine

    lines = [f"HEADER    {name}"]

    for i, (xyz, aa) in enumerate(zip(coords, seq)):
        res_name = aa_map.get(aa.upper(), "UNK")
        atom_num = i + 1
        res_num = i + 1
        x, y, z = xyz

        # PDB ATOM record format
        line = (
            f"ATOM  {atom_num:5d}  CA  {res_name} {chain_id}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)

    lines.append("END")
    return "\n".join(lines)


def write_pdb(path: str | Path, pdb_str: str) -> None:
    """Write PDB string to file.

    Args:
        path: Output file path
        pdb_str: PDB format string
    """
    Path(path).write_text(pdb_str)
