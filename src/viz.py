"""Visualization utilities using py3Dmol."""

import numpy as np
import py3Dmol

from .pdb_io import ca_to_pdb_str


def show_pdb(
    pdb_str: str,
    width: int = 600,
    height: int = 400,
    style: str = "cartoon",
    color: str = "spectrum",
) -> py3Dmol.view:
    """Render a PDB string in a 3D viewer.

    Args:
        pdb_str: PDB format string
        width: Viewer width in pixels
        height: Viewer height in pixels
        style: Visualization style ('cartoon', 'stick', 'sphere', 'line')
        color: Color scheme ('spectrum', 'chain', 'residue', or a color name)

    Returns:
        py3Dmol view object
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_str, "pdb")

    style_dict = {style: {"color": color}}
    view.setStyle(style_dict)
    view.zoomTo()

    return view


def show_ca_trace(
    coords: np.ndarray,
    seq: str | None = None,
    width: int = 600,
    height: int = 400,
    color: str = "spectrum",
) -> py3Dmol.view:
    """Render CA coordinates as a trace.

    Args:
        coords: (L, 3) array of CA coordinates
        seq: Optional amino acid sequence
        width: Viewer width in pixels
        height: Viewer height in pixels
        color: Color scheme

    Returns:
        py3Dmol view object
    """
    pdb_str = ca_to_pdb_str(coords, seq=seq)
    return show_pdb(pdb_str, width=width, height=height, style="cartoon", color=color)


def show_multiple(
    coords_list: list[np.ndarray],
    labels: list[str] | None = None,
    width: int = 600,
    height: int = 400,
) -> py3Dmol.view:
    """Show multiple structures overlaid.

    Args:
        coords_list: List of (L, 3) coordinate arrays
        labels: Optional labels for each structure
        width: Viewer width
        height: Viewer height

    Returns:
        py3Dmol view object
    """
    colors = ["blue", "red", "green", "orange", "purple", "cyan"]
    view = py3Dmol.view(width=width, height=height)

    for i, coords in enumerate(coords_list):
        pdb_str = ca_to_pdb_str(coords, name=labels[i] if labels else f"struct_{i}")
        view.addModel(pdb_str, "pdb")
        view.setStyle({"model": i}, {"cartoon": {"color": colors[i % len(colors)]}})

    view.zoomTo()
    return view
