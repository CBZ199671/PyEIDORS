#!/usr/bin/env python3
"""
Visualize electrode indices and lengths on the mesh for quick verification of CEM electrode markers.

Example:
  python scripts/diagnostics/plot_electrode_tags.py --mesh-name mesh_102070 \
      --output results/electrode_visualization/mesh_102070.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from fenics import Measure, assemble

from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh


def collect_electrode_segments(mesh, tags: List[int]) -> Dict[int, List[np.ndarray]]:
    """Collect boundary segment coordinates for each electrode tag."""
    coords = mesh.coordinates()
    segments: Dict[int, List[np.ndarray]] = {tag: [] for tag in tags}
    from dolfin import facets  # Import local iterator for compatibility with older FEniCS API

    for facet in facets(mesh):
        tag = mesh.boundaries_mf[facet.index()]
        if tag in segments:
            vs = facet.entities(0)
            seg_xy = coords[vs][:, :2]  # Use only first two dimensions
            segments[tag].append(seg_xy)
    return segments


def plot_electrodes(mesh, electrode_tags: List[int], lengths: Dict[int, float], output: Path):
    """Plot electrode segments with labels showing index and length."""
    segments = collect_electrode_segments(mesh, electrode_tags)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw all electrode segments
    for tag in electrode_tags:
        segs = segments.get(tag, [])
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], color="tab:blue", lw=2)

    # Label index and length at electrode centroid
    for tag in electrode_tags:
        segs = segments.get(tag, [])
        if not segs:
            continue
        all_pts = np.vstack(segs)
        centroid = all_pts.mean(axis=0)
        length = lengths.get(tag, 0.0)
        ax.text(
            centroid[0],
            centroid[1],
            f"{tag-1}\\n{length:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )

    ax.set_aspect("equal")
    ax.set_title("Electrode tags (label: idx/length)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"), help="Mesh directory")
    parser.add_argument("--mesh-name", type=str, default="mesh_102070", help="Mesh name (without extension)")
    parser.add_argument("--n-elec", type=int, default=16, help="Number of electrodes")
    parser.add_argument("--refinement", type=int, default=12, help="Mesh refinement parameter")
    parser.add_argument("--radius", type=float, default=1.0, help="Mesh radius (for cache key)")
    parser.add_argument("--electrode-coverage", type=float, default=0.5, help="Electrode coverage ratio (for cache key)")
    parser.add_argument("--output", type=Path, default=Path("results/electrode_visualization/electrodes.png"),
                        help="Output PNG path")
    args = parser.parse_args()

    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        mesh_name=args.mesh_name,
        n_elec=args.n_elec,
        refinement=args.refinement,
        radius=args.radius,
        electrode_coverage=args.electrode_coverage,
    )

    # Get electrode tags and lengths
    if not hasattr(mesh, "boundaries_mf") or mesh.boundaries_mf is None:
        raise RuntimeError("Mesh lacks boundaries_mf, cannot visualize electrode tags")

    ds = Measure("ds", domain=mesh, subdomain_data=mesh.boundaries_mf)
    assoc = mesh.association_table

    # Prefer filtering by "electrode_*" key names, ignoring gaps/domain
    electrode_tags: List[int] = []
    for k, v in assoc.items():
        if isinstance(k, str) and k.lower().startswith("electrode"):
            try:
                electrode_tags.append(int(v))
            except Exception:
                continue

    # Fallback: numeric keys >= 2, excluding gaps marker
    if not electrode_tags:
        gap_tag = assoc.get("gaps", None) if isinstance(assoc, dict) else None
        for v in assoc.values():
            if isinstance(v, int) and v >= 2 and v != gap_tag:
                electrode_tags.append(v)

    electrode_tags = sorted(set(electrode_tags))
    lengths = {tag: float(assemble(1 * ds(tag))) for tag in electrode_tags}

    print("Electrode tags:", electrode_tags)
    print("Lengths:", [lengths[t] for t in electrode_tags])

    plot_electrodes(mesh, electrode_tags, lengths, args.output)
    print(f"Saved electrode visualization to {args.output}")


if __name__ == "__main__":
    main()
