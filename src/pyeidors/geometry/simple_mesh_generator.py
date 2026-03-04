"""Simplified EIT mesh generator built on top of DOLFINx mesh pipeline."""

from __future__ import annotations

import logging
from math import pi
from pathlib import Path
from typing import Optional

from .optimized_mesh_generator import create_eit_mesh

logger = logging.getLogger(__name__)


class SimpleEITMeshGenerator:
    """Simplified wrapper around ``create_eit_mesh``."""

    def __init__(
        self,
        n_elec: int = 16,
        radius: float = 1.0,
        mesh_size: float = 0.1,
        electrode_width: float = 0.1,
    ):
        self.n_elec = n_elec
        self.radius = radius
        self.mesh_size = mesh_size
        self.electrode_width = electrode_width

        segment = 2 * pi / max(1, n_elec)
        self.electrode_coverage = min(1.0, max(1e-6, electrode_width / segment))
        self.refinement = max(2, int(round(radius / max(mesh_size, 1e-6) / 2)))

    def generate_circular_mesh(self, output_dir: str = None, save_files: bool = True):
        """Generate a circular EIT mesh.

        Args:
            output_dir: Output directory for ``.msh`` cache.
            save_files: Retained for API stability. Mesh cache is always written.
        """

        if not save_files:
            logger.info("save_files=False is ignored; mesh cache file is always written in DOLFINx mode")

        return create_eit_mesh(
            n_elec=self.n_elec,
            radius=self.radius,
            refinement=self.refinement,
            electrode_coverage=self.electrode_coverage,
            output_dir=output_dir,
        )


def create_simple_eit_mesh(
    n_elec: int = 16,
    radius: float = 1.0,
    mesh_size: float = 0.1,
    output_dir: Optional[str] = None,
):
    generator = SimpleEITMeshGenerator(
        n_elec=n_elec,
        radius=radius,
        mesh_size=mesh_size,
        electrode_width=2 * pi / n_elec / 8,
    )
    return generator.generate_circular_mesh(output_dir=output_dir)
