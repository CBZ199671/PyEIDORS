"""Optimized mesh generator."""

import numpy as np
import gmsh
import meshio
import tempfile
import time
from pathlib import Path
from math import pi, cos, sin
from typing import Optional, Dict, Any, Union, Tuple
from contextlib import contextmanager
import logging

from ..data.structures import MeshConfig, ElectrodePosition
from .mesh_converter import MeshConverter

# Set up logging
logger = logging.getLogger(__name__)

# Check FEniCS availability
try:
    from fenics import Mesh
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS not available, can only generate raw mesh files")


class MeshGenerator:
    """Optimized mesh generator."""

    def __init__(self, config: MeshConfig, electrodes: ElectrodePosition):
        self.config = config
        self.electrodes = electrodes
        self.mesh_data = {}

    @contextmanager
    def gmsh_context(self, model_name: str = "EIT_Mesh"):
        """GMsh context manager."""
        gmsh.initialize()
        gmsh.model.add(model_name)
        try:
            yield
        finally:
            gmsh.finalize()

    def generate(self, output_dir: Optional[Path] = None,
                 use_fenics: bool = True) -> Union[Mesh, Dict[str, Any]]:
        """Generate mesh.

        Args:
            output_dir: Output directory.
            use_fenics: Whether to convert to FEniCS format.

        Returns:
            FEniCS Mesh object or mesh data dictionary.
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        mesh_file = output_dir / f"mesh_{int(time.time() * 1e6) % 1000000}.msh"

        with self.gmsh_context():
            self._create_geometry()
            self._set_physical_groups()
            self._generate_mesh()
            gmsh.write(str(mesh_file))

            # Save electrode vertex data
            self._extract_electrode_vertices()

        if use_fenics and FENICS_AVAILABLE:
            return self._convert_to_fenics(mesh_file, output_dir)
        else:
            return {
                'mesh_file': mesh_file,
                'radius': self.config.radius,
                'electrodes': self.electrodes,
                'vertex_data': self.mesh_data.get('electrode_vertices', [])
            }

    def _create_geometry(self):
        """Create geometry."""
        positions = self.electrodes.positions
        n_in = self.config.electrode_vertices
        n_out = self.config.gap_vertices
        r = self.config.radius

        boundary_points = []
        electrode_ranges = []

        # Create boundary points
        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)

            # Electrode points
            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)

            electrode_ranges.append((start_idx, len(boundary_points) - 1))

            # Gap points
            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi

            gap_points = np.linspace(gap_start, gap_end, n_out + 2)[1:-1]
            for theta in gap_points:
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)

        # Create boundary lines
        lines = []
        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            line = gmsh.model.occ.addLine(boundary_points[i], boundary_points[next_i])
            lines.append(line)

        # Create surface
        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])

        # Add internal control points
        mesh_size_center = 0.095
        cp_distance = 0.1
        center_points = [
            gmsh.model.occ.addPoint(x, y, 0.0, meshSize=mesh_size_center)
            for x, y in [(-cp_distance, cp_distance), (cp_distance, cp_distance),
                         (-cp_distance, -cp_distance), (cp_distance, -cp_distance)]
        ]

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.embed(0, center_points, 2, surface)

        # Save data
        self.mesh_data['boundary_points'] = boundary_points
        self.mesh_data['electrode_ranges'] = electrode_ranges
        self.mesh_data['lines'] = lines
        self.mesh_data['surface'] = surface

    def _set_physical_groups(self):
        """Set physical groups."""
        surface = self.mesh_data['surface']
        lines = self.mesh_data['lines']
        electrode_ranges = self.mesh_data['electrode_ranges']

        # Domain
        gmsh.model.addPhysicalGroup(2, [surface], 1, name="domain")

        # Electrodes
        electrode_lines = []
        for i, (start, end) in enumerate(electrode_ranges):
            lines_for_electrode = []
            for j in range(start, end):
                line_idx = j % len(lines)
                lines_for_electrode.append(lines[line_idx])

            if lines_for_electrode:
                gmsh.model.addPhysicalGroup(1, lines_for_electrode, i + 2,
                                          name=f"electrode_{i+1}")
                electrode_lines.extend(lines_for_electrode)

        # Gaps
        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            gmsh.model.addPhysicalGroup(1, gap_lines, self.electrodes.L + 2, name="gaps")

    def _generate_mesh(self):
        """Generate mesh."""
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
        gmsh.model.mesh.generate(2)

    def _extract_electrode_vertices(self):
        """Extract electrode vertex coordinates."""
        positions = self.electrodes.positions
        r = self.config.radius
        n_in = self.config.electrode_vertices

        electrode_vertices = []
        for start, end in positions:
            vertices = []
            for theta in np.linspace(start, end, n_in):
                vertices.append([r * cos(theta), r * sin(theta)])
            electrode_vertices.append(vertices)

        self.mesh_data['electrode_vertices'] = electrode_vertices

    def _convert_to_fenics(self, mesh_file: Path, output_dir: Path):
        """Convert to FEniCS mesh."""
        converter = MeshConverter(str(mesh_file), str(output_dir))
        mesh, boundaries_mf, association_table = converter.convert()

        # Add compatibility attributes
        mesh.radius = self.config.radius
        mesh.vertex_elec = self.mesh_data.get('electrode_vertices', [])
        mesh.electrodes = self.electrodes
        mesh.boundaries_mf = boundaries_mf
        mesh.association_table = association_table

        return mesh