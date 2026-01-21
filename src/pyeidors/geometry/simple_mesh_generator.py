"""Simplified EIT mesh generator - Uses GMsh to generate FEniCS-compatible meshes."""

import numpy as np
import tempfile
import time
from pathlib import Path
from math import pi, cos, sin
import logging

logger = logging.getLogger(__name__)

# Check dependencies
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    logger.warning("GMsh not available, cannot generate mesh")

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    logger.warning("meshio not available, mesh conversion functionality limited")

try:
    from fenics import Mesh, MeshFunction, HDF5File
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS not available, cannot create FEniCS mesh objects")


class SimpleEITMeshGenerator:
    """Simplified EIT mesh generator."""

    def __init__(self, n_elec: int = 16, radius: float = 1.0,
                 mesh_size: float = 0.1, electrode_width: float = 0.1):
        """Initialize mesh generator.

        Args:
            n_elec: Number of electrodes.
            radius: Circular domain radius.
            mesh_size: Mesh size.
            electrode_width: Electrode width (radians).
        """
        if not GMSH_AVAILABLE:
            raise ImportError("GMsh not available, please install: pip install gmsh")

        self.n_elec = n_elec
        self.radius = radius
        self.mesh_size = mesh_size
        self.electrode_width = electrode_width

        # Calculate electrode positions
        self.electrode_positions = self._calculate_electrode_positions()

    def _calculate_electrode_positions(self):
        """Calculate electrode positions."""
        positions = []
        for i in range(self.n_elec):
            center_angle = 2 * pi * i / self.n_elec
            start_angle = center_angle - self.electrode_width / 2
            end_angle = center_angle + self.electrode_width / 2
            positions.append((start_angle, end_angle))
        return positions

    def generate_circular_mesh(self, output_dir: str = None,
                              save_files: bool = True) -> object:
        """Generate circular EIT mesh.

        Args:
            output_dir: Output directory.
            save_files: Whether to save mesh files.

        Returns:
            FEniCS mesh object (if available) or mesh info dictionary.
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate timestamp
        timestamp = int(time.time() * 1000) % 1000000
        mesh_name = f"eit_mesh_{timestamp}"

        logger.info(f"Starting EIT mesh generation: {mesh_name}")

        # Initialize GMsh
        gmsh.initialize()
        gmsh.model.add(mesh_name)

        try:
            # Create geometry
            self._create_simple_circular_geometry()

            # Set physical groups
            self._set_physical_groups()

            # Generate mesh
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.mesh_size)
            gmsh.model.mesh.generate(2)

            # Save mesh files
            if save_files:
                msh_file = output_path / f"{mesh_name}.msh"
                gmsh.write(str(msh_file))
                logger.info(f"Mesh file saved: {msh_file}")

            # Convert to FEniCS format
            if FENICS_AVAILABLE:
                fenics_mesh = self._convert_to_fenics(mesh_name, output_path, save_files)
                logger.info("FEniCS mesh created successfully")
                return fenics_mesh
            else:
                logger.warning("FEniCS not available, returning mesh info")
                return self._create_mesh_info(mesh_name, output_path)

        finally:
            gmsh.finalize()

    def _create_simple_circular_geometry(self):
        """Create simplified circular geometry."""
        # Create center point
        center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.mesh_size)

        # Create boundary points
        boundary_points = []
        for i in range(self.n_elec * 4):  # 4 points per electrode region
            angle = 2 * pi * i / (self.n_elec * 4)
            x = self.radius * cos(angle)
            y = self.radius * sin(angle)
            point = gmsh.model.geo.addPoint(x, y, 0.0, self.mesh_size)
            boundary_points.append(point)

        # Create boundary lines (arcs)
        boundary_lines = []
        electrode_lines = []
        gap_lines = []

        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            line = gmsh.model.geo.addCircleArc(boundary_points[i], center, boundary_points[next_i])
            boundary_lines.append(line)

            # Determine if this line is electrode or gap
            # Every 4 lines, the 2nd and 3rd are electrodes
            local_pos = i % 4
            if local_pos in [1, 2]:
                electrode_lines.append(line)
            else:
                gap_lines.append(line)

        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop(boundary_lines)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # Synchronize geometry
        gmsh.model.geo.synchronize()

        # Save info
        self.electrode_lines = electrode_lines
        self.gap_lines = gap_lines
        self.boundary_lines = boundary_lines
        self.surface = surface

    def _set_physical_groups(self):
        """Set physical groups."""
        # Domain
        gmsh.model.addPhysicalGroup(2, [self.surface], 1, name="domain")

        # Electrodes - group adjacent electrode lines
        electrode_groups = []
        for i in range(self.n_elec):
            # Each electrode consists of 2 adjacent lines
            start_idx = i * 2
            if start_idx + 1 < len(self.electrode_lines):
                elec_lines = self.electrode_lines[start_idx:start_idx + 2]
                if elec_lines:  # Ensure there are lines
                    gmsh.model.addPhysicalGroup(1, elec_lines, i + 2, name=f"electrode_{i+1}")
                    electrode_groups.extend(elec_lines)

        # Gaps - all non-electrode boundary lines
        remaining_lines = [line for line in self.boundary_lines if line not in electrode_groups]
        if remaining_lines:
            gmsh.model.addPhysicalGroup(1, remaining_lines, self.n_elec + 2, name="boundary")

    def _convert_to_fenics(self, mesh_name: str, output_path: Path, save_files: bool):
        """Convert to FEniCS mesh."""
        # Use meshio for conversion
        if MESHIO_AVAILABLE:
            # Save temporary msh file
            temp_msh = output_path / f"{mesh_name}_temp.msh"
            gmsh.write(str(temp_msh))

            # Read and convert using meshio
            mesh_data = meshio.read(temp_msh)

            if save_files:
                # Save as XDMF format
                xdmf_file = output_path / f"{mesh_name}.xdmf"
                meshio.write(xdmf_file, mesh_data)

                # Create boundary info
                self._create_boundary_files(mesh_data, mesh_name, output_path)

            # Create FEniCS mesh object
            return self._create_fenics_mesh_object(mesh_data, mesh_name)

        else:
            logger.warning("meshio not available, cannot convert mesh format")
            return self._create_mesh_info(mesh_name, output_path)

    def _create_boundary_files(self, mesh_data, mesh_name: str, output_path: Path):
        """Create boundary info files."""
        # Create association table
        association_table = {}
        for i in range(self.n_elec):
            association_table[i + 2] = i + 2  # Electrode tags start from 2

        # Save association table
        import configparser
        config = configparser.ConfigParser()
        config['boundary_ids'] = {str(k): str(v) for k, v in association_table.items()}

        association_file = output_path / f"{mesh_name}_association_table.ini"
        with open(association_file, 'w') as f:
            config.write(f)

    def _create_fenics_mesh_object(self, mesh_data, mesh_name: str):
        """Create FEniCS mesh object."""

        class EnhancedEITMesh:
            """Enhanced EIT mesh object."""

            def __init__(self, mesh_data, mesh_name, generator):
                self.mesh_data = mesh_data
                self.mesh_name = mesh_name
                self.generator = generator

                # Basic attributes
                self.radius = generator.radius
                self.n_elec = generator.n_elec
                self.vertex_elec = []

                # Create association table
                self.association_table = {i + 2: i + 2 for i in range(self.n_elec)}

                # Boundary markers placeholder
                self.boundaries_mf = None

                # Mesh statistics
                self._compute_stats()

            def _compute_stats(self):
                """Compute mesh statistics."""
                points = self.mesh_data.points
                cells = self.mesh_data.cells

                self._num_vertices = len(points)
                self._num_cells = len(cells[0].data) if cells else 0

                # Compute bounding box
                self.bbox_min = np.min(points[:, :2], axis=0)
                self.bbox_max = np.max(points[:, :2], axis=0)
                self.center = np.mean(points[:, :2], axis=0)

            def coordinates(self):
                """Return coordinate array."""
                return self.mesh_data.points[:, :2]  # Return only 2D coordinates

            def num_vertices(self):
                """Return number of vertices."""
                return self._num_vertices

            def num_cells(self):
                """Return number of cells."""
                return self._num_cells

            def cells(self):
                """Return cell connectivity."""
                if self.mesh_data.cells:
                    return self.mesh_data.cells[0].data
                return np.array([])

            def get_info(self):
                """Get mesh info."""
                return {
                    'mesh_name': self.mesh_name,
                    'num_vertices': self.num_vertices(),
                    'num_cells': self.num_cells(),
                    'num_electrodes': self.n_elec,
                    'radius': self.radius,
                    'center': self.center.tolist(),
                    'bbox': [self.bbox_min.tolist(), self.bbox_max.tolist()],
                    'association_table': self.association_table
                }

        return EnhancedEITMesh(mesh_data, mesh_name, self)

    def _create_mesh_info(self, mesh_name: str, output_path: Path):
        """Create mesh info dictionary."""
        return {
            'mesh_name': mesh_name,
            'n_elec': self.n_elec,
            'radius': self.radius,
            'mesh_size': self.mesh_size,
            'output_path': str(output_path),
            'note': 'FEniCS not available, returning basic info only'
        }


def create_simple_eit_mesh(n_elec: int = 16, radius: float = 1.0,
                          mesh_size: float = 0.1, output_dir: str = None):
    """Convenience function for quickly creating EIT mesh.

    Args:
        n_elec: Number of electrodes.
        radius: Radius.
        mesh_size: Mesh size.
        output_dir: Output directory.

    Returns:
        Mesh object.
    """
    generator = SimpleEITMeshGenerator(
        n_elec=n_elec,
        radius=radius,
        mesh_size=mesh_size,
        electrode_width=2 * pi / n_elec / 8  # Electrode spans 1/8 of circumference, smaller electrodes
    )

    return generator.generate_circular_mesh(output_dir=output_dir)