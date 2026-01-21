"""Optimized EIT Mesh Generator - Improved version based on reference implementation."""

import numpy as np
import tempfile
import time
from pathlib import Path
from math import pi, cos, sin
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Any

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
    from fenics import Mesh, MeshFunction, MeshValueCollection, XDMFFile
    from dolfin.cpp.mesh import MeshFunctionSizet
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS not available, cannot create FEniCS mesh objects")


@dataclass
class ElectrodePosition:
    """Electrode position configuration - based on reference implementation."""
    L: int  # Number of electrodes
    coverage: float = 0.5  # Electrode coverage ratio
    rotation: float = 0.0  # Rotation angle
    anticlockwise: bool = True  # Counter-clockwise direction

    def __post_init__(self):
        if not isinstance(self.L, int) or self.L <= 0:
            raise ValueError("Number of electrodes must be a positive integer")
        if not 0 < self.coverage <= 1:
            raise ValueError("Coverage must be in range (0, 1]")

    @property
    def positions(self) -> List[Tuple[float, float]]:
        """Calculate start and end angles for each electrode."""
        electrode_size = 2 * pi / self.L * self.coverage
        gap_size = 2 * pi / self.L * (1 - self.coverage)

        # First electrode center should be at positive Y-axis (π/2)
        # So first electrode start position should be π/2 - electrode_size/2
        first_electrode_center = pi / 2 + self.rotation
        first_electrode_start = first_electrode_center - electrode_size / 2

        positions = []
        for i in range(self.L):
            # Total angular space per electrode (electrode + gap)
            total_space_per_electrode = electrode_size + gap_size

            start = first_electrode_start + i * total_space_per_electrode
            end = start + electrode_size
            positions.append((start, end))

        if not self.anticlockwise:
            positions[1:] = positions[1:][::-1]

        return positions


@dataclass
class OptimizedMeshConfig:
    """Optimized mesh configuration parameters."""
    radius: float = 1.0
    refinement: int = 8
    electrode_vertices: int = 6  # Number of vertices per electrode
    gap_vertices: int = 1       # Number of vertices in gap regions

    @property
    def mesh_size(self) -> float:
        """Calculate mesh size."""
        return self.radius / (self.refinement * 2)


class OptimizedMeshGenerator:
    """Optimized mesh generator - based on reference implementation."""

    def __init__(self, config: OptimizedMeshConfig, electrodes: ElectrodePosition):
        """Initialize mesh generator.

        Args:
            config: Mesh configuration.
            electrodes: Electrode position configuration.
        """
        if not GMSH_AVAILABLE:
            raise ImportError("GMsh not available, please install gmsh: pip install gmsh")

        self.config = config
        self.electrodes = electrodes
        self.mesh_data = {}

    def generate(self, output_dir: Optional[Path] = None, mesh_name: Optional[str] = None) -> object:
        """Generate mesh.

        Args:
            output_dir: Output directory.

        Returns:
            FEniCS mesh object or mesh info.
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Generate unique mesh filename
        if mesh_name is None:
            timestamp = int(time.time() * 1e6) % 1000000
            mesh_base = f"mesh_{timestamp}"
        else:
            mesh_base = mesh_name
        mesh_file = output_dir / f"{mesh_base}.msh"

        logger.info(f"Starting EIT mesh generation: {mesh_file.stem}")

        # Initialize GMsh
        gmsh.initialize()
        gmsh.model.add("EIT_Mesh")

        try:
            # Create geometry
            self._create_geometry()

            # Set physical groups
            self._set_physical_groups()

            # Generate mesh
            self._generate_mesh()

            # Save mesh file
            gmsh.write(str(mesh_file))

            # Extract electrode vertex information
            self._extract_electrode_vertices()

        finally:
            gmsh.finalize()

        # Convert to FEniCS format
        return self._convert_to_fenics(mesh_file, output_dir)
    
    def _create_geometry(self):
        """Create geometry - based on reference implementation logic."""
        positions = self.electrodes.positions
        n_in = self.config.electrode_vertices  # Electrode vertices
        n_out = self.config.gap_vertices       # Gap vertices
        r = self.config.radius

        boundary_points = []
        electrode_ranges = []

        # Create vertices for each electrode
        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)

            # Create vertices in electrode region
            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)

            # Record electrode range
            electrode_ranges.append((start_idx, len(boundary_points) - 1))

            # Create vertices in gap region
            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi

            # Gap vertices (exclude endpoints to avoid duplicates)
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

        # Create curve loop and surface
        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])

        # Add internal control points to improve mesh quality (scale by radius to avoid overflow for small radii)
        mesh_size_center = 0.095 * r
        cp_distance = 0.1 * r
        center_points = [
            gmsh.model.occ.addPoint(x, y, 0.0, meshSize=mesh_size_center)
            for x, y in [(-cp_distance, cp_distance), (cp_distance, cp_distance),
                         (-cp_distance, -cp_distance), (cp_distance, -cp_distance)]
        ]

        # Synchronize geometry model
        gmsh.model.occ.synchronize()

        # Embed control points
        gmsh.model.mesh.embed(0, center_points, 2, surface)

        # Save geometry information
        self.mesh_data['boundary_points'] = boundary_points
        self.mesh_data['electrode_ranges'] = electrode_ranges
        self.mesh_data['lines'] = lines
        self.mesh_data['surface'] = surface
    
    def _set_physical_groups(self):
        """Set physical groups - for boundary conditions."""
        surface = self.mesh_data['surface']
        lines = self.mesh_data['lines']
        electrode_ranges = self.mesh_data['electrode_ranges']

        # Set domain physical group
        gmsh.model.addPhysicalGroup(2, [surface], 1, name="domain")

        # Set electrode physical groups
        electrode_lines = []
        for i, (start, end) in enumerate(electrode_ranges):
            lines_for_electrode = []
            for j in range(start, end):
                line_idx = j % len(lines)
                lines_for_electrode.append(lines[line_idx])

            if lines_for_electrode:
                # Electrode numbering starts from 2 (1 is domain)
                gmsh.model.addPhysicalGroup(1, lines_for_electrode, i + 2,
                                          name=f"electrode_{i+1}")
                electrode_lines.extend(lines_for_electrode)

        # Set gap (non-electrode boundary) physical group
        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            gmsh.model.addPhysicalGroup(1, gap_lines, self.electrodes.L + 2, name="gaps")

    def _generate_mesh(self):
        """Generate mesh."""
        # Set mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)

        # Generate 2D mesh
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
        """Convert to FEniCS mesh format."""
        if not FENICS_AVAILABLE:
            logger.warning("FEniCS not available, returning basic mesh info")
            return self._create_mesh_info(mesh_file.stem, output_dir)

        try:
            # Use converter to convert mesh
            converter = OptimizedMeshConverter(str(mesh_file), str(output_dir))
            mesh, boundaries_mf, association_table = converter.convert()

            # Add EIT-specific attributes
            mesh.radius = self.config.radius
            mesh.vertex_elec = self.mesh_data.get('electrode_vertices', [])
            mesh.electrodes = self.electrodes
            mesh.boundaries_mf = boundaries_mf
            mesh.association_table = association_table

            logger.info("FEniCS mesh conversion successful")
            return mesh

        except Exception as e:
            logger.error(f"FEniCS mesh conversion failed: {e}")
            return self._create_mesh_info(mesh_file.stem, output_dir)

    def _create_mesh_info(self, mesh_name: str, output_dir: Path):
        """Create mesh info dictionary (fallback when FEniCS not available)."""
        return {
            'mesh_name': mesh_name,
            'n_electrodes': self.electrodes.L,
            'radius': self.config.radius,
            'refinement': self.config.refinement,
            'output_dir': str(output_dir),
            'electrode_vertices': self.mesh_data.get('electrode_vertices', []),
            'note': 'FEniCS not available, returning basic info only'
        }


class OptimizedMeshConverter:
    """Optimized mesh format converter - based on reference implementation."""

    def __init__(self, mesh_file: str, output_dir: str):
        """Initialize converter.

        Args:
            mesh_file: GMsh mesh file path.
            output_dir: Output directory.
        """
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio not available, please install meshio: pip install meshio")

        self.mesh_file = mesh_file
        self.output_dir = output_dir
        self.prefix = Path(mesh_file).stem

    def convert(self):
        """Convert mesh format.

        Returns:
            (mesh, boundaries_mf, association_table) tuple.
        """
        # Read GMsh mesh
        msh = meshio.read(self.mesh_file)

        # Export domain
        self._export_domain(msh)

        # Export boundaries
        self._export_boundaries(msh)

        # Export association table
        association_table = self._export_association_table(msh)

        # Import as FEniCS mesh
        return self._import_fenics_mesh(association_table)

    def _export_domain(self, msh):
        """Export domain to XDMF format."""
        cell_type = "triangle"

        # Get triangle cells
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            raise ValueError("No triangle cells found")

        # Merge all triangle cell data
        data = np.concatenate([cell.data for cell in cells])
        domain_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]

        # Set cell data (subdomain markers)
        cell_data = {
            "subdomains": [
                np.concatenate([
                    msh.cell_data["gmsh:physical"][i]
                    for i, cell in enumerate(msh.cells)
                    if cell.type == cell_type
                ])
            ]
        }

        # Create domain mesh
        domain = meshio.Mesh(
            points=msh.points[:, :2],  # Use 2D coordinates only
            cells=domain_cells,
            cell_data=cell_data
        )

        # Save domain mesh
        domain_file = f"{self.output_dir}/{self.prefix}_domain.xdmf"
        meshio.write(domain_file, domain)
        logger.debug(f"Domain mesh saved: {domain_file}")

    def _export_boundaries(self, msh):
        """Export boundaries to XDMF format."""
        cell_type = "line"

        # Get line cells
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            logger.warning("No boundary line cells found")
            return

        # Merge all line cell data
        data = np.concatenate([cell.data for cell in cells])
        boundary_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]

        # Set boundary data
        cell_data = {
            "boundaries": [
                np.concatenate([
                    msh.cell_data["gmsh:physical"][i]
                    for i, cell in enumerate(msh.cells)
                    if cell.type == cell_type
                ])
            ]
        }

        # Create boundary mesh
        boundaries = meshio.Mesh(
            points=msh.points[:, :2],  # Use 2D coordinates only
            cells=boundary_cells,
            cell_data=cell_data
        )

        # Save boundary mesh
        boundaries_file = f"{self.output_dir}/{self.prefix}_boundaries.xdmf"
        meshio.write(boundaries_file, boundaries)
        logger.debug(f"Boundary mesh saved: {boundaries_file}")

    def _export_association_table(self, msh):
        """Export association table to INI file."""
        association_table = {}

        try:
            # Extract association table from GMsh physical group info
            for label, arrays in msh.cell_sets.items():
                for i, array in enumerate(arrays):
                    if array.size != 0 and label != "gmsh:bounding_entities":
                        if i < len(msh.cell_data["gmsh:physical"]):
                            value = msh.cell_data["gmsh:physical"][i][0]
                            association_table[label] = int(value)
                        break
        except Exception as e:
            logger.warning(f"Error processing association table: {e}")

        # Save association table
        from configparser import ConfigParser
        config = ConfigParser()
        config["ASSOCIATION TABLE"] = {k: str(v) for k, v in association_table.items()}

        association_file = f"{self.output_dir}/{self.prefix}_association_table.ini"
        with open(association_file, 'w') as f:
            config.write(f)

        logger.debug(f"Association table saved: {association_file}")
        return association_table

    def _import_fenics_mesh(self, association_table):
        """Import as FEniCS mesh object."""
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS not available, cannot import mesh")

        # Read domain mesh
        mesh = Mesh()
        domain_file = f"{self.output_dir}/{self.prefix}_domain.xdmf"
        with XDMFFile(domain_file) as infile:
            infile.read(mesh)

        # Read boundary markers
        boundaries_mvc = MeshValueCollection("size_t", mesh, dim=1)
        boundaries_file = f"{self.output_dir}/{self.prefix}_boundaries.xdmf"

        try:
            with XDMFFile(boundaries_file) as infile:
                infile.read(boundaries_mvc, 'boundaries')
            boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
        except Exception as e:
            logger.warning(f"Failed to read boundary markers: {e}")
            # Create empty boundary markers
            boundaries_mf = MeshFunction("size_t", mesh, 1, 0)

        logger.info(f"FEniCS mesh imported: {mesh.num_vertices()} vertices, {mesh.num_cells()} cells")

        return mesh, boundaries_mf, association_table


# Convenience functions
def create_eit_mesh(n_elec: int = 16,
                   radius: float = 1.0,
                   refinement: int = 6,
                   electrode_coverage: float = 0.5,
                   output_dir: str = None,
                   mesh_name: Optional[str] = None) -> object:
    """Convenience function: Create standard EIT mesh.

    Args:
        n_elec: Number of electrodes.
        radius: Circular domain radius.
        refinement: Mesh refinement level.
        electrode_coverage: Electrode coverage ratio.
        output_dir: Output directory.

    Returns:
        FEniCS mesh object.
    """
    # Create configuration
    mesh_config = OptimizedMeshConfig(
        radius=radius,
        refinement=refinement,
        electrode_vertices=6,
        gap_vertices=1
    )

    electrode_config = ElectrodePosition(
        L=n_elec,
        coverage=electrode_coverage,
        rotation=0.0,
        anticlockwise=True
    )

    # Generate mesh
    generator = OptimizedMeshGenerator(mesh_config, electrode_config)
    return generator.generate(output_dir=Path(output_dir) if output_dir else None, mesh_name=mesh_name)


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip('0').rstrip('.').replace('.', 'p')


def _build_cache_name(n_elec: int, radius: float, refinement: int, electrode_coverage: float) -> str:
    radius_str = _format_float(radius)
    coverage_str = _format_float(electrode_coverage)
    return f"mesh_{n_elec}e_r{radius_str}_ref{refinement}_cov{coverage_str}"


def _load_cached_mesh(mesh_dir: Path, mesh_name: str):
    if not FENICS_AVAILABLE:
        return None

    domain_file = mesh_dir / f"{mesh_name}_domain.xdmf"
    boundaries_file = mesh_dir / f"{mesh_name}_boundaries.xdmf"
    association_file = mesh_dir / f"{mesh_name}_association_table.ini"

    if not (domain_file.exists() and boundaries_file.exists() and association_file.exists()):
        return None

    mesh = Mesh()
    with XDMFFile(str(domain_file)) as infile:
        infile.read(mesh)

    boundaries_mvc = MeshValueCollection("size_t", mesh, 1)
    try:
        with XDMFFile(str(boundaries_file)) as infile:
            try:
                infile.read(boundaries_mvc, 'boundaries')
            except RuntimeError:
                infile.read(boundaries_mvc)
        boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
    except Exception:
        boundaries_mf = MeshFunction("size_t", mesh, 1, 0)

    from configparser import ConfigParser
    config = ConfigParser()
    config.read(association_file)
    association_table = {}
    for section in config.sections():
        for key, value in config[section].items():
            try:
                association_table[key] = int(value)
            except ValueError:
                association_table[key] = value

    mesh.boundaries_mf = boundaries_mf
    mesh.association_table = association_table
    return mesh


def load_or_create_mesh(mesh_dir: str = "eit_meshes",
                       mesh_name: str = None,
                       n_elec: int = 16,
                       **kwargs) -> object:
    """Load existing mesh or create new mesh.

    Args:
        mesh_dir: Mesh directory.
        mesh_name: Mesh name (if None, creates new mesh).
        n_elec: Number of electrodes.
        **kwargs: Other parameters passed to create_eit_mesh.

    Returns:
        Mesh object.
    """
    mesh_dir_path = Path(mesh_dir)
    mesh_dir_path.mkdir(parents=True, exist_ok=True)

    params = dict(kwargs)
    radius = params.pop("radius", 1.0)
    refinement = params.pop("refinement", 6)
    electrode_coverage = params.pop("electrode_coverage", 0.5)

    cache_name = mesh_name or _build_cache_name(n_elec, radius, refinement, electrode_coverage)

    cached_mesh = _load_cached_mesh(mesh_dir_path, cache_name)
    if cached_mesh is not None:
        logger.info(f"Loaded cached mesh: {cache_name}")
        return cached_mesh

    logger.info(f"Cached mesh not found, generating: {cache_name}")
    if params:
        logger.debug(f"Unused mesh parameters: {params}")

    return create_eit_mesh(
        n_elec=n_elec,
        radius=radius,
        refinement=refinement,
        electrode_coverage=electrode_coverage,
        output_dir=str(mesh_dir_path),
        mesh_name=cache_name,
    )
