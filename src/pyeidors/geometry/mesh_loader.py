"""Mesh Loader - Supports loading multiple mesh formats."""

import numpy as np
import configparser
from pathlib import Path
import h5py
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Check FEniCS availability
try:
    from fenics import Mesh, MeshFunction, HDF5File, XDMFFile
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS not available, cannot load FEniCS mesh formats")


class MeshLoader:
    """Mesh Loader - Supports loading existing mesh files."""

    def __init__(self, mesh_dir: str = "eit_meshes"):
        """Initialize mesh loader.

        Args:
            mesh_dir: Mesh file directory.
        """
        self.mesh_dir = Path(mesh_dir)
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"Mesh directory does not exist: {mesh_dir}")

    def load_fenics_mesh(self, mesh_name: str = "mesh_506999") -> object:
        """Load FEniCS format mesh.

        Args:
            mesh_name: Mesh name (without extension).

        Returns:
            Mesh object containing mesh, boundary info, and association table.
        """
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS not available, cannot load mesh")

        # Build file paths
        domain_file = self.mesh_dir / f"{mesh_name}_domain.h5"
        boundaries_file = self.mesh_dir / f"{mesh_name}_boundaries.h5"
        association_file = self.mesh_dir / f"{mesh_name}_association_table.ini"

        # Check if files exist
        for file_path in [domain_file, boundaries_file, association_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Mesh file does not exist: {file_path}")

        logger.info(f"Loading mesh: {mesh_name}")

        # Load main mesh
        mesh = Mesh()
        with HDF5File(mesh.mpi_comm(), str(domain_file), "r") as hdf:
            hdf.read(mesh, "/mesh", False)

        # Load boundary markers
        boundaries_mf = MeshFunction("size_t", mesh, 1)
        with HDF5File(mesh.mpi_comm(), str(boundaries_file), "r") as hdf:
            hdf.read(boundaries_mf, "/boundaries")

        # Load association table
        association_table = self._load_association_table(association_file)

        # Create enhanced mesh object
        enhanced_mesh = self._create_enhanced_mesh(mesh, boundaries_mf, association_table)

        logger.info(f"Mesh loaded - vertices: {mesh.num_vertices()}, cells: {mesh.num_cells()}")

        return enhanced_mesh

    def _load_association_table(self, file_path: Path) -> Dict[Any, int]:
        """Load association table."""
        config = configparser.ConfigParser()
        config.read(file_path)

        association_table: Dict[Any, int] = {}
        section = None
        # Compatible with old [boundary_ids] and new [ASSOCIATION TABLE] formats
        if 'ASSOCIATION TABLE' in config:
            section = config['ASSOCIATION TABLE']
        elif 'boundary_ids' in config:
            section = config['boundary_ids']

        if section is None:
            logger.warning(f"Association table file missing expected section: {file_path}")
            return association_table

        for key, value in section.items():
            try:
                value_int = int(value)
            except ValueError:
                logger.debug(f"Skipping unparseable association table entry {key}={value}")
                continue

            clean_key = key.strip()
            # Convert key to int if it's a pure number, for backward compatibility
            association_table[int(clean_key) if clean_key.isdigit() else clean_key] = value_int

        return association_table

    def _create_enhanced_mesh(self, mesh, boundaries_mf, association_table) -> object:
        """Create enhanced mesh object."""

        class EnhancedMesh:
            """Enhanced mesh object with all necessary EIT attributes."""

            def __init__(self, mesh, boundaries_mf, association_table):
                # Copy all original mesh attributes and methods
                for attr in dir(mesh):
                    if not attr.startswith('_'):
                        try:
                            setattr(self, attr, getattr(mesh, attr))
                        except AttributeError:
                            pass

                # EIT-specific attributes
                self.boundaries_mf = boundaries_mf
                self.association_table = association_table

                # Default geometry parameters
                self.radius = 1.0
                self.vertex_elec = []

                # Infer electrode count
                self.electrode_tags = self._extract_electrode_tags()
                self.n_electrodes = len(self.electrode_tags)

                # Compute mesh statistics
                self._compute_mesh_stats()

            def _compute_mesh_stats(self):
                """Compute mesh statistics."""
                coords = mesh.coordinates()
                self.center = np.mean(coords, axis=0)
                self.bbox_min = np.min(coords, axis=0)
                self.bbox_max = np.max(coords, axis=0)

                # Estimate radius
                distances = np.linalg.norm(coords - self.center, axis=1)
                self.radius = np.max(distances)

            def get_info(self) -> Dict[str, Any]:
                """Get mesh information."""
                return {
                    'num_vertices': mesh.num_vertices(),
                    'num_cells': mesh.num_cells(),
                    'num_electrodes': self.n_electrodes,
                    'radius': self.radius,
                    'center': self.center.tolist(),
                    'bbox': [self.bbox_min.tolist(), self.bbox_max.tolist()],
                    'association_table': self.association_table
                }

            def _extract_electrode_tags(self):
                """Extract electrode boundary tags from association table."""
                tags = []
                for key, val in association_table.items():
                    try:
                        tag_val = int(val)
                    except (TypeError, ValueError):
                        continue

                    if isinstance(key, str) and key.lower().startswith("electrode"):
                        tags.append(tag_val)
                    elif isinstance(key, (int, np.integer)) and key >= 2:
                        tags.append(tag_val)
                return sorted(set(tags))

        return EnhancedMesh(mesh, boundaries_mf, association_table)

    def load_numpy_mesh(self, file_path: str) -> np.ndarray:
        """Load numpy format mesh data.

        Args:
            file_path: Numpy file path.

        Returns:
            Mesh data array.
        """
        mesh_file = self.mesh_dir / file_path
        if not mesh_file.exists():
            raise FileNotFoundError(f"File does not exist: {mesh_file}")

        return np.load(mesh_file)

    def list_available_meshes(self) -> Dict[str, list]:
        """List available mesh files."""
        meshes = {
            'fenics_h5': [],
            'xdmf': [],
            'msh': [],
            'numpy': []
        }

        for file_path in self.mesh_dir.glob("*"):
            if file_path.suffix == '.h5':
                base_name = file_path.stem
                if base_name.endswith('_domain'):
                    meshes['fenics_h5'].append(base_name[:-7])  # Remove '_domain'
            elif file_path.suffix == '.xdmf':
                meshes['xdmf'].append(file_path.stem)
            elif file_path.suffix == '.msh':
                meshes['msh'].append(file_path.stem)
            elif file_path.suffix == '.npy':
                meshes['numpy'].append(file_path.stem)

        return meshes

    def get_default_mesh(self) -> object:
        """Get default mesh (if available)."""
        available = self.list_available_meshes()

        # Prefer FEniCS H5 format
        if available['fenics_h5']:
            mesh_name = available['fenics_h5'][0]
            logger.info(f"Using default FEniCS mesh: {mesh_name}")
            return self.load_fenics_mesh(mesh_name)

        raise FileNotFoundError("No available default mesh files found")


def create_simple_mesh_loader(mesh_dir: str = "eit_meshes") -> MeshLoader:
    """Create simple mesh loader instance."""
    return MeshLoader(mesh_dir)
