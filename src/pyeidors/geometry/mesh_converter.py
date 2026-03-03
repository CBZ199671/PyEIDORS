"""Mesh format converter."""

import numpy as np
import meshio
from pathlib import Path
from configparser import ConfigParser
from typing import Dict, Tuple, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Check FEniCS availability
try:
    from fenics import Mesh
    from dolfin import XDMFFile, MeshValueCollection
    from dolfin.cpp.mesh import MeshFunctionSizet
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS/dolfin not available")


class MeshConverter:
    """Mesh format converter."""

    def __init__(self, mesh_file: str, output_dir: str):
        self.mesh_file = mesh_file
        self.output_dir = output_dir
        self.prefix = Path(mesh_file).stem

    def convert(self) -> Tuple[Any, Any, Dict[str, int]]:
        """Convert MSH to FEniCS format."""
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS not available, cannot convert mesh")

        # Read MSH file
        msh = meshio.read(self.mesh_file)

        # Export XDMF files
        self._export_domain(msh)
        self._export_boundaries(msh)
        association_table = self._export_association_table(msh)

        # Import to FEniCS
        return self._import_fenics_mesh(association_table)

    def _export_domain(self, msh):
        """Export domain XDMF file."""
        cell_type = "triangle"

        # Extract domain cells
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            raise ValueError("Domain physical group not found")

        data = np.concatenate([cell.data for cell in cells])
        domain_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]

        # Extract cell data
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
            points=msh.points[:, :2],
            cells=domain_cells,
            cell_data=cell_data
        )

        # Export XDMF
        meshio.write(f"{self.output_dir}/{self.prefix}_domain.xdmf", domain)

    def _export_boundaries(self, msh):
        """Export boundary XDMF file."""
        cell_type = "line"

        # Extract boundary cells
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            logger.warning("Boundary physical group not found")
            return

        data = np.concatenate([cell.data for cell in cells])
        boundary_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]

        # Extract cell data
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
            points=msh.points[:, :2],
            cells=boundary_cells,
            cell_data=cell_data
        )

        # Export XDMF
        meshio.write(f"{self.output_dir}/{self.prefix}_boundaries.xdmf", boundaries)

    def _export_association_table(self, msh) -> Dict[str, int]:
        """Export association table."""
        association_table = {}

        try:
            for label, arrays in msh.cell_sets.items():
                # Find non-empty array
                for i, array in enumerate(arrays):
                    if array.size != 0 and label != "gmsh:bounding_entities":
                        if i < len(msh.cell_data["gmsh:physical"]):
                            value = msh.cell_data["gmsh:physical"][i][0]
                            association_table[label] = int(value)
                        break
        except Exception as e:
            logger.warning(f"Error processing association table: {e}")

        # Save association table
        config = ConfigParser()
        config["ASSOCIATION TABLE"] = {k: str(v) for k, v in association_table.items()}

        with open(f"{self.output_dir}/{self.prefix}_association_table.ini", 'w') as f:
            config.write(f)

        return association_table

    def _import_fenics_mesh(self, association_table: Dict[str, int]) -> Tuple[Any, Any, Dict[str, int]]:
        """Import FEniCS mesh."""
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS/dolfin not available")

        # Import domain
        mesh = Mesh()
        with XDMFFile(f"{self.output_dir}/{self.prefix}_domain.xdmf") as infile:
            infile.read(mesh)

        # Import boundaries
        boundaries_mvc = MeshValueCollection("size_t", mesh, dim=1)
        with XDMFFile(f"{self.output_dir}/{self.prefix}_boundaries.xdmf") as infile:
            infile.read(boundaries_mvc, 'boundaries')
        boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)

        return mesh, boundaries_mf, association_table