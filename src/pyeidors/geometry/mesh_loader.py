"""Mesh loader for DOLFINx ``.msh`` caches."""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from ..data.structures import EITMesh
from ..femx import build_eit_mesh, estimate_radius
from ..perf.policy import LEGACY_3D_GENERATOR_REVISION
from ._helpers import (
    infer_generator_revision,
    infer_geometry_version,
    infer_mesh_family_from_mesh,
)
from .mesh3d_generator import (
    STRUCTURED_SIDECAR_VERSION,
    load_structured_sidecar,
    structured_sidecar_path_for_mesh,
)
from .process_mesh_cache import (
    build_process_mesh_cache_key,
    get_process_cached_mesh,
    put_process_cached_mesh,
)

logger = logging.getLogger(__name__)


class MeshLoader:
    """Load cached EIT meshes from ``.msh`` files."""

    def __init__(self, mesh_dir: str = "eit_meshes", gdim: int = 2):
        self.mesh_dir = Path(mesh_dir)
        self.gdim = int(gdim)
        if self.gdim not in {2, 3}:
            raise ValueError(f"gdim must be 2 or 3, got {gdim!r}")
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"Mesh directory does not exist: {mesh_dir}")

    def load_mesh(self, mesh_name: str) -> EITMesh:
        """Load mesh from ``<mesh_name>.msh``."""
        msh_file = self.mesh_dir / f"{mesh_name}.msh"
        association_file = self.mesh_dir / f"{mesh_name}_association_table.ini"

        if not msh_file.exists():
            raise FileNotFoundError(f"Mesh file does not exist: {msh_file}")

        sidecar_path = structured_sidecar_path_for_mesh(msh_file)
        process_mesh_key = build_process_mesh_cache_key(
            mesh_file=msh_file,
            association_file=association_file if association_file.exists() else None,
            sidecar_file=sidecar_path if sidecar_path.exists() else None,
            gdim=self.gdim,
            mesh_name=mesh_name,
        )
        process_mesh = get_process_cached_mesh(process_mesh_key)
        if process_mesh is not None:
            logger.info(
                "Mesh loaded from process cache %s (vertices=%d, cells=%d)",
                msh_file,
                process_mesh.num_vertices(),
                process_mesh.num_cells(),
            )
            return process_mesh

        mesh_data = gmshio.read_from_msh(
            str(msh_file),
            MPI.COMM_WORLD,
            rank=0,
            gdim=self.gdim,
        )
        association_table = self._load_association_table(association_file)
        if not association_table:
            association_table = {
                name: int(group.tag) for name, group in (mesh_data.physical_groups or {}).items()
            }

        geometry_version = infer_geometry_version(mesh_name)
        generator_revision = infer_generator_revision(mesh_name)
        if sidecar_path.exists():
            try:
                sidecar = load_structured_sidecar(sidecar_path)
                geometry_version = (
                    str(sidecar.get("geometry_version", geometry_version)).strip().lower()
                    or geometry_version
                )
                generator_revision = (
                    str(sidecar.get("generator_revision", generator_revision)).strip().lower()
                    or generator_revision
                )
            except Exception:
                pass

        sidecar_exists = sidecar_path.exists()
        eit_mesh = build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=estimate_radius(mesh_data.mesh),
            mesh_file=str(msh_file),
            geometry_version=geometry_version,
            generator_revision=generator_revision,
            structured_sidecar_file=str(sidecar_path) if sidecar_exists else None,
            structured_sidecar_version=STRUCTURED_SIDECAR_VERSION if sidecar_exists else None,
        )
        eit_mesh.mesh_family = infer_mesh_family_from_mesh(eit_mesh)
        put_process_cached_mesh(process_mesh_key, eit_mesh)
        logger.info(
            "Mesh loaded from %s (vertices=%d, cells=%d)",
            msh_file,
            eit_mesh.num_vertices(),
            eit_mesh.num_cells(),
        )
        return eit_mesh

    def _load_association_table(self, file_path: Path) -> Dict[str, int]:
        if not file_path.exists():
            return {}

        config = configparser.ConfigParser()
        config.read(file_path)

        section = None
        if "ASSOCIATION TABLE" in config:
            section = config["ASSOCIATION TABLE"]
        elif "boundary_ids" in config:
            section = config["boundary_ids"]
        if section is None:
            return {}

        association_table: Dict[str, int] = {}
        for key, value in section.items():
            try:
                association_table[str(key).strip()] = int(value)
            except ValueError:
                logger.debug("Skipping invalid association entry %s=%s", key, value)
        return association_table

    def load_numpy_mesh(self, file_path: str) -> np.ndarray:
        mesh_file = self.mesh_dir / file_path
        if not mesh_file.exists():
            raise FileNotFoundError(f"File does not exist: {mesh_file}")
        return np.load(mesh_file)

    def list_available_meshes(self) -> Dict[str, list[str]]:
        meshes: Dict[str, list[str]] = {"msh": [], "xdmf": [], "numpy": []}
        for file_path in self.mesh_dir.glob("*"):
            if file_path.suffix == ".msh":
                meshes["msh"].append(file_path.stem)
            elif file_path.suffix == ".xdmf":
                meshes["xdmf"].append(file_path.stem)
            elif file_path.suffix == ".npy":
                meshes["numpy"].append(file_path.stem)

        for key in meshes:
            meshes[key].sort()
        return meshes

    def get_default_mesh(self) -> EITMesh:
        available = self.list_available_meshes()
        if not available["msh"]:
            raise FileNotFoundError(
                f"No .msh caches found under {self.mesh_dir}. "
                "Generate one with scripts/mesh_tools/build_matlab_mesh_cache.py "
                "or pyeidors.geometry.optimized_mesh_generator.create_eit_mesh()."
            )

        candidates = sorted(
            available["msh"],
            key=lambda name: (
                0 if self._mesh_name_matches_gdim(name) else 1,
                -self._mesh_mtime(name),
                name,
            ),
        )
        last_error: Exception | None = None
        for mesh_name in candidates:
            try:
                return self.load_mesh(mesh_name)
            except Exception as exc:
                last_error = exc
                logger.debug(
                    "Skipping cached mesh candidate %s for gdim=%d: %s",
                    mesh_name,
                    self.gdim,
                    exc,
                )

        raise RuntimeError(
            f"No compatible .msh cache could be loaded from {self.mesh_dir} for gdim={self.gdim}."
        ) from last_error

    def _mesh_name_matches_gdim(self, mesh_name: str) -> bool:
        normalized = str(mesh_name).strip().lower()
        is_3d_named = normalized.startswith("mesh3d_")
        if self.gdim == 3:
            return is_3d_named
        return not is_3d_named

    def _mesh_mtime(self, mesh_name: str) -> float:
        msh_file = self.mesh_dir / f"{mesh_name}.msh"
        try:
            return msh_file.stat().st_mtime
        except FileNotFoundError:
            return float("-inf")


def create_simple_mesh_loader(mesh_dir: str = "eit_meshes", gdim: int = 2) -> MeshLoader:
    return MeshLoader(mesh_dir, gdim=gdim)
