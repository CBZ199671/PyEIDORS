"""Mesh loader for DOLFINx ``.msh`` caches."""

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from dolfinx.io import gmsh as gmshio

from ..data.structures import EITMesh
from ..femx import build_eit_mesh, estimate_radius
from ..perf.policy import LEGACY_3D_GENERATOR_REVISION  # noqa: F401  re-exported for in-tree callers
from ._helpers import (
    infer_generator_revision,
    infer_geometry_version,
    infer_mesh_family_from_mesh,
    validate_mesh_data_tags,
)
from .dolfinx_mesh_cache import (
    dolfinx_cache_metadata_path_for_mesh,
    load_dolfinx_mesh_cache,
    write_dolfinx_mesh_cache,
    xdmf_cache_path_for_mesh,
    xdmf_h5_path_for_mesh,
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
from ._runtime import mpi_comm_world

logger = logging.getLogger(__name__)


class MeshLoader:
    """Load cached EIT meshes, preferring DOLFINx-native XDMF caches."""

    def __init__(self, mesh_dir: str = "eit_meshes", gdim: int = 2):
        self.mesh_dir = Path(mesh_dir)
        self.gdim = int(gdim)
        if self.gdim not in {2, 3}:
            raise ValueError(f"gdim must be 2 or 3, got {gdim!r}")
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"Mesh directory does not exist: {mesh_dir}")

    def load_mesh(self, mesh_name: str) -> EITMesh:
        """Load mesh from ``<mesh_name>.xdmf`` cache or source ``<mesh_name>.msh``."""
        msh_file = self.mesh_dir / f"{mesh_name}.msh"
        association_file = self.mesh_dir / f"{mesh_name}_association_table.ini"
        xdmf_file = xdmf_cache_path_for_mesh(msh_file)
        metadata_file = dolfinx_cache_metadata_path_for_mesh(msh_file)

        if not msh_file.exists() and not xdmf_file.exists():
            raise FileNotFoundError(
                f"Mesh file does not exist: {xdmf_file} or {msh_file}"
            )

        sidecar_path = structured_sidecar_path_for_mesh(msh_file)
        process_mesh_file = xdmf_file if xdmf_file.exists() else msh_file
        extra_files: list[Path] = []
        h5_file = xdmf_h5_path_for_mesh(msh_file)
        if h5_file.exists():
            extra_files.append(h5_file)
        if msh_file.exists() and process_mesh_file != msh_file:
            extra_files.append(msh_file)
        process_mesh_key = build_process_mesh_cache_key(
            mesh_file=process_mesh_file,
            association_file=(
                metadata_file
                if metadata_file.exists()
                else association_file
                if association_file.exists()
                else None
            ),
            sidecar_file=sidecar_path if sidecar_path.exists() else None,
            extra_files=extra_files,
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

        cache_data = load_dolfinx_mesh_cache(msh_file, gdim=self.gdim)
        if cache_data is not None:
            metadata = cache_data.metadata
            source_mesh_file = cache_data.source_msh_file or cache_data.xdmf_file
            sidecar_file = metadata.get("structured_sidecar_file")
            eit_mesh = build_eit_mesh(
                cache_data.mesh,
                facet_tags=cache_data.facet_tags,
                cell_tags=cache_data.cell_tags,
                association_table=cache_data.association_table,
                physical_groups=cache_data.physical_groups,
                radius=estimate_radius(cache_data.mesh),
                mesh_file=source_mesh_file,
                mesh_family=metadata.get("mesh_family"),
                geometry_version=metadata.get("geometry_version")
                or infer_geometry_version(mesh_name),
                generator_revision=metadata.get("generator_revision")
                or infer_generator_revision(mesh_name),
                structured_sidecar_file=sidecar_file,
                structured_sidecar_version=metadata.get("structured_sidecar_version"),
            )
            if not eit_mesh.mesh_family:
                eit_mesh.mesh_family = infer_mesh_family_from_mesh(eit_mesh)
            put_process_cached_mesh(process_mesh_key, eit_mesh)
            logger.info(
                "Mesh loaded from DOLFINx cache %s (vertices=%d, cells=%d)",
                xdmf_file,
                eit_mesh.num_vertices(),
                eit_mesh.num_cells(),
            )
            return eit_mesh

        if not msh_file.exists():
            raise FileNotFoundError(
                f"Stale or unreadable DOLFINx mesh cache {xdmf_file}; source {msh_file} is missing"
            )

        mesh_data = gmshio.read_from_msh(
            str(msh_file),
            mpi_comm_world(),
            rank=0,
            gdim=self.gdim,
        )
        association_table = validate_mesh_data_tags(mesh_data, gdim=self.gdim)
        if not association_table:
            association_table = self._load_association_table(association_file)

        geometry_version = infer_geometry_version(mesh_name)
        generator_revision = infer_generator_revision(mesh_name)
        if sidecar_path.exists():
            try:
                sidecar = load_structured_sidecar(sidecar_path)
                geometry_version = (
                    str(sidecar.get("geometry_version", geometry_version))
                    .strip()
                    .lower()
                    or geometry_version
                )
                generator_revision = (
                    str(sidecar.get("generator_revision", generator_revision))
                    .strip()
                    .lower()
                    or generator_revision
                )
            except Exception:
                pass

        sidecar_exists = sidecar_path.exists()
        structured_sidecar_file = str(sidecar_path) if sidecar_exists else None
        structured_sidecar_version = (
            STRUCTURED_SIDECAR_VERSION if sidecar_exists else None
        )
        write_dolfinx_mesh_cache(
            mesh_data,
            source_msh_file=msh_file,
            association_table=association_table,
            gdim=self.gdim,
            geometry_version=geometry_version,
            generator_revision=generator_revision,
            structured_sidecar_file=structured_sidecar_file,
            structured_sidecar_version=structured_sidecar_version,
        )
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
            structured_sidecar_file=structured_sidecar_file,
            structured_sidecar_version=structured_sidecar_version,
        )
        eit_mesh.mesh_family = infer_mesh_family_from_mesh(eit_mesh)
        put_process_cached_mesh(process_mesh_key, eit_mesh)
        if xdmf_file.exists():
            h5_file = xdmf_h5_path_for_mesh(msh_file)
            extra_files = [h5_file] if h5_file.exists() else []
            if msh_file.exists():
                extra_files.append(msh_file)
            xdmf_process_key = build_process_mesh_cache_key(
                mesh_file=xdmf_file,
                association_file=metadata_file if metadata_file.exists() else None,
                sidecar_file=sidecar_path if sidecar_path.exists() else None,
                extra_files=extra_files,
                gdim=self.gdim,
                mesh_name=mesh_name,
            )
            put_process_cached_mesh(xdmf_process_key, eit_mesh)
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
        meshes: Dict[str, list[str]] = {
            "msh": [],
            "xdmf": [],
            "adios2": [],
            "numpy": [],
        }
        for file_path in self.mesh_dir.glob("*"):
            if file_path.suffix == ".msh":
                meshes["msh"].append(file_path.stem)
            elif file_path.suffix == ".xdmf":
                meshes["xdmf"].append(file_path.stem)
            elif file_path.suffix == ".bp":
                meshes["adios2"].append(file_path.stem)
            elif file_path.suffix == ".npy":
                meshes["numpy"].append(file_path.stem)

        for key in meshes:
            meshes[key].sort()
        return meshes

    def get_default_mesh(self) -> EITMesh:
        available = self.list_available_meshes()
        mesh_names = sorted(set(available["xdmf"]) | set(available["msh"]))
        if not mesh_names:
            raise FileNotFoundError(
                f"No DOLFINx .xdmf or source .msh caches found under {self.mesh_dir}. "
                "Generate one with scripts/mesh_tools/build_matlab_mesh_cache.py "
                "or pyeidors.geometry.optimized_mesh_generator.create_eit_mesh()."
            )

        candidates = sorted(
            mesh_names,
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
            f"No compatible DOLFINx mesh cache could be loaded from {self.mesh_dir} for gdim={self.gdim}."
        ) from last_error

    def _mesh_name_matches_gdim(self, mesh_name: str) -> bool:
        normalized = str(mesh_name).strip().lower()
        is_3d_named = normalized.startswith("mesh3d_")
        if self.gdim == 3:
            return is_3d_named
        return not is_3d_named

    def _mesh_mtime(self, mesh_name: str) -> float:
        msh_file = self.mesh_dir / f"{mesh_name}.msh"
        xdmf_file = xdmf_cache_path_for_mesh(msh_file)
        mtimes: list[float] = []
        for path in (xdmf_file, msh_file):
            try:
                mtimes.append(path.stat().st_mtime)
            except FileNotFoundError:
                pass
        return max(mtimes) if mtimes else float("-inf")


def create_simple_mesh_loader(
    mesh_dir: str = "eit_meshes", gdim: int = 2
) -> MeshLoader:
    return MeshLoader(mesh_dir, gdim=gdim)
