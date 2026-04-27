"""DOLFINx-native mesh cache helpers.

The readable cache is XDMF/HDF5 because DOLFINx exposes both mesh and
MeshTags read APIs for it. ADIOS2/VTX output is optional and write-only in the
official DOLFINx Python API, so it is treated as a side artifact rather than a
reload source.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI

from ..cache.disk_artifacts import (
    build_disk_artifact_manifest,
    ensure_disk_artifact_metadata,
    file_sha256,
)
from ..cache.keys import hash_array
from ._helpers import (
    association_from_mesh_data,
    physical_group_dimensions_from_mesh_data,
    validate_mesh_data_tags,
)
from .adios4dolfinx_checkpoint import (
    ADIOS4DOLFINX_CHECKPOINT_ENV,
    ADIOS4DOLFINX_DEFAULT_ENGINE,
    ADIOS4DOLFINX_ENGINE_ENV,
    adios4dolfinx_available,
    write_adios4dolfinx_checkpoint,
)

logger = logging.getLogger(__name__)

DOLFINX_MESH_CACHE_VERSION = 1
MESH_NAME = "mesh"
FACET_TAGS_NAME = "facet_tags"
CELL_TAGS_NAME = "cell_tags"
ADIOS2_CACHE_ENV = "PYEIDORS_WRITE_ADIOS2_MESH_CACHE"


@dataclass(frozen=True)
class DolfinxMeshCacheData:
    """Loaded mesh and tag payload from a DOLFINx-native cache."""

    mesh: Any
    facet_tags: Any | None
    cell_tags: Any | None
    association_table: dict[str, int]
    physical_groups: dict[str, Any]
    metadata: dict[str, Any]
    xdmf_file: str
    source_msh_file: str | None


def xdmf_cache_path_for_mesh(mesh_file: str | Path) -> Path:
    """Return the XDMF cache path paired with a source ``.msh`` file."""
    return Path(mesh_file).with_suffix(".xdmf")


def xdmf_h5_path_for_mesh(mesh_file: str | Path) -> Path:
    """Return the HDF5 sidecar path written by DOLFINx XDMFFile."""
    return xdmf_cache_path_for_mesh(mesh_file).with_suffix(".h5")


def dolfinx_cache_metadata_path_for_mesh(mesh_file: str | Path) -> Path:
    """Return the metadata JSON path for a DOLFINx-native mesh cache."""
    mesh_path = Path(mesh_file)
    return mesh_path.with_name(f"{mesh_path.stem}_dolfinx_cache.json")


def adios2_cache_path_for_mesh(mesh_file: str | Path) -> Path:
    """Return the optional ADIOS2/VTX cache path for a source mesh."""
    return Path(mesh_file).with_suffix(".bp")


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_paths_for_mesh(mesh_file: str | Path) -> tuple[Path, Path, Path]:
    mesh_path = Path(mesh_file)
    xdmf_file = (
        mesh_path
        if mesh_path.suffix == ".xdmf"
        else xdmf_cache_path_for_mesh(mesh_path)
    )
    metadata_file = dolfinx_cache_metadata_path_for_mesh(mesh_path)
    return xdmf_file, xdmf_file.with_suffix(".h5"), metadata_file


def _source_signature(
    path: str | Path | None, *, include_hash: bool = False
) -> dict[str, Any] | None:
    if path is None:
        return None
    mesh_path = Path(path)
    try:
        stat = mesh_path.stat()
    except OSError:
        return None
    signature = {
        "path": str(mesh_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        signature["sha256"] = file_sha256(mesh_path)
    return signature


def _mesh_content_signature(mesh: Any) -> dict[str, Any] | None:
    try:
        geometry = np.asarray(mesh.geometry.x)
        tdim = int(mesh.topology.dim)
        mesh.topology.create_connectivity(tdim, 0)
        connectivity = mesh.topology.connectivity(tdim, 0)
        topology = np.asarray(connectivity.array)
    except Exception:
        return None
    return {
        "tdim": tdim,
        "geometry_hash": hash_array(geometry),
        "topology_hash": hash_array(topology),
    }


def _physical_groups_for_manifest(
    physical_groups: dict[str, Any],
) -> dict[str, dict[str, int | None]]:
    payload: dict[str, dict[str, int | None]] = {}
    for name, group in physical_groups.items():
        tag = getattr(group, "tag", None)
        dim = getattr(group, "dim", None)
        if tag is None:
            continue
        payload[str(name)] = {
            "tag": int(tag),
            "dim": None if dim is None else int(dim),
        }
    return payload


def _dolfinx_manifest_key_payload(
    metadata: dict[str, Any],
    *,
    mesh_content_signature: dict[str, Any] | None,
    association_table: dict[str, int],
    physical_groups: dict[str, Any],
) -> dict[str, Any]:
    physical_payload = metadata.get("physical_groups")
    if not isinstance(physical_payload, dict):
        physical_payload = _physical_groups_for_manifest(physical_groups)
    return {
        "format": metadata.get("format", "dolfinx-xdmf-hdf5"),
        "gdim": int(metadata.get("gdim", 0)),
        "mesh_content_signature": (
            metadata.get("mesh_content_signature") or mesh_content_signature
        ),
        "source_msh_signature": metadata.get("source_msh_signature"),
        "association_table": {
            str(name): int(tag) for name, tag in association_table.items()
        },
        "physical_groups": physical_payload,
        "mesh_family": metadata.get("mesh_family"),
        "geometry_version": metadata.get("geometry_version"),
        "generator_revision": metadata.get("generator_revision"),
        "structured_sidecar_signature": metadata.get("structured_sidecar_signature"),
        "structured_sidecar_version": metadata.get("structured_sidecar_version"),
    }


def _ensure_dolfinx_cache_artifact_metadata(
    metadata: dict[str, Any],
    *,
    mesh: Any,
    xdmf_file: Path,
    metadata_file: Path,
    association_table: dict[str, int],
    physical_groups: dict[str, Any],
) -> dict[str, Any]:
    mesh_content_signature = _mesh_content_signature(mesh)
    if metadata.get("mesh_content_signature") is None and mesh_content_signature:
        metadata = dict(metadata)
        metadata["mesh_content_signature"] = mesh_content_signature
    key_payload = _dolfinx_manifest_key_payload(
        metadata,
        mesh_content_signature=mesh_content_signature,
        association_table=association_table,
        physical_groups=physical_groups,
    )
    return ensure_disk_artifact_metadata(
        metadata,
        "dolfinx-mesh-cache",
        key_payload,
        files={
            "xdmf": xdmf_file,
            "hdf5": Path(metadata.get("hdf5_file") or xdmf_file.with_suffix(".h5")),
            "metadata": metadata_file,
            "adios2": metadata.get("adios2_file"),
            "adios4dolfinx": metadata.get("adios4dolfinx_file"),
        },
        manifest_metadata={"cache_format": metadata.get("format", "dolfinx-xdmf-hdf5")},
        subkey_payloads={"mesh_provenance": key_payload},
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _read_metadata(metadata_file: Path) -> dict[str, Any] | None:
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(
            "Unable to read DOLFINx mesh cache metadata %s: %s", metadata_file, exc
        )
        return None


def _physical_group_metadata(
    mesh_data, association_table: dict[str, int]
) -> dict[str, dict[str, int | None]]:
    dimensions = physical_group_dimensions_from_mesh_data(mesh_data)
    return {
        str(name): {
            "tag": int(tag),
            "dim": None if str(name) not in dimensions else int(dimensions[str(name)]),
        }
        for name, tag in association_table.items()
    }


def _metadata_physical_groups(metadata: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name, payload in (metadata.get("physical_groups") or {}).items():
        if not isinstance(payload, dict) or "tag" not in payload:
            continue
        dim = payload.get("dim")
        groups[str(name)] = SimpleNamespace(
            tag=int(payload["tag"]),
            dim=None if dim is None else int(dim),
        )
    return groups


def dolfinx_cache_is_fresh(mesh_file: str | Path) -> bool:
    """Return True when the paired XDMF/HDF5 cache exists and matches source metadata."""
    mesh_path = Path(mesh_file)
    xdmf_file, h5_file, metadata_file = _cache_paths_for_mesh(mesh_path)
    if not xdmf_file.exists() or not metadata_file.exists() or not h5_file.exists():
        return False

    metadata = _read_metadata(metadata_file)
    if metadata is None:
        return False
    if int(metadata.get("version", -1)) != DOLFINX_MESH_CACHE_VERSION:
        return False

    source_payload = metadata.get("source_msh_file")
    source_msh = Path(source_payload) if source_payload else mesh_path
    if mesh_path.suffix == ".xdmf" and not source_payload:
        return True

    current_source = _source_signature(source_msh)
    recorded_source = metadata.get("source_msh_signature")
    if current_source is not None and isinstance(recorded_source, dict):
        same_size = int(recorded_source.get("size", -1)) == current_source["size"]
        same_mtime = (
            int(recorded_source.get("mtime_ns", -1)) == current_source["mtime_ns"]
        )
        if same_size and same_mtime:
            return True
        if same_size and recorded_source.get("sha256"):
            return recorded_source["sha256"] == file_sha256(source_msh)
        return False

    if current_source is not None:
        source_mtime = source_msh.stat().st_mtime
        return (
            xdmf_file.stat().st_mtime >= source_mtime
            and metadata_file.stat().st_mtime >= source_mtime
        )
    return True


def _read_meshtags_optional(
    xdmf: XDMFFile, mesh: Any, *, name: str, dim: int
) -> Any | None:
    try:
        tdim = int(mesh.topology.dim)
        if dim < tdim:
            mesh.topology.create_entities(dim)
            mesh.topology.create_connectivity(dim, tdim)
            mesh.topology.create_connectivity(tdim, dim)
        tags = xdmf.read_meshtags(mesh, name=name)
        try:
            tags.name = name
        except Exception:
            pass
        return tags
    except Exception as exc:
        logger.debug("Unable to read XDMF MeshTags %s: %s", name, exc)
        return None


def load_dolfinx_mesh_cache(
    mesh_file: str | Path,
    *,
    gdim: int,
    required_names: list[str] | tuple[str, ...] = (),
    required_facet_names: list[str] | tuple[str, ...] = (),
) -> DolfinxMeshCacheData | None:
    """Load a paired XDMF/HDF5 mesh cache, returning None if unavailable or stale."""
    mesh_path = Path(mesh_file)
    if not dolfinx_cache_is_fresh(mesh_path):
        return None

    xdmf_file, _, metadata_file = _cache_paths_for_mesh(mesh_path)
    metadata = _read_metadata(metadata_file)
    if metadata is None or int(metadata.get("gdim", gdim)) != int(gdim):
        return None

    try:
        with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
            mesh = xdmf.read_mesh(name=str(metadata.get("mesh_name", MESH_NAME)))
            tdim = int(mesh.topology.dim)
            facet_tags = _read_meshtags_optional(
                xdmf,
                mesh,
                name=str(metadata.get("facet_tags_name", FACET_TAGS_NAME)),
                dim=tdim - 1,
            )
            cell_tags = _read_meshtags_optional(
                xdmf,
                mesh,
                name=str(metadata.get("cell_tags_name", CELL_TAGS_NAME)),
                dim=tdim,
            )
    except Exception as exc:
        logger.warning(
            "Skipping DOLFINx mesh cache %s due to load failure: %s", xdmf_file, exc
        )
        return None

    physical_groups = _metadata_physical_groups(metadata)
    mesh_data = SimpleNamespace(physical_groups=physical_groups, facet_tags=facet_tags)
    association_table = validate_mesh_data_tags(
        mesh_data,
        gdim=int(gdim),
        required_names=required_names,
        required_facet_names=required_facet_names,
    )
    if not association_table:
        association_table = {
            str(name): int(value)
            for name, value in (metadata.get("association_table") or {}).items()
        }

    source_payload = metadata.get("source_msh_file")
    source_path = Path(source_payload) if source_payload else mesh_path
    source_msh_file = (
        str(source_path)
        if source_path.suffix == ".msh" and source_path.exists()
        else None
    )
    metadata = _ensure_dolfinx_cache_artifact_metadata(
        metadata,
        mesh=mesh,
        xdmf_file=xdmf_file,
        metadata_file=metadata_file,
        association_table=association_table,
        physical_groups=physical_groups,
    )
    return DolfinxMeshCacheData(
        mesh=mesh,
        facet_tags=facet_tags,
        cell_tags=cell_tags,
        association_table=association_table,
        physical_groups=physical_groups,
        metadata=metadata,
        xdmf_file=str(xdmf_file),
        source_msh_file=source_msh_file,
    )


def _clear_existing_adios2_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _write_adios2_mesh_cache(mesh: Any, path: Path) -> bool:
    try:
        from dolfinx.cpp.io import VTXMeshPolicy
        from dolfinx.io import VTXWriter

        _clear_existing_adios2_path(path)
        try:
            writer = VTXWriter(
                MPI.COMM_WORLD,
                path,
                mesh,
                engine="BPFile",
                mesh_policy=VTXMeshPolicy.reuse,
            )
        except TypeError:
            writer = VTXWriter(MPI.COMM_WORLD, path, mesh, engine="BPFile")
        with writer:
            writer.write(0.0)
        return True
    except Exception as exc:
        logger.debug("Unable to write optional ADIOS2/VTX mesh cache %s: %s", path, exc)
        return False


def write_dolfinx_mesh_cache(
    mesh_data,
    *,
    source_msh_file: str | Path | None,
    cache_base_file: str | Path | None = None,
    association_table: dict[str, int] | None = None,
    gdim: int,
    mesh_family: str | None = None,
    geometry_version: str | None = None,
    generator_revision: str | None = None,
    structured_sidecar_file: str | Path | None = None,
    structured_sidecar_version: str | None = None,
    write_adios2: bool | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    """Write the DOLFINx-native XDMF/HDF5 cache for a Gmsh-imported mesh."""
    if source_msh_file is None and cache_base_file is None:
        raise ValueError("source_msh_file or cache_base_file is required")
    source_msh = None if source_msh_file is None else Path(source_msh_file)
    cache_base = (
        Path(cache_base_file) if cache_base_file is not None else Path(source_msh_file)
    )
    xdmf_file = xdmf_cache_path_for_mesh(cache_base)
    metadata_file = dolfinx_cache_metadata_path_for_mesh(cache_base)
    adios2_file = adios2_cache_path_for_mesh(cache_base)

    try:
        mesh = mesh_data.mesh
        mesh.name = MESH_NAME
        if getattr(mesh_data, "facet_tags", None) is not None:
            mesh_data.facet_tags.name = FACET_TAGS_NAME
        if getattr(mesh_data, "cell_tags", None) is not None:
            mesh_data.cell_tags.name = CELL_TAGS_NAME

        xdmf_file.parent.mkdir(parents=True, exist_ok=True)
        with XDMFFile(MPI.COMM_WORLD, xdmf_file, "w") as xdmf:
            xdmf.write_mesh(mesh)
            if getattr(mesh_data, "facet_tags", None) is not None:
                xdmf.write_meshtags(mesh_data.facet_tags, mesh.geometry)
            if getattr(mesh_data, "cell_tags", None) is not None:
                xdmf.write_meshtags(mesh_data.cell_tags, mesh.geometry)

        association = {
            str(name): int(tag)
            for name, tag in (
                association_table or association_from_mesh_data(mesh_data)
            ).items()
        }
        adios2_written = False
        if bool(write_adios2) or (
            write_adios2 is None and _truthy_env(ADIOS2_CACHE_ENV)
        ):
            adios2_written = _write_adios2_mesh_cache(mesh, adios2_file)
        adios4dolfinx_file = None
        adios4dolfinx_engine = (
            os.environ.get(
                ADIOS4DOLFINX_ENGINE_ENV,
                ADIOS4DOLFINX_DEFAULT_ENGINE,
            ).strip()
            or ADIOS4DOLFINX_DEFAULT_ENGINE
        )
        if _truthy_env(ADIOS4DOLFINX_CHECKPOINT_ENV):
            adios4dolfinx_file = write_adios4dolfinx_checkpoint(
                mesh_data,
                source_msh_file=cache_base,
                engine=adios4dolfinx_engine,
            )

        physical_groups = _physical_group_metadata(mesh_data, association)
        source_signature = _source_signature(source_msh, include_hash=True)
        structured_sidecar_signature = _source_signature(
            structured_sidecar_file,
            include_hash=True,
        )
        mesh_content_signature = _mesh_content_signature(mesh)
        metadata = {
            "version": DOLFINX_MESH_CACHE_VERSION,
            "format": "dolfinx-xdmf-hdf5",
            "gdim": int(gdim),
            "mesh_content_signature": mesh_content_signature,
            "mesh_name": MESH_NAME,
            "facet_tags_name": FACET_TAGS_NAME,
            "cell_tags_name": CELL_TAGS_NAME,
            "source_msh_file": None if source_msh is None else str(source_msh),
            "source_msh_signature": source_signature,
            "xdmf_file": str(xdmf_file),
            "hdf5_file": str(xdmf_file.with_suffix(".h5")),
            "adios2_file": str(adios2_file) if adios2_written else None,
            "adios4dolfinx_available": adios4dolfinx_available(),
            "adios4dolfinx_file": adios4dolfinx_file,
            "adios4dolfinx_engine": adios4dolfinx_engine,
            "association_table": association,
            "physical_groups": physical_groups,
            "mesh_family": mesh_family,
            "geometry_version": geometry_version,
            "generator_revision": generator_revision,
            "structured_sidecar_file": (
                None
                if structured_sidecar_file is None
                else str(structured_sidecar_file)
            ),
            "structured_sidecar_version": structured_sidecar_version,
            "structured_sidecar_signature": structured_sidecar_signature,
        }
        key_payload = {
            "format": metadata["format"],
            "gdim": int(gdim),
            "mesh_content_signature": mesh_content_signature,
            "source_msh_signature": source_signature,
            "association_table": association,
            "physical_groups": physical_groups,
            "mesh_family": mesh_family,
            "geometry_version": geometry_version,
            "generator_revision": generator_revision,
            "structured_sidecar_signature": structured_sidecar_signature,
            "structured_sidecar_version": structured_sidecar_version,
        }
        manifest = build_disk_artifact_manifest(
            "dolfinx-mesh-cache",
            key_payload,
            files={
                "xdmf": xdmf_file,
                "hdf5": xdmf_file.with_suffix(".h5"),
                "metadata": metadata_file,
                "adios2": adios2_file if adios2_written else None,
                "adios4dolfinx": adios4dolfinx_file,
            },
            metadata={"cache_format": metadata["format"]},
            subkey_payloads={"mesh_provenance": key_payload},
        )
        metadata["artifact_key"] = manifest.artifact_key
        metadata["artifact_manifest"] = manifest.to_metadata()
        if extra_metadata:
            metadata.update(_json_safe(extra_metadata))

        if MPI.COMM_WORLD.rank == 0:
            tmp_file = metadata_file.with_suffix(metadata_file.suffix + ".tmp")
            tmp_file.write_text(
                json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_file.replace(metadata_file)
        MPI.COMM_WORLD.barrier()
        return True
    except Exception as exc:
        logger.warning("Unable to write DOLFINx mesh cache for %s: %s", cache_base, exc)
        return False
