"""Optional ADIOS4DOLFINx checkpoint helpers.

This module deliberately keeps ADIOS4DOLFINx optional. The project default mesh
reload path remains DOLFINx XDMF/HDF5, while this layer can emit scalable
checkpoint artifacts when the third-party package is installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ADIOS4DOLFINX_CHECKPOINT_ENV = "PYEIDORS_WRITE_ADIOS4DOLFINX_CHECKPOINT"
ADIOS4DOLFINX_ENGINE_ENV = "PYEIDORS_ADIOS4DOLFINX_ENGINE"
ADIOS4DOLFINX_DEFAULT_ENGINE = "BP4"
FACET_TAGS_NAME = "facet_tags"
CELL_TAGS_NAME = "cell_tags"


def adios4dolfinx_checkpoint_path_for_mesh(mesh_file: str | Path) -> Path:
    mesh_path = Path(mesh_file)
    return mesh_path.with_name(f"{mesh_path.stem}_adios4dolfinx.bp")


def adios4dolfinx_available() -> bool:
    return importlib.util.find_spec("adios4dolfinx") is not None


def _load_adios4dolfinx():
    return importlib.import_module("adios4dolfinx")


def _set_tag_name(tags: Any, name: str) -> str:
    resolved = str(getattr(tags, "name", "") or name)
    try:
        tags.name = resolved
    except Exception:
        pass
    return resolved


def write_adios4dolfinx_checkpoint(
    mesh_data,
    *,
    source_msh_file: str | Path,
    engine: str = ADIOS4DOLFINX_DEFAULT_ENGINE,
    store_partition_info: bool = True,
) -> str | None:
    """Write mesh and MeshTags through ADIOS4DOLFINx when available."""
    if not adios4dolfinx_available():
        logger.info(
            "ADIOS4DOLFINx checkpoint requested but adios4dolfinx is not installed"
        )
        return None

    checkpoint_file = adios4dolfinx_checkpoint_path_for_mesh(source_msh_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    adx = _load_adios4dolfinx()
    try:
        try:
            adx.write_mesh(
                checkpoint_file,
                mesh_data.mesh,
                engine=engine,
                store_partition_info=bool(store_partition_info),
            )
        except TypeError:
            adx.write_mesh(checkpoint_file, mesh_data.mesh, engine=engine)

        if getattr(mesh_data, "facet_tags", None) is not None:
            facet_name = _set_tag_name(mesh_data.facet_tags, FACET_TAGS_NAME)
            adx.write_meshtags(
                checkpoint_file,
                mesh_data.mesh,
                mesh_data.facet_tags,
                engine=engine,
                meshtag_name=facet_name,
            )
        if getattr(mesh_data, "cell_tags", None) is not None:
            cell_name = _set_tag_name(mesh_data.cell_tags, CELL_TAGS_NAME)
            adx.write_meshtags(
                checkpoint_file,
                mesh_data.mesh,
                mesh_data.cell_tags,
                engine=engine,
                meshtag_name=cell_name,
            )
        return str(checkpoint_file)
    except Exception as exc:
        logger.warning(
            "Unable to write ADIOS4DOLFINx checkpoint %s: %s",
            checkpoint_file,
            exc,
        )
        return None
