"""HDF5 mesh-array bridge for MATLAB / PyEIDORS mesh tools."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import numpy as np

SRC_PATH = Path(__file__).resolve().parents[2] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.io._json import json_ready as _json_ready  # noqa: E402


MATLAB_MESH_HDF5_SCHEMA = "pyeidors-matlab-mesh-hdf5-v1"
_HDF5_SUFFIXES = {".h5", ".hdf5"}


def matlab_mesh_hdf5_path(path: str | Path) -> Path:
    """Normalize MATLAB mesh bridge array output to ``.h5`` / ``.hdf5``."""

    target = Path(path)
    if target.suffix == "":
        return target.with_suffix(".h5")
    if target.suffix.lower() not in _HDF5_SUFFIXES:
        raise ValueError(
            f"MATLAB mesh bridge arrays must use .h5 or .hdf5, got {target}"
        )
    return target


def write_matlab_mesh_hdf5(
    path: str | Path,
    *,
    nodes: Any,
    elements: Any,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write MATLAB mesh arrays as a simple HDF5 package.

    The datasets stay MATLAB-friendly: ``/nodes`` and ``/elements`` can be
    loaded via ``h5read``.  Element indices remain in the source convention
    (MATLAB 1-based for files produced by ``convert_matlab_mesh.py``).
    """

    target = matlab_mesh_hdf5_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    nodes_arr = np.asarray(nodes, dtype=np.float64)
    elements_arr = np.asarray(elements, dtype=np.int64)
    meta = {
        "artifact_schema": MATLAB_MESH_HDF5_SCHEMA,
        "artifact_format": "hdf5",
        "package_role": "matlab_mesh_bridge_arrays",
        "index_base": 1,
    }
    if metadata:
        meta.update(dict(metadata))
    with h5py.File(target, "w") as handle:
        handle.attrs["schema"] = MATLAB_MESH_HDF5_SCHEMA
        handle.attrs["metadata_json"] = json.dumps(_json_ready(meta), sort_keys=True)
        handle.create_dataset("nodes", data=nodes_arr, **_dataset_kwargs(nodes_arr))
        handle.create_dataset(
            "elements",
            data=elements_arr,
            **_dataset_kwargs(elements_arr),
        )
    return target


def load_matlab_mesh_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load HDF5 mesh arrays, or a legacy ``.npz`` read-only bridge file."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in _HDF5_SUFFIXES:
        with h5py.File(source, "r") as handle:
            return (
                np.asarray(handle["nodes"], dtype=np.float64),
                np.asarray(handle["elements"], dtype=np.int64),
            )
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            return (
                np.asarray(payload["nodes"], dtype=np.float64),
                np.asarray(payload["elements"], dtype=np.int64),
            )
    raise ValueError(
        f"Unsupported MATLAB mesh bridge suffix {suffix!r}; expected .h5 or legacy .npz."
    )


def _dataset_kwargs(arr: np.ndarray) -> dict[str, Any]:
    if arr.ndim == 0 or arr.size == 0:
        return {}
    return {"compression": "gzip", "shuffle": True, "chunks": True}


__all__ = [
    "MATLAB_MESH_HDF5_SCHEMA",
    "load_matlab_mesh_arrays",
    "matlab_mesh_hdf5_path",
    "write_matlab_mesh_hdf5",
]
