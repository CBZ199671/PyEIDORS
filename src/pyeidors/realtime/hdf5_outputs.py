"""HDF5 output bundles for scripts and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pyeidors.io.hdf5_artifacts import read_hdf5_artifact, write_hdf5_artifact


OUTPUT_BUNDLE_SCHEMA = "pyeidors-script-output-bundle-hdf5-v1"
RECONSTRUCTION_ARRAYS_SCHEMA = "pyeidors-reconstruction-arrays-hdf5-v1"
DIAGNOSTICS_ARRAYS_SCHEMA = "pyeidors-diagnostics-arrays-hdf5-v1"
GALLERY_ARRAYS_SCHEMA = "pyeidors-gallery-arrays-hdf5-v1"
DEMO_ARRAYS_SCHEMA = "pyeidors-demo-arrays-hdf5-v1"

_HDF5_SUFFIXES = {".h5", ".hdf5"}


def hdf5_output_path(path: str | Path) -> Path:
    """Normalize script output bundle path to ``.h5`` / ``.hdf5``."""

    target = Path(path)
    if target.suffix == "":
        return target.with_suffix(".h5")
    if target.suffix.lower() in _HDF5_SUFFIXES:
        return target
    raise ValueError(f"HDF5 output bundle path must end with .h5 or .hdf5: {target}")


def write_output_bundle(
    path: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    schema: str = OUTPUT_BUNDLE_SCHEMA,
) -> Path:
    """Write one script array bundle to HDF5."""

    target = hdf5_output_path(path)
    meta = {
        "artifact_schema": schema,
        "artifact_format": "hdf5",
    }
    if metadata:
        meta.update(dict(metadata))
    return write_hdf5_artifact(target, arrays, meta, schema=schema)


def read_output_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """Read an HDF5 output bundle, with read-only legacy ``.npz`` support."""

    source = Path(path)
    if source.suffix.lower() in _HDF5_SUFFIXES:
        return dict(read_hdf5_artifact(source).arrays)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=True) as payload:
            return {str(name): np.asarray(payload[name]) for name in payload.files}
    raise ValueError(f"Unsupported output bundle suffix: {source.suffix!r}")


__all__ = [
    "DEMO_ARRAYS_SCHEMA",
    "DIAGNOSTICS_ARRAYS_SCHEMA",
    "GALLERY_ARRAYS_SCHEMA",
    "OUTPUT_BUNDLE_SCHEMA",
    "RECONSTRUCTION_ARRAYS_SCHEMA",
    "hdf5_output_path",
    "read_output_bundle",
    "write_output_bundle",
]
