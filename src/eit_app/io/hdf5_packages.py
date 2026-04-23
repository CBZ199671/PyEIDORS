"""HDF5 package writers for GUI-generated numeric payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


DATASET_MESH_SCHEMA = "pyeidors-dataset-mesh-hdf5-v1"
DATASET_SAMPLE_SCHEMA = "pyeidors-dataset-sample-hdf5-v1"
SIMULATION_RESULTS_SCHEMA = "pyeidors-simulation-results-hdf5-v1"

_HDF5_SUFFIXES = {".h5", ".hdf5"}


def normalize_hdf5_package_path(path: str | Path) -> Path:
    """Return a writable HDF5 package path, appending ``.h5`` if omitted."""

    target = Path(path)
    if target.suffix == "":
        return target.with_suffix(".h5")
    if target.suffix.lower() not in _HDF5_SUFFIXES:
        raise ValueError(f"HDF5 package path must end with .h5 or .hdf5, got {target}")
    return target


def write_dataset_mesh_info_package(
    output_dir: str | Path,
    *,
    node_coords: Any,
    cell_connectivity: Any,
    n_electrodes: int,
    homogeneous_voltages: Any,
    forward_model_config: Mapping[str, Any],
    total_electrodes: int,
) -> Path:
    """Write dataset-wide mesh and homogeneous-reference arrays."""

    out_dir = Path(output_dir)
    arrays = {
        "node_coords": np.asarray(node_coords),
        "cell_connectivity": np.asarray(cell_connectivity),
        "homogeneous_voltages": np.asarray(homogeneous_voltages),
        "n_electrodes": np.asarray(n_electrodes, dtype=np.int64),
        "total_electrodes": np.asarray(total_electrodes, dtype=np.int64),
    }
    metadata = {
        "artifact_schema": DATASET_MESH_SCHEMA,
        "artifact_format": "hdf5",
        "package_role": "dataset_mesh_info",
        "forward_model_config": dict(forward_model_config),
    }
    return _write_hdf5_artifact(
        out_dir / "mesh_info.h5",
        arrays,
        metadata,
        schema=DATASET_MESH_SCHEMA,
    )


def write_dataset_sample_package(
    output_dir: str | Path,
    sample_index: int,
    *,
    ground_truth: Any,
    boundary_voltages: Any,
    background_conductivity: float,
    n_inhomogeneities: int,
) -> Path:
    """Write one dataset sample as an HDF5 package."""

    out_dir = Path(output_dir)
    arrays = {
        "ground_truth": np.asarray(ground_truth),
        "boundary_voltages": np.asarray(boundary_voltages),
        "background_conductivity": np.asarray(
            background_conductivity, dtype=np.float64
        ),
        "n_inhomogeneities": np.asarray(n_inhomogeneities, dtype=np.int64),
    }
    metadata = {
        "artifact_schema": DATASET_SAMPLE_SCHEMA,
        "artifact_format": "hdf5",
        "package_role": "dataset_sample",
        "sample_index": int(sample_index),
    }
    return _write_hdf5_artifact(
        out_dir / f"sample_{sample_index:06d}.h5",
        arrays,
        metadata,
        schema=DATASET_SAMPLE_SCHEMA,
    )


def write_simulation_results_package(
    path: str | Path,
    *,
    ground_truth: Any,
    boundary_voltages: Any,
    homogeneous_voltages: Any,
    node_coords: Any,
    cell_connectivity: Any,
) -> Path:
    """Write interactive simulation results as one HDF5 package."""

    target = normalize_hdf5_package_path(path)
    arrays = {
        "ground_truth": np.asarray(ground_truth),
        "boundary_voltages": np.asarray(boundary_voltages),
        "homogeneous_voltages": np.asarray(homogeneous_voltages),
        "node_coords": np.asarray(node_coords),
        "cell_connectivity": np.asarray(cell_connectivity),
    }
    metadata = {
        "artifact_schema": SIMULATION_RESULTS_SCHEMA,
        "artifact_format": "hdf5",
        "package_role": "simulation_results",
    }
    return _write_hdf5_artifact(
        target,
        arrays,
        metadata,
        schema=SIMULATION_RESULTS_SCHEMA,
    )


def _write_hdf5_artifact(
    path: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    schema: str,
) -> Path:
    # Keep pyeidors runtime imports out of GUI startup; HDF5 is needed only
    # when the user explicitly exports or generates a dataset.
    from pyeidors.io.hdf5_artifacts import write_hdf5_artifact

    return write_hdf5_artifact(path, arrays, metadata, schema=schema)


__all__ = [
    "DATASET_MESH_SCHEMA",
    "DATASET_SAMPLE_SCHEMA",
    "SIMULATION_RESULTS_SCHEMA",
    "normalize_hdf5_package_path",
    "write_dataset_mesh_info_package",
    "write_dataset_sample_package",
    "write_simulation_results_package",
]
