"""Online reconstruction-matrix helpers for difference EIT."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from pyeidors.cache.keys import update_digest_with_array_payload
from pyeidors.data.channels import (
    apply_measurement_contract_to_jacobian,
    apply_measurement_contract_to_vector,
    normalize_bad_channel_mask,
    prepare_measurement_contract,
    zero_bad_channel_weights,
)
from pyeidors.data.difference import (
    build_difference_frames,
    normalize_time_difference,
)
from pyeidors.data._temporal_core import as_real_float_array as _as_real_float_array
from pyeidors.data.temporal_filtering import (
    MeasurementTemporalFilterResult,
    filter_measurement_frames,
)
from pyeidors.inverse.prior import RtRPrior, as_rtr_prior
from pyeidors.perf.gpu_kernels import RMMatmulResult, rm_matmul
from pyeidors.utils.numeric_ops import (
    add_scaled_diagonal_in_place,
    add_scaled_values_in_place,
    all_finite_values,
    safe_dot,
)


@dataclass(frozen=True)
class OneStepRMResult:
    """One-step reconstruction matrix plus build metadata."""

    rm: np.ndarray
    metadata: MappingProxyType

    @property
    def shape(self) -> tuple[int, int]:
        return self.rm.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.rm, dtype=dtype)


@dataclass(frozen=True)
class RMArtifact:
    """Persisted reconstruction matrix artifact payload."""

    rm: np.ndarray
    metadata: MappingProxyType
    voxel_shape: tuple[int, ...] = ()
    node_coords: np.ndarray | None = None
    cell_connectivity: np.ndarray | None = None
    channel_mask: np.ndarray | None = None
    measurement_weights: np.ndarray | None = None
    rec_model: np.ndarray | None = None
    greit_y: np.ndarray | None = None
    greit_d: np.ndarray | None = None
    path: str | None = None
    schema: str | None = None


_RM_HDF5_STREAMING_CHUNK_TARGET_BYTES = 8 * 1024 * 1024
_RM_HDF5_DEFAULT_COMPRESSION = "lzf"


def rm_signature_payload(
    *,
    forward_mesh_hash: str,
    inverse_mesh_hash: str,
    electrode_geometry: Any,
    stim_meas_protocol: Any,
    background: Any,
    difference_mode: str,
    regularization_type: str,
    hyperparameters: Any,
    coarse2fine: Any | None = None,
    coarse2fine_hash: str | None = None,
    bad_channel_mask: Any | None = None,
    noise_covariance: Any | None = None,
    device: Any | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    """Return canonical mathematical RM-cache signature payload.

    ``device`` and ``backend`` are accepted for callers that pass full runtime
    context, but are intentionally excluded from the returned payload.
    """

    _ = (device, backend)
    forward_hash = str(forward_mesh_hash or "").strip()
    inverse_hash = str(inverse_mesh_hash or "").strip()
    if not forward_hash:
        raise ValueError("forward_mesh_hash is required for RM signature.")
    if not inverse_hash:
        raise ValueError("inverse_mesh_hash is required for RM signature.")
    c2f_hash = str(coarse2fine_hash or "").strip()
    if not c2f_hash:
        if coarse2fine is None:
            raise ValueError("coarse2fine or coarse2fine_hash is required.")
        c2f_hash = _digest_value(coarse2fine)
    return {
        "schema": "pyeidors-rm-signature-v1",
        "forward_mesh_hash": forward_hash,
        "inverse_mesh_hash": inverse_hash,
        "coarse2fine_hash": c2f_hash,
        "electrode_geometry": _canonical_signature_value(electrode_geometry),
        "stim_meas_protocol": _canonical_signature_value(stim_meas_protocol),
        "background": _canonical_signature_value(background),
        "difference_mode": str(difference_mode).strip().lower(),
        "bad_channel_mask": _canonical_signature_value(bad_channel_mask),
        "noise_covariance": _canonical_signature_value(noise_covariance),
        "regularization_type": str(regularization_type).strip().lower(),
        "hyperparameters": _canonical_signature_value(hyperparameters),
    }


def rm_signature(**kwargs) -> str:
    """Hash the canonical mathematical RM-cache signature payload."""

    payload = rm_signature_payload(**kwargs)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_forward_rm_benchmark_artifact(
    path: str | Path,
    *,
    offline_rm_build_seconds: float,
    online_rm_apply_seconds: float,
    online_hot_path: str = "rm_matmul",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write benchmark artifact with cold build and warm apply split."""

    build_seconds = _nonnegative_seconds(
        offline_rm_build_seconds, name="offline_rm_build_seconds"
    )
    apply_seconds = _nonnegative_seconds(
        online_rm_apply_seconds, name="online_rm_apply_seconds"
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "pyeidors-forward-rm-benchmark-v1",
        "offline_rm_build_seconds": build_seconds,
        "online_rm_apply_seconds": apply_seconds,
        "online_hot_path": str(online_hot_path),
        "env_path": shutil.which("env") or "",
        "metadata": _canonical_signature_value(metadata or {}),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_rm_artifact(
    path: str | Path,
    rm: Any,
    *,
    metadata: dict[str, Any] | None = None,
    voxel_shape: Any | None = None,
    node_coords: Any | None = None,
    cell_connectivity: Any | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    jacobian: Any | None = None,
) -> Path:
    """Write a reconstruction matrix artifact in HDF5 format."""

    from pyeidors.io.hdf5_artifacts import write_hdf5_artifact

    matrix = _as_rm_matrix(rm)
    meta = dict(metadata or {})
    meta.setdefault("artifact_schema", "pyeidors-rm-hdf5-v1")
    meta.setdefault("artifact_format", "hdf5")
    meta.setdefault("online_hot_path", "rm_matmul")
    meta["rm_shape"] = [int(v) for v in matrix.shape]
    meta["rm_dtype"] = str(matrix.dtype)
    arrays: dict[str, Any] = {"rm": matrix}
    shape = _positive_int_shape(voxel_shape)
    if shape:
        arrays["voxel_shape"] = np.asarray(shape, dtype=np.int64)
    for key, value in (
        ("node_coords", node_coords),
        ("cell_connectivity", cell_connectivity),
        ("channel_mask", channel_mask),
        ("measurement_weights", measurement_weights),
        ("jacobian", jacobian),
    ):
        if value is not None:
            arrays[key] = np.asarray(value)
    rm_chunk_target_bytes = _positive_chunk_target_bytes(
        meta.get(
            "rm_hdf5_streaming_chunk_bytes",
            _RM_HDF5_STREAMING_CHUNK_TARGET_BYTES,
        )
    )
    rm_chunks = _rm_hdf5_streaming_chunks(
        matrix,
        target_chunk_bytes=rm_chunk_target_bytes,
    )
    chunks: bool | dict[str, tuple[int, ...]]
    if rm_chunks is None:
        chunks = True
    else:
        meta.setdefault("rm_hdf5_chunk_layout", "row_full_width_v1")
        meta.setdefault("rm_hdf5_chunk_target_bytes", rm_chunk_target_bytes)
        meta.setdefault("rm_hdf5_chunks", [int(v) for v in rm_chunks])
        chunks = {"rm": rm_chunks}
    compression = _rm_hdf5_compression(meta.get("rm_hdf5_compression"))
    meta.setdefault("rm_hdf5_compression", compression or "none")
    return write_hdf5_artifact(
        path,
        arrays,
        meta,
        schema="pyeidors-rm-hdf5-v1",
        compression=compression,
        chunks=chunks,
    )


def _positive_chunk_target_bytes(value: Any) -> int:
    try:
        target = int(value)
    except (TypeError, ValueError):
        target = _RM_HDF5_STREAMING_CHUNK_TARGET_BYTES
    return max(1, target)


def _rm_hdf5_compression(value: Any) -> str | None:
    raw = str(_RM_HDF5_DEFAULT_COMPRESSION if value is None else value).strip().lower()
    if raw in {"", "auto", "fast"}:
        return _RM_HDF5_DEFAULT_COMPRESSION
    if raw in {"none", "off", "false", "0"}:
        return None
    return raw


def _rm_hdf5_streaming_chunks(
    matrix: np.ndarray,
    *,
    target_chunk_bytes: int = _RM_HDF5_STREAMING_CHUNK_TARGET_BYTES,
) -> tuple[int, int] | None:
    """Choose row-block/full-width chunks for streaming ``RM @ dv`` reads."""

    if matrix.ndim != 2 or matrix.size == 0:
        return None
    n_rows, n_cols = (int(v) for v in matrix.shape)
    if n_cols <= 0:
        return None
    itemsize = max(1, int(matrix.dtype.itemsize))
    row_bytes = max(1, n_cols * itemsize)
    rows_per_chunk = max(1, int(target_chunk_bytes) // row_bytes)
    rows_per_chunk = min(n_rows, rows_per_chunk)
    return (int(rows_per_chunk), n_cols)


def load_rm_artifact(path: str | Path) -> RMArtifact:
    """Load an HDF5 RM artifact, or read a legacy NPZ/NPY artifact."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5_rm_artifact(source)
    if suffix == ".npz":
        return _load_legacy_npz_rm_artifact(source)
    if suffix == ".npy":
        rm = _as_rm_matrix(np.load(source, allow_pickle=False))
        return RMArtifact(
            rm=rm,
            metadata=MappingProxyType(
                {"artifact_format": "legacy-npy", "legacy_read_only": True}
            ),
            path=str(source),
            schema="legacy-npy",
        )
    raise ValueError(
        f"Unsupported RM artifact suffix {suffix!r}; expected .h5, .npz, or .npy."
    )


def migrate_rm_artifact_to_hdf5(
    src: str | Path,
    dst: str | Path | None = None,
) -> Path:
    """Migrate a legacy RM artifact into HDF5 without deleting the source."""

    source = Path(src)
    artifact = load_rm_artifact(source)
    target = Path(dst) if dst is not None else source.with_suffix(".h5")
    metadata = dict(artifact.metadata)
    metadata.update(
        {
            "migrated_from": str(source),
            "legacy_format": source.suffix.lower().lstrip("."),
        }
    )
    return write_rm_artifact(
        target,
        artifact.rm,
        metadata=metadata,
        voxel_shape=artifact.voxel_shape,
        node_coords=artifact.node_coords,
        cell_connectivity=artifact.cell_connectivity,
        channel_mask=artifact.channel_mask,
        measurement_weights=artifact.measurement_weights,
    )


def _resolve_float_dtype(
    dtype: str | np.dtype[Any] | type | None,
    *,
    values: Any | None = None,
) -> np.dtype[Any]:
    if dtype is None and values is not None:
        resolved = np.asarray(values).dtype
    else:
        resolved = np.dtype(np.float64 if dtype is None else dtype)
    if resolved == np.dtype(np.float32):
        return np.dtype(np.float32)
    if resolved == np.dtype(np.float64):
        return np.dtype(np.float64)
    if resolved == np.dtype(np.complex64):
        return np.dtype(np.complex64)
    if resolved == np.dtype(np.complex128):
        return np.dtype(np.complex128)
    return np.dtype(np.float64)


def _resolve_linear_algebra_dtype(
    dtype: str | np.dtype[Any] | type | None,
    *,
    values: Any,
) -> np.dtype[Any]:
    requested = _resolve_float_dtype(dtype)
    if not np.iscomplexobj(values):
        return requested
    if requested in {np.dtype(np.float32), np.dtype(np.complex64)}:
        return np.dtype(np.complex64)
    return np.dtype(np.complex128)


def _resolve_apply_dtype(
    dtype: str | np.dtype[Any] | type | None,
    *values: Any,
) -> np.dtype[Any]:
    requested = _resolve_float_dtype(dtype)
    if not any(np.iscomplexobj(value) for value in values):
        return requested
    if requested in {np.dtype(np.float32), np.dtype(np.complex64)}:
        return np.dtype(np.complex64)
    return np.dtype(np.complex128)


def _as_numeric_float_array(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if np.issubdtype(raw.dtype, np.complexfloating):
        dtype = (
            np.complex64
            if raw.dtype.itemsize <= np.dtype(np.complex64).itemsize
            else np.complex128
        )
        return np.asarray(raw, dtype=dtype)
    return _as_real_float_array(raw)


def _as_measurement_vector(values: Any, *, name: str) -> np.ndarray:
    vector = _as_numeric_float_array(values)
    if vector.ndim > 2:
        raise ValueError(f"{name} must be a 1D or column-vector measurement array.")
    vector = vector.reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not all_finite_values(vector):
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector)


def _as_rm_matrix(
    values: Any,
    *,
    dtype: str | np.dtype[Any] | type | None = None,
) -> np.ndarray:
    resolved_dtype = _resolve_float_dtype(dtype, values=values)
    matrix = np.asarray(values, dtype=resolved_dtype)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(
            f"RM artifact matrix must be non-empty 2D, got {matrix.shape}."
        )
    if not all_finite_values(matrix):
        raise FloatingPointError("RM artifact matrix contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=resolved_dtype)


def _positive_int_shape(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return ()
    return tuple(int(v) for v in arr if int(v) > 0)


def _optional_artifact_array(
    arrays: Mapping[str, Any], key: str, *, dtype: Any
) -> np.ndarray | None:
    if key not in arrays:
        return None
    if arrays[key] is None:
        return None
    arr = np.asarray(arrays[key], dtype=dtype)
    if arr.size == 0:
        return None
    return arr


def _optional_node_coords_array(
    arrays: Mapping[str, Any], key: str
) -> np.ndarray | None:
    if key not in arrays:
        return None
    if arrays[key] is None:
        return None
    arr = np.asarray(arrays[key])
    if arr.size == 0:
        return None
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    if np.issubdtype(np.asarray(arr).dtype, np.floating):
        return np.asarray(arr)
    return np.asarray(arr, dtype=np.float64)


def _optional_artifact_array_aliases(
    arrays: Mapping[str, Any],
    *keys: str,
    dtype: Any,
) -> np.ndarray | None:
    for key in keys:
        arr = _optional_artifact_array(arrays, key, dtype=dtype)
        if arr is not None:
            return arr
    return None


def _load_hdf5_rm_artifact(path: Path) -> RMArtifact:
    from pyeidors.io.hdf5_artifacts import read_hdf5_artifact

    artifact = read_hdf5_artifact(path, lazy=True)
    arrays = dict(artifact.arrays)
    rm_array = arrays.get("rm")
    if rm_array is None:
        rm_array = arrays.get("RM")
    if rm_array is None:
        raise ValueError(f"RM artifact is missing 'rm': {path}")
    return RMArtifact(
        rm=_as_rm_matrix(rm_array),
        metadata=MappingProxyType(dict(artifact.metadata)),
        voxel_shape=_positive_int_shape(arrays.get("voxel_shape")),
        node_coords=_optional_node_coords_array(arrays, "node_coords"),
        cell_connectivity=_optional_artifact_array(
            arrays, "cell_connectivity", dtype=np.int32
        ),
        channel_mask=_optional_artifact_array(arrays, "channel_mask", dtype=bool),
        measurement_weights=_optional_artifact_array(
            arrays, "measurement_weights", dtype=np.float64
        ),
        rec_model=_optional_artifact_array(arrays, "rec_model", dtype=np.float64),
        greit_y=_optional_artifact_array_aliases(
            arrays,
            "y",
            "Y",
            dtype=np.float64,
        ),
        greit_d=_optional_artifact_array_aliases(
            arrays,
            "d",
            "D",
            dtype=np.float64,
        ),
        path=str(path),
        schema=artifact.schema,
    )


def _load_legacy_npz_rm_artifact(path: Path) -> RMArtifact:
    with np.load(path, allow_pickle=False) as payload:
        if "rm" not in payload:
            raise ValueError(f"RM artifact is missing 'rm': {path}")
        metadata: dict[str, Any] = {
            "artifact_format": "legacy-npz",
            "legacy_read_only": True,
        }
        if "metadata_json" in payload:
            raw = str(payload["metadata_json"].item())
            try:
                metadata.update(json.loads(raw))
            except json.JSONDecodeError:
                metadata["metadata_json"] = raw
        arrays = {str(name): np.asarray(payload[name]) for name in payload.files}
    node_coords = _optional_node_coords_array(arrays, "node_coords")
    if node_coords is None:
        node_coords = _optional_node_coords_array(arrays, "display_node_coords")
    cell_connectivity = _optional_artifact_array(
        arrays, "cell_connectivity", dtype=np.int32
    )
    if cell_connectivity is None:
        cell_connectivity = _optional_artifact_array(
            arrays, "display_cell_connectivity", dtype=np.int32
        )
    return RMArtifact(
        rm=_as_rm_matrix(arrays["rm"]),
        metadata=MappingProxyType(metadata),
        voxel_shape=_positive_int_shape(arrays.get("voxel_shape")),
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
        channel_mask=_optional_artifact_array(arrays, "channel_mask", dtype=bool),
        measurement_weights=_optional_artifact_array(
            arrays, "measurement_weights", dtype=np.float64
        ),
        rec_model=_optional_artifact_array(arrays, "rec_model", dtype=np.float64),
        greit_y=_optional_artifact_array_aliases(
            arrays,
            "y",
            "Y",
            dtype=np.float64,
        ),
        greit_d=_optional_artifact_array_aliases(
            arrays,
            "d",
            "D",
            dtype=np.float64,
        ),
        path=str(path),
        schema="legacy-npz",
    )


def _as_measurement_frames(values: Any, *, name: str) -> tuple[np.ndarray, bool]:
    array = _as_numeric_float_array(values)
    if array.ndim == 1:
        frames = array.reshape(1, -1)
        was_vector = True
    elif array.ndim == 2:
        frames = array
        was_vector = False
    else:
        raise ValueError(f"{name} must be a 1D vector or 2D frame batch.")
    if frames.shape[0] == 0 or frames.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not all_finite_values(frames):
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(frames), was_vector


def _reference_frames(
    reference: Any, *, n_frames: int, n_measurements: int
) -> np.ndarray:
    ref = _as_numeric_float_array(reference)
    if ref.ndim == 1:
        if ref.size != n_measurements:
            raise ValueError(
                f"v_ref length {ref.size} does not match {n_measurements} measurements."
            )
        return np.ascontiguousarray(ref.reshape(1, -1), dtype=ref.dtype)
    if ref.ndim == 2:
        if ref.shape != (n_frames, n_measurements):
            raise ValueError(
                f"v_ref shape {ref.shape} does not match {(n_frames, n_measurements)}."
            )
        return np.ascontiguousarray(ref, dtype=ref.dtype)
    raise ValueError("v_ref must be a 1D reference vector or 2D frame batch.")


def _normalize_time_difference_frames(
    targets: np.ndarray,
    reference: Any,
    *,
    floor: float | None,
    orientation: str = "target_minus_reference",
) -> np.ndarray:
    refs = _reference_frames(
        reference,
        n_frames=targets.shape[0],
        n_measurements=targets.shape[1],
    )
    return build_difference_frames(
        targets,
        refs,
        mode="normalized",
        orientation=orientation,
        floor=floor,
    )


def _apply_measurement_contract_to_frames(
    frames: np.ndarray,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> np.ndarray:
    payload, _, _ = _apply_measurement_contract_to_frames_with_metadata(
        frames,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    return payload


def _apply_measurement_contract_to_frames_with_metadata(
    frames: np.ndarray,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> tuple[np.ndarray, int, str]:
    n_measurements = int(frames.shape[1])
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=n_measurements)
    bad_channel_count = int(np.count_nonzero(mask))
    out = np.array(_as_numeric_float_array(frames), copy=True, order="C")
    if bad_channel_count:
        _zero_bad_measurement_columns_in_place(out, mask)
    weights, weight_kind = zero_bad_channel_weights(
        measurement_weights,
        mask,
        n_measurements=n_measurements,
    )
    if weights.ndim == 1:
        np.sqrt(weights, out=weights)
        out *= weights.reshape(1, -1)
        return (
            np.ascontiguousarray(out),
            bad_channel_count,
            weight_kind,
        )

    contract = prepare_measurement_contract(
        n_measurements=n_measurements,
        channel_mask=mask,
        measurement_weights=weights,
    )
    return (
        np.asarray(out @ contract.weight_transform.T, dtype=out.dtype),
        bad_channel_count,
        contract.weight_kind,
    )


def _zero_bad_measurement_columns_in_place(
    frames: np.ndarray,
    mask: np.ndarray,
) -> None:
    for col_idx, is_bad in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        if bool(is_bad):
            frames[:, col_idx] = 0.0


def _as_jacobian(
    jacobian: Any,
    *,
    dtype: str | np.dtype[Any] | type = np.float64,
) -> np.ndarray:
    resolved_dtype = _resolve_float_dtype(dtype)
    if sparse.issparse(jacobian):
        matrix = np.asarray(jacobian.toarray(), dtype=resolved_dtype)
    else:
        matrix = np.asarray(jacobian, dtype=resolved_dtype)
    if matrix.ndim != 2:
        raise ValueError("J must be a 2D measurement-by-parameter matrix.")
    if 0 in matrix.shape:
        raise ValueError("J must be non-empty.")
    if not all_finite_values(matrix):
        raise FloatingPointError("J contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=resolved_dtype)


def _as_regularization_prior(
    regularization: Any,
    *,
    n_parameters: int,
    name: str,
    metadata: Mapping[str, Any] | None = None,
) -> RtRPrior:
    prior = as_rtr_prior(
        regularization,
        n_parameters=n_parameters,
        name=name,
        metadata=metadata,
    )
    if prior.shape != (n_parameters, n_parameters):
        raise ValueError(
            "regularization must have shape "
            f"{(n_parameters, n_parameters)}, got {prior.shape}."
        )
    return prior


def _prior_to_dense_matrix(
    prior: RtRPrior,
    *,
    name: str,
    dtype: str | np.dtype[Any] | type = np.float64,
) -> np.ndarray:
    resolved_dtype = _resolve_float_dtype(dtype)
    explicit = prior.as_RtR(dense=True)
    if sparse.issparse(explicit):
        matrix = np.asarray(explicit.toarray(), dtype=resolved_dtype)
    elif isinstance(explicit, np.ndarray):
        matrix = np.asarray(explicit, dtype=resolved_dtype)
    else:
        raise TypeError(f"{name} RtR prior did not produce an explicit matrix.")
    if matrix.shape != prior.shape:
        raise ValueError(f"{name} RtR shape mismatch: expected {prior.shape}.")
    if not all_finite_values(matrix):
        raise FloatingPointError(f"{name} RtR contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=resolved_dtype)


def _prior_diagonal(
    prior: RtRPrior,
    *,
    dtype: str | np.dtype[Any] | type = np.float64,
) -> np.ndarray | None:
    diag = prior.diag()
    if diag is None:
        return None
    resolved_dtype = _resolve_float_dtype(dtype)
    diag = np.asarray(diag, dtype=resolved_dtype).reshape(-1)
    if diag.size != prior.shape[0]:
        raise ValueError(
            f"RtR diag length {diag.size} does not match {prior.shape[0]}."
        )
    if not all_finite_values(diag):
        raise FloatingPointError("RtR diag contains non-finite values.")
    return np.ascontiguousarray(diag, dtype=resolved_dtype)


def _prior_nnz(prior: RtRPrior) -> int:
    if prior.nnz is not None:
        return int(prior.nnz)
    diag = prior.diag()
    if diag is not None:
        return int(np.count_nonzero(diag))
    return int(np.prod(prior.shape))


def _as_measurement_regularization(
    measurement_regularization: Any,
    *,
    n_measurements: int,
    dtype: str | np.dtype[Any] | type = np.float64,
) -> tuple[np.ndarray, str, str]:
    resolved_dtype = _resolve_float_dtype(dtype)
    if measurement_regularization is None:
        return np.ones(n_measurements, dtype=resolved_dtype), "identity", "diagonal"
    if sparse.issparse(measurement_regularization):
        matrix = np.asarray(measurement_regularization.toarray(), dtype=resolved_dtype)
        kind = "matrix"
    else:
        array = np.asarray(measurement_regularization, dtype=resolved_dtype)
        if array.ndim == 1:
            diag = np.ascontiguousarray(array.reshape(-1), dtype=resolved_dtype)
            if diag.shape != (n_measurements,):
                raise ValueError(
                    "measurement_regularization must have shape "
                    f"{(n_measurements, n_measurements)} or {(n_measurements,)}, "
                    f"got {array.shape}."
                )
            if not all_finite_values(diag):
                raise FloatingPointError(
                    "measurement_regularization contains non-finite values."
                )
            return diag, "provided", "diagonal"
        matrix = array
        kind = "matrix"
    if matrix.shape != (n_measurements, n_measurements):
        raise ValueError(
            "measurement_regularization must have shape "
            f"{(n_measurements, n_measurements)} or {(n_measurements,)}, "
            f"got {matrix.shape}."
        )
    if not all_finite_values(matrix):
        raise FloatingPointError(
            "measurement_regularization contains non-finite values."
        )
    return np.ascontiguousarray(matrix, dtype=resolved_dtype), "provided", kind


def _noser_regularization(
    jacobian: np.ndarray,
    *,
    floor: float,
    exponent: float,
) -> np.ndarray:
    if floor < 0.0:
        raise ValueError("noser_floor must be non-negative.")
    if exponent <= 0.0:
        raise ValueError("noser_exponent must be positive.")
    diag = np.sum(jacobian * jacobian, axis=0)
    if np.iscomplexobj(diag):
        diag = np.asarray(diag, dtype=jacobian.dtype)
        diag[np.abs(diag) < float(floor)] = complex(float(floor), 0.0)
    else:
        diag = np.maximum(diag, float(floor))
    if exponent != 1.0:
        diag = diag ** float(exponent)
    if np.iscomplexobj(diag):
        # EIDORS' complex NOSER prior follows MATLAB's projected Jacobian
        # convention; after importing the same physical Jacobian into NumPy,
        # the matching RtR diagonal is the conjugate of sum(J.^2).^exponent.
        diag = np.conj(diag)
    return diag


def _regularization_for_mode(
    jacobian: np.ndarray,
    regularization: Any,
    *,
    mode: str,
    noser_floor: float,
    noser_exponent: float,
) -> tuple[RtRPrior, str]:
    n_parameters = int(jacobian.shape[1])
    if mode == "noser":
        return (
            as_rtr_prior(
                _noser_regularization(
                    jacobian,
                    floor=float(noser_floor),
                    exponent=float(noser_exponent),
                ),
                n_parameters=n_parameters,
                name="noser",
                metadata={
                    "regularization_source": "diag_jtj",
                    "noser_exponent": float(noser_exponent),
                    "signature_hint": "noser",
                },
            ),
            "diag_jtj",
        )
    if mode in {"laplace", "curvature", "graph_ltl", "tv_irls"}:
        if regularization is None:
            if mode == "laplace":
                raise ValueError(
                    "mode='laplace' requires a graph-Laplacian regularization."
                )
            raise ValueError(
                f"mode={mode!r} requires a graph_ltl/curvature/TV-IRLS regularization."
            )
        if mode == "laplace":
            family = "laplace"
            source = "provided_laplace"
        elif mode == "tv_irls":
            family = "tv_irls"
            source = "provided_tv_irls"
        else:
            family = "graph_ltl"
            source = "provided_graph_ltl"
        return (
            _as_regularization_prior(
                regularization,
                n_parameters=n_parameters,
                name=mode,
                metadata={
                    "prior_family": family,
                    "regularization_source": source,
                    "signature_hint": family,
                },
            ),
            source,
        )
    return (
        _as_regularization_prior(
            regularization,
            n_parameters=n_parameters,
            name="tikhonov",
            metadata={
                "regularization_source": "identity"
                if regularization is None
                else "provided",
                "signature_hint": "tikhonov",
            },
        ),
        "identity" if regularization is None else "provided",
    )


def _solve_or_pinv(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        return np.linalg.solve(lhs, rhs), "solve"
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs, "pinv"


def build_one_step_rm(
    J: Any,
    regularization: Any = None,
    lambda_: float = 1e-2,
    *,
    mode: str = "tikhonov",
    form: str = "param",
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    measurement_regularization: Any = None,
    noser_floor: float = 1e-12,
    noser_exponent: float = 0.5,
    dtype: str | np.dtype[Any] = "float64",
    return_metadata: bool = False,
) -> np.ndarray | OneStepRMResult:
    """Build a one-step GN/NOSER/Laplace/curvature/TV-IRLS reconstruction matrix.

    ``form="param"`` uses ``RM = (J.T @ J + lambda_**2 R)^-1 @ J.T``.
    ``form="measurement"`` uses
    ``RM = P J.T (J P J.T + lambda_**2 Rn)^-1`` with ``P≈R^-1`` and
    identity ``Rn`` by default.
    ``mode="noser"`` defaults to the EIDORS-style
    ``R = diag(J.T @ J)**0.5`` for real-valued Jacobians and the EIDORS
    complex projected-Jacobian convention for complex-valued Jacobians; pass
    ``noser_exponent=1.0`` to reproduce the legacy dense ``diag(J.T @ J)``
    variant.

    ``channel_mask`` uses the data-channel contract where ``True`` marks a
    bad channel. ``measurement_weights`` is the symmetric precision matrix
    ``W`` from ``J.T @ W @ J``; diagonal vectors are accepted. The returned
    RM expects online residuals passed through the same contract.
    """

    resolved_form = str(form).strip().lower()
    if resolved_form not in {"param", "measurement"}:
        raise ValueError("form must be one of: 'param', 'measurement'.")
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {
        "tikhonov",
        "noser",
        "laplace",
        "curvature",
        "graph_ltl",
        "tv_irls",
    }:
        raise ValueError(
            "mode must be one of: 'tikhonov', 'noser', 'laplace', 'curvature', 'graph_ltl', 'tv_irls'."
        )
    lam = float(lambda_)
    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError("lambda_ must be finite and non-negative.")
    calc_dtype = _resolve_linear_algebra_dtype(dtype, values=J)

    jac_raw = _as_jacobian(J, dtype=calc_dtype)
    jac, measurement_contract = apply_measurement_contract_to_jacobian(
        jac_raw,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    jac = np.ascontiguousarray(jac, dtype=calc_dtype)
    jac_adjoint = np.ascontiguousarray(jac.conj().T, dtype=calc_dtype)
    reg_prior, regularization_source = _regularization_for_mode(
        jac,
        regularization,
        mode=resolved_mode,
        noser_floor=float(noser_floor),
        noser_exponent=float(noser_exponent),
    )

    if resolved_form == "measurement":
        rn, rn_source, rn_kind = _as_measurement_regularization(
            measurement_regularization,
            n_measurements=int(jac.shape[0]),
            dtype=calc_dtype,
        )
        diag = _prior_diagonal(reg_prior, dtype=calc_dtype)
        diag_real = np.real_if_close(diag, tol=1000) if diag is not None else None
        diag_positive = bool(
            diag_real is not None
            and not np.iscomplexobj(diag_real)
            and np.all(np.asarray(diag_real) > 0.0)
        )
        if reg_prior.kind == "diagonal_sparse" and diag is not None and diag_positive:
            p_jt = jac_adjoint / np.asarray(diag_real, dtype=calc_dtype).reshape(-1, 1)
            prior_inverse_solver = "diagonal"
        else:
            reg = _prior_to_dense_matrix(
                reg_prior,
                name=resolved_mode,
                dtype=calc_dtype,
            )
            p_jt, prior_inverse_solver = _solve_or_pinv(reg, jac_adjoint)
            p_jt = np.asarray(p_jt, dtype=calc_dtype)
        lhs = np.ascontiguousarray(jac @ p_jt, dtype=calc_dtype)
        if rn_kind == "diagonal":
            add_scaled_diagonal_in_place(lhs, rn, lam * lam)
        else:
            add_scaled_values_in_place(lhs, rn, lam * lam)
        rm_t, solver = _solve_or_pinv(lhs.T, p_jt.T)
        rm = rm_t.T
        inversion_dimension = "measurement"
    else:
        rn_source = "unused"
        prior_inverse_solver = "unused"
        reg = _prior_to_dense_matrix(reg_prior, name=resolved_mode, dtype=calc_dtype)
        lhs = np.asarray(jac_adjoint @ jac + (lam * lam) * reg, dtype=calc_dtype)
        rm, solver = _solve_or_pinv(lhs, jac_adjoint)
        inversion_dimension = "parameter"
    rm = np.asarray(rm, dtype=calc_dtype)
    if not all_finite_values(rm):
        raise FloatingPointError("one-step RM contains non-finite values.")
    try:
        condition_estimate = float(np.linalg.cond(lhs))
    except np.linalg.LinAlgError:
        condition_estimate = float("inf")

    metadata = MappingProxyType(
        {
            "algorithm": "one-step-gn",
            "solver_family": "gauss-newton",
            "mode": resolved_mode,
            "regularization_type": resolved_mode,
            "form": resolved_form,
            "lambda": lam,
            "hyperparameter_name": "hp",
            "hyperparameter": lam,
            "hp": lam,
            "hp_squared": lam * lam,
            "lambda_squared": lam * lam,
            "n_measurements": int(jac.shape[0]),
            "n_parameters": int(jac.shape[1]),
            "rm_dtype": str(calc_dtype),
            "build_dtype": str(calc_dtype),
            "bad_channel_count": int(measurement_contract.bad_channel_count),
            "measurement_weight_kind": measurement_contract.weight_kind,
            "expects_measurement_contract": True,
            "normal_equation_formula": "JhWJ_plus_hp2_RtR"
            if np.issubdtype(calc_dtype, np.complexfloating)
            else "JtWJ_plus_hp2_RtR",
            "adjoint_operator": "hermitian"
            if np.issubdtype(calc_dtype, np.complexfloating)
            else "transpose",
            "regularization_matrix_role": "RtR",
            "RtR_shape": tuple(int(v) for v in reg_prior.shape),
            "RtR_nnz": _prior_nnz(reg_prior),
            "RtR_kind": reg_prior.kind,
            "RtR_signature_hash": reg_prior.signature_hash,
            "RtR_metadata": dict(reg_prior.metadata),
            "inversion_dimension": inversion_dimension,
            "regularization_source": regularization_source,
            "regularization_nnz": _prior_nnz(reg_prior),
            "noser_exponent": float(noser_exponent)
            if resolved_mode == "noser"
            else None,
            "measurement_regularization_source": rn_source,
            "condition_estimate": condition_estimate,
            "solver": solver,
            "prior_inverse_solver": prior_inverse_solver,
            "system_shape": tuple(int(v) for v in lhs.shape),
            "rm_shape": tuple(int(v) for v in rm.shape),
        }
    )
    if return_metadata:
        return OneStepRMResult(rm=rm, metadata=metadata)
    return rm


def _matvec(rm: Any, vector: np.ndarray) -> np.ndarray:
    if sparse.issparse(rm):
        matrix = rm.tocsr()
        if matrix.ndim != 2:
            raise ValueError("rm must be a 2D reconstruction matrix.")
        if matrix.shape[1] != vector.size:
            raise ValueError(
                f"RM column count {matrix.shape[1]} does not match dv length {vector.size}."
            )
        out = np.asarray(matrix @ vector, dtype=_resolve_apply_dtype(None, rm, vector))
    else:
        matrix = np.asarray(rm, dtype=_resolve_apply_dtype(None, rm, vector))
        if matrix.ndim != 2:
            raise ValueError("rm must be a 2D reconstruction matrix.")
        if matrix.shape[1] != vector.size:
            raise ValueError(
                f"RM column count {matrix.shape[1]} does not match dv length {vector.size}."
            )
        out = np.asarray(
            safe_dot(matrix, vector, "reconstruction_matrix.apply"),
            dtype=matrix.dtype,
        )
    if not all_finite_values(out):
        raise FloatingPointError("RM application produced non-finite values.")
    return out.reshape(-1)


def reconstruct_difference(
    rm: Any,
    dv,
    *,
    normalize: bool = True,
    v_ref=None,
    floor: float | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    device: str = "cpu",
    dtype: str | np.dtype[Any] = "float64",
) -> np.ndarray:
    """Apply a precomputed reconstruction matrix to one difference frame.

    If ``normalize`` is true and ``v_ref`` is provided, ``dv`` is interpreted
    as target voltages ``v_t`` and first converted with
    :func:`normalize_time_difference`. Otherwise ``dv`` is treated as an
    already-projected measurement vector. The hot path is deliberately just
    ``RM @ dv_projected``; RM construction belongs to later T16/T17 tasks.
    """

    if normalize and v_ref is not None:
        measurement = normalize_time_difference(dv, v_ref, floor=floor)
    else:
        measurement = _as_measurement_vector(dv, name="dv")
    measurement, _ = apply_measurement_contract_to_vector(
        measurement,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    apply_dtype = _resolve_apply_dtype(dtype, rm, measurement)
    return np.asarray(rm_matmul(rm, measurement, device=device, dtype=apply_dtype))


def reconstruct_difference_batch(
    rm: Any,
    frames,
    *,
    normalize: bool = True,
    v_ref=None,
    floor: float | None = None,
    difference_orientation: str = "target_minus_reference",
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
    return_metadata: bool = False,
) -> np.ndarray | RMMatmulResult:
    """Apply a precomputed RM to one or more online difference frames."""

    frame_batch, was_vector = _as_measurement_frames(frames, name="frames")
    if normalize and v_ref is not None:
        measurement_batch = _normalize_time_difference_frames(
            frame_batch,
            v_ref,
            floor=floor,
            orientation=difference_orientation,
        )
    else:
        measurement_batch = frame_batch
    measurement_batch = _apply_measurement_contract_to_frames(
        measurement_batch,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    payload: np.ndarray
    if was_vector:
        payload = measurement_batch.reshape(-1)
    else:
        payload = measurement_batch
    apply_dtype = _resolve_apply_dtype(dtype, rm, payload)
    result = rm_matmul(
        rm,
        payload,
        device=device,
        dtype=apply_dtype,
        return_metadata=return_metadata,
    )
    if return_metadata:
        return _annotate_online_hot_path_metadata(result)
    return result


def reconstruct_temporal_difference_batch(
    rm: Any,
    frames,
    *,
    normalize: bool = True,
    v_ref=None,
    floor: float | None = None,
    difference_orientation: str = "target_minus_reference",
    temporal: str = "none",
    moving_window: int = 3,
    exponential_alpha: float = 0.5,
    filter_state: Mapping[str, Any] | None = None,
    timestamps: Any | None = None,
    sample_rate_hz: float | None = None,
    filter_hook: Any | None = None,
    hook_kind: str | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
    return_metadata: bool = False,
) -> np.ndarray | RMMatmulResult:
    """Filter measurement frames causally, then apply a cached RM.

    This hot path remains measurement-space preprocessing followed by one
    ``RM @ delta_v`` batch. It never rebuilds a Jacobian, KSP, or forward solve.
    """

    total_start = time.perf_counter()
    projection_start = total_start
    frame_batch, was_vector = _as_measurement_frames(frames, name="frames")
    if normalize and v_ref is not None:
        measurement_batch = _normalize_time_difference_frames(
            frame_batch,
            v_ref,
            floor=floor,
            orientation=difference_orientation,
        )
        projection_kind = "normalized_time_difference"
    else:
        measurement_batch = frame_batch
        projection_kind = "preprojected_measurement"
    projection_seconds = time.perf_counter() - projection_start

    filter_start = time.perf_counter()
    filter_result = filter_measurement_frames(
        measurement_batch,
        temporal=temporal,
        moving_window=moving_window,
        exponential_alpha=exponential_alpha,
        initial_state=filter_state,
        timestamps=timestamps,
        sample_rate_hz=sample_rate_hz,
        hook=filter_hook,
        hook_kind=hook_kind,
        return_metadata=True,
    )
    assert isinstance(filter_result, MeasurementTemporalFilterResult)
    filtered_batch, _ = _as_measurement_frames(filter_result.values, name="filtered")
    filter_seconds = time.perf_counter() - filter_start

    contract_start = time.perf_counter()
    (
        measurement_payload,
        bad_channel_count,
        measurement_weight_kind,
    ) = _apply_measurement_contract_to_frames_with_metadata(
        filtered_batch,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    contract_seconds = time.perf_counter() - contract_start

    payload: np.ndarray
    if was_vector:
        payload = measurement_payload.reshape(-1)
    else:
        payload = measurement_payload
    rm_start = time.perf_counter()
    matmul = rm_matmul(
        rm,
        payload,
        device=device,
        dtype=dtype,
        return_metadata=True,
    )
    assert isinstance(matmul, RMMatmulResult)
    rm_seconds = time.perf_counter() - rm_start
    total_seconds = time.perf_counter() - total_start

    metadata = dict(matmul.metadata)
    metadata.update(
        {
            "schema": "pyeidors-temporal-rm-online-v1",
            "online_hot_path": "temporal_filter_plus_rm_matmul",
            "rm_online_hot_path": "rm_matmul",
            "projection_kind": projection_kind,
            "normalize": bool(normalize and v_ref is not None),
            "difference_orientation": str(difference_orientation),
            "measurement_contract_applied": True,
            "bad_channel_count": bad_channel_count,
            "measurement_weight_kind": measurement_weight_kind,
            "temporal_filter_metadata": MappingProxyType(dict(filter_result.metadata)),
            "temporal_filter_state": filter_result.metadata["final_state"],
            "timestamp_policy": "metadata_only_no_smoothing",
            "timestamps": filter_result.metadata["timestamps"],
            "offline_rm_build_seconds": 0.0,
            "online_projection_seconds": float(projection_seconds),
            "online_temporal_filter_seconds": float(filter_seconds),
            "online_measurement_contract_seconds": float(contract_seconds),
            "online_rm_apply_seconds": float(rm_seconds),
            "online_total_seconds": float(total_seconds),
            "forward_solve_count": 0,
            "adjoint_solve_count": 0,
            "ksp_solve_count": 0,
            "jacobian_rebuild_count": 0,
        }
    )
    result = RMMatmulResult(
        values=np.asarray(matmul.values),
        metadata=MappingProxyType(metadata),
    )
    return result if return_metadata else result.values


def _annotate_online_hot_path_metadata(result: RMMatmulResult) -> RMMatmulResult:
    meta = dict(result.metadata)
    meta.update(
        {
            "online_hot_path": "rm_matmul",
            "forward_solve_count": 0,
            "adjoint_solve_count": 0,
            "ksp_solve_count": 0,
            "jacobian_rebuild_count": 0,
        }
    )
    return RMMatmulResult(
        values=np.asarray(result.values),
        metadata=MappingProxyType(meta),
    )


def _canonical_signature_value(value: Any) -> Any:
    if value is None:
        return None
    if sparse.issparse(value):
        matrix = value.tocsr()
        return {
            "sparse": "csr",
            "shape": [int(v) for v in matrix.shape],
            "data_hash": _digest_value(matrix.data),
            "indices_hash": _digest_value(matrix.indices),
            "indptr_hash": _digest_value(matrix.indptr),
        }
    if isinstance(value, MappingProxyType):
        return _canonical_signature_value(dict(value))
    if isinstance(value, dict):
        return {
            str(key): _canonical_signature_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_signature_value(item) for item in value]
    array = np.asarray(value) if _looks_array_like(value) else None
    if array is not None and array.ndim > 0:
        return {
            "shape": [int(v) for v in array.shape],
            "dtype": str(array.dtype),
            "hash": _digest_value(array),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def _looks_array_like(value: Any) -> bool:
    return isinstance(value, (np.ndarray, list, tuple)) and not isinstance(value, str)


def _digest_value(value: Any) -> str:
    if sparse.issparse(value):
        return _digest_value(_canonical_signature_value(value))
    array = np.asarray(value)
    digest = hashlib.sha256()
    if array.dtype == object:
        encoded = json.dumps(
            _canonical_signature_value(array.tolist()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(encoded)
    else:
        digest.update(str(array.dtype).encode())
        digest.update(b"|")
        digest.update(json.dumps([int(v) for v in array.shape]).encode())
        digest.update(b"|")
        update_digest_with_array_payload(digest, array)
    return digest.hexdigest()


def _nonnegative_seconds(value: float, *, name: str) -> float:
    seconds = float(value)
    if seconds < 0.0 or not np.isfinite(seconds):
        raise ValueError(f"{name} must be finite and non-negative.")
    return seconds


__all__ = [
    "OneStepRMResult",
    "RMArtifact",
    "build_one_step_rm",
    "load_rm_artifact",
    "migrate_rm_artifact_to_hdf5",
    "reconstruct_difference",
    "reconstruct_difference_batch",
    "reconstruct_temporal_difference_batch",
    "rm_signature",
    "rm_signature_payload",
    "write_forward_rm_benchmark_artifact",
    "write_rm_artifact",
]
