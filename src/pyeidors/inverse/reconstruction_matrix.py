"""Online reconstruction-matrix helpers for difference EIT."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import sparse

from pyeidors.data.channels import (
    apply_measurement_contract_to_jacobian,
    apply_measurement_contract_to_vector,
    normalize_bad_channel_mask,
    prepare_measurement_contract,
    zero_bad_channel_weights,
)
from pyeidors.data.difference import normalize_time_difference
from pyeidors.inverse.prior import RtRPrior, as_rtr_prior
from pyeidors.perf.gpu_kernels import RMMatmulResult, rm_matmul
from pyeidors.utils.numeric_ops import safe_dot


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
    path: str | None = None
    schema: str | None = None


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
) -> Path:
    """Write a reconstruction matrix artifact in HDF5 format."""

    from pyeidors.io.hdf5_artifacts import write_hdf5_artifact

    matrix = _as_rm_matrix(rm)
    meta = dict(metadata or {})
    meta.setdefault("artifact_schema", "pyeidors-rm-hdf5-v1")
    meta.setdefault("artifact_format", "hdf5")
    meta.setdefault("online_hot_path", "rm_matmul")
    meta["rm_shape"] = [int(v) for v in matrix.shape]
    arrays: dict[str, Any] = {"rm": matrix}
    shape = _positive_int_shape(voxel_shape)
    if shape:
        arrays["voxel_shape"] = np.asarray(shape, dtype=np.int64)
    for key, value in (
        ("node_coords", node_coords),
        ("cell_connectivity", cell_connectivity),
        ("channel_mask", channel_mask),
        ("measurement_weights", measurement_weights),
    ):
        if value is not None:
            arrays[key] = np.asarray(value)
    return write_hdf5_artifact(
        path,
        arrays,
        meta,
        schema="pyeidors-rm-hdf5-v1",
    )


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


def _as_measurement_vector(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim > 2:
        raise ValueError(f"{name} must be a 1D or column-vector measurement array.")
    vector = vector.reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _as_rm_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(
            f"RM artifact matrix must be non-empty 2D, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("RM artifact matrix contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


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
    arr = np.asarray(arrays[key], dtype=dtype)
    if arr.size == 0:
        return None
    return arr


def _load_hdf5_rm_artifact(path: Path) -> RMArtifact:
    from pyeidors.io.hdf5_artifacts import read_hdf5_artifact

    artifact = read_hdf5_artifact(path)
    arrays = dict(artifact.arrays)
    if "rm" not in arrays:
        raise ValueError(f"RM artifact is missing 'rm': {path}")
    return RMArtifact(
        rm=_as_rm_matrix(arrays["rm"]),
        metadata=MappingProxyType(dict(artifact.metadata)),
        voxel_shape=_positive_int_shape(arrays.get("voxel_shape")),
        node_coords=_optional_artifact_array(arrays, "node_coords", dtype=np.float64),
        cell_connectivity=_optional_artifact_array(
            arrays, "cell_connectivity", dtype=np.int32
        ),
        channel_mask=_optional_artifact_array(arrays, "channel_mask", dtype=bool),
        measurement_weights=_optional_artifact_array(
            arrays, "measurement_weights", dtype=np.float64
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
    node_coords = _optional_artifact_array(arrays, "node_coords", dtype=np.float64)
    if node_coords is None:
        node_coords = _optional_artifact_array(
            arrays, "display_node_coords", dtype=np.float64
        )
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
        path=str(path),
        schema="legacy-npz",
    )


def _as_measurement_frames(values: Any, *, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
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
    if not np.isfinite(frames).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(frames, dtype=np.float64), was_vector


def _reference_frames(
    reference: Any, *, n_frames: int, n_measurements: int
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim == 1:
        if ref.size != n_measurements:
            raise ValueError(
                f"v_ref length {ref.size} does not match {n_measurements} measurements."
            )
        return np.broadcast_to(ref.reshape(1, -1), (n_frames, n_measurements)).copy()
    if ref.ndim == 2:
        if ref.shape != (n_frames, n_measurements):
            raise ValueError(
                f"v_ref shape {ref.shape} does not match {(n_frames, n_measurements)}."
            )
        return np.ascontiguousarray(ref, dtype=np.float64)
    raise ValueError("v_ref must be a 1D reference vector or 2D frame batch.")


def _normalize_time_difference_frames(
    targets: np.ndarray,
    reference: Any,
    *,
    floor: float | None,
) -> np.ndarray:
    refs = _reference_frames(
        reference,
        n_frames=targets.shape[0],
        n_measurements=targets.shape[1],
    )
    safe = refs.copy()
    eps = (
        np.finfo(np.float64).eps
        if floor is None
        else float(max(floor, np.finfo(np.float64).eps))
    )
    small = np.abs(safe) < eps
    if np.any(small):
        signs = np.sign(safe[small])
        signs[signs == 0.0] = 1.0
        safe[small] = signs * eps
    return np.asarray((targets - refs) / safe, dtype=np.float64)


def _apply_measurement_contract_to_frames(
    frames: np.ndarray,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
) -> np.ndarray:
    n_measurements = int(frames.shape[1])
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=n_measurements)
    out = np.asarray(frames, dtype=np.float64).copy()
    if np.any(mask):
        out[:, mask] = 0.0
    weights, _ = zero_bad_channel_weights(
        measurement_weights,
        mask,
        n_measurements=n_measurements,
    )
    if weights.ndim == 1:
        out *= np.sqrt(weights).reshape(1, -1)
        return np.ascontiguousarray(out, dtype=np.float64)

    contract = prepare_measurement_contract(
        n_measurements=n_measurements,
        channel_mask=mask,
        measurement_weights=weights,
    )
    return np.asarray(out @ contract.weight_transform.T, dtype=np.float64)


def _as_jacobian(jacobian: Any) -> np.ndarray:
    if sparse.issparse(jacobian):
        matrix = np.asarray(jacobian.toarray(), dtype=np.float64)
    else:
        matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("J must be a 2D measurement-by-parameter matrix.")
    if 0 in matrix.shape:
        raise ValueError("J must be non-empty.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("J contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


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


def _prior_to_dense_matrix(prior: RtRPrior, *, name: str) -> np.ndarray:
    explicit = prior.as_RtR(dense=True)
    if sparse.issparse(explicit):
        matrix = np.asarray(explicit.toarray(), dtype=np.float64)
    elif isinstance(explicit, np.ndarray):
        matrix = np.asarray(explicit, dtype=np.float64)
    else:
        raise TypeError(f"{name} RtR prior did not produce an explicit matrix.")
    if matrix.shape != prior.shape:
        raise ValueError(f"{name} RtR shape mismatch: expected {prior.shape}.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError(f"{name} RtR contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _as_measurement_regularization(
    measurement_regularization: Any,
    *,
    n_measurements: int,
) -> tuple[np.ndarray, str]:
    if measurement_regularization is None:
        return np.eye(n_measurements, dtype=np.float64), "identity"
    if sparse.issparse(measurement_regularization):
        matrix = np.asarray(measurement_regularization.toarray(), dtype=np.float64)
    else:
        array = np.asarray(measurement_regularization, dtype=np.float64)
        matrix = np.diag(array) if array.ndim == 1 else array
    if matrix.shape != (n_measurements, n_measurements):
        raise ValueError(
            "measurement_regularization must have shape "
            f"{(n_measurements, n_measurements)}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError(
            "measurement_regularization contains non-finite values."
        )
    return np.ascontiguousarray(matrix, dtype=np.float64), "provided"


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
    diag = np.maximum(diag, float(floor))
    if exponent != 1.0:
        diag = diag ** float(exponent)
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
    if mode in {"laplace", "curvature", "graph_ltl"}:
        if regularization is None:
            if mode == "laplace":
                raise ValueError(
                    "mode='laplace' requires a graph-Laplacian regularization."
                )
            raise ValueError(
                f"mode={mode!r} requires a graph_ltl/curvature regularization."
            )
        family = "laplace" if mode == "laplace" else "graph_ltl"
        source = "provided_laplace" if mode == "laplace" else "provided_graph_ltl"
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
    return_metadata: bool = False,
) -> np.ndarray | OneStepRMResult:
    """Build a one-step GN/NOSER/Laplace/curvature reconstruction matrix.

    ``form="param"`` uses ``RM = (J.T @ J + lambda_**2 R)^-1 @ J.T``.
    ``form="measurement"`` uses
    ``RM = P J.T (J P J.T + lambda_**2 Rn)^-1`` with ``P≈R^-1`` and
    identity ``Rn`` by default.
    ``mode="noser"`` defaults to the EIDORS-style
    ``R = diag(J.T @ J)**0.5``; pass ``noser_exponent=1.0`` to reproduce the
    legacy dense ``diag(J.T @ J)`` variant.

    ``channel_mask`` uses the data-channel contract where ``True`` marks a
    bad channel. ``measurement_weights`` is the symmetric precision matrix
    ``W`` from ``J.T @ W @ J``; diagonal vectors are accepted. The returned
    RM expects online residuals passed through the same contract.
    """

    resolved_form = str(form).strip().lower()
    if resolved_form not in {"param", "measurement"}:
        raise ValueError("form must be one of: 'param', 'measurement'.")
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"tikhonov", "noser", "laplace", "curvature", "graph_ltl"}:
        raise ValueError(
            "mode must be one of: 'tikhonov', 'noser', 'laplace', 'curvature', 'graph_ltl'."
        )
    lam = float(lambda_)
    if lam < 0.0 or not np.isfinite(lam):
        raise ValueError("lambda_ must be finite and non-negative.")

    jac_raw = _as_jacobian(J)
    jac, measurement_contract = apply_measurement_contract_to_jacobian(
        jac_raw,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    reg_prior, regularization_source = _regularization_for_mode(
        jac,
        regularization,
        mode=resolved_mode,
        noser_floor=float(noser_floor),
        noser_exponent=float(noser_exponent),
    )
    reg = _prior_to_dense_matrix(reg_prior, name=resolved_mode)

    if resolved_form == "measurement":
        rn, rn_source = _as_measurement_regularization(
            measurement_regularization,
            n_measurements=int(jac.shape[0]),
        )
        p_jt, prior_inverse_solver = _solve_or_pinv(reg, jac.T)
        lhs = np.asarray(jac @ p_jt + (lam * lam) * rn, dtype=np.float64)
        rm_t, solver = _solve_or_pinv(lhs.T, p_jt.T)
        rm = rm_t.T
        inversion_dimension = "measurement"
    else:
        rn_source = "unused"
        prior_inverse_solver = "unused"
        lhs = np.asarray(jac.T @ jac + (lam * lam) * reg, dtype=np.float64)
        rm, solver = _solve_or_pinv(lhs, jac.T)
        inversion_dimension = "parameter"
    rm = np.asarray(rm, dtype=np.float64)
    if not np.isfinite(rm).all():
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
            "bad_channel_count": int(measurement_contract.bad_channel_count),
            "measurement_weight_kind": measurement_contract.weight_kind,
            "expects_measurement_contract": True,
            "normal_equation_formula": "JtWJ_plus_hp2_RtR",
            "regularization_matrix_role": "RtR",
            "RtR_shape": tuple(int(v) for v in reg.shape),
            "RtR_nnz": int(
                reg_prior.nnz if reg_prior.nnz is not None else np.count_nonzero(reg)
            ),
            "RtR_kind": reg_prior.kind,
            "RtR_signature_hash": reg_prior.signature_hash,
            "RtR_metadata": dict(reg_prior.metadata),
            "inversion_dimension": inversion_dimension,
            "regularization_source": regularization_source,
            "regularization_nnz": int(
                reg_prior.nnz if reg_prior.nnz is not None else np.count_nonzero(reg)
            ),
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
        out = np.asarray(matrix @ vector, dtype=np.float64)
    else:
        matrix = np.asarray(rm, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("rm must be a 2D reconstruction matrix.")
        if matrix.shape[1] != vector.size:
            raise ValueError(
                f"RM column count {matrix.shape[1]} does not match dv length {vector.size}."
            )
        out = np.asarray(
            safe_dot(matrix, vector, "reconstruction_matrix.apply"), dtype=np.float64
        )
    if not np.isfinite(out).all():
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
    return np.asarray(
        rm_matmul(rm, measurement, device=device, dtype=dtype),
        dtype=np.float64,
    )


def reconstruct_difference_batch(
    rm: Any,
    frames,
    *,
    normalize: bool = True,
    v_ref=None,
    floor: float | None = None,
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
    result = rm_matmul(
        rm,
        payload,
        device=device,
        dtype=dtype,
        return_metadata=return_metadata,
    )
    if return_metadata:
        return _annotate_online_hot_path_metadata(result)
    return result


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
        values=np.asarray(result.values, dtype=np.float64),
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
    if array.dtype == object:
        encoded = json.dumps(
            _canonical_signature_value(array.tolist()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    else:
        contiguous = np.ascontiguousarray(array)
        encoded = (
            str(contiguous.dtype).encode()
            + b"|"
            + json.dumps([int(v) for v in contiguous.shape]).encode()
            + b"|"
            + contiguous.tobytes()
        )
    return hashlib.sha256(encoded).hexdigest()


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
    "rm_signature",
    "rm_signature_payload",
    "write_forward_rm_benchmark_artifact",
    "write_rm_artifact",
]
