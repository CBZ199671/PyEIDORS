"""Generic RtR/R_prior contract for inverse regularization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator


RTR_PRIOR_SCHEMA = "pyeidors-rtr-prior-v1"
RTR_PRIOR_HDF5_SCHEMA = "pyeidors-rtr-prior-hdf5-v1"


@dataclass(frozen=True)
class RtRPrior:
    """Regularization prior with matrix-free and explicit-matrix views."""

    shape: tuple[int, int]
    kind: str
    metadata: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    _payload: Any = field(default=None, repr=False, compare=False)
    _apply_fn: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )
    _signature_hash: str = field(default="", repr=False, compare=False)

    @property
    def signature_hash(self) -> str:
        return self._signature_hash

    @property
    def nnz(self) -> int | None:
        if sparse.issparse(self._payload):
            return int(self._payload.nnz)
        if isinstance(self._payload, np.ndarray):
            return int(np.count_nonzero(self._payload))
        return None

    def apply(self, vector: Any) -> np.ndarray:
        vec = _as_vector(vector, name="vector")
        if vec.size != self.shape[1]:
            raise ValueError(
                f"RtR vector length {vec.size} does not match {self.shape[1]}."
            )
        if sparse.issparse(self._payload):
            out = self._payload @ vec
        elif isinstance(self._payload, np.ndarray):
            out = self._payload @ vec
        elif isinstance(self._payload, LinearOperator):
            out = self._payload.matvec(vec)
        elif self._apply_fn is not None:
            out = self._apply_fn(vec)
        else:  # pragma: no cover - constructor guards this
            raise RuntimeError("RtRPrior has no apply backend.")
        result = np.asarray(out, dtype=np.float64).reshape(-1)
        if result.size != self.shape[0]:
            raise ValueError(
                f"RtR output length {result.size} does not match {self.shape[0]}."
            )
        if not np.isfinite(result).all():
            raise FloatingPointError("RtR apply produced non-finite values.")
        return np.ascontiguousarray(result, dtype=np.float64)

    def diag(self) -> np.ndarray | None:
        if sparse.issparse(self._payload):
            return np.asarray(self._payload.diagonal(), dtype=np.float64)
        if isinstance(self._payload, np.ndarray):
            return np.asarray(np.diag(self._payload), dtype=np.float64)
        diag_hint = self.metadata.get("diag")
        if diag_hint is None:
            return None
        diag = np.asarray(diag_hint, dtype=np.float64).reshape(-1)
        if diag.size != self.shape[0]:
            raise ValueError(
                f"RtR diag length {diag.size} does not match {self.shape[0]}."
            )
        if not np.isfinite(diag).all():
            raise FloatingPointError("RtR diag contains non-finite values.")
        return np.ascontiguousarray(diag, dtype=np.float64)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(self.shape, matvec=self.apply, dtype=np.float64)

    def as_RtR(
        self,
        *,
        dense: bool = False,
        max_dense_n: int | None = None,
    ) -> np.ndarray | sparse.csr_matrix | LinearOperator:
        if sparse.issparse(self._payload):
            matrix = self._payload.tocsr()
            return matrix.toarray() if dense else matrix
        if isinstance(self._payload, np.ndarray):
            return np.asarray(self._payload, dtype=np.float64)
        if not dense:
            return self.as_linear_operator()
        _check_dense_materialization(self.shape, max_dense_n=max_dense_n)
        eye = np.eye(self.shape[1], dtype=np.float64)
        cols = [self.apply(eye[:, idx]) for idx in range(self.shape[1])]
        return np.column_stack(cols)

    def as_rtr(
        self,
        *,
        dense: bool = False,
        max_dense_n: int | None = None,
    ) -> np.ndarray | sparse.csr_matrix | LinearOperator:
        return self.as_RtR(dense=dense, max_dense_n=max_dense_n)


def as_rtr_prior(
    value: Any,
    *,
    n_parameters: int | None = None,
    shape: tuple[int, int] | None = None,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RtRPrior:
    """Coerce dense/sparse/operator/callable payload into :class:`RtRPrior`."""

    resolved_shape = _resolve_shape(value, n_parameters=n_parameters, shape=shape)
    meta = dict(metadata or {})
    if name is not None:
        meta.setdefault("name", str(name))
    meta.setdefault("schema", RTR_PRIOR_SCHEMA)

    if isinstance(value, RtRPrior):
        if tuple(value.shape) != resolved_shape:
            raise ValueError(
                f"RtRPrior shape mismatch: expected {resolved_shape}, got {value.shape}."
            )
        return value

    if value is None:
        payload = sparse.identity(resolved_shape[0], format="csr", dtype=np.float64)
        kind = "identity_sparse"
    elif sparse.issparse(value):
        payload = sparse.csr_matrix(value, dtype=np.float64)
        _validate_sparse_payload(payload)
        kind = "sparse"
    elif isinstance(value, LinearOperator):
        payload = value
        _validate_operator_shape(payload.shape, resolved_shape)
        kind = "linear_operator"
    elif callable(value):
        payload = None
        kind = "callable"
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            if array.size != resolved_shape[0]:
                raise ValueError(
                    f"RtR diagonal length {array.size} does not match {resolved_shape[0]}."
                )
            if not np.isfinite(array).all():
                raise FloatingPointError("RtR diagonal contains non-finite values.")
            payload = sparse.diags(array, offsets=0, format="csr")
            kind = "diagonal_sparse"
        elif array.ndim == 2:
            if tuple(array.shape) != resolved_shape:
                raise ValueError(
                    f"RtR matrix shape mismatch: expected {resolved_shape}, got {array.shape}."
                )
            if not np.isfinite(array).all():
                raise FloatingPointError("RtR matrix contains non-finite values.")
            payload = np.ascontiguousarray(array, dtype=np.float64)
            kind = "dense"
        else:
            raise ValueError("RtR payload must be 1D diagonal, 2D matrix, or operator.")

    apply_fn = (
        value if callable(value) and not isinstance(value, LinearOperator) else None
    )
    signature = _signature_for_payload(
        payload if apply_fn is None else value,
        shape=resolved_shape,
        kind=kind,
        metadata=meta,
    )
    meta.update(
        {
            "kind": kind,
            "shape": tuple(int(v) for v in resolved_shape),
            "signature_hash": signature,
        }
    )
    if sparse.issparse(payload):
        meta["nnz"] = int(payload.nnz)
    elif isinstance(payload, np.ndarray):
        meta["nnz"] = int(np.count_nonzero(payload))
    return RtRPrior(
        shape=resolved_shape,
        kind=kind,
        metadata=MappingProxyType(_json_ready(meta)),
        _payload=payload,
        _apply_fn=apply_fn,
        _signature_hash=signature,
    )


def write_rtr_prior_artifact(path: str | Path, prior: RtRPrior | Any) -> Path:
    """Persist an explicit matrix-backed RtR prior as HDF5."""

    from pyeidors.io.hdf5_artifacts import write_hdf5_artifact

    resolved = as_rtr_prior(prior)
    arrays: dict[str, Any]
    storage_kind: str
    explicit = resolved.as_RtR(dense=False)
    if sparse.issparse(explicit):
        csr = explicit.tocsr()
        storage_kind = "csr"
        arrays = {
            "data": csr.data,
            "indices": csr.indices.astype(np.int64, copy=False),
            "indptr": csr.indptr.astype(np.int64, copy=False),
            "shape": np.asarray(csr.shape, dtype=np.int64),
        }
    elif isinstance(explicit, np.ndarray):
        storage_kind = "dense"
        arrays = {"matrix": explicit}
    else:
        raise ValueError("Only dense/sparse matrix-backed RtR priors can be persisted.")
    metadata = dict(resolved.metadata)
    metadata.update(
        {
            "artifact_schema": RTR_PRIOR_HDF5_SCHEMA,
            "storage_kind": storage_kind,
            "signature_hash": resolved.signature_hash,
        }
    )
    return write_hdf5_artifact(
        path,
        arrays,
        metadata,
        schema=RTR_PRIOR_HDF5_SCHEMA,
    )


def load_rtr_prior_artifact(path: str | Path) -> RtRPrior:
    """Load an HDF5-persisted RtR prior."""

    from pyeidors.io.hdf5_artifacts import read_hdf5_artifact

    artifact = read_hdf5_artifact(path)
    if artifact.schema != RTR_PRIOR_HDF5_SCHEMA:
        raise ValueError(f"Unsupported RtR prior artifact schema {artifact.schema!r}.")
    metadata = dict(artifact.metadata)
    storage_kind = str(metadata.get("storage_kind", ""))
    if storage_kind == "csr":
        shape = tuple(int(v) for v in np.asarray(artifact.arrays["shape"]).reshape(-1))
        matrix = sparse.csr_matrix(
            (
                np.asarray(artifact.arrays["data"], dtype=np.float64),
                np.asarray(artifact.arrays["indices"], dtype=np.int64),
                np.asarray(artifact.arrays["indptr"], dtype=np.int64),
            ),
            shape=shape,
        )
    elif storage_kind == "dense":
        matrix = np.asarray(artifact.arrays["matrix"], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported RtR prior storage_kind {storage_kind!r}.")
    loaded = as_rtr_prior(matrix, metadata=metadata)
    expected = str(metadata.get("signature_hash", ""))
    if expected and loaded.signature_hash != expected:
        raise ValueError("RtR prior artifact signature mismatch.")
    return loaded


def _resolve_shape(
    value: Any,
    *,
    n_parameters: int | None,
    shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if isinstance(value, RtRPrior):
        candidate = tuple(int(v) for v in value.shape)
    elif shape is not None:
        candidate = tuple(int(v) for v in shape)
    elif n_parameters is not None:
        n = int(n_parameters)
        candidate = (n, n)
    else:
        raw_shape = getattr(value, "shape", None)
        if raw_shape is None:
            raise ValueError(
                "shape or n_parameters is required for callable RtR prior."
            )
        raw_tuple = tuple(int(v) for v in raw_shape)
        if len(raw_tuple) == 1:
            candidate = (raw_tuple[0], raw_tuple[0])
        else:
            candidate = raw_tuple
    _validate_operator_shape(candidate, candidate)
    if candidate[0] != candidate[1]:
        raise ValueError(f"RtR prior must be square, got {candidate}.")
    return candidate


def _validate_operator_shape(actual: Any, expected: tuple[int, int]) -> None:
    shape = tuple(int(v) for v in actual)
    if len(shape) != 2 or 0 in shape:
        raise ValueError(f"RtR shape must be non-empty 2D, got {shape}.")
    if shape != expected:
        raise ValueError(f"RtR shape mismatch: expected {expected}, got {shape}.")


def _validate_sparse_payload(matrix: sparse.spmatrix) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or 0 in matrix.shape:
        raise ValueError(
            f"RtR sparse matrix must be non-empty square, got {matrix.shape}."
        )
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise FloatingPointError("RtR sparse matrix contains non-finite values.")


def _as_vector(value: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _check_dense_materialization(
    shape: tuple[int, int],
    *,
    max_dense_n: int | None,
) -> None:
    if max_dense_n is None:
        return
    n = max(int(v) for v in shape)
    if n > int(max_dense_n):
        raise ValueError(
            f"Refusing to materialize RtR dense matrix with n={n} > {max_dense_n}."
        )


def _signature_for_payload(
    payload: Any,
    *,
    shape: tuple[int, int],
    kind: str,
    metadata: Mapping[str, Any],
) -> str:
    semantic = json.dumps(
        {
            "schema": RTR_PRIOR_SCHEMA,
            "signature_hint": metadata.get("signature_hint"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if sparse.issparse(payload):
        mat = payload.tocsr()
        encoded = (
            semantic
            + b"|"
            + str(mat.dtype).encode()
            + b"|"
            + json.dumps(list(mat.shape)).encode()
            + b"|"
            + np.ascontiguousarray(mat.indptr, dtype=np.int64).tobytes()
            + b"|"
            + np.ascontiguousarray(mat.indices, dtype=np.int64).tobytes()
            + b"|"
            + np.ascontiguousarray(mat.data, dtype=np.float64).tobytes()
        )
        return hashlib.sha256(encoded).hexdigest()
    if isinstance(payload, np.ndarray):
        arr = np.ascontiguousarray(payload, dtype=np.float64)
        encoded = (
            semantic
            + b"|"
            + str(arr.dtype).encode()
            + b"|"
            + json.dumps(list(arr.shape)).encode()
            + b"|"
            + arr.tobytes()
        )
        return hashlib.sha256(encoded).hexdigest()
    callable_id = _callable_identity(payload)
    payload_json = {
        "schema": RTR_PRIOR_SCHEMA,
        "kind": kind,
        "shape": list(shape),
        "callable": callable_id,
        "signature_hint": metadata.get("signature_hint"),
    }
    return hashlib.sha256(
        json.dumps(payload_json, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _callable_identity(value: Any) -> str:
    if isinstance(value, LinearOperator):
        return f"{type(value).__module__}.{type(value).__qualname__}:{value.shape}"
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, np.ndarray):
        return tuple(_json_ready(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "RTR_PRIOR_HDF5_SCHEMA",
    "RTR_PRIOR_SCHEMA",
    "RtRPrior",
    "as_rtr_prior",
    "load_rtr_prior_artifact",
    "write_rtr_prior_artifact",
]
