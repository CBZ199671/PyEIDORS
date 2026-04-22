"""Small HDF5 artifact helpers for numeric cache/save payloads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


DEFAULT_SCHEMA = "pyeidors-hdf5-artifact-v1"


@dataclass(frozen=True)
class HDF5Artifact:
    """Eagerly loaded HDF5 artifact payload."""

    path: Path
    schema: str
    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def write_hdf5_artifact(
    path: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    compression: str | None = "gzip",
    chunks: bool | tuple[int, ...] | Mapping[str, Any] | None = True,
) -> Path:
    """Write numeric arrays and JSON metadata into one HDF5 artifact."""

    target = _hdf5_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = _json_ready(dict(metadata or {}))
    with h5py.File(target, "w") as handle:
        handle.attrs["schema"] = str(schema)
        handle.attrs["metadata_json"] = json.dumps(meta, sort_keys=True)
        arrays_group = handle.create_group("arrays")
        names: list[str] = []
        for name, value in sorted(arrays.items(), key=lambda item: str(item[0])):
            if value is None:
                continue
            arr = np.asarray(value)
            data, dtype = _dataset_data(arr)
            dset = arrays_group.create_dataset(
                str(name),
                data=data,
                dtype=dtype,
                **_dataset_kwargs(
                    arr,
                    name=str(name),
                    compression=compression,
                    chunks=chunks,
                ),
            )
            dset.attrs["dtype"] = str(arr.dtype)
            dset.attrs["shape_json"] = json.dumps([int(v) for v in arr.shape])
            dset.attrs["sha256"] = _array_digest(arr)
            names.append(str(name))
        handle.attrs["array_names_json"] = json.dumps(names, sort_keys=True)
    return target


def read_hdf5_artifact(path: str | Path, *, lazy: bool = False) -> HDF5Artifact:
    """Read an HDF5 artifact.

    ``lazy`` is accepted for API stability; current callers need eager arrays so
    the file can be closed immediately.
    """

    _ = lazy
    source = Path(path)
    with h5py.File(source, "r") as handle:
        schema = str(handle.attrs.get("schema", DEFAULT_SCHEMA))
        raw_meta = str(handle.attrs.get("metadata_json", "{}"))
        metadata = json.loads(raw_meta)
        group = handle.get("arrays")
        arrays = {
            str(name): np.asarray(dataset)
            for name, dataset in (group.items() if group is not None else ())
        }
    return HDF5Artifact(
        path=source,
        schema=schema,
        arrays=arrays,
        metadata=metadata,
    )


def migrate_npz_to_hdf5(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    schema: str = DEFAULT_SCHEMA,
) -> Path:
    """Copy a legacy NumPy archive into HDF5 without modifying the source."""

    source = Path(src)
    target = Path(dst) if dst is not None else source.with_suffix(".h5")
    with np.load(source, allow_pickle=False) as payload:
        arrays = {str(name): np.asarray(payload[name]) for name in payload.files}
    meta = {
        "migrated_from": str(source),
        "legacy_format": source.suffix.lower().lstrip("."),
    }
    if metadata:
        meta.update(dict(metadata))
    return write_hdf5_artifact(target, arrays, meta, schema=schema)


def _hdf5_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix == "":
        return target.with_suffix(".h5")
    if target.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"HDF5 artifact path must end with .h5 or .hdf5: {target}")
    return target


def _dataset_kwargs(
    arr: np.ndarray,
    *,
    name: str,
    compression: str | None,
    chunks: bool | tuple[int, ...] | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if arr.ndim == 0 or arr.size == 0:
        return {}
    kwargs: dict[str, Any] = {}
    if compression:
        kwargs["compression"] = compression
        kwargs["shuffle"] = True
    if isinstance(chunks, Mapping):
        chunk_value = chunks.get(name)
    else:
        chunk_value = chunks
    if chunk_value is not None:
        kwargs["chunks"] = chunk_value
    return kwargs


def _dataset_data(arr: np.ndarray) -> tuple[Any, Any | None]:
    if arr.dtype.kind in {"U", "O"}:
        return arr.astype(str).astype(object), h5py.string_dtype(encoding="utf-8")
    return arr, None


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    if arr.dtype.kind in {"U", "O"}:
        arr = np.ascontiguousarray(arr.astype(str))
    encoded = (
        str(arr.dtype).encode()
        + b"|"
        + json.dumps([int(v) for v in arr.shape]).encode()
        + b"|"
        + arr.tobytes()
    )
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "DEFAULT_SCHEMA",
    "HDF5Artifact",
    "migrate_npz_to_hdf5",
    "read_hdf5_artifact",
    "write_hdf5_artifact",
]
