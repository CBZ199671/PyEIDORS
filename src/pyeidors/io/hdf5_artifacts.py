"""HDF5 artifact helpers for numeric cache/save payloads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from pyeidors.cache.disk_artifacts import (
    DiskArtifactManifest,
    build_disk_artifact_manifest,
    ensure_disk_artifact_metadata,
)


DEFAULT_SCHEMA = "pyeidors-hdf5-artifact-v1"
_MANIFEST_METADATA_KEYS = {"artifact_key", "artifact_manifest"}


@dataclass(frozen=True)
class HDF5Artifact:
    """Loaded HDF5 artifact payload."""

    path: Path
    schema: str
    arrays: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class HDF5DatasetInfo:
    """Metadata for one dataset inside an HDF5 artifact."""

    path: Path
    name: str
    shape: tuple[int, ...]
    dtype: str
    compression: str | None
    chunks: tuple[int, ...] | None
    sha256: str | None


class HDF5LazyDataset:
    """Lazy dataset reader that opens the file only on access."""

    def __init__(self, info: HDF5DatasetInfo, *, verify_checksum: bool = True) -> None:
        self.info = info
        self._verify_checksum = bool(verify_checksum)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.info.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.info.dtype)

    @property
    def ndim(self) -> int:
        return len(self.info.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.info.shape)) if self.info.shape else 1

    @property
    def compression(self) -> str | None:
        return self.info.compression

    @property
    def chunks(self) -> tuple[int, ...] | None:
        return self.info.chunks

    @property
    def sha256(self) -> str | None:
        return self.info.sha256

    def read(self, *, verify_checksum: bool | None = None) -> np.ndarray:
        """Read full dataset and optionally verify its checksum."""

        should_verify = (
            self._verify_checksum if verify_checksum is None else bool(verify_checksum)
        )
        with h5py.File(self.info.path, "r") as handle:
            data = np.asarray(handle["arrays"][self.info.name])
        if should_verify:
            _verify_array_checksum(data, self.info.sha256, self.info.name)
        return data

    def __array__(self, dtype=None) -> np.ndarray:
        data = self.read()
        if dtype is not None:
            return np.asarray(data, dtype=dtype)
        return data

    def __getitem__(self, key: Any) -> np.ndarray:
        with h5py.File(self.info.path, "r") as handle:
            return np.asarray(handle["arrays"][self.info.name][key])


def write_hdf5_artifact(
    path: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    compression: str | None = "gzip",
    chunks: bool | tuple[int, ...] | Mapping[str, Any] | None = True,
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write numeric arrays and JSON metadata into one HDF5 artifact."""

    target = _hdf5_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = _json_ready(dict(metadata or {}))
    meta.setdefault("artifact_format", "hdf5")
    meta.setdefault("checksum_algorithm", "sha256")
    manifest = _hdf5_disk_artifact_manifest(
        target,
        arrays,
        metadata=meta,
        schema=schema,
        subkey_payloads=subkey_payloads,
    )
    meta.setdefault("artifact_key", manifest.artifact_key)
    meta.setdefault("artifact_manifest", manifest.to_metadata())
    with h5py.File(target, "w") as handle:
        handle.attrs["schema"] = str(schema)
        handle.attrs["metadata_json"] = json.dumps(meta, sort_keys=True)
        handle.attrs["checksum_algorithm"] = "sha256"
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
            dset.attrs["compression"] = (
                "" if dset.compression is None else str(dset.compression)
            )
            dset.attrs["chunks_json"] = json.dumps(
                [] if dset.chunks is None else [int(v) for v in dset.chunks]
            )
            names.append(str(name))
        handle.attrs["array_names_json"] = json.dumps(names, sort_keys=True)
    return target


def write_large_cache_hdf5_artifact(
    path: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    compression: str | None = "gzip",
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write a large array cache with deterministic chunk/compression metadata."""

    meta = {
        "large_cache": True,
        "artifact_format": "hdf5",
        "checksum_algorithm": "sha256",
        "compression": compression or "none",
    }
    if metadata:
        meta.update(dict(metadata))
    return write_hdf5_artifact(
        path,
        arrays,
        meta,
        schema=schema,
        compression=compression,
        chunks=large_cache_chunks_for_arrays(arrays),
        subkey_payloads=subkey_payloads,
    )


def read_hdf5_artifact(
    path: str | Path,
    *,
    lazy: bool = False,
    verify_checksums: bool = True,
) -> HDF5Artifact:
    """Read an HDF5 artifact eagerly or as lazy dataset handles."""

    source = Path(path)
    with h5py.File(source, "r") as handle:
        schema = str(handle.attrs.get("schema", DEFAULT_SCHEMA))
        raw_meta = str(handle.attrs.get("metadata_json", "{}"))
        metadata = json.loads(raw_meta)
        group = handle.get("arrays")
        arrays: dict[str, Any] = {}
        if group is not None:
            for name, dataset in group.items():
                key = str(name)
                if lazy:
                    arrays[key] = HDF5LazyDataset(
                        _dataset_info(source, key, dataset),
                        verify_checksum=verify_checksums,
                    )
                else:
                    data = np.asarray(dataset)
                    if verify_checksums:
                        _verify_array_checksum(data, _dataset_sha256(dataset), key)
                    arrays[key] = data
        metadata = _ensure_hdf5_artifact_manifest(
            source,
            schema=schema,
            metadata=metadata,
            group=group,
        )
    return HDF5Artifact(
        path=source,
        schema=schema,
        arrays=arrays,
        metadata=metadata,
    )


def large_cache_chunks_for_arrays(
    arrays: Mapping[str, Any],
    *,
    target_chunk_bytes: int = 4 * 1024 * 1024,
) -> dict[str, tuple[int, ...] | bool | None]:
    """Return deterministic chunk settings for large numeric cache datasets."""

    return {
        str(name): _auto_chunk_shape(
            np.asarray(value),
            target_chunk_bytes=target_chunk_bytes,
        )
        for name, value in arrays.items()
        if value is not None
    }


def verify_hdf5_artifact_checksums(path: str | Path) -> dict[str, str]:
    """Verify all recorded dataset checksums and return digest map."""

    verified: dict[str, str] = {}
    with h5py.File(Path(path), "r") as handle:
        group = handle.get("arrays")
        if group is None:
            return verified
        for name, dataset in group.items():
            key = str(name)
            digest = _dataset_sha256(dataset)
            _verify_array_checksum(np.asarray(dataset), digest, key)
            if digest:
                verified[key] = digest
    return verified


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
        "legacy_source_read_only": True,
        "artifact_format": "hdf5",
    }
    if metadata:
        meta.update(dict(metadata))
    return write_large_cache_hdf5_artifact(target, arrays, meta, schema=schema)


def _hdf5_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix == "":
        return target.with_suffix(".h5")
    if target.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(f"HDF5 artifact path must end with .h5 or .hdf5: {target}")
    return target


def _dataset_info(path: Path, name: str, dataset: h5py.Dataset) -> HDF5DatasetInfo:
    return HDF5DatasetInfo(
        path=path,
        name=name,
        shape=tuple(int(v) for v in dataset.shape),
        dtype=str(dataset.dtype),
        compression=dataset.compression,
        chunks=None
        if dataset.chunks is None
        else tuple(int(v) for v in dataset.chunks),
        sha256=_dataset_sha256(dataset),
    )


def _dataset_sha256(dataset: h5py.Dataset) -> str | None:
    digest = dataset.attrs.get("sha256")
    if digest is None:
        return None
    return digest.decode("utf-8") if isinstance(digest, bytes) else str(digest)


def _verify_array_checksum(
    values: Any,
    expected_sha256: str | None,
    name: str,
) -> None:
    if not expected_sha256:
        return
    actual = _array_digest(values)
    if actual != expected_sha256:
        raise ValueError(
            f"HDF5 dataset checksum mismatch for {name!r}: "
            f"{actual} != {expected_sha256}"
        )


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


def _auto_chunk_shape(
    arr: np.ndarray,
    *,
    target_chunk_bytes: int,
) -> tuple[int, ...] | bool | None:
    if arr.ndim == 0 or arr.size == 0:
        return None
    if arr.dtype.kind in {"O", "S", "U"}:
        return True
    itemsize = max(int(arr.dtype.itemsize), 1)
    chunks = [max(1, int(v)) for v in arr.shape]
    max_elements = max(1, int(target_chunk_bytes) // itemsize)
    while int(np.prod(chunks)) > max_elements:
        axis = int(np.argmax(chunks))
        if chunks[axis] <= 1:
            break
        chunks[axis] = max(1, chunks[axis] // 2)
    return tuple(chunks)


def _dataset_data(arr: np.ndarray) -> tuple[Any, Any | None]:
    if arr.dtype.kind in {"U", "O"}:
        return arr.astype(str).astype(object), h5py.string_dtype(encoding="utf-8")
    return arr, None


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    if arr.dtype.kind in {"S", "U", "O"}:
        arr = _canonical_string_array(arr)
    encoded = (
        str(arr.dtype).encode()
        + b"|"
        + json.dumps([int(v) for v in arr.shape]).encode()
        + b"|"
        + arr.tobytes()
    )
    return hashlib.sha256(encoded).hexdigest()


def _hdf5_disk_artifact_manifest(
    target: Path,
    arrays: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    schema: str,
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> DiskArtifactManifest:
    array_payload = {
        str(name): {
            "shape": [int(dim) for dim in np.asarray(value).shape],
            "dtype": str(np.asarray(value).dtype),
            "sha256": _array_digest(value),
        }
        for name, value in sorted(arrays.items(), key=lambda item: str(item[0]))
        if value is not None
    }
    metadata_payload = {
        str(key): val
        for key, val in dict(metadata).items()
        if str(key) not in _MANIFEST_METADATA_KEYS
    }
    return build_disk_artifact_manifest(
        "hdf5-artifact",
        {
            "schema": str(schema),
            "arrays": array_payload,
            "metadata": metadata_payload,
        },
        files={"artifact": target},
        metadata={"artifact_format": "hdf5"},
        subkey_payloads=subkey_payloads,
    )


def _hdf5_manifest_array_payload_from_group(group: Any) -> dict[str, dict[str, Any]]:
    if group is None:
        return {}
    payload: dict[str, dict[str, Any]] = {}
    for name, dataset in sorted(group.items(), key=lambda item: str(item[0])):
        key = str(name)
        shape_raw = dataset.attrs.get("shape_json")
        if shape_raw is not None:
            shape = json.loads(
                shape_raw.decode("utf-8") if isinstance(shape_raw, bytes) else shape_raw
            )
        else:
            shape = [int(dim) for dim in dataset.shape]
        dtype_raw = dataset.attrs.get("dtype")
        dtype = dtype_raw.decode("utf-8") if isinstance(dtype_raw, bytes) else dtype_raw
        payload[key] = {
            "shape": [int(dim) for dim in shape],
            "dtype": str(dtype or dataset.dtype),
            "sha256": _dataset_sha256(dataset) or _array_digest(np.asarray(dataset)),
        }
    return payload


def _ensure_hdf5_artifact_manifest(
    source: Path,
    *,
    schema: str,
    metadata: Mapping[str, Any],
    group: Any,
) -> dict[str, Any]:
    metadata_payload = {
        str(key): val
        for key, val in dict(metadata).items()
        if str(key) not in _MANIFEST_METADATA_KEYS
    }
    return ensure_disk_artifact_metadata(
        metadata,
        "hdf5-artifact",
        {
            "schema": str(schema),
            "arrays": _hdf5_manifest_array_payload_from_group(group),
            "metadata": metadata_payload,
        },
        files={"artifact": source},
        manifest_metadata={"artifact_format": "hdf5"},
    )


def _canonical_string_array(arr: np.ndarray) -> np.ndarray:
    def normalize(item: Any) -> str:
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)

    flat = [normalize(item) for item in arr.reshape(-1)]
    return np.asarray(flat, dtype=np.str_).reshape(arr.shape)


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
    "HDF5DatasetInfo",
    "HDF5LazyDataset",
    "large_cache_chunks_for_arrays",
    "migrate_npz_to_hdf5",
    "read_hdf5_artifact",
    "verify_hdf5_artifact_checksums",
    "write_hdf5_artifact",
    "write_large_cache_hdf5_artifact",
]
