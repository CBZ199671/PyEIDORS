"""HDF5 artifact helpers for numeric cache/save payloads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from pyeidors.cache.disk_artifacts import (
    DiskArtifactManifest,
    build_disk_artifact_manifest,
    ensure_disk_artifact_metadata,
)
from pyeidors.io._json import json_ready as _json_ready


DEFAULT_SCHEMA = "pyeidors-hdf5-artifact-v1"
_DATASET_DIGEST_CHUNK_BYTES = 16 * 1024 * 1024
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
    swmr: bool = False


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
        with _open_hdf5_read(self.info.path, swmr=self.info.swmr) as handle:
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
        with _open_hdf5_read(self.info.path, swmr=self.info.swmr) as handle:
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
    swmr: bool | None = None,
) -> Path:
    """Write numeric arrays and JSON metadata into one HDF5 artifact."""

    target = _hdf5_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = _json_ready(dict(metadata or {}))
    meta.setdefault("artifact_format", "hdf5")
    meta.setdefault("checksum_algorithm", "sha256")
    use_swmr = _resolve_hdf5_swmr(swmr, default=True)
    meta.setdefault("hdf5_swmr_ready", bool(use_swmr))
    meta.setdefault("hdf5_libver", "latest" if use_swmr else "default")
    array_payload = _hdf5_manifest_array_payload_from_arrays(arrays)
    manifest = _hdf5_disk_artifact_manifest(
        target,
        arrays,
        metadata=meta,
        schema=schema,
        array_payload=array_payload,
        subkey_payloads=subkey_payloads,
    )
    meta.setdefault("artifact_key", manifest.artifact_key)
    meta.setdefault("artifact_manifest", manifest.to_metadata())
    with _open_hdf5_write(target, swmr=use_swmr) as handle:
        handle.attrs["schema"] = str(schema)
        handle.attrs["metadata_json"] = json.dumps(meta, sort_keys=True)
        handle.attrs["checksum_algorithm"] = "sha256"
        handle.attrs["swmr_ready"] = bool(use_swmr)
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
            dset.attrs["sha256"] = str(array_payload[str(name)]["sha256"])
            dset.attrs["compression"] = (
                "" if dset.compression is None else str(dset.compression)
            )
            dset.attrs["chunks_json"] = json.dumps(
                [] if dset.chunks is None else [int(v) for v in dset.chunks]
            )
            names.append(str(name))
        handle.attrs["array_names_json"] = json.dumps(names, sort_keys=True)
        if use_swmr:
            handle.flush()
            handle.swmr_mode = True
            handle.flush()
    return target


def write_large_cache_hdf5_artifact(
    path: str | Path,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    compression: str | None = "gzip",
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    swmr: bool | None = None,
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
        swmr=_resolve_hdf5_swmr(swmr, default=True),
    )


def read_hdf5_artifact(
    path: str | Path,
    *,
    lazy: bool = False,
    verify_checksums: bool = True,
    swmr: bool | None = None,
) -> HDF5Artifact:
    """Read an HDF5 artifact eagerly or as lazy dataset handles."""

    source = Path(path)
    use_swmr = _resolve_hdf5_swmr(swmr, default=False)
    with _open_hdf5_read(source, swmr=use_swmr) as handle:
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
                        _dataset_info(source, key, dataset, swmr=use_swmr),
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


def _resolve_hdf5_swmr(value: bool | None, *, default: bool) -> bool:
    """Resolve SWMR mode from an explicit value plus the process override env."""

    if value is not None:
        return bool(value)
    raw = os.getenv("PYEIDORS_HDF5_SWMR", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _open_hdf5_write(path: str | Path, *, swmr: bool):
    kwargs = {"libver": "latest"} if swmr else {}
    return h5py.File(Path(path), "w", **kwargs)


def _open_hdf5_read(path: str | Path, *, swmr: bool):
    if not swmr:
        return h5py.File(Path(path), "r")
    try:
        return h5py.File(Path(path), "r", libver="latest", swmr=True)
    except OSError:
        return h5py.File(Path(path), "r")


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
            if not digest:
                continue
            actual = _dataset_array_digest(dataset)
            if actual != digest:
                raise ValueError(
                    f"HDF5 dataset checksum mismatch for {key!r}: {actual} != {digest}"
                )
            verified[key] = digest
    return verified


def migrate_npz_to_hdf5(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    schema: str = DEFAULT_SCHEMA,
) -> Path:
    """Copy a legacy NumPy artifact into HDF5 without modifying the source."""

    source = Path(src)
    target = Path(dst) if dst is not None else source.with_suffix(".h5")
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            arrays = {str(name): np.asarray(payload[name]) for name in payload.files}
    elif suffix == ".npy":
        arrays = {"array": np.asarray(np.load(source, allow_pickle=False))}
    else:
        raise ValueError(f"legacy NumPy artifact must be .npz or .npy: {source}")
    meta = {
        "migrated_from": str(source),
        "legacy_format": suffix.lstrip("."),
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


def _dataset_info(
    path: Path, name: str, dataset: h5py.Dataset, *, swmr: bool = False
) -> HDF5DatasetInfo:
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
        swmr=bool(swmr),
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
    arr = np.asarray(value)
    if arr.dtype.kind in {"S", "U", "O"}:
        arr = _canonical_string_array(arr)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode())
    digest.update(b"|")
    digest.update(json.dumps([int(v) for v in arr.shape]).encode())
    digest.update(b"|")
    _update_digest_with_array_bytes(digest, arr)
    return digest.hexdigest()


def _update_digest_with_array_bytes(digest: Any, arr: np.ndarray) -> None:
    if not arr.nbytes:
        return
    if arr.flags.c_contiguous:
        digest.update(memoryview(arr).cast("B"))
        return
    if arr.ndim == 0:
        scalar = np.ascontiguousarray(arr)
        if scalar.nbytes:
            digest.update(memoryview(scalar).cast("B"))
        return

    trailing = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
    row_bytes = max(int(arr.dtype.itemsize) * max(trailing, 1), 1)
    rows_per_chunk = max(1, int(_DATASET_DIGEST_CHUNK_BYTES) // row_bytes)
    for start in range(0, arr.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, arr.shape[0])
        chunk = np.ascontiguousarray(arr[start:stop])
        if chunk.nbytes:
            digest.update(memoryview(chunk).cast("B"))


def _dataset_array_digest(dataset: h5py.Dataset) -> str:
    dtype = np.dtype(dataset.dtype)
    if dtype.kind in {"S", "U", "O"}:
        return _array_digest(dataset[()])

    shape = tuple(int(v) for v in dataset.shape)
    digest = hashlib.sha256()
    digest.update(str(dtype).encode())
    digest.update(b"|")
    digest.update(json.dumps([int(v) for v in shape]).encode())
    digest.update(b"|")
    if not shape:
        scalar = np.ascontiguousarray(dataset[()])
        if scalar.nbytes:
            digest.update(memoryview(scalar).cast("B"))
        return digest.hexdigest()
    if any(dim == 0 for dim in shape):
        return digest.hexdigest()

    trailing = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    row_bytes = max(int(dtype.itemsize) * max(trailing, 1), 1)
    rows_per_chunk = max(1, int(_DATASET_DIGEST_CHUNK_BYTES) // row_bytes)
    for start in range(0, shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, shape[0])
        chunk = np.ascontiguousarray(dataset[start:stop])
        if chunk.nbytes:
            digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _hdf5_disk_artifact_manifest(
    target: Path,
    arrays: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    schema: str,
    array_payload: Mapping[str, Mapping[str, Any]] | None = None,
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> DiskArtifactManifest:
    payload = (
        {str(key): dict(value) for key, value in array_payload.items()}
        if array_payload is not None
        else _hdf5_manifest_array_payload_from_arrays(arrays)
    )
    metadata_payload = {
        str(key): val
        for key, val in dict(metadata).items()
        if str(key) not in _MANIFEST_METADATA_KEYS
    }
    return build_disk_artifact_manifest(
        "hdf5-artifact",
        {
            "schema": str(schema),
            "arrays": payload,
            "metadata": metadata_payload,
        },
        files={"artifact": target},
        metadata={"artifact_format": "hdf5"},
        subkey_payloads=subkey_payloads,
    )


def _hdf5_manifest_array_payload_from_arrays(
    arrays: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(name): {
            "shape": [int(dim) for dim in np.asarray(value).shape],
            "dtype": str(np.asarray(value).dtype),
            "sha256": _array_digest(value),
        }
        for name, value in sorted(arrays.items(), key=lambda item: str(item[0]))
        if value is not None
    }


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
            "sha256": _dataset_sha256(dataset) or _dataset_array_digest(dataset),
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
