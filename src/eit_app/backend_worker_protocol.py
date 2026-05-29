"""HDF5 protocol for profile-isolated GUI backend workers."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from eit_app.controllers.forward_solver_controller import (
        ForwardSolverRequest,
        ForwardSolverResult,
    )
    from eit_app.controllers.reconstruction_controller import (
        ReconstructionRequest,
        ReconstructionResult,
    )
    from eit_app.models.frame_model import FrameData

_FORWARD_REQUEST_SCHEMA = "eit_app_forward_request_h5_v1"
_FORWARD_RESULT_SCHEMA = "eit_app_forward_result_h5_v1"
_RECONSTRUCTION_REQUEST_SCHEMA = "eit_app_reconstruction_request_h5_v1"
_RECONSTRUCTION_RESULT_SCHEMA = "eit_app_reconstruction_result_h5_v1"
_DEFAULT_DATASET_COMPRESSION = "lzf"
_DEFAULT_DATASET_SHUFFLE = True
_DEFAULT_DATASET_CHUNK_BYTES = 1024 * 1024


def _encode_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": value.tolist(),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return _encode_json_value(value.item())
    if isinstance(value, complex):
        return {"__complex__": [float(value.real), float(value.imag)]}
    if isinstance(value, dict):
        return {str(key): _encode_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_json_value(item) for item in value]
    return value


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "__complex__" in value:
            real, imag = value["__complex__"]
            return complex(float(real), float(imag))
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=np.dtype(value["dtype"]))
        return {str(key): _decode_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _json_attr(obj: Any, key: str, default: Any = None) -> Any:
    raw = obj.attrs.get(key)
    if raw is None:
        return default
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _dataset_compression() -> str | None:
    raw = (
        os.getenv(
            "EIT_APP_BACKEND_WORKER_HDF5_COMPRESSION",
            _DEFAULT_DATASET_COMPRESSION,
        )
        .strip()
        .lower()
    )
    if raw in {"", "none", "off", "false", "no", "0", "uncompressed"}:
        return None
    if raw in {"lzf", "gzip"}:
        return raw
    return _DEFAULT_DATASET_COMPRESSION


def _dataset_shuffle(arr: np.ndarray, compression: str | None) -> bool:
    if not compression or arr.ndim == 0 or arr.size == 0:
        return False
    raw = (
        os.getenv(
            "EIT_APP_BACKEND_WORKER_HDF5_SHUFFLE",
            "1" if _DEFAULT_DATASET_SHUFFLE else "0",
        )
        .strip()
        .lower()
    )
    if raw in {"", "none", "off", "false", "no", "0", "disabled"}:
        return False
    return arr.dtype.kind in {"b", "i", "u", "f", "c"}


def _dataset_chunk_bytes() -> int:
    raw = os.getenv("EIT_APP_BACKEND_WORKER_HDF5_CHUNK_BYTES", "").strip().lower()
    if raw in {"", "default"}:
        return _DEFAULT_DATASET_CHUNK_BYTES
    if raw in {"none", "off", "false", "no", "0", "disabled"}:
        return 0
    try:
        value = int(float(raw))
    except ValueError:
        return _DEFAULT_DATASET_CHUNK_BYTES
    return max(0, value)


def _dataset_chunks(arr: np.ndarray, compression: str | None) -> tuple[int, ...] | None:
    if not compression or arr.ndim == 0 or arr.size == 0 or arr.dtype.hasobject:
        return None
    target_bytes = _dataset_chunk_bytes()
    if target_bytes <= 0:
        return None
    itemsize = max(1, int(arr.dtype.itemsize))
    if arr.ndim == 1:
        chunk_len = max(1, min(int(arr.shape[0]), target_bytes // itemsize))
        return (chunk_len,)

    row_items = int(np.prod(arr.shape[1:], dtype=np.int64))
    row_bytes = max(1, row_items * itemsize)
    chunk_rows = max(1, min(int(arr.shape[0]), target_bytes // row_bytes))
    return (chunk_rows, *tuple(int(size) for size in arr.shape[1:]))


def _write_dataset(group: h5py.Group, name: str, value: Any) -> None:
    arr = np.asarray(value)
    compression = _dataset_compression() if arr.ndim > 0 and arr.size > 0 else None
    kwargs = {"compression": compression} if compression else {}
    chunks = _dataset_chunks(arr, compression)
    if chunks is not None:
        kwargs["chunks"] = chunks
    if _dataset_shuffle(arr, compression):
        kwargs["shuffle"] = True
    dataset = group.create_dataset(name, data=arr, **kwargs)
    dataset.attrs["compression"] = compression or "none"
    dataset.attrs["shuffle"] = bool(dataset.shuffle)
    dataset.attrs["chunk_bytes_target"] = (
        int(_dataset_chunk_bytes()) if compression else 0
    )
    dataset.attrs["chunk_shape"] = np.asarray(dataset.chunks or (), dtype=np.int64)


def _read_dataset_array(dataset: h5py.Dataset) -> np.ndarray:
    if dataset.shape == ():
        return np.asarray(dataset[()])
    out = np.empty(dataset.shape, dtype=dataset.dtype, order="C")
    dataset.read_direct(out)
    return out


def forward_request_to_payload(request: ForwardSolverRequest) -> dict[str, Any]:
    return {
        "mesh_dimension": int(request.mesh_dimension),
        "mesh_refinement": float(request.mesh_refinement),
        "n_electrodes": int(request.n_electrodes),
        "background_conductivity": _encode_json_value(request.background_conductivity),
        "inhomogeneities": [
            _encode_json_value(asdict(spec)) for spec in request.inhomogeneities
        ],
        "noise_level": float(request.noise_level),
        "forward_model_config": _encode_json_value(request.forward_model_config),
    }


def forward_request_from_payload(payload: dict[str, Any]) -> ForwardSolverRequest:
    from eit_app.controllers.forward_solver_controller import ForwardSolverRequest
    from eit_app.models.simulation_state import InhomogeneitySpec

    raw = _decode_json_value(dict(payload))
    return ForwardSolverRequest(
        mesh_dimension=int(raw.get("mesh_dimension", 2)),
        mesh_refinement=float(raw.get("mesh_refinement", 0.1)),
        n_electrodes=int(raw.get("n_electrodes", 16)),
        background_conductivity=raw.get("background_conductivity", 1.0),
        inhomogeneities=[
            InhomogeneitySpec(**dict(spec))
            for spec in list(raw.get("inhomogeneities", []))
        ],
        noise_level=float(raw.get("noise_level", 0.0)),
        forward_model_config=dict(raw.get("forward_model_config") or {}),
    )


def write_forward_request(path: str | Path, request: ForwardSolverRequest) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = _FORWARD_REQUEST_SCHEMA
        handle.attrs["payload_json"] = _json_dumps(forward_request_to_payload(request))


def read_forward_request(path: str | Path) -> ForwardSolverRequest:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return forward_request_from_payload(json.loads(path.read_text("utf-8")))
    with h5py.File(path, "r") as handle:
        return forward_request_from_payload(_json_attr(handle, "payload_json", {}))


def write_forward_result(path: str | Path, result: ForwardSolverResult) -> None:
    metadata = {
        "n_elements": int(result.n_elements),
        "n_measurements": int(result.n_measurements),
        "forward_model_config": _encode_json_value(result.forward_model_config),
        "error_msg": result.error_msg,
        "has_homogeneous_voltages": result.homogeneous_voltages is not None,
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = _FORWARD_RESULT_SCHEMA
        handle.attrs["metadata_json"] = _json_dumps(metadata)
        _write_dataset(handle, "boundary_voltages", result.boundary_voltages)
        _write_dataset(
            handle,
            "ground_truth_conductivity",
            result.ground_truth_conductivity,
        )
        _write_dataset(handle, "node_coords", result.node_coords)
        _write_dataset(handle, "cell_connectivity", result.cell_connectivity)
        if result.homogeneous_voltages is not None:
            _write_dataset(handle, "homogeneous_voltages", result.homogeneous_voltages)


def read_forward_result(path: str | Path) -> ForwardSolverResult:
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult

    with h5py.File(path, "r") as data:
        metadata = _decode_json_value(_json_attr(data, "metadata_json", {}))
        has_homogeneous = bool(metadata.get("has_homogeneous_voltages", False))
        homogeneous = (
            _read_dataset_array(data["homogeneous_voltages"])
            if has_homogeneous and "homogeneous_voltages" in data
            else None
        )
        return ForwardSolverResult(
            boundary_voltages=_read_dataset_array(data["boundary_voltages"]),
            ground_truth_conductivity=_read_dataset_array(
                data["ground_truth_conductivity"]
            ),
            node_coords=_read_dataset_array(data["node_coords"]),
            cell_connectivity=_read_dataset_array(data["cell_connectivity"]),
            n_elements=int(metadata["n_elements"]),
            n_measurements=int(metadata["n_measurements"]),
            homogeneous_voltages=homogeneous,
            forward_model_config=dict(metadata.get("forward_model_config") or {}),
            error_msg=metadata.get("error_msg"),
        )


def frame_to_payload(frame: FrameData) -> dict[str, Any]:
    return {
        "real": _encode_json_value(frame.real),
        "imag": _encode_json_value(frame.imag),
        "timestamp": float(frame.timestamp),
        "frame_index": int(frame.frame_index),
        "metadata": _encode_json_value(frame.metadata),
    }


def frame_from_payload(payload: dict[str, Any]) -> FrameData:
    from eit_app.models.frame_model import FrameData

    raw = _decode_json_value(dict(payload))
    return FrameData(
        real=np.asarray(raw.get("real", [])),
        imag=np.asarray(raw.get("imag", [])),
        timestamp=float(raw.get("timestamp", 0.0)),
        frame_index=int(raw.get("frame_index", 0)),
        metadata=dict(raw.get("metadata") or {}),
    )


def _write_frame(group: h5py.Group, frame: FrameData) -> None:
    group.attrs["timestamp"] = float(frame.timestamp)
    group.attrs["frame_index"] = int(frame.frame_index)
    group.attrs["metadata_json"] = _json_dumps(_encode_json_value(frame.metadata))
    _write_dataset(group, "real", frame.real)
    _write_dataset(group, "imag", frame.imag)


def _read_frame(group: h5py.Group) -> FrameData:
    from eit_app.models.frame_model import FrameData

    return FrameData(
        real=_read_dataset_array(group["real"]),
        imag=_read_dataset_array(group["imag"]),
        timestamp=float(group.attrs.get("timestamp", 0.0)),
        frame_index=int(group.attrs.get("frame_index", 0)),
        metadata=dict(_decode_json_value(_json_attr(group, "metadata_json", {})) or {}),
    )


def reconstruction_request_to_payload(
    request: ReconstructionRequest,
) -> dict[str, Any]:
    return {
        "reference_frame": frame_to_payload(request.reference_frame),
        "target_frame": frame_to_payload(request.target_frame),
        "use_part": str(request.use_part),
        "method": str(request.method),
        "regularization_alpha": float(request.regularization_alpha),
        "max_iterations": int(request.max_iterations),
        "mesh_dimension": int(request.mesh_dimension),
        "mesh_refinement": float(request.mesh_refinement),
        "metadata": _encode_json_value(request.metadata),
    }


def reconstruction_request_from_payload(
    payload: dict[str, Any],
) -> ReconstructionRequest:
    from eit_app.controllers.reconstruction_controller import ReconstructionRequest

    raw = _decode_json_value(dict(payload))
    return ReconstructionRequest(
        reference_frame=frame_from_payload(dict(raw["reference_frame"])),
        target_frame=frame_from_payload(dict(raw["target_frame"])),
        use_part=str(raw.get("use_part", "real")),
        method=str(raw.get("method", "gn-difference")),
        regularization_alpha=float(raw.get("regularization_alpha", 1.0)),
        max_iterations=int(raw.get("max_iterations", 10)),
        mesh_dimension=int(raw.get("mesh_dimension", 2)),
        mesh_refinement=float(raw.get("mesh_refinement", 4.0)),
        metadata=dict(raw.get("metadata") or {}),
    )


def write_reconstruction_request(
    path: str | Path,
    request: ReconstructionRequest,
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = _RECONSTRUCTION_REQUEST_SCHEMA
        handle.attrs["use_part"] = str(request.use_part)
        handle.attrs["method"] = str(request.method)
        handle.attrs["regularization_alpha"] = float(request.regularization_alpha)
        handle.attrs["max_iterations"] = int(request.max_iterations)
        handle.attrs["mesh_dimension"] = int(request.mesh_dimension)
        handle.attrs["mesh_refinement"] = float(request.mesh_refinement)
        handle.attrs["metadata_json"] = _json_dumps(
            _encode_json_value(request.metadata)
        )
        _write_frame(handle.create_group("reference_frame"), request.reference_frame)
        _write_frame(handle.create_group("target_frame"), request.target_frame)


def read_reconstruction_request(path: str | Path) -> ReconstructionRequest:
    from eit_app.controllers.reconstruction_controller import ReconstructionRequest

    path = Path(path)
    if path.suffix.lower() == ".json":
        return reconstruction_request_from_payload(json.loads(path.read_text("utf-8")))
    with h5py.File(path, "r") as handle:
        return ReconstructionRequest(
            reference_frame=_read_frame(handle["reference_frame"]),
            target_frame=_read_frame(handle["target_frame"]),
            use_part=str(handle.attrs.get("use_part", "real")),
            method=str(handle.attrs.get("method", "gn-difference")),
            regularization_alpha=float(handle.attrs.get("regularization_alpha", 1.0)),
            max_iterations=int(handle.attrs.get("max_iterations", 10)),
            mesh_dimension=int(handle.attrs.get("mesh_dimension", 2)),
            mesh_refinement=float(handle.attrs.get("mesh_refinement", 4.0)),
            metadata=dict(
                _decode_json_value(_json_attr(handle, "metadata_json", {})) or {}
            ),
        )


def write_reconstruction_result(path: str | Path, result: ReconstructionResult) -> None:
    metadata = {
        "error_msg": result.error_msg,
        "metadata": _encode_json_value(result.metadata),
        "has_measured": result.measured is not None,
        "has_simulated": result.simulated is not None,
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = _RECONSTRUCTION_RESULT_SCHEMA
        handle.attrs["metadata_json"] = _json_dumps(metadata)
        _write_dataset(handle, "conductivity", result.conductivity)
        _write_dataset(handle, "node_coords", result.node_coords)
        _write_dataset(handle, "cell_connectivity", result.cell_connectivity)
        if result.measured is not None:
            _write_dataset(handle, "measured", result.measured)
        if result.simulated is not None:
            _write_dataset(handle, "simulated", result.simulated)


def read_reconstruction_result(path: str | Path) -> ReconstructionResult:
    from eit_app.controllers.reconstruction_controller import ReconstructionResult

    with h5py.File(path, "r") as data:
        metadata = _decode_json_value(_json_attr(data, "metadata_json", {}))
        has_measured = bool(metadata.get("has_measured", False))
        has_simulated = bool(metadata.get("has_simulated", False))
        measured = (
            _read_dataset_array(data["measured"])
            if has_measured and "measured" in data
            else None
        )
        simulated = (
            _read_dataset_array(data["simulated"])
            if has_simulated and "simulated" in data
            else None
        )
        return ReconstructionResult(
            conductivity=_read_dataset_array(data["conductivity"]),
            node_coords=_read_dataset_array(data["node_coords"]),
            cell_connectivity=_read_dataset_array(data["cell_connectivity"]),
            measured=measured,
            simulated=simulated,
            error_msg=metadata.get("error_msg"),
            metadata=dict(metadata.get("metadata") or {}),
        )
