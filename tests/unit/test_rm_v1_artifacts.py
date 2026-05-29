"""Tests for RM v1 signatures, hot-path diagnostics, and benchmark artifacts."""

from __future__ import annotations

import hashlib
import inspect
import json

import h5py
import numpy as np
import pytest
from scipy import sparse

import pyeidors.inverse.reconstruction_matrix as rm_module
from pyeidors.inverse import (
    load_rm_artifact,
    migrate_rm_artifact_to_hdf5,
    reconstruct_difference_batch,
    rm_signature,
    rm_signature_payload,
    write_forward_rm_benchmark_artifact,
    write_rm_artifact,
)


def _signature_kwargs() -> dict:
    return {
        "forward_mesh_hash": "fine-mesh-sha256",
        "inverse_mesh_hash": "coarse-grid-sha256",
        "coarse2fine": sparse.eye(3, format="csr"),
        "electrode_geometry": {"count": 48, "rings": [16, 16, 16]},
        "stim_meas_protocol": {"stim": "adjacent", "meas": "skip-current"},
        "background": {"sigma0": 1.0, "z0": 0.01},
        "difference_mode": "normalized",
        "bad_channel_mask": np.array([False, True, False]),
        "noise_covariance": np.diag([1.0, 2.0, 3.0]),
        "regularization_type": "noser",
        "hyperparameters": {"lambda": 0.1, "noise_figure": 0.5},
    }


def test_rm_digest_value_streams_payload_without_tobytes_copy() -> None:
    base = np.array(
        [[1.0, 0.0], [-2.0, 4.0], [3.5, 5.25], [6.0, -7.5]],
        dtype=np.float64,
    )
    array = base[::2].T
    assert not array.flags.c_contiguous
    contiguous = np.ascontiguousarray(array)
    expected = hashlib.sha256(
        str(contiguous.dtype).encode()
        + b"|"
        + json.dumps([int(v) for v in contiguous.shape]).encode()
        + b"|"
        + contiguous.tobytes()
    ).hexdigest()

    assert rm_module._digest_value(array) == expected
    source = inspect.getsource(rm_module._digest_value)
    assert "update_digest_with_array_payload" in source
    assert "np.ascontiguousarray" not in source
    assert ".tobytes(" not in source


def test_rm_reference_frames_broadcasts_vector_without_broadcast_to_copy() -> None:
    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    frames = rm_module._reference_frames(
        reference, n_frames=2, n_measurements=reference.size
    )

    assert frames.shape == (1, reference.size)
    np.testing.assert_allclose(frames[0], reference)
    assert frames.flags.c_contiguous
    assert np.shares_memory(frames, reference)
    source = inspect.getsource(rm_module._reference_frames)
    assert "broadcast_to" not in source
    assert "np.copyto" not in source


def test_rm_signature_is_math_strong_and_device_independent() -> None:
    base = _signature_kwargs()

    cpu = rm_signature(**base, device="cpu", backend="numpy")
    cuda = rm_signature(**base, device="cuda", backend="torch")
    changed_mask = rm_signature(
        **{**base, "bad_channel_mask": np.array([False, False, False])}
    )
    changed_reg = rm_signature(**{**base, "regularization_type": "laplace"})
    changed_graph_ltl = rm_signature(**{**base, "regularization_type": "graph_ltl"})

    assert cpu == cuda
    assert cpu != changed_mask
    assert cpu != changed_reg
    assert changed_reg != changed_graph_ltl
    payload = rm_signature_payload(**base, device="cuda")
    assert "device" not in payload
    assert "backend" not in payload
    assert payload["coarse2fine_hash"]


def test_rm_signature_requires_mesh_and_projection_identity() -> None:
    base = _signature_kwargs()
    with pytest.raises(ValueError, match="forward_mesh_hash"):
        rm_signature(**{**base, "forward_mesh_hash": ""})
    with pytest.raises(ValueError, match="coarse2fine"):
        rm_signature(**{**base, "coarse2fine": None, "coarse2fine_hash": ""})


def test_online_rm_apply_metadata_reports_zero_forward_work() -> None:
    rm = np.eye(3, dtype=float)
    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    frames = np.vstack(
        [
            reference * np.array([1.1, 0.9, 1.2]),
            reference * np.array([0.8, 1.5, 1.0]),
            reference * np.array([1.0, 1.0, 0.75]),
        ]
    )

    result = reconstruct_difference_batch(
        rm,
        frames,
        normalize=True,
        v_ref=reference,
        device="cpu",
        return_metadata=True,
    )

    assert result.metadata["online_hot_path"] == "rm_matmul"
    assert result.metadata["forward_solve_count"] == 0
    assert result.metadata["adjoint_solve_count"] == 0
    assert result.metadata["ksp_solve_count"] == 0
    assert result.metadata["jacobian_rebuild_count"] == 0
    assert result.metadata["n_frames"] == 3


def test_forward_rm_benchmark_artifact_splits_cold_build_and_warm_apply(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pyeidors.inverse.reconstruction_matrix.shutil.which",
        lambda name: "/home/tom/.local/bin/env" if name == "env" else None,
    )
    path = write_forward_rm_benchmark_artifact(
        tmp_path / "forward_rm_benchmark.json",
        offline_rm_build_seconds=12.5,
        online_rm_apply_seconds=0.03,
        metadata={"case": "unit"},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "pyeidors-forward-rm-benchmark-v1"
    assert payload["offline_rm_build_seconds"] == pytest.approx(12.5)
    assert payload["online_rm_apply_seconds"] == pytest.approx(0.03)
    assert payload["online_hot_path"] == "rm_matmul"
    assert payload["env_path"] == "/home/tom/.local/bin/env"
    assert payload["metadata"] == {"case": "unit"}


def test_rm_artifact_hdf5_roundtrip_and_legacy_migration(tmp_path) -> None:
    rm = np.arange(6, dtype=np.float64).reshape(2, 3)
    path = write_rm_artifact(
        tmp_path / "one_step_rm.h5",
        rm,
        metadata={"algorithm": "one-step-noser"},
        voxel_shape=(2, 1, 1),
        channel_mask=np.array([False, True, False]),
        measurement_weights=np.array([1.0, 2.0, 3.0]),
    )

    loaded = load_rm_artifact(path)

    with h5py.File(path, "r") as handle:
        dataset = handle["arrays"]["rm"]
        metadata = json.loads(handle.attrs["metadata_json"])
        assert dataset.chunks == rm.shape
        assert dataset.compression == "lzf"
        assert metadata["rm_hdf5_chunk_layout"] == "row_full_width_v1"
        assert metadata["rm_hdf5_chunk_target_bytes"] == 8 * 1024 * 1024
        assert metadata["rm_hdf5_chunks"] == [2, 3]
        assert metadata["rm_hdf5_compression"] == "lzf"

    assert path.suffix == ".h5"
    assert loaded.schema == "pyeidors-rm-hdf5-v1"
    assert loaded.metadata["artifact_format"] == "hdf5"
    assert loaded.metadata["online_hot_path"] == "rm_matmul"
    assert loaded.voxel_shape == (2, 1, 1)
    np.testing.assert_allclose(loaded.rm, rm)
    np.testing.assert_array_equal(loaded.channel_mask, [False, True, False])
    np.testing.assert_allclose(loaded.measurement_weights, [1.0, 2.0, 3.0])

    legacy_path = tmp_path / "legacy_rm.npz"
    np.savez_compressed(
        legacy_path,
        rm=rm,
        voxel_shape=np.asarray([2, 1, 1], dtype=np.int64),
        metadata_json=np.asarray(json.dumps({"algorithm": "legacy-one-step"})),
    )
    migrated_path = migrate_rm_artifact_to_hdf5(legacy_path)
    migrated = load_rm_artifact(migrated_path)

    assert migrated_path.suffix == ".h5"
    assert migrated.metadata["migrated_from"] == str(legacy_path)
    assert migrated.metadata["legacy_format"] == "npz"
    np.testing.assert_allclose(migrated.rm, rm)


def test_v210_rm_artifact_preserves_float32_node_coords(tmp_path) -> None:
    rm = np.arange(6, dtype=np.float32).reshape(2, 3)
    node_coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    cells = np.array([[0, 1, 2]], dtype=np.int32)

    path = write_rm_artifact(
        tmp_path / "float32_geometry_rm.h5",
        rm,
        node_coords=node_coords,
        cell_connectivity=cells,
    )

    loaded = load_rm_artifact(path)

    assert loaded.rm.dtype == np.dtype(np.float32)
    assert loaded.node_coords is not None
    assert loaded.node_coords.dtype == np.dtype(np.float32)
    assert loaded.cell_connectivity is not None
    assert loaded.cell_connectivity.dtype == np.dtype(np.int32)
    np.testing.assert_allclose(loaded.node_coords, node_coords)
    np.testing.assert_array_equal(loaded.cell_connectivity, cells)


def test_rm_artifact_uses_row_full_width_hdf5_chunks(tmp_path) -> None:
    rm = np.arange(100 * 16, dtype=np.float64).reshape(100, 16)
    path = write_rm_artifact(
        tmp_path / "chunked_rm.h5",
        rm,
        metadata={
            "algorithm": "one-step-noser",
            "rm_hdf5_streaming_chunk_bytes": 16 * 8 * 7,
        },
    )

    with h5py.File(path, "r") as handle:
        dataset = handle["arrays"]["rm"]
        metadata = json.loads(handle.attrs["metadata_json"])
        assert dataset.chunks == (7, 16)
        assert dataset.compression == "lzf"
        assert metadata["rm_hdf5_chunk_layout"] == "row_full_width_v1"
        assert metadata["rm_hdf5_chunk_target_bytes"] == 16 * 8 * 7
        assert metadata["rm_hdf5_chunks"] == [7, 16]
        assert metadata["rm_hdf5_compression"] == "lzf"
