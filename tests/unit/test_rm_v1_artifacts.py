"""Tests for RM v1 signatures, hot-path diagnostics, and benchmark artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import sparse

from pyeidors.inverse import (
    reconstruct_difference_batch,
    rm_signature,
    rm_signature_payload,
    write_forward_rm_benchmark_artifact,
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


def test_rm_signature_is_math_strong_and_device_independent() -> None:
    base = _signature_kwargs()

    cpu = rm_signature(**base, device="cpu", backend="numpy")
    cuda = rm_signature(**base, device="cuda", backend="torch")
    changed_mask = rm_signature(
        **{**base, "bad_channel_mask": np.array([False, False, False])}
    )
    changed_reg = rm_signature(**{**base, "regularization_type": "laplace"})

    assert cpu == cuda
    assert cpu != changed_mask
    assert cpu != changed_reg
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
