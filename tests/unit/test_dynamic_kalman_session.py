from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np

import eit_app.dynamic_kalman_runtime as dynamic_kalman_runtime
from eit_app.backend_worker_protocol import (
    read_reconstruction_result,
    write_reconstruction_result,
)
from eit_app.controllers.reconstruction_controller import ReconstructionResult
from eit_app.dynamic_kalman_runtime import apply_dynamic_kalman_to_reconstruction
from pyeidors.inverse.dynamic_session import (
    DYNAMIC_DIAGONAL_SESSION_SCHEMA,
    DYNAMIC_MEASUREMENT_DIAGONAL_SESSION_SCHEMA,
    DiagonalKalmanConfig,
    PersistentDiagonalKalmanRegistry,
    PersistentDiagonalKalmanSession,
    PersistentMeasurementDiagonalKalmanSession,
)


def test_v675_session_initializes_without_changing_first_image() -> None:
    observation = np.array([0.2, 1.0, -0.5], dtype=np.float64)
    session = PersistentDiagonalKalmanSession(
        fingerprint="route=noser;mesh=0.08;ref=0",
        config=DiagonalKalmanConfig(upstream_latency_frames=2),
    )

    update = session.update(observation, block_number=10, timestamp=1.0)

    np.testing.assert_array_equal(update.state, observation)
    assert update.metadata["schema"] == DYNAMIC_DIAGONAL_SESSION_SCHEMA
    assert update.metadata["action"] == "initialize"
    assert update.metadata["online_fixed_lag_frames"] == 0
    assert update.metadata["total_latency_frames"] == 2


def test_v675_noncandidate_step_is_updated_not_rejected() -> None:
    session = PersistentDiagonalKalmanSession(
        fingerprint="plant-step",
        config=DiagonalKalmanConfig(
            innovation_gate="reject",
            innovation_nis_threshold_per_dof=0.1,
        ),
    )
    session.update(np.zeros(4), block_number=1)

    update = session.update(
        np.ones(4),
        block_number=2,
        innovation_candidate=False,
    )

    assert update.metadata["innovation_nis_per_dof"] > 0.1
    assert update.metadata["action"] == "update"
    assert np.all(update.state > 0.0)


def test_v675_candidate_spike_can_reject_measurement_update() -> None:
    session = PersistentDiagonalKalmanSession(
        fingerprint="isolated-spike",
        config=DiagonalKalmanConfig(
            innovation_gate="reject",
            innovation_nis_threshold_per_dof=1.0,
        ),
    )
    session.update(np.zeros(3), block_number=4)

    update = session.update(
        np.full(3, 100.0),
        block_number=5,
        innovation_candidate=True,
    )

    assert update.metadata["action"] == "reject"
    np.testing.assert_allclose(update.state, 0.0)


def test_v675_registry_isolates_resets_and_evicts_sessions() -> None:
    registry = PersistentDiagonalKalmanRegistry(max_sessions=2)
    config = DiagonalKalmanConfig()
    first = registry.update(
        "set-a:ref0",
        np.array([1.0]),
        fingerprint="a0",
        config=config,
        block_number=1,
    )
    registry.update(
        "set-b:ref0",
        np.array([5.0]),
        fingerprint="b0",
        config=config,
        block_number=1,
    )
    reset = registry.update(
        "set-a:ref0",
        np.array([2.0]),
        fingerprint="a0",
        config=config,
        block_number=9,
        reset=True,
    )
    registry.update(
        "set-c:ref0",
        np.array([8.0]),
        fingerprint="c0",
        config=config,
        block_number=1,
    )

    assert first.metadata["registry_action"] == "created"
    assert reset.metadata["registry_action"] == "reset"
    np.testing.assert_array_equal(reset.state, np.array([2.0]))
    assert registry.session_count == 2
    assert registry.eviction_count == 1
    assert "set-b:ref0" not in registry.session_ids


def test_v675_worker_runtime_applies_session_and_preserves_raw_image() -> None:
    metadata = {
        "dynamic_kalman_enabled": True,
        "dynamic_kalman_session_id": "runtime-test-session",
        "dynamic_kalman_fingerprint": "route=noser;mesh=0.08;ref=0",
        "dynamic_kalman_reset": True,
        "dynamic_kalman_upstream_latency_frames": 2,
        "block_number": 1,
        "measurement_weights": [1.0] * 208,
    }
    request = SimpleNamespace(
        metadata=metadata,
        target_frame=SimpleNamespace(frame_index=1, timestamp=10.0),
    )
    result = ReconstructionResult(
        conductivity=np.array([1.0, 2.0]),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
    )

    filtered = apply_dynamic_kalman_to_reconstruction(request, result)

    np.testing.assert_array_equal(filtered.raw_conductivity, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(filtered.conductivity, np.array([1.0, 2.0]))
    assert filtered.metadata["dynamic_kalman"]["applied"] is True
    assert filtered.metadata["dynamic_kalman"]["total_latency_frames"] == 2


def test_v675_result_hdf5_roundtrip_keeps_raw_and_dynamic_diagnostics(tmp_path) -> None:
    path = tmp_path / "dynamic-result.h5"
    result = ReconstructionResult(
        conductivity=np.array([0.8, 1.2]),
        raw_conductivity=np.array([0.5, 1.5]),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
        metadata={
            "dynamic_kalman": {
                "applied": True,
                "action": "update",
                "innovation_nis_per_dof": 2.5,
                "kalman_gain_mean": 0.75,
                "variance_inflation": 1.0,
                "update_count": 3,
                "total_latency_frames": 2,
            }
        },
    )

    write_reconstruction_result(path, result)
    loaded = read_reconstruction_result(path)

    np.testing.assert_array_equal(loaded.conductivity, result.conductivity)
    np.testing.assert_array_equal(loaded.raw_conductivity, result.raw_conductivity)
    assert loaded.metadata["dynamic_kalman"]["innovation_nis_per_dof"] == 2.5


def test_v675_backend_worker_exposes_dynamic_session_lifecycle_commands(
    monkeypatch,
    capsys,
) -> None:
    import eit_app.backend_worker as worker

    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "id": "kalman-status-1",
                    "command": "dynamic_kalman_status",
                }
            )
            + "\n"
        ),
    )

    assert worker._serve(SimpleNamespace()) == 0

    messages = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert messages[0]["id"] == "kalman-status-1"
    assert messages[0]["status"] == "ok"
    assert messages[0]["metadata"]["command"] == "status"
    assert "session_count" in messages[0]["metadata"]
    assert "session_modes" in messages[0]["metadata"]


def test_v676_measurement_session_updates_from_cached_jacobian() -> None:
    session = PersistentMeasurementDiagonalKalmanSession(
        fingerprint="measurement-model",
        config=DiagonalKalmanConfig(
            process_noise_relative_std=0.2,
            measurement_noise_relative_std=0.05,
        ),
    )
    model = np.eye(2, dtype=np.float64)
    session.update(
        np.zeros(2),
        np.zeros(2),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.ones(2),
        block_number=1,
    )

    update = session.update(
        np.array([2.0, -1.0]),
        np.array([2.0, -1.0]),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.ones(2),
        block_number=2,
    )

    assert update.metadata["schema"] == DYNAMIC_MEASUREMENT_DIAGONAL_SESSION_SCHEMA
    assert update.metadata["effective_mode"] == "measurement"
    assert update.metadata["measurement_size"] == 2
    assert update.metadata["solve_seconds"] >= 0.0
    assert update.state[0] > 0.0
    assert update.state[1] < 0.0


def test_v676_measurement_candidate_gate_and_channel_weight_are_respected() -> None:
    config = DiagonalKalmanConfig(
        process_noise_relative_std=0.1,
        measurement_noise_relative_std=0.05,
        innovation_gate="reject",
        innovation_nis_threshold_per_dof=0.1,
    )
    rejected_session = PersistentMeasurementDiagonalKalmanSession(
        fingerprint="candidate-reject",
        config=config,
    )
    model = np.eye(2, dtype=np.float64)
    rejected_session.update(
        np.zeros(2),
        np.zeros(2),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.ones(2),
        block_number=1,
    )
    rejected = rejected_session.update(
        np.full(2, 100.0),
        np.full(2, 100.0),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.ones(2),
        block_number=2,
        innovation_candidate=True,
    )
    assert rejected.metadata["action"] == "reject"
    np.testing.assert_allclose(rejected.state, 0.0)

    weighted_session = PersistentMeasurementDiagonalKalmanSession(
        fingerprint="channel-weights",
        config=DiagonalKalmanConfig(
            process_noise_relative_std=0.2,
            measurement_noise_relative_std=0.05,
        ),
    )
    weighted_session.update(
        np.zeros(2),
        np.zeros(2),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.ones(2),
        block_number=1,
    )
    weighted = weighted_session.update(
        np.array([1.0, 0.0]),
        np.ones(2),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.array([1.0, 0.0]),
        block_number=2,
    )
    assert weighted.state[0] > weighted.state[1] * 100.0


def test_v682_measurement_session_fuses_same_frame_static_noser_anchor() -> None:
    session = PersistentMeasurementDiagonalKalmanSession(
        fingerprint="noser-anchor",
        config=DiagonalKalmanConfig(),
    )
    model = np.eye(2, dtype=np.float64)
    session.update(
        np.zeros(2),
        np.zeros(2),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.ones(2),
        block_number=1,
    )

    static_noser = np.array([0.1, -0.1], dtype=np.float64)
    update = session.update(
        static_noser,
        np.array([10.0, -10.0], dtype=np.float64),
        model,
        measurement_scale=np.ones(2),
        measurement_weights=np.ones(2),
        block_number=2,
    )

    assert update.metadata["static_noser_anchor_applied"] is True
    assert update.metadata["static_noser_anchor_gain_min"] >= 0.75
    assert np.max(np.abs(update.state - static_noser)) <= 0.05


def test_v677_runtime_auto_prefers_noser_preserving_fast_image() -> None:
    metadata = {
        "dynamic_kalman_enabled": True,
        "dynamic_kalman_mode": "auto",
        "dynamic_kalman_session_id": "runtime-measurement-session",
        "dynamic_kalman_fingerprint": "route=noser;mode=auto;ref=0",
        "dynamic_kalman_reset": True,
        "dynamic_kalman_upstream_latency_frames": 2,
        "block_number": 1,
        "measurement_weights": [1.0, 1.0],
    }
    request = SimpleNamespace(
        metadata=metadata,
        target_frame=SimpleNamespace(frame_index=1, timestamp=10.0),
    )
    result = ReconstructionResult(
        conductivity=np.array([1.1, 0.9]),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
        dynamic_observation_model=np.eye(2),
        dynamic_observation=np.array([0.1, -0.1]),
        dynamic_measurement_scale=np.ones(2),
        dynamic_state_offset=1.0,
    )

    filtered = apply_dynamic_kalman_to_reconstruction(request, result)

    np.testing.assert_array_equal(filtered.raw_conductivity, np.array([1.1, 0.9]))
    np.testing.assert_allclose(filtered.conductivity, np.array([1.1, 0.9]))
    dynamic = filtered.metadata["dynamic_kalman"]
    assert dynamic["requested_mode"] == "auto"
    assert dynamic["effective_mode"] == "fast_image"
    assert dynamic["fallback_reason"] == ""


def test_v682_runtime_divergence_guard_returns_static_and_resets_session(
    monkeypatch,
) -> None:
    class DivergentRegistry:
        def __init__(self) -> None:
            self.reset_sessions: list[str] = []

        def update_measurement(self, session_id, *_args, **_kwargs):
            return SimpleNamespace(
                state=np.array([4.0, -4.0], dtype=np.float64),
                metadata={
                    "applied": True,
                    "action": "update",
                    "effective_mode": "measurement",
                    "innovation_nis_per_dof": 0.01,
                    "kalman_gain_mean": 0.1,
                    "variance_inflation": 1.0,
                    "update_count": 2,
                    "total_latency_frames": 2,
                },
            )

        def update(self, *_args, **_kwargs):
            raise AssertionError("explicit measurement mode must not silently continue")

        def reset(self, session_id: str) -> bool:
            self.reset_sessions.append(session_id)
            return True

    registry = DivergentRegistry()
    monkeypatch.setattr(dynamic_kalman_runtime, "_REGISTRY", registry)
    metadata = {
        "dynamic_kalman_enabled": True,
        "dynamic_kalman_mode": "measurement",
        "dynamic_kalman_session_id": "runtime-divergence-session",
        "dynamic_kalman_fingerprint": "route=noser;mode=measurement;ref=0",
        "dynamic_kalman_reset": False,
        "dynamic_kalman_upstream_latency_frames": 2,
        "block_number": 2,
        "measurement_weights": [1.0, 1.0],
    }
    request = SimpleNamespace(
        metadata=metadata,
        target_frame=SimpleNamespace(frame_index=2, timestamp=11.0),
    )
    raw = np.array([1.02, 0.98], dtype=np.float64)
    result = ReconstructionResult(
        conductivity=raw.copy(),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
        dynamic_observation_model=np.eye(2),
        dynamic_observation=np.array([0.02, -0.02]),
        dynamic_measurement_scale=np.ones(2),
        dynamic_state_offset=1.0,
    )

    filtered = dynamic_kalman_runtime.apply_dynamic_kalman_to_reconstruction(
        request,
        result,
    )

    np.testing.assert_array_equal(filtered.conductivity, raw)
    dynamic = filtered.metadata["dynamic_kalman"]
    assert dynamic["action"] == "static_guard_reset"
    assert dynamic["fallback_reason"].startswith("spatial_guard:")
    assert dynamic["spatial_guard_triggered"] is True
    assert registry.reset_sessions == ["runtime-divergence-session"]


def test_v677_runtime_auto_needs_no_measurement_context() -> None:
    metadata = {
        "dynamic_kalman_enabled": True,
        "dynamic_kalman_mode": "auto",
        "dynamic_kalman_session_id": "runtime-auto-fallback-session",
        "dynamic_kalman_fingerprint": "route=noser;mode=auto;ref=0",
        "dynamic_kalman_reset": True,
        "dynamic_kalman_upstream_latency_frames": 2,
        "block_number": 1,
        "measurement_weights": [1.0] * 208,
    }
    request = SimpleNamespace(
        metadata=metadata,
        target_frame=SimpleNamespace(frame_index=1, timestamp=10.0),
    )
    result = ReconstructionResult(
        conductivity=np.array([1.0, 2.0]),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
    )

    filtered = apply_dynamic_kalman_to_reconstruction(request, result)

    dynamic = filtered.metadata["dynamic_kalman"]
    assert dynamic["effective_mode"] == "fast_image"
    assert dynamic["fallback_reason"] == ""
    assert dynamic["mode_selection"] == "auto_safe_image"
    assert dynamic["total_latency_frames"] == 2


def test_v677_runtime_explicit_measurement_respects_state_product_budget() -> None:
    metadata = {
        "dynamic_kalman_enabled": True,
        "dynamic_kalman_mode": "measurement",
        "dynamic_kalman_max_measurement_state_product": 3,
        "dynamic_kalman_session_id": "runtime-budget-fallback-session",
        "dynamic_kalman_fingerprint": "route=noser;mode=auto;budget=3",
        "dynamic_kalman_reset": True,
        "dynamic_kalman_upstream_latency_frames": 2,
        "block_number": 1,
        "measurement_weights": [1.0, 1.0],
    }
    request = SimpleNamespace(
        metadata=metadata,
        target_frame=SimpleNamespace(frame_index=1, timestamp=10.0),
    )
    result = ReconstructionResult(
        conductivity=np.array([1.1, 0.9]),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
        dynamic_observation_model=np.eye(2),
        dynamic_observation=np.array([0.1, -0.1]),
        dynamic_measurement_scale=np.ones(2),
        dynamic_state_offset=1.0,
    )

    filtered = apply_dynamic_kalman_to_reconstruction(request, result)

    dynamic = filtered.metadata["dynamic_kalman"]
    assert dynamic["effective_mode"] == "fast_image"
    assert dynamic["fallback_reason"] == "measurement_state_product_budget:4>3"


def test_v678_ephemeral_measurement_context_is_not_written_to_result_hdf5(
    tmp_path,
) -> None:
    path = tmp_path / "ephemeral-context.h5"
    result = ReconstructionResult(
        conductivity=np.array([1.0, 2.0]),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
        dynamic_observation_model=np.eye(2),
        dynamic_observation=np.ones(2),
        dynamic_measurement_scale=np.ones(2),
        dynamic_state_offset=1.0,
        metadata={
            "dynamic_kalman": {
                "applied": True,
                "effective_mode": "measurement",
                "action": "update",
                "solve_seconds": 0.012,
                "total_latency_frames": 2,
            }
        },
    )

    write_reconstruction_result(path, result)
    loaded = read_reconstruction_result(path)

    assert loaded.dynamic_observation_model is None
    with h5py.File(path, "r") as handle:
        assert "dynamic_observation_model" not in handle
        assert int(handle["dynamic_kalman_mode_code"][()]) == 1
        assert float(handle["dynamic_kalman_solve_seconds"][()]) == 0.012


def test_v682_static_guard_reset_action_is_traceable_in_result_hdf5(tmp_path) -> None:
    path = tmp_path / "dynamic-static-guard.h5"
    result = ReconstructionResult(
        conductivity=np.array([0.99, 1.01]),
        raw_conductivity=np.array([0.99, 1.01]),
        node_coords=np.zeros((2, 2)),
        cell_connectivity=np.zeros((1, 3), dtype=np.int32),
        metadata={
            "dynamic_kalman": {
                "applied": True,
                "effective_mode": "measurement",
                "action": "static_guard_reset",
                "fallback_reason": "spatial_guard:rms_ratio=10",
                "total_latency_frames": 2,
            }
        },
    )

    write_reconstruction_result(path, result)

    with h5py.File(path, "r") as handle:
        assert int(handle["dynamic_kalman_action_code"][()]) == 4
        assert int(handle["dynamic_kalman_fallback_code"][()]) == 1
