"""Persistent O(n) diagonal Kalman sessions for realtime EIT images."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


DYNAMIC_DIAGONAL_SESSION_SCHEMA = "pyeidors-dynamic-diagonal-session-v1"
DYNAMIC_MEASUREMENT_DIAGONAL_SESSION_SCHEMA = (
    "pyeidors-dynamic-measurement-diagonal-session-v2"
)


@dataclass(frozen=True)
class DiagonalKalmanConfig:
    process_noise_relative_std: float = 0.15
    measurement_noise_relative_std: float = 0.10
    initial_relative_std: float = 0.50
    transition_decay_per_block: float = 1.0
    minimum_scale: float = 1.0e-6
    minimum_variance: float = 1.0e-16
    maximum_variance: float = 1.0e12
    measurement_weight_floor: float = 1.0e-6
    innovation_gate: str = "inflate"
    innovation_nis_threshold_per_dof: float = 9.0
    innovation_max_variance_inflation: float = 100.0
    static_noser_anchor_relative_std: float = 0.10
    static_noser_anchor_minimum_gain: float = 0.75
    upstream_latency_frames: int = 2

    def __post_init__(self) -> None:
        positive = {
            "process_noise_relative_std": self.process_noise_relative_std,
            "measurement_noise_relative_std": self.measurement_noise_relative_std,
            "initial_relative_std": self.initial_relative_std,
            "minimum_scale": self.minimum_scale,
            "minimum_variance": self.minimum_variance,
            "maximum_variance": self.maximum_variance,
            "measurement_weight_floor": self.measurement_weight_floor,
            "innovation_nis_threshold_per_dof": self.innovation_nis_threshold_per_dof,
            "innovation_max_variance_inflation": self.innovation_max_variance_inflation,
            "static_noser_anchor_relative_std": self.static_noser_anchor_relative_std,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.transition_decay_per_block) or not (
            0.0 < float(self.transition_decay_per_block) <= 1.0
        ):
            raise ValueError("transition_decay_per_block must be in (0, 1].")
        if self.maximum_variance < self.minimum_variance:
            raise ValueError("maximum_variance must be >= minimum_variance.")
        if self.innovation_max_variance_inflation < 1.0:
            raise ValueError("innovation_max_variance_inflation must be >= 1.")
        if not np.isfinite(self.static_noser_anchor_minimum_gain) or not (
            0.0 <= float(self.static_noser_anchor_minimum_gain) <= 1.0
        ):
            raise ValueError("static_noser_anchor_minimum_gain must be in [0, 1].")
        if int(self.upstream_latency_frames) < 0:
            raise ValueError("upstream_latency_frames must be nonnegative.")
        object.__setattr__(self, "innovation_gate", _gate_policy(self.innovation_gate))
        object.__setattr__(
            self,
            "upstream_latency_frames",
            int(self.upstream_latency_frames),
        )


@dataclass(frozen=True)
class DiagonalKalmanUpdate:
    state: np.ndarray
    raw_observation: np.ndarray
    metadata: Mapping[str, Any]


class PersistentDiagonalKalmanSession:
    def __init__(self, *, fingerprint: str, config: DiagonalKalmanConfig) -> None:
        self.fingerprint = _nonempty_text(fingerprint, name="fingerprint")
        self.config = config
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._last_block_number: int | None = None
        self._last_timestamp: float | None = None
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def state_size(self) -> int:
        return 0 if self._state is None else int(self._state.size)

    def update(
        self,
        observation: Any,
        *,
        block_number: int,
        timestamp: float | None = None,
        measurement_confidence: float = 1.0,
        innovation_candidate: bool = False,
    ) -> DiagonalKalmanUpdate:
        raw = _state_vector(observation)
        block = int(block_number)
        time_value = _optional_finite_timestamp(timestamp)
        confidence = float(measurement_confidence)
        if not np.isfinite(confidence) or not (0.0 < confidence <= 1.0):
            raise ValueError("measurement_confidence must be finite and in (0, 1].")

        if self._state is None:
            scale = _state_scale(raw, None, minimum=self.config.minimum_scale)
            covariance = np.square(self.config.initial_relative_std * scale)
            self._state = raw.copy()
            self._covariance = np.clip(
                covariance,
                self.config.minimum_variance,
                self.config.maximum_variance,
            )
            self._last_block_number = block
            self._last_timestamp = time_value
            self._update_count = 1
            return self._result(
                raw,
                action="initialize",
                nis_per_dof=0.0,
                gain_mean=1.0,
                variance_inflation=1.0,
                block_step=1,
                measurement_confidence=confidence,
                innovation_candidate=bool(innovation_candidate),
            )

        assert self._covariance is not None
        assert self._last_block_number is not None
        if raw.size != self._state.size:
            raise ValueError(
                f"observation size {raw.size} does not match session state {self._state.size}."
            )
        if block <= self._last_block_number:
            raise ValueError("block_number must increase within a Kalman session.")

        block_step = block - self._last_block_number
        transition = float(self.config.transition_decay_per_block) ** block_step
        predicted = transition * self._state
        scale = _state_scale(raw, predicted, minimum=self.config.minimum_scale)
        process_variance = np.square(
            self.config.process_noise_relative_std * scale
        ) * float(block_step)
        predicted_covariance = (
            transition * transition * self._covariance + process_variance
        )
        measurement_variance = (
            np.square(self.config.measurement_noise_relative_std * scale) / confidence
        )
        innovation = raw - predicted
        innovation_variance = predicted_covariance + measurement_variance
        nis_per_dof = float(np.mean(np.square(innovation) / innovation_variance))
        gate_triggered = bool(
            innovation_candidate
            and self.config.innovation_gate != "none"
            and nis_per_dof > self.config.innovation_nis_threshold_per_dof
        )
        action = "update"
        variance_inflation = 1.0
        if gate_triggered and self.config.innovation_gate == "reject":
            state = predicted
            covariance = predicted_covariance
            gain = np.zeros_like(raw)
            action = "reject"
        else:
            if gate_triggered and self.config.innovation_gate == "inflate":
                variance_inflation = min(
                    self.config.innovation_max_variance_inflation,
                    max(
                        1.0,
                        nis_per_dof / self.config.innovation_nis_threshold_per_dof,
                    ),
                )
                measurement_variance = measurement_variance * variance_inflation
                innovation_variance = predicted_covariance + measurement_variance
                action = "inflate"
            gain = predicted_covariance / innovation_variance
            state = predicted + gain * innovation
            covariance = (
                np.square(1.0 - gain) * predicted_covariance
                + np.square(gain) * measurement_variance
            )

        self._state = np.ascontiguousarray(state, dtype=np.float64)
        self._covariance = np.ascontiguousarray(
            np.clip(
                covariance,
                self.config.minimum_variance,
                self.config.maximum_variance,
            ),
            dtype=np.float64,
        )
        self._last_block_number = block
        self._last_timestamp = time_value
        self._update_count += 1
        return self._result(
            raw,
            action=action,
            nis_per_dof=nis_per_dof,
            gain_mean=float(np.mean(gain)),
            variance_inflation=variance_inflation,
            block_step=block_step,
            measurement_confidence=confidence,
            innovation_candidate=bool(innovation_candidate),
        )

    def _result(
        self,
        raw: np.ndarray,
        *,
        action: str,
        nis_per_dof: float,
        gain_mean: float,
        variance_inflation: float,
        block_step: int,
        measurement_confidence: float,
        innovation_candidate: bool,
    ) -> DiagonalKalmanUpdate:
        assert self._state is not None
        total_latency = int(self.config.upstream_latency_frames)
        metadata = MappingProxyType(
            {
                "schema": DYNAMIC_DIAGONAL_SESSION_SCHEMA,
                "applied": True,
                "algorithm": "persistent-diagonal-kalman",
                "observation_model": "identity-on-static-rm-image",
                "covariance_model": "diagonal-relative",
                "fingerprint": self.fingerprint,
                "state_size": int(self._state.size),
                "update_count": int(self._update_count),
                "block_number": int(self._last_block_number),
                "block_step": int(block_step),
                "timestamp": self._last_timestamp,
                "action": action,
                "innovation_candidate": bool(innovation_candidate),
                "innovation_nis_per_dof": float(nis_per_dof),
                "innovation_nis_threshold_per_dof": float(
                    self.config.innovation_nis_threshold_per_dof
                ),
                "variance_inflation": float(variance_inflation),
                "kalman_gain_mean": float(gain_mean),
                "measurement_confidence": float(measurement_confidence),
                "online_fixed_lag_frames": 0,
                "upstream_centered_latency_frames": total_latency,
                "total_latency_frames": total_latency,
                "process_noise_relative_std": float(
                    self.config.process_noise_relative_std
                ),
                "measurement_noise_relative_std": float(
                    self.config.measurement_noise_relative_std
                ),
            }
        )
        return DiagonalKalmanUpdate(
            state=self._state.copy(),
            raw_observation=raw.copy(),
            metadata=metadata,
        )


class PersistentMeasurementDiagonalKalmanSession:
    """Ordered linear Kalman session using cached J and diagonal state covariance."""

    def __init__(self, *, fingerprint: str, config: DiagonalKalmanConfig) -> None:
        self.fingerprint = _nonempty_text(fingerprint, name="fingerprint")
        self.config = config
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._last_block_number: int | None = None
        self._last_timestamp: float | None = None
        self._model_shape: tuple[int, int] | None = None
        self._update_count = 0

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def state_size(self) -> int:
        return 0 if self._state is None else int(self._state.size)

    def update(
        self,
        static_observation: Any,
        measurement: Any,
        observation_model: Any,
        *,
        measurement_scale: Any | None,
        measurement_weights: Any | None,
        block_number: int,
        timestamp: float | None = None,
        innovation_candidate: bool = False,
    ) -> DiagonalKalmanUpdate:
        raw = _state_vector(static_observation)
        measured = _measurement_vector(measurement)
        model = _observation_matrix(
            observation_model,
            n_measurements=measured.size,
            n_state=raw.size,
        )
        scales = _measurement_scales(
            measurement_scale,
            measured,
            minimum=self.config.minimum_scale,
        )
        weights = _measurement_weights(
            measurement_weights,
            n_measurements=measured.size,
        )
        block = int(block_number)
        time_value = _optional_finite_timestamp(timestamp)

        if self._state is None:
            state_scale = _state_scale(raw, None, minimum=self.config.minimum_scale)
            covariance = np.square(self.config.initial_relative_std * state_scale)
            self._state = raw.copy()
            self._covariance = np.clip(
                covariance,
                self.config.minimum_variance,
                self.config.maximum_variance,
            )
            self._last_block_number = block
            self._last_timestamp = time_value
            self._model_shape = model.shape
            self._update_count = 1
            return self._result(
                raw,
                action="initialize",
                nis_per_dof=0.0,
                gain_mean=1.0,
                variance_inflation=1.0,
                block_step=1,
                innovation_candidate=bool(innovation_candidate),
                n_measurements=measured.size,
                solve_seconds=0.0,
                factor_jitter=0.0,
                static_anchor_applied=True,
                static_anchor_gain_mean=1.0,
                static_anchor_gain_min=1.0,
            )

        assert self._covariance is not None
        assert self._last_block_number is not None
        if raw.size != self._state.size or model.shape != self._model_shape:
            raise ValueError("measurement Kalman model shape changed within a session.")
        if block <= self._last_block_number:
            raise ValueError("block_number must increase within a Kalman session.")

        started = perf_counter()
        block_step = block - self._last_block_number
        transition = float(self.config.transition_decay_per_block) ** block_step
        predicted = transition * self._state
        state_scale = _state_scale(raw, predicted, minimum=self.config.minimum_scale)
        process_variance = np.square(
            self.config.process_noise_relative_std * state_scale
        ) * float(block_step)
        predicted_covariance = np.clip(
            transition * transition * self._covariance + process_variance,
            self.config.minimum_variance,
            self.config.maximum_variance,
        )
        safe_weights = np.maximum(weights, self.config.measurement_weight_floor)
        measurement_variance = (
            np.square(self.config.measurement_noise_relative_std * scales)
            / safe_weights
        )
        innovation = measured - model @ predicted
        weighted_model = model * predicted_covariance[np.newaxis, :]
        innovation_covariance = weighted_model @ model.T
        diagonal = np.diag_indices(measured.size)
        innovation_covariance[diagonal] += measurement_variance
        factor, factor_jitter = _factor_spd(innovation_covariance)
        normalized_innovation = _solve_factored(factor, innovation)
        nis_per_dof = float(
            max(0.0, np.vdot(innovation, normalized_innovation).real) / measured.size
        )
        gate_triggered = bool(
            innovation_candidate
            and self.config.innovation_gate != "none"
            and nis_per_dof > self.config.innovation_nis_threshold_per_dof
        )
        action = "update"
        variance_inflation = 1.0
        if gate_triggered and self.config.innovation_gate == "reject":
            state = predicted
            covariance = predicted_covariance
            gain_mean = 0.0
            action = "reject"
        else:
            if gate_triggered and self.config.innovation_gate == "inflate":
                variance_inflation = min(
                    self.config.innovation_max_variance_inflation,
                    max(
                        1.0,
                        nis_per_dof / self.config.innovation_nis_threshold_per_dof,
                    ),
                )
                innovation_covariance[diagonal] += measurement_variance * (
                    variance_inflation - 1.0
                )
                factor, factor_jitter = _factor_spd(innovation_covariance)
                normalized_innovation = _solve_factored(factor, innovation)
                action = "inflate"
            state = predicted + predicted_covariance * (model.T @ normalized_innovation)
            solved_weighted_model = _solve_factored(factor, weighted_model)
            covariance_reduction = np.einsum(
                "ij,ij->j",
                weighted_model,
                solved_weighted_model,
                optimize=True,
            )
            covariance = predicted_covariance - covariance_reduction
            gain_mean = float(
                np.mean(
                    np.clip(
                        covariance_reduction / predicted_covariance,
                        0.0,
                        1.0,
                    )
                )
            )

        static_anchor_applied = action != "reject"
        static_anchor_gain_mean = 0.0
        static_anchor_gain_min = 0.0
        if static_anchor_applied:
            state, covariance, anchor_gain = _fuse_static_noser_anchor(
                state,
                covariance,
                raw,
                config=self.config,
            )
            static_anchor_gain_mean = float(np.mean(anchor_gain))
            static_anchor_gain_min = float(np.min(anchor_gain))

        self._state = np.ascontiguousarray(state, dtype=np.float64)
        self._covariance = np.ascontiguousarray(
            np.clip(
                covariance,
                self.config.minimum_variance,
                self.config.maximum_variance,
            ),
            dtype=np.float64,
        )
        self._last_block_number = block
        self._last_timestamp = time_value
        self._update_count += 1
        return self._result(
            raw,
            action=action,
            nis_per_dof=nis_per_dof,
            gain_mean=gain_mean,
            variance_inflation=variance_inflation,
            block_step=block_step,
            innovation_candidate=bool(innovation_candidate),
            n_measurements=measured.size,
            solve_seconds=perf_counter() - started,
            factor_jitter=factor_jitter,
            static_anchor_applied=static_anchor_applied,
            static_anchor_gain_mean=static_anchor_gain_mean,
            static_anchor_gain_min=static_anchor_gain_min,
        )

    def _result(
        self,
        raw: np.ndarray,
        *,
        action: str,
        nis_per_dof: float,
        gain_mean: float,
        variance_inflation: float,
        block_step: int,
        innovation_candidate: bool,
        n_measurements: int,
        solve_seconds: float,
        factor_jitter: float,
        static_anchor_applied: bool,
        static_anchor_gain_mean: float,
        static_anchor_gain_min: float,
    ) -> DiagonalKalmanUpdate:
        assert self._state is not None
        metadata = MappingProxyType(
            {
                "schema": DYNAMIC_MEASUREMENT_DIAGONAL_SESSION_SCHEMA,
                "applied": True,
                "algorithm": "persistent-measurement-diagonal-kalman",
                "effective_mode": "measurement",
                "observation_model": "cached-jacobian-measurement-space",
                "covariance_model": "diagonal-state-measurement-update",
                "fingerprint": self.fingerprint,
                "state_size": int(self._state.size),
                "measurement_size": int(n_measurements),
                "update_count": int(self._update_count),
                "block_number": int(self._last_block_number),
                "block_step": int(block_step),
                "timestamp": self._last_timestamp,
                "action": action,
                "innovation_candidate": bool(innovation_candidate),
                "innovation_nis_per_dof": float(nis_per_dof),
                "innovation_nis_threshold_per_dof": float(
                    self.config.innovation_nis_threshold_per_dof
                ),
                "variance_inflation": float(variance_inflation),
                "kalman_gain_mean": float(gain_mean),
                "measurement_weight_floor": float(self.config.measurement_weight_floor),
                "online_fixed_lag_frames": 0,
                "upstream_centered_latency_frames": int(
                    self.config.upstream_latency_frames
                ),
                "total_latency_frames": int(self.config.upstream_latency_frames),
                "solve_seconds": float(solve_seconds),
                "factor_jitter": float(factor_jitter),
                "static_noser_anchor_applied": bool(static_anchor_applied),
                "static_noser_anchor_gain_mean": float(static_anchor_gain_mean),
                "static_noser_anchor_gain_min": float(static_anchor_gain_min),
                "static_noser_anchor_relative_std": float(
                    self.config.static_noser_anchor_relative_std
                ),
            }
        )
        return DiagonalKalmanUpdate(
            state=self._state.copy(),
            raw_observation=raw.copy(),
            metadata=metadata,
        )


class PersistentDiagonalKalmanRegistry:
    def __init__(self, *, max_sessions: int = 16) -> None:
        if int(max_sessions) <= 0:
            raise ValueError("max_sessions must be positive.")
        self.max_sessions = int(max_sessions)
        self._sessions: OrderedDict[
            str,
            PersistentDiagonalKalmanSession
            | PersistentMeasurementDiagonalKalmanSession,
        ] = OrderedDict()
        self.eviction_count = 0

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    @property
    def session_modes(self) -> dict[str, str]:
        return {
            session_id: (
                "measurement"
                if isinstance(session, PersistentMeasurementDiagonalKalmanSession)
                else "fast_image"
            )
            for session_id, session in self._sessions.items()
        }

    def update(
        self,
        session_id: str,
        observation: Any,
        *,
        fingerprint: str,
        config: DiagonalKalmanConfig,
        block_number: int,
        timestamp: float | None = None,
        measurement_confidence: float = 1.0,
        innovation_candidate: bool = False,
        reset: bool = False,
    ) -> DiagonalKalmanUpdate:
        key = _nonempty_text(session_id, name="session_id")
        registry_action = "reused"
        if reset and key in self._sessions:
            del self._sessions[key]
            registry_action = "reset"
        session = self._sessions.get(key)
        if session is not None and (
            not isinstance(session, PersistentDiagonalKalmanSession)
            or session.fingerprint != fingerprint
            or session.config != config
        ):
            del self._sessions[key]
            session = None
            registry_action = "reconfigured"
        if session is None:
            session = PersistentDiagonalKalmanSession(
                fingerprint=fingerprint,
                config=config,
            )
            self._sessions[key] = session
            if registry_action == "reused":
                registry_action = "created"
        self._sessions.move_to_end(key)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
            self.eviction_count += 1

        result = session.update(
            observation,
            block_number=block_number,
            timestamp=timestamp,
            measurement_confidence=measurement_confidence,
            innovation_candidate=innovation_candidate,
        )
        metadata = dict(result.metadata)
        metadata.update(
            {
                "session_id": key,
                "registry_action": registry_action,
                "registry_session_count": int(self.session_count),
                "registry_eviction_count": int(self.eviction_count),
            }
        )
        return DiagonalKalmanUpdate(
            state=result.state,
            raw_observation=result.raw_observation,
            metadata=MappingProxyType(metadata),
        )

    def update_measurement(
        self,
        session_id: str,
        static_observation: Any,
        measurement: Any,
        observation_model: Any,
        *,
        fingerprint: str,
        config: DiagonalKalmanConfig,
        measurement_scale: Any | None,
        measurement_weights: Any | None,
        block_number: int,
        timestamp: float | None = None,
        innovation_candidate: bool = False,
        reset: bool = False,
    ) -> DiagonalKalmanUpdate:
        key = _nonempty_text(session_id, name="session_id")
        registry_action = "reused"
        if reset and key in self._sessions:
            del self._sessions[key]
            registry_action = "reset"
        session = self._sessions.get(key)
        if session is not None and (
            not isinstance(session, PersistentMeasurementDiagonalKalmanSession)
            or session.fingerprint != fingerprint
            or session.config != config
        ):
            del self._sessions[key]
            session = None
            registry_action = "reconfigured"
        if session is None:
            session = PersistentMeasurementDiagonalKalmanSession(
                fingerprint=fingerprint,
                config=config,
            )
            self._sessions[key] = session
            if registry_action == "reused":
                registry_action = "created"
        self._sessions.move_to_end(key)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
            self.eviction_count += 1

        result = session.update(
            static_observation,
            measurement,
            observation_model,
            measurement_scale=measurement_scale,
            measurement_weights=measurement_weights,
            block_number=block_number,
            timestamp=timestamp,
            innovation_candidate=innovation_candidate,
        )
        metadata = dict(result.metadata)
        metadata.update(
            {
                "session_id": key,
                "registry_action": registry_action,
                "registry_session_count": int(self.session_count),
                "registry_eviction_count": int(self.eviction_count),
            }
        )
        return DiagonalKalmanUpdate(
            state=result.state,
            raw_observation=result.raw_observation,
            metadata=MappingProxyType(metadata),
        )

    def reset(self, session_id: str) -> bool:
        return self._sessions.pop(str(session_id), None) is not None

    def close(self, session_id: str) -> bool:
        return self.reset(session_id)

    def clear(self) -> int:
        count = len(self._sessions)
        self._sessions.clear()
        return count


def _state_vector(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        if not np.allclose(raw.imag, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("diagonal realtime Kalman requires a real image state.")
        raw = raw.real
    vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("observation must be a non-empty finite vector.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _measurement_vector(value: Any) -> np.ndarray:
    vector = _state_vector(value)
    if vector.size == 0:
        raise ValueError("measurement must be non-empty.")
    return vector


def _observation_matrix(
    value: Any,
    *,
    n_measurements: int,
    n_state: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        if not np.allclose(raw.imag, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("measurement Kalman currently requires a real Jacobian.")
        raw = raw.real
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (n_measurements, n_state):
        raise ValueError(
            "observation_model shape must equal (measurement size, state size)."
        )
    if not _all_finite_values(matrix):
        raise ValueError("observation_model must contain finite values only.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _measurement_scales(
    value: Any | None,
    measured: np.ndarray,
    *,
    minimum: float,
) -> np.ndarray:
    if value is None:
        floor = max(float(minimum), 0.05 * float(np.median(np.abs(measured))))
        return np.maximum(np.abs(measured), floor)
    scales = _measurement_vector(value)
    if scales.size != measured.size:
        raise ValueError("measurement_scale size must match measurement size.")
    if np.any(scales <= 0.0):
        raise ValueError("measurement_scale values must be positive.")
    return np.maximum(scales, float(minimum))


def _measurement_weights(value: Any | None, *, n_measurements: int) -> np.ndarray:
    if value is None:
        return np.ones(n_measurements, dtype=np.float64)
    weights = _measurement_vector(value)
    if weights.size != n_measurements:
        raise ValueError("measurement_weights size must match measurement size.")
    if np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("measurement_weights must be in [0, 1].")
    return weights


def _fuse_static_noser_anchor(
    state: np.ndarray,
    covariance: np.ndarray,
    static_observation: np.ndarray,
    *,
    config: DiagonalKalmanConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_scale = _state_scale(
        static_observation,
        None,
        minimum=config.minimum_scale,
    )
    anchor_variance = np.square(config.static_noser_anchor_relative_std * anchor_scale)
    anchor_gain = covariance / (covariance + anchor_variance)
    anchor_gain = np.maximum(
        anchor_gain,
        float(config.static_noser_anchor_minimum_gain),
    )
    anchored_state = state + anchor_gain * (static_observation - state)
    anchored_covariance = (
        np.square(1.0 - anchor_gain) * covariance
        + np.square(anchor_gain) * anchor_variance
    )
    return (
        np.ascontiguousarray(anchored_state, dtype=np.float64),
        np.ascontiguousarray(anchored_covariance, dtype=np.float64),
        np.ascontiguousarray(anchor_gain, dtype=np.float64),
    )


def _factor_spd(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    symmetric = np.asarray(0.5 * (matrix + matrix.T), dtype=np.float64)
    diagonal_scale = max(
        1.0e-18,
        float(np.median(np.abs(np.diag(symmetric)))),
    )
    jitter = 0.0
    for attempt in range(7):
        try:
            if jitter == 0.0:
                return np.linalg.cholesky(symmetric), 0.0
            candidate = symmetric.copy()
            candidate[np.diag_indices_from(candidate)] += jitter
            return np.linalg.cholesky(candidate), jitter
        except np.linalg.LinAlgError:
            jitter = diagonal_scale * (10.0 ** (-12 + attempt))
    raise np.linalg.LinAlgError("innovation covariance is not positive definite.")


def _solve_factored(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


def _all_finite_values(value: np.ndarray, *, chunk_size: int = 65_536) -> bool:
    flat = np.asarray(value).reshape(-1)
    for start in range(0, flat.size, chunk_size):
        if not np.isfinite(flat[start : start + chunk_size]).all():
            return False
    return True


def _state_scale(
    observation: np.ndarray,
    predicted: np.ndarray | None,
    *,
    minimum: float,
) -> np.ndarray:
    global_floor = max(
        float(minimum),
        0.05 * float(np.median(np.abs(observation))),
    )
    scale = np.maximum(np.abs(observation), global_floor)
    if predicted is not None:
        scale = np.maximum(scale, np.abs(predicted))
    return np.ascontiguousarray(scale, dtype=np.float64)


def _optional_finite_timestamp(value: float | None) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError("timestamp must be finite when provided.")
    return resolved


def _gate_policy(value: Any) -> str:
    resolved = str(value).strip().lower().replace("-", "_")
    aliases = {"": "none", "off": "none", "hard": "reject", "variance": "inflate"}
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"none", "reject", "inflate"}:
        raise ValueError("innovation_gate must be 'none', 'reject', or 'inflate'.")
    return resolved


def _nonempty_text(value: Any, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


__all__ = [
    "DYNAMIC_DIAGONAL_SESSION_SCHEMA",
    "DYNAMIC_MEASUREMENT_DIAGONAL_SESSION_SCHEMA",
    "DiagonalKalmanConfig",
    "DiagonalKalmanUpdate",
    "PersistentDiagonalKalmanRegistry",
    "PersistentDiagonalKalmanSession",
    "PersistentMeasurementDiagonalKalmanSession",
]
