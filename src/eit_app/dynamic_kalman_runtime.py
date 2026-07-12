"""Backend-worker integration for persistent realtime diagonal Kalman sessions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pyeidors.inverse.dynamic_session import (
    DiagonalKalmanConfig,
    PersistentDiagonalKalmanRegistry,
)


_REGISTRY = PersistentDiagonalKalmanRegistry(max_sessions=16)
_DEFAULT_MAX_MEASUREMENT_STATE_PRODUCT = 2_000_000
_DEFAULT_STATIC_GUARD_RMS_RATIO = 1.75
_DEFAULT_STATIC_GUARD_ROBUST_RATIO = 2.0
_DEFAULT_STATIC_GUARD_MINIMUM_DEVIATION_RELATIVE = 0.01


def apply_dynamic_kalman_to_reconstruction(request: Any, result: Any) -> Any:
    metadata = dict(getattr(request, "metadata", {}) or {})
    if not _as_bool(metadata.get("dynamic_kalman_enabled", False)):
        return result
    if getattr(result, "error_msg", None) or np.asarray(result.conductivity).size == 0:
        return result

    result_metadata = dict(getattr(result, "metadata", {}) or {})
    try:
        session_id = _required_text(metadata, "dynamic_kalman_session_id")
        fingerprint = _required_text(metadata, "dynamic_kalman_fingerprint")
        requested_mode = _dynamic_mode(
            metadata.get("dynamic_kalman_mode", "fast_image")
        )
        weights = np.asarray(metadata.get("measurement_weights", [1.0]))
        finite_weights = weights[np.isfinite(weights)]
        confidence = (
            float(np.clip(np.mean(finite_weights), 0.02, 1.0))
            if finite_weights.size
            else 1.0
        )
        config = DiagonalKalmanConfig(
            process_noise_relative_std=float(
                metadata.get("dynamic_kalman_process_noise_relative_std", 0.15)
            ),
            measurement_noise_relative_std=float(
                metadata.get("dynamic_kalman_measurement_noise_relative_std", 0.10)
            ),
            initial_relative_std=float(
                metadata.get("dynamic_kalman_initial_relative_std", 0.50)
            ),
            transition_decay_per_block=float(
                metadata.get("dynamic_kalman_transition_decay_per_block", 1.0)
            ),
            innovation_gate=str(
                metadata.get("dynamic_kalman_innovation_gate", "inflate")
            ),
            innovation_nis_threshold_per_dof=float(
                metadata.get("dynamic_kalman_nis_threshold_per_dof", 9.0)
            ),
            innovation_max_variance_inflation=float(
                metadata.get("dynamic_kalman_max_variance_inflation", 100.0)
            ),
            static_noser_anchor_relative_std=float(
                metadata.get("dynamic_kalman_static_noser_anchor_relative_std", 0.10)
            ),
            static_noser_anchor_minimum_gain=float(
                metadata.get("dynamic_kalman_static_noser_anchor_minimum_gain", 0.75)
            ),
            upstream_latency_frames=int(
                metadata.get("dynamic_kalman_upstream_latency_frames", 2)
            ),
        )
        block_number = int(
            metadata.get(
                "block_number",
                getattr(getattr(request, "target_frame", None), "frame_index", 0),
            )
        )
        timestamp = float(
            getattr(getattr(request, "target_frame", None), "timestamp", 0.0)
        )
        innovation_candidate = _as_bool(
            metadata.get("dynamic_kalman_innovation_candidate", False)
        )
        reset = _as_bool(metadata.get("dynamic_kalman_reset", False))
        raw_conductivity = _real_vector(result.conductivity, name="conductivity")
        update = None
        filtered_conductivity = None
        fallback_reason = ""
        if requested_mode == "measurement":
            context, fallback_reason = _measurement_context(
                result,
                raw_conductivity,
                weights,
                max_product=int(
                    metadata.get(
                        "dynamic_kalman_max_measurement_state_product",
                        _DEFAULT_MAX_MEASUREMENT_STATE_PRODUCT,
                    )
                ),
            )
            if context is not None:
                try:
                    update = _REGISTRY.update_measurement(
                        session_id,
                        context["state"],
                        context["measurement"],
                        context["model"],
                        fingerprint=fingerprint,
                        config=config,
                        measurement_scale=context["measurement_scale"],
                        measurement_weights=context["measurement_weights"],
                        block_number=block_number,
                        timestamp=timestamp,
                        innovation_candidate=innovation_candidate,
                        reset=reset,
                    )
                    filtered_conductivity = update.state + context["state_offset"]
                except Exception as exc:
                    fallback_reason = (
                        f"measurement_update_failed:{type(exc).__name__}:{exc}"
                    )

        if update is None:
            update = _REGISTRY.update(
                session_id,
                raw_conductivity,
                fingerprint=fingerprint,
                config=config,
                block_number=block_number,
                timestamp=timestamp,
                measurement_confidence=confidence,
                innovation_candidate=innovation_candidate,
                reset=reset,
            )
            filtered_conductivity = update.state

        dynamic_metadata = dict(update.metadata)
        effective_mode = str(dynamic_metadata.get("effective_mode", "fast_image"))
        dynamic_metadata.update(
            {
                "requested_mode": requested_mode,
                "effective_mode": effective_mode,
                "fallback_reason": fallback_reason,
                "mode_selection": (
                    "auto_safe_image" if requested_mode == "auto" else "explicit"
                ),
                "max_measurement_state_product": int(
                    metadata.get(
                        "dynamic_kalman_max_measurement_state_product",
                        _DEFAULT_MAX_MEASUREMENT_STATE_PRODUCT,
                    )
                ),
            }
        )
        if effective_mode == "measurement":
            guard = _spatial_stability_guard(
                raw_conductivity,
                np.asarray(filtered_conductivity, dtype=np.float64),
                rms_ratio_limit=float(
                    metadata.get(
                        "dynamic_kalman_static_guard_rms_ratio",
                        _DEFAULT_STATIC_GUARD_RMS_RATIO,
                    )
                ),
                robust_ratio_limit=float(
                    metadata.get(
                        "dynamic_kalman_static_guard_robust_ratio",
                        _DEFAULT_STATIC_GUARD_ROBUST_RATIO,
                    )
                ),
                minimum_deviation_relative=float(
                    metadata.get(
                        "dynamic_kalman_static_guard_minimum_deviation_relative",
                        _DEFAULT_STATIC_GUARD_MINIMUM_DEVIATION_RELATIVE,
                    )
                ),
            )
            dynamic_metadata.update(guard)
            if bool(guard["spatial_guard_triggered"]):
                _REGISTRY.reset(session_id)
                filtered_conductivity = raw_conductivity.copy()
                fallback_reason = f"spatial_guard:{guard['spatial_guard_reason']}"
                dynamic_metadata.update(
                    {
                        "action": "static_guard_reset",
                        "fallback_reason": fallback_reason,
                        "registry_action": "guard_reset",
                    }
                )
        else:
            dynamic_metadata.update(
                {
                    "spatial_guard_triggered": False,
                    "spatial_guard_reason": "not_measurement_mode",
                }
            )
        result.raw_conductivity = raw_conductivity.copy()
        result.conductivity = np.ascontiguousarray(
            filtered_conductivity,
            dtype=np.float64,
        )
        result_metadata["dynamic_kalman"] = dynamic_metadata
    except Exception as exc:
        result_metadata["dynamic_kalman"] = {
            "applied": False,
            "error": str(exc),
            "fallback": "static_weighted_reconstruction",
            "total_latency_frames": int(
                metadata.get("dynamic_kalman_upstream_latency_frames", 2)
            ),
        }
    result.metadata = result_metadata
    return result


def dynamic_kalman_registry_command(
    command: str,
    session_id: str | None = None,
) -> dict[str, object]:
    normalized = str(command).strip().lower().replace("-", "_")
    if normalized == "reset":
        changed = _REGISTRY.reset(_required_session_id(session_id))
    elif normalized == "close":
        changed = _REGISTRY.close(_required_session_id(session_id))
    elif normalized == "clear":
        count = _REGISTRY.clear()
        return {"command": normalized, "cleared": count, "session_count": 0}
    elif normalized == "status":
        changed = False
    else:
        raise ValueError(
            "dynamic Kalman command must be reset, close, clear, or status."
        )
    return {
        "command": normalized,
        "session_id": session_id,
        "changed": changed,
        "session_count": _REGISTRY.session_count,
        "session_ids": list(_REGISTRY.session_ids),
        "session_modes": _REGISTRY.session_modes,
        "eviction_count": _REGISTRY.eviction_count,
    }


def _measurement_context(
    result: Any,
    raw_conductivity: np.ndarray,
    weights: np.ndarray,
    *,
    max_product: int,
) -> tuple[dict[str, Any] | None, str]:
    model_value = getattr(result, "dynamic_observation_model", None)
    measurement_value = getattr(result, "dynamic_observation", None)
    if measurement_value is None:
        measurement_value = getattr(result, "measured", None)
    if model_value is None or measurement_value is None:
        return None, "cached_jacobian_or_measurement_unavailable"
    model = np.asarray(model_value)
    measurement = np.asarray(measurement_value).reshape(-1)
    if model.ndim != 2 or model.shape != (measurement.size, raw_conductivity.size):
        return None, "cached_jacobian_shape_mismatch"
    product = int(model.shape[0]) * int(model.shape[1])
    if max_product <= 0 or product > max_product:
        return None, f"measurement_state_product_budget:{product}>{max_product}"
    if weights.size != measurement.size:
        return None, "measurement_weight_shape_mismatch"
    try:
        offset = _real_scalar(
            getattr(result, "dynamic_state_offset", 0.0),
            name="dynamic_state_offset",
        )
        scale_value = getattr(result, "dynamic_measurement_scale", None)
        measurement_scale = (
            None
            if scale_value is None
            else _real_vector(scale_value, name="dynamic_measurement_scale")
        )
        return (
            {
                "state": raw_conductivity - offset,
                "measurement": measurement,
                "model": model,
                "measurement_scale": measurement_scale,
                "measurement_weights": weights,
                "state_offset": offset,
            },
            "",
        )
    except (TypeError, ValueError) as exc:
        return None, f"measurement_context_invalid:{exc}"


def _dynamic_mode(value: Any) -> str:
    resolved = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "fast_image",
        "fast": "fast_image",
        "image": "fast_image",
        "diagonal": "fast_image",
        "measurement_space": "measurement",
        "advanced": "measurement",
    }
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"auto", "measurement", "fast_image"}:
        raise ValueError(
            "dynamic_kalman_mode must be auto, measurement, or fast_image."
        )
    return resolved


def _spatial_stability_guard(
    raw: np.ndarray,
    filtered: np.ndarray,
    *,
    rms_ratio_limit: float,
    robust_ratio_limit: float,
    minimum_deviation_relative: float,
) -> dict[str, float | bool | str]:
    limits = (rms_ratio_limit, robust_ratio_limit, minimum_deviation_relative)
    if not all(np.isfinite(value) and value > 0.0 for value in limits):
        raise ValueError("dynamic Kalman spatial guard limits must be positive.")
    raw_vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    filtered_vector = np.asarray(filtered, dtype=np.float64).reshape(-1)
    if raw_vector.size == 0 or filtered_vector.shape != raw_vector.shape:
        return {
            "spatial_guard_triggered": True,
            "spatial_guard_reason": "shape_mismatch",
        }
    if not np.isfinite(filtered_vector).all():
        return {
            "spatial_guard_triggered": True,
            "spatial_guard_reason": "nonfinite_filtered_state",
        }

    raw_center = float(np.median(raw_vector))
    filtered_center = float(np.median(filtered_vector))
    raw_spatial = raw_vector - raw_center
    filtered_spatial = filtered_vector - filtered_center
    spatial_delta = filtered_spatial - raw_spatial
    reference_scale = max(
        abs(raw_center),
        float(np.median(np.abs(raw_vector))),
        1.0e-6,
    )
    raw_rms = float(np.sqrt(np.mean(np.square(raw_spatial))))
    filtered_rms = float(np.sqrt(np.mean(np.square(filtered_spatial))))
    deviation_relative = float(
        np.sqrt(np.mean(np.square(spatial_delta))) / reference_scale
    )
    raw_robust = float(np.quantile(np.abs(raw_spatial), 0.995))
    filtered_robust = float(np.quantile(np.abs(filtered_spatial), 0.995))
    rms_ratio = filtered_rms / max(raw_rms, 0.005 * reference_scale, 1.0e-12)
    robust_ratio = filtered_robust / max(
        raw_robust,
        0.01 * reference_scale,
        1.0e-12,
    )
    triggered = bool(
        deviation_relative > minimum_deviation_relative
        and (rms_ratio > rms_ratio_limit or robust_ratio > robust_ratio_limit)
    )
    reason = (
        f"rms_ratio={rms_ratio:.6g};robust_ratio={robust_ratio:.6g};"
        f"deviation_relative={deviation_relative:.6g}"
    )
    return {
        "spatial_guard_triggered": triggered,
        "spatial_guard_reason": reason,
        "spatial_guard_raw_rms": raw_rms,
        "spatial_guard_filtered_rms": filtered_rms,
        "spatial_guard_rms_ratio": rms_ratio,
        "spatial_guard_raw_robust_spread": raw_robust,
        "spatial_guard_filtered_robust_spread": filtered_robust,
        "spatial_guard_robust_ratio": robust_ratio,
        "spatial_guard_deviation_relative": deviation_relative,
        "spatial_guard_rms_ratio_limit": float(rms_ratio_limit),
        "spatial_guard_robust_ratio_limit": float(robust_ratio_limit),
        "spatial_guard_minimum_deviation_relative": float(minimum_deviation_relative),
    }


def _real_vector(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        if not np.allclose(raw.imag, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{name} must be real-valued.")
        raw = raw.real
    vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be non-empty and finite.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _real_scalar(value: Any, *, name: str) -> float:
    vector = _real_vector([value], name=name)
    return float(vector[0])


def _required_text(metadata: Mapping[str, Any], key: str) -> str:
    value = str(metadata.get(key, "")).strip()
    if not value:
        raise ValueError(f"{key} is required when dynamic Kalman is enabled.")
    return value


def _required_session_id(value: str | None) -> str:
    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError("session_id is required.")
    return resolved


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


__all__ = [
    "apply_dynamic_kalman_to_reconstruction",
    "dynamic_kalman_registry_command",
]
