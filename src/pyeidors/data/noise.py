"""EIDORS-style measurement noise helpers."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
import math
from typing import Any

import numpy as np

from pyeidors.utils.numeric_ops import all_finite_values

EIDORS_NOISE_NORM_OPTION = "norm"


def add_noise(
    snr: float,
    v1: Any,
    v2: Any | None = None,
    options: str | None = None,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> Any:
    """Add EIDORS-compatible Gaussian noise at the requested SNR.

    The noise vector is scaled so ``norm(signal) / norm(noise) == snr``.  Signal
    follows EIDORS ``add_noise``:

    - ``v1`` only: ``signal = v1``
    - ``v1, v2``: ``signal = v1 - v2``
    - ``v1, v2, "norm"``: ``signal = (v1 - v2) / v2``

    Array inputs return a noisy ``ndarray``.  Objects with ``.meas`` return a
    shallow metadata-preserving copy with ``meas`` replaced.
    """

    snr_value = _validate_snr(snr)
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")

    arr1, output_template = _extract_measurements(v1, name="v1")
    arr2 = None
    if v2 is not None:
        raw_v2, _ = _extract_measurements(v2, name="v2")
        arr2 = _broadcast_v2(raw_v2, target_shape=arr1.shape)

    signal = _eidors_noise_signal(arr1, arr2, options=options)
    generator = rng if rng is not None else np.random.default_rng(seed)
    noise = _scaled_standard_normal(generator, signal, snr=snr_value)
    noisy = arr1 + noise
    return _restore_output(noisy, output_template)


def _validate_snr(snr: float) -> float:
    value = float(snr)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("snr must be positive and finite")
    return value


def _extract_measurements(value: Any, *, name: str) -> tuple[np.ndarray, Any | None]:
    template = value if hasattr(value, "meas") else None
    meas = value.meas if hasattr(value, "meas") else value
    arr = np.asarray(meas, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not all_finite_values(arr):
        raise FloatingPointError(f"{name} contains non-finite values")
    return arr, template


def _broadcast_v2(v2: np.ndarray, *, target_shape: tuple[int, ...]) -> np.ndarray:
    if v2.shape == target_shape:
        return v2

    if len(target_shape) == 2 and v2.ndim == 1 and v2.shape[0] == target_shape[0]:
        out = np.empty(target_shape, dtype=np.float64)
        out[...] = v2.reshape(-1, 1)
        return out

    out = np.empty(target_shape, dtype=np.float64)
    try:
        np.copyto(out, v2, casting="unsafe")
        return out
    except ValueError as exc:
        raise ValueError(
            "v2 measurements must match v1 shape or be broadcastable to v1 shape: "
            f"{v2.shape!r} vs {target_shape!r}."
        ) from exc


def _eidors_noise_signal(
    v1: np.ndarray, v2: np.ndarray | None, *, options: str | None
) -> np.ndarray:
    if v2 is None:
        if options is not None:
            raise ValueError("options='norm' requires v2")
        signal = v1
    elif options is None:
        signal = v1 - v2
    elif str(options).strip().lower() == EIDORS_NOISE_NORM_OPTION:
        signal = (v1 - v2) / v2
    else:
        raise ValueError("options must be None or 'norm'")

    if not all_finite_values(signal):
        raise FloatingPointError("EIDORS noise signal contains non-finite values")
    return np.asarray(signal, dtype=np.float64)


def _scaled_standard_normal(
    rng: np.random.Generator, signal: np.ndarray, *, snr: float
) -> np.ndarray:
    noise = rng.standard_normal(size=signal.shape)
    signal_norm = float(np.linalg.norm(signal))
    noise_norm = float(np.linalg.norm(noise))
    if signal_norm == 0.0:
        return np.zeros_like(signal, dtype=np.float64)
    if noise_norm == 0.0:
        raise FloatingPointError("random noise vector has zero norm")
    return noise * signal_norm / noise_norm / snr


def _restore_output(noisy: np.ndarray, template: Any | None) -> Any:
    if template is None:
        return noisy
    if is_dataclass(template):
        return replace(template, meas=noisy)
    copied = copy.copy(template)
    copied.meas = noisy
    return copied


__all__ = ["EIDORS_NOISE_NORM_OPTION", "add_noise"]
