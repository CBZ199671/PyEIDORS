#!/usr/bin/env python3
"""Validate dynamic EIT fidelity on synthetic travelling-wave fixtures."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.data.dynamic_sequence import DynamicMeasurementSequence
from pyeidors.inverse import (
    GREIT_METRIC_KEYS,
    VoxelGrid,
    build_one_step_rm,
    graph_laplacian,
    greit_metrics,
    reconstruct_temporal_difference_batch,
)


SCHEMA = "pyeidors-dynamic-validation-benchmark-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=32)
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--n-measurements", type=int, default=20)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.08)
    parser.add_argument("--ridge", type=float, default=1.0e-9)
    parser.add_argument("--noise-std", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--temporal-alpha", type=float, default=0.75)
    parser.add_argument("--onset-fraction", type=float, default=0.30)
    parser.add_argument("--peak-delay-tolerance", type=float, default=0.16)
    parser.add_argument(
        "--no-fail-on-delay",
        action="store_true",
        help="Write the report but return success even when the peak-delay gate fails.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/runtime_benchmarks/dynamic_validation.json"),
    )
    return parser.parse_args(argv)


def run_benchmark(
    *,
    n_cells: int = 32,
    n_frames: int = 32,
    n_measurements: int = 20,
    lambda_: float = 0.08,
    ridge: float = 1.0e-9,
    noise_std: float = 2.0e-3,
    seed: int = 20260424,
    temporal_alpha: float = 0.75,
    onset_fraction: float = 0.30,
    peak_delay_tolerance: float = 0.16,
) -> dict[str, Any]:
    """Run deterministic dynamic validation fixtures."""

    configs = (
        ("travelling_wave", build_travelling_wave_fixture),
        ("plant_slow_pulse", build_plant_slow_pulse_fixture),
    )
    fixtures: dict[str, dict[str, Any]] = {}
    for index, (name, builder) in enumerate(configs):
        fixture = builder(
            n_cells=n_cells,
            n_frames=n_frames,
            n_measurements=n_measurements,
            noise_std=noise_std,
            seed=int(seed) + index * 997,
        )
        fixtures[name] = evaluate_fixture(
            fixture,
            lambda_=lambda_,
            ridge=ridge,
            temporal_alpha=temporal_alpha,
            onset_fraction=onset_fraction,
        )
    gate = evaluate_peak_delay_gate(
        fixtures,
        peak_delay_tolerance=peak_delay_tolerance,
    )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_cells": int(n_cells),
            "n_frames": int(n_frames),
            "n_measurements": int(n_measurements),
            "lambda": float(lambda_),
            "ridge": float(ridge),
            "noise_std": float(noise_std),
            "seed": int(seed),
            "temporal_alpha": float(temporal_alpha),
            "onset_fraction": float(onset_fraction),
            "peak_delay_tolerance": float(peak_delay_tolerance),
        },
        "fixtures": fixtures,
        "summary": summarize_benchmark(fixtures),
        "gate": gate,
    }


def evaluate_fixture(
    fixture: dict[str, Any],
    *,
    lambda_: float,
    ridge: float,
    temporal_alpha: float,
    onset_fraction: float,
) -> dict[str, Any]:
    mesh = fixture["mesh"]
    truth = np.asarray(fixture["truth"], dtype=np.float64)
    clean_measurements = np.asarray(fixture["clean_measurements"], dtype=np.float64)
    noisy_measurements = np.asarray(fixture["measurements"], dtype=np.float64)
    jacobian = np.asarray(fixture["jacobian"], dtype=np.float64)
    times = np.asarray(fixture["times"], dtype=np.float64)
    positions = np.asarray(fixture["positions"], dtype=np.float64)

    sequence = DynamicMeasurementSequence.from_arrays(
        noisy_measurements,
        t=times,
        sampling_rate_hz=fixture["sampling_rate_hz"],
        frame_id=np.arange(times.size, dtype=np.int64),
        reference_policy="difference_measurements_preprojected",
        stim_meas_signature=f"{fixture['name']}:synthetic-jacobian",
        bad_channel_mask=np.zeros(jacobian.shape[0], dtype=bool),
        measurement_weights=np.ones(jacobian.shape[0], dtype=np.float64),
        frequency_hz=fixture["frequency_hz"],
        context_metadata={
            "fixture": fixture["name"],
            "domain": fixture["domain"],
            "dynamic_validation": True,
        },
        data_type="difference",
    )

    prior = graph_laplacian(mesh) + sparse.eye(
        mesh.num_cells(), format="csr", dtype=np.float64
    ) * float(ridge)
    cold_start = time.perf_counter()
    rm = build_one_step_rm(
        jacobian,
        regularization=prior,
        lambda_=lambda_,
        mode="laplace",
        return_metadata=True,
    )
    cold_seconds = time.perf_counter() - cold_start

    methods = {
        "rm_raw": {
            "temporal": "none",
            "exponential_alpha": None,
        },
        "measurement_ema": {
            "temporal": "ema",
            "exponential_alpha": float(temporal_alpha),
        },
    }
    method_payloads: dict[str, dict[str, Any]] = {}
    for method_name, method_cfg in methods.items():
        result = reconstruct_temporal_difference_batch(
            rm.rm,
            sequence.frames,
            normalize=False,
            temporal=str(method_cfg["temporal"]),
            exponential_alpha=float(method_cfg["exponential_alpha"] or 0.5),
            timestamps=sequence.t,
            sample_rate_hz=sequence.sampling_rate_hz,
            channel_mask=sequence.bad_channel_mask[0],
            measurement_weights=sequence.measurement_weights,
            device="cpu",
            return_metadata=True,
        )
        values = np.asarray(result.values, dtype=np.float64)
        method_payloads[method_name] = {
            "mode": "cached_rm_laplace",
            "temporal": method_cfg["temporal"],
            "exponential_alpha": method_cfg["exponential_alpha"],
            "cold_metadata": {
                "offline_rm_build_seconds": float(cold_seconds),
                "rm_shape": [int(v) for v in rm.shape],
                "RtR_signature_hash": rm.metadata["RtR_signature_hash"],
                "regularization_source": rm.metadata["regularization_source"],
            },
            "online_metadata": dict(result.metadata),
            "fidelity": dynamic_fidelity_metrics(
                truth,
                values,
                clean_measurements=clean_measurements,
                noisy_measurements=noisy_measurements,
                positions=positions,
                times=times,
                onset_fraction=onset_fraction,
            ),
        }

    return {
        "name": fixture["name"],
        "domain": fixture["domain"],
        "sequence": sequence.summary(),
        "truth_shape": [int(v) for v in truth.shape],
        "measurement_shape": [int(v) for v in noisy_measurements.shape],
        "methods": method_payloads,
    }


def build_travelling_wave_fixture(
    *,
    n_cells: int,
    n_frames: int,
    n_measurements: int,
    noise_std: float,
    seed: int,
) -> dict[str, Any]:
    _validate_fixture_inputs(n_cells, n_frames, n_measurements, noise_std)
    mesh = VoxelGrid.from_bounds([0.0], [1.0], shape=(int(n_cells),))
    positions = np.asarray(mesh.cell_centers(), dtype=np.float64).reshape(-1)
    times = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)
    centers = 0.16 + 0.68 * times
    width = 0.065
    amplitude = 0.85 + 0.15 * np.sin(np.pi * times)
    truth = np.vstack(
        [
            amp * np.exp(-0.5 * ((positions - center) / width) ** 2)
            for amp, center in zip(amplitude, centers, strict=True)
        ]
    )
    return _fixture_payload(
        name="travelling_wave",
        domain="neural_fast_conduction",
        mesh=mesh,
        positions=positions,
        times=times,
        truth=truth,
        n_measurements=n_measurements,
        frequency_hz=2000.0,
        noise_std=noise_std,
        seed=seed,
    )


def build_plant_slow_pulse_fixture(
    *,
    n_cells: int,
    n_frames: int,
    n_measurements: int,
    noise_std: float,
    seed: int,
) -> dict[str, Any]:
    _validate_fixture_inputs(n_cells, n_frames, n_measurements, noise_std)
    mesh = VoxelGrid.from_bounds([0.0], [1.0], shape=(int(n_cells),))
    positions = np.asarray(mesh.cell_centers(), dtype=np.float64).reshape(-1)
    times = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)
    progress = smoothstep(np.clip((times - 0.08) / 0.84, 0.0, 1.0))
    centers = 0.18 + 0.55 * progress
    width = 0.10
    rise = sigmoid((times - 0.18) / 0.045)
    decay = 1.0 - 0.35 * sigmoid((times - 0.76) / 0.08)
    amplitude = 0.75 * rise * decay
    truth = np.vstack(
        [
            amp * np.exp(-0.5 * ((positions - center) / width) ** 2)
            for amp, center in zip(amplitude, centers, strict=True)
        ]
    )
    return _fixture_payload(
        name="plant_slow_pulse",
        domain="plant_slow_pulse",
        mesh=mesh,
        positions=positions,
        times=times,
        truth=truth,
        n_measurements=n_measurements,
        frequency_hz=250.0,
        noise_std=noise_std,
        seed=seed,
    )


def _fixture_payload(
    *,
    name: str,
    domain: str,
    mesh: VoxelGrid,
    positions: np.ndarray,
    times: np.ndarray,
    truth: np.ndarray,
    n_measurements: int,
    frequency_hz: float,
    noise_std: float,
    seed: int,
) -> dict[str, Any]:
    jacobian = synthetic_measurement_jacobian(
        positions,
        n_measurements=int(n_measurements),
    )
    clean_measurements = np.asarray(truth @ jacobian.T, dtype=np.float64)
    measurements = clean_measurements.copy()
    if noise_std > 0.0:
        rng = np.random.default_rng(int(seed))
        measurements = measurements + rng.normal(
            scale=float(noise_std),
            size=measurements.shape,
        )
    sample_rate = 1.0 / max(float(np.mean(np.diff(times))), np.finfo(np.float64).eps)
    return {
        "name": name,
        "domain": domain,
        "mesh": mesh,
        "positions": np.ascontiguousarray(positions, dtype=np.float64),
        "times": np.ascontiguousarray(times, dtype=np.float64),
        "sampling_rate_hz": float(sample_rate),
        "frequency_hz": float(frequency_hz),
        "truth": np.ascontiguousarray(truth, dtype=np.float64),
        "jacobian": jacobian,
        "clean_measurements": np.ascontiguousarray(
            clean_measurements, dtype=np.float64
        ),
        "measurements": np.ascontiguousarray(measurements, dtype=np.float64),
    }


def synthetic_measurement_jacobian(
    positions: np.ndarray,
    *,
    n_measurements: int,
) -> np.ndarray:
    x = np.asarray(positions, dtype=np.float64).reshape(1, -1)
    sensor_centers = np.linspace(0.0, 1.0, int(n_measurements), dtype=np.float64)
    rows = []
    for idx, center in enumerate(sensor_centers):
        width = 0.10 + 0.045 * ((idx % 4) / 3.0)
        positive = np.exp(-0.5 * ((x - center) / width) ** 2).reshape(-1)
        negative_center = (center + 0.41) % 1.0
        negative = np.exp(-0.5 * ((x - negative_center) / (width * 1.35)) ** 2).reshape(
            -1
        )
        row = positive - 0.32 * negative + 0.04
        row /= max(float(np.linalg.norm(row)), 1.0e-12)
        rows.append(row)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def dynamic_fidelity_metrics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    *,
    clean_measurements: np.ndarray,
    noisy_measurements: np.ndarray,
    positions: np.ndarray,
    times: np.ndarray,
    onset_fraction: float,
) -> dict[str, Any]:
    truth_arr = _frame_array(truth, name="truth")
    recon_arr = _frame_array(reconstruction, name="reconstruction")
    if truth_arr.shape != recon_arr.shape:
        raise ValueError(
            f"truth and reconstruction shape mismatch: {truth_arr.shape} vs {recon_arr.shape}."
        )
    times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    positions_arr = np.asarray(positions, dtype=np.float64).reshape(-1)
    if times_arr.size != truth_arr.shape[0]:
        raise ValueError("times length must match frame count.")
    if positions_arr.size != truth_arr.shape[1]:
        raise ValueError("positions length must match parameter count.")

    error = recon_arr - truth_arr
    center_truth = center_trace(truth_arr, positions_arr)
    center_recon = center_trace(recon_arr, positions_arr)
    speed_truth = speed_from_center_trace(center_truth, times_arr)
    speed_recon = speed_from_center_trace(center_recon, times_arr)
    onset_truth = onset_time_trace(truth_arr, times_arr, fraction=onset_fraction)
    onset_recon = onset_time_trace(recon_arr, times_arr, fraction=onset_fraction)
    peak_truth = peak_time_trace(truth_arr, times_arr)
    peak_recon = peak_time_trace(recon_arr, times_arr)
    active = np.max(truth_arr, axis=0) >= 0.20 * float(np.max(truth_arr))
    onset_delta = onset_recon[active] - onset_truth[active]
    peak_delta = peak_recon[active] - peak_truth[active]
    amplitude_ratio = float(
        np.max(np.abs(recon_arr)) / max(float(np.max(np.abs(truth_arr))), 1.0e-12)
    )
    measurement_snr = snr_db(clean_measurements, noisy_measurements)
    reconstruction_snr = snr_db(truth_arr, recon_arr)
    spatial = spatial_metric_summary(
        truth_arr,
        recon_arr,
        positions=positions_arr,
    )
    metrics: dict[str, Any] = {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_l2_error": float(
            np.linalg.norm(error) / max(float(np.linalg.norm(truth_arr)), 1.0e-12)
        ),
        "onset_time_mean_abs_error": float(
            np.mean(np.abs(onset_delta)) if onset_delta.size else 0.0
        ),
        "onset_time_max_positive_delay": float(
            np.max(np.maximum(onset_delta, 0.0)) if onset_delta.size else 0.0
        ),
        "peak_time_mean_abs_error": float(
            np.mean(np.abs(peak_delta)) if peak_delta.size else 0.0
        ),
        "peak_time_max_positive_delay": float(
            np.max(np.maximum(peak_delta, 0.0)) if peak_delta.size else 0.0
        ),
        "propagation_speed_true": float(speed_truth),
        "propagation_speed_estimate": float(speed_recon),
        "propagation_speed_abs_error": float(abs(speed_recon - speed_truth)),
        "amplitude_ratio": amplitude_ratio,
        "amplitude_attenuation": float(1.0 - amplitude_ratio),
        "measurement_snr_db": float(measurement_snr),
        "reconstruction_snr_db": float(reconstruction_snr),
        "snr_gain_db": float(reconstruction_snr - measurement_snr),
        "spatial_metrics": spatial,
    }
    _assert_finite_metrics(metrics)
    return metrics


def spatial_metric_summary(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    *,
    positions: np.ndarray,
) -> dict[str, float]:
    records: list[dict[str, float]] = []
    centers = np.asarray(positions, dtype=np.float64).reshape(-1, 1)
    for truth_frame, recon_frame in zip(truth, reconstruction, strict=True):
        peak = float(np.max(np.abs(truth_frame)))
        if peak <= 1.0e-12:
            continue
        mask = np.asarray(truth_frame >= 0.25 * peak, dtype=bool)
        if not np.any(mask):
            continue
        records.append(
            greit_metrics(
                recon_frame,
                mask,
                centers=centers,
                target_values=truth_frame,
            )
        )
    if not records:
        return {key: 0.0 for key in GREIT_METRIC_KEYS}
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in GREIT_METRIC_KEYS
    }


def evaluate_peak_delay_gate(
    fixtures: dict[str, dict[str, Any]],
    *,
    peak_delay_tolerance: float,
) -> dict[str, Any]:
    tolerance = _nonnegative_finite(peak_delay_tolerance, name="peak_delay_tolerance")
    violations: list[dict[str, Any]] = []
    max_delay = 0.0
    for fixture_name, fixture in fixtures.items():
        for method_name, method in fixture["methods"].items():
            delay = float(method["fidelity"]["peak_time_max_positive_delay"])
            max_delay = max(max_delay, delay)
            if delay > tolerance + 1.0e-12:
                violations.append(
                    {
                        "fixture": fixture_name,
                        "method": method_name,
                        "peak_time_max_positive_delay": delay,
                    }
                )
    return {
        "passed": not violations,
        "peak_delay_tolerance": tolerance,
        "max_peak_time_positive_delay": float(max_delay),
        "violations": violations,
    }


def summarize_benchmark(fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for fixture_name, fixture in fixtures.items():
        for method_name, method in fixture["methods"].items():
            candidates.append((fixture_name, method_name, method["fidelity"]))
    best_rmse = min(candidates, key=lambda item: float(item[2]["rmse"]))
    best_speed = min(
        candidates,
        key=lambda item: float(item[2]["propagation_speed_abs_error"]),
    )
    return {
        "fixture_count": int(len(fixtures)),
        "method_count": int(
            sum(len(fixture["methods"]) for fixture in fixtures.values())
        ),
        "best_rmse": {
            "fixture": best_rmse[0],
            "method": best_rmse[1],
            "value": float(best_rmse[2]["rmse"]),
        },
        "best_propagation_speed": {
            "fixture": best_speed[0],
            "method": best_speed[1],
            "value": float(best_speed[2]["propagation_speed_abs_error"]),
        },
    }


def center_trace(frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [center_of_mass(frame, positions) for frame in np.asarray(frames)],
        dtype=np.float64,
    )


def center_of_mass(values: np.ndarray, positions: np.ndarray) -> float:
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    weights = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    if weights.size != pos.size:
        raise ValueError("values and positions must have the same length.")
    total = float(np.sum(weights))
    if total <= 1.0e-12:
        weights = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
        total = float(np.sum(weights))
    if total <= 1.0e-12:
        return float(np.mean(pos))
    return float(np.dot(weights, pos) / total)


def speed_from_center_trace(center: np.ndarray, times: np.ndarray) -> float:
    coeff = np.polyfit(
        np.asarray(times, dtype=np.float64).reshape(-1),
        np.asarray(center, dtype=np.float64).reshape(-1),
        deg=1,
    )
    return float(coeff[0])


def onset_time_trace(
    frames: np.ndarray, times: np.ndarray, *, fraction: float
) -> np.ndarray:
    frac = _unit_interval(fraction, name="onset_fraction")
    arr = np.asarray(frames, dtype=np.float64)
    out = np.empty(arr.shape[1], dtype=np.float64)
    for col in range(arr.shape[1]):
        trace = arr[:, col]
        threshold = frac * float(np.max(trace))
        hits = np.flatnonzero(trace >= threshold)
        out[col] = float(times[int(hits[0])]) if hits.size else float(times[-1])
    return out


def peak_time_trace(frames: np.ndarray, times: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    idx = np.argmax(arr, axis=0)
    return np.asarray(times, dtype=np.float64)[idx]


def snr_db(reference: np.ndarray, observed: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if ref.shape != obs.shape:
        raise ValueError("reference and observed must have matching shapes.")
    signal = float(np.linalg.norm(ref))
    noise = float(np.linalg.norm(obs - ref))
    if noise <= 1.0e-15:
        return 300.0
    return float(20.0 * np.log10(max(signal, 1.0e-15) / noise))


def smoothstep(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    return x * x * (3.0 - 2.0 * x)


def sigmoid(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def write_payload(path: Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def _validate_fixture_inputs(
    n_cells: int,
    n_frames: int,
    n_measurements: int,
    noise_std: float,
) -> None:
    if int(n_cells) < 4:
        raise ValueError("n_cells must be at least 4.")
    if int(n_frames) < 4:
        raise ValueError("n_frames must be at least 4.")
    if int(n_measurements) < 2:
        raise ValueError("n_measurements must be at least 2.")
    if float(noise_std) < 0.0 or not np.isfinite(float(noise_std)):
        raise ValueError("noise_std must be finite and non-negative.")


def _frame_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError(f"{name} must be a non-empty 2D frame array.")
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _nonnegative_finite(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return out


def _unit_interval(value: float, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0 or out > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return out


def _assert_finite_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, dict):
            _assert_finite_metrics(value)
            continue
        if not np.isfinite(float(value)):
            raise FloatingPointError(f"dynamic metric {key!r} is non-finite.")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_benchmark(
        n_cells=args.n_cells,
        n_frames=args.n_frames,
        n_measurements=args.n_measurements,
        lambda_=args.lambda_,
        ridge=args.ridge,
        noise_std=args.noise_std,
        seed=args.seed,
        temporal_alpha=args.temporal_alpha,
        onset_fraction=args.onset_fraction,
        peak_delay_tolerance=args.peak_delay_tolerance,
    )
    output = write_payload(args.output_json, payload)
    print(f"[OK] dynamic validation benchmark saved: {output}")
    if payload["gate"]["passed"] or args.no_fail_on_delay:
        return 0
    print(
        "[FAIL] peak delay gate exceeded: "
        f"{payload['gate']['max_peak_time_positive_delay']:.6g} > "
        f"{payload['gate']['peak_delay_tolerance']:.6g}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
