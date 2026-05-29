#!/usr/bin/env python3
"""Sweep T65 4D GN, T66 TV/Huber, and T67 Kalman on a neural travelling wave."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.inverse import (  # noqa: E402
    build_one_step_rm,
    graph_laplacian,
    run_dynamic_kalman_filter,
    solve_batch_spatiotemporal_gn,
    solve_spatiotemporal_tv_huber,
)
from pyeidors.io._json import json_ready as _json_ready  # noqa: E402
from scripts.benchmarks.benchmark_dynamic_validation import (  # noqa: E402
    build_travelling_wave_fixture,
    dynamic_fidelity_metrics,
)


SCHEMA = "pyeidors-dynamic-t65-t66-t67-sweep-v1"


def _strictly_increasing(values: np.ndarray, *, chunk_size: int = 1_048_576) -> bool:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return True
    block_size = max(2, min(int(chunk_size), int(arr.size)))
    work = np.empty(block_size - 1, dtype=bool)
    previous = float(arr[0])
    for start in range(1, int(arr.size), block_size - 1):
        stop = min(start + block_size - 1, int(arr.size))
        chunk = arr[start:stop]
        if chunk.size == 0:
            continue
        if float(chunk[0]) <= previous:
            return False
        if chunk.size > 1:
            mask = work[: chunk.size - 1]
            np.greater(chunk[1:], chunk[:-1], out=mask)
            if not bool(mask.all()):
                return False
        previous = float(chunk[-1])
    return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=32)
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--n-measurements", type=int, default=20)
    parser.add_argument("--lambda-s", type=float, default=0.08)
    parser.add_argument("--lambda-t", default="0.02,0.04,0.08,0.12,0.20,0.35")
    parser.add_argument("--huber-delta", default="0.01,0.02,0.03,0.05,0.08,0.12")
    parser.add_argument("--temporal-order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--noise-std", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--onset-fraction", type=float, default=0.30)
    parser.add_argument("--max-outer-iterations", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--peak-delay-limit", type=float, default=0.05)
    parser.add_argument("--rmse-ratio-limit", type=float, default=1.08)
    parser.add_argument("--kalman-lag", default="0,1,2,3")
    parser.add_argument("--kalman-process-noise", default="0.005,0.01,0.02,0.04,0.08")
    parser.add_argument("--kalman-measurement-noise", default="0.01,0.02,0.04,0.08")
    parser.add_argument("--kalman-initial-covariance", type=float, default=1.0)
    parser.add_argument(
        "--kalman-transition",
        default="identity",
        help="Comma-separated transition models: identity, propagation.",
    )
    parser.add_argument(
        "--kalman-velocity",
        default="0.50,0.68,0.85",
        help="Comma-separated propagation velocities in fixture position units per time unit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(
            "reports/runtime_benchmarks/dynamic_t65_t66_t67_sweep_20260426.json"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(
            "reports/runtime_benchmarks/dynamic_t65_t66_t67_sweep_20260426.md"
        ),
    )
    return parser.parse_args(argv)


def run_sweep(
    *,
    n_cells: int = 32,
    n_frames: int = 32,
    n_measurements: int = 20,
    lambda_s: float = 0.08,
    lambda_t_values: Sequence[float] = (0.02, 0.04, 0.08, 0.12, 0.20, 0.35),
    huber_delta_values: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12),
    temporal_order: int = 2,
    noise_std: float = 2.0e-3,
    seed: int = 20260425,
    onset_fraction: float = 0.30,
    max_outer_iterations: int = 6,
    tolerance: float = 1.0e-6,
    peak_delay_limit: float = 0.05,
    rmse_ratio_limit: float = 1.08,
    kalman_lag_values: Sequence[int] = (0, 1, 2, 3),
    kalman_process_noise_values: Sequence[float] = (0.005, 0.01, 0.02, 0.04, 0.08),
    kalman_measurement_noise_values: Sequence[float] = (0.01, 0.02, 0.04, 0.08),
    kalman_initial_covariance: float = 1.0,
    kalman_transition_modes: Sequence[str] = ("identity",),
    kalman_velocity_values: Sequence[float] = (0.50, 0.68, 0.85),
) -> dict[str, Any]:
    """Run deterministic T65/T66/T67 hyperparameter sweep."""

    lambda_t_grid = _positive_grid(lambda_t_values, name="lambda_t_values")
    huber_delta_grid = _positive_grid(huber_delta_values, name="huber_delta_values")
    kalman_lag_grid = _nonnegative_int_grid(kalman_lag_values, name="kalman_lag_values")
    kalman_q_grid = _positive_grid(
        kalman_process_noise_values,
        name="kalman_process_noise_values",
    )
    kalman_r_grid = _positive_grid(
        kalman_measurement_noise_values,
        name="kalman_measurement_noise_values",
    )
    kalman_transition_grid = _transition_mode_grid(kalman_transition_modes)
    kalman_velocity_grid = _positive_grid(
        kalman_velocity_values,
        name="kalman_velocity_values",
    )
    fixture = build_travelling_wave_fixture(
        n_cells=n_cells,
        n_frames=n_frames,
        n_measurements=n_measurements,
        noise_std=noise_std,
        seed=seed,
    )
    mesh = fixture["mesh"]
    spatial_prior = graph_laplacian(mesh) + 1.0e-8 * sparse.eye(
        mesh.num_cells(),
        format="csr",
    )
    truth = np.asarray(fixture["truth"], dtype=np.float64)
    t66_rows: list[dict[str, Any]] = []
    t65_by_lambda: dict[float, dict[str, Any]] = {}

    for lambda_t in lambda_t_grid:
        l2_start = time.perf_counter()
        l2_result = solve_batch_spatiotemporal_gn(
            fixture["jacobian"],
            fixture["measurements"],
            spatial_prior=spatial_prior,
            lambda_s=lambda_s,
            lambda_t=lambda_t,
            temporal_order=temporal_order,
            rowwise_rm_baseline=False,
        )
        l2_seconds = time.perf_counter() - l2_start
        l2_metrics = dynamic_fidelity_metrics(
            truth,
            l2_result.values,
            clean_measurements=fixture["clean_measurements"],
            noisy_measurements=fixture["measurements"],
            jacobian=fixture["jacobian"],
            positions=fixture["positions"],
            times=fixture["times"],
            onset_fraction=onset_fraction,
        )
        t65_by_lambda[lambda_t] = _t65_row_payload(
            lambda_t=lambda_t,
            solve_seconds=l2_seconds,
            metrics=l2_metrics,
            metadata=dict(l2_result.metadata),
        )
        for huber_delta in huber_delta_grid:
            tv_start = time.perf_counter()
            tv_result = solve_spatiotemporal_tv_huber(
                fixture["jacobian"],
                fixture["measurements"],
                spatial_graph=mesh,
                lambda_s=lambda_s,
                lambda_t=lambda_t,
                huber_delta=huber_delta,
                temporal_order=temporal_order,
                max_outer_iterations=max_outer_iterations,
                tolerance=tolerance,
            )
            tv_seconds = time.perf_counter() - tv_start
            tv_metrics = dynamic_fidelity_metrics(
                truth,
                tv_result.values,
                clean_measurements=fixture["clean_measurements"],
                noisy_measurements=fixture["measurements"],
                jacobian=fixture["jacobian"],
                positions=fixture["positions"],
                times=fixture["times"],
                onset_fraction=onset_fraction,
            )
            t66_rows.append(
                _t66_row_payload(
                    lambda_t=lambda_t,
                    huber_delta=huber_delta,
                    tv_metrics=tv_metrics,
                    l2_metrics=l2_metrics,
                    solve_seconds=tv_seconds,
                    metadata=dict(tv_result.metadata),
                    peak_delay_limit=peak_delay_limit,
                    rmse_ratio_limit=rmse_ratio_limit,
                )
            )

    best_t65 = min(
        t65_by_lambda.values(),
        key=lambda row: float(row["fast_conduction_score"]),
    )
    best_t66 = _best_t66_row(t66_rows)
    rm_start = time.perf_counter()
    rm = build_one_step_rm(
        fixture["jacobian"],
        regularization=spatial_prior,
        lambda_=lambda_s,
        mode="laplace",
        return_metadata=True,
    )
    rm_seconds = time.perf_counter() - rm_start
    t67_rows: list[dict[str, Any]] = []
    transitions = build_kalman_transition_payloads(
        kalman_transition_grid,
        positions=fixture["positions"],
        times=fixture["times"],
        velocity_values=kalman_velocity_grid,
    )
    for transition_payload in transitions:
        for lag in kalman_lag_grid:
            for process_noise in kalman_q_grid:
                for measurement_noise in kalman_r_grid:
                    kalman_start = time.perf_counter()
                    kalman = run_dynamic_kalman_filter(
                        rm.rm,
                        fixture["measurements"],
                        observation_mode="rm_observation",
                        transition=transition_payload["matrix"],
                        process_noise=process_noise,
                        measurement_noise=measurement_noise,
                        initial_covariance=kalman_initial_covariance,
                        fixed_lag=lag,
                        timestamps=fixture["times"],
                    )
                    kalman_seconds = time.perf_counter() - kalman_start
                    kalman_metrics = dynamic_fidelity_metrics(
                        truth,
                        kalman.values,
                        clean_measurements=fixture["clean_measurements"],
                        noisy_measurements=fixture["measurements"],
                        jacobian=fixture["jacobian"],
                        positions=fixture["positions"],
                        times=fixture["times"],
                        onset_fraction=onset_fraction,
                    )
                    t67_rows.append(
                        _t67_row_payload(
                            transition_payload=transition_payload,
                            fixed_lag=lag,
                            process_noise=process_noise,
                            measurement_noise=measurement_noise,
                            initial_covariance=kalman_initial_covariance,
                            kalman_metrics=kalman_metrics,
                            solve_seconds=kalman_seconds,
                            rm_build_seconds=rm_seconds,
                            metadata=dict(kalman.metadata),
                            best_t65=best_t65,
                            best_t66=best_t66,
                            peak_delay_limit=peak_delay_limit,
                            rmse_ratio_limit=rmse_ratio_limit,
                        )
                    )

    summary = summarize_sweep(
        t66_rows,
        t67_rows,
        t65_by_lambda=t65_by_lambda,
        lambda_t_grid=lambda_t_grid,
        huber_delta_grid=huber_delta_grid,
        kalman_lag_grid=kalman_lag_grid,
        kalman_q_grid=kalman_q_grid,
        kalman_r_grid=kalman_r_grid,
        kalman_transition_grid=kalman_transition_grid,
        kalman_velocity_grid=kalman_velocity_grid,
        peak_delay_limit=peak_delay_limit,
        rmse_ratio_limit=rmse_ratio_limit,
    )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "fixture": "travelling_wave",
            "domain": "neural_fast_conduction",
            "n_cells": int(n_cells),
            "n_frames": int(n_frames),
            "n_measurements": int(n_measurements),
            "lambda_s": float(lambda_s),
            "lambda_t_values": [float(v) for v in lambda_t_grid],
            "huber_delta_values": [float(v) for v in huber_delta_grid],
            "temporal_order": int(temporal_order),
            "noise_std": float(noise_std),
            "seed": int(seed),
            "onset_fraction": float(onset_fraction),
            "max_outer_iterations": int(max_outer_iterations),
            "tolerance": float(tolerance),
            "peak_delay_limit": float(peak_delay_limit),
            "rmse_ratio_limit": float(rmse_ratio_limit),
            "kalman_lag_values": [int(v) for v in kalman_lag_grid],
            "kalman_process_noise_values": [float(v) for v in kalman_q_grid],
            "kalman_measurement_noise_values": [float(v) for v in kalman_r_grid],
            "kalman_initial_covariance": float(kalman_initial_covariance),
            "kalman_transition_modes": list(kalman_transition_grid),
            "kalman_velocity_values": [float(v) for v in kalman_velocity_grid],
        },
        "t65_l2_by_lambda_t": {str(key): value for key, value in t65_by_lambda.items()},
        "t66_rows": t66_rows,
        "t67_kalman_rows": t67_rows,
        "rows": t66_rows,
        "summary": summary,
    }


def summarize_sweep(
    t66_rows: Sequence[Mapping[str, Any]],
    t67_rows: Sequence[Mapping[str, Any]],
    *,
    t65_by_lambda: Mapping[float, Mapping[str, Any]],
    lambda_t_grid: Sequence[float],
    huber_delta_grid: Sequence[float],
    kalman_lag_grid: Sequence[int],
    kalman_q_grid: Sequence[float],
    kalman_r_grid: Sequence[float],
    kalman_transition_grid: Sequence[str],
    kalman_velocity_grid: Sequence[float],
    peak_delay_limit: float,
    rmse_ratio_limit: float,
) -> dict[str, Any]:
    """Choose fidelity-oriented operating regions for fast conduction."""

    t65_rows = list(t65_by_lambda.values())
    best_t65 = min(t65_rows, key=lambda row: float(row["fast_conduction_score"]))
    t66_summary = _summarize_t66_rows(
        t66_rows,
        lambda_t_grid=lambda_t_grid,
        huber_delta_grid=huber_delta_grid,
        peak_delay_limit=peak_delay_limit,
        rmse_ratio_limit=rmse_ratio_limit,
    )
    t67_summary = _summarize_t67_rows(
        t67_rows,
        kalman_lag_grid=kalman_lag_grid,
        kalman_q_grid=kalman_q_grid,
        kalman_r_grid=kalman_r_grid,
        kalman_transition_grid=kalman_transition_grid,
        kalman_velocity_grid=kalman_velocity_grid,
        peak_delay_limit=peak_delay_limit,
        rmse_ratio_limit=rmse_ratio_limit,
    )
    best_t66 = t66_summary["best_score"]
    best_t67 = t67_summary["best_score"]
    best_overall = min(
        (
            _overall_candidate("t65_spatiotemporal_l2", best_t65),
            _overall_candidate("t66_spatiotemporal_tv_huber", best_t66),
            _overall_candidate("t67_kalman_fixed_lag", best_t67),
        ),
        key=lambda row: float(row["fast_conduction_score"]),
    )
    return {
        "row_count": int(len(t66_rows) + len(t67_rows)),
        "t65_row_count": int(len(t65_rows)),
        "t66_row_count": int(len(t66_rows)),
        "t67_row_count": int(len(t67_rows)),
        "peak_delay_limit": float(peak_delay_limit),
        "rmse_ratio_limit": float(rmse_ratio_limit),
        "best_t65_l2": _compact_t65_row(best_t65),
        "best_t66_tv_huber": best_t66,
        "best_t67_kalman": best_t67,
        "best_overall_by_fast_conduction_score": best_overall,
        "best_score": best_t66,
        "best_speed": t66_summary["best_speed"],
        "best_peak_time": t66_summary["best_peak_time"],
        "best_rmse": t66_summary["best_rmse"],
        "candidate_count": t66_summary["candidate_count"],
        "t67_candidate_count": t67_summary["candidate_count"],
        "recommended_lambda_t_range": t66_summary["recommended_lambda_t_range"],
        "recommended_huber_delta_range": t66_summary["recommended_huber_delta_range"],
        "recommended_region_rows": t66_summary["recommended_region_rows"],
        "recommended_kalman_fixed_lag_range": t67_summary[
            "recommended_fixed_lag_range"
        ],
        "recommended_kalman_process_noise_range": t67_summary[
            "recommended_process_noise_range"
        ],
        "recommended_kalman_measurement_noise_range": t67_summary[
            "recommended_measurement_noise_range"
        ],
        "recommended_kalman_transition_modes": t67_summary[
            "recommended_transition_modes"
        ],
        "recommended_kalman_velocity_range": t67_summary["recommended_velocity_range"],
        "recommended_kalman_region_rows": t67_summary["recommended_region_rows"],
        "propagation_aware_A_review": _propagation_aware_review(t67_rows),
        "comparison": {
            "best_t67_vs_best_t65": _reference_delta(
                best_t67,
                best_t65,
                method_key="t67",
                reference_key="t65",
            ),
            "best_t67_vs_best_t66": _reference_delta(
                best_t67,
                best_t66,
                method_key="t67",
                reference_key="t66",
            ),
        },
        "grid": {
            "lambda_t": [float(v) for v in lambda_t_grid],
            "huber_delta": [float(v) for v in huber_delta_grid],
            "kalman_lag": [int(v) for v in kalman_lag_grid],
            "kalman_process_noise": [float(v) for v in kalman_q_grid],
            "kalman_measurement_noise": [float(v) for v in kalman_r_grid],
            "kalman_transition": list(kalman_transition_grid),
            "kalman_velocity": [float(v) for v in kalman_velocity_grid],
        },
    }


def write_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def write_markdown_report(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_markdown_report(payload), encoding="utf-8")
    return target


def build_kalman_transition_payloads(
    modes: Sequence[str],
    *,
    positions: Sequence[float],
    times: Sequence[float],
    velocity_values: Sequence[float],
) -> tuple[dict[str, Any], ...]:
    """Build opt-in Kalman transition matrices for the travelling-wave fixture."""

    resolved_modes = _transition_mode_grid(modes)
    positions_arr = np.asarray(positions, dtype=np.float64).reshape(-1)
    times_arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if positions_arr.size <= 1:
        raise ValueError("positions must contain at least two cells.")
    if times_arr.size <= 1:
        raise ValueError("times must contain at least two frames.")
    dt = float(np.median(np.diff(times_arr)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("times must be strictly increasing enough to infer dt.")
    payloads: list[dict[str, Any]] = []
    n_state = int(positions_arr.size)
    for mode in resolved_modes:
        if mode == "identity":
            payloads.append(
                {
                    "kind": "identity",
                    "matrix": None,
                    "velocity": None,
                    "shift_per_frame": 0.0,
                    "metadata": {
                        "kind": "identity",
                        "matrix_shape": [n_state, n_state],
                        "boundary": "none",
                    },
                }
            )
            continue
        for velocity in velocity_values:
            matrix = propagation_transition_matrix(
                positions_arr,
                velocity=float(velocity),
                dt=dt,
            )
            row_sums = np.sum(matrix, axis=1)
            payloads.append(
                {
                    "kind": "propagation",
                    "matrix": matrix,
                    "velocity": float(velocity),
                    "shift_per_frame": float(velocity) * dt,
                    "metadata": {
                        "kind": "propagation",
                        "velocity": float(velocity),
                        "dt": dt,
                        "shift_per_frame": float(velocity) * dt,
                        "boundary": "zero_open",
                        "matrix_shape": [int(v) for v in matrix.shape],
                        "matrix_nnz": int(np.count_nonzero(matrix)),
                        "row_sum_min": float(np.min(row_sums)),
                        "row_sum_max": float(np.max(row_sums)),
                    },
                }
            )
    return tuple(payloads)


def propagation_transition_matrix(
    positions: Sequence[float],
    *,
    velocity: float,
    dt: float,
) -> np.ndarray:
    """Construct a 1D open-boundary advection matrix for ``x_t = A x_{t-1}``."""

    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    if pos.size <= 1:
        raise ValueError("positions must contain at least two cells.")
    if not _strictly_increasing(pos):
        raise ValueError("positions must be strictly increasing.")
    vel = float(velocity)
    step = float(dt)
    if not np.isfinite(vel) or vel <= 0.0:
        raise ValueError("velocity must be finite and positive.")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("dt must be finite and positive.")
    shift = vel * step
    matrix = np.zeros((pos.size, pos.size), dtype=np.float64)
    for row, target_pos in enumerate(pos):
        source_pos = target_pos - shift
        if source_pos < pos[0] or source_pos > pos[-1]:
            continue
        right = int(np.searchsorted(pos, source_pos, side="left"))
        if right == 0:
            matrix[row, 0] = 1.0
            continue
        if right < pos.size and np.isclose(source_pos, pos[right]):
            matrix[row, right] = 1.0
            continue
        if right >= pos.size:
            matrix[row, -1] = 1.0
            continue
        left = right - 1
        span = float(pos[right] - pos[left])
        frac = float((source_pos - pos[left]) / span)
        matrix[row, left] = 1.0 - frac
        matrix[row, right] = frac
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _t65_row_payload(
    *,
    lambda_t: float,
    solve_seconds: float,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "lambda_t": float(lambda_t),
        "method": "t65_spatiotemporal_l2",
        "solve_seconds": float(solve_seconds),
        "rmse_t65": float(metrics["rmse"]),
        "speed_error_t65": float(metrics["propagation_speed_abs_error"]),
        "peak_time_mae_t65": float(metrics["peak_time_mean_abs_error"]),
        "peak_time_max_positive_delay_t65": float(
            metrics["peak_time_max_positive_delay"]
        ),
        "onset_time_mae_t65": float(metrics["onset_time_mean_abs_error"]),
        "amplitude_attenuation_t65": float(metrics["amplitude_attenuation"]),
        "snr_gain_db_t65": float(metrics["snr_gain_db"]),
        "fast_conduction_score": _fast_conduction_score(metrics, rmse_ratio=1.0),
        "metrics": dict(metrics),
        "metadata": {
            "schema": metadata["schema"],
            "lambda_s": metadata["lambda_s"],
            "lambda_t": metadata["lambda_t"],
            "temporal_order": metadata["temporal_order"],
        },
    }


def _t66_row_payload(
    *,
    lambda_t: float,
    huber_delta: float,
    tv_metrics: Mapping[str, Any],
    l2_metrics: Mapping[str, Any],
    solve_seconds: float,
    metadata: Mapping[str, Any],
    peak_delay_limit: float,
    rmse_ratio_limit: float,
) -> dict[str, Any]:
    rmse_ratio = float(tv_metrics["rmse"]) / max(float(l2_metrics["rmse"]), 1.0e-15)
    speed_delta = float(l2_metrics["propagation_speed_abs_error"]) - float(
        tv_metrics["propagation_speed_abs_error"]
    )
    peak_delta = float(l2_metrics["peak_time_mean_abs_error"]) - float(
        tv_metrics["peak_time_mean_abs_error"]
    )
    onset_delta = float(l2_metrics["onset_time_mean_abs_error"]) - float(
        tv_metrics["onset_time_mean_abs_error"]
    )
    score = _fast_conduction_score(tv_metrics, rmse_ratio=rmse_ratio)
    passes = (
        float(tv_metrics["peak_time_max_positive_delay"]) <= float(peak_delay_limit)
        and rmse_ratio <= float(rmse_ratio_limit)
        and float(tv_metrics["propagation_speed_abs_error"])
        <= float(l2_metrics["propagation_speed_abs_error"])
    )
    return {
        "lambda_t": float(lambda_t),
        "huber_delta": float(huber_delta),
        "method": "t66_spatiotemporal_tv_huber",
        "baseline": "t65_spatiotemporal_l2",
        "solve_seconds": float(solve_seconds),
        "outer_iterations": int(metadata["outer_iterations"]),
        "rmse_t66": float(tv_metrics["rmse"]),
        "rmse_t65": float(l2_metrics["rmse"]),
        "rmse_delta_t65_minus_t66": float(l2_metrics["rmse"])
        - float(tv_metrics["rmse"]),
        "rmse_ratio_t66_over_t65": rmse_ratio,
        "speed_error_t66": float(tv_metrics["propagation_speed_abs_error"]),
        "speed_error_t65": float(l2_metrics["propagation_speed_abs_error"]),
        "speed_error_delta_t65_minus_t66": speed_delta,
        "peak_time_mae_t66": float(tv_metrics["peak_time_mean_abs_error"]),
        "peak_time_mae_t65": float(l2_metrics["peak_time_mean_abs_error"]),
        "peak_time_mae_delta_t65_minus_t66": peak_delta,
        "peak_time_max_positive_delay_t66": float(
            tv_metrics["peak_time_max_positive_delay"]
        ),
        "onset_time_mae_t66": float(tv_metrics["onset_time_mean_abs_error"]),
        "onset_time_mae_t65": float(l2_metrics["onset_time_mean_abs_error"]),
        "onset_time_mae_delta_t65_minus_t66": onset_delta,
        "amplitude_attenuation_t66": float(tv_metrics["amplitude_attenuation"]),
        "snr_gain_db_t66": float(tv_metrics["snr_gain_db"]),
        "fast_conduction_score": score,
        "passes_fast_conduction_gate": bool(passes),
        "metrics": dict(tv_metrics),
        "metadata": {
            "schema": metadata["schema"],
            "penalty": metadata["penalty"],
            "spatial_weight_range": metadata["spatial_weight_range"],
            "temporal_weight_range": metadata["temporal_weight_range"],
            "t65_l2_comparison": metadata["t65_l2_comparison"],
        },
    }


def _t67_row_payload(
    *,
    transition_payload: Mapping[str, Any],
    fixed_lag: int,
    process_noise: float,
    measurement_noise: float,
    initial_covariance: float,
    kalman_metrics: Mapping[str, Any],
    solve_seconds: float,
    rm_build_seconds: float,
    metadata: Mapping[str, Any],
    best_t65: Mapping[str, Any],
    best_t66: Mapping[str, Any],
    peak_delay_limit: float,
    rmse_ratio_limit: float,
) -> dict[str, Any]:
    rmse_ratio_t65 = float(kalman_metrics["rmse"]) / max(
        float(best_t65["rmse_t65"]),
        1.0e-15,
    )
    rmse_ratio_t66 = float(kalman_metrics["rmse"]) / max(
        float(best_t66["rmse_t66"]),
        1.0e-15,
    )
    score = _fast_conduction_score(kalman_metrics, rmse_ratio=rmse_ratio_t65)
    reference_speed = min(
        float(best_t65["speed_error_t65"]),
        float(best_t66["speed_error_t66"]),
    )
    passes = (
        float(kalman_metrics["peak_time_max_positive_delay"]) <= float(peak_delay_limit)
        and rmse_ratio_t65 <= float(rmse_ratio_limit)
        and float(kalman_metrics["propagation_speed_abs_error"]) <= reference_speed
    )
    latency_seconds = float(metadata["latency_seconds"])
    transition_meta = dict(transition_payload["metadata"])
    return {
        "transition_kind": str(transition_payload["kind"]),
        "transition_velocity": transition_payload["velocity"],
        "transition_shift_per_frame": float(transition_payload["shift_per_frame"]),
        "fixed_lag": int(fixed_lag),
        "process_noise": float(process_noise),
        "measurement_noise": float(measurement_noise),
        "initial_covariance": float(initial_covariance),
        "method": "t67_kalman_fixed_lag",
        "method_variant": f"t67_kalman_{transition_payload['kind']}_transition",
        "baseline": "t65_best_l2_and_t66_best_tv_huber",
        "solve_seconds": float(solve_seconds),
        "rm_build_seconds": float(rm_build_seconds),
        "latency_seconds": latency_seconds,
        "rmse_t67": float(kalman_metrics["rmse"]),
        "rmse_best_t65": float(best_t65["rmse_t65"]),
        "rmse_best_t66": float(best_t66["rmse_t66"]),
        "rmse_ratio_t67_over_best_t65": rmse_ratio_t65,
        "rmse_ratio_t67_over_best_t66": rmse_ratio_t66,
        "speed_error_t67": float(kalman_metrics["propagation_speed_abs_error"]),
        "speed_error_best_t65": float(best_t65["speed_error_t65"]),
        "speed_error_best_t66": float(best_t66["speed_error_t66"]),
        "speed_error_delta_t65_minus_t67": float(best_t65["speed_error_t65"])
        - float(kalman_metrics["propagation_speed_abs_error"]),
        "speed_error_delta_t66_minus_t67": float(best_t66["speed_error_t66"])
        - float(kalman_metrics["propagation_speed_abs_error"]),
        "peak_time_mae_t67": float(kalman_metrics["peak_time_mean_abs_error"]),
        "peak_time_mae_best_t65": float(best_t65["peak_time_mae_t65"]),
        "peak_time_mae_best_t66": float(best_t66["peak_time_mae_t66"]),
        "peak_time_mae_delta_t65_minus_t67": float(best_t65["peak_time_mae_t65"])
        - float(kalman_metrics["peak_time_mean_abs_error"]),
        "peak_time_mae_delta_t66_minus_t67": float(best_t66["peak_time_mae_t66"])
        - float(kalman_metrics["peak_time_mean_abs_error"]),
        "peak_time_max_positive_delay_t67": float(
            kalman_metrics["peak_time_max_positive_delay"]
        ),
        "onset_time_mae_t67": float(kalman_metrics["onset_time_mean_abs_error"]),
        "onset_time_mae_best_t65": float(best_t65["onset_time_mae_t65"]),
        "onset_time_mae_best_t66": float(best_t66["onset_time_mae_t66"]),
        "amplitude_attenuation_t67": float(kalman_metrics["amplitude_attenuation"]),
        "snr_gain_db_t67": float(kalman_metrics["snr_gain_db"]),
        "fast_conduction_score": score,
        "passes_fast_conduction_gate": bool(passes),
        "metrics": dict(kalman_metrics),
        "metadata": {
            "transition": transition_meta,
            "schema": metadata["schema"],
            "observation_mode": metadata["observation_mode"],
            "online_hot_path": metadata["online_hot_path"],
            "fixed_lag": metadata["fixed_lag"],
            "latency_frames": metadata["latency_frames"],
            "latency_seconds": metadata["latency_seconds"],
            "smoother": metadata["smoother"],
            "default_enabled": metadata["default_enabled"],
            "requires_t69_gate_before_default": metadata[
                "requires_t69_gate_before_default"
            ],
            "forward_solve_count": metadata["forward_solve_count"],
            "adjoint_solve_count": metadata["adjoint_solve_count"],
            "ksp_solve_count": metadata["ksp_solve_count"],
            "jacobian_rebuild_count": metadata["jacobian_rebuild_count"],
        },
    }


def _summarize_t66_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    lambda_t_grid: Sequence[float],
    huber_delta_grid: Sequence[float],
    peak_delay_limit: float,
    rmse_ratio_limit: float,
) -> dict[str, Any]:
    candidates = [row for row in rows if bool(row["passes_fast_conduction_gate"])]
    pool = candidates or list(rows)
    best_score = min(pool, key=lambda row: float(row["fast_conduction_score"]))
    best_speed = min(rows, key=lambda row: float(row["speed_error_t66"]))
    best_peak = min(rows, key=lambda row: float(row["peak_time_mae_t66"]))
    best_rmse = min(rows, key=lambda row: float(row["rmse_t66"]))
    threshold = float(best_score["fast_conduction_score"]) * 1.25 + 1.0e-12
    stable = [
        row
        for row in pool
        if float(row["fast_conduction_score"]) <= threshold
        and float(row["rmse_ratio_t66_over_t65"]) <= rmse_ratio_limit
    ]
    if not stable:
        stable = [best_score]
    return {
        "row_count": int(len(rows)),
        "candidate_count": int(len(candidates)),
        "peak_delay_limit": float(peak_delay_limit),
        "rmse_ratio_limit": float(rmse_ratio_limit),
        "best_score": _compact_t66_row(best_score),
        "best_speed": _compact_t66_row(best_speed),
        "best_peak_time": _compact_t66_row(best_peak),
        "best_rmse": _compact_t66_row(best_rmse),
        "recommended_lambda_t_range": _range(
            [float(row["lambda_t"]) for row in stable]
        ),
        "recommended_huber_delta_range": _range(
            [float(row["huber_delta"]) for row in stable]
        ),
        "recommended_region_rows": [_compact_t66_row(row) for row in stable],
        "grid": {
            "lambda_t": [float(v) for v in lambda_t_grid],
            "huber_delta": [float(v) for v in huber_delta_grid],
        },
    }


def _summarize_t67_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    kalman_lag_grid: Sequence[int],
    kalman_q_grid: Sequence[float],
    kalman_r_grid: Sequence[float],
    kalman_transition_grid: Sequence[str],
    kalman_velocity_grid: Sequence[float],
    peak_delay_limit: float,
    rmse_ratio_limit: float,
) -> dict[str, Any]:
    candidates = [row for row in rows if bool(row["passes_fast_conduction_gate"])]
    pool = candidates or list(rows)
    best_score = min(pool, key=lambda row: float(row["fast_conduction_score"]))
    best_speed = min(rows, key=lambda row: float(row["speed_error_t67"]))
    best_peak = min(rows, key=lambda row: float(row["peak_time_mae_t67"]))
    best_rmse = min(rows, key=lambda row: float(row["rmse_t67"]))
    threshold = float(best_score["fast_conduction_score"]) * 1.25 + 1.0e-12
    stable = [
        row
        for row in pool
        if float(row["fast_conduction_score"]) <= threshold
        and float(row["rmse_ratio_t67_over_best_t65"]) <= rmse_ratio_limit
    ]
    if not stable:
        stable = [best_score]
    return {
        "row_count": int(len(rows)),
        "candidate_count": int(len(candidates)),
        "peak_delay_limit": float(peak_delay_limit),
        "rmse_ratio_limit": float(rmse_ratio_limit),
        "best_score": _compact_t67_row(best_score),
        "best_speed": _compact_t67_row(best_speed),
        "best_peak_time": _compact_t67_row(best_peak),
        "best_rmse": _compact_t67_row(best_rmse),
        "recommended_fixed_lag_range": _int_range(
            [int(row["fixed_lag"]) for row in stable]
        ),
        "recommended_process_noise_range": _range(
            [float(row["process_noise"]) for row in stable]
        ),
        "recommended_measurement_noise_range": _range(
            [float(row["measurement_noise"]) for row in stable]
        ),
        "recommended_transition_modes": sorted(
            {str(row["transition_kind"]) for row in stable}
        ),
        "recommended_velocity_range": _optional_range(
            [
                float(row["transition_velocity"])
                for row in stable
                if row["transition_velocity"] is not None
            ]
        ),
        "recommended_region_rows": [_compact_t67_row(row) for row in stable],
        "grid": {
            "kalman_lag": [int(v) for v in kalman_lag_grid],
            "kalman_process_noise": [float(v) for v in kalman_q_grid],
            "kalman_measurement_noise": [float(v) for v in kalman_r_grid],
            "kalman_transition": list(kalman_transition_grid),
            "kalman_velocity": [float(v) for v in kalman_velocity_grid],
        },
    }


def _best_t66_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates = [row for row in rows if bool(row["passes_fast_conduction_gate"])]
    pool = candidates or list(rows)
    return min(pool, key=lambda row: float(row["fast_conduction_score"]))


def _propagation_aware_review(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    propagation = [row for row in rows if row.get("transition_kind") == "propagation"]
    identity = [row for row in rows if row.get("transition_kind") == "identity"]
    if not propagation:
        return {
            "enabled": False,
            "best_propagation": None,
            "best_identity": _compact_t67_row(
                min(rows, key=lambda row: float(row["fast_conduction_score"]))
            )
            if rows
            else None,
        }
    best_prop = min(propagation, key=lambda row: float(row["fast_conduction_score"]))
    best_identity = (
        min(identity, key=lambda row: float(row["fast_conduction_score"]))
        if identity
        else None
    )
    review: dict[str, Any] = {
        "enabled": True,
        "propagation_row_count": int(len(propagation)),
        "identity_row_count": int(len(identity)),
        "best_propagation": _compact_t67_row(best_prop),
        "best_identity": _compact_t67_row(best_identity) if best_identity else None,
        "gate_passing_propagation_count": int(
            sum(bool(row["passes_fast_conduction_gate"]) for row in propagation)
        ),
    }
    if best_identity is not None:
        review["best_propagation_vs_identity"] = {
            "score_delta_identity_minus_propagation": float(
                best_identity["fast_conduction_score"]
            )
            - float(best_prop["fast_conduction_score"]),
            "speed_delta_identity_minus_propagation": float(
                best_identity["speed_error_t67"]
            )
            - float(best_prop["speed_error_t67"]),
            "rmse_delta_identity_minus_propagation": float(best_identity["rmse_t67"])
            - float(best_prop["rmse_t67"]),
        }
    return review


def _fast_conduction_score(metrics: Mapping[str, Any], *, rmse_ratio: float) -> float:
    return float(
        2.5 * float(metrics["propagation_speed_abs_error"])
        + 1.5 * float(metrics["peak_time_mean_abs_error"])
        + 1.0 * float(metrics["onset_time_mean_abs_error"])
        + 0.25 * abs(float(metrics["amplitude_attenuation"]))
        + 0.10 * max(0.0, rmse_ratio - 1.0)
    )


def _compact_t65_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "method",
        "lambda_t",
        "fast_conduction_score",
        "rmse_t65",
        "speed_error_t65",
        "peak_time_mae_t65",
        "onset_time_mae_t65",
        "peak_time_max_positive_delay_t65",
    )
    return {key: row[key] for key in keys}


def _compact_t66_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "method",
        "lambda_t",
        "huber_delta",
        "fast_conduction_score",
        "rmse_t66",
        "rmse_t65",
        "rmse_ratio_t66_over_t65",
        "speed_error_t66",
        "speed_error_t65",
        "peak_time_mae_t66",
        "onset_time_mae_t66",
        "peak_time_max_positive_delay_t66",
        "passes_fast_conduction_gate",
    )
    return {key: row[key] for key in keys}


def _compact_t67_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "method",
        "method_variant",
        "transition_kind",
        "transition_velocity",
        "transition_shift_per_frame",
        "fixed_lag",
        "process_noise",
        "measurement_noise",
        "latency_seconds",
        "fast_conduction_score",
        "rmse_t67",
        "rmse_ratio_t67_over_best_t65",
        "rmse_ratio_t67_over_best_t66",
        "speed_error_t67",
        "speed_error_best_t65",
        "speed_error_best_t66",
        "speed_error_delta_t65_minus_t67",
        "speed_error_delta_t66_minus_t67",
        "peak_time_mae_t67",
        "onset_time_mae_t67",
        "peak_time_max_positive_delay_t67",
        "passes_fast_conduction_gate",
    )
    return {key: row[key] for key in keys}


def _overall_candidate(method: str, row: Mapping[str, Any]) -> dict[str, Any]:
    out = {"method": method, "fast_conduction_score": row["fast_conduction_score"]}
    if method.startswith("t65"):
        out.update(
            {
                "speed_error": row["speed_error_t65"],
                "peak_time_mae": row["peak_time_mae_t65"],
                "rmse": row["rmse_t65"],
            }
        )
    elif method.startswith("t66"):
        out.update(
            {
                "speed_error": row["speed_error_t66"],
                "peak_time_mae": row["peak_time_mae_t66"],
                "rmse": row["rmse_t66"],
            }
        )
    else:
        out.update(
            {
                "speed_error": row["speed_error_t67"],
                "peak_time_mae": row["peak_time_mae_t67"],
                "rmse": row["rmse_t67"],
                "fixed_lag": row["fixed_lag"],
                "process_noise": row["process_noise"],
                "measurement_noise": row["measurement_noise"],
                "transition_kind": row.get("transition_kind"),
                "transition_velocity": row.get("transition_velocity"),
            }
        )
    return out


def _reference_delta(
    method: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    method_key: str,
    reference_key: str,
) -> dict[str, Any]:
    method_metrics = _metric_triplet(method, method_key)
    reference_metrics = _metric_triplet(reference, reference_key)
    return {
        "method": method["method"],
        "reference": reference["method"],
        "rmse_delta_reference_minus_method": float(reference_metrics["rmse"])
        - float(method_metrics["rmse"]),
        "speed_error_delta_reference_minus_method": float(reference_metrics["speed"])
        - float(method_metrics["speed"]),
        "peak_time_mae_delta_reference_minus_method": float(reference_metrics["peak"])
        - float(method_metrics["peak"]),
        "onset_time_mae_delta_reference_minus_method": float(reference_metrics["onset"])
        - float(method_metrics["onset"]),
        "fast_conduction_score_delta_reference_minus_method": float(
            reference["fast_conduction_score"]
        )
        - float(method["fast_conduction_score"]),
    }


def _metric_triplet(row: Mapping[str, Any], key: str) -> dict[str, float]:
    return {
        "rmse": float(row[f"rmse_{key}"]),
        "speed": float(row[f"speed_error_{key}"]),
        "peak": float(row[f"peak_time_mae_{key}"]),
        "onset": float(row[f"onset_time_mae_{key}"]),
    }


def _markdown_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    config = payload["config"]
    best_t66 = summary["best_t66_tv_huber"]
    best_t67 = summary["best_t67_kalman"]
    lines = [
        "# Dynamic Sweep: T65 4D GN vs T66 TV/Huber vs T67 Kalman",
        "",
        f"- schema: `{payload['schema']}`",
        f"- created_utc: `{payload['created_utc']}`",
        f"- fixture: `{config['fixture']}` / `{config['domain']}`",
        f"- n_cells/n_frames/n_measurements: `{config['n_cells']}/{config['n_frames']}/{config['n_measurements']}`",
        f"- lambda_s: `{config['lambda_s']}`",
        f"- temporal_order: `{config['temporal_order']}`",
        f"- noise_std: `{config['noise_std']}`",
        f"- gate: peak_delay<=`{config['peak_delay_limit']}`, rmse_ratio<=`{config['rmse_ratio_limit']}`",
        "",
        "## Recommended Fast-Conduction Regions",
        "",
        f"- T66 lambda_t range: `{_range_text(summary['recommended_lambda_t_range'])}`",
        f"- T66 huber_delta range: `{_range_text(summary['recommended_huber_delta_range'])}`",
        f"- T66 gate-passing rows: `{summary['candidate_count']}/{summary['t66_row_count']}`",
        f"- T67 fixed_lag range: `{_int_range_text(summary['recommended_kalman_fixed_lag_range'])}`",
        f"- T67 process_noise Q range: `{_range_text(summary['recommended_kalman_process_noise_range'])}`",
        f"- T67 measurement_noise R range: `{_range_text(summary['recommended_kalman_measurement_noise_range'])}`",
        f"- T67 transition modes: `{', '.join(summary['recommended_kalman_transition_modes'])}`",
        f"- T67 propagation velocity range: `{_optional_range_text(summary['recommended_kalman_velocity_range'])}`",
        f"- T67 gate-passing rows: `{summary['t67_candidate_count']}/{summary['t67_row_count']}`; if zero, the range is the best-scored fallback region.",
        f"- best overall score: `{summary['best_overall_by_fast_conduction_score']['method']}`",
        "",
        "## Best Points",
        "",
        "| method | params | score | speed err | peak MAE | onset MAE | RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
        _best_t65_markdown_row(summary["best_t65_l2"]),
        _best_t66_markdown_row(best_t66),
        _best_t67_markdown_row(best_t67),
        "",
        "## Top T67 Kalman Lag/Q/R Rows",
        "",
        "| A | lag | Q | R | latency | score | speed err | peak MAE | onset MAE | RMSE ratio vs T65 | speed delta vs T66 | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in sorted(
        payload["t67_kalman_rows"],
        key=lambda item: (
            not item["passes_fast_conduction_gate"],
            item["fast_conduction_score"],
        ),
    )[:12]:
        lines.append(
            "| {transition} | {lag} | {q} | {r} | {latency} | {score} | {speed} | {peak} | {onset} | {rmse_ratio} | {speed_delta} | {passed} |".format(
                transition=_transition_label(row),
                lag=int(row["fixed_lag"]),
                q=_fmt(row["process_noise"]),
                r=_fmt(row["measurement_noise"]),
                latency=_fmt(row["latency_seconds"]),
                score=_fmt(row["fast_conduction_score"]),
                speed=_fmt(row["speed_error_t67"]),
                peak=_fmt(row["peak_time_mae_t67"]),
                onset=_fmt(row["onset_time_mae_t67"]),
                rmse_ratio=_fmt(row["rmse_ratio_t67_over_best_t65"]),
                speed_delta=_fmt(row["speed_error_delta_t66_minus_t67"]),
                passed="yes" if row["passes_fast_conduction_gate"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Top T66 TV/Huber Rows",
            "",
            "| lambda_t | huber_delta | score | speed err | peak MAE | onset MAE | RMSE ratio | pass |",
            "|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in sorted(
        payload["t66_rows"],
        key=lambda item: (
            not item["passes_fast_conduction_gate"],
            item["fast_conduction_score"],
        ),
    )[:12]:
        lines.append(
            "| {lambda_t} | {huber_delta} | {score} | {speed} | {peak} | {onset} | {rmse_ratio} | {passed} |".format(
                lambda_t=_fmt(row["lambda_t"]),
                huber_delta=_fmt(row["huber_delta"]),
                score=_fmt(row["fast_conduction_score"]),
                speed=_fmt(row["speed_error_t66"]),
                peak=_fmt(row["peak_time_mae_t66"]),
                onset=_fmt(row["onset_time_mae_t66"]),
                rmse_ratio=_fmt(row["rmse_ratio_t66_over_t65"]),
                passed="yes" if row["passes_fast_conduction_gate"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- T65 baseline is L2 spatiotemporal GN at each `lambda_t`.",
            "- T66 uses Huber IRLS over spatial graph differences and temporal differences.",
            "- T67 uses the cached Laplace RM as a state observation, then applies online Kalman filtering plus optional fixed-lag RTS smoothing.",
            "- Lower score favours propagation-speed, peak-time, onset-time fidelity first; RMSE ratio is a guard, not the main objective.",
            "",
        ]
    )
    return "\n".join(lines)


def _best_t65_markdown_row(row: Mapping[str, Any]) -> str:
    return "| T65 4D GN | lambda_t={lambda_t} | {score} | {speed} | {peak} | {onset} | {rmse} |".format(
        lambda_t=_fmt(row["lambda_t"]),
        score=_fmt(row["fast_conduction_score"]),
        speed=_fmt(row["speed_error_t65"]),
        peak=_fmt(row["peak_time_mae_t65"]),
        onset=_fmt(row["onset_time_mae_t65"]),
        rmse=_fmt(row["rmse_t65"]),
    )


def _best_t66_markdown_row(row: Mapping[str, Any]) -> str:
    return "| T66 TV/Huber | lambda_t={lambda_t}, delta={delta} | {score} | {speed} | {peak} | {onset} | {rmse} |".format(
        lambda_t=_fmt(row["lambda_t"]),
        delta=_fmt(row["huber_delta"]),
        score=_fmt(row["fast_conduction_score"]),
        speed=_fmt(row["speed_error_t66"]),
        peak=_fmt(row["peak_time_mae_t66"]),
        onset=_fmt(row["onset_time_mae_t66"]),
        rmse=_fmt(row["rmse_t66"]),
    )


def _best_t67_markdown_row(row: Mapping[str, Any]) -> str:
    return "| T67 Kalman | A={transition}, lag={lag}, Q={q}, R={r} | {score} | {speed} | {peak} | {onset} | {rmse} |".format(
        transition=_transition_label(row),
        lag=int(row["fixed_lag"]),
        q=_fmt(row["process_noise"]),
        r=_fmt(row["measurement_noise"]),
        score=_fmt(row["fast_conduction_score"]),
        speed=_fmt(row["speed_error_t67"]),
        peak=_fmt(row["peak_time_mae_t67"]),
        onset=_fmt(row["onset_time_mae_t67"]),
        rmse=_fmt(row["rmse_t67"]),
    )


def _transition_label(row: Mapping[str, Any]) -> str:
    kind = str(row.get("transition_kind", "identity"))
    velocity = row.get("transition_velocity")
    if velocity is None:
        return kind
    return f"{kind}@v={_fmt(velocity)}"


def _parse_grid(text: str) -> tuple[float, ...]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    return tuple(values)


def _parse_int_grid(text: str) -> tuple[int, ...]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    return tuple(values)


def _parse_text_grid(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(text).split(",") if item.strip())


def _positive_grid(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out:
        raise ValueError(f"{name} must not be empty.")
    if any((not np.isfinite(value) or value <= 0.0) for value in out):
        raise ValueError(f"{name} entries must be finite and positive.")
    return out


def _nonnegative_int_grid(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    out = tuple(int(value) for value in values)
    if not out:
        raise ValueError(f"{name} must not be empty.")
    if any(value < 0 for value in out):
        raise ValueError(f"{name} entries must be non-negative.")
    return out


def _transition_mode_grid(
    values: Sequence[str], *, name: str = "kalman_transition"
) -> tuple[str, ...]:
    out = tuple(str(value).strip().lower() for value in values)
    if not out:
        raise ValueError(f"{name} must not be empty.")
    allowed = {"identity", "propagation"}
    bad = [value for value in out if value not in allowed]
    if bad:
        raise ValueError(f"{name} entries must be one of {sorted(allowed)}; got {bad}.")
    seen: set[str] = set()
    unique: list[str] = []
    for value in out:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return tuple(unique)


def _range(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(arr)), "max": float(np.max(arr))}


def _optional_range(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return _range(values)


def _int_range(values: Sequence[int]) -> dict[str, int]:
    arr = np.asarray(values, dtype=np.int64)
    return {"min": int(np.min(arr)), "max": int(np.max(arr))}


def _range_text(value: Mapping[str, Any]) -> str:
    return f"{_fmt(value['min'])}..{_fmt(value['max'])}"


def _optional_range_text(value: Mapping[str, Any] | None) -> str:
    return "none" if value is None else _range_text(value)


def _int_range_text(value: Mapping[str, Any]) -> str:
    return f"{int(value['min'])}..{int(value['max'])}"


def _fmt(value: Any) -> str:
    return f"{float(value):.6g}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_sweep(
        n_cells=args.n_cells,
        n_frames=args.n_frames,
        n_measurements=args.n_measurements,
        lambda_s=args.lambda_s,
        lambda_t_values=_parse_grid(args.lambda_t),
        huber_delta_values=_parse_grid(args.huber_delta),
        temporal_order=args.temporal_order,
        noise_std=args.noise_std,
        seed=args.seed,
        onset_fraction=args.onset_fraction,
        max_outer_iterations=args.max_outer_iterations,
        tolerance=args.tolerance,
        peak_delay_limit=args.peak_delay_limit,
        rmse_ratio_limit=args.rmse_ratio_limit,
        kalman_lag_values=_parse_int_grid(args.kalman_lag),
        kalman_process_noise_values=_parse_grid(args.kalman_process_noise),
        kalman_measurement_noise_values=_parse_grid(args.kalman_measurement_noise),
        kalman_initial_covariance=args.kalman_initial_covariance,
        kalman_transition_modes=_parse_text_grid(args.kalman_transition),
        kalman_velocity_values=_parse_grid(args.kalman_velocity),
    )
    json_path = write_payload(args.output_json, payload)
    md_path = write_markdown_report(args.output_md, payload)
    summary = payload["summary"]
    print(f"[OK] dynamic T65/T66/T67 sweep saved: {json_path}")
    print(f"[OK] dynamic T65/T66/T67 report saved: {md_path}")
    print(
        "[OK] recommended T67 lag="
        f"{_int_range_text(summary['recommended_kalman_fixed_lag_range'])}, "
        f"Q={_range_text(summary['recommended_kalman_process_noise_range'])}, "
        f"R={_range_text(summary['recommended_kalman_measurement_noise_range'])} "
        f"A={','.join(summary['recommended_kalman_transition_modes'])} "
        f"(gate candidates={summary['t67_candidate_count']}/{summary['t67_row_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
