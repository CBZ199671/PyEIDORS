#!/usr/bin/env python3
"""Compare Laplace, graph-LtL, and TV-IRLS priors on a travelling-wave fixture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.inverse import (
    VoxelGrid,
    build_one_step_rm,
    graph_curvature_prior,
    graph_laplacian,
    graph_ltl,
    reconstruct_difference_batch,
    solve_tv_irls_batch,
)


SCHEMA = "pyeidors-prior-travelling-wave-benchmark-v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cells", type=int, default=32)
    parser.add_argument("--n-frames", type=int, default=24)
    parser.add_argument("--n-measurements", type=int, default=20)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.08)
    parser.add_argument("--ridge", type=float, default=1.0e-9)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--tv-irls-beta", type=float, default=1.0e-4)
    parser.add_argument("--tv-irls-iterations", type=int, default=4)
    parser.add_argument("--tv-irls-tolerance", type=float, default=1.0e-5)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/runtime_benchmarks/prior_travelling_wave.json"),
    )
    return parser.parse_args(argv)


def run_benchmark(
    *,
    n_cells: int = 32,
    n_frames: int = 24,
    n_measurements: int = 20,
    lambda_: float = 0.08,
    ridge: float = 1.0e-9,
    noise_std: float = 0.0,
    seed: int = 20260424,
    tv_irls_beta: float = 1.0e-4,
    tv_irls_iterations: int = 4,
    tv_irls_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Run the deterministic travelling-wave prior comparison."""

    fixture = build_travelling_wave_fixture(
        n_cells=n_cells,
        n_frames=n_frames,
        n_measurements=n_measurements,
        noise_std=noise_std,
        seed=seed,
    )
    mesh = fixture["mesh"]
    jacobian = fixture["jacobian"]
    truth = fixture["truth"]
    measurements = fixture["measurements"]
    positions = fixture["positions"]
    times = fixture["times"]

    priors = build_prior_payloads(mesh, ridge=float(ridge))
    method_payloads: dict[str, dict[str, Any]] = {}
    reconstructions: dict[str, np.ndarray] = {}
    laplace_matrix = priors["laplace"]["matrix"].toarray()

    for name, prior in priors.items():
        rm = build_one_step_rm(
            jacobian,
            regularization=prior["regularization"],
            lambda_=lambda_,
            mode=prior["mode"],
            return_metadata=True,
        )
        result = reconstruct_difference_batch(
            rm.rm,
            measurements,
            normalize=False,
            device="cpu",
            return_metadata=True,
        )
        values = np.asarray(result.values, dtype=np.float64)
        reconstructions[name] = values
        matrix = prior["matrix"].toarray()
        method_payloads[name] = {
            "mode": prior["mode"],
            "regularization_source": rm.metadata["regularization_source"],
            "RtR_signature_hash": rm.metadata["RtR_signature_hash"],
            "RtR_metadata": dict(rm.metadata["RtR_metadata"]),
            "rm_shape": [int(v) for v in rm.shape],
            "online_metadata": dict(result.metadata),
            "matrix_delta_fro_vs_laplace": float(
                np.linalg.norm(matrix - laplace_matrix)
            ),
            "fidelity": fidelity_metrics(
                truth,
                values,
                positions=positions,
                times=times,
            ),
        }

    tv_irls = solve_tv_irls_batch(
        jacobian,
        measurements,
        mesh,
        lambda_=lambda_,
        initial=reconstructions["laplace"],
        beta=tv_irls_beta,
        max_outer_iterations=tv_irls_iterations,
        tolerance=tv_irls_tolerance,
    )
    tv_values = np.asarray(tv_irls.values, dtype=np.float64)
    reconstructions["tv_irls"] = tv_values
    method_payloads["tv_irls"] = {
        "mode": "tv_irls",
        "regularization_source": "iterative_tv_irls",
        "RtR_signature_hash": str(tv_irls.metadata["final_RtR_signature_hashes"][-1]),
        "RtR_metadata": dict(
            tv_irls.metadata["frame_metadata"][-1]["final_prior_metadata"]
        ),
        "rm_shape": [int(tv_values.shape[-1]), int(jacobian.shape[0])],
        "online_metadata": {
            "online_hot_path": "iterative_rm_rebuild",
            "forward_solve_count": 0,
            "adjoint_solve_count": 0,
            "ksp_solve_count": 0,
            "jacobian_rebuild_count": 0,
        },
        "matrix_delta_fro_vs_laplace": None,
        "fidelity": fidelity_metrics(
            truth,
            tv_values,
            positions=positions,
            times=times,
        ),
        "tv_irls_metadata": dict(tv_irls.metadata),
    }

    laplace_recon = reconstructions["laplace"]
    for name, values in reconstructions.items():
        delta = np.asarray(values - laplace_recon, dtype=np.float64)
        method_payloads[name]["reconstruction_delta_l2_vs_laplace"] = float(
            np.linalg.norm(delta)
        )
        method_payloads[name]["reconstruction_delta_max_abs_vs_laplace"] = float(
            np.max(np.abs(delta)) if delta.size else 0.0
        )

    summary = summarize_methods(method_payloads)
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
            "tv_irls_beta": float(tv_irls_beta),
            "tv_irls_iterations": int(tv_irls_iterations),
            "tv_irls_tolerance": float(tv_irls_tolerance),
        },
        "fixture": {
            "name": "travelling_gaussian_1d",
            "mesh": {
                "type": "VoxelGrid",
                "shape": [int(v) for v in mesh.shape],
                "bounds": [float(positions[0]), float(positions[-1])],
            },
            "truth_shape": [int(v) for v in truth.shape],
            "measurement_shape": [int(v) for v in measurements.shape],
            "center_start": float(center_of_mass(truth[0], positions)),
            "center_end": float(center_of_mass(truth[-1], positions)),
        },
        "methods": method_payloads,
        "summary": summary,
    }


def build_travelling_wave_fixture(
    *,
    n_cells: int,
    n_frames: int,
    n_measurements: int,
    noise_std: float,
    seed: int,
) -> dict[str, Any]:
    if n_cells < 4:
        raise ValueError("n_cells must be at least 4.")
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2.")
    if n_measurements < 2:
        raise ValueError("n_measurements must be at least 2.")
    if noise_std < 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and non-negative.")

    mesh = VoxelGrid.from_bounds([0.0], [1.0], shape=(int(n_cells),))
    positions = np.asarray(mesh.cell_centers(), dtype=np.float64).reshape(-1)
    times = np.linspace(0.0, 1.0, int(n_frames), dtype=np.float64)
    centers = 0.18 + 0.64 * times
    width = 0.07
    truth = np.vstack(
        [np.exp(-0.5 * ((positions - center) / width) ** 2) for center in centers]
    )
    truth /= np.maximum(np.max(truth, axis=1, keepdims=True), 1.0e-12)
    amplitude = 0.8 + 0.2 * np.sin(np.pi * times)
    truth *= amplitude.reshape(-1, 1)

    jacobian = synthetic_measurement_jacobian(
        positions,
        n_measurements=int(n_measurements),
    )
    measurements = np.asarray(truth @ jacobian.T, dtype=np.float64)
    if noise_std > 0.0:
        rng = np.random.default_rng(int(seed))
        measurements = measurements + rng.normal(
            scale=float(noise_std),
            size=measurements.shape,
        )
    return {
        "mesh": mesh,
        "positions": positions,
        "times": times,
        "truth": np.ascontiguousarray(truth, dtype=np.float64),
        "jacobian": jacobian,
        "measurements": np.ascontiguousarray(measurements, dtype=np.float64),
    }


def synthetic_measurement_jacobian(
    positions: np.ndarray,
    *,
    n_measurements: int,
) -> np.ndarray:
    x = np.asarray(positions, dtype=np.float64).reshape(1, -1)
    sensor_centers = np.linspace(0.0, 1.0, int(n_measurements), dtype=np.float64)
    widths = 0.11 + 0.04 * ((np.arange(int(n_measurements)) % 3) / 2.0)
    rows = []
    for idx, center in enumerate(sensor_centers):
        left = np.exp(-0.5 * ((x - center) / widths[idx]) ** 2).reshape(-1)
        right_center = (center + 0.37) % 1.0
        right = np.exp(-0.5 * ((x - right_center) / (widths[idx] * 1.2)) ** 2).reshape(
            -1
        )
        row = left - 0.35 * right + 0.05
        row /= max(float(np.linalg.norm(row)), 1.0e-12)
        rows.append(row)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def build_prior_payloads(mesh: VoxelGrid, *, ridge: float) -> dict[str, dict[str, Any]]:
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative.")
    n_cells = int(mesh.num_cells())
    ridge_matrix = sparse.eye(n_cells, format="csr", dtype=np.float64) * float(ridge)
    laplace_matrix = (graph_laplacian(mesh) + ridge_matrix).tocsr()
    graph_ltl_matrix = (graph_ltl(mesh) + ridge_matrix).tocsr()
    curvature_matrix = (
        graph_curvature_prior(mesh).as_RtR(dense=False) + ridge_matrix
    ).tocsr()
    curvature_prior = graph_curvature_prior(mesh)
    curvature_regularization = (
        curvature_prior.as_RtR(dense=False) + ridge_matrix
    ).tocsr()
    return {
        "laplace": {
            "mode": "laplace",
            "regularization": laplace_matrix,
            "matrix": laplace_matrix,
        },
        "graph_ltl": {
            "mode": "graph_ltl",
            "regularization": graph_ltl_matrix,
            "matrix": graph_ltl_matrix,
        },
        "curvature": {
            "mode": "curvature",
            "regularization": curvature_regularization,
            "matrix": curvature_matrix,
        },
    }


def fidelity_metrics(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    *,
    positions: np.ndarray,
    times: np.ndarray,
) -> dict[str, float]:
    truth_arr = np.asarray(truth, dtype=np.float64)
    recon_arr = np.asarray(reconstruction, dtype=np.float64)
    if truth_arr.shape != recon_arr.shape:
        raise ValueError(
            f"truth and reconstruction shape mismatch: {truth_arr.shape} vs {recon_arr.shape}."
        )
    error = recon_arr - truth_arr
    center_truth = center_trace(truth_arr, positions)
    center_recon = center_trace(recon_arr, positions)
    speed_truth = speed_from_center_trace(center_truth, times)
    speed_recon = speed_from_center_trace(center_recon, times)
    peak_time_truth = peak_time_trace(truth_arr, times)
    peak_time_recon = peak_time_trace(recon_arr, times)
    peak_mask = np.max(truth_arr, axis=0) >= 0.20 * float(np.max(truth_arr))
    peak_delta = np.abs(peak_time_recon[peak_mask] - peak_time_truth[peak_mask])
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_l2_error": float(
            np.linalg.norm(error) / max(float(np.linalg.norm(truth_arr)), 1.0e-12)
        ),
        "mean_frame_correlation": float(mean_frame_correlation(truth_arr, recon_arr)),
        "amplitude_ratio": float(
            np.max(np.abs(recon_arr)) / max(float(np.max(np.abs(truth_arr))), 1.0e-12)
        ),
        "center_rmse": float(np.sqrt(np.mean((center_recon - center_truth) ** 2))),
        "center_max_abs_error": float(np.max(np.abs(center_recon - center_truth))),
        "speed_true": float(speed_truth),
        "speed_estimate": float(speed_recon),
        "speed_abs_error": float(abs(speed_recon - speed_truth)),
        "peak_time_mean_abs_error": float(
            np.mean(peak_delta) if peak_delta.size else 0.0
        ),
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


def peak_time_trace(frames: np.ndarray, times: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    idx = np.argmax(arr, axis=0)
    return np.asarray(times, dtype=np.float64)[idx]


def mean_frame_correlation(truth: np.ndarray, reconstruction: np.ndarray) -> float:
    values: list[float] = []
    for truth_frame, recon_frame in zip(truth, reconstruction, strict=True):
        truth_centered = truth_frame - np.mean(truth_frame)
        recon_centered = recon_frame - np.mean(recon_frame)
        denom = float(np.linalg.norm(truth_centered) * np.linalg.norm(recon_centered))
        if denom <= 1.0e-12:
            continue
        values.append(float(np.dot(truth_centered, recon_centered) / denom))
    return float(np.mean(values) if values else 0.0)


def summarize_methods(methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    laplace_signature = str(methods["laplace"]["RtR_signature_hash"])
    return {
        "best_rmse_method": min(
            methods,
            key=lambda name: float(methods[name]["fidelity"]["rmse"]),
        ),
        "best_center_rmse_method": min(
            methods,
            key=lambda name: float(methods[name]["fidelity"]["center_rmse"]),
        ),
        "signatures_distinct_from_laplace": {
            name: str(payload["RtR_signature_hash"]) != laplace_signature
            for name, payload in methods.items()
            if name != "laplace"
        },
        "matches_laplace_reconstruction": {
            name: bool(
                float(payload["reconstruction_delta_max_abs_vs_laplace"]) <= 1.0e-10
            )
            for name, payload in methods.items()
        },
        "max_reconstruction_delta_vs_laplace": max(
            float(payload["reconstruction_delta_max_abs_vs_laplace"])
            for payload in methods.values()
        ),
    }


def write_payload(path: Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


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
        tv_irls_beta=args.tv_irls_beta,
        tv_irls_iterations=args.tv_irls_iterations,
        tv_irls_tolerance=args.tv_irls_tolerance,
    )
    output = write_payload(args.output_json, payload)
    print(f"[OK] travelling-wave prior benchmark saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
