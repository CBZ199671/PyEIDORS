#!/usr/bin/env python3
"""Benchmark the v1 dual-model reconstruction-matrix path.

This benchmark intentionally targets the EIDORS-style v1 online path:

    fine forward mesh + coarse inverse grid + offline RM + online RM @ dv

It uses a deterministic synthetic linearized CEM-like Jacobian so the RM
layer can be timed without re-entering the expensive 48-electrode FEniCSx
Jacobian cold path. Real-CEM coverage is guarded separately by the V25 smoke.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.data.channels import bad_channel_mask
from pyeidors.inverse import (
    GREIT_METRIC_KEYS,
    CellMesh,
    DualMesh,
    DualMeshJacobianOperator,
    VoxelGrid,
    build_3d_greit_rm,
    build_one_step_rm,
    graph_laplacian,
    greit_metrics,
    reconstruct_difference_batch,
    rm_signature,
    write_forward_rm_benchmark_artifact,
    write_greit_metrics_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/runtime_benchmarks/dual_model_rm_v1_20260421"),
    )
    parser.add_argument("--coarse-shape", default="6,6,4")
    parser.add_argument("--fine-per-coarse", type=int, default=4)
    parser.add_argument("--n-measurements", type=int, default=256)
    parser.add_argument("--n-elec", type=int, default=48)
    parser.add_argument("--n-rings", type=int, default=3)
    parser.add_argument("--n-frames", type=int, default=512)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0e-2)
    parser.add_argument("--noise-figure", type=float, default=1.0e-2)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument(
        "--devices",
        default="cpu,auto",
        help="Comma-separated online devices to try: cpu, auto, cuda.",
    )
    return parser.parse_args()


def _parse_shape(raw: str) -> tuple[int, int, int]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError("--coarse-shape must contain three positive integers.")
    return values


def _cell_mesh_from_centers(centers: np.ndarray, *, name: str) -> CellMesh:
    centers = np.asarray(centers, dtype=np.float64)
    offsets = np.array(
        [
            [-1.0e-3, -1.0e-3, -1.0e-3],
            [1.0e-3, 0.0, 0.0],
            [0.0, 1.0e-3, 0.0],
            [0.0, 0.0, 1.0e-3],
        ],
        dtype=np.float64,
    )
    coordinates: list[np.ndarray] = []
    cells: list[list[int]] = []
    for center in centers:
        start = len(coordinates)
        coordinates.extend(center + offsets)
        cells.append(list(range(start, start + offsets.shape[0])))
    return CellMesh(np.asarray(coordinates), np.asarray(cells), name=name)


def _build_fine_mesh(coarse: VoxelGrid, *, fine_per_coarse: int) -> CellMesh:
    if fine_per_coarse <= 0:
        raise ValueError("fine_per_coarse must be positive.")
    rng_offsets = np.array(
        [
            [-0.22, -0.10, -0.05],
            [0.20, 0.12, 0.06],
            [-0.08, 0.20, 0.12],
            [0.10, -0.18, -0.10],
            [0.00, 0.00, 0.00],
            [0.18, -0.02, 0.16],
        ],
        dtype=np.float64,
    )
    if fine_per_coarse > rng_offsets.shape[0]:
        raise ValueError(
            f"fine_per_coarse is capped at {rng_offsets.shape[0]} for this benchmark."
        )
    spacing = np.asarray(coarse.spacing, dtype=np.float64)
    offsets = rng_offsets[:fine_per_coarse] * spacing.reshape(1, -1)
    centers = np.vstack(
        [center + offset for center in coarse.cell_centers() for offset in offsets]
    )
    return _cell_mesh_from_centers(centers, name="fine-cem-surrogate")


def _electrode_positions(*, n_elec: int, n_rings: int) -> np.ndarray:
    if n_elec <= 0 or n_rings <= 0 or n_elec % n_rings:
        raise ValueError("n_elec must be divisible by n_rings.")
    per_ring = n_elec // n_rings
    levels = np.linspace(0.15, 0.85, n_rings)
    positions = []
    for z in levels:
        for idx in range(per_ring):
            theta = 2.0 * np.pi * idx / per_ring
            positions.append(
                [0.5 + 0.52 * np.cos(theta), 0.5 + 0.52 * np.sin(theta), z]
            )
    return np.asarray(positions, dtype=np.float64)


def _build_synthetic_coarse_j(
    centers: np.ndarray,
    *,
    n_measurements: int,
    n_elec: int,
    n_rings: int,
) -> np.ndarray:
    electrodes = _electrode_positions(n_elec=n_elec, n_rings=n_rings)
    diff = centers[None, :, :] - electrodes[:, None, :]
    dist2 = np.sum(diff * diff, axis=2)
    fields = 1.0 / np.sqrt(dist2 + 2.5e-3)
    fields -= fields.mean(axis=1, keepdims=True)
    fields /= np.maximum(np.linalg.norm(fields, axis=1, keepdims=True), 1.0e-12)

    rows = []
    for meas in range(n_measurements):
        a = meas % n_elec
        b = (meas * 7 + 5) % n_elec
        c = (meas * 11 + 3) % n_elec
        d = (meas * 13 + 1) % n_elec
        row = (fields[a] - fields[b]) * (fields[c] - fields[d])
        row += 0.05 * np.sin((meas + 1) * centers[:, 0])
        norm = float(np.linalg.norm(row))
        rows.append(row / max(norm, 1.0e-12))
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def _coarse_j_to_fine_j(coarse_j: np.ndarray, dual: DualMesh) -> np.ndarray:
    counts = np.asarray(dual.coarse2fine.sum(axis=0)).reshape(-1)
    inv_counts = np.diag(1.0 / np.maximum(counts, 1.0))
    return np.asarray(coarse_j @ inv_counts @ dual.coarse2fine.T.toarray())


def _target_vector(coarse: VoxelGrid) -> tuple[np.ndarray, np.ndarray]:
    centers = coarse.cell_centers()
    domain_center = np.asarray([0.55, 0.45, 0.55], dtype=np.float64)
    distance = np.linalg.norm(centers - domain_center.reshape(1, -1), axis=1)
    radius = 0.30
    mask = distance <= radius
    if not np.any(mask):
        mask[int(np.argmin(distance))] = True
    target = np.zeros(coarse.num_cells(), dtype=np.float64)
    target[mask] = 1.0
    return target, mask


def _measurement_frames(
    *,
    reference: np.ndarray,
    normalized_delta: np.ndarray,
    n_frames: int,
) -> np.ndarray:
    scales = 1.0 + 0.04 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames))
    frames = [reference * (1.0 + scale * normalized_delta) for scale in scales]
    return np.asarray(frames, dtype=np.float64)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _timed(fn):
    _sync_cuda()
    started = time.perf_counter()
    out = fn()
    _sync_cuda()
    return out, float(time.perf_counter() - started)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    args = _parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = _parse_shape(args.coarse_shape)
    devices = [part.strip().lower() for part in args.devices.split(",") if part.strip()]

    rng = np.random.default_rng(args.seed)
    coarse = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        shape=shape,
        name="coarse-inverse-voxels",
    )
    fine, dual_setup_seconds = _timed(
        lambda: _build_fine_mesh(coarse, fine_per_coarse=args.fine_per_coarse)
    )
    dual, projection_seconds = _timed(lambda: DualMesh(fine, coarse))

    coarse_j = _build_synthetic_coarse_j(
        coarse.cell_centers(),
        n_measurements=args.n_measurements,
        n_elec=args.n_elec,
        n_rings=args.n_rings,
    )
    fine_j = _coarse_j_to_fine_j(coarse_j, dual)
    operator = DualMeshJacobianOperator(dual, fine_j)
    dense_from_operator, materialize_seconds = _timed(operator.to_dense)
    max_operator_error = float(np.max(np.abs(dense_from_operator - coarse_j)))

    bad_channels = np.arange(0, args.n_measurements, 31, dtype=np.int64)
    mask = bad_channel_mask(args.n_measurements, bad_channels=bad_channels)
    weights = 0.75 + 0.5 * rng.random(args.n_measurements)
    weights[mask] = 1.0e6
    laplace = (
        graph_laplacian(coarse) + sparse.eye(coarse.num_cells(), format="csr") * 1e-9
    )

    rm_builds: dict[str, dict[str, Any]] = {}
    rm_objects = {}
    for mode, regularization in (
        ("tikhonov", None),
        ("noser", None),
        ("laplace", laplace),
    ):
        result, seconds = _timed(
            lambda mode=mode, regularization=regularization: build_one_step_rm(
                coarse_j,
                regularization=regularization,
                lambda_=args.lambda_,
                mode=mode,
                form="measurement",
                channel_mask=mask,
                measurement_weights=weights,
                return_metadata=True,
            )
        )
        rm_objects[mode] = result.rm
        rm_builds[mode] = {
            "seconds": seconds,
            "metadata": _jsonable(dict(result.metadata)),
        }

    greit_path = out_dir / "greit_rm.npz"
    greit, greit_seconds = _timed(
        lambda: build_3d_greit_rm(
            jacobian=coarse_j,
            inverse_mesh=coarse,
            target_radius=0.18,
            noise_figure=args.noise_figure,
            channel_mask=mask,
            measurement_weights=weights,
            artifact_path=greit_path,
        )
    )

    target, target_mask = _target_vector(coarse)
    normalized_delta = 0.04 * (coarse_j @ target)
    normalized_delta[mask] = 0.0
    reference = np.linspace(2.0, 4.0, args.n_measurements)
    frames = _measurement_frames(
        reference=reference,
        normalized_delta=normalized_delta,
        n_frames=args.n_frames,
    )

    online: dict[str, Any] = {}
    for device in devices:
        try:
            result, seconds = _timed(
                lambda device=device: reconstruct_difference_batch(
                    rm_objects["noser"],
                    frames,
                    normalize=True,
                    v_ref=reference,
                    channel_mask=mask,
                    measurement_weights=weights,
                    device=device,
                    return_metadata=True,
                )
            )
            online[device] = {
                "seconds": seconds,
                "metadata": _jsonable(dict(result.metadata)),
                "output_norm": float(np.linalg.norm(np.asarray(result.values))),
            }
        except Exception as exc:
            online[device] = {
                "seconds": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    greit_recon = greit.reconstruct(
        frames[0],
        normalize=True,
        v_ref=reference,
        device="cpu",
    )
    metrics = greit_metrics(
        greit_recon,
        target_mask,
        centers=coarse.cell_centers(),
    )
    metrics_path = write_greit_metrics_artifact(
        metrics,
        out_dir / "greit_metrics.json",
        metadata={"case": "dual_model_rm_v1_synthetic"},
    )
    if set(metrics) != set(GREIT_METRIC_KEYS):
        raise RuntimeError("GREIT metric key set is incomplete.")

    signature = rm_signature(
        forward_mesh_hash=f"fine-surrogate-{dual.n_fine_cells}",
        inverse_mesh_hash=f"coarse-voxel-{shape}",
        coarse2fine=dual.coarse2fine,
        electrode_geometry={"count": args.n_elec, "rings": args.n_rings},
        stim_meas_protocol={
            "family": "synthetic-cem-like",
            "n_measurements": args.n_measurements,
        },
        background={"sigma0": 1.0, "z0": 0.01},
        difference_mode="normalized",
        bad_channel_mask=mask,
        noise_covariance=weights,
        regularization_type="noser",
        hyperparameters={"lambda": args.lambda_, "noise_figure": args.noise_figure},
        device="cuda",
    )
    rm_benchmark_path = write_forward_rm_benchmark_artifact(
        out_dir / "forward_rm_benchmark.json",
        offline_rm_build_seconds=rm_builds["noser"]["seconds"],
        online_rm_apply_seconds=float(
            next(
                entry["seconds"]
                for entry in online.values()
                if entry.get("seconds") is not None
            )
        ),
        metadata={
            "rm_signature": signature,
            "case": "dual_model_rm_v1_synthetic",
        },
    )

    payload = {
        "schema": "pyeidors-dual-model-rm-v1-benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "scope": "RM-layer benchmark, synthetic linearized CEM-like Jacobian",
        "config": {
            "coarse_shape": list(shape),
            "fine_per_coarse": int(args.fine_per_coarse),
            "n_measurements": int(args.n_measurements),
            "n_elec": int(args.n_elec),
            "n_rings": int(args.n_rings),
            "n_frames": int(args.n_frames),
            "lambda": float(args.lambda_),
            "noise_figure": float(args.noise_figure),
            "seed": int(args.seed),
            "devices": devices,
        },
        "sizes": {
            "coarse_unknowns": int(coarse.num_cells()),
            "fine_cells": int(dual.n_fine_cells),
            "coarse2fine_shape": list(dual.coarse2fine.shape),
            "coarse2fine_nnz": int(dual.coarse2fine.nnz),
            "measurements": int(coarse_j.shape[0]),
            "bad_channels": int(np.count_nonzero(mask)),
            "frames": int(args.n_frames),
        },
        "timings_seconds": {
            "fine_mesh_setup": float(dual_setup_seconds),
            "coarse2fine_projection": float(projection_seconds),
            "dual_operator_materialize_dense_check": float(materialize_seconds),
            "greit_rm_build": float(greit_seconds),
        },
        "dual_mesh": _jsonable(dual.summary()),
        "operator_dense_check": {
            "max_abs_error": max_operator_error,
        },
        "rm_builds": rm_builds,
        "online_apply": online,
        "greit": {
            "metadata": _jsonable(dict(greit.metadata)),
            "metrics": _jsonable(metrics),
            "metric_keys": list(GREIT_METRIC_KEYS),
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "forward_rm_benchmark": str(rm_benchmark_path),
            "greit_rm": str(greit_path),
            "greit_metrics": str(metrics_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
