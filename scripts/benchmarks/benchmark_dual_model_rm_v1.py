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
    load_greit_rm,
    load_rm_artifact as load_hdf5_rm_artifact,
    reconstruct_difference_batch,
    rm_signature,
    write_forward_rm_benchmark_artifact,
    write_greit_metrics_artifact,
    write_rm_artifact as write_hdf5_rm_artifact,
)
from pyeidors.perf.gpu_kernels import prepare_rm_matmul
from pyeidors.io._json import json_ready as _jsonable


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
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument(
        "--devices",
        default="cpu,auto",
        help="Comma-separated online devices to try: cpu, auto, cuda.",
    )
    parser.add_argument(
        "--forward-reference",
        type=Path,
        default=Path(
            "reports/benchmarks/forward_spd_gamg_cuda_48e_repeat2_20260421.json"
        ),
        help="Existing real forward-solver benchmark JSON to cite in the report.",
    )
    parser.add_argument(
        "--lazy-reference",
        type=Path,
        default=Path(
            "reports/runtime_benchmarks/lazy_48e_spd_gamg_cuda_b4_20260421/summary.json"
        ),
        help="Existing real 48e context/Jacobian benchmark JSON to cite in the report.",
    )
    parser.add_argument(
        "--previous-greit-reference",
        type=Path,
        default=Path(
            "reports/runtime_benchmarks/greit_48e_5936_rm_layer_20260421/summary.json"
        ),
        help="Previous GREIT RM-layer benchmark JSON used for hot-path speedup comparison.",
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


def _write_rm_artifact(path: Path, rm: np.ndarray, metadata: dict[str, Any]) -> Path:
    return write_hdf5_rm_artifact(
        path,
        rm,
        metadata=metadata,
    )


def _load_rm_artifact(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = load_hdf5_rm_artifact(path)
    return artifact.rm, dict(artifact.metadata)


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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _forward_reference_summary(
    *,
    forward_path: Path,
    lazy_path: Path,
) -> dict[str, Any]:
    forward = _read_json(forward_path)
    lazy = _read_json(lazy_path)
    summary: dict[str, Any] = {
        "forward_reference_path": str(forward_path),
        "lazy_reference_path": str(lazy_path),
        "forward_reference_found": forward is not None,
        "lazy_reference_found": lazy is not None,
    }
    if forward is not None:
        solver = dict(forward.get("forward_solver_benchmark", {}) or {})
        mesh = dict(forward.get("mesh_info", {}) or {})
        summary["forward_solver"] = {
            "solver_preset": solver.get("solver_preset"),
            "ksp_type": solver.get("ksp_type"),
            "pc_type": solver.get("pc_type"),
            "pc_subtype": solver.get("pc_subtype"),
            "mat_type": solver.get("mat_type"),
            "vec_type": solver.get("vec_type"),
            "dense_mat_type": solver.get("dense_mat_type"),
            "mat_solve_effective": solver.get("mat_solve_effective"),
            "petsc_device_effective": solver.get("petsc_device_effective"),
            "setup_seconds": solver.get("setup_seconds"),
            "solve_seconds": solver.get("solve_seconds"),
            "n_patterns": solver.get("n_patterns"),
        }
        summary["forward_mesh"] = {
            "nodes": mesh.get("nodes"),
            "elements": mesh.get("elements"),
            "potential_dofs": mesh.get("potential_dofs"),
            "sigma_dofs": mesh.get("sigma_dofs"),
            "mesh_family": mesh.get("mesh_family"),
            "geometry_version": mesh.get("geometry_version"),
        }
    if lazy is not None:
        cold = dict(lazy.get("cold_context", {}) or {})
        cache_build = dict(cold.get("cache_build_seconds", {}) or {})
        backend = dict(cold.get("petsc_backend_info", {}) or {})
        summary["lazy_context"] = {
            "context_build_seconds": cold.get("context_build_seconds"),
            "mesh_cache_hit": cold.get("mesh_cache_hit"),
            "mesh_cache_layer": cold.get("mesh_cache_layer"),
            "jacobian_shape": cold.get("jacobian_shape"),
            "n_meas_total": cold.get("n_meas_total"),
            "torch_device": cold.get("torch_device"),
            "cache_build_seconds": {
                "mesh": cache_build.get("mesh"),
                "base_meas": cache_build.get("base_meas"),
                "jacobian": cache_build.get("jacobian"),
                "operator_noser": cache_build.get("operator_noser"),
                "operator_precond": cache_build.get("operator_precond"),
            },
            "forward_policy": {
                "solver_preset": backend.get("solver_preset"),
                "petsc_device_effective": backend.get("petsc_device_effective"),
                "forward_mat_solve_effective": backend.get(
                    "forward_mat_solve_effective"
                ),
                "pc_type": backend.get("pc_type"),
                "pc_gamg_type": backend.get("pc_gamg_type"),
            },
        }
    return summary


def _previous_greit_summary(
    path: Path,
    *,
    current_online: dict[str, Any],
) -> dict[str, Any]:
    previous = _read_json(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "found": previous is not None,
    }
    if previous is None:
        return summary
    warm = dict(previous.get("warm_seconds", {}) or {})
    summary["previous_warm_seconds"] = {
        "apply_cpu_1_frame": warm.get("apply_cpu_1_frame"),
        "apply_cpu_512_frames": warm.get("apply_cpu_512_frames"),
        "apply_auto_1_frame": warm.get("apply_auto_1_frame"),
        "apply_auto_512_frames": warm.get("apply_auto_512_frames"),
        "artifact_load": warm.get("artifact_load"),
    }
    comparisons: dict[str, Any] = {}
    for device in ("cpu", "cuda", "auto"):
        current = current_online.get(device)
        if not isinstance(current, dict) or current.get("error"):
            continue
        current_batch = current.get("apply_batch_seconds")
        old_key = (
            "apply_auto_512_frames"
            if device in {"cuda", "auto"}
            else "apply_cpu_512_frames"
        )
        old_batch = warm.get(old_key)
        try:
            if (
                old_batch is not None
                and current_batch is not None
                and float(current_batch) > 0.0
            ):
                comparisons[device] = {
                    "previous_batch_seconds": float(old_batch),
                    "current_batch_seconds": float(current_batch),
                    "speedup": float(old_batch) / float(current_batch),
                }
        except (TypeError, ValueError):
            continue
    summary["comparisons"] = comparisons
    return summary


def _apply_one_step_rm(
    *,
    rm: np.ndarray,
    frames: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    device: str,
    dtype: str,
) -> tuple[dict[str, Any], np.ndarray]:
    handle, prepare_seconds = _timed(
        lambda: prepare_rm_matmul(
            rm,
            device=device,
            dtype=dtype,
            cache_key=f"one-step:{device}:{dtype}",
        )
    )
    one_frame = frames[:1]
    result_1, seconds_1 = _timed(
        lambda: reconstruct_difference_batch(
            handle,
            one_frame,
            normalize=True,
            v_ref=reference,
            channel_mask=mask,
            measurement_weights=weights,
            device=device,
            dtype=dtype,
            return_metadata=True,
        )
    )
    result_n, seconds_n = _timed(
        lambda: reconstruct_difference_batch(
            handle,
            frames,
            normalize=True,
            v_ref=reference,
            channel_mask=mask,
            measurement_weights=weights,
            device=device,
            dtype=dtype,
            return_metadata=True,
        )
    )
    entry = {
        "prepare_seconds": prepare_seconds,
        "apply_1_frame_seconds": seconds_1,
        "apply_batch_seconds": seconds_n,
        "apply_batch_n_frames": int(frames.shape[0]),
        "metadata_1_frame": _jsonable(dict(result_1.metadata)),
        "metadata_batch": _jsonable(dict(result_n.metadata)),
        "output_norm_1_frame": float(np.linalg.norm(np.asarray(result_1.values))),
        "output_norm_batch": float(np.linalg.norm(np.asarray(result_n.values))),
    }
    return entry, np.asarray(result_1.values).reshape(-1)


def _apply_greit_rm(
    *,
    greit,
    frames: np.ndarray,
    reference: np.ndarray,
    device: str,
    dtype: str,
) -> tuple[dict[str, Any], np.ndarray]:
    prepared, prepare_seconds = _timed(
        lambda: greit.prepare_online(
            device=device,
            dtype=dtype,
            cache_key=f"greit:{device}:{dtype}",
        )
    )
    result_1, seconds_1 = _timed(
        lambda: prepared.reconstruct(
            frames[:1],
            normalize=True,
            v_ref=reference,
            device=device,
            dtype=dtype,
            return_metadata=True,
        )
    )
    result_n, seconds_n = _timed(
        lambda: prepared.reconstruct(
            frames,
            normalize=True,
            v_ref=reference,
            device=device,
            dtype=dtype,
            return_metadata=True,
        )
    )
    entry = {
        "prepare_seconds": prepare_seconds,
        "apply_1_frame_seconds": seconds_1,
        "apply_batch_seconds": seconds_n,
        "apply_batch_n_frames": int(frames.shape[0]),
        "metadata_1_frame": _jsonable(dict(result_1.metadata)),
        "metadata_batch": _jsonable(dict(result_n.metadata)),
        "output_norm_1_frame": float(np.linalg.norm(np.asarray(result_1.values))),
        "output_norm_batch": float(np.linalg.norm(np.asarray(result_n.values))),
    }
    return entry, np.asarray(result_1.values).reshape(-1)


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> Path:
    lines = [
        "# 48e/5936 Dual-Model RM Runtime Report",
        "",
        f"- schema: `{payload['schema']}`",
        f"- scope: {payload['scope']}",
        f"- generated: {payload['timestamp_utc']}",
        f"- git: `{payload['git_commit']}`",
        "",
        "## Forward Reference",
        "",
    ]
    forward_ref = payload.get("forward_reference", {})
    solver = forward_ref.get("forward_solver", {})
    lazy = forward_ref.get("lazy_context", {})
    if solver:
        lines.extend(
            [
                f"- solver: `{solver.get('solver_preset')}` / `{solver.get('ksp_type')}` + `{solver.get('pc_type')}`",
                f"- PETSc device: `{solver.get('petsc_device_effective')}`; matSolve: `{solver.get('mat_solve_effective')}`",
                f"- setup seconds: {_format_seconds(solver.get('setup_seconds'))}; solve seconds: {_format_seconds(solver.get('solve_seconds'))}",
            ]
        )
    if lazy:
        cache = lazy.get("cache_build_seconds", {})
        lines.extend(
            [
                f"- lazy context seconds: {_format_seconds(lazy.get('context_build_seconds'))}",
                f"- lazy jacobian seconds: {_format_seconds(cache.get('jacobian'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## RM Build And Load",
            "",
            "| algorithm | rm build s | artifact load s |",
            "|---|---:|---:|",
        ]
    )
    artifact_load = payload.get("artifact_load", {})
    for name in ("noser", "laplace", "greit"):
        if name == "greit":
            build_s = payload["timings_seconds"].get("greit_rm_build")
        else:
            build_s = payload["rm_builds"][name].get("seconds")
        load_s = artifact_load.get(name, {}).get("seconds")
        lines.append(
            f"| {name} | {_format_seconds(build_s)} | {_format_seconds(load_s)} |"
        )
    lines.extend(
        [
            "",
            "## Online Apply",
            "",
            "| algorithm | device | prepare s | 1 frame s | batch frames | batch s | effective device | resident |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for algorithm, by_device in payload.get("online_apply", {}).items():
        for device, entry in by_device.items():
            if entry.get("error"):
                lines.append(
                    f"| {algorithm} | {device} | n/a | n/a | n/a | n/a | error | {entry['error']} |"
                )
                continue
            meta = entry.get("metadata_batch", {})
            lines.append(
                "| {algorithm} | {device} | {prep} | {one} | {frames} | {batch} | {effective} | {resident} |".format(
                    algorithm=algorithm,
                    device=device,
                    prep=_format_seconds(entry.get("prepare_seconds")),
                    one=_format_seconds(entry.get("apply_1_frame_seconds")),
                    frames=int(entry.get("apply_batch_n_frames", 0)),
                    batch=_format_seconds(entry.get("apply_batch_seconds")),
                    effective=meta.get("device_effective", ""),
                    resident=meta.get("rm_matrix_resident", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Previous GREIT Baseline",
            "",
        ]
    )
    previous = payload.get("previous_greit_reference", {})
    comparisons = previous.get("comparisons", {})
    if comparisons:
        lines.extend(
            [
                "| device | previous 512-frame s | current 512-frame s | speedup |",
                "|---|---:|---:|---:|",
            ]
        )
        for device, item in comparisons.items():
            lines.append(
                f"| {device} | {_format_seconds(item.get('previous_batch_seconds'))} | "
                f"{_format_seconds(item.get('current_batch_seconds'))} | "
                f"{float(item.get('speedup', 0.0)):.2f}x |"
            )
    else:
        lines.append("- previous GREIT reference unavailable")
    lines.extend(
        [
            "",
            "## GREIT Metrics",
            "",
            "| metric | value |",
            "|---|---:|",
        ]
    )
    for key in payload.get("greit", {}).get("metric_keys", []):
        lines.append(f"| {key} | {payload['greit']['metrics'].get(key)} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


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

    coarse_j, coarse_j_seconds = _timed(
        lambda: _build_synthetic_coarse_j(
            coarse.cell_centers(),
            n_measurements=args.n_measurements,
            n_elec=args.n_elec,
            n_rings=args.n_rings,
        )
    )
    fine_j, fine_j_seconds = _timed(lambda: _coarse_j_to_fine_j(coarse_j, dual))
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
    rm_artifacts: dict[str, Path] = {}
    for mode, regularization in (
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
        rm_builds[mode] = {
            "seconds": seconds,
            "metadata": _jsonable(dict(result.metadata)),
        }
        rm_artifacts[mode] = _write_rm_artifact(
            out_dir / f"one_step_{mode}_rm.h5",
            result.rm,
            {
                "algorithm": f"one-step-{mode}",
                "metadata": dict(result.metadata),
            },
        )

    greit_path = out_dir / "greit_rm.h5"
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
    artifact_load: dict[str, dict[str, Any]] = {}
    loaded_rm_objects: dict[str, np.ndarray] = {}
    for mode, path in rm_artifacts.items():
        loaded, load_seconds = _timed(lambda path=path: _load_rm_artifact(path))
        loaded_rm_objects[mode] = loaded[0]
        artifact_load[mode] = {
            "seconds": load_seconds,
            "path": str(path),
            "metadata": _jsonable(loaded[1]),
        }
    loaded_greit, greit_load_seconds = _timed(lambda: load_greit_rm(greit_path))
    artifact_load["greit"] = {
        "seconds": greit_load_seconds,
        "path": str(greit_path),
        "metadata": _jsonable(dict(loaded_greit.metadata)),
    }

    target, target_mask = _target_vector(coarse)
    normalized_delta = 0.04 * (coarse_j @ target)
    normalized_delta[mask] = 0.0
    reference = np.linspace(2.0, 4.0, args.n_measurements)
    frames = _measurement_frames(
        reference=reference,
        normalized_delta=normalized_delta,
        n_frames=args.n_frames,
    )

    online: dict[str, Any] = {"noser": {}, "laplace": {}, "greit": {}}
    one_frame_recons: dict[str, np.ndarray] = {}
    for device in devices:
        for mode in ("noser", "laplace"):
            try:
                entry, one_frame = _apply_one_step_rm(
                    rm=loaded_rm_objects[mode],
                    frames=frames,
                    reference=reference,
                    mask=mask,
                    weights=weights,
                    device=device,
                    dtype=args.dtype,
                )
                online[mode][device] = entry
                one_frame_recons.setdefault(mode, one_frame)
            except Exception as exc:
                online[mode][device] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        try:
            entry, one_frame = _apply_greit_rm(
                greit=loaded_greit,
                frames=frames,
                reference=reference,
                device=device,
                dtype=args.dtype,
            )
            online["greit"][device] = entry
            one_frame_recons.setdefault("greit", one_frame)
        except Exception as exc:
            online["greit"][device] = {
                "error": f"{type(exc).__name__}: {exc}",
            }

    greit_recon = one_frame_recons.get(
        "greit",
        np.asarray(
            loaded_greit.reconstruct(
                frames[0],
                normalize=True,
                v_ref=reference,
                device="cpu",
            )
        ).reshape(-1),
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
                entry["apply_batch_seconds"]
                for by_device in online.values()
                for entry in by_device.values()
                if entry.get("apply_batch_seconds") is not None
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
        "scope": "48e/5936 RM-layer benchmark; forward/J cold path cited from real spd_gamg CUDA reports",
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
            "dtype": str(args.dtype),
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
            "synthetic_coarse_j_build": float(coarse_j_seconds),
            "fine_j_projection": float(fine_j_seconds),
            "dual_operator_materialize_dense_check": float(materialize_seconds),
            "greit_rm_build": float(greit_seconds),
        },
        "forward_reference": _forward_reference_summary(
            forward_path=args.forward_reference,
            lazy_path=args.lazy_reference,
        ),
        "dual_mesh": _jsonable(dual.summary()),
        "operator_dense_check": {
            "max_abs_error": max_operator_error,
        },
        "rm_builds": rm_builds,
        "artifact_load": artifact_load,
        "online_apply": online,
        "previous_greit_reference": _previous_greit_summary(
            args.previous_greit_reference,
            current_online=online["greit"],
        ),
        "greit": {
            "metadata": _jsonable(dict(greit.metadata)),
            "metrics": _jsonable(metrics),
            "metric_keys": list(GREIT_METRIC_KEYS),
        },
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "markdown_report": str(out_dir / "README.md"),
            "forward_rm_benchmark": str(rm_benchmark_path),
            "greit_rm": str(greit_path),
            "one_step_noser_rm": str(rm_artifacts["noser"]),
            "one_step_laplace_rm": str(rm_artifacts["laplace"]),
            "greit_metrics": str(metrics_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown_report(out_dir / "README.md", payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
