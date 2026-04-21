"""Benchmark lazy-adjoint 48-electrode 3D difference reconstruction runtime."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common import gn_difference_runner as runner  # noqa: E402


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_context(label: str, kwargs: dict) -> tuple[dict, dict]:
    start = time.perf_counter()
    ctx = runner.build_shared_context(**kwargs)
    elapsed = time.perf_counter() - start
    operator_bundle = ctx["operator_bundle"]
    return ctx, {
        "label": label,
        "elapsed_sec": elapsed,
        "context_build_seconds": ctx.get("context_build_seconds"),
        "cache_build_seconds": ctx.get("cache_build_seconds", {}),
        "cache_lookups": ctx.get("cache_lookups", {}),
        "mesh_cache_hit": ctx.get("mesh_cache_hit"),
        "mesh_cache_layer": ctx.get("mesh_cache_layer"),
        "mesh_cache_name": ctx.get("mesh_cache_name"),
        "n_stim": ctx.get("n_stim"),
        "n_meas_total": ctx.get("n_meas_total"),
        "jacobian_shape": tuple(int(v) for v in ctx["J"].shape),
        "jacobian_representation": ctx.get("jacobian_representation"),
        "linearized_solver_strategy": ctx.get("linearized_solver_strategy"),
        "linearized_maxiter": ctx.get("linearized_maxiter"),
        "lazy_preconditioner_mode": ctx.get("lazy_preconditioner_mode"),
        "preconditioner_info": operator_bundle.get(
            "linearized_preconditioner_info", {}
        ),
        "petsc_backend_info": ctx.get("petsc_backend_info", {}),
        "execution_profile": ctx.get("execution_profile"),
        "torch_device": ctx.get("torch_device"),
    }


def _process_context(label: str, ctx: dict, output_dir: Path) -> dict:
    vh = np.asarray(ctx["base_meas"], dtype=float)
    vi = vh * 1.001
    start = time.perf_counter()
    metrics = runner.process_frames(
        vh=vh,
        vi=vi,
        output_dir=output_dir,
        ctx=ctx,
        step_size_calib=False,
        step_size_min=1.0e-3,
        step_size_max=1.0,
        step_size_maxiter=5,
        lam=1.0e-2,
        colormap="viridis",
        colorbar_scientific=False,
        colorbar_format=None,
        transparent=False,
        write_plots=False,
        measurement_gain=1.0,
    )
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "elapsed_sec": elapsed,
        "stage_timings": metrics.get("stage_timings", {}),
        "linearized_last_solve": metrics.get("linearized_last_solve", {}),
        "linearized_preconditioner_info": metrics.get(
            "linearized_preconditioner_info", {}
        ),
        "rmse_abs": metrics.get("rmse_abs"),
        "jacobian_block_backend": metrics.get("jacobian_block_backend"),
        "execution_profile": metrics.get("execution_profile"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--linearized-maxiter", type=int, default=3)
    parser.add_argument(
        "--linearized-solver-strategy",
        default="cg_only",
        choices=("auto", "cg_only", "cg_lsmr", "lsmr", "cgls"),
    )
    parser.add_argument(
        "--lazy-preconditioner-mode",
        default="auto",
        choices=("auto", "approx", "batch_noser", "coarse", "prior"),
    )
    parser.add_argument("--skip-process", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "reports" / "runtime_benchmarks" / f"lazy_48e_cuda_{stamp}"
    )
    bench_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "mesh_dir": str(bench_dir / "meshes"),
        "mesh_name": None,
        "mesh_dim": 3,
        "mesh_height": 0.16,
        "electrode_height_ratio": 0.2,
        "z_center": 0.0,
        "electrode_level_fractions": (0.15, 0.5, 0.85),
        "refinement": 2,
        "n_elec": 16,
        "n_rings": 3,
        "radius": 0.18,
        "drive_mode": "total_current",
        "drive_value": 1.0e-5,
        "contact_impedance": 0.01,
        "electrode_coverage": 0.5,
        "electrode_layout": "ring_major",
        "measurement_protocol": "hybrid_full_3d",
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "difference_mode": "normalized",
        "difference_orientation": "target_minus_reference",
        "background_sigma": 1.0,
        "lam": 1.0e-2,
        "cache_scope": "both",
        "cache_dir": str(bench_dir / "cache"),
        "solver_mode": "fast",
        "linear_solver": "auto",
        "preconditioner": "auto",
        "rom_mode": "off",
        "lowrank_mode": "off",
        "forward_solver_preset": "spd_gamg",
        "forward_mat_solve": "off",
        "petsc_device": "cuda",
        "device": "cuda",
        "jacobian_representation": "lazy",
        "linearized_solver_strategy": args.linearized_solver_strategy,
        "linearized_maxiter": int(args.linearized_maxiter),
        "lazy_preconditioner_mode": args.lazy_preconditioner_mode,
        "lazy_diag_batch_max_measurements": 512,
        "forward_backend": "dolfinx",
        "mesh_family": "tetra",
        "geometry_version": "geomv2",
    }

    cold_ctx, cold_context = _build_context("cold_context", kwargs)
    cold_process = None
    if not args.skip_process:
        cold_process = _process_context(
            "cold_process",
            cold_ctx,
            bench_dir / "process_cold",
        )
    warm_ctx, warm_context = _build_context("warm_context", kwargs)
    warm_process = None
    if not args.skip_process:
        warm_process = _process_context(
            "warm_process",
            warm_ctx,
            bench_dir / "process_warm",
        )

    summary = {
        "bench_dir": str(bench_dir),
        "config": {
            "total_electrodes": 48,
            "rings": 3,
            "per_ring_electrodes": 16,
            "mesh_family": "tetra",
            "radius": 0.18,
            "height": 0.16,
            "refinement": 2,
            "measurement_protocol": "hybrid_full_3d",
            "linearized_solver_strategy": args.linearized_solver_strategy,
            "linearized_maxiter": int(args.linearized_maxiter),
            "lazy_preconditioner_mode": args.lazy_preconditioner_mode,
        },
        "cold_context": cold_context,
        "cold_process": cold_process,
        "warm_context": warm_context,
        "warm_process": warm_process,
    }
    summary = _jsonable(summary)
    (bench_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
