#!/usr/bin/env python3
"""Benchmark 3D EIT runtime stages for difference and absolute workflows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
from dolfinx import fem

try:  # pragma: no cover - optional in lean environments
    import torch
except Exception:  # pragma: no cover
    torch = None

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import function_get_array
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.perf.capabilities import detect_performance_capabilities
from pyeidors.perf.policy import (
    DEFAULT_ABSOLUTE_STARTUP_CACHE,
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    DEFAULT_CHOLMOD_MAX_N,
    DEFAULT_INEXACT_ETA0,
    DEFAULT_INEXACT_ETA_MAX,
    DEFAULT_INEXACT_ETA_MIN,
    DEFAULT_INEXACT_FORCING,
    DEFAULT_INEXACT_MODE,
    DEFAULT_JACOBIAN_BLOCK_CANDIDATES,
    DEFAULT_JACOBIAN_BLOCK_SIZE,
    DEFAULT_JACOBIAN_BLOCK_TUNE,
    DEFAULT_LOWRANK_ENERGY,
    DEFAULT_LOWRANK_METHOD,
    DEFAULT_LOWRANK_MODE,
    DEFAULT_LOWRANK_RANK,
    DEFAULT_PETSC_DEVICE,
    DEFAULT_ROM_MODE,
    DEFAULT_ROM_RANK_ADAPTIVE,
    DEFAULT_ROM_RANK_GLOBAL,
    DEFAULT_ROM_REFRESH_EVERY,
    DEFAULT_ROM_SNAPSHOT_SOURCE,
    parse_block_size_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import gn_difference_runner

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".pyeidors_cache") / "v2")
    parser.add_argument("--output-json", type=Path, default=Path("reports") / "benchmark_3d_runtime.json")
    parser.add_argument("--perf-report", type=Path, default=None)
    parser.add_argument("--profile-label", type=str, default="default")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.18)
    parser.add_argument("--height", type=float, default=0.16)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--lambda", dest="lam", type=float, default=1e-2)
    parser.add_argument("--background", type=float, default=1.0)
    parser.add_argument("--contact-impedance", type=float, default=1e-5)
    parser.add_argument("--solver-mode", choices=["strict", "fast"], default="fast")
    parser.add_argument(
        "--linear-solver",
        choices=["auto", "petsc-ksp", "scipy-lsmr", "pyamg-cg", "cholmod"],
        default="auto",
    )
    parser.add_argument(
        "--preconditioner",
        choices=["auto", "diag", "pyamg", "cholmod", "petsc-gamg"],
        default="auto",
    )
    parser.add_argument(
        "--fast-linear-path",
        choices=["auto", "woodbury", "pcg", "cholmod-direct", "strict"],
        default="auto",
    )
    parser.add_argument("--rom-mode", choices=["off", "auto", "on"], default=DEFAULT_ROM_MODE)
    parser.add_argument("--rom-rank-global", type=int, default=DEFAULT_ROM_RANK_GLOBAL)
    parser.add_argument("--rom-rank-adaptive", type=int, default=DEFAULT_ROM_RANK_ADAPTIVE)
    parser.add_argument("--rom-refresh-every", type=int, default=DEFAULT_ROM_REFRESH_EVERY)
    parser.add_argument(
        "--rom-snapshot-source",
        choices=["cache", "synthetic", "hybrid"],
        default=DEFAULT_ROM_SNAPSHOT_SOURCE,
    )
    parser.add_argument("--inexact-mode", choices=["off", "auto", "on"], default=DEFAULT_INEXACT_MODE)
    parser.add_argument(
        "--inexact-forcing",
        choices=["fixed", "eisenstat-walker"],
        default=DEFAULT_INEXACT_FORCING,
    )
    parser.add_argument("--inexact-eta0", type=float, default=DEFAULT_INEXACT_ETA0)
    parser.add_argument("--inexact-eta-min", type=float, default=DEFAULT_INEXACT_ETA_MIN)
    parser.add_argument("--inexact-eta-max", type=float, default=DEFAULT_INEXACT_ETA_MAX)
    parser.add_argument("--lowrank-mode", choices=["off", "auto", "on"], default=DEFAULT_LOWRANK_MODE)
    parser.add_argument("--lowrank-rank", type=int, default=DEFAULT_LOWRANK_RANK)
    parser.add_argument(
        "--lowrank-method",
        choices=["tsvd", "randomized"],
        default=DEFAULT_LOWRANK_METHOD,
    )
    parser.add_argument("--lowrank-energy", type=float, default=DEFAULT_LOWRANK_ENERGY)
    parser.add_argument(
        "--forward-mat-solve",
        choices=["auto", "off", "on"],
        default="auto",
    )
    parser.add_argument(
        "--petsc-device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_PETSC_DEVICE,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--jacobian-block-tune",
        choices=["auto", "off"],
        default=DEFAULT_JACOBIAN_BLOCK_TUNE,
    )
    parser.add_argument(
        "--jacobian-block-size",
        type=int,
        default=DEFAULT_JACOBIAN_BLOCK_SIZE,
    )
    parser.add_argument(
        "--jacobian-block-candidates",
        type=str,
        default=",".join(str(value) for value in DEFAULT_JACOBIAN_BLOCK_CANDIDATES),
    )
    parser.add_argument("--cholmod-max-n", type=int, default=DEFAULT_CHOLMOD_MAX_N)
    parser.add_argument("--cholmod-max-memory-gib", type=float, default=DEFAULT_CHOLMOD_MAX_MEMORY_GIB)
    parser.add_argument(
        "--absolute-startup-cache",
        choices=["on", "off"],
        default=DEFAULT_ABSOLUTE_STARTUP_CACHE,
    )
    parser.add_argument("--absolute-iters", type=int, default=2)
    parser.add_argument("--run-diff", choices=["on", "off"], default="on")
    parser.add_argument("--run-absolute", choices=["on", "off"], default="on")
    return parser.parse_args()


def _maybe_cuda_sync() -> None:
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        return


def _timed(name: str, fn):
    tracemalloc.start()
    _maybe_cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    _maybe_cuda_sync()
    elapsed = float(time.perf_counter() - t0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, {"stage": name, "elapsed_sec": elapsed, "peak_mib": float(peak / (1024 * 1024))}


def _build_phantom_sigma(system: EITSystem, *, background: float) -> np.ndarray:
    sigma_fn = fem.Function(system.fwd_model.V_sigma)
    sigma_fn.x.array[:] = float(background)
    coords = system.fwd_model.V_sigma.tabulate_dof_coordinates()
    center = np.array([0.35 * system.mesh.radius, 0.0, 0.0], dtype=float)
    dist = np.linalg.norm(coords[:, :3] - center[None, :], axis=1)
    sigma = function_get_array(sigma_fn).copy()
    sigma[dist <= 0.22 * system.mesh.radius] = float(background) * 1.8
    return sigma


def main() -> None:
    args = _parse_args()
    if int(args.repeat) <= 0:
        raise ValueError("--repeat must be a positive integer.")
    jacobian_block_candidates = parse_block_size_candidates(args.jacobian_block_candidates)
    cache_dir = args.cache_dir.resolve()

    def _run_once(run_index: int) -> dict:
        run_cache_dir = cache_dir if int(args.repeat) == 1 else cache_dir / f"run_{run_index:02d}"
        if run_cache_dir.exists():
            shutil.rmtree(run_cache_dir)
        run_cache_dir.mkdir(parents=True, exist_ok=True)

        run_diff = str(args.run_diff) == "on"
        run_absolute = str(args.run_absolute) == "on"
        stages: list[dict[str, float | str]] = []
        diff_metrics: dict[str, object] = {}
        cold_ctx: dict[str, object] = {}
        warm_ctx: dict[str, object] = {}
        absolute_timing: dict[str, object] = {}
        absolute_backend_info: dict[str, object] = {}
        absolute_forward_probe_stage: dict[str, float | str] | None = None
        absolute_forward_probe_repeats = 5

        if run_diff:
            cold_ctx, cold_stage = _timed(
                "diff_context_cold",
                lambda: gn_difference_runner.build_shared_context(
                    mesh_dir=str(args.mesh_dir),
                    mesh_name=None,
                    mesh_dim=3,
                    mesh_height=float(args.height),
                    electrode_height_ratio=0.2,
                    z_center=0.0,
                    refinement=int(args.refinement),
                    n_elec=int(args.n_elec),
                    radius=float(args.radius),
                    drive_value=1.0,
                    contact_impedance=float(args.contact_impedance),
                    background_sigma=float(args.background),
                    lam=float(args.lam),
                    cache_scope="both",
                    cache_dir=str(run_cache_dir),
                    solver_mode=str(args.solver_mode),
                    linear_solver=str(args.linear_solver),
                    preconditioner=str(args.preconditioner),
                    rom_mode=str(args.rom_mode),
                    rom_rank_global=int(args.rom_rank_global),
                    rom_rank_adaptive=int(args.rom_rank_adaptive),
                    rom_snapshot_source=str(args.rom_snapshot_source),
                    lowrank_mode=str(args.lowrank_mode),
                    lowrank_rank=int(args.lowrank_rank),
                    lowrank_method=str(args.lowrank_method),
                    lowrank_energy=float(args.lowrank_energy),
                    forward_mat_solve=str(args.forward_mat_solve),
                    petsc_device=str(args.petsc_device),
                    device=str(args.device),
                ),
            )
            warm_ctx, warm_stage = _timed(
                "diff_context_warm",
                lambda: gn_difference_runner.build_shared_context(
                    mesh_dir=str(args.mesh_dir),
                    mesh_name=None,
                    mesh_dim=3,
                    mesh_height=float(args.height),
                    electrode_height_ratio=0.2,
                    z_center=0.0,
                    refinement=int(args.refinement),
                    n_elec=int(args.n_elec),
                    radius=float(args.radius),
                    drive_value=1.0,
                    contact_impedance=float(args.contact_impedance),
                    background_sigma=float(args.background),
                    lam=float(args.lam),
                    cache_scope="both",
                    cache_dir=str(run_cache_dir),
                    solver_mode=str(args.solver_mode),
                    linear_solver=str(args.linear_solver),
                    preconditioner=str(args.preconditioner),
                    rom_mode=str(args.rom_mode),
                    rom_rank_global=int(args.rom_rank_global),
                    rom_rank_adaptive=int(args.rom_rank_adaptive),
                    rom_snapshot_source=str(args.rom_snapshot_source),
                    lowrank_mode=str(args.lowrank_mode),
                    lowrank_rank=int(args.lowrank_rank),
                    lowrank_method=str(args.lowrank_method),
                    lowrank_energy=float(args.lowrank_energy),
                    forward_mat_solve=str(args.forward_mat_solve),
                    petsc_device=str(args.petsc_device),
                    device=str(args.device),
                ),
            )

            vh = np.asarray(cold_ctx["base_meas"], dtype=float)
            vi = vh + 1e-4
            diff_metrics, diff_stage = _timed(
                "diff_process_one_case",
                lambda: gn_difference_runner.process_frames(
                    vh=vh,
                    vi=vi,
                    output_dir=Path("reports") / "bench_diff_tmp",
                    ctx=warm_ctx,
                    step_size_calib=False,
                    step_size_min=1e-3,
                    step_size_max=1.0,
                    step_size_maxiter=20,
                    lam=float(args.lam),
                    colormap="viridis",
                    colorbar_scientific=False,
                    colorbar_format="plain",
                    transparent=False,
                    write_plots=False,
                    measurement_gain=1.0,
                ),
            )
            stages.extend([cold_stage, warm_stage, diff_stage])

        if run_absolute:
            mesh = load_or_create_mesh(
                mesh_dir=str(args.mesh_dir),
                mesh_name=None,
                n_elec=int(args.n_elec),
                dimension=3,
                radius=float(args.radius),
                refinement=int(args.refinement),
                height=float(args.height),
                electrode_height_ratio=0.2,
                z_center=0.0,
                electrode_coverage=0.5,
            )
            pattern = PatternConfig(
                n_elec=int(args.n_elec),
                stim_pattern="{ad}",
                meas_pattern="{ad}",
                drive_mode="total_current",
                drive_value=1.0,
                geometry_scale_to_m=1.0,
            )
            system = EITSystem(
                n_elec=int(args.n_elec),
                pattern_config=pattern,
                contact_impedance=np.full(int(args.n_elec), float(args.contact_impedance), dtype=float),
                base_conductivity=float(args.background),
                regularization_type="noser",
                regularization_alpha=1.0,
                cache_scope="both",
                cache_dir=str(run_cache_dir),
                solver_mode=str(args.solver_mode),
                linear_solver=str(args.linear_solver),
                jacobian_update_every=2,
                jacobian_reuse_tol=1e-3,
                line_search_mode="fast" if str(args.solver_mode) == "fast" else "full",
                preconditioner=str(args.preconditioner),
                fast_linear_path=str(args.fast_linear_path),
                rom_mode=str(args.rom_mode),
                rom_rank_global=int(args.rom_rank_global),
                rom_rank_adaptive=int(args.rom_rank_adaptive),
                rom_refresh_every=int(args.rom_refresh_every),
                rom_snapshot_source=str(args.rom_snapshot_source),
                inexact_mode=str(args.inexact_mode),
                inexact_forcing=str(args.inexact_forcing),
                inexact_eta0=float(args.inexact_eta0),
                inexact_eta_min=float(args.inexact_eta_min),
                inexact_eta_max=float(args.inexact_eta_max),
                lowrank_mode=str(args.lowrank_mode),
                lowrank_rank=int(args.lowrank_rank),
                lowrank_method=str(args.lowrank_method),
                lowrank_energy=float(args.lowrank_energy),
                absolute_startup_cache=str(args.absolute_startup_cache) == "on",
                cholmod_max_n=int(args.cholmod_max_n),
                cholmod_max_memory_gib=float(args.cholmod_max_memory_gib),
                jacobian_block_tune=str(args.jacobian_block_tune),
                jacobian_block_size=int(args.jacobian_block_size),
                jacobian_block_candidates=jacobian_block_candidates,
                petsc_device=str(args.petsc_device),
                device=str(args.device),
                linear_backend_config={"mat_solve_mode": str(args.forward_mat_solve), "petsc_device": str(args.petsc_device)},
            )
            system.setup(mesh=mesh)
            system.reconstructor.max_iterations = int(args.absolute_iters)
            system.reconstructor.verbose = False

            baseline = system.create_homogeneous_image(conductivity=float(args.background))
            baseline_data = system.forward_solve(baseline)
            _, absolute_forward_probe_stage = _timed(
                "absolute_forward_probe",
                lambda: [system.forward_solve(baseline) for _ in range(int(absolute_forward_probe_repeats))],
            )
            stages.append(absolute_forward_probe_stage)
            phantom_sigma = _build_phantom_sigma(system, background=float(args.background))
            phantom = EITImage(elem_data=phantom_sigma, fwd_model=system.fwd_model)
            target_data = system.forward_solve(phantom)

            absolute_result, absolute_stage = _timed(
                "absolute_reconstruct",
                lambda: system.inverse_solve(data=target_data, reference_data=baseline_data),
            )
            stages.append(absolute_stage)
            if hasattr(absolute_result, "diagnostics") and isinstance(absolute_result.diagnostics, dict):
                absolute_timing = absolute_result.diagnostics.get("timing", {})
                backend = absolute_result.diagnostics.get("backend_info", {})
                if isinstance(backend, dict):
                    absolute_backend_info = backend

        if absolute_forward_probe_stage is not None:
            absolute_timing = dict(absolute_timing) if isinstance(absolute_timing, dict) else {}
            absolute_timing["forward_probe"] = float(absolute_forward_probe_stage.get("elapsed_sec", 0.0))
            absolute_timing["forward_probe_repeats"] = int(absolute_forward_probe_repeats)

        return {
            "run_index": int(run_index),
            "stages": stages,
            "stage_breakdown": {
                "difference": diff_metrics.get("stage_timings", {}) if isinstance(diff_metrics, dict) else {},
                "absolute": absolute_timing,
            },
            "difference_solver": {
                "solver_mode": diff_metrics.get("solver_mode") if isinstance(diff_metrics, dict) else None,
                "linear_solver": diff_metrics.get("linear_solver") if isinstance(diff_metrics, dict) else None,
                "preconditioner": diff_metrics.get("preconditioner") if isinstance(diff_metrics, dict) else None,
                "strict_solver_backend_requested": diff_metrics.get("strict_solver_backend_requested") if isinstance(diff_metrics, dict) else None,
                "strict_solver_backend_effective": diff_metrics.get("strict_solver_backend_effective") if isinstance(diff_metrics, dict) else None,
                "strict_memory_guard_triggered": diff_metrics.get("strict_memory_guard_triggered") if isinstance(diff_metrics, dict) else None,
                "strict_memory_guard_reason": diff_metrics.get("strict_memory_guard_reason") if isinstance(diff_metrics, dict) else None,
                "strict_dense_estimated_peak_gib": diff_metrics.get("strict_dense_estimated_peak_gib") if isinstance(diff_metrics, dict) else None,
                "strict_measurement_system_shape": diff_metrics.get("strict_measurement_system_shape") if isinstance(diff_metrics, dict) else None,
                "inverse_device_requested": diff_metrics.get("inverse_device_requested") if isinstance(diff_metrics, dict) else None,
                "inverse_device_effective": diff_metrics.get("inverse_device_effective") if isinstance(diff_metrics, dict) else None,
                "execution_profile": diff_metrics.get("execution_profile") if isinstance(diff_metrics, dict) else None,
                "jacobian_block_backend": diff_metrics.get("jacobian_block_backend") if isinstance(diff_metrics, dict) else None,
            },
            "absolute_solver": {
                "resolved_preconditioner": absolute_backend_info.get("resolved_preconditioner"),
                "fast_solver_path": absolute_backend_info.get("fast_solver_path"),
                "fast_linear_path_selected": absolute_backend_info.get("fast_linear_path_selected"),
                "fast_linear_path_reason": absolute_backend_info.get("fast_linear_path_reason"),
                "fallback_reason": absolute_backend_info.get("fallback_reason"),
                "rom_enabled_effective": absolute_backend_info.get("rom_enabled_effective", False),
                "rom_rank_effective": absolute_backend_info.get("rom_rank_effective", 0),
                "lowrank_rank_effective": absolute_backend_info.get("lowrank_rank_effective", 0),
                "inexact_eta_history": absolute_backend_info.get("inexact_eta_history", []),
                "degrade_stage_counts": absolute_backend_info.get("degrade_stage_counts", {}),
                "effective_solver_path_counts": absolute_backend_info.get("effective_solver_path_counts", {}),
                "petsc_device_requested": absolute_backend_info.get("petsc_device_requested"),
                "petsc_device_effective": absolute_backend_info.get("petsc_device_effective"),
                "petsc_mat_type": absolute_backend_info.get("petsc_mat_type"),
                "petsc_vec_type": absolute_backend_info.get("petsc_vec_type"),
                "gpu_fallback_reason": absolute_backend_info.get("gpu_fallback_reason"),
                "forward_factor_backend": absolute_backend_info.get("forward_factor_backend"),
                "forward_mat_solve_effective": absolute_backend_info.get("forward_mat_solve_effective"),
                "inverse_device_requested": absolute_backend_info.get("inverse_device_requested"),
                "inverse_device_effective": absolute_backend_info.get("inverse_device_effective"),
                "execution_profile": absolute_backend_info.get("execution_profile"),
                "jacobian_backend_requested": absolute_backend_info.get("jacobian_backend_requested"),
                "jacobian_backend_effective": absolute_backend_info.get("jacobian_backend_effective"),
                "jacobian_block_backend": absolute_backend_info.get("jacobian_block_backend"),
                "jacobian_transfer_estimate": absolute_backend_info.get("jacobian_transfer_estimate"),
                "jacobian_cuda_threshold_hit": absolute_backend_info.get("jacobian_cuda_threshold_hit"),
                "jacobian_block_tune": absolute_backend_info.get("jacobian_block_tune", {}),
                "jacobian_assembly_elapsed_only": absolute_backend_info.get("jacobian_assembly_elapsed_only", 0.0),
                "startup_jacobian_elapsed": float(absolute_timing.get("jacobian", 0.0))
                if isinstance(absolute_timing, dict)
                else 0.0,
                "forward_probe_elapsed_sec": float(absolute_timing.get("forward_probe", 0.0))
                if isinstance(absolute_timing, dict)
                else 0.0,
                "forward_probe_repeats": int(absolute_timing.get("forward_probe_repeats", absolute_forward_probe_repeats))
                if isinstance(absolute_timing, dict)
                else int(absolute_forward_probe_repeats),
                "startup_cache_lookup": absolute_backend_info.get("startup_cache_lookup", {}),
            },
            "cache": {
                "cold_build": cold_ctx.get("cache_build_seconds", {}) if isinstance(cold_ctx, dict) else {},
                "warm_build": warm_ctx.get("cache_build_seconds", {}) if isinstance(warm_ctx, dict) else {},
                "warm_lookups": warm_ctx.get("cache_lookups", {}) if isinstance(warm_ctx, dict) else {},
            },
        }

    runs = [_run_once(i) for i in range(int(args.repeat))]

    def _absolute_elapsed(run_payload: dict) -> float:
        for item in run_payload.get("stages", []):
            if item.get("stage") == "absolute_reconstruct":
                return float(item.get("elapsed_sec", 0.0))
        return float("inf")

    median_run = sorted(runs, key=_absolute_elapsed)[len(runs) // 2]

    payload = {
        "config": {
            "n_elec": int(args.n_elec),
            "radius": float(args.radius),
            "height": float(args.height),
            "refinement": int(args.refinement),
            "lambda": float(args.lam),
            "solver_mode": str(args.solver_mode),
            "linear_solver": str(args.linear_solver),
            "preconditioner": str(args.preconditioner),
            "fast_linear_path": str(args.fast_linear_path),
            "rom_mode": str(args.rom_mode),
            "rom_rank_global": int(args.rom_rank_global),
            "rom_rank_adaptive": int(args.rom_rank_adaptive),
            "rom_refresh_every": int(args.rom_refresh_every),
            "rom_snapshot_source": str(args.rom_snapshot_source),
            "inexact_mode": str(args.inexact_mode),
            "inexact_forcing": str(args.inexact_forcing),
            "inexact_eta0": float(args.inexact_eta0),
            "inexact_eta_min": float(args.inexact_eta_min),
            "inexact_eta_max": float(args.inexact_eta_max),
            "lowrank_mode": str(args.lowrank_mode),
            "lowrank_rank": int(args.lowrank_rank),
            "lowrank_method": str(args.lowrank_method),
            "lowrank_energy": float(args.lowrank_energy),
            "forward_mat_solve": str(args.forward_mat_solve),
            "petsc_device": str(args.petsc_device),
            "device": str(args.device),
            "jacobian_block_tune": str(args.jacobian_block_tune),
            "jacobian_block_size": int(args.jacobian_block_size),
            "jacobian_block_candidates": jacobian_block_candidates,
            "cholmod_max_n": int(args.cholmod_max_n),
            "cholmod_max_memory_gib": float(args.cholmod_max_memory_gib),
            "absolute_startup_cache": str(args.absolute_startup_cache),
            "absolute_iters": int(args.absolute_iters),
            "run_diff": str(args.run_diff),
            "run_absolute": str(args.run_absolute),
        },
        "profile_label": str(args.profile_label),
        "repeat": int(args.repeat),
        "capabilities": detect_performance_capabilities(),
        "stages": median_run.get("stages", []),
        "stage_breakdown": median_run.get("stage_breakdown", {}),
        "difference_solver": median_run.get("difference_solver", {}),
        "absolute_solver": median_run.get("absolute_solver", {}),
        "cache": median_run.get("cache", {}),
        "runs": runs if int(args.repeat) > 1 else [],
    }
    output_path = args.perf_report if args.perf_report is not None else args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] 3D benchmark report saved: {output_path}")


if __name__ == "__main__":
    main()
