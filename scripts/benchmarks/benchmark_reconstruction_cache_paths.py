#!/usr/bin/env python3
"""Benchmark realtime reconstruction cache paths on a real recorded session."""

from __future__ import annotations

import argparse
import contextlib
import cProfile
import io
import json
import os
import pstats
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eit_app.controllers import reconstruction_controller as rc
from eit_app.controllers.reconstruction_controller import (
    ReconstructionRequest,
    clear_reconstruction_system_cache,
    run_reconstruction_request,
)
from eit_app.models.frame_model import FrameData
from pyeidors import EITSystem
from pyeidors.data import MeasurementDataset, PatternConfig
from pyeidors.data.frame_io import (
    read_frame_csv,
    read_frame_yaml,
    read_session_metadata,
    scan_frame_dir,
)
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.utils.numeric_ops import safe_dot
from scripts.common.gn_difference_runner import (
    STRICT_SOLVER_BACKEND_MEASUREMENT,
    _measurement_space_delta,
    _solve_linear_from_bundle,
    build_shared_context,
    process_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("data/measurements/test_for_gui/session_20260413_140902"),
        help="Real hardware session directory.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("eit_meshes"),
        help="Mesh cache directory.",
    )
    parser.add_argument(
        "--mesh-refinement",
        type=int,
        default=4,
        help="GUI request-side refinement knob.",
    )
    parser.add_argument(
        "--compute-cycles",
        type=int,
        default=3,
        help="Number of warm cycles over all target frames.",
    )
    parser.add_argument(
        "--use-part",
        choices=["real", "imag", "mag"],
        default="real",
        help="Measurement component for reconstruction.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/benchmarks/reconstruction_cache_paths_session_20260413_140902.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }
    arr = np.asarray(samples, dtype=np.float64)
    total = float(arr.sum())
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean() * 1000.0),
        "median_ms": float(np.median(arr) * 1000.0),
        "p95_ms": float(np.percentile(arr, 95) * 1000.0),
        "min_ms": float(arr.min() * 1000.0),
        "max_ms": float(arr.max() * 1000.0),
        "fps": float(arr.size / total) if total > 0 else 0.0,
    }


def _quiet(fn):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return fn()


def _time_call(fn) -> tuple[Any, float]:
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


def _load_session_frames(session_dir: Path) -> tuple[dict[str, Any], list[FrameData]]:
    session_meta = read_session_metadata(session_dir / "session_metadata.yaml")
    frames: list[FrameData] = []
    for csv_path, yaml_path in scan_frame_dir(session_dir):
        real, imag = read_frame_csv(csv_path)
        frame_meta = dict(session_meta)
        frame_meta.update(read_frame_yaml(yaml_path))
        frames.append(
            FrameData(
                real=real,
                imag=imag,
                timestamp=float(frame_meta.get("timestamp", 0.0)),
                frame_index=int(frame_meta.get("frame_index", len(frames))),
                metadata=frame_meta,
            )
        )
    if len(frames) < 2:
        raise ValueError(f"Session 至少需要 2 帧: {session_dir}")
    return session_meta, frames


def _gui_effective_refinement(radius: float, mesh_refinement: int) -> int:
    mesh_size = max(0.02, 0.25 / max(1, int(mesh_refinement)))
    return max(2, int(round(float(radius) / max(mesh_size, 1e-6) / 2.0)))


def _build_metadata(session_meta: dict[str, Any], mesh_dir: Path) -> dict[str, Any]:
    return {
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": int(session_meta.get("n_elec", session_meta.get("total_electrodes", 16))),
        "stim_pattern": session_meta.get("stim_pattern", "{ad}"),
        "meas_pattern": session_meta.get("meas_pattern", "{ad}"),
        "drive_mode": "total_current",
        "drive_value": 1.0e-5,
        "geometry_scale_to_m": 1.0,
        "radius": float(session_meta.get("radius", 1.0)),
        "mesh_dir": str(mesh_dir),
    }


def _build_gui_request(
    *,
    reference_frame: FrameData,
    target_frame: FrameData,
    session_meta: dict[str, Any],
    mesh_dir: Path,
    mesh_refinement: int,
    use_part: str,
) -> ReconstructionRequest:
    metadata = _build_metadata(session_meta, mesh_dir)
    return ReconstructionRequest(
        reference_frame=reference_frame,
        target_frame=target_frame,
        use_part=use_part,
        method="gn-difference",
        regularization_alpha=1.0,
        max_iterations=1,
        mesh_dimension=int(session_meta.get("mea_mode", 2)),
        mesh_refinement=int(mesh_refinement),
        metadata=metadata,
    )


def _build_eit_data(frame: FrameData, metadata: dict[str, Any], use_part: str):
    data_type = use_part if use_part in {"real", "imag", "mag"} else "real"
    dataset = MeasurementDataset.from_metadata(
        measurements=frame.to_measurement_vector(use_part).reshape(1, -1),
        metadata=metadata,
        data_type=data_type,
    )
    return dataset.to_eit_data(frame_index=0)


def _profile_to_lines(profile: cProfile.Profile, limit: int = 20) -> list[str]:
    output = io.StringIO()
    stats = pstats.Stats(profile, stream=output).sort_stats("cumulative")
    stats.print_stats(limit)
    return [line.rstrip() for line in output.getvalue().splitlines() if line.strip()]


def _benchmark_gui_default(
    *,
    reference_frame: FrameData,
    target_frames: list[FrameData],
    session_meta: dict[str, Any],
    mesh_dir: Path,
    mesh_refinement: int,
    use_part: str,
    compute_cycles: int,
) -> dict[str, Any]:
    clear_reconstruction_system_cache()

    cold_request = _build_gui_request(
        reference_frame=reference_frame,
        target_frame=target_frames[0],
        session_meta=session_meta,
        mesh_dir=mesh_dir,
        mesh_refinement=mesh_refinement,
        use_part=use_part,
    )
    cold_result, cold_elapsed = _time_call(
        lambda: _quiet(lambda: run_reconstruction_request(cold_request))
    )
    if cold_result.error_msg:
        raise RuntimeError(f"GUI 默认链冷启动失败: {cold_result.error_msg}")

    warm_times: list[float] = []
    last_result = cold_result
    for _ in range(compute_cycles):
        for target_frame in target_frames:
            request = _build_gui_request(
                reference_frame=reference_frame,
                target_frame=target_frame,
                session_meta=session_meta,
                mesh_dir=mesh_dir,
                mesh_refinement=mesh_refinement,
                use_part=use_part,
            )
            result, elapsed = _time_call(
                lambda req=request: _quiet(lambda: run_reconstruction_request(req))
            )
            if result.error_msg:
                raise RuntimeError(f"GUI 默认链暖启动失败: {result.error_msg}")
            last_result = result
            warm_times.append(elapsed)

    profile = cProfile.Profile()
    profile.enable()
    prof_result = _quiet(lambda: run_reconstruction_request(cold_request))
    profile.disable()
    if prof_result.error_msg:
        raise RuntimeError(f"GUI 默认链 profile 失败: {prof_result.error_msg}")

    cache_stats: dict[str, Any] = {}
    cache_key_repr = None
    if rc._SYSTEM_CACHE:
        cache_key_repr = repr(next(iter(rc._SYSTEM_CACHE.keys())))
        system = next(iter(rc._SYSTEM_CACHE.values()))
        try:
            cache_stats = system.get_cache_stats()
        except Exception as exc:  # pragma: no cover
            cache_stats = {"error": str(exc)}

    solver_diag = getattr(last_result, "metadata", {}).get("solver_diagnostics", {}) or {}
    return {
        "cold_ms": float(cold_elapsed * 1000.0),
        "warm": _stats(warm_times),
        "solver_diagnostics": {
            "difference_step_size": solver_diag.get("difference_step_size"),
            "best_homog": solver_diag.get("best_homog"),
            "timing": solver_diag.get("timing"),
            "measurement_space": solver_diag.get("measurement_space"),
            "preset_name": solver_diag.get("preset_name"),
            "lambda_eff": solver_diag.get("lambda_eff"),
        },
        "process_cache": {
            "system_cache_items": len(rc._SYSTEM_CACHE),
            "system_cache_key": cache_key_repr,
            "eit_cache_stats": cache_stats,
        },
        "profile_top_cumulative": _profile_to_lines(profile),
    }


def _benchmark_direct_gn_variant(
    *,
    name: str,
    reference_eit,
    target_eits: list[Any],
    session_meta: dict[str, Any],
    mesh_dir: Path,
    effective_refinement: int,
    use_part: str,
    compute_cycles: int,
    difference_step_size_mode: str,
    best_homog_mode: str,
    solver_mode: str = "strict",
    line_search_mode: str = "full",
) -> dict[str, Any]:
    metadata = _build_metadata(session_meta, mesh_dir)
    pattern_config = PatternConfig(
        n_elec=int(metadata["n_elec"]),
        stim_pattern=str(metadata["stim_pattern"]),
        meas_pattern=str(metadata["meas_pattern"]),
        drive_mode=str(metadata["drive_mode"]),
        drive_value=float(metadata["drive_value"]),
        geometry_scale_to_m=float(metadata["geometry_scale_to_m"]),
    )
    mesh = load_or_create_mesh(
        mesh_dir=str(mesh_dir),
        n_elec=int(metadata["n_elec"]),
        dimension=int(session_meta.get("mea_mode", 2)),
        radius=float(metadata["radius"]),
        refinement=int(effective_refinement),
    )

    system = EITSystem(
        n_elec=int(metadata["n_elec"]),
        pattern_config=pattern_config,
        regularization_alpha=1.0,
        difference_mode="raw",
        difference_orientation="target_minus_reference",
        difference_step_size_mode=difference_step_size_mode,
        best_homog_mode=best_homog_mode,
        solver_mode=solver_mode,
        line_search_mode=line_search_mode,
    )
    _quiet(lambda: system.setup(mesh=mesh))

    cold_result, cold_elapsed = _time_call(
        lambda: _quiet(
            lambda: system.difference_reconstruct(
                measurement_data=target_eits[0],
                reference_data=reference_eit,
            )
        )
    )

    warm_times: list[float] = []
    last_result = cold_result
    for _ in range(compute_cycles):
        for target_eit in target_eits:
            result, elapsed = _time_call(
                lambda tgt=target_eit: _quiet(
                    lambda: system.difference_reconstruct(
                        measurement_data=tgt,
                        reference_data=reference_eit,
                    )
                )
            )
            warm_times.append(elapsed)
            last_result = result

    solver_diag = getattr(last_result, "metadata", {}).get("solver_diagnostics", {}) or {}
    return {
        "variant": name,
        "cold_ms": float(cold_elapsed * 1000.0),
        "warm": _stats(warm_times),
        "solver_diagnostics": {
            "difference_step_size": solver_diag.get("difference_step_size"),
            "best_homog": solver_diag.get("best_homog"),
            "timing": solver_diag.get("timing"),
            "measurement_space": solver_diag.get("measurement_space"),
            "preset_name": solver_diag.get("preset_name"),
            "lambda_eff": solver_diag.get("lambda_eff"),
        },
        "cache_stats": system.get_cache_stats(),
    }


def _benchmark_single_step_paths(
    *,
    reference_frame: FrameData,
    target_frames: list[FrameData],
    mesh_dir: Path,
    session_meta: dict[str, Any],
    effective_refinement: int,
    compute_cycles: int,
) -> dict[str, Any]:
    ctx = _quiet(
        lambda: build_shared_context(
            mesh_dir=str(mesh_dir),
            mesh_name=None,
            mesh_dim=int(session_meta.get("mea_mode", 2)),
            mesh_height=1.0,
            electrode_height_ratio=0.2,
            z_center=0.0,
            refinement=int(effective_refinement),
            n_elec=int(session_meta.get("n_elec", session_meta.get("total_electrodes", 16))),
            radius=float(session_meta.get("radius", 1.0)),
            drive_value=1.0e-5,
            contact_impedance=0.01,
            background_sigma=1.0,
            lam=1e-2,
            cache_scope="both",
            solver_mode="strict",
            linear_solver="auto",
            preconditioner="auto",
            rom_mode="off",
            lowrank_mode="off",
            forward_mat_solve="off",
            petsc_device="auto",
            device="auto",
        )
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)

        full_cold_result, full_cold_elapsed = _time_call(
            lambda: _quiet(
                lambda: process_frames(
                    vh=reference_frame.real,
                    vi=target_frames[0].real,
                    output_dir=output_dir,
                    ctx=ctx,
                    step_size_calib=False,
                    step_size_min=1.0e-5,
                    step_size_max=1.0e1,
                    step_size_maxiter=50,
                    lam=1.0e-2,
                    colormap="eidors_diff",
                    colorbar_scientific=False,
                    colorbar_format=None,
                    transparent=False,
                    write_plots=False,
                    measurement_gain=1.0,
                )
            )
        )

        full_warm_times: list[float] = []
        last_full_result = full_cold_result
        for _ in range(compute_cycles):
            for target_frame in target_frames:
                result, elapsed = _time_call(
                    lambda frame=target_frame: _quiet(
                        lambda: process_frames(
                            vh=reference_frame.real,
                            vi=frame.real,
                            output_dir=output_dir,
                            ctx=ctx,
                            step_size_calib=False,
                            step_size_min=1.0e-5,
                            step_size_max=1.0e1,
                            step_size_maxiter=50,
                            lam=1.0e-2,
                            colormap="eidors_diff",
                            colorbar_scientific=False,
                            colorbar_format=None,
                            transparent=False,
                            write_plots=False,
                            measurement_gain=1.0,
                        )
                    )
                )
                last_full_result = result
                full_warm_times.append(elapsed)

        operator_bundle = ctx["operator_bundle"]
        strict_backend = str(
            operator_bundle.get(
                "strict_solver_backend_effective",
                "dense-param",
            )
        )

        def _solve_delta(vi: np.ndarray) -> np.ndarray:
            dv = np.asarray(vi - reference_frame.real, dtype=np.float64)
            if strict_backend == STRICT_SOLVER_BACKEND_MEASUREMENT:
                return _measurement_space_delta(operator_bundle=operator_bundle, rhs=dv)
            rhs = np.asarray(
                safe_dot(operator_bundle["Jt"], dv, "bench.Jt_dv"),
                dtype=float,
            )
            return _solve_linear_from_bundle(operator_bundle, rhs)

        _quiet(lambda: _solve_delta(target_frames[0].real))
        pure_warm_times: list[float] = []
        for _ in range(compute_cycles):
            for target_frame in target_frames:
                _, elapsed = _time_call(
                    lambda frame=target_frame: _quiet(
                        lambda: _solve_delta(frame.real)
                    )
                )
                pure_warm_times.append(elapsed)

    return {
        "full_process": {
            "cold_ms": float(full_cold_elapsed * 1000.0),
            "warm": _stats(full_warm_times),
            "stage_timings_ms": {
                key: float(value * 1000.0)
                for key, value in (last_full_result.get("stage_timings") or {}).items()
            },
            "cache_build_seconds": dict(last_full_result.get("cache_build_seconds") or {}),
            "cache_miss_reasons": dict(last_full_result.get("cache_miss_reasons") or {}),
            "strict_backend": strict_backend,
        },
        "pure_single_step_core": {
            "warm": _stats(pure_warm_times),
            "strict_backend": strict_backend,
        },
    }


def main() -> None:
    args = parse_args()
    session_meta, frames = _load_session_frames(args.session_dir)
    reference_frame = frames[0]
    target_frames = frames[1:]
    gui_metadata = _build_metadata(session_meta, args.mesh_dir)
    reference_eit = _build_eit_data(reference_frame, gui_metadata, args.use_part)
    target_eits = [
        _build_eit_data(frame, gui_metadata, args.use_part) for frame in target_frames
    ]

    effective_refinement = _gui_effective_refinement(
        radius=float(gui_metadata["radius"]),
        mesh_refinement=int(args.mesh_refinement),
    )

    report = {
        "benchmark": "reconstruction_cache_paths",
        "session_dir": str(args.session_dir),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pyeidors_env_profile": os.environ.get("PYEIDORS_ENV_PROFILE", "default"),
        },
        "session": {
            "frame_count": int(len(frames)),
            "target_count": int(len(target_frames)),
            "points_per_frame": int(session_meta.get("points_per_frame", reference_frame.n_meas)),
            "n_elec": int(session_meta.get("n_elec", session_meta.get("total_electrodes", 16))),
            "frequency_hz": float(session_meta.get("frequency_hz", 0.0)),
            "stim_amp_uA": float(session_meta.get("stim_amp_uA", 0.0)),
            "use_part": args.use_part,
        },
        "mesh": {
            "mesh_dir": str(args.mesh_dir),
            "gui_requested_refinement": int(args.mesh_refinement),
            "gui_effective_refinement": int(effective_refinement),
        },
    }

    report["gui_default"] = _benchmark_gui_default(
        reference_frame=reference_frame,
        target_frames=target_frames,
        session_meta=session_meta,
        mesh_dir=args.mesh_dir,
        mesh_refinement=args.mesh_refinement,
        use_part=args.use_part,
        compute_cycles=args.compute_cycles,
    )
    report["gn_variant_default"] = _benchmark_direct_gn_variant(
        name="gn_default",
        reference_eit=reference_eit,
        target_eits=target_eits,
        session_meta=session_meta,
        mesh_dir=args.mesh_dir,
        effective_refinement=effective_refinement,
        use_part=args.use_part,
        compute_cycles=args.compute_cycles,
        difference_step_size_mode="optimize",
        best_homog_mode="off",
    )
    report["gn_variant_no_step_search"] = _benchmark_direct_gn_variant(
        name="gn_no_step_search",
        reference_eit=reference_eit,
        target_eits=target_eits,
        session_meta=session_meta,
        mesh_dir=args.mesh_dir,
        effective_refinement=effective_refinement,
        use_part=args.use_part,
        compute_cycles=args.compute_cycles,
        difference_step_size_mode="off",
        best_homog_mode="off",
    )
    report["gn_variant_fast_mode"] = _benchmark_direct_gn_variant(
        name="gn_fast_mode",
        reference_eit=reference_eit,
        target_eits=target_eits,
        session_meta=session_meta,
        mesh_dir=args.mesh_dir,
        effective_refinement=effective_refinement,
        use_part=args.use_part,
        compute_cycles=args.compute_cycles,
        difference_step_size_mode="off",
        best_homog_mode="off",
        solver_mode="fast",
        line_search_mode="fast",
    )
    report["single_step_cached"] = _benchmark_single_step_paths(
        reference_frame=reference_frame,
        target_frames=target_frames,
        mesh_dir=args.mesh_dir,
        session_meta=session_meta,
        effective_refinement=effective_refinement,
        compute_cycles=args.compute_cycles,
    )

    gui_fps = float(report["gui_default"]["warm"]["fps"])
    no_step_fps = float(report["gn_variant_no_step_search"]["warm"]["fps"])
    single_step_fps = float(report["single_step_cached"]["full_process"]["warm"]["fps"])
    pure_core_fps = float(report["single_step_cached"]["pure_single_step_core"]["warm"]["fps"])
    report["summary"] = {
        "gui_default_fps": gui_fps,
        "gn_default_fps": float(report["gn_variant_default"]["warm"]["fps"]),
        "gn_no_step_search_fps": no_step_fps,
        "gn_fast_mode_fps": float(report["gn_variant_fast_mode"]["warm"]["fps"]),
        "single_step_cached_full_fps": single_step_fps,
        "single_step_pure_core_fps": pure_core_fps,
        "speedup_vs_gui": {
            "gn_no_step_search": (no_step_fps / gui_fps) if gui_fps > 0 else 0.0,
            "single_step_cached_full": (single_step_fps / gui_fps) if gui_fps > 0 else 0.0,
            "single_step_pure_core": (pure_core_fps / gui_fps) if gui_fps > 0 else 0.0,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
