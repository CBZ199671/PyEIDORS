#!/usr/bin/env python3
"""Benchmark the realtime hardware imaging chain with real recorded frames."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from PySide6.QtWidgets import QApplication

from eit_app.controllers.reconstruction_controller import (
    ReconstructionRequest,
    clear_reconstruction_system_cache,
    run_reconstruction_request,
)
from eit_app.models.frame_model import FrameData
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget
from pyeidors.data.frame_io import (
    read_frame_csv,
    read_frame_yaml,
    read_session_metadata,
    scan_frame_dir,
)
from pyeidors.runtime_paths import pyeidors_cache_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=Path("data/measurements/test_for_gui/session_20260413_140902"),
        help="包含真实采集帧的 session 目录。",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=pyeidors_cache_path("eit_meshes"),
        help="Mesh cache 目录。",
    )
    parser.add_argument(
        "--mesh-refinement",
        type=int,
        default=4,
        help="与 GUI 自动重构一致的 refinement 参数。",
    )
    parser.add_argument(
        "--compute-cycles",
        type=int,
        default=3,
        help="缓存后重构吞吐的循环次数；每次会遍历所有 target 帧。",
    )
    parser.add_argument(
        "--render-cycles",
        type=int,
        default=40,
        help="纯渲染吞吐的循环次数；每次会遍历一轮预先计算好的结果。",
    )
    parser.add_argument(
        "--use-part",
        choices=["real", "imag", "mag"],
        default="real",
        help="重构使用的测量分量。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("docs/benchmarks/realtime_imaging_chain.json"),
        help="可选 JSON 报告输出路径。",
    )
    return parser.parse_args()


def _stats_from_times(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "count": 0,
            "total_sec": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }
    total = float(sum(samples))
    mean = float(statistics.mean(samples))
    median = float(statistics.median(samples))
    p95 = float(np.percentile(np.asarray(samples, dtype=np.float64), 95))
    return {
        "count": int(len(samples)),
        "total_sec": total,
        "mean_ms": mean * 1000.0,
        "median_ms": median * 1000.0,
        "p95_ms": p95 * 1000.0,
        "min_ms": min(samples) * 1000.0,
        "max_ms": max(samples) * 1000.0,
        "fps": (len(samples) / total) if total > 0 else 0.0,
    }


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
        raise ValueError(f"Session 至少需要 2 帧才能做差分 benchmark: {session_dir}")
    return session_meta, frames


def _build_request(
    *,
    reference_frame: FrameData,
    target_frame: FrameData,
    session_meta: dict[str, Any],
    mesh_dir: Path,
    mesh_refinement: int,
    use_part: str,
) -> ReconstructionRequest:
    n_elec = int(session_meta.get("n_elec", session_meta.get("total_electrodes", 16)))
    metadata = {
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": n_elec,
        "stim_pattern": session_meta.get("stim_pattern", "{ad}"),
        "meas_pattern": session_meta.get("meas_pattern", "{ad}"),
        "drive_mode": "total_current",
        "drive_value": 1.0e-5,
        "geometry_scale_to_m": 1.0,
        "radius": float(session_meta.get("radius", 1.0)),
        "mesh_dir": str(mesh_dir),
    }
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


def _time_call(fn) -> tuple[Any, float]:
    started = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - started


def _run_quietly(fn):
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return fn()


def _ensure_qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def main() -> None:
    args = parse_args()
    session_meta, frames = _load_session_frames(args.session_dir)
    reference_frame = frames[0]
    target_frames = frames[1:]

    app = _ensure_qapp()
    widget = ReconstructionWidget()
    widget.resize(640, 640)
    widget.configure_layout(
        n_elec=int(
            session_meta.get("n_elec", session_meta.get("total_electrodes", 16))
        ),
        radius=float(session_meta.get("radius", 1.0)),
    )
    widget.show()
    app.processEvents()

    static_prepare_started = time.perf_counter()
    widget.configure_layout(
        n_elec=int(
            session_meta.get("n_elec", session_meta.get("total_electrodes", 16))
        ),
        radius=float(session_meta.get("radius", 1.0)),
    )
    app.processEvents()
    static_prepare_sec = time.perf_counter() - static_prepare_started

    cold_request = _build_request(
        reference_frame=reference_frame,
        target_frame=target_frames[0],
        session_meta=session_meta,
        mesh_dir=args.mesh_dir,
        mesh_refinement=args.mesh_refinement,
        use_part=args.use_part,
    )

    clear_reconstruction_system_cache()
    cold_result, cold_compute_sec = _time_call(
        lambda: _run_quietly(lambda: run_reconstruction_request(cold_request))
    )
    if cold_result.error_msg:
        raise RuntimeError(f"冷启动重构失败: {cold_result.error_msg}")

    _, cold_render_sec = _time_call(
        lambda: (widget.update_reconstruction(cold_result), app.processEvents())
    )

    compute_times: list[float] = []
    warm_results: list[Any] = []
    for _ in range(args.compute_cycles):
        for target_frame in target_frames:
            request = _build_request(
                reference_frame=reference_frame,
                target_frame=target_frame,
                session_meta=session_meta,
                mesh_dir=args.mesh_dir,
                mesh_refinement=args.mesh_refinement,
                use_part=args.use_part,
            )
            result, elapsed = _time_call(
                lambda req=request: _run_quietly(
                    lambda: run_reconstruction_request(req)
                )
            )
            if result.error_msg:
                raise RuntimeError(f"缓存后重构失败: {result.error_msg}")
            compute_times.append(elapsed)
            if len(warm_results) < len(target_frames):
                warm_results.append(result)

    render_times: list[float] = []
    render_results = warm_results or [cold_result]
    for _ in range(args.render_cycles):
        for result in render_results:
            _, elapsed = _time_call(
                lambda res=result: (
                    widget.update_reconstruction(res),
                    app.processEvents(),
                )
            )
            render_times.append(elapsed)

    combined_times: list[float] = []
    for _ in range(args.compute_cycles):
        for target_frame in target_frames:
            request = _build_request(
                reference_frame=reference_frame,
                target_frame=target_frame,
                session_meta=session_meta,
                mesh_dir=args.mesh_dir,
                mesh_refinement=args.mesh_refinement,
                use_part=args.use_part,
            )

            def _combined(req=request) -> Any:
                result = _run_quietly(lambda: run_reconstruction_request(req))
                if result.error_msg:
                    raise RuntimeError(result.error_msg)
                widget.update_reconstruction(result)
                app.processEvents()
                return result

            _, elapsed = _time_call(_combined)
            combined_times.append(elapsed)

    widget.close()
    app.processEvents()

    report = {
        "benchmark": "realtime_imaging_chain",
        "session_dir": str(args.session_dir),
        "session": {
            "reference_frame_index": int(reference_frame.frame_index),
            "target_frames": [int(frame.frame_index) for frame in target_frames],
            "frame_count": int(len(frames)),
            "points_per_frame": int(
                session_meta.get("points_per_frame", reference_frame.n_meas)
            ),
            "n_elec": int(
                session_meta.get("n_elec", session_meta.get("total_electrodes", 16))
            ),
            "frequency_hz": float(session_meta.get("frequency_hz", 0.0)),
            "stim_amp_uA": float(session_meta.get("stim_amp_uA", 0.0)),
            "use_part": args.use_part,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "qt_platform": os.environ.get("QT_QPA_PLATFORM", ""),
            "pyeidors_env_profile": os.environ.get("PYEIDORS_ENV_PROFILE", "default"),
        },
        "static_prepare": {
            "elapsed_ms": static_prepare_sec * 1000.0,
        },
        "cold": {
            "compute_ms": cold_compute_sec * 1000.0,
            "render_ms": cold_render_sec * 1000.0,
            "end_to_end_ms": (cold_compute_sec + cold_render_sec) * 1000.0,
        },
        "warm_compute": _stats_from_times(compute_times),
        "warm_render": _stats_from_times(render_times),
        "warm_end_to_end": _stats_from_times(combined_times),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Realtime imaging benchmark")
    print(f"  Session: {args.session_dir}")
    print(
        "  Sample: "
        f"{report['session']['frame_count']} frame(s), "
        f"{report['session']['points_per_frame']} pts/frame, "
        f"{report['session']['frequency_hz']:.0f} Hz"
    )
    print(
        "  Cold path: "
        f"compute={report['cold']['compute_ms']:.2f} ms, "
        f"render={report['cold']['render_ms']:.2f} ms, "
        f"end-to-end={report['cold']['end_to_end_ms']:.2f} ms"
    )
    for label in ("warm_compute", "warm_render", "warm_end_to_end"):
        stats = report[label]
        print(
            f"  {label}: count={stats['count']}, fps={stats['fps']:.2f}, "
            f"mean={stats['mean_ms']:.2f} ms, median={stats['median_ms']:.2f} ms, "
            f"p95={stats['p95_ms']:.2f} ms"
        )
    print(f"  Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
