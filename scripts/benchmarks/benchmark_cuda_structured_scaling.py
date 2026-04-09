#!/usr/bin/env python3
"""Compare 3D direct-forward scaling for CPU dolfinx vs GPU cuda_structured."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional in lean environments
    import torch
except Exception:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.perf import ACCELERATION_PROFILE_GPU3D, DEFAULT_ACCELERATION_PROFILE
from scripts.common.acceleration_profiles import (
    add_acceleration_profile_argument,
    resolve_3d_mesh_contract,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=None, help="Explicit mesh root. Defaults to ephemeral /tmp.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Explicit cache root. Defaults to ephemeral /tmp.")
    parser.add_argument(
        "--refinements",
        type=str,
        default="3,4,8",
        help="Comma-separated 3D refinement levels for the scaling sweep.",
    )
    parser.add_argument("--warm-forward-repeats", type=int, default=2)
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.18)
    parser.add_argument("--height", type=float, default=0.16)
    parser.add_argument("--electrode-height-ratio", type=float, default=0.2)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--contact-impedance", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path, default=Path("reports") / "cuda_structured_scaling.json")
    parser.add_argument("--gate", choices=["strict", "off"], default="strict")
    add_acceleration_profile_argument(
        parser,
        flag="--gpu-acceleration-profile",
        default=ACCELERATION_PROFILE_GPU3D,
        help_suffix="Used for the GPU-side 3D case only.",
    )
    return parser.parse_args()


def _parse_refinements(raw: str) -> list[int]:
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    values = sorted({int(token) for token in tokens if int(token) > 0})
    if not values:
        raise ValueError("at least one positive refinement is required")
    return values


def _sync_cuda() -> None:
    if torch is None or not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()


def _timed(fn):
    _sync_cuda()
    started = time.perf_counter()
    out = fn()
    _sync_cuda()
    return out, float(time.perf_counter() - started)


def _pattern(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        rotate_meas=True,
    )


def _build_system(
    *,
    mesh,
    cache_dir: Path,
    backend: str,
    petsc_device: str,
    device: str,
    n_elec: int,
    contact_impedance: float,
    mesh_family: str,
    geometry_version: str,
    generator_revision: str,
    acceleration_profile: str = DEFAULT_ACCELERATION_PROFILE,
) -> EITSystem:
    system = EITSystem(
        n_elec=n_elec,
        pattern_config=_pattern(n_elec),
        contact_impedance=np.full(n_elec, float(contact_impedance), dtype=float),
        base_conductivity=1.0,
        regularization_type="noser",
        regularization_alpha=1.0,
        cache_dir=str(cache_dir),
        cache_scope="both",
        forward_backend=backend,
        mesh_family=str(mesh_family),
        geometry_version=str(geometry_version),
        generator_revision=str(generator_revision),
        solver_mode="fast",
        line_search_mode="fast",
        petsc_device=petsc_device,
        device=device,
        acceleration_profile=str(acceleration_profile),
        linear_backend_config={"petsc_device": petsc_device},
    )
    system.setup(mesh=mesh)
    system.reconstructor.max_iterations = 1
    system.reconstructor.min_iterations = 1
    system.reconstructor.verbose = False
    return system


def _clone_image(system: EITSystem, values: np.ndarray) -> EITImage:
    return EITImage(elem_data=np.asarray(values, dtype=np.float64).copy(), fwd_model=system.fwd_model)


def _make_phantom(system: EITSystem) -> EITImage:
    baseline = system.create_homogeneous_image(conductivity=1.0)
    sigma = np.asarray(baseline.elem_data, dtype=np.float64).copy()
    sigma[: max(1, sigma.size // 10)] = 1.2
    return _clone_image(system, sigma)


def _evaluate_gate(refinement: int, *, first_forward_speedup: float, warm_forward_speedup: float) -> dict[str, object]:
    first_required = 3.0 if refinement in {3, 8} else None
    warm_required = 5.0 if refinement in {3, 4, 8} else None
    first_ok = True if first_required is None else bool(first_forward_speedup >= first_required)
    warm_ok = True if warm_required is None else bool(warm_forward_speedup >= warm_required)
    return {
        "first_forward_required_x": first_required,
        "warm_forward_required_x": warm_required,
        "first_forward_passed": first_ok,
        "warm_forward_passed": warm_ok,
        "passed": bool(first_ok and warm_ok),
    }


def _run_refinement(args: argparse.Namespace, *, refinement: int, mesh_root: Path, cache_root: Path) -> dict[str, object]:
    mesh_dir = mesh_root / f"ref{refinement}"
    cpu_cache = cache_root / f"ref{refinement}" / "cpu"
    gpu_cache = cache_root / f"ref{refinement}" / "gpu"
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir, ignore_errors=True)
    if cpu_cache.exists():
        shutil.rmtree(cpu_cache, ignore_errors=True)
    if gpu_cache.exists():
        shutil.rmtree(gpu_cache, ignore_errors=True)
    mesh_family, geometry_version, generator_revision = resolve_3d_mesh_contract(
        acceleration_profile=args.gpu_acceleration_profile,
    )
    mesh = load_or_create_mesh(
        mesh_dir=str(mesh_dir),
        mesh_name=(
            f"cuda_structured_scaling_ref{refinement}"
            f"_cf{mesh_family}_{geometry_version}_{generator_revision}"
        ),
        n_elec=int(args.n_elec),
        dimension=3,
        radius=float(args.radius),
        refinement=int(refinement),
        height=float(args.height),
        electrode_height_ratio=float(args.electrode_height_ratio),
        z_center=0.0,
        electrode_coverage=float(args.electrode_coverage),
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
    )
    cpu = _build_system(
        mesh=mesh,
        cache_dir=cpu_cache,
        backend="dolfinx",
        petsc_device="cpu",
        device="cpu",
        n_elec=int(args.n_elec),
        contact_impedance=float(args.contact_impedance),
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
        acceleration_profile=DEFAULT_ACCELERATION_PROFILE,
    )
    gpu = _build_system(
        mesh=mesh,
        cache_dir=gpu_cache,
        backend="cuda_structured",
        petsc_device="cuda",
        device="cuda",
        n_elec=int(args.n_elec),
        contact_impedance=float(args.contact_impedance),
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
        acceleration_profile=str(args.gpu_acceleration_profile),
    )

    cpu_baseline = cpu.create_homogeneous_image(conductivity=1.0)
    gpu_baseline = gpu.create_homogeneous_image(conductivity=1.0)
    cpu_phantom = _make_phantom(cpu)
    gpu_phantom = _clone_image(gpu, np.asarray(cpu_phantom.elem_data, dtype=np.float64))

    _, cpu_first = _timed(lambda: cpu.forward_solve(cpu_baseline))
    _, gpu_first = _timed(lambda: gpu.forward_solve(gpu_baseline))

    _, cpu_warm_total = _timed(
        lambda: [cpu.forward_solve(cpu_baseline) for _ in range(int(args.warm_forward_repeats))]
    )
    _, gpu_warm_total = _timed(
        lambda: [gpu.forward_solve(gpu_baseline) for _ in range(int(args.warm_forward_repeats))]
    )
    _, cpu_target = _timed(lambda: cpu.forward_solve(cpu_phantom))
    _, gpu_target = _timed(lambda: gpu.forward_solve(gpu_phantom))

    cpu_warm_avg = float(cpu_warm_total / max(1, int(args.warm_forward_repeats)))
    gpu_warm_avg = float(gpu_warm_total / max(1, int(args.warm_forward_repeats)))
    speedup = {
        "first_forward_x": float(cpu_first / gpu_first) if gpu_first > 0 else 0.0,
        "warm_forward_avg_x": float(cpu_warm_avg / gpu_warm_avg) if gpu_warm_avg > 0 else 0.0,
        "target_forward_x": float(cpu_target / gpu_target) if gpu_target > 0 else 0.0,
    }
    return {
        "refinement": int(refinement),
        "mesh": {
            "nodes": int(mesh.num_vertices()),
            "cells": int(mesh.num_cells()),
            "mesh_file": getattr(mesh, "mesh_file", None),
            "mesh_family": getattr(mesh, "mesh_family", None),
            "geometry_version": getattr(mesh, "geometry_version", None),
            "generator_revision": getattr(mesh, "generator_revision", None),
        },
        "cpu": {
            "forward_backend": "dolfinx",
            "first_forward_elapsed_sec": float(cpu_first),
            "warm_forward_total_sec": float(cpu_warm_total),
            "warm_forward_avg_sec": float(cpu_warm_avg),
            "target_forward_elapsed_sec": float(cpu_target),
            "backend_info": cpu.fwd_model.get_backend_diagnostics(),
        },
        "gpu": {
            "forward_backend": "cuda_structured",
            "first_forward_elapsed_sec": float(gpu_first),
            "warm_forward_total_sec": float(gpu_warm_total),
            "warm_forward_avg_sec": float(gpu_warm_avg),
            "target_forward_elapsed_sec": float(gpu_target),
            "backend_info": gpu.fwd_model.get_backend_diagnostics(),
        },
        "speedup": speedup,
        "gate": _evaluate_gate(
            int(refinement),
            first_forward_speedup=float(speedup["first_forward_x"]),
            warm_forward_speedup=float(speedup["warm_forward_avg_x"]),
        ),
    }


def main() -> None:
    args = _parse_args()
    refinements = _parse_refinements(args.refinements)
    mesh_family, geometry_version, generator_revision = resolve_3d_mesh_contract(
        acceleration_profile=args.gpu_acceleration_profile,
    )
    ephemeral_mesh_root = None
    ephemeral_cache_root = None
    mesh_root = args.mesh_dir
    cache_root = args.cache_root
    if mesh_root is None:
        ephemeral_mesh_root = Path(tempfile.mkdtemp(prefix="pyeidors-cuda-structured-mesh-", dir="/tmp"))
        mesh_root = ephemeral_mesh_root
    if cache_root is None:
        ephemeral_cache_root = Path(tempfile.mkdtemp(prefix="pyeidors-cuda-structured-cache-", dir="/tmp"))
        cache_root = ephemeral_cache_root

    try:
        results = [_run_refinement(args, refinement=refinement, mesh_root=mesh_root, cache_root=cache_root) for refinement in refinements]
        payload = {
            "config": {
                "refinements": refinements,
                "n_elec": int(args.n_elec),
                "radius": float(args.radius),
                "height": float(args.height),
                "electrode_height_ratio": float(args.electrode_height_ratio),
                "electrode_coverage": float(args.electrode_coverage),
                "contact_impedance": float(args.contact_impedance),
                "warm_forward_repeats": int(args.warm_forward_repeats),
                "gpu_acceleration_profile": str(args.gpu_acceleration_profile),
                "mesh_family": str(mesh_family),
                "geometry_version": str(geometry_version),
                "generator_revision": str(generator_revision),
            },
            "mesh_root_mode": "ephemeral" if ephemeral_mesh_root is not None else "explicit",
            "cache_root_mode": "ephemeral" if ephemeral_cache_root is not None else "explicit",
            "results": results,
            "all_gates_passed": bool(all(bool(result["gate"]["passed"]) for result in results)),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))

        if args.gate == "strict" and not payload["all_gates_passed"]:
            raise SystemExit(1)
    finally:
        if ephemeral_cache_root is not None and ephemeral_cache_root.exists():
            shutil.rmtree(ephemeral_cache_root, ignore_errors=True)
        if ephemeral_mesh_root is not None and ephemeral_mesh_root.exists():
            shutil.rmtree(ephemeral_mesh_root, ignore_errors=True)


if __name__ == "__main__":
    main()
