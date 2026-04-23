#!/usr/bin/env python3
"""Worker for the real-valued 2D/3D CPU/GPU reconstruction gallery."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from pyeidors.perf import ACCELERATION_PROFILE_GPU3D, DEFAULT_ACCELERATION_PROFILE
from scripts.common.acceleration_profiles import (
    add_acceleration_profile_argument,
    resolve_3d_mesh_contract,
)
from scripts.common.hdf5_outputs import GALLERY_ARRAYS_SCHEMA, write_output_bundle
from scripts.diagnostics.gallery_shared import (
    consistency_metrics as _shared_consistency_metrics,
    jsonable as _jsonable,
    relative_l2 as _relative_l2,
    rmse as _rmse,
    safe_pearson as _safe_pearson,
    save_case_data as _save_case_data,
    truth_metrics as _truth_metrics,
)
from pyeidors import EITSystem
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.geometry.simple_mesh_generator import create_simple_eit_mesh


BACKGROUND_CONDUCTIVITY = 1.0
REAL_PHANTOM_HIGH = 1.6
REAL_PHANTOM_LOW = 0.65
_MEASUREMENT_REL_TOL = 1e-6
_IMAGE_REL_TOL = 5e-5
_IMAGE_RMSE_TOL = {2: 1e-6, 3: 1.25e-6}


@dataclass(frozen=True)
class AnomalySpec:
    label: str
    center_norm: tuple[float, ...]
    radius_norm: float
    conductivity: float


ANOMALIES_2D = (
    AnomalySpec(
        label="high",
        center_norm=(0.35, 0.0),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_HIGH,
    ),
    AnomalySpec(
        label="low",
        center_norm=(-0.35, 0.0),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_LOW,
    ),
)
ANOMALIES_3D = (
    AnomalySpec(
        label="high",
        center_norm=(0.35, 0.0, 0.22),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_HIGH,
    ),
    AnomalySpec(
        label="low",
        center_norm=(-0.35, 0.0, -0.22),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_LOW,
    ),
)


def _consistency_metrics(
    *,
    dim: int,
    baseline_cpu_meas: np.ndarray | None,
    baseline_gpu_meas: np.ndarray | None,
    target_cpu_meas: np.ndarray,
    target_gpu_meas: np.ndarray,
    cpu_recon: np.ndarray,
    gpu_recon: np.ndarray,
) -> dict[str, Any]:
    return _shared_consistency_metrics(
        dim=dim,
        baseline_cpu_meas=baseline_cpu_meas,
        baseline_gpu_meas=baseline_gpu_meas,
        target_cpu_meas=target_cpu_meas,
        target_gpu_meas=target_gpu_meas,
        cpu_recon=cpu_recon,
        gpu_recon=gpu_recon,
        measurement_rel_tol=_MEASUREMENT_REL_TOL,
        image_rel_tol=_IMAGE_REL_TOL,
        image_rmse_tol_by_dim=_IMAGE_RMSE_TOL,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=[2, 3], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-output-json", type=Path, required=True)
    parser.add_argument(
        "--run-kind", choices=["correctness", "fairness"], default="correctness"
    )
    parser.add_argument(
        "--backend-order", choices=["cpu-first", "gpu-first"], default="cpu-first"
    )
    parser.add_argument("--backend-key", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--mesh-size-2d", type=float, default=0.08)
    parser.add_argument("--radius-2d", type=float, default=1.0)
    parser.add_argument("--radius-3d", type=float, default=0.18)
    parser.add_argument("--height-3d", type=float, default=0.16)
    parser.add_argument("--refinement-3d", type=int, default=3)
    parser.add_argument("--electrode-height-ratio", type=float, default=0.2)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--contact-impedance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=2)
    add_acceleration_profile_argument(
        parser,
        flag="--gpu-acceleration-profile",
        default=ACCELERATION_PROFILE_GPU3D,
        help_suffix="Used for the 3D GPU case only.",
    )
    return parser.parse_args()


def _maybe_cuda_sync() -> None:
    try:  # pragma: no cover - only needed for cuda_structured timing
        import torch
    except Exception:
        return
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        return


def _timed(fn, *, sync_cuda: bool = False):
    if sync_cuda:
        _maybe_cuda_sync()
    started = time.perf_counter()
    out = fn()
    if sync_cuda:
        _maybe_cuda_sync()
    return out, float(time.perf_counter() - started)


def _sigma_coordinates(system: EITSystem) -> np.ndarray:
    coords = np.asarray(
        system.fwd_model.V_sigma.tabulate_dof_coordinates(), dtype=np.float64
    )
    return coords[:, : system.mesh.geometry.dim]


def _actual_anomalies(system: EITSystem, *, dim: int) -> list[dict[str, Any]]:
    coords = _sigma_coordinates(system)
    center = coords.mean(axis=0)
    radius = float(system.mesh.radius)
    anomalies = ANOMALIES_2D if int(dim) == 2 else ANOMALIES_3D
    actual: list[dict[str, Any]] = []
    if int(dim) == 2:
        for item in anomalies:
            actual.append(
                {
                    "label": item.label,
                    "center": np.array(
                        [
                            center[0] + item.center_norm[0] * radius,
                            center[1] + item.center_norm[1] * radius,
                        ],
                        dtype=np.float64,
                    ),
                    "radius": float(item.radius_norm * radius),
                    "conductivity": float(item.conductivity),
                    "center_norm": tuple(float(v) for v in item.center_norm),
                    "radius_norm": float(item.radius_norm),
                }
            )
        return actual

    z_min = float(np.min(coords[:, 2]))
    z_max = float(np.max(coords[:, 2]))
    z_center = 0.5 * (z_min + z_max)
    half_height = 0.5 * (z_max - z_min)
    for item in anomalies:
        actual.append(
            {
                "label": item.label,
                "center": np.array(
                    [
                        center[0] + item.center_norm[0] * radius,
                        center[1] + item.center_norm[1] * radius,
                        z_center + item.center_norm[2] * half_height,
                    ],
                    dtype=np.float64,
                ),
                "radius": float(item.radius_norm * radius),
                "conductivity": float(item.conductivity),
                "center_norm": tuple(float(v) for v in item.center_norm),
                "radius_norm": float(item.radius_norm),
            }
        )
    return actual


def _build_truth_values(
    system: EITSystem, *, dim: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coords = _sigma_coordinates(system)
    values = np.full(coords.shape[0], BACKGROUND_CONDUCTIVITY, dtype=np.float64)
    anomalies = _actual_anomalies(system, dim=dim)
    for item in anomalies:
        center = np.asarray(item["center"], dtype=np.float64)
        dist = np.linalg.norm(coords - center[None, :], axis=1)
        values[dist <= float(item["radius"])] = float(item["conductivity"])
    return values, anomalies


def _summary_case(
    case: dict[str, Any], *, data_path: Path, output_dir: Path
) -> dict[str, Any]:
    return {
        "label": case["label"],
        "forward_backend": case["forward_backend"],
        "petsc_device": case["petsc_device"],
        "forward_baseline_elapsed_sec": case["forward_baseline_elapsed_sec"],
        "forward_elapsed_sec": case["forward_elapsed_sec"],
        "inverse_total_elapsed_sec": case["inverse_total_elapsed_sec"],
        "measurement_relative_error": case["measurement_relative_error"],
        "backend_info": case["backend_info"],
        "measurement_backend_info": case["measurement_backend_info"],
        "truth_metrics": case["truth_metrics"],
        "data_path": str(data_path.relative_to(output_dir)),
    }


def _run_2d_stable_combined(
    args: argparse.Namespace, output_dir: Path, *, backend_order: str
) -> dict[str, Any]:
    mesh = create_simple_eit_mesh(
        n_elec=int(args.n_elec),
        radius=float(args.radius_2d),
        mesh_size=float(args.mesh_size_2d),
        output_dir=str(output_dir / "runtime" / "meshes_2d"),
    )
    pattern = PatternConfig(
        n_elec=int(args.n_elec),
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        rotate_meas=True,
    )

    def _make(device_name: str, petsc_name: str, cache_dir: Path) -> EITSystem:
        system = EITSystem(
            n_elec=int(args.n_elec),
            pattern_config=pattern,
            contact_impedance=np.full(
                int(args.n_elec), float(args.contact_impedance), dtype=np.float64
            ),
            base_conductivity=BACKGROUND_CONDUCTIVITY,
            regularization_type="noser",
            regularization_alpha=1.0,
            cache_scope="both",
            cache_dir=str(cache_dir),
            solver_mode="strict",
            line_search_mode="full",
            jacobian_update_every=1,
            jacobian_reuse_tol=0.0,
            petsc_device=petsc_name,
            device=device_name,
            linear_backend_config={"petsc_device": petsc_name},
        )
        system.setup(mesh=mesh)
        system.reconstructor.max_iterations = int(max(1, args.max_iterations))
        system.reconstructor.min_iterations = 1
        system.reconstructor.verbose = False
        return system

    cpu_cache = output_dir / "runtime" / "2d_cache" / "cpu"
    gpu_cache = output_dir / "runtime" / "2d_cache" / "gpu"
    cpu_cache.mkdir(parents=True, exist_ok=True)
    gpu_cache.mkdir(parents=True, exist_ok=True)

    print("[gallery-worker] 2D: build CPU system", flush=True)
    cpu = _make("cpu", "cpu", cpu_cache)
    print("[gallery-worker] 2D: build GPU system", flush=True)
    gpu = _make("cuda", "cuda", gpu_cache)

    truth_values, anomalies = _build_truth_values(cpu, dim=2)
    truth_cpu = EITImage(elem_data=truth_values.copy(), fwd_model=cpu.fwd_model)
    truth_gpu = EITImage(elem_data=truth_values.copy(), fwd_model=gpu.fwd_model)
    baseline_cpu = cpu.create_homogeneous_image(conductivity=BACKGROUND_CONDUCTIVITY)
    baseline_gpu = gpu.create_homogeneous_image(conductivity=BACKGROUND_CONDUCTIVITY)

    state: dict[str, Any] = {}

    def _run_cpu() -> None:
        print("[gallery-worker] 2D CPU: forward baseline", flush=True)
        baseline_cpu_data, cpu_forward_baseline_elapsed = _timed(
            lambda: cpu.forward_solve(baseline_cpu)
        )
        print("[gallery-worker] 2D CPU: forward target", flush=True)
        target_cpu_data, cpu_forward_target_elapsed = _timed(
            lambda: cpu.forward_solve(truth_cpu)
        )
        state.update(
            {
                "baseline_cpu_data": baseline_cpu_data,
                "cpu_forward_baseline_elapsed": cpu_forward_baseline_elapsed,
                "target_cpu_data": target_cpu_data,
                "cpu_forward_target_elapsed": cpu_forward_target_elapsed,
            }
        )

    def _run_gpu() -> None:
        print("[gallery-worker] 2D GPU: forward target", flush=True)
        target_gpu_data, gpu_forward_target_elapsed = _timed(
            lambda: gpu.forward_solve(truth_gpu)
        )
        state.update(
            {
                "target_gpu_data": target_gpu_data,
                "gpu_forward_target_elapsed": gpu_forward_target_elapsed,
            }
        )

    order = ["cpu", "gpu"] if backend_order == "cpu-first" else ["gpu", "cpu"]
    for backend_key in order:
        if backend_key == "cpu":
            _run_cpu()
        else:
            _run_gpu()

    print("[gallery-worker] 2D CPU: inverse absolute", flush=True)
    cpu_recon, cpu_inverse_elapsed = _timed(
        lambda: cpu.absolute_reconstruct(
            state["target_cpu_data"], baseline_image=baseline_cpu
        )
    )
    print("[gallery-worker] 2D GPU: inverse absolute", flush=True)
    gpu_recon, gpu_inverse_elapsed = _timed(
        lambda: gpu.absolute_reconstruct(
            state["target_cpu_data"], baseline_image=baseline_gpu
        )
    )
    print("[gallery-worker] 2D GPU: forward baseline (post-inverse)", flush=True)
    baseline_gpu_data, gpu_forward_baseline_elapsed = _timed(
        lambda: gpu.forward_solve(baseline_gpu)
    )
    state.update(
        {
            "baseline_gpu_data": baseline_gpu_data,
            "gpu_forward_baseline_elapsed": gpu_forward_baseline_elapsed,
        }
    )

    coords = _sigma_coordinates(cpu)
    cpu_truth_metrics = _truth_metrics(
        truth=truth_values,
        recon=np.asarray(cpu_recon.conductivity, dtype=np.float64),
        coords=coords,
        anomalies=anomalies,
        background_conductivity=BACKGROUND_CONDUCTIVITY,
    )
    gpu_truth_metrics = _truth_metrics(
        truth=truth_values,
        recon=np.asarray(gpu_recon.conductivity, dtype=np.float64),
        coords=coords,
        anomalies=anomalies,
        background_conductivity=BACKGROUND_CONDUCTIVITY,
    )
    consistency = _consistency_metrics(
        dim=2,
        baseline_cpu_meas=np.asarray(state["baseline_cpu_data"].meas, dtype=np.float64),
        baseline_gpu_meas=np.asarray(state["baseline_gpu_data"].meas, dtype=np.float64),
        target_cpu_meas=np.asarray(state["target_cpu_data"].meas, dtype=np.float64),
        target_gpu_meas=np.asarray(state["target_gpu_data"].meas, dtype=np.float64),
        cpu_recon=np.asarray(cpu_recon.conductivity, dtype=np.float64),
        gpu_recon=np.asarray(gpu_recon.conductivity, dtype=np.float64),
    )
    return {
        "mesh": mesh,
        "coords": coords,
        "truth_values": truth_values,
        "cpu_reconstruction": np.asarray(cpu_recon.conductivity, dtype=np.float64),
        "gpu_reconstruction": np.asarray(gpu_recon.conductivity, dtype=np.float64),
        "anomalies": anomalies,
        "cpu_case": {
            "label": "2D CPU",
            "forward_backend": "dolfinx",
            "petsc_device": "cpu",
            "forward_elapsed_sec": state["cpu_forward_target_elapsed"],
            "inverse_total_elapsed_sec": cpu_inverse_elapsed,
            "measurement_relative_error": float(cpu_recon.relative_error),
            "backend_info": cpu.fwd_model.get_backend_diagnostics(),
            "measurement_backend_info": cpu.fwd_model.get_backend_diagnostics(),
            "truth_metrics": cpu_truth_metrics,
            "baseline_measured": np.asarray(
                state["baseline_cpu_data"].meas, dtype=np.float64
            ),
            "measured": np.asarray(state["target_cpu_data"].meas, dtype=np.float64),
            "simulated": np.asarray(cpu_recon.simulated, dtype=np.float64),
            "residual": np.asarray(cpu_recon.residual, dtype=np.float64),
            "forward_baseline_elapsed_sec": state["cpu_forward_baseline_elapsed"],
        },
        "gpu_case": {
            "label": "2D GPU",
            "forward_backend": "dolfinx",
            "petsc_device": "cuda",
            "forward_elapsed_sec": state["gpu_forward_target_elapsed"],
            "inverse_total_elapsed_sec": gpu_inverse_elapsed,
            "measurement_relative_error": float(gpu_recon.relative_error),
            "backend_info": gpu.fwd_model.get_backend_diagnostics(),
            "measurement_backend_info": gpu.fwd_model.get_backend_diagnostics(),
            "truth_metrics": gpu_truth_metrics,
            "baseline_measured": np.asarray(
                state["baseline_gpu_data"].meas, dtype=np.float64
            ),
            "measured": np.asarray(state["target_gpu_data"].meas, dtype=np.float64),
            "simulated": np.asarray(gpu_recon.simulated, dtype=np.float64),
            "residual": np.asarray(gpu_recon.residual, dtype=np.float64),
            "forward_baseline_elapsed_sec": state["gpu_forward_baseline_elapsed"],
        },
        "cpu_truth_metrics": cpu_truth_metrics,
        "gpu_truth_metrics": gpu_truth_metrics,
        "consistency": consistency,
    }


def _backend_settings(
    *,
    dim: int,
    backend_key: str,
    gpu_acceleration_profile: str = DEFAULT_ACCELERATION_PROFILE,
) -> dict[str, str]:
    if int(dim) == 2:
        if backend_key == "cpu":
            return {
                "label": "2D CPU",
                "forward_backend": "dolfinx",
                "device": "cpu",
                "petsc_device": "cpu",
            }
        if backend_key == "gpu":
            return {
                "label": "2D GPU",
                "forward_backend": "dolfinx",
                "device": "cuda",
                "petsc_device": "cuda",
            }
    if int(dim) == 3:
        if backend_key == "cpu":
            return {
                "label": "3D CPU",
                "forward_backend": "dolfinx",
                "device": "cpu",
                "petsc_device": "cpu",
            }
        if backend_key == "gpu":
            return {
                "label": "3D GPU",
                "forward_backend": "cuda_structured",
                "device": "cuda",
                "petsc_device": "cuda",
                "acceleration_profile": str(gpu_acceleration_profile),
            }
    raise ValueError(f"unsupported backend {backend_key!r} for dim={dim}")


def _build_mesh(*, dim: int, args: argparse.Namespace, output_dir: Path):
    if int(dim) == 2:
        return create_simple_eit_mesh(
            n_elec=int(args.n_elec),
            radius=float(args.radius_2d),
            mesh_size=float(args.mesh_size_2d),
            output_dir=str(output_dir / "runtime" / "meshes_2d"),
        )
    mesh_family, geometry_version, generator_revision = resolve_3d_mesh_contract(
        acceleration_profile=args.gpu_acceleration_profile,
    )
    return load_or_create_mesh(
        mesh_dir=str(output_dir / "runtime" / "meshes_3d"),
        mesh_name=(
            f"gallery_real_ref{int(args.refinement_3d)}"
            f"_cf{mesh_family}_{geometry_version}_{generator_revision}"
        ),
        n_elec=int(args.n_elec),
        dimension=3,
        radius=float(args.radius_3d),
        refinement=int(args.refinement_3d),
        height=float(args.height_3d),
        electrode_height_ratio=float(args.electrode_height_ratio),
        z_center=0.0,
        electrode_coverage=float(args.electrode_coverage),
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
    )


def _build_system(
    *,
    dim: int,
    args: argparse.Namespace,
    mesh,
    cache_dir: Path,
    backend_key: str,
) -> EITSystem:
    settings = _backend_settings(
        dim=dim,
        backend_key=backend_key,
        gpu_acceleration_profile=str(args.gpu_acceleration_profile),
    )
    pattern = PatternConfig(
        n_elec=int(args.n_elec),
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized" if int(dim) == 2 else "total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        rotate_meas=True,
    )
    kwargs: dict[str, Any] = {
        "n_elec": int(args.n_elec),
        "pattern_config": pattern,
        "contact_impedance": np.full(
            int(args.n_elec), float(args.contact_impedance), dtype=np.float64
        ),
        "base_conductivity": BACKGROUND_CONDUCTIVITY,
        "regularization_type": "noser",
        "regularization_alpha": 1.0,
        "cache_scope": "both",
        "cache_dir": str(cache_dir),
        "solver_mode": "fast" if int(dim) == 3 else "strict",
        "line_search_mode": "fast" if int(dim) == 3 else "full",
        "jacobian_update_every": 1,
        "jacobian_reuse_tol": 0.0,
        "petsc_device": settings["petsc_device"],
        "device": settings["device"],
        "acceleration_profile": str(
            settings.get("acceleration_profile", DEFAULT_ACCELERATION_PROFILE)
        ),
        "linear_backend_config": {"petsc_device": settings["petsc_device"]},
    }
    if int(dim) == 3:
        mesh_family, geometry_version, generator_revision = resolve_3d_mesh_contract(
            acceleration_profile=settings.get(
                "acceleration_profile", DEFAULT_ACCELERATION_PROFILE
            ),
        )
        kwargs.update(
            {
                "forward_backend": settings["forward_backend"],
                "mesh_family": mesh_family,
                "geometry_version": geometry_version,
                "generator_revision": generator_revision,
            }
        )
    system = EITSystem(**kwargs)
    system.setup(mesh=mesh)
    system.reconstructor.max_iterations = int(max(1, args.max_iterations))
    system.reconstructor.min_iterations = 1
    system.reconstructor.verbose = False
    return system


def _run_backend_case(
    *,
    dim: int,
    args: argparse.Namespace,
    mesh,
    truth_values: np.ndarray,
    anomalies: list[dict[str, Any]],
    backend_key: str,
    cache_root: Path,
) -> dict[str, Any]:
    settings = _backend_settings(
        dim=dim,
        backend_key=backend_key,
        gpu_acceleration_profile=str(args.gpu_acceleration_profile),
    )
    measure_cache = cache_root / backend_key / "measure"
    inverse_cache = cache_root / backend_key / "inverse"
    measure_cache.mkdir(parents=True, exist_ok=True)
    inverse_cache.mkdir(parents=True, exist_ok=True)

    print(f"[gallery-worker] {settings['label']}: build measurement system", flush=True)
    measure_system = _build_system(
        dim=dim, args=args, mesh=mesh, cache_dir=measure_cache, backend_key=backend_key
    )
    print(f"[gallery-worker] {settings['label']}: build inverse system", flush=True)
    inverse_system = _build_system(
        dim=dim, args=args, mesh=mesh, cache_dir=inverse_cache, backend_key=backend_key
    )

    truth_measure = EITImage(
        elem_data=np.asarray(truth_values, dtype=np.float64).copy(),
        fwd_model=measure_system.fwd_model,
    )
    baseline_measure = measure_system.create_homogeneous_image(
        conductivity=BACKGROUND_CONDUCTIVITY
    )
    baseline_inverse = inverse_system.create_homogeneous_image(
        conductivity=BACKGROUND_CONDUCTIVITY
    )
    sync_cuda = settings["forward_backend"] == "cuda_structured"

    print(f"[gallery-worker] {settings['label']}: forward baseline", flush=True)
    baseline_data, baseline_elapsed = _timed(
        lambda: measure_system.forward_solve(baseline_measure), sync_cuda=sync_cuda
    )
    print(f"[gallery-worker] {settings['label']}: forward target", flush=True)
    target_data, target_elapsed = _timed(
        lambda: measure_system.forward_solve(truth_measure), sync_cuda=sync_cuda
    )
    print(f"[gallery-worker] {settings['label']}: inverse absolute", flush=True)
    recon, inverse_elapsed = _timed(
        lambda: inverse_system.absolute_reconstruct(
            target_data, baseline_image=baseline_inverse
        ),
        sync_cuda=sync_cuda,
    )

    coords = _sigma_coordinates(inverse_system)
    recon_values = np.asarray(recon.conductivity, dtype=np.float64)
    truth_metrics = _truth_metrics(
        truth=np.asarray(truth_values, dtype=np.float64),
        recon=recon_values,
        coords=coords,
        anomalies=anomalies,
        background_conductivity=BACKGROUND_CONDUCTIVITY,
    )
    return {
        "label": settings["label"],
        "forward_backend": settings["forward_backend"],
        "petsc_device": settings["petsc_device"],
        "backend_info": inverse_system.fwd_model.get_backend_diagnostics(),
        "measurement_backend_info": measure_system.fwd_model.get_backend_diagnostics(),
        "forward_baseline_elapsed_sec": baseline_elapsed,
        "forward_elapsed_sec": target_elapsed,
        "inverse_total_elapsed_sec": inverse_elapsed,
        "measurement_relative_error": float(recon.relative_error),
        "baseline_measured": np.asarray(baseline_data.meas, dtype=np.float64),
        "measured": np.asarray(target_data.meas, dtype=np.float64),
        "simulated": np.asarray(recon.simulated, dtype=np.float64),
        "residual": np.asarray(recon.residual, dtype=np.float64),
        "truth_metrics": truth_metrics,
        "coords": coords,
        "reconstruction": recon_values,
    }


def _run_correctness(
    args: argparse.Namespace, output_dir: Path, *, dim: int
) -> dict[str, Any]:
    if int(dim) == 2:
        return _run_2d_stable_combined(
            args, output_dir, backend_order=str(args.backend_order)
        )
    mesh = _build_mesh(dim=dim, args=args, output_dir=output_dir)
    bootstrap = _build_system(
        dim=dim,
        args=args,
        mesh=mesh,
        cache_dir=output_dir / "runtime" / f"{dim}d_bootstrap",
        backend_key="cpu",
    )
    truth_values, anomalies = _build_truth_values(bootstrap, dim=dim)
    cache_root = output_dir / "runtime" / f"{dim}d_correctness_cache"
    cpu_case = _run_backend_case(
        dim=dim,
        args=args,
        mesh=mesh,
        truth_values=truth_values,
        anomalies=anomalies,
        backend_key="cpu",
        cache_root=cache_root,
    )
    gpu_case = _run_backend_case(
        dim=dim,
        args=args,
        mesh=mesh,
        truth_values=truth_values,
        anomalies=anomalies,
        backend_key="gpu",
        cache_root=cache_root,
    )
    consistency = _consistency_metrics(
        dim=dim,
        baseline_cpu_meas=np.asarray(cpu_case["baseline_measured"], dtype=np.float64),
        baseline_gpu_meas=np.asarray(gpu_case["baseline_measured"], dtype=np.float64),
        target_cpu_meas=np.asarray(cpu_case["measured"], dtype=np.float64),
        target_gpu_meas=np.asarray(gpu_case["measured"], dtype=np.float64),
        cpu_recon=np.asarray(cpu_case["reconstruction"], dtype=np.float64),
        gpu_recon=np.asarray(gpu_case["reconstruction"], dtype=np.float64),
    )
    return {
        "mesh": mesh,
        "coords": np.asarray(cpu_case["coords"], dtype=np.float64),
        "truth_values": np.asarray(truth_values, dtype=np.float64),
        "cpu_reconstruction": np.asarray(cpu_case["reconstruction"], dtype=np.float64),
        "gpu_reconstruction": np.asarray(gpu_case["reconstruction"], dtype=np.float64),
        "anomalies": anomalies,
        "cpu_case": cpu_case,
        "gpu_case": gpu_case,
        "cpu_truth_metrics": cpu_case["truth_metrics"],
        "gpu_truth_metrics": gpu_case["truth_metrics"],
        "consistency": consistency,
    }


def _run_single_backend_correctness(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    dim: int,
    backend_key: str,
) -> dict[str, Any]:
    mesh = _build_mesh(dim=dim, args=args, output_dir=output_dir)
    bootstrap = _build_system(
        dim=dim,
        args=args,
        mesh=mesh,
        cache_dir=output_dir / "runtime" / f"{dim}d_bootstrap",
        backend_key="cpu",
    )
    truth_values, anomalies = _build_truth_values(bootstrap, dim=dim)
    cache_root = output_dir / "runtime" / f"{dim}d_backend_case_cache"
    case = _run_backend_case(
        dim=dim,
        args=args,
        mesh=mesh,
        truth_values=truth_values,
        anomalies=anomalies,
        backend_key=backend_key,
        cache_root=cache_root,
    )
    return {
        "dim": int(dim),
        "backend_key": str(backend_key),
        "mesh": mesh,
        "coords": np.asarray(case["coords"], dtype=np.float64),
        "truth_values": np.asarray(truth_values, dtype=np.float64),
        "reconstruction": np.asarray(case["reconstruction"], dtype=np.float64),
        "anomalies": anomalies,
        "case": case,
        "truth_metrics": case["truth_metrics"],
    }


def _run_fairness(
    args: argparse.Namespace, output_dir: Path, *, dim: int, backend_order: str
) -> dict[str, Any]:
    mesh = _build_mesh(dim=dim, args=args, output_dir=output_dir)
    bootstrap = _build_system(
        dim=dim,
        args=args,
        mesh=mesh,
        cache_dir=output_dir / "runtime" / f"{dim}d_bootstrap",
        backend_key="cpu",
    )
    truth_values, anomalies = _build_truth_values(bootstrap, dim=dim)
    order = ["cpu", "gpu"] if backend_order == "cpu-first" else ["gpu", "cpu"]
    cache_root = output_dir / "runtime" / f"{dim}d_fairness_cache"

    backend_runs: dict[str, Any] = {}
    for backend_key in order:
        cold = _run_backend_case(
            dim=dim,
            args=args,
            mesh=mesh,
            truth_values=truth_values,
            anomalies=anomalies,
            backend_key=backend_key,
            cache_root=cache_root,
        )
        hot = _run_backend_case(
            dim=dim,
            args=args,
            mesh=mesh,
            truth_values=truth_values,
            anomalies=anomalies,
            backend_key=backend_key,
            cache_root=cache_root,
        )
        backend_runs[backend_key] = {
            "label": cold["label"],
            "forward_backend": cold["forward_backend"],
            "petsc_device": cold["petsc_device"],
            "cold": {
                "forward_baseline_elapsed_sec": cold["forward_baseline_elapsed_sec"],
                "forward_elapsed_sec": cold["forward_elapsed_sec"],
                "inverse_total_elapsed_sec": cold["inverse_total_elapsed_sec"],
            },
            "hot": {
                "forward_baseline_elapsed_sec": hot["forward_baseline_elapsed_sec"],
                "forward_elapsed_sec": hot["forward_elapsed_sec"],
                "inverse_total_elapsed_sec": hot["inverse_total_elapsed_sec"],
            },
        }
    return {
        "dim": int(dim),
        "run_kind": "fairness",
        "backend_order": order,
        "backend_runs": backend_runs,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    dim = int(args.dim)
    if args.run_kind == "fairness":
        summary = _run_fairness(
            args, output_dir, dim=dim, backend_order=str(args.backend_order)
        )
        args.worker_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output_json.write_text(
            json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
        )
        print(json.dumps(_jsonable(summary), indent=2))
        return

    if args.backend_key != "both":
        result = _run_single_backend_correctness(
            args, output_dir, dim=dim, backend_key=str(args.backend_key)
        )
        bundle_path = output_dir / "data" / f"{dim}d_{args.backend_key}_bundle.h5"
        case_path = output_dir / "data" / f"{dim}d_{args.backend_key}_case.h5"
        write_output_bundle(
            bundle_path,
            {
                "coords": np.asarray(result["coords"], dtype=np.float64),
                "truth_values": np.asarray(result["truth_values"], dtype=np.float64),
                "reconstruction": np.asarray(
                    result["reconstruction"], dtype=np.float64
                ),
            },
            {"package_role": "gallery_backend_bundle"},
            schema=GALLERY_ARRAYS_SCHEMA,
        )
        _save_case_data(
            case_path,
            {
                "coords": np.asarray(result["coords"], dtype=np.float64),
                "truth": np.asarray(result["truth_values"], dtype=np.float64),
                "reconstruction": np.asarray(
                    result["reconstruction"], dtype=np.float64
                ),
                **{
                    k: v
                    for k, v in result["case"].items()
                    if isinstance(v, np.ndarray) or not isinstance(v, dict)
                },
            },
        )
        summary = {
            "dim": dim,
            "backend_key": str(args.backend_key),
            "anomalies": _jsonable(result["anomalies"]),
            "truth_metrics": _jsonable(result["truth_metrics"]),
            "case": _jsonable(
                _summary_case(
                    result["case"], data_path=case_path, output_dir=output_dir
                )
            ),
            "bundle_path": str(bundle_path.relative_to(output_dir)),
            "mesh_radius": float(result["mesh"].radius),
        }
        args.worker_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output_json.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    result = _run_correctness(args, output_dir, dim=dim)
    bundle_path = output_dir / "data" / f"{dim}d_worker_bundle.h5"
    cpu_case_path = output_dir / "data" / f"{dim}d_cpu_case.h5"
    gpu_case_path = output_dir / "data" / f"{dim}d_gpu_case.h5"
    write_output_bundle(
        bundle_path,
        {
            "coords": np.asarray(result["coords"], dtype=np.float64),
            "truth_values": np.asarray(result["truth_values"], dtype=np.float64),
            "cpu_reconstruction": np.asarray(
                result["cpu_reconstruction"], dtype=np.float64
            ),
            "gpu_reconstruction": np.asarray(
                result["gpu_reconstruction"], dtype=np.float64
            ),
        },
        {"package_role": "gallery_worker_bundle"},
        schema=GALLERY_ARRAYS_SCHEMA,
    )
    _save_case_data(
        cpu_case_path,
        {
            "coords": np.asarray(result["coords"], dtype=np.float64),
            "truth": np.asarray(result["truth_values"], dtype=np.float64),
            "reconstruction": np.asarray(
                result["cpu_reconstruction"], dtype=np.float64
            ),
            **{
                k: v
                for k, v in result["cpu_case"].items()
                if isinstance(v, np.ndarray) or not isinstance(v, dict)
            },
        },
    )
    _save_case_data(
        gpu_case_path,
        {
            "coords": np.asarray(result["coords"], dtype=np.float64),
            "truth": np.asarray(result["truth_values"], dtype=np.float64),
            "reconstruction": np.asarray(
                result["gpu_reconstruction"], dtype=np.float64
            ),
            **{
                k: v
                for k, v in result["gpu_case"].items()
                if isinstance(v, np.ndarray) or not isinstance(v, dict)
            },
        },
    )

    summary = {
        "dim": dim,
        "anomalies": _jsonable(result["anomalies"]),
        "cpu_truth_metrics": _jsonable(result["cpu_truth_metrics"]),
        "gpu_truth_metrics": _jsonable(result["gpu_truth_metrics"]),
        "consistency": _jsonable(result["consistency"]),
        "cpu_case": _jsonable(
            _summary_case(
                result["cpu_case"], data_path=cpu_case_path, output_dir=output_dir
            )
        ),
        "gpu_case": _jsonable(
            _summary_case(
                result["gpu_case"], data_path=gpu_case_path, output_dir=output_dir
            )
        ),
        "bundle_path": str(bundle_path.relative_to(output_dir)),
        "mesh_radius": float(result["mesh"].radius),
    }
    args.worker_output_json.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
