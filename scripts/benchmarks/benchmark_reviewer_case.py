#!/usr/bin/env python3
"""Single-case benchmark runner for reviewer-facing SoftwareX benchmarks.

This script benchmarks one framework/task/configuration at a time and writes a
single JSON result row using a shared schema. Running each case in its own
process keeps timing and peak RSS measurements isolated.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fenics import Function  # noqa: E402

from pyeidors import EITSystem  # noqa: E402
from pyeidors.data.structures import EITImage, PatternConfig  # noqa: E402
from pyeidors.data.synthetic_data import create_custom_phantom  # noqa: E402
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh  # noqa: E402
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian  # noqa: E402
from pyeidors.inverse.regularization.smoothness import NOSERRegularization  # noqa: E402
from pyeidors.inverse.solvers.gauss_newton import ModularGaussNewtonReconstructor  # noqa: E402

from benchmark_difference_runtime import (  # noqa: E402
    build_single_step_namespace,
    compute_difference,
    run_single_step_benchmark,
)

GPU_SCOPE_NOTE = (
    "GPU acceleration currently benefits inverse/tensor operations; "
    "forward PDE assembly remains on the FEniCS/CPU side."
)

ACTIVE_TRACKER: "BenchmarkStateTracker | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["pyeidors", "pyeit"], required=True)
    parser.add_argument(
        "--task",
        choices=["forward", "jacobian", "difference", "absolute_gn", "multi_frame_difference"],
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--scenario", default="low_z", choices=["low_z", "high_z"])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--n-frames", type=int, default=1)
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"))
    parser.add_argument("--difference-hyperparameter", type=float, default=None)
    parser.add_argument("--absolute-lambda", type=float, default=1e-2)
    parser.add_argument("--absolute-max-iter", type=int, default=15)
    parser.add_argument("--gn-path", choices=["optimized", "legacy_dense"], default="optimized")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--phase", default="")
    parser.add_argument("--state-json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


PYEIDORS_REFINEMENTS = {
    "coarse": 6,
    "medium": 12,
    "fine": 18,
}

PYEIT_H0 = {
    "coarse": 0.22,
    "medium": 0.14,
    "fine": 0.10,
}

SCENARIO_CONFIG = {
    "low_z": {
        "contact_impedance": 1e-6,
        "background": 1.0,
        "phantom_conductivity": 2.0,
        "phantom_center": (0.30, 0.20),
        "phantom_radius": 0.20,
        "label": "saline-like",
    },
    "high_z": {
        "contact_impedance": 1e-2,
        "background": 1.0,
        "phantom_conductivity": 2.0,
        "phantom_center": (0.25, -0.22),
        "phantom_radius": 0.18,
        "label": "plant-like",
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def make_pattern_config(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        rotate_meas=True,
    )


def summarise_times(times: Iterable[float]) -> Dict[str, float]:
    values = [float(t) for t in times]
    if not values:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "iqr": math.nan}
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(values)),
        "iqr": float(q75 - q25),
    }


def conductivity_metrics(true_values: np.ndarray, recon_values: np.ndarray) -> Dict[str, float]:
    error = np.asarray(recon_values, dtype=float).reshape(-1) - np.asarray(true_values, dtype=float).reshape(-1)
    true_values = np.asarray(true_values, dtype=float).reshape(-1)
    return {
        "conductivity_mae": float(np.mean(np.abs(error))),
        "conductivity_rmse": float(np.sqrt(np.mean(error ** 2))),
        "conductivity_relative_error_pct": float(
            np.linalg.norm(error) / max(np.linalg.norm(true_values), np.finfo(float).eps) * 100.0
        ),
    }


def voltage_metrics(target: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    error = np.asarray(predicted, dtype=float).reshape(-1) - np.asarray(target, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    return {
        "voltage_rmse": float(np.sqrt(np.mean(error ** 2))),
        "voltage_mae": float(np.mean(np.abs(error))),
        "voltage_relative_error_pct": float(
            np.linalg.norm(error) / max(np.linalg.norm(target), np.finfo(float).eps) * 100.0
        ),
    }


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


class BenchmarkStateTracker:
    """Persist progress for long-running benchmark cases."""

    def __init__(self, args: argparse.Namespace):
        self.path = args.state_json
        self.args = args
        self.started_at = utc_now_iso()
        self.start_perf = time.perf_counter()
        self.completed_warmups = 0
        self.completed_repeats = 0
        self.last_duration_s: Optional[float] = None

    def update(
        self,
        status: str,
        *,
        case=None,
        completed_warmups: Optional[int] = None,
        completed_repeats: Optional[int] = None,
        last_duration_s: Optional[float] = None,
    ) -> None:
        if self.path is None:
            return
        if completed_warmups is not None:
            self.completed_warmups = int(completed_warmups)
        if completed_repeats is not None:
            self.completed_repeats = int(completed_repeats)
        if last_duration_s is not None:
            self.last_duration_s = float(last_duration_s)

        payload = {
            "run_id": self.args.run_id or "",
            "case_id": self.args.case_id or "",
            "framework": self.args.framework,
            "task": self.args.task,
            "mesh_level": self.args.mesh_level,
            "scenario": self.args.scenario,
            "device": self.args.device,
            "n_frames": int(self.args.n_frames),
            "output_json": self.args.output_json.as_posix(),
            "started_at": self.started_at,
            "updated_at": utc_now_iso(),
            "phase": self.args.phase or "",
            "completed_warmups": self.completed_warmups,
            "completed_repeats": self.completed_repeats,
            "elapsed_s": float(time.perf_counter() - self.start_perf),
            "last_duration_s": self.last_duration_s,
            "status": status,
        }
        if case is not None:
            payload["mesh_name"] = case.mesh_name
            payload["n_nodes"] = int(case.n_nodes)
            payload["n_elements"] = int(case.n_elements)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def handle_termination(signum, _frame) -> None:
    if ACTIVE_TRACKER is not None:
        ACTIVE_TRACKER.update("stopped")
    raise SystemExit(128 + signum)


class PyEidorsCase:
    def __init__(self, args: argparse.Namespace):
        cfg = SCENARIO_CONFIG[args.scenario]
        self.args = args
        self.cfg = cfg
        self.pattern = make_pattern_config(args.n_elec)
        self.refinement = PYEIDORS_REFINEMENTS[args.mesh_level]
        self.mesh = load_or_create_mesh(
            mesh_dir=str(args.mesh_dir),
            n_elec=args.n_elec,
            refinement=self.refinement,
            radius=1.0,
            electrode_coverage=0.5,
        )
        z_contact = np.ones(args.n_elec) * cfg["contact_impedance"]
        self.system = EITSystem(
            n_elec=args.n_elec,
            pattern_config=self.pattern,
            contact_impedance=z_contact,
            base_conductivity=cfg["background"],
            regularization_type="noser",
            regularization_alpha=1.0,
            noser_exponent=0.5,
        )
        self.system.setup(mesh=self.mesh)
        self.absolute_jacobian_backend = ""

        if args.task == "absolute_gn":
            use_torch = args.device == "gpu"
            use_legacy_dense = args.gn_path == "legacy_dense"
            jacobian_calculator = EidorsStyleAdjointJacobian(
                self.system.fwd_model,
                use_torch=use_torch,
                device="cuda:0" if use_torch else None,
            )
            regularization = NOSERRegularization(
                self.system.fwd_model,
                jacobian_calculator,
                base_conductivity=cfg["background"],
                alpha=1.0,
                exponent=0.5,
                floor=1e-12,
            )
            device_name = "cuda:0" if args.device == "gpu" else "cpu"
            self.system.reconstructor = ModularGaussNewtonReconstructor(
                fwd_model=self.system.fwd_model,
                jacobian_calculator=jacobian_calculator,
                regularization=regularization,
                max_iterations=args.absolute_max_iter,
                regularization_param=args.absolute_lambda,
                device=device_name,
                verbose=False,
                negate_jacobian=False,
                use_prior_term=True,
                enable_measurement_space_fast_path=not use_legacy_dense,
                enable_diagonal_regularization_cache=not use_legacy_dense,
            )
            self.absolute_jacobian_backend = type(jacobian_calculator).__name__

        self.baseline_image = self.system.create_homogeneous_image(conductivity=cfg["background"])
        sigma = create_custom_phantom(
            self.system.fwd_model,
            background_conductivity=cfg["background"],
            anomalies=[{
                "center": tuple(cfg["phantom_center"]),
                "radius": cfg["phantom_radius"],
                "conductivity": cfg["phantom_conductivity"],
            }],
        )
        self.truth_image = EITImage(elem_data=sigma.vector()[:], fwd_model=self.system.fwd_model)
        self.truth_sigma = self.truth_image.elem_data.copy()
        self.baseline_meas, _ = self.system.fwd_model.fwd_solve(self.baseline_image)
        self.target_meas, _ = self.system.fwd_model.fwd_solve(self.truth_image)

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.num_vertices())

    @property
    def n_elements(self) -> int:
        return len(self.system.fwd_model.V_sigma.dofmap().dofs())

    @property
    def mesh_name(self) -> str:
        return getattr(self.mesh, "mesh_name", f"ref{self.refinement}")

    def forward(self) -> Dict[str, float]:
        sim, _ = self.system.fwd_model.fwd_solve(self.truth_image)
        return voltage_metrics(self.target_meas.meas, sim.meas)

    def jacobian(self) -> Dict[str, float]:
        calc = EidorsStyleAdjointJacobian(self.system.fwd_model, use_torch=self.args.device == "gpu")
        matrix = calc.calculate_from_image(self.truth_image)
        return {
            "jacobian_rows": int(matrix.shape[0]),
            "jacobian_cols": int(matrix.shape[1]),
            "matrix_norm": float(np.linalg.norm(matrix)),
        }

    def difference(self) -> Dict[str, float]:
        diff_vector = compute_difference(self.target_meas.meas, self.baseline_meas.meas, False)
        single_step_kwargs: Dict[str, float] = {}
        if self.args.difference_hyperparameter is not None:
            single_step_kwargs["difference_hyperparameter"] = self.args.difference_hyperparameter
        single_step_args = build_single_step_namespace(self.system, **single_step_kwargs)
        recon_image, predicted_diff, step_size = run_single_step_benchmark(
            self.system,
            self.baseline_image,
            self.baseline_meas.meas,
            self.target_meas.meas,
            single_step_args,
            target_diff=diff_vector,
        )
        result = {}
        result.update(voltage_metrics(diff_vector, predicted_diff))
        result.update(conductivity_metrics(self.truth_sigma, recon_image.elem_data))
        result["optimal_step_size"] = float(step_size)
        return result

    def absolute_gn(self) -> Dict[str, float]:
        initial_sigma = np.full(self.n_elements, self.cfg["background"], dtype=float)
        recon_result = self.system.reconstructor.reconstruct(
            measured_data=self.target_meas,
            initial_conductivity=initial_sigma,
            jacobian_method="efficient",
        )
        conductivity_fn = recon_result["conductivity"]
        conductivity_vec = conductivity_fn.vector()[:]
        sim_data, _ = self.system.fwd_model.fwd_solve(EITImage(elem_data=conductivity_vec, fwd_model=self.system.fwd_model))
        result = {}
        result.update(voltage_metrics(self.target_meas.meas, sim_data.meas))
        result.update(conductivity_metrics(self.truth_sigma, conductivity_vec))
        result["final_residual"] = float(recon_result.get("final_residual", np.nan))
        result["iterations"] = int(recon_result.get("iterations", self.args.absolute_max_iter))
        result["jacobian_backend"] = recon_result.get("jacobian_backend", self.absolute_jacobian_backend)
        result["linear_solver"] = recon_result.get("linear_solver", "unknown")
        result["linear_system_rows"] = int(recon_result.get("linear_system_rows", self.n_elements))
        result["linear_system_cols"] = int(recon_result.get("linear_system_cols", self.n_elements))
        result["regularization_structure"] = recon_result.get("regularization_structure", "unknown")
        result["gpu_scope_note"] = recon_result.get("gpu_scope_note", GPU_SCOPE_NOTE)
        result["used_measurement_space_fast_path"] = bool(
            recon_result.get("used_measurement_space_fast_path", False)
        )
        result["gn_path"] = self.args.gn_path
        result["enable_measurement_space_fast_path"] = bool(
            recon_result.get("enable_measurement_space_fast_path", self.args.gn_path != "legacy_dense")
        )
        result["enable_diagonal_regularization_cache"] = bool(
            recon_result.get("enable_diagonal_regularization_cache", self.args.gn_path != "legacy_dense")
        )
        return result

    def multi_frame_difference(self) -> Dict[str, float]:
        frames = max(1, self.args.n_frames)
        metrics = self.difference()
        total_voltage_rmse = metrics["voltage_rmse"]
        total_cond_rel = metrics["conductivity_relative_error_pct"]
        for _ in range(frames - 1):
            metrics = self.difference()
            total_voltage_rmse += metrics["voltage_rmse"]
            total_cond_rel += metrics["conductivity_relative_error_pct"]
        return {
            "avg_voltage_rmse": total_voltage_rmse / frames,
            "avg_conductivity_relative_error_pct": total_cond_rel / frames,
            "n_frames": int(frames),
        }


class PyEitCase:
    def __init__(self, args: argparse.Namespace):
        from pyeit.eit.fem import EITForward
        from pyeit.eit.jac import JAC
        from pyeit.eit.protocol import create as create_protocol
        from pyeit.mesh import create
        from pyeit.mesh.wrapper import PyEITAnomaly_Circle, set_perm

        cfg = SCENARIO_CONFIG[args.scenario]
        self.args = args
        self.cfg = cfg
        self._EITForward = EITForward
        self._JAC = JAC
        self._set_perm = set_perm
        self._anomaly_cls = PyEITAnomaly_Circle
        h0 = PYEIT_H0[args.mesh_level]
        self.mesh = create(n_el=args.n_elec, h0=h0)
        self.protocol = create_protocol(n_el=args.n_elec, dist_exc=1, step_meas=1, parser_meas="std")
        self.forward_solver = EITForward(self.mesh, self.protocol)
        self.baseline_perm = np.ones(self.mesh.n_elems, dtype=float) * cfg["background"]
        anomaly = PyEITAnomaly_Circle(center=list(cfg["phantom_center"]), r=cfg["phantom_radius"], perm=cfg["phantom_conductivity"])
        self.truth_mesh = set_perm(self.mesh, anomaly=anomaly, background=cfg["background"])
        self.truth_perm = np.asarray(self.truth_mesh.perm_array, dtype=float).copy()
        self.v0 = np.asarray(self.forward_solver.solve_eit(perm=self.baseline_perm)).ravel()
        self.v1 = np.asarray(EITForward(self.truth_mesh, self.protocol).solve_eit(perm=self.truth_perm)).ravel()

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.n_nodes)

    @property
    def n_elements(self) -> int:
        return int(self.mesh.n_elems)

    @property
    def mesh_name(self) -> str:
        return f"pyeit_{self.args.mesh_level}"

    def forward(self) -> Dict[str, float]:
        pred = np.asarray(self.forward_solver.solve_eit(perm=self.truth_perm)).ravel()
        return voltage_metrics(self.v1, pred)

    def jacobian(self) -> Dict[str, float]:
        jac = self._JAC(self.mesh, self.protocol)
        jac.setup(p=0.5, lamb=0.01, method="kotre", perm=self.baseline_perm)
        matrix, _ = jac.fwd.compute_jac(self.baseline_perm)
        return {
            "jacobian_rows": int(matrix.shape[0]),
            "jacobian_cols": int(matrix.shape[1]),
            "matrix_norm": float(np.linalg.norm(matrix)),
        }

    def difference(self) -> Dict[str, float]:
        jac = self._JAC(self.mesh, self.protocol)
        jac.setup(p=0.5, lamb=0.01, method="kotre", perm=self.baseline_perm)
        ds = np.asarray(jac.solve(self.v1, self.v0)).ravel()
        predicted_perm = self.baseline_perm + ds
        pred_v = np.asarray(self.forward_solver.solve_eit(perm=predicted_perm)).ravel()
        result = {}
        result.update(voltage_metrics(self.v1 - self.v0, pred_v - self.v0))
        result.update(conductivity_metrics(self.truth_perm, predicted_perm))
        return result

    def absolute_gn(self) -> Dict[str, float]:
        jac = self._JAC(self.mesh, self.protocol)
        jac.setup(p=0.5, lamb=0.01, method="kotre", perm=self.baseline_perm)
        sigma = np.asarray(
            jac.gn(
                self.v1,
                x0=self.baseline_perm,
                maxiter=self.args.absolute_max_iter,
                p=0.5,
                lamb=0.01,
                method="kotre",
                verbose=False,
            )
        ).ravel()
        pred_v = np.asarray(self.forward_solver.solve_eit(perm=sigma)).ravel()
        result = {}
        result.update(voltage_metrics(self.v1, pred_v))
        result.update(conductivity_metrics(self.truth_perm, sigma))
        result["iterations"] = int(self.args.absolute_max_iter)
        result["jacobian_backend"] = "JAC"
        result["linear_solver"] = "pyeit_builtin_gn"
        result["linear_system_rows"] = int(self.n_elements)
        result["linear_system_cols"] = int(self.n_elements)
        result["regularization_structure"] = "unknown"
        result["gpu_scope_note"] = GPU_SCOPE_NOTE
        result["used_measurement_space_fast_path"] = False
        return result

    def multi_frame_difference(self) -> Dict[str, float]:
        frames = max(1, self.args.n_frames)
        metrics = self.difference()
        total_voltage_rmse = metrics["voltage_rmse"]
        total_cond_rel = metrics["conductivity_relative_error_pct"]
        for _ in range(frames - 1):
            metrics = self.difference()
            total_voltage_rmse += metrics["voltage_rmse"]
            total_cond_rel += metrics["conductivity_relative_error_pct"]
        return {
            "avg_voltage_rmse": total_voltage_rmse / frames,
            "avg_conductivity_relative_error_pct": total_cond_rel / frames,
            "n_frames": int(frames),
        }


def run_task(
    case,
    task_name: str,
    args: argparse.Namespace,
    tracker: BenchmarkStateTracker,
) -> tuple[Dict[str, float], float]:
    task_fn: Callable[[], Dict[str, float]] = getattr(case, task_name)
    tracker.update("running", case=case, completed_warmups=0, completed_repeats=0)

    for warmup_index in range(max(0, args.warmups)):
        warmup_start = time.perf_counter()
        task_fn()
        tracker.update(
            "running",
            case=case,
            completed_warmups=warmup_index + 1,
            completed_repeats=0,
            last_duration_s=time.perf_counter() - warmup_start,
        )

    timings = []
    last_metrics: Dict[str, float] = {}
    last_duration = 0.0
    for repeat_index in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        last_metrics = task_fn()
        last_duration = time.perf_counter() - t0
        timings.append(last_duration)
        tracker.update(
            "running",
            case=case,
            completed_warmups=args.warmups,
            completed_repeats=repeat_index + 1,
            last_duration_s=last_duration,
        )

    summary = summarise_times(timings)
    summary["peak_rss_mb"] = get_peak_rss_mb()
    summary.update(last_metrics)
    return summary, last_duration


def build_row(case, metrics: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    mean_value = metrics.pop("mean")
    std_value = metrics.pop("std")
    median_value = metrics.pop("median")
    iqr_value = metrics.pop("iqr")
    row = {
        "framework": args.framework,
        "task": args.task,
        "mesh_level": args.mesh_level,
        "mesh_name": case.mesh_name,
        "scenario": args.scenario,
        "n_nodes": int(case.n_nodes),
        "n_elements": int(case.n_elements),
        "n_frames": int(args.n_frames if args.task == "multi_frame_difference" else 1),
        "device": args.device if args.framework == "pyeidors" and args.task in {"jacobian", "absolute_gn"} else "cpu",
        "warmups": int(args.warmups),
        "repeats": int(args.repeats),
        "commit": get_git_commit(),
        "peak_rss_mb": metrics.pop("peak_rss_mb"),
        "gn_path": getattr(args, "gn_path", ""),
        "mean": mean_value,
        "std": std_value,
        "median": median_value,
        "iqr": iqr_value,
        # Keep temporary aliases until all downstream consumers switch to the canonical schema.
        "mean_sec": mean_value,
        "std_sec": std_value,
        "median_sec": median_value,
        "iqr_sec": iqr_value,
    }
    row.update(metrics)
    return row


def main(args: argparse.Namespace) -> None:
    global ACTIVE_TRACKER
    ACTIVE_TRACKER = BenchmarkStateTracker(args)
    ACTIVE_TRACKER.update("running")

    case = PyEidorsCase(args) if args.framework == "pyeidors" else PyEitCase(args)
    metrics, last_duration = run_task(case, args.task, args, ACTIVE_TRACKER)
    row = build_row(case, metrics, args)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(row, indent=2), encoding="utf-8")

    ACTIVE_TRACKER.update(
        "completed",
        case=case,
        completed_warmups=args.warmups,
        completed_repeats=max(1, args.repeats),
        last_duration_s=last_duration,
    )
    if args.verbose:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    args = parse_args()
    signal.signal(signal.SIGINT, handle_termination)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_termination)
    try:
        main(args)
    except KeyboardInterrupt:
        if ACTIVE_TRACKER is not None:
            ACTIVE_TRACKER.update("stopped")
        raise
    except SystemExit:
        raise
    except Exception:
        if ACTIVE_TRACKER is not None:
            ACTIVE_TRACKER.update("failed")
        raise
