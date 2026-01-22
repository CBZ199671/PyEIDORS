#!/usr/bin/env python3
"""Benchmark difference reconstruction runtime (PyEIDORS single-step).

Focus: inverse solve timing only (no rendering).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from fenics import Function
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors import EITSystem
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"),
                        help="Directory containing cached meshes.")
    parser.add_argument("--n-elec", type=int, default=16,
                        help="Number of electrodes.")
    parser.add_argument("--refinements", type=int, nargs="+",
                        default=list(range(6, 25, 2)),
                        help="Mesh refinements to benchmark.")
    parser.add_argument("--radius", type=float, default=1.0,
                        help="Domain radius passed to mesh generator.")
    parser.add_argument("--electrode-coverage", type=float, default=0.5,
                        help="Electrode coverage ratio.")
    parser.add_argument("--background", type=float, default=0.008,
                        help="Background conductivity.")
    parser.add_argument("--phantom-center", type=float, nargs=2, default=(0.3, 0.2),
                        help="Center (x, y) of the circular phantom.")
    parser.add_argument("--phantom-radius", type=float, default=0.2,
                        help="Radius of the circular phantom.")
    parser.add_argument("--phantom-contrast", type=float, default=0.02,
                        help="Conductivity inside the phantom.")
    parser.add_argument("--normalized-diff", action="store_true",
                        help="Use normalized difference (vi/vh - 1) instead of raw difference.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of timing repeats per mesh (median is reported).")
    parser.add_argument("--measure-cold", action="store_true",
                        help="Measure cold timing (first call).")
    parser.add_argument("--measure-warm", action="store_true",
                        help="Measure warm timing (after warm-up).")
    parser.add_argument("--jacobian-method", choices=["efficient", "traditional"], default="efficient",
                        help="Jacobian calculator method.")
    parser.add_argument("--negate-jacobian", action="store_true",
                        help="Negate Jacobian to match EIDORS sign convention.")
    parser.add_argument("--single-step-space", choices=["parameter", "measurement"], default="measurement",
                        help="Solve single-step in parameter space (J^T J) or measurement space (J J^T).")
    parser.add_argument("--compare-solvers", action="store_true",
                        help="Compare parameter vs measurement space accuracy at a selected refinement.")
    parser.add_argument("--compare-refinement", type=int, default=None,
                        help="Refinement used for solver comparison (defaults to max refinement).")
    parser.add_argument("--difference-hyperparameter", type=float, default=1e-2,
                        help="Spatial hyperparameter (hp) applied as hp^2 * RtR.")
    parser.add_argument("--noser-exponent", type=float, default=0.5,
                        help="Exponent applied to diag(J'WJ) when forming the NOSER prior.")
    parser.add_argument("--noser-floor", type=float, default=1e-12,
                        help="Numerical floor for NOSER diagonal entries.")
    parser.add_argument("--meas-weight-strategy", choices=["baseline", "difference", "none"], default="baseline",
                        help="Measurement weighting strategy.")
    parser.add_argument("--meas-weight-floor", type=float, default=1e-9,
                        help="Minimum allowable measurement weight.")
    parser.add_argument("--conductivity-bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=(1e-6, 10.0),
                        help="Bounds enforced on reconstructed conductivities.")
    parser.add_argument("--step-size-calibration", action="store_true",
                        help="Enable 1-D scale search on the single-step update.")
    parser.add_argument("--step-size-min", type=float, default=1e-5,
                        help="Lower bound for the step-size calibration interval.")
    parser.add_argument("--step-size-max", type=float, default=1e1,
                        help="Upper bound for the step-size calibration interval.")
    parser.add_argument("--step-size-maxiter", type=int, default=50,
                        help="Maximum iterations for step-size optimization.")
    parser.add_argument("--csv-out", type=Path,
                        default=Path("docs/benchmarks/difference_runtime.csv"),
                        help="CSV output path.")
    parser.add_argument("--plot-out", type=Path, default=None,
                        help="Optional plot output path.")
    return parser.parse_args()


def build_pattern_config(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        rotate_meas=True,
    )


def compute_difference(target: np.ndarray, baseline: np.ndarray, normalized: bool) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if not normalized:
        return target - baseline
    safe = baseline.copy()
    safe[np.abs(safe) < np.finfo(float).eps] = np.finfo(float).eps
    return target / safe - 1.0


def build_measurement_weights(baseline_vector: np.ndarray,
                              diff_vector: np.ndarray,
                              strategy: str,
                              floor: float) -> Optional[np.ndarray]:
    if strategy == "none":
        return None
    if strategy == "difference":
        reference = np.asarray(diff_vector, dtype=float)
    else:
        reference = np.asarray(baseline_vector, dtype=float)
    weights = reference ** 2
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, floor)
    return weights


def solve_single_step_delta(fwd_model,
                            baseline_image: EITImage,
                            raw_diff: np.ndarray,
                            raw_baseline: np.ndarray,
                            args: argparse.Namespace,
                            solver_space: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    jacobian_calculator = DirectJacobianCalculator(fwd_model)
    sigma_fn = Function(fwd_model.V_sigma)
    sigma_fn.vector()[:] = baseline_image.elem_data
    jacobian = jacobian_calculator.calculate(sigma_fn, method=args.jacobian_method)
    if args.negate_jacobian:
        jacobian = -jacobian

    weights = build_measurement_weights(raw_baseline, raw_diff,
                                        args.meas_weight_strategy, args.meas_weight_floor)
    dv = np.asarray(raw_diff, dtype=float)
    if weights is None:
        jacobian_weighted = jacobian
        dv_weighted = dv
    else:
        sqrt_weights = np.sqrt(weights)
        jacobian_weighted = jacobian * sqrt_weights[:, None]
        dv_weighted = dv * sqrt_weights

    diag_entries = np.sum(jacobian_weighted ** 2, axis=0)
    diag_entries = np.maximum(diag_entries, args.noser_floor)
    noser_diag = diag_entries ** args.noser_exponent

    hp = max(args.difference_hyperparameter, 0.0)
    solver_choice = solver_space or args.single_step_space
    if solver_choice == "measurement":
        inv_noser = 1.0 / noser_diag
        jw_scaled = jacobian_weighted * inv_noser[None, :]
        lhs = jw_scaled @ jacobian_weighted.T
        if hp > 0:
            lhs = lhs + (hp ** 2) * np.eye(lhs.shape[0])
        rhs = dv_weighted
        try:
            y = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            y, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        delta = inv_noser * (jacobian_weighted.T @ y)
    else:
        RtR = np.diag(noser_diag)
        lhs = jacobian_weighted.T @ jacobian_weighted + (hp ** 2) * RtR
        rhs = jacobian_weighted.T @ dv_weighted
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    return delta, weights


def build_single_step_image(fwd_model: object,
                            baseline_image: EITImage,
                            delta: np.ndarray,
                            step_size: float,
                            bounds: Tuple[float, float]) -> EITImage:
    min_cond, max_cond = bounds
    elem = baseline_image.elem_data + step_size * delta
    elem = np.clip(elem, min_cond, max_cond)
    return EITImage(elem_data=elem, fwd_model=fwd_model)


def predict_difference(system: EITSystem,
                       image: EITImage,
                       baseline_meas: np.ndarray,
                       normalized: bool) -> np.ndarray:
    sim, _ = system.fwd_model.fwd_solve(image)
    return compute_difference(sim.meas, baseline_meas, normalized)


def optimize_step_size(system: EITSystem,
                       baseline_image: EITImage,
                       delta: np.ndarray,
                       target_diff: np.ndarray,
                       baseline_meas: np.ndarray,
                       normalized: bool,
                       args: argparse.Namespace) -> float:
    bounds = (args.step_size_min, args.step_size_max)

    def objective(step: float) -> float:
        test_img = build_single_step_image(system.fwd_model, baseline_image,
                                           delta, step, args.conductivity_bounds)
        pred = predict_difference(system, test_img, baseline_meas, normalized)
        residual = pred - target_diff
        return float(np.mean(residual ** 2))

    result = minimize_scalar(objective, bounds=bounds, method="bounded",
                             options={"maxiter": args.step_size_maxiter})
    if result.success:
        return float(result.x)
    return 1.0


def timed_single_step(system: EITSystem,
                      baseline_image: EITImage,
                      baseline_meas: np.ndarray,
                      target_meas: np.ndarray,
                      args: argparse.Namespace) -> None:
    diff = compute_difference(target_meas, baseline_meas, args.normalized_diff)
    delta, _ = solve_single_step_delta(system.fwd_model, baseline_image, diff,
                                       baseline_meas, args)
    if args.step_size_calibration:
        step_size = optimize_step_size(system, baseline_image, delta, diff,
                                       baseline_meas, args.normalized_diff, args)
    else:
        step_size = 1.0
    _ = build_single_step_image(system.fwd_model, baseline_image,
                                delta, step_size, args.conductivity_bounds)


def main() -> None:
    args = parse_args()
    if not args.measure_cold and not args.measure_warm:
        args.measure_cold = True
        args.measure_warm = True

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    if args.plot_out is not None:
        args.plot_out.parent.mkdir(parents=True, exist_ok=True)

    compare_refinement = args.compare_refinement
    if compare_refinement is None and args.refinements:
        compare_refinement = max(args.refinements)

    rows = []
    for refinement in args.refinements:
        pattern = build_pattern_config(args.n_elec)
        system = EITSystem(n_elec=args.n_elec, pattern_config=pattern)
        mesh = load_or_create_mesh(
            mesh_dir=str(args.mesh_dir),
            n_elec=args.n_elec,
            refinement=refinement,
            radius=args.radius,
            electrode_coverage=args.electrode_coverage,
        )
        system.setup(mesh=mesh)

        baseline_image = system.create_homogeneous_image(conductivity=args.background)
        sigma = create_custom_phantom(
            system.fwd_model,
            background_conductivity=args.background,
            anomalies=[{
                "center": tuple(args.phantom_center),
                "radius": args.phantom_radius,
                "conductivity": args.phantom_contrast,
            }],
        )
        phantom_image = EITImage(elem_data=sigma.vector()[:], fwd_model=system.fwd_model)

        baseline_meas, _ = system.fwd_model.fwd_solve(baseline_image)
        target_meas, _ = system.fwd_model.fwd_solve(phantom_image)

        cold_time: Optional[float] = None
        warm_time: Optional[float] = None

        if args.measure_cold:
            t0 = time.perf_counter()
            timed_single_step(system, baseline_image,
                              baseline_meas.meas, target_meas.meas, args)
            cold_time = time.perf_counter() - t0

        if args.measure_warm:
            timed_single_step(system, baseline_image,
                              baseline_meas.meas, target_meas.meas, args)
            times: List[float] = []
            for _ in range(max(1, args.repeat)):
                t0 = time.perf_counter()
                timed_single_step(system, baseline_image,
                                  baseline_meas.meas, target_meas.meas, args)
                times.append(time.perf_counter() - t0)
            warm_time = float(np.median(times))

        n_elements = len(system.fwd_model.V_sigma.dofmap().dofs())
        n_nodes = int(mesh.num_vertices())
        rows.append({
            "refinement": refinement,
            "nodes": n_nodes,
            "elements": n_elements,
            "cold_sec": cold_time,
            "warm_sec": warm_time,
        })

        line = f"ref={refinement} nodes={n_nodes} elems={n_elements}"
        if cold_time is not None:
            line += f" cold={cold_time:.4f}s"
        if warm_time is not None:
            line += f" warm={warm_time:.4f}s"
        print(line)

        if args.compare_solvers and refinement == compare_refinement:
            diff_target = compute_difference(target_meas.meas, baseline_meas.meas, args.normalized_diff)
            delta_param, _ = solve_single_step_delta(
                system.fwd_model,
                baseline_image,
                diff_target,
                baseline_meas.meas,
                args,
                solver_space="parameter",
            )
            delta_meas, _ = solve_single_step_delta(
                system.fwd_model,
                baseline_image,
                diff_target,
                baseline_meas.meas,
                args,
                solver_space="measurement",
            )

            def _solve_with_delta(delta: np.ndarray) -> Tuple[float, np.ndarray]:
                if args.step_size_calibration:
                    step = optimize_step_size(
                        system, baseline_image, delta, diff_target,
                        baseline_meas.meas, args.normalized_diff, args,
                    )
                else:
                    step = 1.0
                img = build_single_step_image(
                    system.fwd_model, baseline_image, delta,
                    step, args.conductivity_bounds,
                )
                pred = predict_difference(system, img, baseline_meas.meas, args.normalized_diff)
                rmse = float(np.sqrt(np.mean((pred - diff_target) ** 2)))
                return rmse, pred

            rmse_param, pred_param = _solve_with_delta(delta_param)
            rmse_meas, pred_meas = _solve_with_delta(delta_meas)
            delta_rel = float(np.linalg.norm(delta_meas - delta_param) /
                              (np.linalg.norm(delta_param) + 1e-12))
            pred_rel = float(np.linalg.norm(pred_meas - pred_param) /
                             (np.linalg.norm(pred_param) + 1e-12))
            print(
                "Solver comparison (parameter vs measurement): "
                f"delta_rel={delta_rel:.3e}, "
                f"rmse_param={rmse_param:.3e}, rmse_meas={rmse_meas:.3e}, "
                f"pred_rel={pred_rel:.3e}"
            )

    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["refinement", "nodes", "elements", "cold_sec", "warm_sec"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.plot_out is not None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        elements = [row["elements"] for row in rows]
        if args.measure_cold:
            cold_times = [row["cold_sec"] for row in rows]
            ax.plot(elements, cold_times, marker="o", linewidth=2,
                    label="Cold", color="#4C78A8")
        if args.measure_warm:
            warm_times = [row["warm_sec"] for row in rows]
            ax.plot(elements, warm_times, marker="o", linewidth=2,
                    label="Warm", color="#F58518")
        ax.set_ylabel("Time (s)")
        ax.set_xlabel("Elements")
        ax.set_title("PyEIDORS Difference Runtime vs Mesh Density")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(args.plot_out, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
