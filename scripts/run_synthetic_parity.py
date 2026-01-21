#!/usr/bin/env python3
"""Generate synthetic EIT data and compare PyEIDORS predictions against references.

This script serves two roles:
  1. Create a ground-truth phantom, run forward simulations and store the
     resulting voltage measurements for later reuse (e.g., inside EIDORS).
  2. Reconstruct the phantom with PyEIDORS (absolute and/or difference modes)
     and report voltage/residual metrics, optionally comparing against a set of
     reference voltages exported from another tool.

Example usage (difference mode only):

    python scripts/run_synthetic_parity.py \
      --output-root results/simulation_parity/run01 \
      --mode both \
      --phantom-center 0.3 0.2 --phantom-radius 0.2 --phantom-contrast 1.5

Example usage comparing against EIDORS voltages:

    python scripts/run_synthetic_parity.py \
      --output-root results/simulation_parity/eidors_cmp \
      --eidors-csv external/eidors_diff_voltages.csv
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from fenics import Function
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

from pyeidors import EITSystem
from pyeidors.data.synthetic_data import create_custom_phantom
from pyeidors.data.structures import EITData, EITImage, PatternConfig
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from pyeidors.visualization import create_visualizer


@dataclass
class Metrics:
    rmse: float
    mae: float
    max_abs_error: float
    relative_error_pct: float
    correlation: Optional[float]

    def to_dict(self) -> Dict[str, float]:
        data: Dict[str, float] = {
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "relative_error_pct": self.relative_error_pct,
        }
        if self.correlation is not None and not np.isnan(self.correlation):
            data["correlation"] = self.correlation
        else:
            data["correlation"] = None
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/simulation_parity"),
                        help="Directory where metrics, figures and CSV files will be stored.")
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"),
                        help="Directory containing cached meshes (passed to load_or_create_mesh).")
    parser.add_argument("--mesh-name", type=str, default=None,
                        help="Optional mesh name to reuse; if omitted a mesh will be created.")
    parser.add_argument("--n-elec", type=int, default=16,
                        help="Number of electrodes for the synthetic test.")
    parser.add_argument("--refinement", type=int, default=12,
                        help="Mesh refinement passed to load_or_create_mesh when generating a mesh.")
    parser.add_argument("--mesh-radius", type=float, default=1.0,
                        help="Radius passed to load_or_create_mesh (used for caching key).")
    parser.add_argument("--electrode-coverage", type=float, default=0.5,
                        help="Fraction of perimeter covered by each electrode (affects cache key).")
    parser.add_argument("--background", type=float, default=1.0,
                        help="Background conductivity used for simulations.")
    parser.add_argument("--phantom-center", type=float, nargs=2, default=(0.3, 0.2),
                        help="Center (x, y) of the circular phantom.")
    parser.add_argument("--phantom-radius", type=float, default=0.2,
                        help="Radius of the circular phantom.")
    parser.add_argument("--phantom-contrast", type=float, default=1.5,
                        help="Conductivity inside the phantom (absolute units).")
    parser.add_argument("--mode", choices=["difference", "absolute", "both"], default="both",
                        help="Which reconstruction pipeline(s) to execute.")
    parser.add_argument("--eidors-csv", type=Path,
                        help="Optional CSV file produced by EIDORS to compare voltages against.")
    parser.add_argument("--csv-delimiter", type=str, default=",",
                        help="Delimiter used when parsing --eidors-csv (default ',').")
    parser.add_argument("--save-forward-csv", action="store_true",
                        help="Whether to save synthetic voltages (baseline & phantom) to CSV.")
    parser.add_argument("--figure-dpi", type=int, default=300,
                        help="DPI used when saving matplotlib figures.")
    parser.add_argument("--difference-solver", choices=["gauss-newton", "single-step"], default="gauss-newton",
                        help="Select PyEIDORS' iterative GN solver or an EIDORS-style single-step solve.")
    parser.add_argument("--difference-max-iterations", type=int, default=None,
                        help="Override the Gauss–Newton iteration count when using the iterative solver.")
    parser.add_argument("--difference-hyperparameter", type=float, default=1e-2,
                        help="Spatial hyperparameter (hp) applied as hp^2 * RtR in single-step solves.")
    parser.add_argument("--noser-exponent", type=float, default=0.5,
                        help="Exponent applied to diag(J'WJ) when forming the NOSER prior (default 0.5 like EIDORS).")
    parser.add_argument("--noser-floor", type=float, default=1e-12,
                        help="Numerical floor for NOSER diagonal entries before applying the exponent.")
    parser.add_argument("--meas-weight-strategy", choices=["baseline", "difference", "none"], default="baseline",
                        help="Strategy for estimating measurement inverse covariance weights.")
    parser.add_argument("--meas-weight-floor", type=float, default=1e-9,
                        help="Minimum allowable measurement weight to avoid singular systems.")
    parser.add_argument("--single-step-jacobian-method", choices=["efficient", "traditional"], default="efficient",
                        help="Jacobian calculator mode for the single-step solver.")
    parser.add_argument("--single-step-negate-jacobian", action="store_true",
                        help="Match PyEIDORS' sign convention by negating the Jacobian before solving.")
    parser.add_argument("--conductivity-bounds", type=float, nargs=2, metavar=("MIN", "MAX"),
                        default=(1e-6, 10.0),
                        help="Bounds enforced on reconstructed conductivities in single-step mode.")
    parser.add_argument("--step-size-calibration", action="store_true",
                        help="Enable 1-D scale search (fminbnd-style) on the single-step update.")
    parser.add_argument("--step-size-min", type=float, default=1e-5,
                        help="Lower bound for the step-size calibration interval.")
    parser.add_argument("--step-size-max", type=float, default=1e1,
                        help="Upper bound for the step-size calibration interval.")
    parser.add_argument("--step-size-maxiter", type=int, default=50,
                        help="Maximum iterations passed to the bounded step-size optimizer.")
    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_pattern_config(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        rotate_meas=True,
    )


def setup_eit_system(args: argparse.Namespace) -> Tuple[EITSystem, object]:
    pattern = build_pattern_config(args.n_elec)
    eit_system = EITSystem(n_elec=args.n_elec, pattern_config=pattern)
    mesh = load_or_create_mesh(mesh_dir=str(args.mesh_dir), mesh_name=args.mesh_name,
                               n_elec=args.n_elec, refinement=args.refinement,
                               radius=args.mesh_radius, electrode_coverage=args.electrode_coverage)
    eit_system.setup(mesh=mesh)
    return eit_system, mesh


def make_phantom_image(system: EITSystem, args: argparse.Namespace) -> EITImage:
    anomalies = [
        {
            "center": tuple(args.phantom_center),
            "radius": args.phantom_radius,
            "conductivity": args.phantom_contrast,
        }
    ]
    sigma = create_custom_phantom(system.fwd_model,
                                  background_conductivity=args.background,
                                  anomalies=anomalies)
    return EITImage(elem_data=sigma.vector()[:], fwd_model=system.fwd_model)


def compute_metrics(measured: np.ndarray, predicted: np.ndarray) -> Metrics:
    measured = measured.reshape(-1)
    predicted = predicted.reshape(-1)
    error = predicted - measured
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    max_abs = float(np.max(np.abs(error)))
    # Avoid division by zero when computing relative error
    safe_measured = measured.copy()
    safe_measured[np.abs(safe_measured) < np.finfo(float).eps] = np.finfo(float).eps
    relative = np.abs(error / safe_measured) * 100.0
    rel_mean = float(np.mean(relative))
    if np.std(measured) < np.finfo(float).eps or np.std(predicted) < np.finfo(float).eps:
        corr = None
    else:
        corr = float(np.corrcoef(predicted, measured)[0, 1])
    return Metrics(rmse, mae, max_abs, rel_mean, corr)

def compute_scale_bias(measured: np.ndarray, model: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(model, dtype=float)
    y = np.asarray(measured, dtype=float)
    denom = float(np.dot(x, x))
    if denom < 1e-18:
        return 1.0, float(y.mean())
    scale = float(np.dot(y, x) / denom)
    if abs(scale) < 1e-12:
        scale = 1.0 if scale >= 0 else -1.0
    bias = float(y.mean() - scale * x.mean())
    return scale, bias


def apply_calibration(vector: np.ndarray, scale: float, bias: float) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    safe_scale = abs(scale)
    if safe_scale < np.finfo(float).eps:
        safe_scale = 1.0
    return (arr - bias) / safe_scale


def simulate_calibrated_difference(system: EITSystem,
                                   image: EITImage,
                                   scale: float,
                                   bias: float,
                                   baseline_calibrated: np.ndarray) -> np.ndarray:
    sim_recon, _ = system.fwd_model.fwd_solve(image)
    calibrated = apply_calibration(sim_recon.meas, scale, bias)
    return calibrated - baseline_calibrated


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


def solve_single_step_delta(system: EITSystem,
                            baseline_image: EITImage,
                            raw_diff: np.ndarray,
                            raw_baseline: np.ndarray,
                            args: argparse.Namespace) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    jacobian_calculator = getattr(getattr(system, "reconstructor", None),
                                  "jacobian_calculator", None)
    if jacobian_calculator is None:
        jacobian_calculator = DirectJacobianCalculator(system.fwd_model)

    sigma_fn = Function(system.fwd_model.V_sigma)
    sigma_fn.vector()[:] = baseline_image.elem_data
    jacobian = jacobian_calculator.calculate(sigma_fn, method=args.single_step_jacobian_method)
    negate = args.single_step_negate_jacobian
    recon_negate = getattr(getattr(system, "reconstructor", None), "negate_jacobian", False)
    negate = negate or recon_negate
    if negate:
        jacobian = -jacobian

    weights = build_measurement_weights(raw_baseline, raw_diff, args.meas_weight_strategy,
                                        args.meas_weight_floor)
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
    RtR = np.diag(noser_diag)

    hp = max(args.difference_hyperparameter, 0.0)
    lhs = jacobian_weighted.T @ jacobian_weighted + (hp ** 2) * RtR
    rhs = jacobian_weighted.T @ dv_weighted

    try:
        delta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        delta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    return delta, weights


def build_single_step_image(system: EITSystem,
                            baseline_image: EITImage,
                            delta: np.ndarray,
                            step_size: float,
                            bounds: Tuple[float, float]) -> EITImage:
    min_cond, max_cond = bounds
    elem = baseline_image.elem_data + step_size * delta
    elem = np.clip(elem, min_cond, max_cond)
    return EITImage(elem_data=elem, fwd_model=system.fwd_model)


def optimize_step_size(system: EITSystem,
                       baseline_image: EITImage,
                       delta: np.ndarray,
                       target_diff: np.ndarray,
                       baseline_calibrated: np.ndarray,
                       scale: float,
                       bias: float,
                       args: argparse.Namespace) -> float:
    bounds = (args.step_size_min, args.step_size_max)

    def objective(step: float) -> float:
        test_img = build_single_step_image(system, baseline_image, delta, step, args.conductivity_bounds)
        pred = simulate_calibrated_difference(system, test_img, scale, bias, baseline_calibrated)
        residual = pred - target_diff
        return float(np.mean(residual ** 2))

    result = minimize_scalar(objective, bounds=bounds, method="bounded",
                             options={"maxiter": args.step_size_maxiter})
    if result.success:
        return float(result.x)
    return 1.0


def run_single_step_difference(system: EITSystem,
                               baseline_image: EITImage,
                               raw_diff: np.ndarray,
                               raw_baseline: np.ndarray,
                               target_diff: np.ndarray,
                               baseline_calibrated: np.ndarray,
                               scale: float,
                               bias: float,
                               args: argparse.Namespace) -> Tuple[EITImage, np.ndarray, Dict[str, object]]:
    delta, weights = solve_single_step_delta(system, baseline_image, raw_diff, raw_baseline, args)
    step_size = 1.0
    if args.step_size_calibration:
        step_size = optimize_step_size(system, baseline_image, delta, target_diff,
                                       baseline_calibrated, scale, bias, args)

    recon_image = build_single_step_image(system, baseline_image, delta, step_size, args.conductivity_bounds)
    predicted_diff = simulate_calibrated_difference(system, recon_image, scale, bias, baseline_calibrated)

    metadata: Dict[str, object] = {
        "step_size": step_size,
        "jacobian_method": args.single_step_jacobian_method,
        "measurement_weight_strategy": args.meas_weight_strategy,
        "hyperparameter": args.difference_hyperparameter,
        "noser_exponent": args.noser_exponent,
        "conductivity_bounds": args.conductivity_bounds,
        "delta_norm": float(np.linalg.norm(delta)),
    }
    if weights is not None:
        metadata["weight_stats"] = {
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "median": float(np.median(weights)),
        }

    return recon_image, predicted_diff, metadata


def save_conductivity_figures(system: EITSystem,
                              ground_truth: EITImage,
                              reconstruction: EITImage,
                              output_dir: Path,
                              dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = create_visualizer()
    fig_gt = visualizer.plot_conductivity(system.mesh, ground_truth.elem_data,
                                          title=None, minimal=True, show_electrodes=True)
    fig_rec = visualizer.plot_conductivity(system.mesh, reconstruction.elem_data,
                                           title=None, minimal=True, show_electrodes=True)
    fig_gt.savefig(output_dir / "phantom_ground_truth.png", dpi=dpi)
    fig_rec.savefig(output_dir / "reconstruction.png", dpi=dpi)


def save_voltage_comparison(measured: np.ndarray,
                            predicted: np.ndarray,
                            output_path: Path,
                            title: str,
                            dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.arange(measured.size)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(idx, measured, label="Measured", linewidth=1.2)
    ax.plot(idx, predicted, label="Predicted", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Voltage")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_metrics(metrics: Dict[str, Metrics], output_path: Path,
                 extra: Optional[Dict[str, object]] = None) -> None:
    payload: Dict[str, object] = {mode: m.to_dict() for mode, m in metrics.items()}
    if extra:
        payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2))


def load_eidors_vector(path: Path, delimiter: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=delimiter)
    return data.reshape(-1)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_root)

    system, _ = setup_eit_system(args)
    reconstructor = getattr(system, "reconstructor", None)
    if reconstructor is not None:
        reconstructor.measurement_weight_strategy = args.meas_weight_strategy
        reconstructor.use_measurement_weights = args.meas_weight_strategy != "none"
        reconstructor.weight_floor = args.meas_weight_floor
        if args.difference_max_iterations is not None:
            reconstructor.max_iterations = args.difference_max_iterations

    homogeneous = system.create_homogeneous_image(conductivity=args.background)
    phantom_img = make_phantom_image(system, args)

    # Forward simulations
    meas_homogeneous, _ = system.fwd_model.fwd_solve(homogeneous)
    meas_phantom, _ = system.fwd_model.fwd_solve(phantom_img)

    # Emulate measurement calibration (scale/bias) to stay consistent with real-data pipeline.
    scale, bias = compute_scale_bias(meas_homogeneous.meas, meas_homogeneous.meas)
    meas_h_calibrated = apply_calibration(meas_homogeneous.meas, scale, bias)
    meas_p_calibrated = apply_calibration(meas_phantom.meas, scale, bias)
    diff_vector = meas_p_calibrated - meas_h_calibrated
    raw_diff_vector = meas_phantom.meas - meas_homogeneous.meas

    if args.save_forward_csv:
        header = "meas_homogeneous,meas_phantom,difference"
        forward_matrix = np.column_stack([meas_homogeneous.meas,
                                          meas_phantom.meas,
                                          diff_vector])
        np.savetxt(output_dir / "synthetic_forward_data.csv",
                   forward_matrix, delimiter=",", header=header, comments="")

    metrics: Dict[str, Metrics] = {}
    extra_payload: Dict[str, object] = {
        "n_measurements": int(diff_vector.size),
        "metadata": {
            "n_elec": args.n_elec,
            "phantom": {
                "center": args.phantom_center,
                "radius": args.phantom_radius,
                "contrast": args.phantom_contrast,
            },
        },
        "calibration": {"scale": scale, "bias": bias},
        "difference_solver": args.difference_solver,
    }

    if args.mode in {"difference", "both"}:
        diff_meta: Dict[str, object] = {"solver": args.difference_solver}
        if args.difference_solver == "single-step":
            recon_image, predicted_diff, solver_meta = run_single_step_difference(
                system=system,
                baseline_image=homogeneous,
                raw_diff=raw_diff_vector,
                raw_baseline=meas_homogeneous.meas,
                target_diff=diff_vector,
                baseline_calibrated=meas_h_calibrated,
                scale=scale,
                bias=bias,
                args=args,
            )
            diff_meta.update(solver_meta)
        else:
            result = system.difference_reconstruct(meas_phantom, meas_homogeneous)
            recon_image = result.conductivity_image
            predicted_diff = simulate_calibrated_difference(
                system, recon_image, scale, bias, meas_h_calibrated
            )
            history = result.residual_history or []
            diff_meta.update({
                "iterations": len(history),
                "final_residual": history[-1] if history else None,
            })

        metrics["difference"] = compute_metrics(diff_vector, predicted_diff)
        diff_meta["rmse"] = metrics["difference"].rmse
        save_conductivity_figures(system, phantom_img, recon_image,
                                  output_dir / "difference", args.figure_dpi)
        np.savetxt(output_dir / "difference" / "predicted_difference.csv",
                   predicted_diff, delimiter=",")
        save_voltage_comparison(
            measured=diff_vector,
            predicted=predicted_diff,
            output_path=output_dir / "difference" / "voltage_comparison.png",
            title="Difference reconstruction voltages",
            dpi=args.figure_dpi,
        )
        extra_payload["difference_metadata"] = diff_meta

    if args.mode in {"absolute", "both"}:
        abs_result = system.absolute_reconstruct(meas_phantom, baseline_image=homogeneous)
        metrics["absolute"] = compute_metrics(meas_phantom.meas, abs_result.simulated)
        recon_image = abs_result.conductivity_image
        save_conductivity_figures(system, phantom_img, recon_image,
                                  output_dir / "absolute", args.figure_dpi)
        np.savetxt(output_dir / "absolute" / "predicted_absolute.csv",
                   abs_result.simulated, delimiter=",")
        save_voltage_comparison(
            measured=meas_phantom.meas,
            predicted=abs_result.simulated,
            output_path=output_dir / "absolute" / "voltage_comparison.png",
            title="Absolute reconstruction voltages",
            dpi=args.figure_dpi,
        )

    if args.eidors_csv:
        reference_vector = load_eidors_vector(args.eidors_csv, args.csv_delimiter)
        key = "eidors_reference"
        extra_payload[key] = compute_metrics(diff_vector, reference_vector).to_dict()

    save_metrics(metrics, output_dir / "metrics.json", extra=extra_payload)
    print(f"Synthetic parity results stored in {output_dir}")


if __name__ == "__main__":
    main()
