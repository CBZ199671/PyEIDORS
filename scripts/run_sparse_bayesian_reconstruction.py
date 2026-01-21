#!/usr/bin/env python3
"""Run sparse Bayesian EIT reconstructions and compare with Gauss-Newton results."""

from __future__ import annotations

import argparse
import json
import logging
from ast import literal_eval
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("This script requires PyYAML. Install via: pip install pyyaml") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SRC_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPTS_PATH))

from pyeidors.core_system import EITSystem
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.data.structures import PatternConfig, EITData
from pyeidors.inverse import (
    perform_sparse_absolute_reconstruction,
    perform_sparse_difference_reconstruction,
    SparseBayesianConfig,
    SparseBayesianReconstructor,
)
from pyeidors.visualization import create_visualizer
from scripts.common.io_utils import align_frames_polarity  # type: ignore

LOGGER = logging.getLogger("sparse_bayes_reconstruction")

DEFAULT_METADATA = {
    "n_elec": 16,
    "stim_pattern": "{ad}",
    "meas_pattern": "{ad}",
    "amplitude": 1.0e-4,
    "use_meas_current": False,
    "use_meas_current_next": 0,
    "rotate_meas": True,
}


def compute_scale_bias(measured: np.ndarray, model: np.ndarray) -> Tuple[float, float]:
    """Linear calibration parameters aligning model response to measured data."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse Bayesian EIT reconstruction (absolute & difference)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, required=True, help="CSV measurement file")
    parser.add_argument("--metadata", type=Path, help="Optional metadata YAML/JSON file")
    parser.add_argument(
        "--mode",
        choices=["absolute", "difference", "both"],
        default="both",
        help="Reconstruction mode(s) to run",
    )
    parser.add_argument(
        "--absolute-col",
        type=int,
        default=2,
        help="Column index used for absolute reconstruction",
    )
    parser.add_argument(
        "--reference-col",
        type=int,
        default=0,
        help="Column index representing the reference frame (difference mode)",
    )
    parser.add_argument(
        "--target-col",
        type=int,
        default=2,
        help="Column index representing the target frame (difference mode)",
    )
    parser.add_argument(
        "--calibration-col",
        type=int,
        default=-1,
        help="Column index used for linear calibration. "
             "Defaults to absolute column or reference column depending on mode.",
    )
    parser.add_argument(
        "--difference-calibration",
        choices=["before", "after", "none"],
        default="after",
        help="When running difference mode, choose calibration order: "
             "'after' (default, calibrate post-subtraction), 'before' (legacy flow), "
             "or 'none' to skip calibration.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "sparse_bayesian",
        help="Output directory root",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=REPO_ROOT / "eit_meshes",
        help="Directory containing (or receiving) FEniCS meshes",
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        help="Optional mesh name (without extension) to load",
    )
    parser.add_argument(
        "--refinement",
        type=int,
        default=12,
        help="Mesh refinement level used when creating/loading meshes",
    )
    parser.add_argument(
        "--mesh-radius",
        type=float,
        default=1.0,
        help="Mesh radius (participates in cache key)",
    )
    parser.add_argument(
        "--electrode-coverage",
        type=float,
        default=0.5,
        help="Electrode coverage fraction (participates in cache key)",
    )
    parser.add_argument(
        "--contact-impedance",
        type=float,
        default=1e-5,
        help="Per-electrode contact impedance (Ω·m²); default matches common EIDORS setups",
    )
    parser.add_argument(
        "--prior-scale",
        type=float,
        help="Override Smoothed Laplace prior scale parameter",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        help="Override Gaussian likelihood noise standard deviation",
    )
    parser.add_argument(
        "--subspace-rank",
        type=int,
        help="Use truncated SVD subspace of this rank for the Bayesian solve",
    )
    parser.add_argument(
        "--coarse-group-size",
        type=int,
        help="Aggregate conductivity elements into groups of this size before inversion",
    )
    parser.add_argument(
        "--coarse-levels",
        type=int,
        nargs="+",
        help="Specify additional coarse levels (group sizes) applied from coarse to fine",
    )
    parser.add_argument(
        "--linear-warm-start",
        action="store_true",
        help="Initialise MAP optimisation with the linear least-squares solution (when using subspace)",
    )
    parser.add_argument(
        "--solver",
        default="map",
        choices=["map", "fista", "irls"],
        help="Sparse solver to use for the Bayesian update (fista/irls avoid CUQI MAP)",
    )
    parser.add_argument(
        "--linear-max-iters",
        type=int,
        default=200,
        help="Maximum iterations for the linearised solvers (FISTA/IRLS)",
    )
    parser.add_argument(
        "--linear-tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for the linearised solvers",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Run the linearised solvers (FISTA/IRLS) on GPU when available",
    )
    parser.add_argument(
        "--gpu-dtype",
        type=str,
        default="float32",
        help="GPU tensor data type: float16/float32/float64",
    )
    parser.add_argument(
        "--block-iterations",
        type=int,
        default=0,
        help="Number of block Gauss-Seidel refinement passes (0 to disable)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        help="Block size used during local block refinement",
    )
    parser.add_argument(
        "--refinement-gradient-tol",
        type=float,
        default=1e-5,
        help="Gradient norm threshold used to skip coarse/block refinement updates",
    )
    parser.add_argument(
        "--coarse-iterations",
        type=int,
        default=0,
        help="Number of coarse-to-fine correction sweeps (0 to disable)",
    )
    parser.add_argument(
        "--coarse-relaxation",
        type=float,
        default=1.0,
        help="Relaxation factor applied to coarse-level corrections (1.0 keeps full update)",
    )
    parser.add_argument(
        "--jacobian-cache",
        action="store_true",
        help="Enable Jacobian caching between reconstructions",
    )
    parser.add_argument(
        "--gn-absolute-dir",
        type=Path,
        help="Path to Gauss-Newton absolute result directory for comparison",
    )
    parser.add_argument(
        "--gn-difference-dir",
        type=Path,
        help="Path to Gauss-Newton difference result directory for comparison",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")


def load_metadata(path: Optional[Path]) -> Dict[str, any]:
    if path is None:
        LOGGER.info("Metadata not provided; using default hardware configuration.")
        return dict(DEFAULT_METADATA)

    LOGGER.info("Loading metadata from %s", path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            metadata = yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            metadata = json.load(handle)
        else:
            raise ValueError("Metadata file must be YAML or JSON.")
    if not isinstance(metadata, dict):
        raise TypeError("Metadata file must define a mapping.")

    merged = dict(DEFAULT_METADATA)
    merged.update(metadata)
    return merged


def select_frames(raw: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Extract frames based on column indices."""
    if len(indices) == 1:
        return raw[:, indices[0]]
    return raw[:, list(indices)].T


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def measurement_to_dataset(
    measurements: np.ndarray, metadata: Dict[str, any], data_type: str = "real"
) -> MeasurementDataset:
    return MeasurementDataset.from_metadata(
        measurements=measurements,
        metadata=metadata,
        data_type=data_type,
    )


def calibrate_measurements(dataset: MeasurementDataset, baseline_vector: np.ndarray, frame_index: int) -> Dict[str, float]:
    """Apply linear calibration to align dataset with baseline model measurements."""
    ref_vector = dataset.measurements[frame_index].copy()
    scale, bias = compute_scale_bias(ref_vector, baseline_vector)
    LOGGER.info("Calibration parameters: scale=%.3e, bias=%.3e", scale, bias)
    if abs(scale) < 1e-18:
        scale = 1e-18 if scale >= 0 else -1e-18
    dataset.measurements = (dataset.measurements - bias) / scale
    return {"scale": scale, "bias": bias}


def clone_eit_data(data: EITData, new_meas: np.ndarray, data_type: Optional[str] = None) -> EITData:
    """Return a shallow copy of EITData with updated measurements."""
    return EITData(
        meas=new_meas.copy(),
        stim_pattern=data.stim_pattern.copy(),
        n_elec=data.n_elec,
        n_stim=data.n_stim,
        n_meas=data.n_meas,
        type=data.type if data_type is None else data_type,
    )


def calibrate_difference_after_subtraction(
    reference_data: EITData,
    target_data: EITData,
    baseline_vector: np.ndarray,
) -> Tuple[EITData, Dict[str, float]]:
    """Calibrate difference measurements after subtraction to align with baseline scale."""

    diff_vector = target_data.meas - reference_data.meas
    scale, bias = compute_scale_bias(diff_vector, baseline_vector)
    LOGGER.info("Post-difference calibration: scale=%.3e, bias=%.3e", scale, bias)

    if abs(scale) < 1e-18:
        scale = 1.0 if scale >= 0 else -1.0
    calibrated_diff = (diff_vector - bias) / scale
    adjusted_target = reference_data.meas + calibrated_diff

    return clone_eit_data(target_data, adjusted_target), {"diff_scale": scale, "diff_bias": bias}


def run_absolute_pipeline(
    eit_system: EITSystem,
    dataset: MeasurementDataset,
    baseline_image,
    output_dir: Path,
    reconstructor: SparseBayesianReconstructor,
    prior_scale: Optional[float],
    noise_std: Optional[float],
    calibration_info: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    measurement_data = dataset.to_eit_data(frame_index=0, data_type="real")
    metadata: Dict[str, Any] = {"target_idx": 0}
    if calibration_info:
        metadata.update(
            {
                "calibration_scale": calibration_info.get("scale"),
                "calibration_bias": calibration_info.get("bias"),
            }
        )

    result = perform_sparse_absolute_reconstruction(
        eit_system=eit_system,
        measurement_data=measurement_data,
        baseline_image=baseline_image,
        reconstructor=reconstructor,
        noise_std=noise_std,
        prior_scale=prior_scale,
        metadata=metadata,
    )

    save_result_outputs(result, output_dir, mode="absolute")
    return {
        "result": result,
        "summary": summarise_result(result),
    }


def run_difference_pipeline(
    eit_system: EITSystem,
    dataset: MeasurementDataset,
    baseline_image,
    output_dir: Path,
    reconstructor: SparseBayesianReconstructor,
    prior_scale: Optional[float],
    noise_std: Optional[float],
    baseline_vector: np.ndarray,
    calibration_mode: str,
    pre_calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, any]:
    reference_data = dataset.to_eit_data(frame_index=0, data_type="reference")
    target_data = dataset.to_eit_data(frame_index=1, data_type="measurement")
    metadata: Dict[str, Any] = {
        "target_idx": 1,
        "reference_idx": 0,
        "difference_calibration": calibration_mode,
    }

    if calibration_mode == "after":
        target_data, diff_cal = calibrate_difference_after_subtraction(
            reference_data,
            target_data,
            baseline_vector,
        )
        metadata.update(diff_cal)
    else:
        metadata.setdefault("diff_scale", 1.0)
        metadata.setdefault("diff_bias", 0.0)
        if calibration_mode == "before" and pre_calibration:
            metadata.update({
                "pre_scale": pre_calibration.get("scale"),
                "pre_bias": pre_calibration.get("bias"),
            })

    result = perform_sparse_difference_reconstruction(
        eit_system=eit_system,
        measurement_data=target_data,
        reference_data=reference_data,
        baseline_image=baseline_image,
        reconstructor=reconstructor,
        noise_std=noise_std,
        prior_scale=prior_scale,
        metadata=metadata,
    )

    save_result_outputs(result, output_dir, mode="difference")
    return {
        "result": result,
        "summary": summarise_result(result),
    }


def summarise_result(result) -> Dict[str, any]:
    summary = {
        "mode": result.mode,
        "l2_error": result.l2_error,
        "relative_error": result.relative_error,
        "mse": result.mse,
    }
    if result.metadata:
        summary.update(
            {
                "prior_scale": result.metadata.get("prior_scale"),
                "noise_std": result.metadata.get("likelihood_noise_std"),
            }
        )
    return summary


def save_result_outputs(result, output_dir: Path, mode: str) -> None:
    ensure_output_dir(output_dir)
    visualizer = create_visualizer()
    mesh = result.conductivity_image.fwd_model.mesh

    metadata = result.metadata or {}

    measured_plot = result.measured.copy()
    simulated_plot = result.simulated.copy()
    residual_plot = result.residual.copy()
    scale_for_plot: Optional[float] = None
    bias_for_plot: float = 0.0

    if "calibration_scale" in metadata and metadata["calibration_scale"] is not None:
        scale_for_plot = float(metadata["calibration_scale"])
        bias_for_plot = float(metadata.get("calibration_bias", 0.0))
    elif metadata.get("difference_calibration") == "after" and metadata.get("diff_scale") is not None:
        scale_for_plot = float(metadata.get("diff_scale"))
        bias_for_plot = float(metadata.get("diff_bias", 0.0))
    elif metadata.get("difference_calibration") == "before" and metadata.get("pre_scale") is not None:
        scale_for_plot = float(metadata.get("pre_scale"))
        bias_for_plot = float(metadata.get("pre_bias", 0.0))

    if scale_for_plot is not None and abs(scale_for_plot) > 1e-18:
        measured_plot = measured_plot * scale_for_plot + bias_for_plot
        simulated_plot = simulated_plot * scale_for_plot + bias_for_plot
        residual_plot = simulated_plot - measured_plot

        np.savetxt(output_dir / "measured_physical_vector.txt", measured_plot)
        np.savetxt(output_dir / "predicted_physical_vector.txt", simulated_plot)
        np.savetxt(output_dir / "residual_physical_vector.txt", residual_plot)

    fig_cond = visualizer.plot_conductivity(
        mesh,
        result.metadata.get("display_values", result.conductivity),
        title=None,
        save_path=str(output_dir / "reconstruction.png"),
        minimal=True,
    )
    plt.close(fig_cond)

    indices = np.arange(len(result.measured))
    fig, ax = plt.subplots(figsize=visualizer.figsize)
    ax.plot(indices, measured_plot, label="Measured", linewidth=1.5)
    ax.plot(indices, simulated_plot, label="Predicted", linewidth=1.5, alpha=0.85)
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{mode.capitalize()} measurements comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "measurements_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig_res = visualizer.plot_measurements(
        residual_plot,
        title=f"{mode.capitalize()} residual (Sparse Bayesian)",
        save_path=str(output_dir / "measurements_residual.png"),
    )
    plt.close(fig_res)

    np.savetxt(output_dir / "residual_vector.txt", result.residual)
    np.savetxt(output_dir / "measured_vector.txt", result.measured)
    np.savetxt(output_dir / "predicted_vector.txt", result.simulated)

    summary_path = output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write(f"mode={result.mode}\n")
        fh.write(f"L2={result.l2_error}\n")
        fh.write(f"relative_error={result.relative_error}\n")
        fh.write(f"mse={result.mse}\n")
        if result.metadata:
            if "likelihood_noise_std" in result.metadata:
                fh.write(f"likelihood_noise_std={result.metadata['likelihood_noise_std']}\n")
            if "prior_scale" in result.metadata:
                fh.write(f"prior_scale={result.metadata['prior_scale']}\n")


def parse_gn_summary(summary_file: Path) -> Optional[Dict[str, any]]:
    if summary_file is None or not summary_file.exists():
        return None
    data: Dict[str, any] = {}
    for line in summary_file.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            data[key] = literal_eval(value)
        except (ValueError, SyntaxError):
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
    return data


def compare_with_gn(
    sparse_summary: Dict[str, any],
    gn_summary: Optional[Dict[str, any]],
    output_dir: Path,
    tag: str,
) -> None:
    if not gn_summary:
        return

    comparison: Dict[str, Dict[str, float]] = {}
    for metric in ("L2", "relative_error", "mse", "L2_error", "relative_error", "mse"):
        sparse_value = sparse_summary.get(metric) or sparse_summary.get(metric.lower())
        gn_value = gn_summary.get(metric) or gn_summary.get(metric.lower())
        if sparse_value is None or gn_value is None:
            continue
        comparison[metric.lower()] = {
            "gauss_newton": float(gn_value),
            "sparse_bayesian": float(sparse_value),
            "delta": float(sparse_value) - float(gn_value),
            "ratio": float(sparse_value) / float(gn_value) if gn_value else np.inf,
        }

    if not comparison:
        return

    compare_path = output_dir / f"comparison_vs_gn_{tag}.json"
    with compare_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    raw_measurements = np.loadtxt(args.csv, delimiter=",", dtype=float)
    if raw_measurements.ndim == 1:
        raw_measurements = raw_measurements[:, np.newaxis]

    metadata = load_metadata(args.metadata)
    LOGGER.info("Dataset summary: n_meas=%d, n_cols=%d", raw_measurements.shape[0], raw_measurements.shape[1])

    mode_absolute = args.mode in {"absolute", "both"}
    mode_difference = args.mode in {"difference", "both"}

    pattern_config = PatternConfig(
        n_elec=int(metadata["n_elec"]),
        stim_pattern=metadata.get("stim_pattern", "{ad}"),
        meas_pattern=metadata.get("meas_pattern", "{ad}"),
        amplitude=float(metadata.get("amplitude", 1.0)),
        use_meas_current=bool(metadata.get("use_meas_current", False)),
        use_meas_current_next=int(metadata.get("use_meas_current_next", 0)),
        rotate_meas=bool(metadata.get("rotate_meas", True)),
    )

    LOGGER.info("Initialising EITSystem with %d electrodes", pattern_config.n_elec)
    contact_impedance = np.full(pattern_config.n_elec, float(args.contact_impedance))
    eit_system = EITSystem(
        n_elec=pattern_config.n_elec,
        pattern_config=pattern_config,
        contact_impedance=contact_impedance,
    )

    mesh = load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        n_elec=pattern_config.n_elec,
        refinement=max(args.refinement, 4),
        mesh_name=args.mesh_name,
        radius=args.mesh_radius,
        electrode_coverage=args.electrode_coverage,
    )
    eit_system.setup(mesh=mesh)

    baseline_image = eit_system.create_homogeneous_image()
    baseline_data = eit_system.forward_solve(baseline_image)
    baseline_vector = baseline_data.meas

    # Polarity correction: align columns (absolute, reference, target, calibration) to normal U-shape
    cols_to_align = set()
    if mode_absolute:
        cols_to_align.add(args.absolute_col)
    if mode_difference:
        cols_to_align.update({args.reference_col, args.target_col})
    calib_col = (
        args.calibration_col
        if args.calibration_col >= 0
        else (args.absolute_col if mode_absolute else args.reference_col)
    )
    cols_to_align.add(calib_col)

    flipped_cols: list[int] = []
    if cols_to_align:
        col_list = list(cols_to_align)
        selected = np.vstack([raw_measurements[:, c] for c in col_list])
        aligned, flipped_idx = align_frames_polarity(selected, baseline_vector)
        for i, c in enumerate(col_list):
            raw_measurements[:, c] = aligned[i]
        flipped_cols = [col_list[i] for i in flipped_idx]
        if flipped_cols:
            LOGGER.info("Polarity alignment: flipped columns %s", flipped_cols)

    config = SparseBayesianConfig(
        cache_jacobian=args.jacobian_cache,
        subspace_rank=args.subspace_rank,
        use_linear_warm_start=args.linear_warm_start,
        solver=args.solver,
        linear_max_iterations=args.linear_max_iters,
        linear_tolerance=args.linear_tol,
        coarse_group_size=args.coarse_group_size,
        use_gpu=args.use_gpu,
        gpu_dtype=args.gpu_dtype,
        coarse_levels=tuple(args.coarse_levels) if args.coarse_levels else None,
        block_iterations=args.block_iterations,
        block_size=args.block_size,
        refinement_gradient_tol=args.refinement_gradient_tol,
        coarse_iterations=args.coarse_iterations,
        coarse_relaxation=args.coarse_relaxation,
    )
    reconstructor = SparseBayesianReconstructor(
        eit_system=eit_system,
        config=config,
    )

    summaries = {}

    if mode_absolute:
        LOGGER.info("Running sparse Bayesian absolute reconstruction")
        abs_indices = [args.absolute_col]
        abs_measurements = select_frames(raw_measurements, abs_indices)
        abs_dataset = measurement_to_dataset(abs_measurements, dict(metadata))
        abs_calibration = calibrate_measurements(abs_dataset, baseline_data.meas, frame_index=0)

        abs_output_dir = ensure_output_dir(args.output_root / "absolute" / args.csv.stem)
        abs_info = run_absolute_pipeline(
            eit_system,
            abs_dataset,
            baseline_image,
            abs_output_dir,
            reconstructor,
            prior_scale=args.prior_scale,
            noise_std=args.noise_std,
            calibration_info=abs_calibration,
        )
        summaries["absolute"] = abs_info["summary"]
        if args.gn_absolute_dir:
            gn_summary = parse_gn_summary(Path(args.gn_absolute_dir) / "summary.txt")
            compare_with_gn(abs_info["summary"], gn_summary, abs_output_dir, tag="absolute")

    if mode_difference:
        LOGGER.info("Running sparse Bayesian difference reconstruction")
        diff_indices = [args.reference_col, args.target_col]
        diff_measurements = select_frames(raw_measurements, diff_indices)
        diff_dataset = measurement_to_dataset(diff_measurements, dict(metadata))
        calibration_frame = args.calibration_col if args.calibration_col >= 0 else 0
        if not 0 <= calibration_frame < diff_dataset.measurements.shape[0]:
            raise IndexError(
                f"calibration_col must be within [0, {diff_dataset.measurements.shape[0]-1}] "
                f"for selected frames; got {args.calibration_col}"
            )

        pre_calibration: Optional[Dict[str, float]] = None
        if args.difference_calibration == "before":
            pre_calibration = calibrate_measurements(
                diff_dataset,
                baseline_data.meas,
                frame_index=calibration_frame,
            )
        elif args.difference_calibration == "after":
            LOGGER.info("Skipping pre-difference calibration (after-diff mode)")
        else:
            LOGGER.info("Skipping calibration for difference mode")

        diff_output_dir = ensure_output_dir(args.output_root / "difference" / args.csv.stem)
        diff_info = run_difference_pipeline(
            eit_system,
            diff_dataset,
            baseline_image,
            diff_output_dir,
            reconstructor,
            prior_scale=args.prior_scale,
            noise_std=args.noise_std,
            baseline_vector=baseline_data.meas,
            calibration_mode=args.difference_calibration,
            pre_calibration=pre_calibration,
        )
        summaries["difference"] = diff_info["summary"]
        if args.gn_difference_dir:
            gn_summary = parse_gn_summary(Path(args.gn_difference_dir) / "summary.txt")
            compare_with_gn(diff_info["summary"], gn_summary, diff_output_dir, tag="difference")

    if summaries:
        report_path = ensure_output_dir(args.output_root / "reports") / f"{args.csv.stem}_summary.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(summaries, fh, indent=2, ensure_ascii=False)
        LOGGER.info("Saved summary report to %s", report_path)


if __name__ == "__main__":  # pragma: no cover
    main()
