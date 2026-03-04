"""Sparse Bayesian reconstruction utilities for unified CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pyeidors.core_system import EITSystem
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.data.structures import EITData, PatternConfig
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
# Re-export solver classes for the unified CLI runner module.
from pyeidors.inverse import (
    SparseBayesianConfig,
    SparseBayesianReconstructor,
    perform_sparse_difference_reconstruction,
)
from pyeidors.visualization import create_visualizer

from .calibration import compute_scale_bias
from .io_utils import load_metadata as _load_metadata

LOGGER = logging.getLogger("sparse_bayes_runner")

DEFAULT_METADATA = {
    "n_elec": 16,
    "stim_pattern": "{ad}",
    "meas_pattern": "{ad}",
    "drive_mode": "line_current_density",
    "drive_value": 1.0e-4,
    "geometry_scale_to_m": 1.0,
    "use_meas_current": False,
    "use_meas_current_next": 0,
    "rotate_meas": True,
}


def load_metadata(path: Path) -> Dict[str, Any]:
    metadata = _load_metadata(path)
    if not isinstance(metadata, dict):
        raise TypeError("Metadata file must define a mapping.")

    merged = dict(DEFAULT_METADATA)
    merged.update(metadata)
    if "amplitude" in metadata:
        raise ValueError(
            "metadata field 'amplitude' is no longer supported. "
            "Use 'drive_mode' and 'drive_value' instead."
        )
    return merged


def select_frames(raw: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if len(indices) == 1:
        return raw[:, indices[0]]
    return raw[:, list(indices)].T


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def measurement_to_dataset(
    measurements: np.ndarray, metadata: Dict[str, Any], data_type: str = "real"
) -> MeasurementDataset:
    return MeasurementDataset.from_metadata(
        measurements=measurements,
        metadata=metadata,
        data_type=data_type,
    )


def calibrate_measurements(
    dataset: MeasurementDataset,
    baseline_vector: np.ndarray,
    frame_index: int,
) -> Dict[str, float]:
    ref_vector = dataset.measurements[frame_index].copy()
    scale, bias = compute_scale_bias(ref_vector, baseline_vector)
    LOGGER.info("Calibration parameters: scale=%.3e, bias=%.3e", scale, bias)
    if abs(scale) < 1e-18:
        scale = 1e-18 if scale >= 0 else -1e-18
    dataset.measurements = (dataset.measurements - bias) / scale
    return {"scale": scale, "bias": bias}


def clone_eit_data(
    data: EITData,
    new_meas: np.ndarray,
    data_type: Optional[str] = None,
) -> EITData:
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
    diff_vector = target_data.meas - reference_data.meas
    scale, bias = compute_scale_bias(diff_vector, baseline_vector)
    LOGGER.info("Post-difference calibration: scale=%.3e, bias=%.3e", scale, bias)

    if abs(scale) < 1e-18:
        scale = 1.0 if scale >= 0 else -1.0
    calibrated_diff = (diff_vector - bias) / scale
    adjusted_target = reference_data.meas + calibrated_diff

    return clone_eit_data(target_data, adjusted_target), {"diff_scale": scale, "diff_bias": bias}


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
) -> Dict[str, Any]:
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
            metadata.update(
                {
                    "pre_scale": pre_calibration.get("scale"),
                    "pre_bias": pre_calibration.get("bias"),
                }
            )

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


def summarise_result(result) -> Dict[str, Any]:
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
    bias_for_plot = 0.0

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
                fh.write(
                    f"likelihood_noise_std={result.metadata['likelihood_noise_std']}\n"
                )
            if "prior_scale" in result.metadata:
                fh.write(f"prior_scale={result.metadata['prior_scale']}\n")
