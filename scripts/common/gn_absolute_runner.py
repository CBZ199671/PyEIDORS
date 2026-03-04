"""GN absolute reconstruction runner shared by unified CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem

from pyeidors.core_system import EITSystem
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.data.structures import EITData, EITImage
from pyeidors.femx import function_get_array
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.visualization import EITVisualizer

from .io_utils import align_measurement_polarity

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH_DIR = REPO_ROOT / "eit_meshes"


def _configure_reconstructor(
    system: EITSystem,
    *,
    lambda_: float,
    max_iter: int,
    background_sigma: float,
) -> None:
    recon = system.reconstructor
    if recon is None:
        raise RuntimeError("EITSystem reconstructor not initialized")

    recon.max_iterations = int(max_iter)
    recon.regularization_param = float(lambda_)
    recon.line_search_steps = 12
    recon.max_step = 1.0
    recon.min_step = 1e-6
    recon.convergence_tol = 1e-5
    recon.negate_jacobian = True
    recon.use_measurement_weights = True
    recon.measurement_weight_strategy = "scaled_baseline"
    recon.clip_values = (background_sigma * 0.1, background_sigma * 100)
    recon.min_iterations = 1
    recon.use_prior_term = True


def _build_dataset(measurement: np.ndarray, metadata: Dict[str, Any]) -> MeasurementDataset:
    measurements = np.asarray(measurement, dtype=float).reshape(1, -1)
    return MeasurementDataset.from_metadata(measurements, metadata)


def run_absolute_reconstruction(
    *,
    measurement: np.ndarray,
    metadata: Dict[str, Any],
    csv_path: Path,
    metadata_path: Path,
    col_idx: int,
    output_dir: Path,
    mesh_radius: float,
    refinement: int,
    measurement_gain: float,
    background_sigma: float,
    lambda_: float,
    max_iter: int,
    contact_impedance: float,
    cache_scope: str = "both",
    cache_dir: str = ".pyeidors_cache/v2",
) -> None:
    """Execute GN absolute reconstruction and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] CSV data file: {csv_path}")
    print(f"[INFO] YAML metadata file: {metadata_path}")
    print(
        "[INFO] Background sigma: "
        f"{background_sigma}, lambda: {lambda_}, measurement_gain: {measurement_gain}"
    )
    measurement_min = float(np.min(measurement))
    measurement_max = float(np.max(measurement))
    print(
        f"[INFO] Measurement range: [{measurement_min:.6e}, {measurement_max:.6e}]"
    )

    dataset = _build_dataset(measurement, metadata)
    pattern_config = dataset.pattern_config
    n_elec = pattern_config.n_elec

    mesh = load_or_create_mesh(
        mesh_dir=str(DEFAULT_MESH_DIR),
        mesh_name=None,
        n_elec=n_elec,
        radius=float(mesh_radius),
        refinement=int(refinement),
        electrode_coverage=float(metadata.get("electrode_coverage", 0.5)),
    )

    system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        contact_impedance=np.ones(n_elec, dtype=float) * float(contact_impedance),
        base_conductivity=float(background_sigma),
        regularization_type="noser",
        regularization_alpha=1.0,
        noser_exponent=0.5,
        cache_scope=cache_scope,
        cache_dir=cache_dir,
    )
    system.setup(mesh=mesh)
    _configure_reconstructor(
        system,
        lambda_=lambda_,
        max_iter=max_iter,
        background_sigma=background_sigma,
    )

    eit_data: EITData = dataset.to_eit_data(
        frame_index=0,
        data_type="real",
        copy_policy="view",
    )
    baseline_image = system.create_homogeneous_image(conductivity=background_sigma)
    base_forward, _ = system.fwd_model.fwd_solve(baseline_image)

    corrected_meas, was_flipped = align_measurement_polarity(
        eit_data.meas,
        base_forward.meas,
    )
    if was_flipped:
        print(
            "[INFO] Polarity correction: measurement data flipped (inverted U-shape detected)"
        )
        eit_data.meas = corrected_meas

    measured_min = float(np.min(eit_data.meas))
    measured_max = float(np.max(eit_data.meas))
    base_min = float(np.min(base_forward.meas))
    base_max = float(np.max(base_forward.meas))
    scale_ratio = float(
        np.abs(eit_data.meas).max() / (np.abs(base_forward.meas).max() + 1e-12)
    )
    print(f"[INFO] Measured voltage range: [{measured_min:.6e}, {measured_max:.6e}]")
    print(f"[INFO] Model prediction range: [{base_min:.6e}, {base_max:.6e}]")
    print(
        f"[INFO] Meas/Model ratio: {scale_ratio:.2f} "
        "(if far from 1, adjust background conductivity or stimulation current)"
    )

    n_elements = int(fem.Function(system.fwd_model.V_sigma).x.array.size)
    initial_sigma = np.full(n_elements, float(background_sigma), dtype=float)

    system.reconstructor.ensure_regularization_ready()
    recon_result = system.reconstructor.reconstruct(
        measured_data=eit_data,
        initial_conductivity=initial_sigma,
        jacobian_method="efficient",
    )

    conductivity_fn = recon_result.conductivity
    conductivity_vec = function_get_array(conductivity_fn).copy()

    sim_data, _ = system.fwd_model.fwd_solve(
        EITImage(elem_data=conductivity_vec, fwd_model=system.fwd_model)
    )
    measured_vec = eit_data.meas
    predicted_vec = sim_data.meas

    visualizer = EITVisualizer(style="seaborn", figsize=(10, 8))
    fig_cond = visualizer.plot_conductivity(
        mesh,
        conductivity_fn,
        title="GN Absolute Imaging Conductivity Distribution",
        colormap="viridis",
        show_electrodes=True,
    )
    fig_cond.savefig(output_dir / "conductivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cond)

    corr = np.corrcoef(measured_vec, predicted_vec)[0, 1]

    fig_cmp, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_left, ax_right = axes

    ax_left.plot(measured_vec, "b.-", label="Measured", markersize=3)
    ax_left.plot(predicted_vec, "r--", label="Predicted")
    ax_left.set_xlabel("Measurement index")
    ax_left.set_ylabel("Voltage (V)")
    ax_left.set_title("Boundary Voltage Comparison")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend()

    ax_right.scatter(measured_vec, predicted_vec, s=15, alpha=0.7, c="steelblue")
    vmin = min(np.min(measured_vec), np.min(predicted_vec))
    vmax = max(np.max(measured_vec), np.max(predicted_vec))
    ax_right.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label="y=x")
    ax_right.set_xlabel("Measured Voltage (V)")
    ax_right.set_ylabel("Predicted Voltage (V)")
    ax_right.set_title(f"Scatter Plot (r = {corr:.4f})")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend()
    ax_right.set_aspect("equal", adjustable="box")

    fig_cmp.tight_layout()
    fig_cmp.savefig(output_dir / "prediction_vs_measurement.png", dpi=300, bbox_inches="tight")
    plt.close(fig_cmp)

    np.savez(
        output_dir / "result_arrays.npz",
        conductivity=conductivity_vec,
        measured=measured_vec,
        predicted=predicted_vec,
        residual=np.asarray(predicted_vec) - np.asarray(measured_vec),
    )

    summary_payload = {
        "csv": str(csv_path),
        "metadata": str(metadata_path),
        "use_col": int(col_idx),
        "n_elec": n_elec,
        "mesh_radius": mesh_radius,
        "refinement": refinement,
        "regularization": "NOSER",
        "lambda": lambda_,
        "max_iterations": max_iter,
        "initial_sigma": background_sigma,
        "measurement_gain": measurement_gain,
        "contact_impedance": contact_impedance,
        "residual_history": list(recon_result.residual_history or []),
        "sigma_change_history": list(recon_result.sigma_change_history or []),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] GN absolute imaging complete, results saved to: {output_dir}")
    print(f"Conductivity image: {output_dir / 'conductivity.png'}")
    print(f"Prediction vs Measurement: {output_dir / 'prediction_vs_measurement.png'}")
