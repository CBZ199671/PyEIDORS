#!/usr/bin/env python3
"""Run Gauss-Newton absolute imaging with EIDORS-style parameters.

Key points:
- Uses only CSV column 3 (zero-based index 2) as target frame real part.
- Stimulation/measurement patterns and amplitude from companion YAML metadata.
- Mesh: 16 electrodes, cylinder radius 0.03 m, z_contact=1e-5, default refinement 12.
- Regularization: NOSER, lambda=0.02; GN max 15 iterations with backtracking line search.
- Initial conductivity: 0.001 S/m.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from fenics import Function
import numpy as np

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from pyeidors.core_system import EITSystem  # noqa: E402
from pyeidors.data.measurement_dataset import MeasurementDataset  # noqa: E402
from pyeidors.data.structures import PatternConfig  # noqa: E402
from pyeidors.data.structures import EITData, EITImage  # noqa: E402
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh  # noqa: E402
from pyeidors.visualization import EITVisualizer  # noqa: E402

# Use common modules
from common.io_utils import load_metadata, load_single_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyEIDORS GN absolute imaging (EIDORS-style parameters)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Measurement CSV with 4 columns (vh_real, vh_imag, vi_real, vi_imag)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="YAML metadata file corresponding to CSV",
    )
    parser.add_argument(
        "--use-col",
        type=int,
        default=2,
        help="Column index for absolute reconstruction (zero-based), default: column 3 (vi real)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--mesh-radius",
        type=float,
        default=0.03,
        help="Cylinder radius (m)",
    )
    parser.add_argument(
        "--refinement",
        type=int,
        default=12,
        help="Mesh refinement level, passed to load_or_create_mesh",
    )
    parser.add_argument(
        "--measurement-gain",
        type=float,
        default=10.0,
        help="Measurement amplifier gain, CSV voltages are divided by this value",
    )
    parser.add_argument(
        "--background-sigma",
        type=float,
        default=0.001,
        help="Initial/background conductivity (S/m)",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=0.02,
        help="Regularization parameter lambda",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="GN maximum iterations",
    )
    parser.add_argument(
        "--contact-impedance",
        type=float,
        default=1e-5,
        help="Contact impedance (ohm*m^2)",
    )
    return parser.parse_args()


def load_measurement_vector(csv_path: Path, col_idx: int, measurement_gain: float = 1.0) -> np.ndarray:
    """Load single frame from CSV and convert to (1, n_meas) shape."""
    frame = load_single_frame(csv_path, col_idx, measurement_gain)
    return frame.reshape(1, -1)


def build_dataset(measurements: np.ndarray, metadata: dict) -> MeasurementDataset:
    # Ensure metadata has required fields
    required = ["n_elec", "stim_pattern", "meas_pattern"]
    for key in required:
        if key not in metadata:
            raise KeyError(f"metadata missing required field: {key}")
    meta = dict(metadata)
    meta.setdefault("n_frames", int(measurements.shape[0]))
    return MeasurementDataset.from_metadata(measurements, meta)


def configure_reconstructor(system: EITSystem, lambda_: float = 0.02, max_iter: int = 15, background_sigma: float = 0.001) -> None:
    """Configure GN parameters to EIDORS style."""
    recon = system.reconstructor
    recon.max_iterations = max_iter
    recon.regularization_param = lambda_
    recon.line_search_steps = 12
    # Step size limits
    recon.max_step = 1.0
    recon.min_step = 1e-6  # Allow very small step sizes
    recon.convergence_tol = 1e-5
    recon.negate_jacobian = True
    recon.use_measurement_weights = True
    recon.measurement_weight_strategy = "scaled_baseline"
    # Prevent conductivity from dropping too low causing voltage explosion
    recon.clip_values = (background_sigma * 0.1, background_sigma * 100)
    recon.min_iterations = 1
    # EIDORS style: use prior error term
    recon.use_prior_term = True


def run_reconstruction(
    csv_path: Path,
    metadata_path: Path,
    col_idx: int,
    output_dir: Path,
    mesh_radius: float,
    refinement: int,
    measurement_gain: float = 10.0,
    background_sigma: float = 0.001,
    lambda_: float = 0.02,
    max_iter: int = 15,
    contact_impedance: float = 1e-5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] CSV data file: {csv_path}")
    print(f"[INFO] YAML metadata file: {metadata_path}")
    metadata = load_metadata(metadata_path)
    measurements = load_measurement_vector(csv_path, col_idx, measurement_gain=measurement_gain)
    dataset = build_dataset(measurements, metadata)
    print(f"[INFO] Background sigma: {background_sigma}, lambda: {lambda_}, measurement_gain: {measurement_gain}")
    print(f"[INFO] Measurement range: [{measurements.min():.6e}, {measurements.max():.6e}]")

    # Build PatternConfig from metadata to match acquisition settings
    pattern_config = dataset.pattern_config
    n_elec = pattern_config.n_elec

    # Generate/load mesh
    mesh = load_or_create_mesh(
        mesh_dir=str(REPO_ROOT / "eit_meshes"),
        mesh_name=None,
        n_elec=n_elec,
        radius=mesh_radius,
        refinement=refinement,
        electrode_coverage=float(metadata.get("electrode_coverage", 0.5)),
    )

    # EIT system configuration: contact impedance, NOSER regularization
    z_contact = np.ones(n_elec) * contact_impedance
    system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        contact_impedance=z_contact,
        base_conductivity=background_sigma,
        regularization_type="noser",
        regularization_alpha=1.0,
        noser_exponent=0.5,  # EIDORS style
    )
    system.setup(mesh=mesh)
    configure_reconstructor(system, lambda_=lambda_, max_iter=max_iter, background_sigma=background_sigma)

    # Prepare measurement data and initial values
    eit_data: EITData = dataset.to_eit_data(frame_index=0, data_type="real")

    # Baseline forward comparison (for information output only, no scaling)
    base_img = system.create_homogeneous_image(conductivity=background_sigma)
    base_forward, _ = system.fwd_model.fwd_solve(base_img)

    # Polarity correction (normal U/inverted U detection)
    from common.io_utils import align_measurement_polarity
    corrected_meas, was_flipped = align_measurement_polarity(eit_data.meas, base_forward.meas)
    if was_flipped:
        print("[INFO] Polarity correction: measurement data flipped (inverted U-shape detected)")
        eit_data.meas = corrected_meas

    print(f"[INFO] Measured voltage range: [{eit_data.meas.min():.6e}, {eit_data.meas.max():.6e}]")
    print(f"[INFO] Model prediction range: [{base_forward.meas.min():.6e}, {base_forward.meas.max():.6e}]")
    scale_ratio = np.abs(eit_data.meas).max() / (np.abs(base_forward.meas).max() + 1e-12)
    print(f"[INFO] Meas/Model ratio: {scale_ratio:.2f} (if far from 1, adjust background conductivity or stimulation current)")
    
    n_elements = len(Function(system.fwd_model.V_sigma).vector()[:])
    initial_sigma = np.full(n_elements, background_sigma, dtype=float)

    recon_result = system.reconstructor.reconstruct(
        measured_data=eit_data,
        initial_conductivity=initial_sigma,
        jacobian_method="efficient",
    )
    conductivity_fn = recon_result["conductivity"]
    conductivity_vec = conductivity_fn.vector()[:]

    # Forward prediction for curve comparison
    sim_data, _ = system.fwd_model.fwd_solve(EITImage(elem_data=conductivity_vec, fwd_model=system.fwd_model))
    measured_vec = eit_data.meas
    predicted_vec = sim_data.meas

    # Visualization
    visualizer = EITVisualizer(style="seaborn", figsize=(10, 8))
    fig_cond = visualizer.plot_conductivity(
        mesh,
        conductivity_fn,
        title="GN Absolute Imaging Conductivity Distribution",
        colormap="viridis",
        show_electrodes=True,
    )
    fig_cond.savefig(output_dir / "conductivity.png", dpi=300, bbox_inches="tight")

    # Compute correlation coefficient
    corr = np.corrcoef(measured_vec, predicted_vec)[0, 1]

    fig_cmp, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: curve comparison
    ax = axes[0]
    ax.plot(measured_vec, "b.-", label="Measured", markersize=3)
    ax.plot(predicted_vec, "r--", label="Predicted")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Boundary Voltage Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right plot: scatter plot + correlation coefficient
    ax2 = axes[1]
    ax2.scatter(measured_vec, predicted_vec, s=15, alpha=0.7, c='steelblue')
    vmin = min(np.min(measured_vec), np.min(predicted_vec))
    vmax = max(np.max(measured_vec), np.max(predicted_vec))
    ax2.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5, label='y=x')
    ax2.set_xlabel("Measured Voltage (V)")
    ax2.set_ylabel("Predicted Voltage (V)")
    ax2.set_title(f"Scatter Plot (r = {corr:.4f})")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')

    fig_cmp.tight_layout()
    fig_cmp.savefig(output_dir / "prediction_vs_measurement.png", dpi=300, bbox_inches="tight")

    # Save key numerical data
    np.savez(
        output_dir / "result_arrays.npz",
        conductivity=conductivity_vec,
        measured=measured_vec,
        predicted=predicted_vec,
        residual=np.asarray(predicted_vec) - np.asarray(measured_vec),
    )

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "csv": str(csv_path),
                "metadata": str(metadata_path),
                "use_col": col_idx,
                "n_elec": n_elec,
                "mesh_radius": mesh_radius,
                "refinement": refinement,
                "regularization": "NOSER",
                "lambda": lambda_,
                "max_iterations": max_iter,
                "initial_sigma": background_sigma,
                "measurement_gain": measurement_gain,
                "contact_impedance": contact_impedance,
                "residual_history": recon_result.get("residual_history", []),
                "sigma_change_history": recon_result.get("sigma_change_history", []),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] GN absolute imaging complete, results saved to: {output_dir}")
    print(f"Conductivity image: {output_dir/'conductivity.png'}")
    print(f"Prediction vs Measurement: {output_dir/'prediction_vs_measurement.png'}")


def main() -> None:
    args = parse_args()
    run_reconstruction(
        csv_path=args.csv,
        metadata_path=args.metadata,
        col_idx=args.use_col,
        output_dir=args.output_dir,
        mesh_radius=args.mesh_radius,
        refinement=args.refinement,
        measurement_gain=args.measurement_gain,
        background_sigma=args.background_sigma,
        lambda_=args.lambda_,
        max_iter=args.max_iter,
        contact_impedance=args.contact_impedance,
    )


if __name__ == "__main__":
    main()
