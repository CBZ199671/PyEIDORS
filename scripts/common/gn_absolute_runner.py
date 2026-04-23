"""GN absolute reconstruction runner shared by unified CLI."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem

from pyeidors.core_system import EITSystem
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.data.structures import EITData, EITImage
from pyeidors.electrodes.layout import effective_pattern_layout_for_3d_mesh
from pyeidors.femx import function_get_array
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.perf.policy import (
    DEFAULT_ACCELERATION_PROFILE,
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    DEFAULT_CHOLMOD_MAX_N,
    DEFAULT_FORWARD_BACKEND,
    DEFAULT_MESH_FAMILY,
    DEFAULT_INEXACT_ETA0,
    DEFAULT_INEXACT_ETA_MAX,
    DEFAULT_INEXACT_ETA_MIN,
    DEFAULT_INEXACT_FORCING,
    DEFAULT_INEXACT_MODE,
    DEFAULT_JACOBIAN_BLOCK_CANDIDATES,
    DEFAULT_JACOBIAN_BLOCK_SIZE,
    DEFAULT_JACOBIAN_BLOCK_TUNE,
    DEFAULT_LOWRANK_ENERGY,
    DEFAULT_LOWRANK_METHOD,
    DEFAULT_LOWRANK_MODE,
    DEFAULT_LOWRANK_RANK,
    DEFAULT_PRECONDITIONER,
    DEFAULT_ROM_MODE,
    DEFAULT_ROM_RANK_ADAPTIVE,
    DEFAULT_ROM_RANK_GLOBAL,
    DEFAULT_ROM_REFRESH_EVERY,
    DEFAULT_ROM_SNAPSHOT_SOURCE,
    normalize_forward_backend,
    normalize_mesh_family,
)
from pyeidors.physics.current_drive import normalize_pattern_config_for_mesh
from pyeidors.visualization import EITVisualizer

from .hdf5_outputs import RECONSTRUCTION_ARRAYS_SCHEMA, write_output_bundle
from .io_utils import align_measurement_polarity

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MESH_DIR = REPO_ROOT / "eit_meshes"


def _configure_reconstructor(
    system: EITSystem,
    *,
    lambda_: float,
    max_iter: int,
    background_sigma: float,
    solver_mode: str,
    linear_solver: str,
    jacobian_update_every: int,
    jacobian_reuse_tol: float,
    line_search_mode: str,
    preconditioner: str,
    fast_linear_path: str,
    rom_mode: str,
    rom_rank_global: int,
    rom_rank_adaptive: int,
    rom_refresh_every: int,
    rom_snapshot_source: str,
    inexact_mode: str,
    inexact_forcing: str,
    inexact_eta0: float,
    inexact_eta_min: float,
    inexact_eta_max: float,
    lowrank_mode: str,
    lowrank_rank: int,
    lowrank_method: str,
    lowrank_energy: float,
    absolute_startup_cache: bool,
    cholmod_max_n: int,
    cholmod_max_memory_gib: float,
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
    recon.solver_mode = str(solver_mode)
    recon.linear_solver = str(linear_solver)
    recon.jacobian_update_every = int(max(1, jacobian_update_every))
    recon.jacobian_reuse_tol = float(max(0.0, jacobian_reuse_tol))
    recon.line_search_mode = str(line_search_mode)
    recon.preconditioner = str(preconditioner)
    recon.fast_linear_path = str(fast_linear_path)
    recon.rom_mode = str(rom_mode)
    recon.rom_rank_global = int(max(1, rom_rank_global))
    recon.rom_rank_adaptive = int(max(0, rom_rank_adaptive))
    recon.rom_refresh_every = int(max(1, rom_refresh_every))
    recon.rom_snapshot_source = str(rom_snapshot_source)
    recon.inexact_mode = str(inexact_mode)
    recon.inexact_forcing = str(inexact_forcing)
    recon.inexact_eta0 = float(inexact_eta0)
    recon.inexact_eta_min = float(inexact_eta_min)
    recon.inexact_eta_max = float(inexact_eta_max)
    recon.lowrank_mode = str(lowrank_mode)
    recon.lowrank_rank = int(max(1, lowrank_rank))
    recon.lowrank_method = str(lowrank_method)
    recon.lowrank_energy = float(lowrank_energy)
    recon.absolute_startup_cache = bool(absolute_startup_cache)
    recon.cholmod_max_n = int(max(1, cholmod_max_n))
    recon.cholmod_max_memory_gib = float(max(0.25, cholmod_max_memory_gib))


def _build_dataset(
    measurement: np.ndarray, metadata: Dict[str, Any]
) -> MeasurementDataset:
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
    mesh_dim: int,
    mesh_height: float,
    electrode_height_ratio: float,
    z_center: float,
    mesh_dir: Path,
    mesh_name: str | None,
    measurement_gain: float,
    background_sigma: float,
    lambda_: float,
    max_iter: int,
    contact_impedance: float,
    cache_scope: str = "both",
    cache_dir: str = ".pyeidors_cache/v2",
    solver_mode: str = "strict",
    linear_solver: str = "auto",
    jacobian_update_every: int = 1,
    jacobian_reuse_tol: float = 0.0,
    line_search_mode: str = "full",
    preconditioner: str = DEFAULT_PRECONDITIONER,
    fast_linear_path: str = "auto",
    rom_mode: str = DEFAULT_ROM_MODE,
    rom_rank_global: int = DEFAULT_ROM_RANK_GLOBAL,
    rom_rank_adaptive: int = DEFAULT_ROM_RANK_ADAPTIVE,
    rom_refresh_every: int = DEFAULT_ROM_REFRESH_EVERY,
    rom_snapshot_source: str = DEFAULT_ROM_SNAPSHOT_SOURCE,
    inexact_mode: str = DEFAULT_INEXACT_MODE,
    inexact_forcing: str = DEFAULT_INEXACT_FORCING,
    inexact_eta0: float = DEFAULT_INEXACT_ETA0,
    inexact_eta_min: float = DEFAULT_INEXACT_ETA_MIN,
    inexact_eta_max: float = DEFAULT_INEXACT_ETA_MAX,
    lowrank_mode: str = DEFAULT_LOWRANK_MODE,
    lowrank_rank: int = DEFAULT_LOWRANK_RANK,
    lowrank_method: str = DEFAULT_LOWRANK_METHOD,
    lowrank_energy: float = DEFAULT_LOWRANK_ENERGY,
    absolute_startup_cache: bool = True,
    forward_mat_solve: str = "off",
    petsc_device: str = "auto",
    device: str = "auto",
    forward_backend: str = DEFAULT_FORWARD_BACKEND,
    mesh_family: str = DEFAULT_MESH_FAMILY,
    geometry_version: str = DEFAULT_3D_GEOMETRY_VERSION,
    cholmod_max_n: int = DEFAULT_CHOLMOD_MAX_N,
    cholmod_max_memory_gib: float = DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    acceleration_profile: str = DEFAULT_ACCELERATION_PROFILE,
    jacobian_block_tune: str = DEFAULT_JACOBIAN_BLOCK_TUNE,
    jacobian_block_size: int = DEFAULT_JACOBIAN_BLOCK_SIZE,
    jacobian_block_candidates: list[int]
    | tuple[int, ...] = DEFAULT_JACOBIAN_BLOCK_CANDIDATES,
) -> None:
    """Execute GN absolute reconstruction and save outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    forward_backend = normalize_forward_backend(
        forward_backend,
        default=DEFAULT_FORWARD_BACKEND,
    )
    mesh_family = normalize_mesh_family(
        mesh_family,
        default=DEFAULT_MESH_FAMILY,
    )
    geometry_version = (
        str(geometry_version).strip().lower() or DEFAULT_3D_GEOMETRY_VERSION
    )

    print(f"[INFO] CSV data file: {csv_path}")
    print(f"[INFO] YAML metadata file: {metadata_path}")
    print(
        "[INFO] Background sigma: "
        f"{background_sigma}, lambda: {lambda_}, measurement_gain: {measurement_gain}"
    )
    measurement_min = float(np.min(measurement))
    measurement_max = float(np.max(measurement))
    print(f"[INFO] Measurement range: [{measurement_min:.6e}, {measurement_max:.6e}]")

    dataset = _build_dataset(measurement, metadata)
    pattern_config, drive_mode_diag = normalize_pattern_config_for_mesh(
        dataset.pattern_config,
        mesh_tdim=int(mesh_dim),
    )
    total_electrodes = int(pattern_config.n_elec) * max(int(pattern_config.n_rings), 1)
    electrode_layout = str(getattr(pattern_config, "electrode_layout", "ring_major"))
    pattern_n_elec, pattern_n_rings = effective_pattern_layout_for_3d_mesh(
        mesh_tdim=int(mesh_dim),
        n_elec=int(pattern_config.n_elec),
        n_rings=int(pattern_config.n_rings),
        electrode_layout=electrode_layout,
    )
    if (pattern_n_elec, pattern_n_rings) != (
        int(pattern_config.n_elec),
        int(pattern_config.n_rings),
    ):
        pattern_config = replace(
            pattern_config,
            n_elec=pattern_n_elec,
            n_rings=pattern_n_rings,
        )

    mesh = load_or_create_mesh(
        mesh_dir=str(mesh_dir),
        mesh_name=mesh_name,
        n_elec=total_electrodes,
        dimension=int(mesh_dim),
        radius=float(mesh_radius),
        refinement=int(refinement),
        height=float(mesh_height),
        electrode_height_ratio=float(electrode_height_ratio),
        z_center=float(z_center),
        electrode_coverage=float(metadata.get("electrode_coverage", 0.5)),
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        electrode_layout=electrode_layout,
    )

    system = EITSystem(
        n_elec=total_electrodes,
        pattern_config=pattern_config,
        contact_impedance=np.ones(total_electrodes, dtype=float)
        * float(contact_impedance),
        base_conductivity=float(background_sigma),
        regularization_type="noser",
        regularization_alpha=1.0,
        noser_exponent=0.5,
        cache_scope=cache_scope,
        cache_dir=cache_dir,
        solver_mode=solver_mode,
        linear_solver=linear_solver,
        jacobian_update_every=jacobian_update_every,
        jacobian_reuse_tol=jacobian_reuse_tol,
        line_search_mode=line_search_mode,
        preconditioner=preconditioner,
        fast_linear_path=fast_linear_path,
        rom_mode=rom_mode,
        rom_rank_global=int(max(1, rom_rank_global)),
        rom_rank_adaptive=int(max(0, rom_rank_adaptive)),
        rom_refresh_every=int(max(1, rom_refresh_every)),
        rom_snapshot_source=str(rom_snapshot_source),
        inexact_mode=str(inexact_mode),
        inexact_forcing=str(inexact_forcing),
        inexact_eta0=float(inexact_eta0),
        inexact_eta_min=float(inexact_eta_min),
        inexact_eta_max=float(inexact_eta_max),
        lowrank_mode=str(lowrank_mode),
        lowrank_rank=int(max(1, lowrank_rank)),
        lowrank_method=str(lowrank_method),
        lowrank_energy=float(lowrank_energy),
        absolute_startup_cache=absolute_startup_cache,
        cholmod_max_n=int(max(1, cholmod_max_n)),
        cholmod_max_memory_gib=float(max(0.25, cholmod_max_memory_gib)),
        acceleration_profile=str(acceleration_profile),
        jacobian_block_tune=str(jacobian_block_tune),
        jacobian_block_size=int(max(0, jacobian_block_size)),
        jacobian_block_candidates=tuple(
            int(v) for v in jacobian_block_candidates if int(v) > 0
        ),
        petsc_device=str(petsc_device),
        device=str(device),
        forward_backend=str(forward_backend),
        mesh_family=str(mesh_family),
        linear_backend_config={
            "mat_solve_mode": str(forward_mat_solve),
            "petsc_device": str(petsc_device),
        },
    )
    system.setup(mesh=mesh)
    _configure_reconstructor(
        system,
        lambda_=lambda_,
        max_iter=max_iter,
        background_sigma=background_sigma,
        solver_mode=solver_mode,
        linear_solver=linear_solver,
        jacobian_update_every=jacobian_update_every,
        jacobian_reuse_tol=jacobian_reuse_tol,
        line_search_mode=line_search_mode,
        preconditioner=preconditioner,
        fast_linear_path=fast_linear_path,
        rom_mode=rom_mode,
        rom_rank_global=rom_rank_global,
        rom_rank_adaptive=rom_rank_adaptive,
        rom_refresh_every=rom_refresh_every,
        rom_snapshot_source=rom_snapshot_source,
        inexact_mode=inexact_mode,
        inexact_forcing=inexact_forcing,
        inexact_eta0=inexact_eta0,
        inexact_eta_min=inexact_eta_min,
        inexact_eta_max=inexact_eta_max,
        lowrank_mode=lowrank_mode,
        lowrank_rank=lowrank_rank,
        lowrank_method=lowrank_method,
        lowrank_energy=lowrank_energy,
        absolute_startup_cache=absolute_startup_cache,
        cholmod_max_n=cholmod_max_n,
        cholmod_max_memory_gib=cholmod_max_memory_gib,
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
    fig_cmp.savefig(
        output_dir / "prediction_vs_measurement.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig_cmp)

    write_output_bundle(
        output_dir / "result_arrays.h5",
        {
            "conductivity": conductivity_vec,
            "measured": measured_vec,
            "predicted": predicted_vec,
            "residual": np.asarray(predicted_vec) - np.asarray(measured_vec),
        },
        {"package_role": "reconstruction_result_arrays", "mode": "absolute"},
        schema=RECONSTRUCTION_ARRAYS_SCHEMA,
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
        **drive_mode_diag,
        "residual_history": list(recon_result.residual_history or []),
        "sigma_change_history": list(recon_result.sigma_change_history or []),
        "solver_mode": solver_mode,
        "linear_solver": linear_solver,
        "line_search_mode": line_search_mode,
        "preconditioner": preconditioner,
        "fast_linear_path": fast_linear_path,
        "rom_mode": rom_mode,
        "rom_rank_global": int(rom_rank_global),
        "rom_rank_adaptive": int(rom_rank_adaptive),
        "rom_refresh_every": int(rom_refresh_every),
        "rom_snapshot_source": rom_snapshot_source,
        "inexact_mode": inexact_mode,
        "inexact_forcing": inexact_forcing,
        "inexact_eta0": float(inexact_eta0),
        "inexact_eta_min": float(inexact_eta_min),
        "inexact_eta_max": float(inexact_eta_max),
        "lowrank_mode": lowrank_mode,
        "lowrank_rank": int(lowrank_rank),
        "lowrank_method": lowrank_method,
        "lowrank_energy": float(lowrank_energy),
        "absolute_startup_cache": bool(absolute_startup_cache),
        "forward_mat_solve": forward_mat_solve,
        "forward_backend": str(forward_backend),
        "mesh_family": str(mesh_family),
        "geometry_version": str(geometry_version),
        "petsc_device": petsc_device,
        "device": device,
        "cholmod_max_n": int(cholmod_max_n),
        "cholmod_max_memory_gib": float(cholmod_max_memory_gib),
        "jacobian_block_tune": str(jacobian_block_tune),
        "jacobian_block_size": int(jacobian_block_size),
        "jacobian_block_candidates": [int(v) for v in jacobian_block_candidates],
        "diagnostics": recon_result.diagnostics,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] GN absolute imaging complete, results saved to: {output_dir}")
    print(f"Conductivity image: {output_dir / 'conductivity.png'}")
    print(f"Prediction vs Measurement: {output_dir / 'prediction_vs_measurement.png'}")
