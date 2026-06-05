"""Method runners for unified reconstruction CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from pyeidors.perf import (
    DEFAULT_ACCELERATION_PROFILE,
    DEFAULT_ABSOLUTE_STARTUP_CACHE,
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    DEFAULT_CHOLMOD_MAX_N,
    DEFAULT_FORWARD_BACKEND,
    DEFAULT_FORWARD_MAT_SOLVE,
    DEFAULT_INEXACT_ETA0,
    DEFAULT_INEXACT_ETA_MAX,
    DEFAULT_INEXACT_ETA_MIN,
    DEFAULT_INEXACT_FORCING,
    DEFAULT_INEXACT_MODE,
    DEFAULT_JACOBIAN_BLOCK_CANDIDATES,
    DEFAULT_JACOBIAN_BLOCK_SIZE,
    DEFAULT_JACOBIAN_BLOCK_TUNE,
    DEFAULT_LINEAR_SOLVER,
    DEFAULT_MESH_FAMILY,
    DEFAULT_LOWRANK_ENERGY,
    DEFAULT_LOWRANK_METHOD,
    DEFAULT_LOWRANK_MODE,
    DEFAULT_LOWRANK_RANK,
    DEFAULT_PETSC_DEVICE,
    DEFAULT_PRECONDITIONER,
    DEFAULT_ROM_MODE,
    DEFAULT_ROM_RANK_ADAPTIVE,
    DEFAULT_ROM_RANK_GLOBAL,
    DEFAULT_ROM_REFRESH_EVERY,
    DEFAULT_ROM_SNAPSHOT_SOURCE,
    normalize_forward_backend,
    normalize_mesh_family,
    normalize_petsc_device,
    parse_block_size_candidates,
    resolve_experimental_mode,
    resolve_forward_mat_solve,
    resolve_line_search_mode,
    resolve_solver_mode,
)
from pyeidors.runtime_paths import pyeidors_cache_path

from . import gn_absolute_runner
from . import gn_difference_runner
from . import sparse_bayes_runner
from .case_loader import (
    load_absolute_frame_from_paired_csv,
    load_frame_csv,
    load_paired_frames,
)
from .io_utils import align_frames_polarity, align_measurement_polarity, load_metadata
from .hdf5_outputs import RECONSTRUCTION_ARRAYS_SCHEMA, write_output_bundle
from .recon_cli_models import (
    CaseResult,
    InputMode,
    ReconstructionCase,
    ReconstructionMethod,
)


def _stack_frame_rows(*frames: np.ndarray) -> np.ndarray:
    if not frames:
        return np.empty((0, 0), dtype=np.float64)
    arrays = [np.asarray(frame).reshape(-1) for frame in frames]
    n_cols = arrays[0].size
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.size != n_cols:
            raise ValueError(
                f"frame {idx} has {arr.size} measurements, expected {n_cols}"
            )
    out = np.empty((len(arrays), n_cols), dtype=np.result_type(*arrays))
    for row, arr in enumerate(arrays):
        out[row, :] = arr
    return out


def _default(value: Optional[float], fallback: float) -> float:
    return fallback if value is None else float(value)


def _should_skip_case(output_dir: Path, overwrite: bool) -> bool:
    candidate_files = (
        output_dir / "result_arrays.h5",
        output_dir / "outputs.h5",
        output_dir / "result_arrays.npz",
        output_dir / "outputs.npz",
    )
    has_outputs = any(path.exists() for path in candidate_files)
    return has_outputs and not overwrite


def _safe_load_metrics(output_dir: Path) -> Dict[str, Any]:
    run_summary = output_dir / "run_summary.json"
    if run_summary.exists():
        try:
            return json.loads(run_summary.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            return {}

    summary_txt = output_dir / "summary.txt"
    if not summary_txt.exists():
        return {}

    metrics: Dict[str, Any] = {}
    for line in summary_txt.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    return metrics


def _collect_absolute_runtime_kwargs(args) -> Dict[str, Any]:
    mesh_dim = int(getattr(args, "mesh_dim", 2))
    solver_mode = resolve_solver_mode(
        getattr(args, "solver_mode", "auto"), mesh_dim=mesh_dim
    )
    return {
        "solver_mode": solver_mode,
        "linear_solver": str(getattr(args, "linear_solver", DEFAULT_LINEAR_SOLVER)),
        "jacobian_update_every": int(getattr(args, "jacobian_update_every", 1)),
        "jacobian_reuse_tol": float(getattr(args, "jacobian_reuse_tol", 0.0)),
        "line_search_mode": resolve_line_search_mode(
            getattr(args, "line_search_mode", "auto"), mesh_dim=mesh_dim
        ),
        "preconditioner": str(getattr(args, "preconditioner", DEFAULT_PRECONDITIONER)),
        "fast_linear_path": str(getattr(args, "fast_linear_path", "auto")),
        "rom_mode": resolve_experimental_mode(
            getattr(args, "rom_mode", DEFAULT_ROM_MODE)
        ),
        "rom_rank_global": int(
            getattr(args, "rom_rank_global", DEFAULT_ROM_RANK_GLOBAL)
        ),
        "rom_rank_adaptive": int(
            getattr(args, "rom_rank_adaptive", DEFAULT_ROM_RANK_ADAPTIVE)
        ),
        "rom_refresh_every": int(
            getattr(args, "rom_refresh_every", DEFAULT_ROM_REFRESH_EVERY)
        ),
        "rom_snapshot_source": str(
            getattr(args, "rom_snapshot_source", DEFAULT_ROM_SNAPSHOT_SOURCE)
        ),
        "inexact_mode": resolve_experimental_mode(
            getattr(args, "inexact_mode", DEFAULT_INEXACT_MODE)
        ),
        "inexact_forcing": str(
            getattr(args, "inexact_forcing", DEFAULT_INEXACT_FORCING)
        ),
        "inexact_eta0": float(getattr(args, "inexact_eta0", DEFAULT_INEXACT_ETA0)),
        "inexact_eta_min": float(
            getattr(args, "inexact_eta_min", DEFAULT_INEXACT_ETA_MIN)
        ),
        "inexact_eta_max": float(
            getattr(args, "inexact_eta_max", DEFAULT_INEXACT_ETA_MAX)
        ),
        "lowrank_mode": resolve_experimental_mode(
            getattr(args, "lowrank_mode", DEFAULT_LOWRANK_MODE)
        ),
        "lowrank_rank": int(getattr(args, "lowrank_rank", DEFAULT_LOWRANK_RANK)),
        "lowrank_method": str(getattr(args, "lowrank_method", DEFAULT_LOWRANK_METHOD)),
        "lowrank_energy": float(
            getattr(args, "lowrank_energy", DEFAULT_LOWRANK_ENERGY)
        ),
        "absolute_startup_cache": str(
            getattr(args, "absolute_startup_cache", DEFAULT_ABSOLUTE_STARTUP_CACHE)
        ).lower()
        != "off",
        "forward_mat_solve": resolve_forward_mat_solve(
            getattr(args, "forward_mat_solve", DEFAULT_FORWARD_MAT_SOLVE),
            mesh_dim=mesh_dim,
            solver_mode=solver_mode,
        ),
        "petsc_device": normalize_petsc_device(
            getattr(args, "petsc_device", DEFAULT_PETSC_DEVICE),
            default=DEFAULT_PETSC_DEVICE,
        ),
        "forward_backend": normalize_forward_backend(
            getattr(args, "forward_backend", DEFAULT_FORWARD_BACKEND),
            default=DEFAULT_FORWARD_BACKEND,
        ),
        "mesh_family": normalize_mesh_family(
            getattr(args, "mesh_family", DEFAULT_MESH_FAMILY),
            default=DEFAULT_MESH_FAMILY,
        ),
        "geometry_version": str(
            getattr(args, "geometry_version", DEFAULT_3D_GEOMETRY_VERSION)
        )
        .strip()
        .lower()
        or DEFAULT_3D_GEOMETRY_VERSION,
        "device": str(getattr(args, "device", "auto")).strip().lower() or "auto",
        "cholmod_max_n": int(getattr(args, "cholmod_max_n", DEFAULT_CHOLMOD_MAX_N)),
        "cholmod_max_memory_gib": float(
            getattr(args, "cholmod_max_memory_gib", DEFAULT_CHOLMOD_MAX_MEMORY_GIB)
        ),
        "acceleration_profile": str(
            getattr(args, "acceleration_profile", DEFAULT_ACCELERATION_PROFILE)
        )
        .strip()
        .lower()
        or DEFAULT_ACCELERATION_PROFILE,
        "jacobian_block_tune": str(
            getattr(args, "jacobian_block_tune", DEFAULT_JACOBIAN_BLOCK_TUNE)
        ),
        "jacobian_block_size": int(
            getattr(args, "jacobian_block_size", DEFAULT_JACOBIAN_BLOCK_SIZE)
        ),
        "jacobian_block_candidates": parse_block_size_candidates(
            getattr(
                args, "jacobian_block_candidates", DEFAULT_JACOBIAN_BLOCK_CANDIDATES
            )
        ),
    }


def run_gn_absolute_cases(
    *,
    cases: Iterable[ReconstructionCase],
    output_root: Path,
    args,
) -> List[CaseResult]:
    """Run GN absolute reconstruction for each case."""
    if args.metadata is None:
        raise ValueError("gn-absolute requires --metadata")

    output_root.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    runtime_kwargs = _collect_absolute_runtime_kwargs(args)
    results: List[CaseResult] = []

    for case in cases:
        output_dir = output_root / case.case_name
        if _should_skip_case(output_dir, args.overwrite):
            results.append(
                CaseResult(
                    case_name=case.case_name,
                    status="skipped",
                    output_dir=output_dir,
                )
            )
            continue

        try:
            csv_path = case.primary_path()
            col_idx = int(args.absolute_col)

            if case.input_mode == InputMode.FRAME:
                measurement = load_frame_csv(
                    csv_path,
                    measurement_gain=float(args.measurement_gain),
                    layout=str(args.frame_layout),
                    use_part=str(args.use_part),
                )
            else:
                measurement = load_absolute_frame_from_paired_csv(
                    csv_path,
                    col_idx=col_idx,
                    measurement_gain=float(args.measurement_gain),
                )

            gn_absolute_runner.run_absolute_reconstruction(
                measurement=measurement,
                metadata=metadata,
                csv_path=csv_path,
                metadata_path=args.metadata,
                col_idx=col_idx,
                output_dir=output_dir,
                mesh_radius=_default(args.mesh_radius, 0.03),
                refinement=int(args.refinement if args.refinement is not None else 12),
                mesh_dim=int(getattr(args, "mesh_dim", 2)),
                mesh_height=float(getattr(args, "mesh_height", 1.0)),
                electrode_height_ratio=float(
                    getattr(args, "electrode_height_ratio", 0.2)
                ),
                z_center=float(getattr(args, "z_center", 0.0)),
                mesh_dir=Path(args.mesh_dir),
                mesh_name=str(args.mesh_name) if args.mesh_name else None,
                measurement_gain=float(args.measurement_gain),
                background_sigma=_default(args.background_sigma, 0.001),
                lambda_=_default(args.lam, 0.02),
                max_iter=int(args.max_iter if args.max_iter is not None else 15),
                contact_impedance=_default(args.contact_impedance, 1e-5),
                cache_scope=str(getattr(args, "cache_scope", "both")),
                cache_dir=str(getattr(args, "cache_dir", pyeidors_cache_path("v2"))),
                **runtime_kwargs,
            )

            metrics = _safe_load_metrics(output_dir)
            results.append(
                CaseResult(
                    case_name=case.case_name,
                    status="success",
                    output_dir=output_dir,
                    metrics=metrics,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced by CLI tests
            if not args.continue_on_error:
                raise
            results.append(
                CaseResult(
                    case_name=case.case_name,
                    status="failed",
                    output_dir=output_dir,
                    error=str(exc),
                )
            )

    return results


def run_gn_difference_cases(
    *,
    cases: Iterable[ReconstructionCase],
    output_root: Path,
    args,
) -> List[CaseResult]:
    """Run GN single-step difference reconstruction for each case."""
    output_root.mkdir(parents=True, exist_ok=True)
    mesh_dim = int(getattr(args, "mesh_dim", 2))
    mesh_name = str(args.mesh_name) if args.mesh_name is not None else None
    if mesh_dim == 3 and mesh_name == "mesh_16e_r0p025_ref10_cov0p5":
        mesh_name = None
    resolved_solver_mode = (
        "fast"
        if str(getattr(args, "solver_mode", "auto")) == "auto" and mesh_dim == 3
        else str(getattr(args, "solver_mode", "strict"))
    )
    forward_mat_solve = str(getattr(args, "forward_mat_solve", "off"))
    if forward_mat_solve == "auto" and not (
        mesh_dim == 3 and resolved_solver_mode == "fast"
    ):
        forward_mat_solve = "off"
    petsc_device = normalize_petsc_device(
        getattr(args, "petsc_device", DEFAULT_PETSC_DEVICE),
        default=DEFAULT_PETSC_DEVICE,
    )
    forward_backend = normalize_forward_backend(
        getattr(args, "forward_backend", DEFAULT_FORWARD_BACKEND),
        default=DEFAULT_FORWARD_BACKEND,
    )
    mesh_family = normalize_mesh_family(
        getattr(args, "mesh_family", DEFAULT_MESH_FAMILY),
        default=DEFAULT_MESH_FAMILY,
    )
    geometry_version = (
        str(getattr(args, "geometry_version", DEFAULT_3D_GEOMETRY_VERSION))
        .strip()
        .lower()
        or DEFAULT_3D_GEOMETRY_VERSION
    )

    ctx = gn_difference_runner.build_shared_context(
        mesh_dir=str(args.mesh_dir),
        mesh_name=mesh_name,
        mesh_dim=mesh_dim,
        mesh_height=float(getattr(args, "mesh_height", 1.0)),
        electrode_height_ratio=float(getattr(args, "electrode_height_ratio", 0.2)),
        z_center=float(getattr(args, "z_center", 0.0)),
        refinement=int(args.refinement) if args.refinement is not None else None,
        n_elec=int(args.n_elec),
        radius=_default(args.radius, 0.025),
        drive_value=args.drive_value,
        contact_impedance=_default(args.contact_impedance, 1e-6),
        background_sigma=_default(args.background_sigma, 1.0),
        lam=_default(args.lam, 0.1),
        cache_scope=str(getattr(args, "cache_scope", "both")),
        cache_dir=str(getattr(args, "cache_dir", pyeidors_cache_path("v2"))),
        cache_clear_names=list(getattr(args, "cache_clear_name", []) or []),
        solver_mode=resolved_solver_mode,
        linear_solver=str(getattr(args, "linear_solver", "auto")),
        preconditioner=str(getattr(args, "preconditioner", "auto")),
        rom_mode=str(getattr(args, "rom_mode", "off")),
        rom_rank_global=int(getattr(args, "rom_rank_global", 32)),
        rom_rank_adaptive=int(getattr(args, "rom_rank_adaptive", 16)),
        rom_snapshot_source=str(getattr(args, "rom_snapshot_source", "hybrid")),
        lowrank_mode=str(getattr(args, "lowrank_mode", "off")),
        lowrank_rank=int(getattr(args, "lowrank_rank", 16)),
        lowrank_method=str(getattr(args, "lowrank_method", "tsvd")),
        lowrank_energy=float(getattr(args, "lowrank_energy", 0.995)),
        forward_mat_solve=forward_mat_solve,
        forward_backend=forward_backend,
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        petsc_device=petsc_device,
        device=str(getattr(args, "device", "auto")).strip().lower() or "auto",
    )

    expected_len = int(ctx["n_meas_total"])
    n_stim = int(ctx["n_stim"])
    n_meas_per_stim = ctx["n_meas_per_stim"]

    reference_cache: Dict[str, np.ndarray] = {}
    results: List[CaseResult] = []

    for case in cases:
        output_dir = output_root / case.case_name
        if _should_skip_case(output_dir, args.overwrite):
            results.append(
                CaseResult(
                    case_name=case.case_name,
                    status="skipped",
                    output_dir=output_dir,
                )
            )
            continue

        try:
            if case.input_mode == InputMode.PAIRED:
                vh, vi = load_paired_frames(
                    case.paired_csv,
                    use_part=str(args.use_part),
                    measurement_gain=float(args.measurement_gain),
                )
                vh_vi = _stack_frame_rows(vh, vi)
                vh_vi_aligned, _ = align_frames_polarity(vh_vi, ctx["base_meas"])
                vh, vi = vh_vi_aligned
            else:
                if case.reference_csv is None:
                    raise ValueError("Frame-mode difference case missing reference_csv")

                ref_key = str(case.reference_csv.resolve())
                if ref_key not in reference_cache:
                    reference_frame = load_frame_csv(
                        case.reference_csv,
                        measurement_gain=float(args.measurement_gain),
                        layout=str(args.frame_layout),
                        use_part=str(args.use_part),
                        expected_len=expected_len,
                        n_stim=n_stim,
                        n_meas_per_stim=n_meas_per_stim,
                    )
                    reference_frame, _ = align_measurement_polarity(
                        reference_frame,
                        ctx["base_meas"],
                    )
                    reference_cache[ref_key] = reference_frame

                vh = reference_cache[ref_key]
                vi = load_frame_csv(
                    case.target_csv,
                    measurement_gain=float(args.measurement_gain),
                    layout=str(args.frame_layout),
                    use_part=str(args.use_part),
                    expected_len=expected_len,
                    n_stim=n_stim,
                    n_meas_per_stim=n_meas_per_stim,
                )
                vi, _ = align_measurement_polarity(vi, ctx["base_meas"])

            diff_metrics = gn_difference_runner.process_frames(
                vh=vh,
                vi=vi,
                output_dir=output_dir,
                ctx=ctx,
                step_size_calib=bool(args.step_size_calibration),
                step_size_min=float(args.step_size_min),
                step_size_max=float(args.step_size_max),
                step_size_maxiter=int(args.step_size_maxiter),
                lam=_default(args.lam, 0.1),
                colormap=str(args.colormap),
                colorbar_scientific=bool(args.colorbar_scientific),
                colorbar_format=args.colorbar_format,
                transparent=bool(args.transparent),
                write_plots=not bool(args.no_plots),
                measurement_gain=float(args.measurement_gain),
            )

            results.append(
                CaseResult(
                    case_name=case.case_name,
                    status="success",
                    output_dir=output_dir,
                    metrics=diff_metrics,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced by CLI tests
            if not args.continue_on_error:
                raise
            results.append(
                CaseResult(
                    case_name=case.case_name,
                    status="failed",
                    output_dir=output_dir,
                    error=str(exc),
                )
            )

    return results


def _save_sparse_outputs_no_plots(result, output_dir: Path, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_output_bundle(
        output_dir / "result_arrays.h5",
        {
            "conductivity": result.conductivity,
            "measured": result.measured,
            "predicted": result.simulated,
            "residual": result.residual,
        },
        {"mode": mode, "package_role": "reconstruction_result_arrays"},
        schema=RECONSTRUCTION_ARRAYS_SCHEMA,
    )
    summary = {
        "mode": mode,
        "l2_error": float(result.l2_error),
        "relative_error": float(result.relative_error),
        "mse": float(result.mse),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def run_sparse_bayes_difference_cases(
    *,
    cases: Iterable[ReconstructionCase],
    output_root: Path,
    args,
) -> List[CaseResult]:
    """Run sparse Bayesian difference reconstruction for each case."""
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = sparse_bayes_runner.load_metadata(args.metadata)
    pattern_config = sparse_bayes_runner.PatternConfig(
        n_elec=int(metadata["n_elec"]),
        stim_pattern=metadata.get("stim_pattern", "{ad}"),
        meas_pattern=metadata.get("meas_pattern", "{ad}"),
        drive_mode=str(metadata.get("drive_mode", "line_current_density")),
        drive_value=float(metadata.get("drive_value", 1.0)),
        geometry_scale_to_m=float(metadata.get("geometry_scale_to_m", 1.0)),
        electrode_length_m_override=metadata.get("electrode_length_m_override"),
        use_meas_current=bool(metadata.get("use_meas_current", False)),
        use_meas_current_next=int(metadata.get("use_meas_current_next", 0)),
        rotate_meas=bool(metadata.get("rotate_meas", True)),
    )

    contact_impedance = np.full(
        pattern_config.n_elec,
        _default(args.contact_impedance, 1e-5),
    )
    eit_system = sparse_bayes_runner.EITSystem(
        n_elec=pattern_config.n_elec,
        pattern_config=pattern_config,
        contact_impedance=contact_impedance,
        base_conductivity=_default(args.background_sigma, 1.0),
        cache_scope=str(getattr(args, "cache_scope", "both")),
        cache_dir=str(getattr(args, "cache_dir", pyeidors_cache_path("v2"))),
    )
    mesh = sparse_bayes_runner.load_or_create_mesh(
        mesh_dir=str(args.mesh_dir),
        n_elec=pattern_config.n_elec,
        refinement=max(int(args.refinement if args.refinement is not None else 12), 4),
        mesh_name=args.mesh_name,
        radius=_default(args.mesh_radius, 1.0),
        electrode_coverage=float(args.electrode_coverage),
    )
    eit_system.setup(mesh=mesh)

    baseline_image = eit_system.create_homogeneous_image()
    baseline_data = eit_system.forward_solve(baseline_image)
    baseline_vector = baseline_data.meas

    config = sparse_bayes_runner.SparseBayesianConfig(
        cache_jacobian=bool(args.jacobian_cache),
        subspace_rank=args.subspace_rank,
        use_linear_warm_start=bool(args.linear_warm_start),
        solver=str(args.solver),
        linear_max_iterations=int(args.linear_max_iters),
        linear_tolerance=float(args.linear_tol),
        coarse_group_size=args.coarse_group_size,
        use_gpu=bool(args.use_gpu),
        gpu_dtype=str(args.gpu_dtype),
        coarse_levels=tuple(args.coarse_levels) if args.coarse_levels else None,
        block_iterations=int(args.block_iterations),
        block_size=args.block_size,
        refinement_gradient_tol=float(args.refinement_gradient_tol),
        coarse_iterations=int(args.coarse_iterations),
        coarse_relaxation=float(args.coarse_relaxation),
    )
    reconstructor = sparse_bayes_runner.SparseBayesianReconstructor(
        eit_system=eit_system,
        config=config,
    )

    expected_len = int(baseline_vector.shape[0])
    n_stim = int(eit_system.fwd_model.pattern_manager.n_stim)
    n_meas_per_stim_values = sorted(
        set(eit_system.fwd_model.pattern_manager.n_meas_per_stim)
    )
    n_meas_per_stim = (
        n_meas_per_stim_values[0] if len(n_meas_per_stim_values) == 1 else None
    )

    reference_cache: Dict[str, np.ndarray] = {}
    results: List[CaseResult] = []

    original_save = sparse_bayes_runner.save_result_outputs
    if args.no_plots:
        sparse_bayes_runner.save_result_outputs = _save_sparse_outputs_no_plots

    try:
        for case in cases:
            output_dir = output_root / case.case_name
            if _should_skip_case(output_dir, args.overwrite):
                results.append(
                    CaseResult(
                        case_name=case.case_name,
                        status="skipped",
                        output_dir=output_dir,
                    )
                )
                continue

            try:
                if case.input_mode == InputMode.PAIRED:
                    raw_measurements = np.loadtxt(
                        case.paired_csv,
                        delimiter=",",
                        dtype=float,
                    )
                    if raw_measurements.ndim == 1:
                        raw_measurements = raw_measurements[:, np.newaxis]
                    if args.measurement_gain and float(args.measurement_gain) != 1.0:
                        raw_measurements = raw_measurements / float(
                            args.measurement_gain
                        )

                    calib_col = (
                        args.calibration_col
                        if args.calibration_col >= 0
                        else args.reference_col
                    )
                    cols_to_align = [args.reference_col, args.target_col, calib_col]
                    unique_cols = list(dict.fromkeys(cols_to_align))
                    selected = _stack_frame_rows(
                        *(raw_measurements[:, c] for c in unique_cols)
                    )
                    aligned, _ = align_frames_polarity(
                        selected,
                        baseline_vector,
                    )
                    for idx, column in enumerate(unique_cols):
                        raw_measurements[:, column] = aligned[idx]

                    diff_measurements = sparse_bayes_runner.select_frames(
                        raw_measurements,
                        [args.reference_col, args.target_col],
                    )
                else:
                    if case.reference_csv is None:
                        raise ValueError("Frame-mode sparse case missing reference_csv")

                    ref_key = str(case.reference_csv.resolve())
                    if ref_key not in reference_cache:
                        ref_frame = load_frame_csv(
                            case.reference_csv,
                            measurement_gain=float(args.measurement_gain),
                            layout=str(args.frame_layout),
                            use_part=str(args.use_part),
                            expected_len=expected_len,
                            n_stim=n_stim,
                            n_meas_per_stim=n_meas_per_stim,
                        )
                        ref_frame, _ = align_measurement_polarity(
                            ref_frame,
                            baseline_vector,
                        )
                        reference_cache[ref_key] = ref_frame
                    ref_frame = reference_cache[ref_key]

                    target_frame = load_frame_csv(
                        case.target_csv,
                        measurement_gain=float(args.measurement_gain),
                        layout=str(args.frame_layout),
                        use_part=str(args.use_part),
                        expected_len=expected_len,
                        n_stim=n_stim,
                        n_meas_per_stim=n_meas_per_stim,
                    )
                    target_frame, _ = align_measurement_polarity(
                        target_frame,
                        baseline_vector,
                    )
                    diff_measurements = _stack_frame_rows(ref_frame, target_frame)

                diff_dataset = sparse_bayes_runner.measurement_to_dataset(
                    diff_measurements,
                    dict(metadata),
                )
                calibration_frame = (
                    args.calibration_col if args.calibration_col >= 0 else 0
                )
                if not 0 <= calibration_frame < diff_dataset.measurements.shape[0]:
                    raise IndexError(
                        "calibration_col must map to one of the selected difference frames."
                    )

                pre_calibration: Optional[Dict[str, float]] = None
                if args.difference_calibration == "before":
                    pre_calibration = sparse_bayes_runner.calibrate_measurements(
                        diff_dataset,
                        baseline_vector,
                        frame_index=calibration_frame,
                    )

                info = sparse_bayes_runner.run_difference_pipeline(
                    eit_system,
                    diff_dataset,
                    baseline_image,
                    output_dir,
                    reconstructor,
                    prior_scale=args.prior_scale,
                    noise_std=args.noise_std,
                    baseline_vector=baseline_vector,
                    calibration_mode=args.difference_calibration,
                    pre_calibration=pre_calibration,
                )

                results.append(
                    CaseResult(
                        case_name=case.case_name,
                        status="success",
                        output_dir=output_dir,
                        metrics=info.get("summary", {}),
                    )
                )
            except Exception as exc:  # pragma: no cover - surfaced by CLI tests
                if not args.continue_on_error:
                    raise
                results.append(
                    CaseResult(
                        case_name=case.case_name,
                        status="failed",
                        output_dir=output_dir,
                        error=str(exc),
                    )
                )
    finally:
        sparse_bayes_runner.save_result_outputs = original_save

    return results


METHOD_RUNNERS: Dict[ReconstructionMethod, Callable[..., List[CaseResult]]] = {
    ReconstructionMethod.GN_ABSOLUTE: run_gn_absolute_cases,
    ReconstructionMethod.GN_DIFFERENCE: run_gn_difference_cases,
    ReconstructionMethod.SPARSE_BAYES: run_sparse_bayes_difference_cases,
}


def get_method_runner(
    method: ReconstructionMethod,
) -> Callable[..., List[CaseResult]]:
    """Return registered runner for reconstruction method."""
    if method not in METHOD_RUNNERS:
        raise ValueError(f"Unsupported reconstruction method: {method}")
    return METHOD_RUNNERS[method]
