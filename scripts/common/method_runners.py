"""Method runners for unified reconstruction CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from . import gn_absolute_runner
from . import gn_difference_runner
from . import sparse_bayes_runner
from .case_loader import (
    load_absolute_frame_from_paired_csv,
    load_frame_csv,
    load_paired_frames,
)
from .io_utils import align_frames_polarity, align_measurement_polarity, load_metadata
from .recon_cli_models import (
    CaseResult,
    InputMode,
    ReconstructionCase,
    ReconstructionMethod,
)


def _default(value: Optional[float], fallback: float) -> float:
    return fallback if value is None else float(value)


def _should_skip_case(output_dir: Path, overwrite: bool) -> bool:
    result_file = output_dir / "result_arrays.npz"
    outputs_file = output_dir / "outputs.npz"
    has_outputs = result_file.exists() or outputs_file.exists()
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
                measurement_gain=float(args.measurement_gain),
                background_sigma=_default(args.background_sigma, 0.001),
                lambda_=_default(args.lam, 0.02),
                max_iter=int(args.max_iter if args.max_iter is not None else 15),
                contact_impedance=_default(args.contact_impedance, 1e-5),
                cache_scope=str(getattr(args, "cache_scope", "both")),
                cache_dir=str(getattr(args, "cache_dir", ".pyeidors_cache/v2")),
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
    ctx = gn_difference_runner.build_shared_context(
        mesh_dir=str(args.mesh_dir),
        mesh_name=str(args.mesh_name) if args.mesh_name is not None else None,
        n_elec=int(args.n_elec),
        radius=_default(args.radius, 0.025),
        drive_value=args.drive_value,
        contact_impedance=_default(args.contact_impedance, 1e-6),
        background_sigma=_default(args.background_sigma, 1.0),
        lam=_default(args.lam, 0.1),
        cache_scope=str(getattr(args, "cache_scope", "both")),
        cache_dir=str(getattr(args, "cache_dir", ".pyeidors_cache/v2")),
        cache_clear_names=list(getattr(args, "cache_clear_name", []) or []),
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
                vh_vi = np.vstack([vh, vi])
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
    np.savez(
        output_dir / "result_arrays.npz",
        conductivity=result.conductivity,
        measured=result.measured,
        predicted=result.simulated,
        residual=result.residual,
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
        cache_dir=str(getattr(args, "cache_dir", ".pyeidors_cache/v2")),
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
                        raw_measurements = raw_measurements / float(args.measurement_gain)

                    calib_col = (
                        args.calibration_col
                        if args.calibration_col >= 0
                        else args.reference_col
                    )
                    cols_to_align = [args.reference_col, args.target_col, calib_col]
                    unique_cols = list(dict.fromkeys(cols_to_align))
                    selected = np.vstack([raw_measurements[:, c] for c in unique_cols])
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
                    diff_measurements = np.vstack([ref_frame, target_frame])

                diff_dataset = sparse_bayes_runner.measurement_to_dataset(
                    diff_measurements,
                    dict(metadata),
                )
                calibration_frame = args.calibration_col if args.calibration_col >= 0 else 0
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
