#!/usr/bin/env python3
"""Batch runner for single-step diff reconstruction on real CSV data.

Supports paired CSVs (reference/target columns) and single-frame CSVs with an
explicit reference frame. Shared computations (mesh, Jacobian, regularization)
are reused across files to speed up processing when parameters are fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import matplotlib as mpl

mpl.rcParams.update(
    {
        "axes.unicode_minus": False,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    }
)
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.linalg import lu_factor, lu_solve

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian
from pyeidors.visualization import create_visualizer

from common.io_utils import (
    load_csv_measurements,
    align_frames_polarity,
    align_measurement_polarity,
)
from common.mesh_utils import cell_to_node


def _build_noser_matrix(
    jacobian: np.ndarray,
    exponent: float = 0.5,
    alpha: float = 1.0,
    adaptive_floor: bool = True,
    floor: float = 1e-12,
    floor_fraction: float = 1e-6,
) -> np.ndarray:
    """Build NOSER regularization matrix from a precomputed Jacobian."""
    diag_entries = np.sum(jacobian * jacobian, axis=0)
    if adaptive_floor:
        adaptive_floor_value = np.max(diag_entries) * floor_fraction
        effective_floor = max(adaptive_floor_value, 1e-100)
    else:
        effective_floor = floor
    diag_entries = np.maximum(diag_entries, effective_floor)
    scaled_diag = diag_entries**exponent
    return alpha * np.diag(scaled_diag)


def _make_linear_solver(A: np.ndarray) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    try:
        lu, piv = lu_factor(A)
    except Exception:
        return None

    def _solve(b: np.ndarray) -> np.ndarray:
        return lu_solve((lu, piv), b)

    return _solve


def _solve_linear(
    A: np.ndarray,
    b: np.ndarray,
    solver: Optional[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    if solver is not None:
        try:
            return solver(b)
        except Exception:
            pass
    try:
        return np.linalg.solve(A, b)
    except Exception:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _select_complex_part(real: np.ndarray, imag: np.ndarray, use_part: str) -> np.ndarray:
    if use_part == "real":
        return real
    if use_part == "imag":
        return imag
    if use_part == "mag":
        return np.abs(real + 1j * imag)
    raise ValueError(f"Unsupported use_part={use_part}")


def _reshape_frame_matrix(
    arr: np.ndarray,
    expected_len: int,
    n_stim: int,
    n_meas_per_stim: Optional[int],
    layout: str,
) -> np.ndarray:
    if layout == "auto":
        if n_meas_per_stim is not None:
            if arr.shape == (n_stim, n_meas_per_stim):
                return arr.reshape(-1)
            if arr.shape == (n_meas_per_stim, n_stim):
                return arr.T.reshape(-1)
        if arr.size == expected_len and 1 in arr.shape:
            return arr.reshape(-1)
        raise ValueError(
            f"Cannot infer frame layout from shape {arr.shape}. "
            f"Expected {(n_stim, n_meas_per_stim)} or {(n_meas_per_stim, n_stim)}."
        )
    if layout == "stim-meas":
        if n_meas_per_stim is None:
            raise ValueError("n_meas_per_stim is not uniform; use --frame-layout vector.")
        if arr.shape != (n_stim, n_meas_per_stim):
            raise ValueError(
                f"Expected shape {(n_stim, n_meas_per_stim)} for stim-meas, got {arr.shape}."
            )
        return arr.reshape(-1)
    if layout == "meas-stim":
        if n_meas_per_stim is None:
            raise ValueError("n_meas_per_stim is not uniform; use --frame-layout vector.")
        if arr.shape != (n_meas_per_stim, n_stim):
            raise ValueError(
                f"Expected shape {(n_meas_per_stim, n_stim)} for meas-stim, got {arr.shape}."
            )
        return arr.T.reshape(-1)
    if layout == "vector":
        if arr.size != expected_len:
            raise ValueError(f"Expected {expected_len} values, got {arr.size}.")
        return arr.reshape(-1)
    raise ValueError(f"Unsupported frame layout: {layout}")


def _load_frame_csv(
    csv_path: Path,
    expected_len: int,
    n_stim: int,
    n_meas_per_stim: Optional[int],
    measurement_gain: float,
    layout: str,
    use_part: str,
) -> np.ndarray:
    arr = np.loadtxt(csv_path, delimiter=",")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"CSV {csv_path.name} contains a single value.")

    uses_complex = False
    if arr.ndim == 2 and arr.shape == (expected_len, 2):
        uses_complex = True
        frame = _select_complex_part(arr[:, 0], arr[:, 1], use_part)
    elif arr.ndim == 2 and arr.shape == (2, expected_len):
        uses_complex = True
        frame = _select_complex_part(arr[0], arr[1], use_part)
    elif arr.ndim == 2:
        frame = _reshape_frame_matrix(arr, expected_len, n_stim, n_meas_per_stim, layout)
    elif arr.ndim == 1:
        frame = arr
    else:
        raise ValueError(f"Unsupported CSV shape {arr.shape} in {csv_path.name}")

    if not uses_complex and use_part == "imag":
        raise ValueError("use_part=imag requires real/imag columns in the frame CSV.")

    if frame.shape[0] != expected_len:
        raise ValueError(f"Frame length {frame.shape[0]} does not match expected {expected_len}.")

    if measurement_gain != 1.0:
        frame = frame / measurement_gain
    return frame


def _collect_csv_files(
    input_dir: Optional[Path],
    pattern: str,
    csv_files: Optional[List[Path]],
    include_ad: bool,
) -> List[Path]:
    candidates: List[Path] = []
    if csv_files:
        candidates.extend(csv_files)
    if input_dir is not None:
        candidates.extend(sorted(input_dir.glob(pattern)))

    if not candidates:
        return []

    result: List[Path] = []
    seen = set()
    for path in candidates:
        path = path.expanduser()
        if path.suffix.lower() != ".csv":
            continue
        if not include_ad and path.stem.endswith("_AD"):
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return sorted(result, key=lambda p: p.name)


def _prepare_shared_context(
    *,
    mesh_dir: str,
    mesh_name: str,
    n_elec: int,
    radius: float,
    pattern_amplitude: Optional[float],
    contact_impedance: float,
    background_sigma: float,
    lam: float,
) -> dict:
    stim_amplitude = pattern_amplitude if pattern_amplitude is not None else 1.0
    print(f"[INFO] Diff imaging amplitude={stim_amplitude:.2e} (EIDORS-style, relative dSigma)")

    mesh = load_or_create_mesh(
        mesh_dir=mesh_dir,
        mesh_name=mesh_name,
        n_elec=n_elec,
        radius=radius,
    )
    pattern_cfg = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=stim_amplitude,
        use_meas_current=False,
        rotate_meas=True,
    )
    z_contact = np.full(n_elec, contact_impedance, dtype=float)
    fwd_model = EITForwardModel(n_elec=n_elec, pattern_config=pattern_cfg, z=z_contact, mesh=mesh)

    n_elem = len(fwd_model.V_sigma.dofmap().dofs())
    sigma_bg = np.full(n_elem, background_sigma)
    img_bg = EITImage(elem_data=sigma_bg, fwd_model=fwd_model)
    print(f"[INFO] Background conductivity: {background_sigma}")

    base_forward, _ = fwd_model.fwd_solve(img_bg)
    base_meas = base_forward.meas

    pattern_manager = fwd_model.pattern_manager
    n_stim = pattern_manager.n_stim
    n_meas_total = pattern_manager.n_meas_total
    unique_counts = sorted(set(pattern_manager.n_meas_per_stim))
    n_meas_per_stim = unique_counts[0] if len(unique_counts) == 1 else None

    jac_calc = EidorsStyleAdjointJacobian(fwd_model, use_torch=False)
    J = jac_calc.calculate_from_image(img_bg)

    R = _build_noser_matrix(J, exponent=0.5, alpha=1.0)
    Jt = J.T
    A = Jt @ J + lam * R
    solver = _make_linear_solver(A)

    return {
        "mesh": mesh,
        "fwd_model": fwd_model,
        "sigma_bg": sigma_bg,
        "img_bg": img_bg,
        "base_meas": base_meas,
        "n_stim": n_stim,
        "n_meas_total": n_meas_total,
        "n_meas_per_stim": n_meas_per_stim,
        "J": J,
        "Jt": Jt,
        "A": A,
        "solver": solver,
        "stim_amplitude": stim_amplitude,
    }


def _calibrate_step_size(
    *,
    fwd_model: EITForwardModel,
    sigma_bg: np.ndarray,
    delta_sigma: np.ndarray,
    dv: np.ndarray,
    base_meas: np.ndarray,
    step_size_min: float,
    step_size_max: float,
    step_size_maxiter: int,
) -> float:
    def _objective(scale: float) -> float:
        sigma_try = sigma_bg + scale * delta_sigma
        img_try = EITImage(elem_data=sigma_try, fwd_model=fwd_model)
        pred_vi_try, _ = fwd_model.fwd_solve(img_try)
        pred_diff_try = pred_vi_try.meas - base_meas
        residual = pred_diff_try - dv
        return float(np.mean(residual**2))

    result = minimize_scalar(
        _objective,
        bounds=(step_size_min, step_size_max),
        method="bounded",
        options={"maxiter": int(max(1, step_size_maxiter))},
    )
    if result.success:
        print(f"[INFO] Step-size calibration: alpha={result.x:.3g}, diff residual={result.fun:.3e}")
        return float(result.x)

    print("[WARN] Step-size calibration failed, fallback alpha=1.0")
    return 1.0


def _process_frames(
    *,
    vh: np.ndarray,
    vi: np.ndarray,
    output_dir: Path,
    ctx: dict,
    step_size_calib: bool,
    step_size_min: float,
    step_size_max: float,
    step_size_maxiter: int,
    lam: float,
    colormap: str,
    colorbar_scientific: bool,
    colorbar_format: Optional[str],
    transparent: bool,
    write_plots: bool,
    measurement_gain: float,
) -> float:
    dv = vi - vh
    if dv.shape[0] != ctx["J"].shape[0]:
        raise RuntimeError(
            f"Data length {dv.shape[0]} does not match Jacobian rows {ctx['J'].shape[0]}"
        )

    b = ctx["Jt"] @ dv
    delta_sigma = _solve_linear(ctx["A"], b, ctx["solver"])

    alpha = 1.0
    if step_size_calib:
        alpha = _calibrate_step_size(
            fwd_model=ctx["fwd_model"],
            sigma_bg=ctx["sigma_bg"],
            delta_sigma=delta_sigma,
            dv=dv,
            base_meas=ctx["base_meas"],
            step_size_min=step_size_min,
            step_size_max=step_size_max,
            step_size_maxiter=step_size_maxiter,
        )

    sigma_est = ctx["sigma_bg"] + alpha * delta_sigma
    delta_sigma_scaled = alpha * delta_sigma
    img_est = EITImage(elem_data=sigma_est, fwd_model=ctx["fwd_model"])
    pred_vi, _ = ctx["fwd_model"].fwd_solve(img_est)
    pred_diff = pred_vi.meas - ctx["base_meas"]
    meas_diff = dv

    res = pred_vi.meas - vi
    rmse_abs = float(np.sqrt(np.mean(res**2)))

    output_dir.mkdir(parents=True, exist_ok=True)

    if write_plots:
        viz = create_visualizer()
        if len(delta_sigma_scaled) == ctx["mesh"].num_cells():
            node_vals = cell_to_node(ctx["mesh"], delta_sigma_scaled)
        else:
            node_vals = delta_sigma_scaled
        eidors_style = colormap.lower() in {"eidors_diff", "eidors-diff"}
        format_mode = colorbar_format or ("scientific" if colorbar_scientific else "plain")
        fig = viz.plot_conductivity(
            ctx["mesh"],
            node_vals,
            title=f"Reconstruction dSigma (lam={lam})",
            colormap=colormap,
            minimal=not eidors_style,
            show_electrodes=True,
            scientific_notation=colorbar_scientific,
            colorbar_format=format_mode,
            transparent=transparent,
        )
        fig.savefig(
            output_dir / "reconstruction.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.15,
            transparent=transparent,
        )
        plt.close(fig)

        corr_diff = np.corrcoef(meas_diff, pred_diff)[0, 1]
        fig = plt.figure(figsize=(12, 5))
        idx = np.arange(len(meas_diff))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(idx, meas_diff, "b-", lw=1.0, label="Measured diff (vi-vh)")
        ax.plot(idx, pred_diff, "r--", lw=1.0, label="Predicted diff")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlabel("Measurement index")
        ax.set_ylabel("Voltage")
        ax.set_title("Diff comparison")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(meas_diff, pred_diff, s=15, alpha=0.7, c="steelblue")
        vmin = min(meas_diff.min(), pred_diff.min())
        vmax = max(meas_diff.max(), pred_diff.max())
        ax2.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5)
        ax2.set_xlabel("Measured diff")
        ax2.set_ylabel("Predicted diff")
        ax2.grid(alpha=0.3)
        ax2.set_title(f"Scatter (r = {corr_diff:.4f})")
        ax2.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(output_dir / "diff_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(vi, pred_vi.meas, s=10, alpha=0.7)
        vmin = min(vi.min(), pred_vi.meas.min())
        vmax = max(vi.max(), pred_vi.meas.max())
        ax1.plot([vmin, vmax], [vmin, vmax], "r--")
        ax1.set_title("Measured vs Predicted (abs, real)")
        ax1.grid(alpha=0.3)
        ax1.set_xlabel("Measured target")
        ax1.set_ylabel("Predicted")
        ax2 = fig.add_subplot(1, 2, 2)
        idx = np.arange(len(vi))
        ax2.plot(idx, vi, "b-", lw=1.0, label="Measured target")
        ax2.plot(idx, pred_vi.meas, "r--", lw=1.0, label="Predicted")
        ax2.legend()
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "voltage_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    np.savez(
        output_dir / "outputs.npz",
        sigma_est=sigma_est,
        delta_sigma=delta_sigma_scaled,
        sigma_bg=ctx["sigma_bg"],
        dv=meas_diff,
        pred_diff=pred_diff,
        vi=vi,
        pred_vi=pred_vi.meas,
        lambda_=lam,
        rmse_abs=rmse_abs,
        step_size_alpha=alpha,
        pattern_amplitude=ctx["stim_amplitude"],
        measurement_gain=measurement_gain,
    )
    return rmse_abs


def _run_single_file(
    *,
    csv_path: Path,
    output_dir: Path,
    ctx: dict,
    use_part: str,
    measurement_gain: float,
    step_size_calib: bool,
    step_size_min: float,
    step_size_max: float,
    step_size_maxiter: int,
    lam: float,
    colormap: str,
    colorbar_scientific: bool,
    colorbar_format: Optional[str],
    transparent: bool,
    write_plots: bool,
) -> None:
    vh, vi = load_csv_measurements(csv_path, use_part=use_part, measurement_gain=measurement_gain)
    vh_vi = np.vstack([vh, vi])
    vh_vi_aligned, flipped = align_frames_polarity(vh_vi, ctx["base_meas"])
    vh, vi = vh_vi_aligned
    if flipped:
        print(f"[INFO] Polarity flip in {csv_path.name}: frames {flipped}")
    rmse_abs = _process_frames(
        vh=vh,
        vi=vi,
        output_dir=output_dir,
        ctx=ctx,
        step_size_calib=step_size_calib,
        step_size_min=step_size_min,
        step_size_max=step_size_max,
        step_size_maxiter=step_size_maxiter,
        lam=lam,
        colormap=colormap,
        colorbar_scientific=colorbar_scientific,
        colorbar_format=colorbar_format,
        transparent=transparent,
        write_plots=write_plots,
        measurement_gain=measurement_gain,
    )
    print(f"[INFO] {csv_path.name}: RMSE(abs)={rmse_abs:.5f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch single-step diff reconstruction for real CSV data. "
            "Shared computations (mesh/Jacobian) are reused across files."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="directory of CSV files")
    parser.add_argument("--glob", type=str, default="*.csv", help="glob under input-dir")
    parser.add_argument("--csv", type=Path, action="append", default=None, help="explicit CSV path(s)")
    parser.add_argument("--include-ad", action="store_true", help="include *_AD.csv files")
    parser.add_argument(
        "--input-mode",
        choices=["paired", "frame"],
        default="paired",
        help="paired: ref/target columns in one CSV, frame: one CSV per frame",
    )
    parser.add_argument("--reference-csv", type=Path, default=None, help="reference frame CSV (frame mode)")
    parser.add_argument("--reference-index", type=int, default=None, help="reference index in inputs (frame mode)")
    parser.add_argument(
        "--frame-layout",
        choices=["auto", "stim-meas", "meas-stim", "vector"],
        default="auto",
        help="frame CSV layout (frame mode)",
    )
    parser.add_argument("--output-root", type=Path, required=True, help="output root directory")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--continue-on-error", action="store_true", help="continue when a file fails")
    parser.add_argument("--dry-run", action="store_true", help="list inputs and exit")
    parser.add_argument("--no-plots", action="store_true", help="skip plot generation")

    parser.add_argument("--lambda", dest="lam", type=float, default=0.1, help="regularization lambda")
    parser.add_argument("--use-part", choices=["real", "imag", "mag"], default="real")
    parser.add_argument("--pattern-amplitude", type=float, default=None, help="override stimulation amplitude (A)")
    parser.add_argument(
        "--contact-impedance",
        type=float,
        default=1e-6,
        help="contact impedance (ohm*m^2)",
    )
    parser.add_argument("--measurement-gain", type=float, default=10.0, help="divide measured voltages by gain")
    parser.add_argument("--step-size-calibration", action="store_true", help="enable 1-D step-size search")
    parser.add_argument("--step-size-min", type=float, default=1e-3, help="lower bound for step-size search")
    parser.add_argument("--step-size-max", type=float, default=1e1, help="upper bound for step-size search")
    parser.add_argument("--step-size-maxiter", type=int, default=50, help="max iterations for step-size search")
    parser.add_argument("--background-sigma", type=float, default=1.0, help="background conductivity (S/m)")
    parser.add_argument("--colormap", type=str, default="viridis", help="colormap for reconstruction")
    parser.add_argument("--colorbar-scientific", action="store_true", help="scientific notation colorbar")
    parser.add_argument(
        "--colorbar-format",
        type=str,
        default=None,
        choices=["plain", "scientific", "matlab_short"],
        help="colorbar format rule",
    )
    parser.add_argument("--transparent", action="store_true", help="transparent background for plots")

    parser.add_argument("--mesh-dir", type=str, default="eit_meshes", help="mesh cache directory")
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="mesh_16e_r0p025_ref10_cov0p5",
        help="mesh cache name",
    )
    parser.add_argument("--n-elec", type=int, default=16, help="number of electrodes")
    parser.add_argument("--radius", type=float, default=0.025, help="mesh radius (m)")

    args = parser.parse_args()
    if args.input_dir is None and not args.csv:
        parser.error("Provide --input-dir or --csv.")
    if args.input_mode == "frame":
        if args.reference_csv is not None and args.reference_index is not None:
            parser.error("Use only one of --reference-csv or --reference-index.")
        if args.reference_csv is None and args.reference_index is None:
            parser.error("Frame mode requires --reference-csv or --reference-index.")
    return args


def main() -> None:
    args = _parse_args()
    input_files = _collect_csv_files(args.input_dir, args.glob, args.csv, args.include_ad)
    if not input_files:
        raise SystemExit("No CSV files found for batch run.")

    reference_path = None
    target_files = input_files
    if args.input_mode == "frame":
        if args.reference_index is not None:
            if args.reference_index < 0 or args.reference_index >= len(input_files):
                raise SystemExit("reference-index is out of range for the input list.")
            reference_path = input_files[args.reference_index]
        else:
            reference_path = args.reference_csv.expanduser() if args.reference_csv else None
            if reference_path is None or not reference_path.exists():
                raise SystemExit("reference-csv does not exist.")
        target_files = [p for p in input_files if p.resolve() != reference_path.resolve()]
        if not target_files:
            raise SystemExit("No target CSV files found after removing reference.")

    if args.dry_run:
        if args.input_mode == "frame" and reference_path is not None:
            print(f"reference: {reference_path}")
        for path in target_files:
            print(path)
        return

    ctx = _prepare_shared_context(
        mesh_dir=args.mesh_dir,
        mesh_name=args.mesh_name,
        n_elec=args.n_elec,
        radius=args.radius,
        pattern_amplitude=args.pattern_amplitude,
        contact_impedance=args.contact_impedance,
        background_sigma=args.background_sigma,
        lam=args.lam,
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.input_mode == "frame" and reference_path is not None:
        ref_info = output_root / "reference_frame.txt"
        ref_info.write_text(str(reference_path.resolve()) + "\n", encoding="utf-8")

    total = len(target_files)
    processed = 0
    skipped = 0
    failed = 0

    if args.input_mode == "frame":
        expected_len = ctx["n_meas_total"]
        ref_frame = _load_frame_csv(
            reference_path,
            expected_len=expected_len,
            n_stim=ctx["n_stim"],
            n_meas_per_stim=ctx["n_meas_per_stim"],
            measurement_gain=args.measurement_gain,
            layout=args.frame_layout,
            use_part=args.use_part,
        )
        ref_frame, flipped = align_measurement_polarity(ref_frame, ctx["base_meas"])
        if flipped:
            print(f"[INFO] Polarity flip in reference: {reference_path.name}")

        for idx, csv_path in enumerate(target_files, start=1):
            output_dir = output_root / csv_path.stem
            outputs_file = output_dir / "outputs.npz"
            if outputs_file.exists() and not args.overwrite:
                skipped += 1
                print(f"[SKIP] {csv_path.name} (exists)")
                continue
            print(f"[INFO] ({idx}/{total}) {csv_path.name}")
            try:
                target_frame = _load_frame_csv(
                    csv_path,
                    expected_len=expected_len,
                    n_stim=ctx["n_stim"],
                    n_meas_per_stim=ctx["n_meas_per_stim"],
                    measurement_gain=args.measurement_gain,
                    layout=args.frame_layout,
                    use_part=args.use_part,
                )
                target_frame, flipped = align_measurement_polarity(target_frame, ctx["base_meas"])
                if flipped:
                    print(f"[INFO] Polarity flip in {csv_path.name}")
                rmse_abs = _process_frames(
                    vh=ref_frame,
                    vi=target_frame,
                    output_dir=output_dir,
                    ctx=ctx,
                    step_size_calib=args.step_size_calibration,
                    step_size_min=args.step_size_min,
                    step_size_max=args.step_size_max,
                    step_size_maxiter=args.step_size_maxiter,
                    lam=args.lam,
                    colormap=args.colormap,
                    colorbar_scientific=args.colorbar_scientific,
                    colorbar_format=args.colorbar_format,
                    transparent=args.transparent,
                    write_plots=not args.no_plots,
                    measurement_gain=args.measurement_gain,
                )
                print(f"[INFO] {csv_path.name}: RMSE(abs)={rmse_abs:.5f}")
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"[ERROR] {csv_path.name}: {exc}")
                if not args.continue_on_error:
                    raise
    else:
        for idx, csv_path in enumerate(target_files, start=1):
            output_dir = output_root / csv_path.stem
            outputs_file = output_dir / "outputs.npz"
            if outputs_file.exists() and not args.overwrite:
                skipped += 1
                print(f"[SKIP] {csv_path.name} (exists)")
                continue
            print(f"[INFO] ({idx}/{total}) {csv_path.name}")
            try:
                _run_single_file(
                    csv_path=csv_path,
                    output_dir=output_dir,
                    ctx=ctx,
                    use_part=args.use_part,
                    measurement_gain=args.measurement_gain,
                    step_size_calib=args.step_size_calibration,
                    step_size_min=args.step_size_min,
                    step_size_max=args.step_size_max,
                    step_size_maxiter=args.step_size_maxiter,
                    lam=args.lam,
                    colormap=args.colormap,
                    colorbar_scientific=args.colorbar_scientific,
                    colorbar_format=args.colorbar_format,
                    transparent=args.transparent,
                    write_plots=not args.no_plots,
                )
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"[ERROR] {csv_path.name}: {exc}")
                if not args.continue_on_error:
                    raise

    print(
        f"[DONE] processed={processed}, skipped={skipped}, failed={failed}, total={total}"
    )


if __name__ == "__main__":
    main()
