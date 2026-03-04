"""GN single-step difference reconstruction utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize_scalar

from pyeidors.cache import CacheManager, CachePolicy, hash_array
from pyeidors.cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
)
from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian
from pyeidors.utils.numeric_ops import safe_dot
from pyeidors.visualization import create_visualizer

from .mesh_utils import cell_to_node

mpl.rcParams.update(
    {
        "axes.unicode_minus": False,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    }
)


def _build_noser_matrix(
    jacobian: np.ndarray,
    *,
    exponent: float = 0.5,
    alpha: float = 1.0,
    adaptive_floor: bool = True,
    floor: float = 1e-12,
    floor_fraction: float = 1e-6,
) -> np.ndarray:
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


def _build_operator_bundle(jacobian: np.ndarray, lam: float) -> dict:
    reg_matrix = _build_noser_matrix(jacobian, exponent=0.5, alpha=1.0)
    jacobian_t = jacobian.T
    A = np.asarray(
        safe_dot(jacobian_t, jacobian, "gn_difference.operator.JtJ"),
        dtype=float,
    ) + lam * reg_matrix
    try:
        lu, piv = lu_factor(A)
        factor = {"method": "lu", "lu": lu, "piv": piv}
    except Exception:
        factor = {"method": "none"}
    return {
        "Jt": jacobian_t,
        "A": A,
        "reg_matrix": reg_matrix,
        "factor": factor,
    }


def _solve_linear_from_bundle(bundle: dict, b: np.ndarray) -> np.ndarray:
    factor = bundle.get("factor", {})
    if factor.get("method") == "lu":
        try:
            return np.asarray(lu_solve((factor["lu"], factor["piv"]), b), dtype=float)
        except Exception:
            pass
    return _solve_linear(bundle["A"], b, None)


def _to_lookup_payload(lookup) -> dict[str, object]:
    return {
        "hit": bool(lookup.hit),
        "layer": str(lookup.layer),
        "artifact": str(lookup.artifact),
        "key": str(lookup.key),
    }


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
        print(
            f"[INFO] Step-size calibration: alpha={result.x:.3g}, diff residual={result.fun:.3e}"
        )
        return float(result.x)

    print("[WARN] Step-size calibration failed, fallback alpha=1.0")
    return 1.0


def build_shared_context(
    *,
    mesh_dir: str,
    mesh_name: Optional[str],
    n_elec: int,
    radius: float,
    drive_value: Optional[float],
    contact_impedance: float,
    background_sigma: float,
    lam: float,
    cache_scope: str = "both",
    cache_dir: str = ".pyeidors_cache/v2",
    cache_clear_names: Optional[list[str]] = None,
) -> dict:
    stim_drive_value = drive_value if drive_value is not None else 1.0
    print(f"[INFO] Diff imaging drive_mode=normalized, drive_value={stim_drive_value:.2e}")

    cache_manager = CacheManager(
        scope=cache_scope,
        cache_dir=cache_dir,
        policy=CachePolicy(),
    )
    if cache_clear_names:
        for name in cache_clear_names:
            cache_manager.clear_name(name=name)

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
        drive_mode="normalized",
        drive_value=stim_drive_value,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    z_contact = np.full(n_elec, contact_impedance, dtype=float)
    fwd_model = EITForwardModel(
        n_elec=n_elec,
        pattern_config=pattern_cfg,
        z=z_contact,
        mesh=mesh,
        cache_manager=cache_manager,
    )

    n_elem = int(
        fwd_model.V_sigma.dofmap.index_map.size_local
        * fwd_model.V_sigma.dofmap.index_map_bs
    )
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
    sigma_hash = hashlib.sha256(
        np.ascontiguousarray(sigma_bg, dtype=np.float64).tobytes()
    ).hexdigest()
    jacobian_payload = {
        "solver": "gn_difference",
        "method": "adjoint",
        "sigma_hash": sigma_hash,
        "model_signature": model_signature_from_forward_model(fwd_model),
        "pattern_signature": pattern_signature_from_forward_model(fwd_model),
        "backend_signature": backend_signature_from_forward_model(fwd_model),
    }
    jacobian, jacobian_lookup = cache_manager.get_or_compute_semantic(
        artifact="jacobian",
        name="calc_jacobian",
        namespace="difference",
        cache_obj=jacobian_payload,
        payload=jacobian_payload,
        compute_fn=lambda: jac_calc.calculate_from_image(img_bg),
        persist=True,
        cost=12.0,
        effort_seconds=8.0,
    )

    operator_payload = {
        "solver": "gn_difference",
        "sigma_hash": sigma_hash,
        "jacobian_hash": hash_array(np.ascontiguousarray(jacobian, dtype=np.float64)),
        "lambda": float(lam),
        "model_signature": jacobian_payload["model_signature"],
        "pattern_signature": jacobian_payload["pattern_signature"],
        "backend_signature": jacobian_payload["backend_signature"],
    }
    operator_bundle, operator_lookup = cache_manager.get_or_compute_semantic(
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cache_obj=operator_payload,
        payload=operator_payload,
        compute_fn=lambda: _build_operator_bundle(jacobian, lam),
        persist=True,
        cost=16.0,
        effort_seconds=10.0,
    )

    return {
        "mesh": mesh,
        "fwd_model": fwd_model,
        "cache_manager": cache_manager,
        "cache_scope": cache_scope,
        "sigma_bg": sigma_bg,
        "img_bg": img_bg,
        "base_meas": base_meas,
        "n_stim": n_stim,
        "n_meas_total": n_meas_total,
        "n_meas_per_stim": n_meas_per_stim,
        "J": jacobian,
        "operator_bundle": operator_bundle,
        "stim_drive_value": stim_drive_value,
        "cache_lookups": {
            "jacobian": _to_lookup_payload(jacobian_lookup),
            "single_step_operator": _to_lookup_payload(operator_lookup),
            "forward_factor": dict(getattr(fwd_model, "_last_cache_lookup", {})),
        },
    }


def process_frames(
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
) -> dict[str, object]:
    dv = vi - vh
    if dv.shape[0] != ctx["J"].shape[0]:
        raise RuntimeError(
            f"Data length {dv.shape[0]} does not match Jacobian rows {ctx['J'].shape[0]}"
        )

    b = np.asarray(
        safe_dot(ctx["operator_bundle"]["Jt"], dv, "gn_difference.operator.Jt_dv"),
        dtype=float,
    )
    delta_sigma = _solve_linear_from_bundle(ctx["operator_bundle"], b)

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
        format_mode = colorbar_format or (
            "scientific" if colorbar_scientific else "plain"
        )
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
        drive_value=ctx["stim_drive_value"],
        measurement_gain=measurement_gain,
    )
    cache_manager = ctx.get("cache_manager")
    cache_stats = cache_manager.stats() if cache_manager is not None else {}
    forward_lookup = dict(getattr(ctx["fwd_model"], "_last_cache_lookup", {}))
    return {
        "rmse_abs": rmse_abs,
        "step_size_alpha": float(alpha),
        "cache_lookups": {
            "context": dict(ctx.get("cache_lookups", {})),
            "forward_factor": forward_lookup,
        },
        "cache_stats": cache_stats,
    }
