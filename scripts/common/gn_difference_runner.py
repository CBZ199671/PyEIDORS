"""GN single-step difference reconstruction utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize_scalar
from scipy import sparse
from scipy.sparse.linalg import cg, lsmr

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None

try:  # pragma: no cover - optional dependency
    import pyamg
except Exception:  # pragma: no cover
    pyamg = None

from pyeidors.cache import CacheManager, CachePolicy, hash_array
from pyeidors.cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
)
from pyeidors.data.difference import build_difference_vector, project_measurement_jacobian
from pyeidors.data.structures import PatternConfig, EITImage
from pyeidors.electrodes.layout import effective_pattern_layout_for_3d_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsStyleAdjointJacobian
from pyeidors.perf.capabilities import detect_performance_capabilities, select_preconditioner
from pyeidors.perf.policy import (
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_FORWARD_BACKEND,
    DEFAULT_MESH_FAMILY,
    normalize_forward_backend,
    normalize_mesh_family,
)
from pyeidors.inverse.reduced.lowrank_subspace import build_lowrank_subspace
from pyeidors.inverse.reduced.pod_basis import compute_pod_basis, merge_orthonormal_bases
from pyeidors.inverse.reduced.snapshot_bank import select_snapshot_matrix
from pyeidors.inverse.solvers.gauss_newton_device import resolve_torch_device
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

CACHE_NAMESPACE_DIFFERENCE = "difference"
CACHE_NAME_JACOBIAN = "calc_jacobian"
CACHE_NAME_OPERATOR_JT = "gn_diff_operator_jt"
CACHE_NAME_OPERATOR_NOSER = "gn_diff_operator_noser"
CACHE_NAME_OPERATOR_A = "gn_diff_operator_system"
CACHE_NAME_OPERATOR_LU = "gn_diff_operator_lu"
CACHE_NAME_OPERATOR_PRECOND = "gn_diff_operator_precond"
CACHE_NAME_OPERATOR_REDUCED_RM = "gn_diff_operator_reduced_rm"


def _mesh_compatible_drive_mode(drive_mode: str | None, *, mesh_dim: int) -> str:
    if drive_mode is None:
        return "normalized" if int(mesh_dim) == 2 else "total_current"
    resolved = str(drive_mode).strip().lower()
    if int(mesh_dim) == 3 and resolved == "line_current_density":
        return "total_current"
    return resolved or ("normalized" if int(mesh_dim) == 2 else "total_current")


CACHE_NAME_BASE_MEAS = "gn_diff_base_meas"

STRICT_SOLVER_BACKEND_DENSE = "dense-param"
STRICT_SOLVER_BACKEND_MEASUREMENT = "measurement-exact"
STRICT_MEMORY_FALLBACK_BYTES = 8 * 1024**3
STRICT_MEMORY_GUARD_CAP_BYTES = 12 * 1024**3
STRICT_MEMORY_GUARD_FRACTION = 0.60


def _build_noser_diag(
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
    return np.asarray(alpha * scaled_diag, dtype=np.float64)


def _linux_mem_available_bytes() -> int | None:
    meminfo = Path("/proc/meminfo")
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _estimate_strict_dense_peak_bytes(n_param: int) -> float:
    return 4.0 * float(max(1, n_param)) * float(max(1, n_param)) * 8.0


def _select_strict_solver_backend(
    *,
    mesh_dim: int,
    n_param: int,
    n_meas: int,
    mem_available_bytes: int | None = None,
) -> dict[str, object]:
    mem_available = int(mem_available_bytes) if mem_available_bytes is not None else _linux_mem_available_bytes()
    mem_available_source = "linux-memavailable" if mem_available_bytes is None and mem_available is not None else "provided"
    if mem_available is None:
        mem_available = int(STRICT_MEMORY_FALLBACK_BYTES)
        mem_available_source = "fallback"
    estimated_peak_bytes = float(_estimate_strict_dense_peak_bytes(int(n_param)))
    guard_limit_bytes = float(min(float(mem_available) * STRICT_MEMORY_GUARD_FRACTION, float(STRICT_MEMORY_GUARD_CAP_BYTES)))
    triggered = bool(int(mesh_dim) == 3 and estimated_peak_bytes > guard_limit_bytes)
    effective = STRICT_SOLVER_BACKEND_MEASUREMENT if triggered else STRICT_SOLVER_BACKEND_DENSE
    if int(mesh_dim) != 3:
        reason = "dense_allowed_non3d"
    elif triggered:
        reason = "dense_estimate_exceeds_guard"
    else:
        reason = "dense_within_guard"
    return {
        "requested": STRICT_SOLVER_BACKEND_DENSE,
        "effective": effective,
        "strict_memory_guard_triggered": triggered,
        "strict_memory_guard_reason": reason,
        "strict_dense_estimated_peak_bytes": float(estimated_peak_bytes),
        "strict_dense_estimated_peak_gib": float(estimated_peak_bytes / (1024.0**3)),
        "strict_memory_guard_limit_bytes": float(guard_limit_bytes),
        "strict_memory_guard_limit_gib": float(guard_limit_bytes / (1024.0**3)),
        "strict_mem_available_bytes": int(mem_available),
        "strict_mem_available_gib": float(mem_available / (1024.0**3)),
        "strict_mem_available_source": mem_available_source,
        "strict_measurement_system_shape": [int(n_meas), int(n_meas)] if triggered else None,
    }


def _measurement_space_delta(
    *,
    operator_bundle: dict,
    rhs: np.ndarray,
) -> np.ndarray:
    runtime_device = str(operator_bundle.get("device_effective", "cpu"))
    torch_device = str(operator_bundle.get("torch_device", "cuda"))
    y = _solve_linear_from_bundle(operator_bundle, np.asarray(rhs, dtype=float))
    if runtime_device == "cuda" and torch is not None:
        inv_reg_diag_t = _bundle_torch_tensor(operator_bundle, "inv_reg_diag", operator_bundle["inv_reg_diag"])
        Jt_t = _bundle_torch_tensor(operator_bundle, "Jt", operator_bundle["Jt"])
        y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), device=torch_device, dtype=torch.float64)
        return np.asarray((inv_reg_diag_t * torch.mv(Jt_t, y_t)).detach().cpu().numpy(), dtype=float)
    return np.asarray(
        operator_bundle["inv_reg_diag"]
        * safe_dot(
            operator_bundle["Jt"],
            y,
            "gn_difference.operator.strict_measurement_RinvJt_y",
        ),
        dtype=float,
    )


def _make_linear_solver(A: np.ndarray) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    try:
        lu, piv = lu_factor(A)
    except Exception:
        return None

    def _solve(b: np.ndarray) -> np.ndarray:
        return lu_solve((lu, piv), b)

    return _solve


def _factorize_matrix(A: np.ndarray) -> dict:
    try:
        lu, piv = lu_factor(A)
        return {"method": "lu", "lu": lu, "piv": piv}
    except Exception:
        return {"method": "none"}


def _solve_linear_from_bundle(bundle: dict, b: np.ndarray) -> np.ndarray:
    device_effective = str(bundle.get("device_effective", "cpu"))
    torch_device = str(bundle.get("torch_device", "cuda"))
    if device_effective == "cuda" and torch is not None:
        return _solve_linear_torch(bundle["A"], b, device=torch_device)
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


def _lookup_miss_reason(lookup, *, family: str) -> str:
    if bool(lookup.hit):
        return "hit"
    if str(lookup.layer) == "disabled":
        return "cache_disabled"
    if family == CACHE_NAME_JACOBIAN:
        return "sigma_hash_changed"
    if family in {
        CACHE_NAME_OPERATOR_JT,
        CACHE_NAME_OPERATOR_NOSER,
        CACHE_NAME_OPERATOR_A,
        CACHE_NAME_OPERATOR_PRECOND,
        CACHE_NAME_OPERATOR_LU,
        CACHE_NAME_OPERATOR_REDUCED_RM,
    }:
        return "solver_config_changed"
    return "mesh_signature_changed"


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


def _bundle_torch_tensor(bundle: dict, name: str, values: np.ndarray):
    cache = bundle.setdefault("torch_cache", {})
    device = str(bundle.get("torch_device", "cuda"))
    tensor = cache.get(name)
    if tensor is None:
        tensor = torch.as_tensor(np.asarray(values, dtype=np.float64), device=device, dtype=torch.float64)
        cache[name] = tensor
    return tensor


def _solve_linear_torch_cg(
    A_t: "torch.Tensor",
    b_t: "torch.Tensor",
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
    max_iter: int | None = None,
):
    if max_iter is None:
        max_iter = max(512, min(int(A_t.shape[0]) * 4, 8192))
    diag = torch.diagonal(A_t)
    safe_diag = torch.where(torch.abs(diag) > 1e-18, diag, torch.ones_like(diag))

    def _solve_one(rhs_t: "torch.Tensor"):
        x_t = torch.zeros_like(rhs_t)
        r_t = rhs_t.clone()
        z_t = r_t / safe_diag
        p_t = z_t.clone()
        rz_old = torch.dot(r_t, z_t)
        rhs_norm = float(torch.linalg.vector_norm(rhs_t).item())
        tol = max(float(atol), float(rtol) * max(rhs_norm, 1e-18))
        for _ in range(int(max_iter)):
            Ap_t = torch.mv(A_t, p_t)
            denom = torch.dot(p_t, Ap_t)
            if not torch.isfinite(denom) or torch.abs(denom) <= 1e-30:
                break
            alpha = rz_old / denom
            x_t = x_t + alpha * p_t
            r_t = r_t - alpha * Ap_t
            if float(torch.linalg.vector_norm(r_t).item()) <= tol:
                return x_t
            z_t = r_t / safe_diag
            rz_new = torch.dot(r_t, z_t)
            if not torch.isfinite(rz_new):
                break
            beta = rz_new / rz_old
            p_t = z_t + beta * p_t
            rz_old = rz_new
        raise RuntimeError("Torch CUDA CG fallback did not converge")

    if b_t.ndim == 1:
        return _solve_one(b_t)
    return torch.stack([_solve_one(b_t[:, idx]) for idx in range(int(b_t.shape[1]))], dim=1)


def _solve_linear_torch(A: np.ndarray, b: np.ndarray, *, device: str) -> np.ndarray:
    if torch is None:
        raise RuntimeError("Torch runtime is unavailable for CUDA linear solve")
    A_t = torch.as_tensor(np.asarray(A, dtype=np.float64), device=device, dtype=torch.float64)
    b_t = torch.as_tensor(np.asarray(b, dtype=np.float64), device=device, dtype=torch.float64)
    try:
        x_t = torch.linalg.solve(A_t, b_t)
    except Exception as exc:
        message = str(exc).lower()
        runtime_linalg_unavailable = any(
            token in message for token in ("libtorch_cuda_linalg", "cusolver", "undefined symbol", "dlopen")
        )
        if runtime_linalg_unavailable and device.startswith("cuda"):
            x_t = _solve_linear_torch_cg(
                A_t,
                b_t,
                rtol=1e-12,
                atol=1e-14,
                max_iter=max(2048, min(int(A_t.shape[0]) * 4, 8192)),
            )
        else:
            rhs = b_t.unsqueeze(1) if b_t.ndim == 1 else b_t
            x_t = torch.linalg.lstsq(A_t, rhs).solution
            if b_t.ndim == 1:
                x_t = x_t.squeeze(-1)
    if device.startswith("cuda") and hasattr(torch.cuda, "synchronize"):
        torch.cuda.synchronize(device)
    return np.asarray(x_t.detach().cpu().numpy(), dtype=np.float64)


def _build_reduced_rm(
    *,
    jacobian: np.ndarray,
    reg_diag: np.ndarray,
    lam: float,
    basis: np.ndarray,
) -> dict[str, np.ndarray]:
    j_mat = np.asarray(jacobian, dtype=np.float64)
    b_mat = np.asarray(basis, dtype=np.float64)
    r_diag = np.asarray(reg_diag, dtype=np.float64)

    ju = np.asarray(safe_dot(j_mat, b_mat, "gn_difference.rom.JU"), dtype=np.float64)
    rb = np.asarray(r_diag[:, None] * b_mat, dtype=np.float64)
    h_red = np.asarray(
        safe_dot(ju.T, ju, "gn_difference.rom.JUtJU")
        + float(lam) * safe_dot(b_mat.T, rb, "gn_difference.rom.UtRU"),
        dtype=np.float64,
    )
    h_red = 0.5 * (h_red + h_red.T)
    try:
        h_inv = np.linalg.inv(h_red)
    except Exception:
        h_inv = np.linalg.pinv(h_red)
    rm_reduced = np.asarray(
        safe_dot(
            b_mat,
            safe_dot(h_inv, ju.T, "gn_difference.rom.HinvJUt"),
            "gn_difference.rom.U_HinvJUt",
        ),
        dtype=np.float64,
    )
    return {
        "basis": b_mat,
        "JU": ju,
        "H": h_red,
        "H_inv": np.asarray(h_inv, dtype=np.float64),
        "RM_reduced": rm_reduced,
    }


def _solve_measurement_space(
    *,
    system_matrix: np.ndarray,
    rhs: np.ndarray,
    linear_solver: str,
    preconditioner_mode: str = "auto",
    preconditioner: Optional[np.ndarray] = None,
    runtime_device: str = "cpu",
    torch_device: str = "cuda",
) -> np.ndarray:
    """Solve measurement-space system in fast mode."""
    solver = str(linear_solver)
    A = np.asarray(system_matrix, dtype=np.float64)
    b = np.asarray(rhs, dtype=np.float64)
    preconditioner_choice = select_preconditioner(
        preconditioner_mode,
        capabilities=detect_performance_capabilities(),
    )

    if str(runtime_device) == "cuda" and torch is not None:
        return _solve_linear_torch(A, b, device=str(torch_device))

    if solver in {"auto", "petsc-ksp", "cholmod"}:
        try:
            return np.asarray(np.linalg.solve(A, b), dtype=np.float64)
        except Exception:
            return np.asarray(np.linalg.lstsq(A, b, rcond=None)[0], dtype=np.float64)

    if solver == "scipy-lsmr":
        return np.asarray(lsmr(A, b, atol=1e-8, btol=1e-8, maxiter=max(200, A.shape[0] * 4))[0], dtype=np.float64)

    if solver == "pyamg-cg" or (solver == "auto" and preconditioner_choice == "pyamg"):
        mat = sparse.csr_matrix(A)
        if pyamg is not None:
            ml = pyamg.smoothed_aggregation_solver(mat)
            M = ml.aspreconditioner(cycle="V")
            x, info = cg(mat, b, M=M, rtol=1e-8, maxiter=max(200, A.shape[0] * 4))
            if info == 0:
                return np.asarray(x, dtype=np.float64)
        if preconditioner is not None:
            pinv = np.asarray(preconditioner, dtype=np.float64)
            M = sparse.linalg.LinearOperator(
                mat.shape,
                matvec=lambda v: np.asarray(v, dtype=np.float64) / pinv,
                dtype=np.float64,
            )
            x, info = cg(mat, b, M=M, rtol=1e-8, maxiter=max(200, A.shape[0] * 4))
            if info == 0:
                return np.asarray(x, dtype=np.float64)

    if solver == "auto" and preconditioner is not None and preconditioner_choice in {
        "diag",
        "noser",
        "prior",
        "pmat",
        "coarse",
        "custom",
    }:
        mat = sparse.csr_matrix(A)
        pinv = np.asarray(preconditioner, dtype=np.float64)
        M = sparse.linalg.LinearOperator(
            mat.shape,
            matvec=lambda v: np.asarray(v, dtype=np.float64) / pinv,
            dtype=np.float64,
        )
        x, info = cg(mat, b, M=M, rtol=1e-8, maxiter=max(200, A.shape[0] * 4))
        if info == 0:
            return np.asarray(x, dtype=np.float64)

    try:
        return np.asarray(np.linalg.solve(A, b), dtype=np.float64)
    except Exception:
        return np.asarray(np.linalg.lstsq(A, b, rcond=None)[0], dtype=np.float64)


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
    difference_mode: str = "raw",
    difference_orientation: str = "target_minus_reference",
) -> float:
    def _objective(scale: float) -> float:
        sigma_try = sigma_bg + scale * delta_sigma
        img_try = EITImage(elem_data=sigma_try, fwd_model=fwd_model)
        pred_vi_try, _ = fwd_model.fwd_solve(img_try)
        pred_diff_try = build_difference_vector(
            pred_vi_try.meas,
            base_meas,
            mode=difference_mode,
            orientation=difference_orientation,
        )
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
    mesh_dim: int,
    mesh_height: float,
    electrode_height_ratio: float,
    z_center: float,
    electrode_level_fractions: tuple[float, ...] | list[float] | None = None,
    refinement: Optional[int],
    n_elec: int,
    n_rings: int = 1,
    radius: float = 1.0,
    drive_mode: str | None = None,
    drive_value: Optional[float] = None,
    contact_impedance: float = 0.01,
    electrode_length_m_override: float | list[float] | None = None,
    electrode_coverage: float = 0.5,
    geometry_scale_to_m: float = 1.0,
    stim_pattern: str = "{ad}",
    meas_pattern: str = "{ad}",
    electrode_layout: str = "ring_major",
    measurement_protocol: str = "eidors_full_3d",
    custom_stim_matrix: object | None = None,
    custom_meas_matrices: object | None = None,
    rotate_meas: bool = True,
    use_meas_current: bool = False,
    use_meas_current_next: int = 0,
    stim_direction: str = "ccw",
    meas_direction: str = "ccw",
    stim_first_positive: bool = False,
    difference_mode: str = "raw",
    difference_orientation: str = "target_minus_reference",
    background_sigma: float = 1.0,
    lam: float = 1.0e-2,
    cache_scope: str = "both",
    cache_dir: str = ".pyeidors_cache/v2",
    cache_clear_names: Optional[list[str]] = None,
    solver_mode: str = "strict",
    linear_solver: str = "auto",
    preconditioner: str = "auto",
    rom_mode: str = "off",
    rom_rank_global: int = 32,
    rom_rank_adaptive: int = 16,
    rom_snapshot_source: str = "hybrid",
    lowrank_mode: str = "off",
    lowrank_rank: int = 16,
    lowrank_method: str = "tsvd",
    lowrank_energy: float = 0.995,
    forward_mat_solve: str = "off",
    petsc_device: str = "auto",
    device: str = "auto",
    forward_backend: str = DEFAULT_FORWARD_BACKEND,
    mesh_family: str = DEFAULT_MESH_FAMILY,
    geometry_version: str = DEFAULT_3D_GEOMETRY_VERSION,
) -> dict:
    context_start = time.perf_counter()
    build_seconds: dict[str, float] = {}
    if int(mesh_dim) not in {2, 3}:
        raise ValueError(f"mesh_dim must be 2 or 3, got {mesh_dim!r}")
    solver_mode = str(solver_mode).strip().lower()
    linear_solver = str(linear_solver).strip().lower()
    preconditioner = str(preconditioner).strip().lower()
    rom_mode = str(rom_mode).strip().lower()
    lowrank_mode = str(lowrank_mode).strip().lower()
    lowrank_method = str(lowrank_method).strip().lower()
    rom_snapshot_source = str(rom_snapshot_source).strip().lower()
    forward_mat_solve = str(forward_mat_solve).strip().lower()
    petsc_device = str(petsc_device).strip().lower()
    device = str(device).strip().lower()
    forward_backend = normalize_forward_backend(
        forward_backend,
        default=DEFAULT_FORWARD_BACKEND,
    )
    mesh_family = normalize_mesh_family(
        mesh_family,
        default=DEFAULT_MESH_FAMILY,
    )
    geometry_version = str(geometry_version).strip().lower() or DEFAULT_3D_GEOMETRY_VERSION
    if solver_mode not in {"strict", "fast"}:
        raise ValueError(f"solver_mode must be 'strict' or 'fast', got {solver_mode!r}")
    if preconditioner not in {
        "auto",
        "diag",
        "noser",
        "prior",
        "pmat",
        "coarse",
        "custom",
        "pyamg",
        "cholmod",
        "petsc-gamg",
    }:
        raise ValueError(f"preconditioner is invalid: {preconditioner!r}")
    if rom_mode not in {"off", "auto", "on"}:
        raise ValueError(f"rom_mode is invalid: {rom_mode!r}")
    if lowrank_mode not in {"off", "auto", "on"}:
        raise ValueError(f"lowrank_mode is invalid: {lowrank_mode!r}")
    if lowrank_method not in {"tsvd", "randomized"}:
        raise ValueError(f"lowrank_method is invalid: {lowrank_method!r}")
    if rom_snapshot_source not in {"cache", "synthetic", "hybrid"}:
        raise ValueError(f"rom_snapshot_source is invalid: {rom_snapshot_source!r}")
    if int(rom_rank_global) <= 0:
        raise ValueError("rom_rank_global must be positive")
    if int(rom_rank_adaptive) < 0:
        raise ValueError("rom_rank_adaptive must be >= 0")
    if int(lowrank_rank) <= 0:
        raise ValueError("lowrank_rank must be positive")
    if not (0.0 < float(lowrank_energy) <= 1.0):
        raise ValueError("lowrank_energy must be in (0, 1]")
    if forward_mat_solve not in {"auto", "off", "on"}:
        raise ValueError(f"forward_mat_solve must be auto|off|on, got {forward_mat_solve!r}")
    if petsc_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"petsc_device must be auto|cpu|cuda, got {petsc_device!r}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"device must be auto|cpu|cuda, got {device!r}")
    stim_drive_value = drive_value if drive_value is not None else 1.0
    resolved_drive_mode = _mesh_compatible_drive_mode(drive_mode, mesh_dim=int(mesh_dim))
    print(
        f"[INFO] Diff imaging drive_mode={resolved_drive_mode}, "
        f"drive_value={stim_drive_value:.2e}"
    )

    cache_manager = CacheManager(
        scope=cache_scope,
        cache_dir=cache_dir,
        policy=CachePolicy(),
    )
    if cache_clear_names:
        for name in cache_clear_names:
            cache_manager.clear_name(name=name)

    total_electrodes = max(int(n_elec), 1) * max(int(n_rings), 1)
    pattern_n_elec, pattern_n_rings = effective_pattern_layout_for_3d_mesh(
        mesh_tdim=int(mesh_dim),
        n_elec=int(n_elec),
        n_rings=int(n_rings),
        electrode_layout=electrode_layout,
    )

    mesh_start = time.perf_counter()
    mesh = load_or_create_mesh(
        mesh_dir=mesh_dir,
        mesh_name=mesh_name,
        n_elec=total_electrodes,
        dimension=int(mesh_dim),
        radius=radius,
        refinement=refinement if refinement is not None else 6,
        electrode_coverage=float(electrode_coverage),
        height=float(mesh_height),
        electrode_height_ratio=float(electrode_height_ratio),
        electrode_level_fractions=electrode_level_fractions or (0.25, 0.75),
        z_center=float(z_center),
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        electrode_layout=electrode_layout,
    )
    build_seconds["mesh"] = time.perf_counter() - mesh_start
    mesh_cache_hit = getattr(mesh, "_pyeidors_mesh_cache_hit", None)
    mesh_cache_layer = getattr(mesh, "_pyeidors_mesh_cache_layer", None)
    mesh_cache_name = getattr(mesh, "_pyeidors_mesh_cache_name", None)
    pattern_cfg = PatternConfig(
        n_elec=pattern_n_elec,
        n_rings=pattern_n_rings,
        stim_pattern=stim_pattern,
        meas_pattern=meas_pattern,
        electrode_layout=electrode_layout,
        measurement_protocol=measurement_protocol,
        custom_stim_matrix=custom_stim_matrix,
        custom_meas_matrices=custom_meas_matrices,
        drive_mode=resolved_drive_mode,
        drive_value=stim_drive_value,
        geometry_scale_to_m=float(geometry_scale_to_m),
        electrode_length_m_override=electrode_length_m_override,
        use_meas_current=bool(use_meas_current),
        use_meas_current_next=int(use_meas_current_next),
        rotate_meas=bool(rotate_meas),
        stim_direction=str(stim_direction),
        meas_direction=str(meas_direction),
        stim_first_positive=bool(stim_first_positive),
    )
    z_contact = np.full(total_electrodes, contact_impedance, dtype=float)
    fwd_model = EITForwardModel(
        n_elec=total_electrodes,
        pattern_config=pattern_cfg,
        z=z_contact,
        mesh=mesh,
        cache_manager=cache_manager,
        backend_config={"mat_solve_mode": forward_mat_solve, "petsc_device": petsc_device},
        forward_backend=forward_backend,
    )
    petsc_backend_info = dict(getattr(fwd_model, "_petsc_backend_info", {}) or {})
    runtime_selection = resolve_torch_device(
        device,
        verbose=False,
        petsc_device_effective=str(petsc_backend_info.get("petsc_device_effective", "cpu")),
    )

    n_elem = int(
        fwd_model.V_sigma.dofmap.index_map.size_local
        * fwd_model.V_sigma.dofmap.index_map_bs
    )
    sigma_bg = np.full(n_elem, background_sigma)
    img_bg = EITImage(elem_data=sigma_bg, fwd_model=fwd_model)
    print(f"[INFO] Background conductivity: {background_sigma}")

    def _timed(name: str, fn: Callable[[], np.ndarray | dict]) -> np.ndarray | dict:
        start = time.perf_counter()
        value = fn()
        build_seconds[name] = time.perf_counter() - start
        return value

    base_meas_payload = {
        "solver": "gn_difference",
        "part": "base_meas",
        "mesh_dim": int(mesh_dim),
        "mesh_height": float(mesh_height),
        "electrode_height_ratio": float(electrode_height_ratio),
        "z_center": float(z_center),
        "background_sigma": float(background_sigma),
        "mesh_family": str(mesh_family),
        "geometry_version": str(geometry_version),
        "model_signature": model_signature_from_forward_model(fwd_model),
        "pattern_signature": pattern_signature_from_forward_model(fwd_model),
        "backend_signature": backend_signature_from_forward_model(fwd_model),
    }

    def _compute_base_meas() -> np.ndarray:
        base_forward, _ = fwd_model.fwd_solve(img_bg)
        return np.asarray(base_forward.meas, dtype=np.float64)

    base_meas, base_meas_lookup = cache_manager.get_or_compute_semantic(
        artifact="diff_base_meas",
        name=CACHE_NAME_BASE_MEAS,
        namespace=CACHE_NAMESPACE_DIFFERENCE,
        cache_obj=base_meas_payload,
        payload=base_meas_payload,
        compute_fn=lambda: _timed("base_meas", _compute_base_meas),
        persist=True,
        cost=4.0,
        effort_seconds=2.0,
    )
    base_meas = np.asarray(base_meas, dtype=np.float64)

    pattern_manager = fwd_model.pattern_manager
    n_stim = pattern_manager.n_stim
    n_meas_total = pattern_manager.n_meas_total
    unique_counts = sorted(set(pattern_manager.n_meas_per_stim))
    n_meas_per_stim = unique_counts[0] if len(unique_counts) == 1 else None

    jac_calc = EidorsStyleAdjointJacobian(
        fwd_model,
        use_torch=str(runtime_selection.effective) == "cuda",
        device=str(runtime_selection.torch_device),
        torch_dtype="float64",
        torch_batch_all=str(runtime_selection.effective) == "cuda",
    )
    sigma_hash = hashlib.sha256(
        np.ascontiguousarray(sigma_bg, dtype=np.float64).tobytes()
    ).hexdigest()
    jacobian_payload = {
        "solver": "gn_difference",
        "method": "adjoint",
        "mesh_dim": int(mesh_dim),
        "mesh_height": float(mesh_height),
        "electrode_height_ratio": float(electrode_height_ratio),
        "z_center": float(z_center),
        "sigma_hash": sigma_hash,
        "model_signature": model_signature_from_forward_model(fwd_model),
        "pattern_signature": pattern_signature_from_forward_model(fwd_model),
        "backend_signature": backend_signature_from_forward_model(fwd_model),
    }

    raw_jacobian, jacobian_lookup = cache_manager.get_or_compute_semantic(
        artifact="jacobian",
        name=CACHE_NAME_JACOBIAN,
        namespace=CACHE_NAMESPACE_DIFFERENCE,
        cache_obj=jacobian_payload,
        payload=jacobian_payload,
        compute_fn=lambda: _timed(
            "jacobian",
            lambda: jac_calc.calculate_from_image(img_bg),
        ),
        persist=True,
        cost=12.0,
        effort_seconds=8.0,
    )
    build_seconds.setdefault("jacobian", 0.0)
    jacobian = project_measurement_jacobian(
        raw_jacobian,
        measurement_type="difference",
        reference_meas=base_meas,
        difference_mode=difference_mode,
        difference_orientation=difference_orientation,
    )

    strict_backend_info = _select_strict_solver_backend(
        mesh_dim=int(mesh_dim),
        n_param=int(jacobian.shape[1]),
        n_meas=int(jacobian.shape[0]),
    ) if solver_mode == "strict" else {
        "requested": STRICT_SOLVER_BACKEND_DENSE,
        "effective": STRICT_SOLVER_BACKEND_DENSE,
        "strict_memory_guard_triggered": False,
        "strict_memory_guard_reason": "not_strict",
        "strict_dense_estimated_peak_bytes": 0.0,
        "strict_dense_estimated_peak_gib": 0.0,
        "strict_memory_guard_limit_bytes": 0.0,
        "strict_memory_guard_limit_gib": 0.0,
        "strict_mem_available_bytes": 0,
        "strict_mem_available_gib": 0.0,
        "strict_mem_available_source": "not_strict",
        "strict_measurement_system_shape": None,
    }

    operator_payload_base = {
        "solver": "gn_difference",
        "solver_mode": solver_mode,
        "linear_solver": linear_solver,
        "preconditioner": preconditioner,
        "mesh_dim": int(mesh_dim),
        "mesh_height": float(mesh_height),
        "electrode_height_ratio": float(electrode_height_ratio),
        "z_center": float(z_center),
        "sigma_hash": sigma_hash,
        "jacobian_hash": hash_array(np.ascontiguousarray(jacobian, dtype=np.float64)),
        "lambda": float(lam),
        "model_signature": jacobian_payload["model_signature"],
        "pattern_signature": jacobian_payload["pattern_signature"],
        "backend_signature": jacobian_payload["backend_signature"],
        "difference_mode": str(difference_mode),
        "difference_orientation": str(difference_orientation),
        "base_meas_hash": hash_array(np.ascontiguousarray(base_meas, dtype=np.float64)),
        "strict_solver_backend_effective": str(strict_backend_info.get("effective", STRICT_SOLVER_BACKEND_DENSE)),
    }

    jacobian_t, j_t_lookup = cache_manager.get_or_compute_semantic(
        artifact="single_step_operator",
        name=CACHE_NAME_OPERATOR_JT,
        namespace=CACHE_NAMESPACE_DIFFERENCE,
        cache_obj={**operator_payload_base, "part": "Jt"},
        payload={**operator_payload_base, "part": "Jt"},
        compute_fn=lambda: _timed(
            "operator_jt",
            lambda: np.asarray(jacobian.T, dtype=float),
        ),
        persist=True,
        cost=4.0,
        effort_seconds=2.0,
    )
    build_seconds.setdefault("operator_jt", 0.0)

    reg_diag, reg_lookup = cache_manager.get_or_compute_semantic(
        artifact="single_step_operator",
        name=CACHE_NAME_OPERATOR_NOSER,
        namespace=CACHE_NAMESPACE_DIFFERENCE,
        cache_obj={**operator_payload_base, "part": "NOSER_DIAG"},
        payload={**operator_payload_base, "part": "NOSER_DIAG"},
        compute_fn=lambda: _timed(
            "operator_noser",
            lambda: _build_noser_diag(jacobian, exponent=0.5, alpha=1.0),
        ),
        persist=True,
        cost=5.0,
        effort_seconds=3.0,
    )
    build_seconds.setdefault("operator_noser", 0.0)

    inv_reg_diag = np.asarray(1.0 / np.maximum(reg_diag, 1e-12), dtype=float)
    if solver_mode == "fast":
        system_matrix, a_lookup = cache_manager.get_or_compute_semantic(
            artifact="single_step_operator",
            name=CACHE_NAME_OPERATOR_A,
            namespace=CACHE_NAMESPACE_DIFFERENCE,
            cache_obj={**operator_payload_base, "part": "H_FAST"},
            payload={**operator_payload_base, "part": "H_FAST"},
            compute_fn=lambda: _timed(
                "operator_A",
                lambda: np.asarray(
                    safe_dot(jacobian * inv_reg_diag[None, :], jacobian_t, "gn_difference.operator.measurement_H"),
                    dtype=float,
                ) + float(lam) * np.eye(jacobian.shape[0], dtype=float),
            ),
            persist=True,
            cost=6.0,
            effort_seconds=4.0,
        )
        precond_diag, precond_lookup = cache_manager.get_or_compute_semantic(
            artifact="single_step_operator",
            name=CACHE_NAME_OPERATOR_PRECOND,
            namespace=CACHE_NAMESPACE_DIFFERENCE,
            cache_obj={**operator_payload_base, "part": "PRECOND_FAST"},
            payload={**operator_payload_base, "part": "PRECOND_FAST"},
            compute_fn=lambda: _timed(
                "operator_precond",
                lambda: np.asarray(np.maximum(np.diag(system_matrix), 1e-12), dtype=float),
            ),
            persist=True,
            cost=2.0,
            effort_seconds=1.0,
        )
    else:
        strict_backend_effective = str(strict_backend_info.get("effective", STRICT_SOLVER_BACKEND_DENSE))
        if strict_backend_effective == STRICT_SOLVER_BACKEND_MEASUREMENT:
            system_matrix, a_lookup = cache_manager.get_or_compute_semantic(
                artifact="single_step_operator",
                name=CACHE_NAME_OPERATOR_A,
                namespace=CACHE_NAMESPACE_DIFFERENCE,
                cache_obj={**operator_payload_base, "part": "H_STRICT_MEASUREMENT"},
                payload={**operator_payload_base, "part": "H_STRICT_MEASUREMENT"},
                compute_fn=lambda: _timed(
                    "operator_A",
                    lambda: np.asarray(
                        safe_dot(jacobian * inv_reg_diag[None, :], jacobian_t, "gn_difference.operator.strict_measurement_H"),
                        dtype=float,
                    ) + float(lam) * np.eye(jacobian.shape[0], dtype=float),
                ),
                persist=True,
                cost=6.0,
                effort_seconds=4.0,
            )
            precond_diag, precond_lookup = cache_manager.get_or_compute_semantic(
                artifact="single_step_operator",
                name=CACHE_NAME_OPERATOR_PRECOND,
                namespace=CACHE_NAMESPACE_DIFFERENCE,
                cache_obj={**operator_payload_base, "part": "PRECOND_STRICT_MEASUREMENT"},
                payload={**operator_payload_base, "part": "PRECOND_STRICT_MEASUREMENT"},
                compute_fn=lambda: _timed(
                    "operator_precond",
                    lambda: np.asarray(np.maximum(np.diag(system_matrix), 1e-12), dtype=float),
                ),
                persist=True,
                cost=2.0,
                effort_seconds=1.0,
            )
        else:
            system_matrix, a_lookup = cache_manager.get_or_compute_semantic(
                artifact="single_step_operator",
                name=CACHE_NAME_OPERATOR_A,
                namespace=CACHE_NAMESPACE_DIFFERENCE,
                cache_obj={**operator_payload_base, "part": "A"},
                payload={**operator_payload_base, "part": "A"},
                compute_fn=lambda: _timed(
                    "operator_A",
                    lambda: np.asarray(
                        safe_dot(jacobian_t, jacobian, "gn_difference.operator.JtJ"),
                        dtype=float,
                    ) + float(lam) * np.diag(reg_diag),
                ),
                persist=True,
                cost=10.0,
                effort_seconds=6.0,
            )
            precond_diag, precond_lookup = cache_manager.get_or_compute_semantic(
                artifact="single_step_operator",
                name=CACHE_NAME_OPERATOR_PRECOND,
                namespace=CACHE_NAMESPACE_DIFFERENCE,
                cache_obj={**operator_payload_base, "part": "PRECOND_STRICT"},
                payload={**operator_payload_base, "part": "PRECOND_STRICT"},
                compute_fn=lambda: _timed(
                    "operator_precond",
                    lambda: np.asarray(np.maximum(np.diag(system_matrix), 1e-12), dtype=float),
                ),
                persist=True,
                cost=2.0,
                effort_seconds=1.0,
            )
    build_seconds.setdefault("operator_A", 0.0)
    build_seconds.setdefault("operator_precond", 0.0)
    build_seconds.setdefault("operator_rom_snapshots", 0.0)
    build_seconds.setdefault("operator_rom_reduced_rm", 0.0)

    factor_payload = {
        **operator_payload_base,
        "part": "LU",
        "solver_mode": solver_mode,
        "strict_solver_backend_effective": str(strict_backend_info.get("effective", STRICT_SOLVER_BACKEND_DENSE)),
        "A_hash": hash_array(np.ascontiguousarray(system_matrix, dtype=np.float64)),
    }
    factor, factor_lookup = cache_manager.get_or_compute_semantic(
        artifact="single_step_operator",
        name=CACHE_NAME_OPERATOR_LU,
        namespace=CACHE_NAMESPACE_DIFFERENCE,
        cache_obj=factor_payload,
        payload=factor_payload,
        compute_fn=lambda: _timed(
            "operator_lu",
            lambda: _factorize_matrix(system_matrix),
        ),
        persist=True,
        cost=12.0,
        effort_seconds=10.0,
    )
    build_seconds.setdefault("operator_lu", 0.0)

    n_param = int(jacobian.shape[1])
    n_meas = int(jacobian.shape[0])
    ratio_n_over_m = float(n_param) / max(float(n_meas), 1.0)
    enable_rom = solver_mode == "fast" and int(mesh_dim) == 3 and (
        rom_mode == "on" or (rom_mode == "auto" and ratio_n_over_m >= 4.0)
    )
    enable_lowrank = enable_rom and (
        lowrank_mode == "on" or (lowrank_mode == "auto" and ratio_n_over_m >= 5.0)
    )
    reduced_rm = None
    reduced_lookup = None
    reduced_info = {
        "enabled": bool(enable_rom),
        "rom_mode": rom_mode,
        "lowrank_mode": lowrank_mode,
        "ratio_n_over_m": ratio_n_over_m,
    }
    if enable_rom:
        synthetic_snapshots = np.ascontiguousarray(
            np.column_stack(
                [
                    np.asarray(jacobian_t[:, 0], dtype=np.float64),
                    np.asarray(np.mean(jacobian_t, axis=1), dtype=np.float64),
                    np.asarray(inv_reg_diag, dtype=np.float64),
                ]
            ),
            dtype=np.float64,
        )
        snapshot_payload = {
            **operator_payload_base,
            "part": "ROM_SNAPSHOT_BANK",
            "rom_snapshot_source": rom_snapshot_source,
            "rom_rank_global": int(rom_rank_global),
            "rom_rank_adaptive": int(max(0, rom_rank_adaptive)),
        }
        cached_snapshot_matrix, _snapshot_lookup = cache_manager.get_or_compute_semantic(
            artifact="rom_snapshot_bank",
            name="gn_diff_rom_snapshot_bank",
            namespace=CACHE_NAMESPACE_DIFFERENCE,
            cache_obj=snapshot_payload,
            payload=snapshot_payload,
            compute_fn=lambda: _timed(
                "operator_rom_snapshots",
                lambda: synthetic_snapshots,
            ),
            persist=True,
            cost=3.0,
            effort_seconds=1.0,
        )
        snapshot_matrix = select_snapshot_matrix(
            rom_snapshot_source,
            n_param=n_param,
            bank_matrix=synthetic_snapshots,
            synthetic_matrix=synthetic_snapshots,
            cached_matrix=np.asarray(cached_snapshot_matrix, dtype=np.float64),
        )
        global_basis = compute_pod_basis(
            snapshot_matrix,
            rank=int(rom_rank_global),
            energy=float(lowrank_energy),
        )
        adaptive_basis = np.zeros((n_param, 0), dtype=np.float64)
        if enable_lowrank:
            adaptive_basis, _ = build_lowrank_subspace(
                jacobian,
                rank=int(max(1, lowrank_rank)),
                energy=float(lowrank_energy),
                method=str(lowrank_method),
            )
            if int(rom_rank_adaptive) > 0 and adaptive_basis.shape[1] > int(rom_rank_adaptive):
                adaptive_basis = np.asarray(
                    adaptive_basis[:, : int(rom_rank_adaptive)],
                    dtype=np.float64,
                )
        reduced_basis = merge_orthonormal_bases(
            global_basis,
            adaptive_basis,
            rank_cap=int(max(1, int(rom_rank_global) + int(max(0, rom_rank_adaptive)))),
        )

        if reduced_basis.shape[1] > 0:
            reduced_payload = {
                **operator_payload_base,
                "part": "ROM_REDUCED_RM",
                "rom_rank": int(reduced_basis.shape[1]),
                "basis_hash": hash_array(np.ascontiguousarray(reduced_basis, dtype=np.float64)),
                "rom_snapshot_source": rom_snapshot_source,
                "lowrank_method": lowrank_method,
                "lowrank_rank": int(lowrank_rank),
                "lowrank_energy": float(lowrank_energy),
            }
            reduced_rm, reduced_lookup = cache_manager.get_or_compute_semantic(
                artifact="rom_reduced_rm_diff",
                name=CACHE_NAME_OPERATOR_REDUCED_RM,
                namespace=CACHE_NAMESPACE_DIFFERENCE,
                cache_obj=reduced_payload,
                payload=reduced_payload,
                compute_fn=lambda: _timed(
                    "operator_rom_reduced_rm",
                    lambda: _build_reduced_rm(
                        jacobian=np.asarray(jacobian, dtype=np.float64),
                        reg_diag=np.asarray(reg_diag, dtype=np.float64),
                        lam=float(lam),
                        basis=np.asarray(reduced_basis, dtype=np.float64),
                    ),
                ),
                persist=True,
                cost=8.0,
                effort_seconds=4.0,
            )
            reduced_info.update(
                {
                    "basis_rank": int(reduced_basis.shape[1]),
                    "global_rank": int(global_basis.shape[1]),
                    "adaptive_rank": int(adaptive_basis.shape[1]),
                }
            )

    operator_bundle = {
        "mode": solver_mode,
        "linear_solver": linear_solver,
        "preconditioner": preconditioner,
        "J": np.asarray(jacobian, dtype=float),
        "Jt": np.asarray(jacobian_t, dtype=float),
        "A": np.asarray(system_matrix, dtype=float),
        "reg_diag": np.asarray(reg_diag, dtype=float),
        "inv_reg_diag": np.asarray(inv_reg_diag, dtype=float),
        "precond_diag": np.asarray(precond_diag, dtype=float),
        "factor": factor if isinstance(factor, dict) else {"method": "none"},
        "reduced_rm": reduced_rm if isinstance(reduced_rm, dict) else None,
        "reduced_info": reduced_info,
        "device_requested": str(runtime_selection.requested),
        "device_effective": str(runtime_selection.effective),
        "torch_device": str(runtime_selection.torch_device),
        "strict_solver_backend_requested": str(strict_backend_info.get("requested", STRICT_SOLVER_BACKEND_DENSE)),
        "strict_solver_backend_effective": str(strict_backend_info.get("effective", STRICT_SOLVER_BACKEND_DENSE)),
        "strict_memory_guard_triggered": bool(strict_backend_info.get("strict_memory_guard_triggered", False)),
        "strict_memory_guard_reason": str(strict_backend_info.get("strict_memory_guard_reason", "")),
        "strict_dense_estimated_peak_gib": float(strict_backend_info.get("strict_dense_estimated_peak_gib", 0.0)),
        "strict_measurement_system_shape": strict_backend_info.get("strict_measurement_system_shape"),
    }
    perf_caps = detect_performance_capabilities()

    return {
        "mesh": mesh,
        "fwd_model": fwd_model,
        "cache_manager": cache_manager,
        "cache_scope": cache_scope,
        "mesh_dim": int(mesh_dim),
        "mesh_height": float(mesh_height),
        "electrode_height_ratio": float(electrode_height_ratio),
        "z_center": float(z_center),
        "sigma_bg": sigma_bg,
        "img_bg": img_bg,
        "base_meas": base_meas,
        "difference_mode": str(difference_mode),
        "difference_orientation": str(difference_orientation),
        "n_stim": n_stim,
        "n_meas_total": n_meas_total,
        "n_meas_per_stim": n_meas_per_stim,
        "J": jacobian,
        "operator_bundle": operator_bundle,
        "strict_backend_info": dict(strict_backend_info),
        "solver_mode": solver_mode,
        "linear_solver": linear_solver,
        "preconditioner": preconditioner,
        "rom_mode": rom_mode,
        "rom_rank_global": int(rom_rank_global),
        "rom_rank_adaptive": int(rom_rank_adaptive),
        "rom_snapshot_source": rom_snapshot_source,
        "lowrank_mode": lowrank_mode,
        "lowrank_rank": int(lowrank_rank),
        "lowrank_method": lowrank_method,
        "lowrank_energy": float(lowrank_energy),
        "forward_mat_solve": forward_mat_solve,
        "forward_backend": forward_backend,
        "mesh_family": mesh_family,
        "geometry_version": geometry_version,
        "petsc_device": petsc_device,
        "petsc_backend_info": dict(petsc_backend_info),
        "device_requested": str(runtime_selection.requested),
        "device_effective": str(runtime_selection.effective),
        "torch_device": str(runtime_selection.torch_device),
        "mesh_cache_hit": mesh_cache_hit,
        "mesh_cache_layer": mesh_cache_layer,
        "mesh_cache_name": mesh_cache_name,
        "execution_profile": (
            "cuda" if str(getattr(fwd_model, "_petsc_backend_info", {}).get("petsc_device_effective", "cpu")) == "cuda"
            and str(runtime_selection.effective) == "cuda" and bool(jac_calc.use_torch)
            else (
                "mixed" if str(getattr(fwd_model, "_petsc_backend_info", {}).get("petsc_device_effective", "cpu")) == "cuda"
                or str(runtime_selection.effective) == "cuda" or bool(jac_calc.use_torch)
                else "cpu"
            )
        ),
        "stim_drive_mode": resolved_drive_mode,
        "stim_drive_value": stim_drive_value,
        "cache_build_seconds": dict(build_seconds),
        "context_build_seconds": time.perf_counter() - context_start,
        "performance_capabilities": perf_caps,
        "cache_miss_reasons": {
            CACHE_NAME_BASE_MEAS: _lookup_miss_reason(base_meas_lookup, family=CACHE_NAME_BASE_MEAS),
            CACHE_NAME_JACOBIAN: _lookup_miss_reason(jacobian_lookup, family=CACHE_NAME_JACOBIAN),
            CACHE_NAME_OPERATOR_JT: _lookup_miss_reason(j_t_lookup, family=CACHE_NAME_OPERATOR_JT),
            CACHE_NAME_OPERATOR_NOSER: _lookup_miss_reason(reg_lookup, family=CACHE_NAME_OPERATOR_NOSER),
            CACHE_NAME_OPERATOR_A: _lookup_miss_reason(a_lookup, family=CACHE_NAME_OPERATOR_A),
            CACHE_NAME_OPERATOR_PRECOND: _lookup_miss_reason(precond_lookup, family=CACHE_NAME_OPERATOR_PRECOND),
            CACHE_NAME_OPERATOR_LU: _lookup_miss_reason(factor_lookup, family=CACHE_NAME_OPERATOR_LU),
            CACHE_NAME_OPERATOR_REDUCED_RM: (
                _lookup_miss_reason(reduced_lookup, family=CACHE_NAME_OPERATOR_REDUCED_RM)
                if reduced_lookup is not None
                else "disabled"
            ),
        },
        "cache_lookups": {
            "base_meas": _to_lookup_payload(base_meas_lookup),
            "jacobian": _to_lookup_payload(jacobian_lookup),
            "operator_jt": _to_lookup_payload(j_t_lookup),
            "operator_noser": _to_lookup_payload(reg_lookup),
            "operator_A": _to_lookup_payload(a_lookup),
            "operator_precond": _to_lookup_payload(precond_lookup),
            "operator_lu": _to_lookup_payload(factor_lookup),
            "operator_rom_reduced_rm": _to_lookup_payload(reduced_lookup)
            if reduced_lookup is not None
            else {"hit": False, "layer": "disabled", "artifact": "rom_reduced_rm_diff", "key": ""},
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
    difference_mode: str = "raw",
    difference_orientation: str = "target_minus_reference",
) -> dict[str, object]:
    dv = build_difference_vector(
        vi,
        vh,
        mode=difference_mode,
        orientation=difference_orientation,
    )
    if dv.shape[0] != ctx["J"].shape[0]:
        raise RuntimeError(
            f"Data length {dv.shape[0]} does not match Jacobian rows {ctx['J'].shape[0]}"
        )

    operator_bundle = ctx["operator_bundle"]
    mode = str(operator_bundle.get("mode", "strict"))
    stage_timings = {
        "linear_solve": 0.0,
        "forward_validate": 0.0,
    }

    if mode == "fast":
        reduced_rm = operator_bundle.get("reduced_rm")
        if isinstance(reduced_rm, dict) and isinstance(reduced_rm.get("RM_reduced"), np.ndarray):
            linear_start = time.perf_counter()
            runtime_device = str(operator_bundle.get("device_effective", "cpu"))
            if runtime_device == "cuda" and torch is not None:
                rm_t = _bundle_torch_tensor(operator_bundle, "RM_reduced", np.asarray(reduced_rm["RM_reduced"], dtype=np.float64))
                dv_t = torch.as_tensor(np.asarray(dv, dtype=np.float64), device=str(operator_bundle.get("torch_device", "cuda")), dtype=torch.float64)
                delta_sigma = np.asarray(torch.mv(rm_t, dv_t).detach().cpu().numpy(), dtype=float)
            else:
                delta_sigma = np.asarray(
                    safe_dot(
                        np.asarray(reduced_rm["RM_reduced"], dtype=np.float64),
                        dv,
                        "gn_difference.rom.delta_sigma",
                    ),
                    dtype=float,
                )
            reduced_elapsed = time.perf_counter() - linear_start
            stage_timings["linear_solve"] += reduced_elapsed
            stage_timings["linear_solve_reduced_rm"] = (
                stage_timings.get("linear_solve_reduced_rm", 0.0) + reduced_elapsed
            )
        else:
            linear_start = time.perf_counter()
            runtime_device = str(operator_bundle.get("device_effective", "cpu"))
            torch_device = str(operator_bundle.get("torch_device", "cuda"))
            y = _solve_measurement_space(
                system_matrix=operator_bundle["A"],
                rhs=dv,
                linear_solver=str(operator_bundle.get("linear_solver", "auto")),
                preconditioner_mode=str(operator_bundle.get("preconditioner", "auto")),
                preconditioner=operator_bundle.get("precond_diag"),
                runtime_device=runtime_device,
                torch_device=torch_device,
            )
            stage_timings["linear_solve"] += time.perf_counter() - linear_start
            if runtime_device == "cuda" and torch is not None:
                inv_reg_diag_t = _bundle_torch_tensor(operator_bundle, "inv_reg_diag", operator_bundle["inv_reg_diag"])
                Jt_t = _bundle_torch_tensor(operator_bundle, "Jt", operator_bundle["Jt"])
                y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), device=torch_device, dtype=torch.float64)
                delta_sigma = np.asarray((inv_reg_diag_t * torch.mv(Jt_t, y_t)).detach().cpu().numpy(), dtype=float)
            else:
                delta_sigma = np.asarray(
                    operator_bundle["inv_reg_diag"]
                    * safe_dot(
                        operator_bundle["Jt"],
                        y,
                        "gn_difference.operator.fast_RinvJt_y",
                    ),
                    dtype=float,
                )
    else:
        linear_start = time.perf_counter()
        strict_backend = str(operator_bundle.get("strict_solver_backend_effective", STRICT_SOLVER_BACKEND_DENSE))
        runtime_device = str(operator_bundle.get("device_effective", "cpu"))
        if strict_backend == STRICT_SOLVER_BACKEND_MEASUREMENT:
            delta_sigma = _measurement_space_delta(
                operator_bundle=operator_bundle,
                rhs=dv,
            )
        else:
            if runtime_device == "cuda" and torch is not None:
                Jt_t = _bundle_torch_tensor(operator_bundle, "Jt", operator_bundle["Jt"])
                dv_t = torch.as_tensor(np.asarray(dv, dtype=np.float64), device=str(operator_bundle.get("torch_device", "cuda")), dtype=torch.float64)
                b = np.asarray(torch.mv(Jt_t, dv_t).detach().cpu().numpy(), dtype=float)
            else:
                b = np.asarray(
                    safe_dot(operator_bundle["Jt"], dv, "gn_difference.operator.Jt_dv"),
                    dtype=float,
                )
            delta_sigma = _solve_linear_from_bundle(operator_bundle, b)
        stage_timings["linear_solve"] += time.perf_counter() - linear_start

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
            difference_mode=difference_mode,
            difference_orientation=difference_orientation,
        )

    sigma_est = ctx["sigma_bg"] + alpha * delta_sigma
    delta_sigma_scaled = alpha * delta_sigma
    img_est = EITImage(elem_data=sigma_est, fwd_model=ctx["fwd_model"])
    forward_start = time.perf_counter()
    pred_vi, _ = ctx["fwd_model"].fwd_solve(img_est)
    stage_timings["forward_validate"] += time.perf_counter() - forward_start
    pred_diff = build_difference_vector(
        pred_vi.meas,
        ctx["base_meas"],
        mode=difference_mode,
        orientation=difference_orientation,
    )
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
        "solver_mode": str(ctx.get("solver_mode", "strict")),
        "linear_solver": str(ctx.get("linear_solver", "auto")),
        "preconditioner": str(ctx.get("preconditioner", "auto")),
        "forward_mat_solve": str(ctx.get("forward_mat_solve", "off")),
        "petsc_device": str(ctx.get("petsc_device", "auto")),
        "inverse_device_requested": str(ctx.get("device_requested", "auto")),
        "inverse_device_effective": str(ctx.get("device_effective", "cpu")),
        "execution_profile": str(ctx.get("execution_profile", "cpu")),
        "jacobian_block_backend": "torch-cuda" if str(ctx.get("device_effective", "cpu")) == "cuda" else "numpy",
        "strict_solver_backend_requested": str(operator_bundle.get("strict_solver_backend_requested", STRICT_SOLVER_BACKEND_DENSE)),
        "strict_solver_backend_effective": str(operator_bundle.get("strict_solver_backend_effective", STRICT_SOLVER_BACKEND_DENSE)),
        "strict_memory_guard_triggered": bool(operator_bundle.get("strict_memory_guard_triggered", False)),
        "strict_memory_guard_reason": str(operator_bundle.get("strict_memory_guard_reason", "")),
        "strict_dense_estimated_peak_gib": float(operator_bundle.get("strict_dense_estimated_peak_gib", 0.0)),
        "strict_measurement_system_shape": operator_bundle.get("strict_measurement_system_shape"),
        "cache_build_seconds": dict(ctx.get("cache_build_seconds", {})),
        "cache_miss_reasons": dict(ctx.get("cache_miss_reasons", {})),
        "cache_lookups": {
            "context": dict(ctx.get("cache_lookups", {})),
            "forward_factor": forward_lookup,
        },
        "reduced_info": dict(operator_bundle.get("reduced_info", {})),
        "stage_timings": stage_timings,
        "cache_stats": cache_stats,
    }
