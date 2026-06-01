#!/usr/bin/env python3
"""Step-by-step complex-admittance parity harness for EIDORS and PyEIDORS.

This script is intentionally diagnostic, not a benchmark. It exports one
canonical 3D 8x2-electrode mesh and complex-admittance payload for MATLAB
EIDORS, solves the same payload in PyEIDORS, and compares every stage once the
EIDORS result file is available.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pyeidors import EITSystem
from pyeidors.data.difference import (
    build_difference_vector,
    project_measurement_jacobian,
)
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager
from pyeidors.femx import cell_midpoints
from pyeidors.geometry.mesh3d_generator import create_cylinder_3d_eit_mesh
from pyeidors.interop.geometry_exchange import build_electrode_arrays
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsJacobianAdapter
from pyeidors.inverse.reconstruction_matrix import (
    _noser_regularization,
    build_one_step_rm,
)


OUT_ROOT = ROOT / "results" / "complex_eidors_pyeidors_step_compare"


@dataclass(frozen=True)
class ComplexCase:
    name: str = "complex_3d_8x2_center_sphere"
    n_per_ring: int = 8
    n_rings: int = 2
    radius: float = 0.18
    height: float = 0.16
    refinement: int = 2
    mesh_family: str = "tetra"
    electrode_coverage: float = 0.5
    electrode_height_ratio: float = 0.2
    base_sigma: complex = 1.0 + 2.0j
    target_sigma: complex = 2.0 + 3.0j
    contact_impedance: complex = 0.01 + 0.05j
    target_radius: float = 0.063
    target_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hyperparameter: float = 1.0e-2
    measurement_protocol: str = "eidors_full_3d"

    @property
    def total_electrodes(self) -> int:
        return int(self.n_per_ring * self.n_rings)

    @property
    def ring_fractions(self) -> tuple[float, ...]:
        if self.n_rings == 2:
            return (0.25, 0.75)
        return tuple(np.linspace(0.15, 0.85, self.n_rings).tolist())


def _json_default(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _complex_text(value: complex) -> str:
    sign = "+" if value.imag >= 0.0 else "-"
    return f"{value.real:.12g}{sign}{abs(value.imag):.12g}j"


def _complex_scalar(payload: dict[str, np.ndarray], key: str) -> complex:
    return complex(np.asarray(payload[key]).reshape(-1)[0])


def _int_scalar(payload: dict[str, np.ndarray], key: str) -> int:
    return int(np.asarray(payload[key]).reshape(-1)[0])


def _float_scalar(payload: dict[str, np.ndarray], key: str) -> float:
    return float(np.asarray(payload[key]).reshape(-1)[0])


def _abs_phase_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    return {
        "real_min": float(np.min(arr.real)),
        "real_max": float(np.max(arr.real)),
        "imag_min": float(np.min(arr.imag)),
        "imag_max": float(np.max(arr.imag)),
        "abs_min": float(np.min(np.abs(arr))),
        "abs_max": float(np.max(np.abs(arr))),
        "phase_min": float(np.min(np.angle(arr))),
        "phase_max": float(np.max(np.angle(arr))),
    }


def _safe_norm(values: np.ndarray) -> float:
    norm = float(np.linalg.norm(np.asarray(values).reshape(-1)))
    return max(norm, float(np.finfo(np.float64).eps))


def _complex_rel_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    cand = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if ref.shape != cand.shape:
        return float("nan")
    return float(np.linalg.norm(cand - ref) / _safe_norm(ref))


def _complex_rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    cand = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if ref.shape != cand.shape:
        return float("nan")
    return float(np.sqrt(np.mean(np.abs(cand - ref) ** 2)))


def _complex_corr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    cand = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if ref.shape != cand.shape or ref.size < 2:
        return float("nan")
    denom = _safe_norm(ref) * _safe_norm(cand)
    return float(abs(np.vdot(ref, cand)) / denom)


def _complex_scalar_fit(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    cand = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if ref.shape != cand.shape or cand.size == 0:
        return {"alpha": None, "rel_l2_after_fit": float("nan")}
    denom = np.vdot(cand, cand)
    if abs(denom) <= np.finfo(np.float64).eps:
        return {"alpha": None, "rel_l2_after_fit": float("nan")}
    alpha = np.vdot(cand, ref) / denom
    return {
        "alpha": {"real": float(alpha.real), "imag": float(alpha.imag)},
        "rel_l2_after_fit": _complex_rel_l2(ref, alpha * cand),
    }


def _stage_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.complex128).reshape(-1)
    cand = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    same_shape = ref.shape == cand.shape
    metrics: dict[str, Any] = {
        "shape_reference": list(ref.shape),
        "shape_candidate": list(cand.shape),
        "same_shape": same_shape,
    }
    if same_shape:
        metrics.update(
            {
                "rel_l2": _complex_rel_l2(ref, cand),
                "rmse_abs": _complex_rmse(ref, cand),
                "corr_abs_complex": _complex_corr(ref, cand),
                "max_abs_error": float(np.max(np.abs(cand - ref))) if ref.size else 0.0,
                "candidate_is_negative_reference_rel_l2": _complex_rel_l2(ref, -cand),
                "candidate_conjugate_rel_l2": _complex_rel_l2(ref, np.conjugate(cand)),
                "best_complex_scalar_fit": _complex_scalar_fit(ref, cand),
            }
        )
    return metrics


def _find_first_stage_above_threshold(
    stage_metrics: dict[str, Any],
    keys: list[str],
    *,
    threshold: float,
) -> str | None:
    for key in keys:
        metrics = stage_metrics.get(key, {})
        if not metrics.get("available") or not metrics.get("same_shape"):
            return key
        rel = float(metrics.get("rel_l2", 0.0))
        if not math.isfinite(rel) or rel > float(threshold):
            return key
    return None


def _rm_application_metrics(
    eidors: dict[str, np.ndarray],
    py: dict[str, np.ndarray],
) -> dict[str, Any]:
    required = {"rm_matrix", "dv_norm_tmr", "rec_delta"}
    if not required.issubset(eidors) or not required.issubset(py):
        return {"available": False}

    eidors_rm = np.asarray(eidors["rm_matrix"], dtype=np.complex128)
    eidors_dv = np.asarray(eidors["dv_norm_tmr"], dtype=np.complex128).reshape(-1)
    eidors_rec = np.asarray(eidors["rec_delta"], dtype=np.complex128).reshape(-1)
    py_rm = np.asarray(py["rm_matrix"], dtype=np.complex128)
    py_dv = np.asarray(py["dv_norm_tmr"], dtype=np.complex128).reshape(-1)
    py_rec = np.asarray(py["rec_delta"], dtype=np.complex128).reshape(-1)
    if (
        eidors_rm.ndim != 2
        or py_rm.ndim != 2
        or eidors_rm.shape[1] != eidors_dv.size
        or py_rm.shape[1] != py_dv.size
        or eidors_rm.shape[0] != eidors_rec.size
        or py_rm.shape[0] != py_rec.size
        or eidors_rm.shape != py_rm.shape
        or eidors_dv.shape != py_dv.shape
        or eidors_rec.shape != py_rec.shape
    ):
        return {
            "available": False,
            "eidors_rm_shape": list(eidors_rm.shape),
            "pyeidors_rm_shape": list(py_rm.shape),
        }

    return {
        "available": True,
        "eidors_self_rel_l2": _complex_rel_l2(eidors_rec, eidors_rm @ eidors_dv),
        "pyeidors_self_rel_l2": _complex_rel_l2(py_rec, py_rm @ py_dv),
        "eidors_rm_on_pyeidors_dv_rel_l2": _complex_rel_l2(py_rec, eidors_rm @ py_dv),
        "pyeidors_rm_on_eidors_dv_rel_l2": _complex_rel_l2(
            eidors_rec, py_rm @ eidors_dv
        ),
        "eidors_rm_norm": float(np.linalg.norm(eidors_rm)),
        "pyeidors_rm_norm": float(np.linalg.norm(py_rm)),
        "eidors_dv_norm": float(np.linalg.norm(eidors_dv)),
        "pyeidors_dv_norm": float(np.linalg.norm(py_dv)),
    }


def boundary_facets_3d(eit_mesh) -> np.ndarray:
    mesh = eit_mesh.mesh
    tags = eit_mesh.facet_tags
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    if f2v is None:
        raise RuntimeError("mesh has no facet-to-vertex connectivity")
    facets = np.asarray(tags.indices, dtype=np.int64).reshape(-1)
    rows: list[np.ndarray] = []
    for facet in facets:
        verts = np.asarray(f2v.links(int(facet)), dtype=np.int64).reshape(-1)
        if verts.size == 3:
            rows.append(verts + 1)
    if not rows:
        raise RuntimeError("no triangular boundary facets were found")
    return np.vstack(rows).astype(np.int64, copy=False)


def _measurement_starts(meas_counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(meas_counts, dtype=np.int64).reshape(-1)
    starts = np.empty(counts.size, dtype=np.int64)
    if counts.size:
        starts[0] = 0
        if counts.size > 1:
            np.cumsum(counts[:-1], out=starts[1:])
    return starts


def _stack_measurement_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.empty((0, 0), dtype=float)
    arrays = [np.asarray(matrix, dtype=float) for matrix in matrices]
    n_cols = int(arrays[0].shape[1])
    if any(matrix.ndim != 2 or matrix.shape[1] != n_cols for matrix in arrays):
        raise ValueError("measurement matrices must be 2D with matching columns")
    out = np.empty(
        (sum(int(matrix.shape[0]) for matrix in arrays), n_cols), dtype=float
    )
    start = 0
    for matrix in arrays:
        stop = start + int(matrix.shape[0])
        out[start:stop, :] = matrix
        start = stop
    return out


def build_case_mesh(case: ComplexCase, case_dir: Path):
    mesh_dir = case_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    return create_cylinder_3d_eit_mesh(
        n_elec=case.total_electrodes,
        radius=case.radius,
        height=case.height,
        refinement=case.refinement,
        electrode_coverage=case.electrode_coverage,
        electrode_height_ratio=case.electrode_height_ratio,
        electrode_level_fractions=case.ring_fractions,
        z_center=0.0,
        mesh_family=case.mesh_family,
        geometry_version="geomv2",
        electrode_layout="ring_major",
        output_dir=str(mesh_dir),
        mesh_name=case.name,
    )


def build_truth(case: ComplexCase, mesh) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(cell_midpoints(mesh.mesh), dtype=float)
    center = np.asarray(case.target_center, dtype=float).reshape(1, 3)
    dist = np.linalg.norm(centers[:, :3] - center, axis=1)
    mask = dist <= float(case.target_radius)
    truth = np.full(mesh.num_cells(), case.base_sigma, dtype=np.complex128)
    truth[mask] = case.target_sigma
    return truth, mask


def build_pattern_config(case: ComplexCase) -> PatternConfig:
    return PatternConfig(
        n_elec=case.n_per_ring,
        n_rings=case.n_rings,
        electrode_layout="ring_major",
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        measurement_protocol=case.measurement_protocol,
        drive_mode="total_current",
        drive_value=1.0,
        rotate_meas=True,
        use_meas_current=False,
        stim_first_positive=False,
    )


def payload_measurement_matrices(
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    stim = np.asarray(payload["stim_matrix"], dtype=float)
    meas_concat = np.asarray(payload["meas_matrix_concat"], dtype=float)
    starts = np.asarray(payload["meas_start"], dtype=np.int64).reshape(-1) - 1
    counts = np.asarray(payload["meas_counts"], dtype=np.int64).reshape(-1)
    matrices = [
        meas_concat[int(start) : int(start + count)].copy()
        for start, count in zip(starts, counts, strict=True)
    ]
    return stim, matrices, meas_concat, starts, counts


def exact_pattern_from_payload(payload: dict[str, np.ndarray]) -> PatternConfig:
    stim, meas_matrices, _, _, _ = payload_measurement_matrices(payload)
    return PatternConfig(
        n_elec=_int_scalar(payload, "n_per_ring"),
        n_rings=_int_scalar(payload, "n_rings"),
        electrode_layout="ring_major",
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        measurement_protocol="custom",
        custom_stim_matrix=stim,
        custom_meas_matrices=meas_matrices,
        drive_mode="total_current",
        drive_value=1.0,
        rotate_meas=True,
        use_meas_current=False,
        stim_first_positive=False,
    )


def export_payload(case: ComplexCase, out_root: Path) -> Path:
    case_dir = out_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_case_mesh(case, case_dir)
    truth, truth_mask = build_truth(case, mesh)
    cells = np.asarray(mesh.cells(), dtype=np.int64)
    nodes = np.asarray(mesh.coordinates(), dtype=float)
    electrode_nodes, electrode_counts = build_electrode_arrays(mesh)
    boundary = boundary_facets_3d(mesh)

    pattern = build_pattern_config(case)
    pm = StimMeasPatternManager(pattern, mesh_tdim=3)
    meas_counts = np.asarray(pm.n_meas_per_stim, dtype=np.int64)
    meas_start = _measurement_starts(meas_counts)
    payload = {
        "case_name": case.name,
        "dim": 3,
        "nodes": nodes,
        "elems": cells + 1,
        "boundary": boundary,
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "truth_elem_data": truth.reshape(-1, 1),
        "truth_mask": truth_mask.reshape(-1, 1),
        "stim_matrix": np.asarray(pm.stim_matrix, dtype=float),
        "meas_matrix_concat": _stack_measurement_matrices(pm.meas_matrices),
        "meas_start": meas_start.reshape(-1, 1) + 1,
        "meas_counts": meas_counts.reshape(-1, 1),
        "radius": case.radius,
        "height": case.height,
        "z_center": 0.0,
        "refinement": case.refinement,
        "mesh_family": case.mesh_family,
        "n_per_ring": case.n_per_ring,
        "n_rings": case.n_rings,
        "total_electrodes": case.total_electrodes,
        "ring_fractions": np.asarray(case.ring_fractions, dtype=float).reshape(1, -1),
        "base_sigma": np.asarray([[case.base_sigma]], dtype=np.complex128),
        "target_sigma": np.asarray([[case.target_sigma]], dtype=np.complex128),
        "target_radius": case.target_radius,
        "target_center": np.asarray(case.target_center, dtype=float).reshape(1, 3),
        "contact_impedance": np.asarray(
            [[case.contact_impedance]], dtype=np.complex128
        ),
        "hyperparameter": case.hyperparameter,
        "measurement_protocol": case.measurement_protocol,
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "rotate_meas": True,
        "use_meas_current": False,
        "drive_value": 1.0,
        "drive_mode": "total_current",
        "source": "complex_eidors_pyeidors_step_compare",
    }
    savemat(case_dir / "payload.mat", payload, do_compression=True)
    meta = {
        "case": asdict(case),
        "base_sigma_text": _complex_text(case.base_sigma),
        "target_sigma_text": _complex_text(case.target_sigma),
        "contact_impedance_text": _complex_text(case.contact_impedance),
        "nodes": int(nodes.shape[0]),
        "elements": int(cells.shape[0]),
        "boundary_facets": int(boundary.shape[0]),
        "electrodes": int(case.total_electrodes),
        "n_stim": int(pm.n_stim),
        "n_meas_total": int(pm.n_meas_total),
        "meas_per_stim_min": int(meas_counts.min()),
        "meas_per_stim_max": int(meas_counts.max()),
        "truth_inclusion_cells": int(truth_mask.sum()),
        "truth_sigma_summary": _abs_phase_summary(truth),
    }
    (case_dir / "payload_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(
        "[export] "
        f"{case.name}: nodes={nodes.shape[0]} elems={cells.shape[0]} "
        f"stim={pm.n_stim} meas={pm.n_meas_total} "
        f"sigma_bg={_complex_text(case.base_sigma)} "
        f"sigma_obj={_complex_text(case.target_sigma)} "
        f"z={_complex_text(case.contact_impedance)}"
    )
    return case_dir


def verify_mesh_order(payload: dict[str, np.ndarray], mesh) -> bool:
    payload_nodes = np.asarray(payload["nodes"], dtype=float)
    mesh_nodes = np.asarray(mesh.coordinates(), dtype=float)
    payload_elems = np.asarray(payload["elems"], dtype=np.int64)
    mesh_elems = np.asarray(mesh.cells(), dtype=np.int64) + 1
    return bool(
        payload_nodes.shape == mesh_nodes.shape
        and payload_elems.shape == mesh_elems.shape
        and np.allclose(payload_nodes, mesh_nodes)
        and np.array_equal(payload_elems, mesh_elems)
    )


def run_pyeidors(case: ComplexCase, out_root: Path, *, skip_jacobian: bool) -> Path:
    case_dir = out_root / case.name
    payload_path = case_dir / "payload.mat"
    if not payload_path.exists():
        export_payload(case, out_root)
    payload = loadmat(payload_path)
    mesh = build_case_mesh(case, case_dir)
    if not verify_mesh_order(payload, mesh):
        raise RuntimeError("regenerated PyEIDORS mesh differs from payload.mat")

    base_sigma = _complex_scalar(payload, "base_sigma")
    contact_z = _complex_scalar(payload, "contact_impedance")
    truth = np.asarray(payload["truth_elem_data"], dtype=np.complex128).reshape(-1)
    hp = _float_scalar(payload, "hyperparameter")
    total_electrodes = _int_scalar(payload, "total_electrodes")

    system = EITSystem(
        n_elec=total_electrodes,
        pattern_config=exact_pattern_from_payload(payload),
        contact_impedance=np.full(total_electrodes, contact_z, dtype=np.complex128),
        base_conductivity=base_sigma,
        regularization_type="noser",
        regularization_alpha=1.0,
        hyperparameter=hp,
        noser_exponent=0.5,
        difference_mode="normalized",
        difference_step_size_mode="off",
        linear_backend="scipy",
        linear_backend_config={"solver_preset": "direct", "mat_solve_mode": "off"},
        petsc_device="cpu",
        device="cpu",
        solver_mode="fast",
        line_search_mode="fast",
        forward_backend="dolfinx",
        mesh_family=case.mesh_family,
        cache_dir=str(case_dir / ".pyeidors_cache_complex_cpu"),
    )
    setup_start = time.perf_counter()
    system.setup(mesh=mesh, initialize_inverse=False)
    setup_seconds = time.perf_counter() - setup_start

    bg = EITImage(
        elem_data=np.full(mesh.num_cells(), base_sigma, dtype=np.complex128),
        fwd_model=system.fwd_model,
    )
    target = EITImage(elem_data=truth, fwd_model=system.fwd_model)
    forward_start = time.perf_counter()
    vh = system.forward_solve(bg)
    vi = system.forward_solve(target)
    forward_seconds = time.perf_counter() - forward_start

    vh_meas = np.asarray(vh.meas, dtype=np.complex128).reshape(-1)
    vi_meas = np.asarray(vi.meas, dtype=np.complex128).reshape(-1)
    dv_raw_tmr = build_difference_vector(
        vi_meas,
        vh_meas,
        mode="raw",
        orientation="target_minus_reference",
    )
    dv_raw_rmt = build_difference_vector(
        vi_meas,
        vh_meas,
        mode="raw",
        orientation="reference_minus_target",
    )
    dv_norm_tmr = build_difference_vector(
        vi_meas,
        vh_meas,
        mode="normalized",
        orientation="target_minus_reference",
    )
    dv_norm_rmt = build_difference_vector(
        vi_meas,
        vh_meas,
        mode="normalized",
        orientation="reference_minus_target",
    )

    jacobian = np.empty((0, truth.size), dtype=np.complex128)
    noser_prior_diag = np.empty(0, dtype=np.complex128)
    rm_matrix = np.empty((0, 0), dtype=np.complex128)
    rec_delta = np.empty(0, dtype=np.complex128)
    rec_sigma = np.empty(0, dtype=np.complex128)
    jacobian_status: dict[str, Any] = {"attempted": not skip_jacobian, "ok": False}
    if not skip_jacobian:
        try:
            jac_start = time.perf_counter()
            jac_calc = EidorsJacobianAdapter(
                system.fwd_model,
                use_torch=False,
                device="cpu",
                torch_dtype="complex128",
            )
            raw_jacobian = jac_calc.calculate_from_image(bg)
            jacobian = np.asarray(
                project_measurement_jacobian(
                    raw_jacobian,
                    measurement_type="difference",
                    reference_meas=vh_meas,
                    difference_mode="normalized",
                    difference_orientation="target_minus_reference",
                ),
                dtype=np.complex128,
            )
            noser_prior_diag = np.asarray(
                _noser_regularization(jacobian, floor=1.0e-12, exponent=0.5),
                dtype=np.complex128,
            ).reshape(-1)
            rm = build_one_step_rm(jacobian, lambda_=hp, mode="noser", form="param")
            rm_matrix = np.asarray(rm, dtype=np.complex128)
            rec_delta = np.asarray(rm @ dv_norm_tmr, dtype=np.complex128).reshape(-1)
            rec_sigma = base_sigma + rec_delta
            jacobian_status = {
                "attempted": True,
                "ok": True,
                "seconds": time.perf_counter() - jac_start,
                "shape": list(jacobian.shape),
            }
        except Exception as exc:  # pragma: no cover - diagnostic path.
            jacobian_status = {
                "attempted": True,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    summary = {
        "setup_seconds": setup_seconds,
        "forward_seconds": forward_seconds,
        "n_meas": int(vh_meas.size),
        "truth_sigma_summary": _abs_phase_summary(truth),
        "vh_summary": _abs_phase_summary(vh_meas),
        "vi_summary": _abs_phase_summary(vi_meas),
        "dv_norm_tmr_summary": _abs_phase_summary(dv_norm_tmr),
        "jacobian_status": jacobian_status,
        "backend_diagnostics": system.fwd_model.get_backend_diagnostics(),
    }
    py_path = case_dir / "pyeidors_result.h5"
    with h5py.File(py_path, "w") as h5:
        h5.create_dataset("truth_elem_data", data=truth, compression="gzip")
        h5.create_dataset("vh", data=vh_meas, compression="gzip")
        h5.create_dataset("vi", data=vi_meas, compression="gzip")
        h5.create_dataset("dv_raw_tmr", data=dv_raw_tmr, compression="gzip")
        h5.create_dataset("dv_raw_rmt", data=dv_raw_rmt, compression="gzip")
        h5.create_dataset("dv_norm_tmr", data=dv_norm_tmr, compression="gzip")
        h5.create_dataset("dv_norm_rmt", data=dv_norm_rmt, compression="gzip")
        h5.create_dataset("jacobian_projected_norm_tmr", data=jacobian)
        h5.create_dataset("noser_prior_diag", data=noser_prior_diag, compression="gzip")
        h5.create_dataset("rm_matrix", data=rm_matrix, compression="gzip")
        h5.create_dataset("rec_delta", data=rec_delta)
        h5.create_dataset("rec_sigma", data=rec_sigma)
        h5.attrs["summary_json"] = json.dumps(
            summary,
            ensure_ascii=False,
            default=_json_default,
        )
    (case_dir / "pyeidors_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(
        "[pyeidors] "
        f"{case.name}: setup={setup_seconds:.3f}s forward={forward_seconds:.3f}s "
        f"meas={vh_meas.size} jacobian_ok={jacobian_status.get('ok')}"
    )
    return py_path


def load_pyeidors_arrays(case_dir: Path) -> dict[str, np.ndarray]:
    path = case_dir / "pyeidors_result.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for key in h5.keys():
            out[key] = np.asarray(h5[key])
    return out


def load_eidors_arrays(case_dir: Path) -> dict[str, np.ndarray] | None:
    path = case_dir / "eidors_result.mat"
    if not path.exists():
        return None
    mat = loadmat(path)
    keys = [
        "truth_elem_data",
        "vh",
        "vi",
        "dv_raw_tmr",
        "dv_raw_rmt",
        "dv_norm_tmr",
        "dv_norm_rmt",
        "jacobian_projected_norm_tmr",
        "noser_prior_diag",
        "rm_matrix",
        "rec_delta",
        "rec_sigma",
    ]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in mat:
            continue
        array = np.asarray(mat[key])
        out[key] = array if key == "rm_matrix" else array.reshape(-1)
    return out


def compare_results(case: ComplexCase, out_root: Path) -> Path:
    case_dir = out_root / case.name
    payload_path = case_dir / "payload.mat"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    payload = loadmat(payload_path)
    py = load_pyeidors_arrays(case_dir)
    eidors = load_eidors_arrays(case_dir)

    report: dict[str, Any] = {
        "case": case.name,
        "payload": {
            "nodes": int(np.asarray(payload["nodes"]).shape[0]),
            "elements": int(np.asarray(payload["elems"]).shape[0]),
            "n_per_ring": _int_scalar(payload, "n_per_ring"),
            "n_rings": _int_scalar(payload, "n_rings"),
            "total_electrodes": _int_scalar(payload, "total_electrodes"),
            "n_stim": int(np.asarray(payload["stim_matrix"]).shape[0]),
            "n_meas_total": int(np.sum(np.asarray(payload["meas_counts"]))),
            "base_sigma": _complex_text(_complex_scalar(payload, "base_sigma")),
            "target_sigma": _complex_text(_complex_scalar(payload, "target_sigma")),
            "contact_impedance": _complex_text(
                _complex_scalar(payload, "contact_impedance")
            ),
            "truth_sigma_summary": _abs_phase_summary(
                np.asarray(payload["truth_elem_data"], dtype=np.complex128)
            ),
        },
        "eidors_result_present": eidors is not None,
        "stage_metrics": {},
    }
    if eidors is None:
        report["next_step"] = (
            "Run MATLAB/EIDORS script, then rerun this Python script with --stage compare."
        )
    else:
        stages = [
            ("truth_elem_data", "material assignment"),
            ("vh", "background forward voltage"),
            ("vi", "target forward voltage"),
            ("dv_raw_tmr", "raw target-reference difference"),
            ("dv_raw_rmt", "raw reference-target difference"),
            ("dv_norm_tmr", "normalized target-reference difference"),
            ("dv_norm_rmt", "normalized reference-target difference"),
            ("jacobian_projected_norm_tmr", "projected normalized Jacobian"),
            ("noser_prior_diag", "NOSER RtR diagonal"),
            ("rm_matrix", "one-step reconstruction matrix"),
            ("rec_delta", "one-step reconstruction delta"),
            ("rec_sigma", "one-step absolute sigma"),
        ]
        for key, label in stages:
            if key not in py or key not in eidors:
                report["stage_metrics"][key] = {"label": label, "available": False}
                continue
            report["stage_metrics"][key] = {
                "label": label,
                "available": True,
                **_stage_metrics(eidors[key], py[key]),
            }
        candidate_keys = [
            "truth_elem_data",
            "vh",
            "vi",
            "dv_norm_tmr",
            "jacobian_projected_norm_tmr",
            "noser_prior_diag",
            "rm_matrix",
            "rec_delta",
        ]
        report["first_numerical_difference_candidate"] = (
            _find_first_stage_above_threshold(
                report["stage_metrics"],
                candidate_keys,
                threshold=1.0e-6,
            )
        )
        report["first_divergence_candidate"] = _find_first_stage_above_threshold(
            report["stage_metrics"],
            candidate_keys,
            threshold=1.0e-3,
        )
        report["rm_application_metrics"] = _rm_application_metrics(eidors, py)

    json_path = case_dir / "step_compare_summary.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    md_path = case_dir / "step_compare_report.md"
    md_path.write_text(render_markdown_report(report, case_dir), encoding="utf-8")
    print(f"[compare] wrote {json_path}")
    print(f"[compare] wrote {md_path}")
    return json_path


def render_markdown_report(report: dict[str, Any], case_dir: Path) -> str:
    payload = report["payload"]
    lines = [
        "# Complex EIDORS vs PyEIDORS Step Compare",
        "",
        f"- case: `{report['case']}`",
        f"- result dir: `{case_dir}`",
        f"- mesh: nodes={payload['nodes']}, elements={payload['elements']}",
        f"- electrodes: {payload['n_per_ring']} per ring x {payload['n_rings']} rings",
        f"- protocol: `{payload['n_stim']}` stim, `{payload['n_meas_total']}` measurements",
        f"- sigma background: `{payload['base_sigma']}`",
        f"- sigma target: `{payload['target_sigma']}`",
        f"- contact impedance: `{payload['contact_impedance']}`",
        "",
    ]
    if not report["eidors_result_present"]:
        lines.extend(
            [
                "## EIDORS Result Missing",
                "",
                "The shared payload and PyEIDORS result exist, but MATLAB/EIDORS has not",
                "written `eidors_result.mat` yet. Run:",
                "",
                "```matlab",
                "complex_eidors_pyeidors_step_compare('<out_root>')",
                "```",
                "",
                "Then rerun the Python script with `--stage compare`.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Stage Metrics",
            "",
            "| stage | rel_l2 | corr | max_abs_error | sign_flip_rel_l2 | conj_rel_l2 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, metrics in report["stage_metrics"].items():
        if not metrics.get("available"):
            lines.append(f"| {key} | unavailable |  |  |  |  |")
            continue
        lines.append(
            "| {key} | {rel:.6g} | {corr:.6g} | {maxe:.6g} | {neg:.6g} | {conj:.6g} |".format(
                key=key,
                rel=float(metrics.get("rel_l2", float("nan"))),
                corr=float(metrics.get("corr_abs_complex", float("nan"))),
                maxe=float(metrics.get("max_abs_error", float("nan"))),
                neg=float(
                    metrics.get("candidate_is_negative_reference_rel_l2", float("nan"))
                ),
                conj=float(metrics.get("candidate_conjugate_rel_l2", float("nan"))),
            )
        )
    app_metrics = report.get("rm_application_metrics", {})
    if app_metrics.get("available"):
        lines.extend(
            [
                "",
                "## RM Application Metrics",
                "",
                "| check | rel_l2 |",
                "|---|---:|",
                "| EIDORS RM @ EIDORS dV vs EIDORS rec_delta | {value:.6g} |".format(
                    value=float(app_metrics["eidors_self_rel_l2"])
                ),
                "| PyEIDORS RM @ PyEIDORS dV vs PyEIDORS rec_delta | {value:.6g} |".format(
                    value=float(app_metrics["pyeidors_self_rel_l2"])
                ),
                "| EIDORS RM @ PyEIDORS dV vs PyEIDORS rec_delta | {value:.6g} |".format(
                    value=float(app_metrics["eidors_rm_on_pyeidors_dv_rel_l2"])
                ),
                "| PyEIDORS RM @ EIDORS dV vs EIDORS rec_delta | {value:.6g} |".format(
                    value=float(app_metrics["pyeidors_rm_on_eidors_dv_rel_l2"])
                ),
                "",
                "- RM Frobenius norms: EIDORS={eidors:.6g}, PyEIDORS={py:.6g}".format(
                    eidors=float(app_metrics["eidors_rm_norm"]),
                    py=float(app_metrics["pyeidors_rm_norm"]),
                ),
                "- dV norms: EIDORS={eidors:.6g}, PyEIDORS={py:.6g}".format(
                    eidors=float(app_metrics["eidors_dv_norm"]),
                    py=float(app_metrics["pyeidors_dv_norm"]),
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Difference Candidates",
            "",
            "- first numerical difference (>1e-6 rel_l2): "
            f"`{report.get('first_numerical_difference_candidate')}`",
            "- first meaningful divergence (>1e-3 rel_l2): "
            f"`{report.get('first_divergence_candidate')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help="Output root for payload/result/report files.",
    )
    parser.add_argument(
        "--stage",
        choices=("export", "pyeidors", "compare", "all-python"),
        default="all-python",
        help="Diagnostic stage to run.",
    )
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.18)
    parser.add_argument("--height", type=float, default=0.16)
    parser.add_argument("--target-radius", type=float, default=0.063)
    parser.add_argument("--base-sigma", type=complex, default=1.0 + 2.0j)
    parser.add_argument("--target-sigma", type=complex, default=2.0 + 3.0j)
    parser.add_argument("--contact-impedance", type=complex, default=0.01 + 0.05j)
    parser.add_argument(
        "--skip-jacobian",
        action="store_true",
        help="Only run mesh/pattern/forward/difference stages.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = ComplexCase(
        refinement=args.refinement,
        radius=args.radius,
        height=args.height,
        target_radius=args.target_radius,
        base_sigma=args.base_sigma,
        target_sigma=args.target_sigma,
        contact_impedance=args.contact_impedance,
    )
    out_root = args.out_root.resolve()
    if args.stage in {"export", "all-python"}:
        export_payload(case, out_root)
    if args.stage in {"pyeidors", "all-python"}:
        run_pyeidors(case, out_root, skip_jacobian=args.skip_jacobian)
    if args.stage in {"compare", "all-python"}:
        compare_results(case, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
