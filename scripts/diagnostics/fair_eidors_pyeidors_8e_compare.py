#!/usr/bin/env python3
"""Fair 8-electrode PyEIDORS CPU/CUDA vs EIDORS visual comparison."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.io import loadmat, savemat

try:  # pragma: no cover - depends on the CUDA shell.
    import torch
except Exception:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
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
from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
from pyeidors.interop.geometry_exchange import (
    build_boundary_edges,
    build_electrode_arrays,
)
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsJacobianAdapter
from pyeidors.inverse.reconstruction_matrix import build_one_step_rm


OUT_ROOT = ROOT / "results" / "eidors_fair_8e_layers"


@dataclass(frozen=True)
class CaseConfig:
    name: str
    dim: int
    n_per_ring: int
    n_rings: int
    radius: float = 0.18
    height: float = 0.50
    refinement: int = 3
    mesh_family: str = "tetra"
    base_sigma: float = 1.0
    target_sigma: float = 2.0
    target_radius: float = 0.065
    target_center: tuple[float, float, float] = (0.045, -0.025, 0.0)
    contact_impedance: float = 1.0e-5
    hyperparameter: float = 1.0e-2

    @property
    def total_electrodes(self) -> int:
        return int(self.n_per_ring * self.n_rings)

    @property
    def ring_fractions(self) -> tuple[float, ...]:
        if self.n_rings <= 1:
            return (0.5,)
        if self.n_rings == 2:
            return (0.25, 0.75)
        return tuple(np.linspace(0.15, 0.85, self.n_rings).tolist())


CASES = (
    CaseConfig(
        name="2d_8e",
        dim=2,
        n_per_ring=8,
        n_rings=1,
        refinement=5,
        height=0.0,
        target_radius=0.055,
        target_center=(0.050, -0.030, 0.0),
    ),
    CaseConfig(
        name="3d_8x2",
        dim=3,
        n_per_ring=8,
        n_rings=2,
        refinement=2,
        target_radius=0.090,
        target_center=(0.040, -0.025, 0.0),
    ),
    CaseConfig(
        name="3d_8x3",
        dim=3,
        n_per_ring=8,
        n_rings=3,
        refinement=2,
        target_radius=0.090,
        target_center=(0.040, -0.025, 0.0),
    ),
)


def case_by_name(name: str) -> CaseConfig:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(name)


def configure_fonts() -> None:
    for font_path in [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/mnt/c/Windows/Fonts/simhei.ttf"),
    ]:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
    available = {font.name for font in fm.fontManager.ttflist}
    chinese = next(
        (
            name
            for name in [
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "SimHei",
                "WenQuanYi Zen Hei",
            ]
            if name in available
        ),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": [chinese, "Times New Roman", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
        }
    )


def apply_times_ticks(ax) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")
    if hasattr(ax, "get_zticklabels"):
        for label in ax.get_zticklabels():
            label.set_fontname("Times New Roman")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 3:
        return float("nan")
    if np.std(aa[mask]) <= np.finfo(float).eps:
        return float("nan")
    if np.std(bb[mask]) <= np.finfo(float).eps:
        return float("nan")
    return float(np.corrcoef(aa[mask], bb[mask])[0, 1])


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).reshape(-1)
    bb = np.asarray(b, dtype=float).reshape(-1)
    return float(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), np.finfo(float).eps))


def boundary_facets_3d(eit_mesh) -> np.ndarray:
    mesh = eit_mesh.mesh
    tags = eit_mesh.facet_tags
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    if f2v is None:
        raise RuntimeError("mesh has no facet-to-vertex connectivity")
    rows: list[np.ndarray] = []
    for facet in np.asarray(tags.indices, dtype=np.int64).reshape(-1):
        verts = np.asarray(f2v.links(int(facet)), dtype=np.int64).reshape(-1)
        if verts.size == 3:
            rows.append(verts + 1)
    if not rows:
        raise RuntimeError("no triangular boundary facets were found")
    return np.vstack(rows).astype(np.int64)


def build_case_mesh(case: CaseConfig, case_dir: Path):
    mesh_dir = case_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    if case.dim == 2:
        return create_eit_mesh(
            n_elec=case.n_per_ring,
            radius=case.radius,
            refinement=case.refinement,
            electrode_coverage=0.5,
            output_dir=str(mesh_dir),
            mesh_name=f"{case.name}_circle",
        )
    return create_cylinder_3d_eit_mesh(
        n_elec=case.total_electrodes,
        radius=case.radius,
        height=case.height,
        refinement=case.refinement,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        electrode_level_fractions=case.ring_fractions,
        z_center=0.0,
        mesh_family=case.mesh_family,
        geometry_version="geomv2",
        electrode_layout="ring_major",
    )


def build_truth(case: CaseConfig, mesh) -> tuple[np.ndarray, np.ndarray]:
    centres = cell_midpoints(mesh.mesh)
    center = np.asarray(case.target_center[: centres.shape[1]], dtype=float)
    if case.dim == 2:
        dist = np.linalg.norm(centres[:, :2] - center[:2].reshape(1, 2), axis=1)
    else:
        dist = np.linalg.norm(centres[:, :3] - center[:3].reshape(1, 3), axis=1)
    mask = dist <= case.target_radius
    truth = np.full(mesh.num_cells(), case.base_sigma, dtype=float)
    truth[mask] = case.target_sigma
    return truth, mask


def build_pattern_config(case: CaseConfig) -> PatternConfig:
    protocol = "eidors_full_3d" if case.dim == 2 else "hybrid_full_3d"
    return PatternConfig(
        n_elec=case.n_per_ring,
        n_rings=case.n_rings,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        measurement_protocol=protocol,
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
        n_elec=int(np.asarray(payload["n_per_ring"]).reshape(-1)[0]),
        n_rings=int(np.asarray(payload["n_rings"]).reshape(-1)[0]),
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


def export_case(case: CaseConfig, out_root: Path) -> None:
    case_dir = out_root / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    mesh = build_case_mesh(case, case_dir)
    cells = np.asarray(mesh.cells(), dtype=np.int64)
    truth, truth_mask = build_truth(case, mesh)
    electrode_nodes, electrode_counts = build_electrode_arrays(mesh)
    boundary = build_boundary_edges(mesh) if case.dim == 2 else boundary_facets_3d(mesh)

    pattern = build_pattern_config(case)
    pm = StimMeasPatternManager(pattern, mesh_tdim=case.dim)
    meas_counts = np.asarray(pm.n_meas_per_stim, dtype=np.int64)
    meas_start = np.concatenate([[0], np.cumsum(meas_counts[:-1])]).astype(np.int64)

    nodes = np.asarray(mesh.coordinates(), dtype=float)
    if case.dim == 2 and nodes.shape[1] > 2:
        nodes = nodes[:, :2]
    payload = {
        "case_name": case.name,
        "dim": case.dim,
        "nodes": nodes,
        "elems": cells + 1,
        "boundary": boundary,
        "boundary_edges": boundary,
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "truth_elem_data": truth.reshape(-1, 1),
        "truth_mask": truth_mask.reshape(-1, 1),
        "stim_matrix": np.asarray(pm.stim_matrix, dtype=float),
        "meas_matrix_concat": np.vstack(pm.meas_matrices).astype(float),
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
        "base_sigma": case.base_sigma,
        "target_sigma": case.target_sigma,
        "target_radius": case.target_radius,
        "target_center": np.asarray(case.target_center, dtype=float).reshape(1, 3),
        "contact_impedance": case.contact_impedance,
        "hyperparameter": case.hyperparameter,
        "measurement_protocol": "eidors_full_3d" if case.dim == 2 else "hybrid_full_3d",
        "source": "pyeidors_fair_8e_compare",
    }
    savemat(case_dir / "payload.mat", payload, do_compression=True)
    print(
        f"[export] {case.name}: nodes={nodes.shape[0]} elems={cells.shape[0]} "
        f"stim={pm.n_stim} meas={pm.n_meas_total}"
    )


def verify_mesh_order(payload: dict[str, np.ndarray], mesh, case: CaseConfig) -> bool:
    payload_nodes = np.asarray(payload["nodes"], dtype=float)
    mesh_nodes = np.asarray(mesh.coordinates(), dtype=float)
    if case.dim == 2 and mesh_nodes.shape[1] > 2:
        mesh_nodes = mesh_nodes[:, :2]
    payload_elems = np.asarray(payload["elems"], dtype=np.int64)
    mesh_elems = np.asarray(mesh.cells(), dtype=np.int64) + 1
    return bool(
        payload_nodes.shape == mesh_nodes.shape
        and payload_elems.shape == mesh_elems.shape
        and np.allclose(payload_nodes, mesh_nodes)
        and np.array_equal(payload_elems, mesh_elems)
    )


def run_pyeidors_case(
    case: CaseConfig, out_root: Path, device: str
) -> dict[str, object]:
    case_dir = out_root / case.name
    payload = loadmat(case_dir / "payload.mat")
    mesh = build_case_mesh(case, case_dir)
    if not verify_mesh_order(payload, mesh, case):
        raise RuntimeError(f"{case.name}: regenerated PyEIDORS mesh order changed")
    truth = np.asarray(payload["truth_elem_data"], dtype=float).reshape(-1)
    base_sigma = float(np.asarray(payload["base_sigma"]).reshape(-1)[0])
    total_electrodes = int(np.asarray(payload["total_electrodes"]).reshape(-1)[0])
    contact_z = float(np.asarray(payload["contact_impedance"]).reshape(-1)[0])
    hp = float(np.asarray(payload["hyperparameter"]).reshape(-1)[0])
    pattern = exact_pattern_from_payload(payload)

    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    system = EITSystem(
        n_elec=total_electrodes,
        pattern_config=pattern,
        contact_impedance=np.full(total_electrodes, contact_z, dtype=float),
        base_conductivity=base_sigma,
        regularization_type="noser",
        regularization_alpha=1.0,
        hyperparameter=hp,
        noser_exponent=0.5,
        difference_mode="normalized",
        difference_step_size_mode="off",
        petsc_device=device,
        device=device,
        solver_mode="fast",
        line_search_mode="fast",
        forward_backend="dolfinx",
        mesh_family=case.mesh_family,
        cache_dir=str(case_dir / f".pyeidors_cache_{device}"),
    )
    setup_start = time.perf_counter()
    system.setup(mesh=mesh)
    setup_seconds = time.perf_counter() - setup_start

    bg = EITImage(
        elem_data=np.full(mesh.num_cells(), base_sigma, dtype=float),
        fwd_model=system.fwd_model,
    )
    target = EITImage(elem_data=truth, fwd_model=system.fwd_model)

    t0 = time.perf_counter()
    vh = system.forward_solve(bg)
    vi = system.forward_solve(target)
    forward_seconds = time.perf_counter() - t0
    dv_truth = build_difference_vector(
        vi.meas,
        vh.meas,
        mode="normalized",
        orientation="target_minus_reference",
    )

    eidors_dv_path = case_dir / "eidors_dv_measured_normalized.csv"
    forward_parity_corr = float("nan")
    if eidors_dv_path.exists():
        eidors_dv = np.loadtxt(eidors_dv_path, delimiter=",").reshape(-1)
        forward_parity_corr = safe_corr(eidors_dv, dv_truth)

    t1 = time.perf_counter()
    jacobian_path = case_dir / f"pyeidors_{device}_projected_jacobian.npy"
    use_torch_jacobian = device == "cuda"
    jacobian_device = device if use_torch_jacobian else "cpu"
    if jacobian_path.exists():
        jacobian = np.load(jacobian_path)
    else:
        jac_calc = EidorsJacobianAdapter(
            system.fwd_model,
            use_torch=use_torch_jacobian,
            device=jacobian_device,
            torch_dtype="float64",
            torch_batch_all=use_torch_jacobian,
        )
        raw_jacobian = jac_calc.calculate_from_image(bg)
        jacobian = project_measurement_jacobian(
            raw_jacobian,
            measurement_type="difference",
            reference_meas=vh.meas,
            difference_mode="normalized",
            difference_orientation="target_minus_reference",
        )
        np.save(jacobian_path, jacobian)
    rm = build_one_step_rm(jacobian, lambda_=hp, mode="noser", form="param")
    delta = np.asarray(rm @ dv_truth, dtype=np.float64).reshape(-1)
    sigma = base_sigma + delta
    inverse_seconds = time.perf_counter() - t1

    pred = system.forward_solve(
        EITImage(elem_data=np.maximum(sigma, 1.0e-6), fwd_model=system.fwd_model)
    )
    dv_pred = build_difference_vector(
        pred.meas,
        vh.meas,
        mode="normalized",
        orientation="target_minus_reference",
    )

    truth_delta = truth - base_sigma
    metrics = {
        "case": case.name,
        "device": device,
        "setup_seconds": setup_seconds,
        "forward_seconds": forward_seconds,
        "inverse_seconds": inverse_seconds,
        "cond_rel_l2": rel_l2(delta, truth_delta),
        "cond_corr": safe_corr(delta, truth_delta),
        "voltage_rmse": float(np.sqrt(np.mean((dv_pred - dv_truth) ** 2))),
        "voltage_corr": safe_corr(dv_pred, dv_truth),
        "forward_parity_corr_vs_eidors": forward_parity_corr,
        "n_meas": int(dv_truth.size),
        "jacobian_device": jacobian_device,
        "torch_jacobian": bool(use_torch_jacobian),
        "backend_diagnostics": system.fwd_model.get_backend_diagnostics(),
    }
    np.savez_compressed(
        case_dir / f"pyeidors_{device}_result.npz",
        sigma=sigma,
        delta=delta,
        dv_truth=dv_truth,
        dv_pred=dv_pred,
        vh=np.asarray(vh.meas, dtype=float),
        vi=np.asarray(vi.meas, dtype=float),
        metrics=np.asarray(json.dumps(metrics, ensure_ascii=False), dtype=object),
    )
    (case_dir / f"pyeidors_{device}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[pyeidors:{device}] {case.name}: forward={forward_seconds:.3f}s "
        f"inverse={inverse_seconds:.3f}s cond_corr={metrics['cond_corr']:.4f} "
        f"Vcorr={metrics['voltage_corr']:.4f} parity={forward_parity_corr:.4f}"
    )
    return metrics


def load_pyeidors_result(case_dir: Path, device: str) -> dict[str, object]:
    data = np.load(case_dir / f"pyeidors_{device}_result.npz", allow_pickle=True)
    metrics = json.loads(str(data["metrics"].item()))
    return {
        "sigma": np.asarray(data["sigma"], dtype=float),
        "delta": np.asarray(data["delta"], dtype=float),
        "dv_truth": np.asarray(data["dv_truth"], dtype=float),
        "dv_pred": np.asarray(data["dv_pred"], dtype=float),
        "metrics": metrics,
    }


def load_eidors_result(case_dir: Path) -> dict[str, object]:
    mat = loadmat(case_dir / "eidors_result.mat")
    metrics = {
        "case": str(np.asarray(mat["case_name"]).reshape(-1)[0]),
        "device": "EIDORS/MATLAB CPU",
        "forward_seconds": float(np.asarray(mat["forward_seconds"]).reshape(-1)[0]),
        "inverse_seconds": float(np.asarray(mat["inverse_seconds"]).reshape(-1)[0]),
        "cond_rel_l2": float(np.asarray(mat["cond_rel_l2"]).reshape(-1)[0]),
        "cond_corr": float(np.asarray(mat["cond_corr"]).reshape(-1)[0]),
        "voltage_rmse": float(np.asarray(mat["fit_rmse"]).reshape(-1)[0]),
        "voltage_corr": float(np.asarray(mat["fit_corr"]).reshape(-1)[0]),
        "n_meas": int(np.asarray(mat["dv_meas"]).size),
    }
    return {
        "sigma": np.asarray(mat["rec_sigma"], dtype=float).reshape(-1),
        "delta": np.asarray(mat["rec_delta"], dtype=float).reshape(-1),
        "dv_truth": np.asarray(mat["dv_meas"], dtype=float).reshape(-1),
        "dv_pred": np.asarray(mat["dv_pred"], dtype=float).reshape(-1),
        "metrics": metrics,
    }


def panel_mask(delta: np.ndarray, truth_mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(delta, dtype=float).reshape(-1)
    if truth_mask is not None:
        return np.asarray(truth_mask, dtype=bool).reshape(-1)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        order = np.argsort(values)
        mask = np.zeros_like(values, dtype=bool)
        mask[order[-min(120, values.size) :]] = True
        return mask
    threshold = max(0.20 * float(np.max(positive)), float(np.percentile(positive, 82)))
    mask = values >= threshold
    if int(mask.sum()) < 12:
        mask = np.zeros_like(values, dtype=bool)
        mask[np.argsort(values)[-min(120, values.size) :]] = True
    return mask


def cylinder_context(ax, case: CaseConfig) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 96)
    z_min = -0.5 * case.height
    z_max = 0.5 * case.height
    tt, zz = np.meshgrid(theta, np.array([z_min, z_max]))
    ax.plot_surface(
        case.radius * np.cos(tt),
        case.radius * np.sin(tt),
        zz,
        color=(0.66, 0.70, 0.77),
        alpha=0.05,
        linewidth=0,
        shade=False,
    )
    for frac in case.ring_fractions:
        z = z_min + case.height * frac
        ax.plot(
            case.radius * np.cos(theta),
            case.radius * np.sin(theta),
            np.full_like(theta, z),
            color=(0.20, 0.26, 0.34, 0.40),
            linewidth=0.9,
        )


def plot_field_panel(ax, case: CaseConfig, payload, values, mask, title, clim):
    nodes = np.asarray(payload["nodes"], dtype=float)
    elems = np.asarray(payload["elems"], dtype=np.int64) - 1
    values = np.asarray(values, dtype=float).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if case.dim == 2:
        tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems[:, :3])
        plot_values = values.copy()
        plot_values[~mask] = np.nan
        artist = ax.tripcolor(
            tri,
            facecolors=plot_values,
            shading="flat",
            cmap="viridis",
            vmin=clim[0],
            vmax=clim[1],
        )
        circle = plt.Circle(
            (0, 0), case.radius, fill=False, color="0.45", linewidth=0.9
        )
        ax.add_patch(circle)
        ax.set_aspect("equal")
        ax.set_xlim(-case.radius, case.radius)
        ax.set_ylim(-case.radius, case.radius)
        ax.set_xlabel("X (m)", fontname="Times New Roman")
        ax.set_ylabel("Y (m)", fontname="Times New Roman")
    else:
        centres = np.asarray(payload["centres"], dtype=float)
        cylinder_context(ax, case)
        pts = centres[mask]
        artist = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=values[mask],
            s=24,
            cmap="viridis",
            vmin=clim[0],
            vmax=clim[1],
            alpha=0.88,
            edgecolors=(0.08, 0.08, 0.08, 0.14),
            linewidths=0.15,
        )
        ax.set_xlim(-case.radius, case.radius)
        ax.set_ylim(-case.radius, case.radius)
        ax.set_zlim(-0.5 * case.height, 0.5 * case.height)
        ax.set_box_aspect((1, 1, case.height / (2 * case.radius)))
        ax.view_init(elev=22, azim=38)
        ax.set_xlabel("X (m)", fontname="Times New Roman")
        ax.set_ylabel("Y (m)", fontname="Times New Roman")
        ax.set_zlabel("Z (m)", fontname="Times New Roman")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.24)
    apply_times_ticks(ax)
    return artist


def render_case(case: CaseConfig, out_root: Path) -> dict[str, object]:
    case_dir = out_root / case.name
    raw_payload = loadmat(case_dir / "payload.mat")
    payload = dict(raw_payload)
    mesh = build_case_mesh(case, case_dir)
    centres = cell_midpoints(mesh.mesh)
    payload["centres"] = centres
    truth = np.asarray(raw_payload["truth_elem_data"], dtype=float).reshape(-1)
    truth_delta = truth - case.base_sigma
    truth_mask = np.asarray(raw_payload["truth_mask"], dtype=bool).reshape(-1)
    eidors = load_eidors_result(case_dir)
    cpu = load_pyeidors_result(case_dir, "cpu")
    cuda = load_pyeidors_result(case_dir, "cuda")

    recon_values = np.concatenate([eidors["sigma"], cpu["sigma"], cuda["sigma"]])
    clim = (
        float(min(0.95, np.nanpercentile(recon_values, 2))),
        float(max(2.0, np.nanpercentile(recon_values, 98))),
    )
    panels = [
        ("真值", truth, panel_mask(truth_delta, truth_mask)),
        ("EIDORS\nNOSER", eidors["sigma"], panel_mask(eidors["delta"])),
        ("PyEIDORS CPU\nNOSER-RM", cpu["sigma"], panel_mask(cpu["delta"])),
        ("PyEIDORS CUDA\nNOSER-RM", cuda["sigma"], panel_mask(cuda["delta"])),
    ]

    subplot_kw = {"projection": "3d"} if case.dim == 3 else {}
    fig = plt.figure(figsize=(18, 10), dpi=180)
    axes = [fig.add_subplot(2, 4, idx + 1, **subplot_kw) for idx in range(4)]
    for ax, (title, values, mask) in zip(axes, panels, strict=True):
        artist = plot_field_panel(ax, case, payload, values, mask, title, clim)
    fig.colorbar(
        artist,
        ax=axes,
        shrink=0.66,
        pad=0.02,
        label=r"$\sigma$ (S/m)",
    )

    ax_v = fig.add_subplot(2, 2, 3)
    x = np.arange(eidors["dv_truth"].size)
    ax_v.plot(
        x, eidors["dv_truth"], color="#00a7a8", linewidth=0.85, label="EIDORS truth"
    )
    ax_v.plot(
        x,
        eidors["dv_pred"],
        color="#ff5a4f",
        linestyle="--",
        linewidth=0.85,
        label="EIDORS fit",
    )
    ax_v.plot(
        x,
        cpu["dv_pred"],
        color="#2667ff",
        linewidth=0.80,
        alpha=0.86,
        label="PyEIDORS CPU fit",
    )
    ax_v.plot(
        x,
        cuda["dv_pred"],
        color="#7a3cff",
        linewidth=0.80,
        alpha=0.86,
        label="PyEIDORS CUDA fit",
    )
    ax_v.set_title("边界电压拟合", fontweight="bold")
    ax_v.set_xlabel("Boundary-voltage difference index", fontname="Times New Roman")
    ax_v.set_ylabel("Normalized voltage difference", fontname="Times New Roman")
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(prop={"family": "Times New Roman", "size": 8})
    apply_times_ticks(ax_v)

    ax_text = fig.add_subplot(2, 2, 4)
    ax_text.axis("off")
    rows = [
        f"案例：{case.name}，维度：{case.dim}D",
        f"电极：{case.n_per_ring}/环 × {case.n_rings} 层，总计 {case.total_electrodes}",
        f"网格：节点 {np.asarray(raw_payload['nodes']).shape[0]}，单元 {np.asarray(raw_payload['elems']).shape[0]}",
        f"激励/测量：{np.asarray(raw_payload['stim_matrix']).shape[0]} / {np.asarray(raw_payload['meas_matrix_concat']).shape[0]}",
        f"目标：σ={case.target_sigma:g} S/m，背景 σ={case.base_sigma:g} S/m",
        "",
        "指标（重构相对真值 / 电压拟合）",
    ]
    for label, result in [
        ("EIDORS", eidors),
        ("Py CPU", cpu),
        ("Py CUDA", cuda),
    ]:
        m = result["metrics"]
        rows.append(
            f"{label}: corr={m['cond_corr']:.4f}, L2={m['cond_rel_l2']:.3f}, "
            f"Vcorr={m['voltage_corr']:.4f}, inv={m['inverse_seconds']:.2f}s"
        )
    rows.append("")
    rows.append(
        "前向一致性："
        f"CPU={cpu['metrics'].get('forward_parity_corr_vs_eidors', float('nan')):.4f}, "
        f"CUDA={cuda['metrics'].get('forward_parity_corr_vs_eidors', float('nan')):.4f}"
    )
    ax_text.text(
        0.02,
        0.98,
        "\n".join(rows),
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.45,
        color="#1f2d3d",
        transform=ax_text.transAxes,
    )
    fig.suptitle(
        f"公平对比：{case.name}（同网格、同 CEM、电极矩阵、电压排序）",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = case_dir / f"{case.name}_fair_reconstruction_compare.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": case.name,
        "figure": str(out_path),
        "eidors": eidors["metrics"],
        "pyeidors_cpu": cpu["metrics"],
        "pyeidors_cuda": cuda["metrics"],
    }
    (case_dir / "fair_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[render] {case.name}: {out_path}")
    return summary


def render_index(summaries: list[dict[str, object]], out_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.8), dpi=180)
    ax.axis("off")
    lines = ["8 电极公平对比实验汇总", ""]
    for summary in summaries:
        cpu = summary["pyeidors_cpu"]
        cuda = summary["pyeidors_cuda"]
        eidors = summary["eidors"]
        lines.append(
            f"{summary['case']}: "
            f"EIDORS corr={eidors['cond_corr']:.3f}, "
            f"CPU corr={cpu['cond_corr']:.3f}, CUDA corr={cuda['cond_corr']:.3f}; "
            f"前向一致性 CPU/CUDA={cpu.get('forward_parity_corr_vs_eidors', float('nan')):.3f}/"
            f"{cuda.get('forward_parity_corr_vs_eidors', float('nan')):.3f}"
        )
    ax.text(
        0.02,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=13,
        linespacing=1.55,
        transform=ax.transAxes,
    )
    out_path = out_root / "fair_8e_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    (out_root / "fair_8e_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[render] summary: {out_path}")


def selected_cases(names: list[str] | None) -> list[CaseConfig]:
    if not names:
        return list(CASES)
    return [case_by_name(name) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["export", "pyeidors", "render", "all-python"],
        default="all-python",
        help="export payloads, run PyEIDORS, render existing EIDORS/PyEIDORS outputs, or export+pyeidors+render",
    )
    parser.add_argument("--cases", nargs="*", help="Subset: 2d_8e 3d_8x2 3d_8x3")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--devices",
        nargs="*",
        default=["cpu", "cuda"],
        choices=["cpu", "cuda"],
        help="PyEIDORS devices for --stage pyeidors/all-python",
    )
    args = parser.parse_args()

    configure_fonts()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(args.cases)

    if args.stage in {"export", "all-python"}:
        for case in cases:
            export_case(case, out_root)

    if args.stage in {"pyeidors", "all-python"}:
        for case in cases:
            for device in args.devices:
                run_pyeidors_case(case, out_root, device)

    if args.stage in {"render", "all-python"}:
        summaries = [render_case(case, out_root) for case in cases]
        render_index(summaries, out_root)


if __name__ == "__main__":
    main()
