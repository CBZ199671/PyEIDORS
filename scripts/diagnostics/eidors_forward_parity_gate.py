#!/usr/bin/env python3
"""Forward-parity gate using the exact EIDORS stimulation/measurement matrices."""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.common.array_metrics import finite_pearson_correlation

from pyeidors import EITSystem
from pyeidors.data.difference import build_difference_vector
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.geometry.mesh3d_generator import create_cylinder_3d_eit_mesh
from pyeidors.runtime_paths import pyeidors_output_path

DEFAULT_DATA_DIR = pyeidors_output_path("eidors_same_pyeidors_mesh")


@dataclass(frozen=True)
class VectorFit:
    corr: float
    rmse: float
    scale: float
    scaled_rmse: float


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


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    return finite_pearson_correlation(a, b, min_count=3)


def vector_fit(reference: np.ndarray, candidate: np.ndarray) -> VectorFit:
    ref = np.asarray(reference, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    denom = float(np.dot(cand, cand))
    scale = float(np.dot(ref, cand) / denom) if denom > 0.0 else float("nan")
    return VectorFit(
        corr=safe_corr(ref, cand),
        rmse=float(np.sqrt(np.mean((cand - ref) ** 2))),
        scale=scale,
        scaled_rmse=float(np.sqrt(np.mean((scale * cand - ref) ** 2))),
    )


def read_vector(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=float).reshape(-1)


def build_mesh(payload: dict[str, np.ndarray]):
    radius = float(np.asarray(payload["radius"]).reshape(-1)[0])
    height = float(np.asarray(payload["height"]).reshape(-1)[0])
    refinement = int(np.asarray(payload["refinement"]).reshape(-1)[0])
    n_per_ring = int(np.asarray(payload["n_per_ring"]).reshape(-1)[0])
    n_rings = int(np.asarray(payload["n_rings"]).reshape(-1)[0])
    levels = tuple(float(v) for v in np.asarray(payload["ring_fractions"]).reshape(-1))
    return create_cylinder_3d_eit_mesh(
        n_elec=n_per_ring * n_rings,
        radius=radius,
        height=height,
        refinement=refinement,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        electrode_level_fractions=levels,
        z_center=float(np.asarray(payload["z_center"]).reshape(-1)[0]),
        mesh_family="tetra",
        geometry_version="geomv2",
        electrode_layout="ring_major",
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


def _stack_measurement_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.empty((0, 0), dtype=float)
    arrays = [np.asarray(matrix, dtype=float) for matrix in matrices]
    n_cols = arrays[0].shape[1]
    total_rows = 0
    for matrix in arrays:
        if matrix.ndim != 2 or matrix.shape[1] != n_cols:
            raise ValueError("measurement matrices must be 2D with matching columns")
        total_rows += int(matrix.shape[0])
    out = np.empty((total_rows, n_cols), dtype=float)
    start = 0
    for matrix in arrays:
        stop = start + int(matrix.shape[0])
        out[start:stop, :] = matrix
        start = stop
    return out


def build_custom_pattern(payload: dict[str, np.ndarray]) -> PatternConfig:
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


def verify_pattern_manager(system, payload: dict[str, np.ndarray]) -> dict[str, object]:
    stim, _, meas_concat, _, counts = payload_measurement_matrices(payload)
    manager = system.fwd_model.pattern_manager
    actual_concat = _stack_measurement_matrices(manager.meas_matrices)
    actual_counts = np.asarray(manager.n_meas_per_stim, dtype=np.int64)
    return {
        "stim_exact": bool(np.allclose(manager.stim_matrix, stim)),
        "meas_exact": bool(np.allclose(actual_concat, meas_concat)),
        "counts_exact": bool(np.array_equal(actual_counts, counts)),
        "n_stim": int(manager.n_stim),
        "n_meas_total": int(manager.n_meas_total),
        "meas_count_min": int(actual_counts.min()),
        "meas_count_max": int(actual_counts.max()),
    }


def per_stim_corr(
    reference: np.ndarray, candidate: np.ndarray, starts: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    values = []
    for start, count in zip(starts, counts, strict=True):
        rows = slice(int(start), int(start + count))
        values.append(safe_corr(reference[rows], candidate[rows]))
    return np.asarray(values, dtype=float)


def solve_pyeidors(payload: dict[str, np.ndarray], out_dir: Path) -> dict[str, object]:
    mesh = build_mesh(payload)
    base_sigma = float(np.asarray(payload["base_sigma"]).reshape(-1)[0])
    truth_sigma = np.asarray(payload["truth_elem_data"], dtype=float).reshape(-1)
    total_electrodes = int(np.asarray(payload["total_electrodes"]).reshape(-1)[0])
    contact_z = float(np.asarray(payload["contact_impedance"]).reshape(-1)[0])
    device = os.environ.get("PYEIDORS_PARITY_DEVICE", "cpu").strip().lower() or "cpu"

    system = EITSystem(
        n_elec=total_electrodes,
        pattern_config=build_custom_pattern(payload),
        contact_impedance=np.full(total_electrodes, contact_z, dtype=float),
        base_conductivity=base_sigma,
        regularization_type="noser",
        regularization_alpha=1.0,
        hyperparameter=1e-2,
        noser_exponent=0.5,
        difference_mode="normalized",
        difference_step_size_mode="off",
        petsc_device=device,
        device=device,
        solver_mode="fast",
        line_search_mode="fast",
        forward_backend="dolfinx",
        mesh_family="tetra",
        cache_dir=str(out_dir / ".pyeidors_cache"),
    )
    setup_start = time.perf_counter()
    system.setup(mesh=mesh)
    pattern_check = verify_pattern_manager(system, payload)

    bg = EITImage(
        elem_data=np.full(mesh.num_cells(), base_sigma, dtype=float),
        fwd_model=system.fwd_model,
    )
    truth = EITImage(elem_data=truth_sigma, fwd_model=system.fwd_model)
    forward_start = time.perf_counter()
    vh = system.forward_solve(bg)
    vi = system.forward_solve(truth)
    forward_seconds = time.perf_counter() - forward_start
    setup_seconds = forward_start - setup_start

    dv_target_minus_reference = build_difference_vector(
        vi.meas,
        vh.meas,
        mode="normalized",
        orientation="target_minus_reference",
    )
    dv_reference_minus_target = build_difference_vector(
        vi.meas,
        vh.meas,
        mode="normalized",
        orientation="reference_minus_target",
    )
    np.savetxt(out_dir / "forward_parity_pyeidors_vh.csv", vh.meas, delimiter=",")
    np.savetxt(out_dir / "forward_parity_pyeidors_vi.csv", vi.meas, delimiter=",")
    np.savetxt(
        out_dir / "forward_parity_pyeidors_dv_target_minus_reference.csv",
        dv_target_minus_reference,
        delimiter=",",
    )
    return {
        "system": system,
        "vh": vh.meas,
        "vi": vi.meas,
        "dv_target_minus_reference": dv_target_minus_reference,
        "dv_reference_minus_target": dv_reference_minus_target,
        "setup_seconds": setup_seconds,
        "forward_seconds": forward_seconds,
        "device": device,
        "pattern_check": pattern_check,
    }


def make_figure(
    out_dir: Path,
    eidors_dv: np.ndarray,
    pyeidors_dv: np.ndarray,
    per_block_corr: np.ndarray,
    metrics: dict[str, object],
    *,
    device_slug: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=180)
    x = np.arange(eidors_dv.size)
    axes[0, 0].plot(x, eidors_dv, color="#00a7a8", linewidth=0.75, label="EIDORS")
    axes[0, 0].plot(
        x,
        pyeidors_dv,
        color="#d84b3a",
        linewidth=0.65,
        alpha=0.82,
        label="PyEIDORS exact protocol",
    )
    axes[0, 0].set_title("归一化边界电压差")
    axes[0, 0].set_xlabel("Measurement index", fontname="Times New Roman")
    axes[0, 0].set_ylabel("Normalized voltage difference", fontname="Times New Roman")
    axes[0, 0].legend(prop={"family": "Times New Roman", "size": 9})
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].scatter(eidors_dv, pyeidors_dv, s=8, alpha=0.45, color="#2f66c5")
    lim = float(max(np.max(np.abs(eidors_dv)), np.max(np.abs(pyeidors_dv)), 1.0))
    axes[0, 1].plot(
        [-lim, lim], [-lim, lim], color="0.2", linewidth=0.9, linestyle="--"
    )
    axes[0, 1].set_xlim(-lim, lim)
    axes[0, 1].set_ylim(-lim, lim)
    axes[0, 1].set_title("逐测量散点")
    axes[0, 1].set_xlabel("EIDORS", fontname="Times New Roman")
    axes[0, 1].set_ylabel("PyEIDORS", fontname="Times New Roman")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(
        np.arange(per_block_corr.size),
        per_block_corr,
        color="#1f77b4",
        linewidth=1.0,
    )
    axes[1, 0].axhline(0.99, color="#d84b3a", linestyle="--", linewidth=1.0)
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[1, 0].set_title("每次激励内相关系数")
    axes[1, 0].set_xlabel("Stimulation index", fontname="Times New Roman")
    axes[1, 0].set_ylabel("Correlation", fontname="Times New Roman")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].axis("off")
    fit = metrics["target_minus_reference"]
    summary = (
        "前向一致性门禁\n"
        f"stim/meas 矩阵完全一致：{metrics['pattern_exact']}\n"
        f"测量数：{metrics['n_meas_total']}，激励数：{metrics['n_stim']}\n"
        f"PyEIDORS 设备：{metrics['device']}，正问题：{metrics['forward_seconds']:.3f}s\n\n"
        "目标：corr > 0.99\n"
        f"当前 corr：{fit['corr']:.6f}\n"
        f"RMSE：{fit['rmse']:.6e}\n"
        f"最佳尺度因子：{fit['scale']:.6e}\n"
        f"尺度校正后 RMSE：{fit['scaled_rmse']:.6e}\n"
        f"状态：{metrics['gate_status']}\n"
    )
    axes[1, 1].text(
        0.02,
        0.98,
        summary,
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.45,
        transform=axes[1, 1].transAxes,
    )
    for ax in axes.ravel()[:3]:
        apply_times_ticks(ax)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"forward_parity_exact_protocol_{device_slug}.png",
        bbox_inches="tight",
    )
    fig.savefig(out_dir / "forward_parity_exact_protocol.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_fonts()
    data_dir = Path(
        os.environ.get("PYEIDORS_PARITY_DATA_DIR", DEFAULT_DATA_DIR)
    ).resolve()
    out_dir = Path(os.environ.get("PYEIDORS_PARITY_OUTPUT_DIR", data_dir)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = loadmat(data_dir / "pyeidors_same_tetra_mesh.mat")
    _, _, _, starts, counts = payload_measurement_matrices(payload)

    eidors_vh = read_vector(data_dir / "same_mesh_vh_background.csv")
    eidors_vi = read_vector(data_dir / "same_mesh_vi_sphere.csv")
    eidors_dv = read_vector(data_dir / "same_mesh_dv_measured_normalized.csv")
    py = solve_pyeidors(payload, out_dir)
    device_slug = str(py["device"]).replace("/", "_")

    fits = {
        "vh_background": vector_fit(eidors_vh, py["vh"]),
        "vi_sphere": vector_fit(eidors_vi, py["vi"]),
        "target_minus_reference": vector_fit(
            eidors_dv, py["dv_target_minus_reference"]
        ),
        "reference_minus_target": vector_fit(
            eidors_dv, py["dv_reference_minus_target"]
        ),
    }
    primary = fits["target_minus_reference"]
    block_corr = per_stim_corr(
        eidors_dv, py["dv_target_minus_reference"], starts, counts
    )
    pattern_check = py["pattern_check"]
    pattern_exact = bool(
        pattern_check["stim_exact"]
        and pattern_check["meas_exact"]
        and pattern_check["counts_exact"]
    )
    gate_threshold = 0.99
    gate_status = (
        "PASS"
        if pattern_exact
        and np.isfinite(primary.corr)
        and primary.corr >= gate_threshold
        else "FAIL"
    )

    metrics = {
        "gate_threshold": gate_threshold,
        "gate_status": gate_status,
        "pattern_exact": pattern_exact,
        "pattern_check": pattern_check,
        "device": py["device"],
        "setup_seconds": py["setup_seconds"],
        "forward_seconds": py["forward_seconds"],
        "n_stim": int(pattern_check["n_stim"]),
        "n_meas_total": int(pattern_check["n_meas_total"]),
        "target_minus_reference": fits["target_minus_reference"].__dict__,
        "reference_minus_target": fits["reference_minus_target"].__dict__,
        "vh_background": fits["vh_background"].__dict__,
        "vi_sphere": fits["vi_sphere"].__dict__,
        "per_stim_corr_min": float(np.nanmin(block_corr)),
        "per_stim_corr_median": float(np.nanmedian(block_corr)),
        "per_stim_corr_max": float(np.nanmax(block_corr)),
    }
    metrics_text = json.dumps(metrics, indent=2, ensure_ascii=False)
    (out_dir / f"forward_parity_exact_protocol_metrics_{device_slug}.json").write_text(
        metrics_text,
        encoding="utf-8",
    )
    (out_dir / "forward_parity_exact_protocol_metrics.json").write_text(
        metrics_text,
        encoding="utf-8",
    )
    np.savetxt(
        out_dir / f"forward_parity_per_stim_corr_{device_slug}.csv",
        block_corr,
        delimiter=",",
    )
    np.savetxt(out_dir / "forward_parity_per_stim_corr.csv", block_corr, delimiter=",")
    make_figure(
        out_dir,
        eidors_dv,
        py["dv_target_minus_reference"],
        block_corr,
        metrics,
        device_slug=device_slug,
    )
    print(metrics_text)


if __name__ == "__main__":
    main()
