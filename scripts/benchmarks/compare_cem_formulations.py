#!/usr/bin/env python3
"""Compare classic and Robin-transconductance CEM formulations.

The PyEIDORS stage runs both formulations on one DOLFINx mesh.  CSV files
produced by the companion NGSolve and EIDORS scripts can then be supplied to
the aggregate stage.  Curves are kept in raw SI units; no fitted rescaling is
applied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pyeidors.data.structures import PatternConfig
from pyeidors.forward import EITForwardModel, RobinTransconductanceForwardModel
from pyeidors.geometry.optimized_mesh_generator import create_eit_mesh
from pyeidors.interop.geometry_exchange import (
    STANDARD_INTEROP_FORMAT,
    build_electrode_arrays,
    save_exchange_mat,
)

try:
    from scripts.benchmarks.cem_fair_common import (
        MESH_FINGERPRINT_SCHEMA,
        benchmark_preassembled_blocks,
        canonical_mesh_fingerprint,
        validate_solver_reports,
        write_gmsh22,
    )
except ModuleNotFoundError:  # direct execution from scripts/benchmarks
    from cem_fair_common import (  # type: ignore[no-redef]
        MESH_FINGERPRINT_SCHEMA,
        benchmark_preassembled_blocks,
        canonical_mesh_fingerprint,
        validate_solver_reports,
        write_gmsh22,
    )


CSV_FIELDS = (
    "solver",
    "formulation",
    "mode",
    "spatial_frequency",
    "current_norm_a",
    "voltage_norm_v",
    "characteristic_resistance_ohm",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    n_electrodes: int = 16
    radius_m: float = 4.0
    conductivity_s_per_m: float = 0.25
    contact_impedance: float = 1.0
    electrode_coverage: float = 0.7
    mesh_refinement: int = 3
    potential_order: int = 1
    timing_repeats: int = 11


def trigonometric_current_patterns(
    n_electrodes: int,
    coverage: float,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Return paper-style cosine/sine current columns and their labels."""

    count = int(n_electrodes)
    frequencies = np.arange(1, count // 2 + 1, dtype=float)
    electrode_index = np.arange(count, dtype=float)
    mid_angles = 2.0 * np.pi * (electrode_index + float(coverage) / 2.0) / count
    cosine = np.cos(mid_angles[:, None] * frequencies[None, :])
    sine = np.sin(mid_angles[:, None] * frequencies[None, :])
    patterns = np.column_stack((cosine, sine))
    patterns -= np.mean(patterns, axis=0, keepdims=True)
    labels = [
        *(("cosine", int(k)) for k in frequencies),
        *(("sine", int(k)) for k in frequencies),
    ]
    return patterns, labels


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_array = np.asarray(candidate)
    reference_array = np.asarray(reference)
    denominator = max(float(np.linalg.norm(reference_array)), np.finfo(float).eps)
    return float(np.linalg.norm(candidate_array - reference_array) / denominator)


def characteristic_rows(
    solver: str,
    formulation: str,
    currents: np.ndarray,
    voltages: np.ndarray,
    labels: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Build raw characteristic-resistance rows for one voltage matrix."""

    current_matrix = np.asarray(currents)
    voltage_matrix = np.asarray(voltages)
    if current_matrix.shape != voltage_matrix.shape:
        raise ValueError("current and voltage matrices must have matching shapes")
    if current_matrix.shape[1] != len(labels):
        raise ValueError("one mode/frequency label is required per pattern column")

    rows: list[dict[str, Any]] = []
    for column, (mode, frequency) in enumerate(labels):
        current_norm = float(np.linalg.norm(current_matrix[:, column]))
        voltage_norm = float(np.linalg.norm(voltage_matrix[:, column]))
        if current_norm <= np.finfo(float).eps:
            raise ValueError(f"zero current norm for {mode} mode k={frequency}")
        rows.append(
            {
                "solver": str(solver),
                "formulation": str(formulation),
                "mode": str(mode),
                "spatial_frequency": int(frequency),
                "current_norm_a": current_norm,
                "voltage_norm_v": voltage_norm,
                "characteristic_resistance_ohm": voltage_norm / current_norm,
            }
        )
    return rows


def _extract_tagged_boundary_edges(
    eit_mesh,
    electrode_tags: list[int],
) -> np.ndarray:
    mesh = eit_mesh.mesh
    fdim = int(mesh.topology.dim) - 1
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    if facet_to_vertex is None:
        raise ValueError("mesh is missing facet-to-vertex connectivity")
    tag_to_label = {
        int(tag): electrode for electrode, tag in enumerate(electrode_tags, start=1)
    }
    rows: list[tuple[int, int, int]] = []
    for facet, marker in zip(
        eit_mesh.facet_tags.indices,
        eit_mesh.facet_tags.values,
        strict=True,
    ):
        vertices = np.asarray(facet_to_vertex.links(int(facet)), dtype=np.int64)
        if vertices.size != 2:
            continue
        rows.append(
            (
                int(vertices[0]),
                int(vertices[1]),
                int(tag_to_label.get(int(marker), 0)),
            )
        )
    if not rows:
        raise ValueError("mesh has no tagged boundary edges")
    return np.asarray(rows, dtype=np.int64)


def _export_common_mesh(
    config: BenchmarkConfig,
    output_dir: Path,
    eit_mesh,
    electrode_tags: list[int],
    currents: np.ndarray,
) -> dict[str, Any]:
    common_dir = output_dir / "common_mesh"
    nodes = np.asarray(eit_mesh.coordinates(), dtype=np.float64)[:, :2]
    cells = np.asarray(eit_mesh.cells(), dtype=np.int64)
    tagged_edges = _extract_tagged_boundary_edges(eit_mesh, electrode_tags)
    fingerprint = canonical_mesh_fingerprint(nodes, cells, tagged_edges)
    electrode_nodes, electrode_counts = build_electrode_arrays(eit_mesh)
    msh_path = common_dir / "cem_common_p1.msh"
    mat_path = common_dir / "cem_common_p1.mat"
    json_path = common_dir / "cem_common_p1.json"
    write_gmsh22(
        msh_path,
        nodes,
        cells,
        tagged_edges,
        config.n_electrodes,
    )
    payload = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "source_framework": "PyEIDORS/DOLFINx",
        "nodes": nodes,
        "elems": cells + 1,
        "boundary_edges": tagged_edges[:, :2] + 1,
        "tagged_boundary_edges": np.column_stack(
            (tagged_edges[:, :2] + 1, tagged_edges[:, 2])
        ),
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": config.n_electrodes,
        "background": config.conductivity_s_per_m,
        "truth_elem_data": np.full(cells.shape[0], config.conductivity_s_per_m),
        "contact_impedance": config.contact_impedance,
        "mesh_name": "cem_common_p1",
        "mesh_level": f"refinement_{config.mesh_refinement}",
        "scenario_name": "homogeneous_cem_formulation_comparison",
        "electrode_coverage": config.electrode_coverage,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "current_patterns": np.asarray(currents, dtype=np.float64),
    }
    save_exchange_mat(mat_path, payload)
    metadata = {
        "schema": "cem-common-mesh-v1",
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "nodes": int(nodes.shape[0]),
        "cells": int(cells.shape[0]),
        "boundary_edges": int(tagged_edges.shape[0]),
        "electrode_edges": {
            str(electrode): int(np.count_nonzero(tagged_edges[:, 2] == electrode))
            for electrode in range(1, config.n_electrodes + 1)
        },
        "potential_order": 1,
        "scalar_dtype": "float64",
        "msh": str(msh_path),
        "mat": str(mat_path),
    }
    write_json(json_path, metadata)
    return {
        **metadata,
        "nodes_array": nodes,
        "cells_array": cells,
        "tagged_edges_array": tagged_edges,
        "msh_path": msh_path,
        "mat_path": mat_path,
        "json_path": json_path,
    }


def _pattern_config(config: BenchmarkConfig) -> PatternConfig:
    return PatternConfig(
        n_elec=config.n_electrodes,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )


def _assemble_pyeidors_blocks(model: EITForwardModel, sigma: fem.Function):
    electrode = model._ensure_electrode_matrix().tocsr()
    conductivity_petsc = model._assemble_conductivity_matrix(sigma)
    try:
        conductivity = model._petsc_to_csr(conductivity_petsc).astype(
            np.float64, copy=False
        )
    finally:
        destroy = getattr(conductivity_petsc, "destroy", None)
        if callable(destroy):
            destroy()
    dofs = int(model.dofs)
    electrode_stop = dofs + model.n_elec
    robin_matrix = conductivity + electrode[:dofs, :dofs]
    coupling = electrode[:dofs, dofs:electrode_stop]
    electrode_matrix = electrode[dofs:electrode_stop, dofs:electrode_stop]
    return robin_matrix, coupling, electrode_matrix


def run_pyeidors(config: BenchmarkConfig, output_dir: Path) -> dict[str, Any]:
    """Run strict float64/common-P1-mesh classic/Robin PyEIDORS benchmark."""

    mesh_dir = output_dir / "mesh_source"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_started = time.perf_counter()
    eit_mesh = create_eit_mesh(
        n_elec=config.n_electrodes,
        radius=config.radius_m,
        refinement=config.mesh_refinement,
        electrode_coverage=config.electrode_coverage,
        output_dir=str(mesh_dir),
        mesh_name="cem_formulation_disk",
    )
    mesh_seconds = float(time.perf_counter() - mesh_started)
    impedance = np.full(config.n_electrodes, config.contact_impedance, dtype=np.float64)
    setup_started = time.perf_counter()
    classic = EITForwardModel(
        n_elec=config.n_electrodes,
        pattern_config=_pattern_config(config),
        z=impedance,
        mesh=eit_mesh,
        potential_order=config.potential_order,
        linear_backend="scipy",
    )
    model_setup_seconds = float(time.perf_counter() - setup_started)
    scalar_dtype = np.dtype(classic.scalar_dtype)
    if scalar_dtype != np.dtype(np.float64):
        raise RuntimeError(
            "Fair CEM comparison requires PyEIDORS real float64; "
            f"active PETSc scalar dtype is {scalar_dtype}. "
            "Run this benchmark in `nix develop .#default`."
        )
    sigma_classic = fem.Function(classic.V_sigma)
    sigma_classic.x.array[:] = config.conductivity_s_per_m
    currents, labels = trigonometric_current_patterns(
        config.n_electrodes,
        config.electrode_coverage,
    )
    common_mesh = _export_common_mesh(
        config,
        output_dir,
        eit_mesh,
        list(classic.electrode_tags),
        currents,
    )

    assembly_started = time.perf_counter()
    robin_matrix, coupling, electrode_matrix = _assemble_pyeidors_blocks(
        classic, sigma_classic
    )
    assembly_seconds = float(time.perf_counter() - assembly_started)
    timing, voltages, parity = benchmark_preassembled_blocks(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
        repeats=config.timing_repeats,
    )
    timing.update(
        {
            "mesh_generation_seconds": mesh_seconds,
            "model_setup_seconds": model_setup_seconds,
            "assembly_seconds": assembly_seconds,
        }
    )
    voltage_classic = voltages["classic"]
    voltage_robin = voltages["robin_transconductance"]

    production_robin = RobinTransconductanceForwardModel(
        n_elec=config.n_electrodes,
        pattern_config=_pattern_config(config),
        z=impedance,
        mesh=eit_mesh,
        potential_order=config.potential_order,
        linear_backend="scipy",
    )
    sigma_robin = fem.Function(production_robin.V_sigma)
    sigma_robin.x.array[:] = config.conductivity_s_per_m
    _, production_classic_rows = classic.forward_solve(sigma_classic, currents.T)
    _, production_robin_rows = production_robin.forward_solve(sigma_robin, currents.T)
    production_classic = np.asarray(production_classic_rows, dtype=np.float64).T
    production_robin_values = np.asarray(production_robin_rows, dtype=np.float64).T

    rows = characteristic_rows(
        "PyEIDORS/DOLFINx", "classic", currents, voltage_classic, labels
    )
    rows.extend(
        characteristic_rows(
            "PyEIDORS/DOLFINx",
            "robin_transconductance",
            currents,
            voltage_robin,
            labels,
        )
    )
    write_rows(output_dir / "pyeidors_characteristic_resistance.csv", rows)
    diagnostics = production_robin.get_backend_diagnostics()
    report = {
        "solver": "PyEIDORS/DOLFINx",
        "physical_config": asdict(config),
        "discretization": {
            "vertices": common_mesh["nodes"],
            "cells": common_mesh["cells"],
            "degrees_of_freedom": int(classic.dofs),
            "element_family": "DOLFINx P1 Lagrange triangle",
            "potential_order": 1,
            "conductivity_order": int(classic.V_sigma.ufl_element().degree),
            "electrode_integration": "DOLFINx facet forms",
            "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
            "mesh_fingerprint": common_mesh["mesh_fingerprint"],
            "mesh_import_verified": True,
            "common_mesh_role": "canonical source",
        },
        "linear_solver": {
            "classic": "SciPy SuperLU on augmented CEM matrix",
            "robin": "SciPy SuperLU A_R plus SciPy dense reduced LU",
            "scalar_dtype": "float64",
        },
        "timing": timing,
        "within_solver": {
            **parity,
            "classic_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(voltage_classic, axis=0)))
            ),
            "robin_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(voltage_robin, axis=0)))
            ),
            "production_classic_vs_block_classic_relative_l2": relative_l2(
                production_classic, voltage_classic
            ),
            "production_robin_vs_block_robin_relative_l2": relative_l2(
                production_robin_values, voltage_robin
            ),
        },
        "robin_diagnostics": {
            key: diagnostics.get(key)
            for key in (
                "robin_transconductance_rank",
                "robin_transconductance_condition_number",
                "robin_response_residual",
                "robin_reduced_solve_residual",
                "robin_current_balance_residual",
                "robin_voltage_balance_residual",
                "forward_ksp_setup_count",
                "forward_ksp_solve_count",
                "fallback_reason",
            )
        },
        "artifacts": {
            "csv": "pyeidors_characteristic_resistance.csv",
            "common_mesh_msh": str(common_mesh["msh_path"].relative_to(output_dir)),
            "common_mesh_mat": str(common_mesh["mat_path"].relative_to(output_dir)),
            "common_mesh_json": str(common_mesh["json_path"].relative_to(output_dir)),
        },
        "implementation_note": (
            "Production classic and Robin implementations are validated against the "
            "same independently timed A_R/C/D algebraic kernels; validation calls are "
            "excluded from timing."
        ),
    }
    write_json(output_dir / "pyeidors_report.json", report)
    return report


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_FIELDS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing CSV fields: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(
                {
                    "solver": row["solver"],
                    "formulation": row["formulation"],
                    "mode": row["mode"],
                    "spatial_frequency": int(row["spatial_frequency"]),
                    "current_norm_a": float(row["current_norm_a"]),
                    "voltage_norm_v": float(row["voltage_norm_v"]),
                    "characteristic_resistance_ohm": float(
                        row["characteristic_resistance_ohm"]
                    ),
                }
            )
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def configure_fonts() -> None:
    for font_path in (
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    ):
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
    available = {font.name for font in fm.fontManager.ttflist}
    chinese_font = (
        "Microsoft YaHei" if "Microsoft YaHei" in available else "DejaVu Sans"
    )
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", chinese_font, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
        }
    )


def plot_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    configure_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    styles = {"classic": "--", "robin_transconductance": "-"}
    grouped: dict[tuple[str, str], dict[int, list[float]]] = {}
    for row in rows:
        key = (str(row["solver"]), str(row["formulation"]))
        grouped.setdefault(key, {}).setdefault(
            int(row["spatial_frequency"]), []
        ).append(float(row["characteristic_resistance_ohm"]))
    for (solver, formulation), by_frequency in sorted(grouped.items()):
        frequencies = np.asarray(sorted(by_frequency), dtype=float)
        values = np.asarray(
            [np.mean(by_frequency[int(k)]) for k in frequencies],
            dtype=float,
        )
        label = f"{solver} / {formulation}"
        axes[0].plot(
            frequencies,
            values,
            styles.get(formulation, "-"),
            marker="o",
            markersize=3.5,
            label=label,
        )
        axes[1].plot(
            frequencies,
            frequencies * values,
            styles.get(formulation, "-"),
            marker="o",
            markersize=3.5,
            label=label,
        )
    axes[0].set_xlabel("空间频率 k")
    axes[0].set_ylabel("特征电阻 ||U||₂ / ||I||₂ (Ω)")
    axes[0].set_yscale("log")
    axes[1].set_xlabel("空间频率 k")
    axes[1].set_ylabel("k · 特征电阻 (Ω)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontname("Times New Roman")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _resistance_curves(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[int, float]]:
    grouped: dict[tuple[str, str], dict[int, list[float]]] = {}
    for row in rows:
        key = (str(row["solver"]), str(row["formulation"]))
        grouped.setdefault(key, {}).setdefault(
            int(row["spatial_frequency"]), []
        ).append(float(row["characteristic_resistance_ohm"]))
    return {
        key: {
            frequency: float(np.mean(values))
            for frequency, values in by_frequency.items()
        }
        for key, by_frequency in grouped.items()
    }


def comparison_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate within-solver formulation gaps from cross-solver FEM gaps."""

    curves = _resistance_curves(rows)
    solvers = sorted({solver for solver, _ in curves})
    formulations = sorted({formulation for _, formulation in curves})
    within_solver: list[dict[str, Any]] = []
    for solver in solvers:
        classic = curves.get((solver, "classic"))
        robin = curves.get((solver, "robin_transconductance"))
        if classic is None or robin is None:
            continue
        frequencies = sorted(set(classic).intersection(robin))
        classic_values = np.asarray([classic[k] for k in frequencies])
        robin_values = np.asarray([robin[k] for k in frequencies])
        within_solver.append(
            {
                "solver": solver,
                "frequencies": frequencies,
                "curve_relative_l2_robin_vs_classic": relative_l2(
                    robin_values,
                    classic_values,
                ),
                "curve_max_abs_ohm": float(
                    np.max(np.abs(robin_values - classic_values), initial=0.0)
                ),
            }
        )

    cross_solver: list[dict[str, Any]] = []
    for formulation in formulations:
        available = [solver for solver in solvers if (solver, formulation) in curves]
        for candidate_solver, reference_solver in combinations(available, 2):
            candidate = curves[(candidate_solver, formulation)]
            reference = curves[(reference_solver, formulation)]
            frequencies = sorted(set(candidate).intersection(reference))
            candidate_values = np.asarray([candidate[k] for k in frequencies])
            reference_values = np.asarray([reference[k] for k in frequencies])
            denominator = np.maximum(
                np.abs(reference_values),
                np.finfo(float).eps,
            )
            cross_solver.append(
                {
                    "formulation": formulation,
                    "candidate_solver": candidate_solver,
                    "reference_solver": reference_solver,
                    "frequencies": frequencies,
                    "raw_curve_relative_l2": relative_l2(
                        candidate_values,
                        reference_values,
                    ),
                    "raw_curve_max_pointwise_relative": float(
                        np.max(
                            np.abs(candidate_values - reference_values) / denominator
                        )
                    ),
                    "raw_curve_correlation": float(
                        np.corrcoef(candidate_values, reference_values)[0, 1]
                    ),
                }
            )
    return {
        "within_solver_formulation": within_solver,
        "cross_solver_implementation": cross_solver,
    }


def _timing_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        solver = str(report["solver"])
        timing = report["timing"]
        for formulation in ("classic", "robin_transconductance"):
            formulation_timing = timing[formulation]
            for phase in ("cold", "warm"):
                summary = formulation_timing[f"{phase}_seconds"]
                rows.append(
                    {
                        "solver": solver,
                        "formulation": formulation,
                        "phase": phase,
                        "median_seconds": float(summary["median"]),
                        "iqr_seconds": float(summary["iqr"]),
                        "repeats": int(timing["repeats"]),
                        "rhs_count": int(timing["rhs_count"]),
                    }
                )
    return rows


def _write_timing_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "solver",
        "formulation",
        "phase",
        "median_seconds",
        "iqr_seconds",
        "repeats",
        "rhs_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_timings(rows: list[dict[str, Any]], output_path: Path) -> None:
    configure_fonts()
    solvers = sorted({str(row["solver"]) for row in rows})
    formulations = ("classic", "robin_transconductance")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    width = 0.36
    positions = np.arange(len(solvers), dtype=float)
    for axis, phase in zip(axes, ("cold", "warm"), strict=True):
        for offset_index, formulation in enumerate(formulations):
            selected = {
                str(row["solver"]): row
                for row in rows
                if row["phase"] == phase and row["formulation"] == formulation
            }
            medians = [float(selected[solver]["median_seconds"]) for solver in solvers]
            errors = [
                float(selected[solver]["iqr_seconds"]) / 2.0 for solver in solvers
            ]
            offset = (offset_index - 0.5) * width
            axis.bar(
                positions + offset,
                medians,
                width,
                yerr=errors,
                capsize=3,
                label=formulation,
            )
        axis.set_title(f"{phase.capitalize()} solve")
        axis.set_ylabel("Seconds for all RHS")
        axis.set_yscale("log")
        axis.set_xticks(positions, solvers, rotation=12, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontname("Times New Roman")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _timing_metrics(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = []
    for report in reports:
        timing = report["timing"]
        item: dict[str, Any] = {"solver": report["solver"]}
        for phase in ("cold", "warm"):
            classic = float(timing["classic"][f"{phase}_seconds"]["median"])
            robin = float(
                timing["robin_transconductance"][f"{phase}_seconds"]["median"]
            )
            item[f"{phase}_classic_over_robin"] = classic / max(
                robin, np.finfo(float).eps
            )
        metrics.append(item)
    return metrics


def aggregate(
    output_dir: Path,
    csv_paths: list[Path],
    metadata_paths: list[Path],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in csv_paths:
        rows.extend(read_rows(path))
    if not rows:
        raise ValueError("at least one benchmark CSV is required")
    metadata: list[dict[str, Any]] = []
    for path in metadata_paths:
        metadata.append(json.loads(path.read_text(encoding="utf-8")))
    common_fingerprint = validate_solver_reports(metadata)
    combined_csv = output_dir / "cem_formulation_comparison.csv"
    write_rows(combined_csv, rows)
    plot_path = output_dir / "cem_formulation_comparison.png"
    plot_rows(rows, plot_path)
    timing_rows = _timing_rows(metadata)
    timing_csv = output_dir / "cem_formulation_timing.csv"
    _write_timing_rows(timing_csv, timing_rows)
    timing_plot = output_dir / "cem_formulation_timing.png"
    _plot_timings(timing_rows, timing_plot)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.platform(),
        "comparison_policy": {
            "within_solver": "same assembled FEM model; classic versus Robin",
            "cross_solver": (
                "same canonical P1 mesh/fingerprint, float64, physical parameters, "
                "current RHS, and gauge; raw SI values with no fitted scaling"
            ),
            "gauge": "zero-mean electrode voltage",
            "transpose": "non-conjugate transpose for reciprocal complex systems",
            "timing": (
                "preassembled A_R/C/D blocks; independent formulation state; cold "
                "factorization and warm own-cache solves reported separately"
            ),
        },
        "common_mesh_fingerprint": common_fingerprint,
        "input_csvs": [str(path) for path in csv_paths],
        "solver_reports": metadata,
        "metrics": {
            **comparison_metrics(rows),
            "timing": _timing_metrics(metadata),
        },
        "artifacts": {
            "combined_csv": combined_csv.name,
            "plot": plot_path.name,
            "timing_csv": timing_csv.name,
            "timing_plot": timing_plot.name,
        },
    }
    write_json(output_dir / "cem_formulation_comparison.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "cem_formulation_comparison",
    )
    parser.add_argument(
        "--skip-pyeidors",
        action="store_true",
        help="only aggregate already generated CSV/JSON files",
    )
    parser.add_argument(
        "--external-csv",
        action="append",
        type=Path,
        default=[],
        help="NGSolve/EIDORS CSV to include; may be repeated",
    )
    parser.add_argument(
        "--external-report",
        action="append",
        type=Path,
        default=[],
        help="NGSolve/EIDORS JSON metadata to include; may be repeated",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=11,
        help="cold and warm timing repetitions per formulation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [path.resolve() for path in args.external_csv]
    metadata_paths = [path.resolve() for path in args.external_report]
    if not args.skip_pyeidors:
        run_pyeidors(
            BenchmarkConfig(timing_repeats=int(args.timing_repeats)),
            output_dir,
        )
        csv_paths.insert(0, output_dir / "pyeidors_characteristic_resistance.csv")
        metadata_paths.insert(0, output_dir / "pyeidors_report.json")
    aggregate(output_dir, csv_paths, metadata_paths)
    print(f"CEM comparison artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
