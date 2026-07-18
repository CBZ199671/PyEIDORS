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


def _mesh_counts(eit_mesh) -> dict[str, int]:
    mesh = eit_mesh.mesh
    topology = mesh.topology
    topology.create_connectivity(topology.dim, 0)
    cell_map = topology.index_map(topology.dim)
    vertex_map = topology.index_map(0)
    return {
        "vertices": int(vertex_map.size_global),
        "cells": int(cell_map.size_global),
    }


def run_pyeidors(config: BenchmarkConfig, output_dir: Path) -> dict[str, Any]:
    """Run same-mesh classic/Robin PyEIDORS comparison and write raw data."""

    mesh_dir = output_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    eit_mesh = create_eit_mesh(
        n_elec=config.n_electrodes,
        radius=config.radius_m,
        refinement=config.mesh_refinement,
        electrode_coverage=config.electrode_coverage,
        output_dir=str(mesh_dir),
        mesh_name="cem_formulation_disk",
    )
    pattern_config = PatternConfig(
        n_elec=config.n_electrodes,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    impedance = np.full(config.n_electrodes, config.contact_impedance, dtype=float)
    classic = EITForwardModel(
        n_elec=config.n_electrodes,
        pattern_config=pattern_config,
        z=impedance,
        mesh=eit_mesh,
        potential_order=config.potential_order,
        linear_backend="scipy",
    )
    robin = RobinTransconductanceForwardModel(
        n_elec=config.n_electrodes,
        pattern_config=pattern_config,
        z=impedance,
        mesh=eit_mesh,
        potential_order=config.potential_order,
        linear_backend="scipy",
    )
    sigma_classic = fem.Function(classic.V_sigma)
    sigma_robin = fem.Function(robin.V_sigma)
    sigma_classic.x.array[:] = config.conductivity_s_per_m
    sigma_robin.x.array[:] = config.conductivity_s_per_m
    currents, labels = trigonometric_current_patterns(
        config.n_electrodes,
        config.electrode_coverage,
    )

    classic_started = time.perf_counter()
    potential_classic, voltage_classic_rows = classic.forward_solve(
        sigma_classic,
        currents.T,
    )
    classic_seconds = float(time.perf_counter() - classic_started)
    robin_started = time.perf_counter()
    potential_robin, voltage_robin_rows = robin.forward_solve(
        sigma_robin,
        currents.T,
    )
    robin_seconds = float(time.perf_counter() - robin_started)

    voltage_classic = np.asarray(voltage_classic_rows).T
    voltage_robin = np.asarray(voltage_robin_rows).T
    potential_classic_matrix = np.column_stack(potential_classic)
    potential_robin_matrix = np.column_stack(potential_robin)
    rows = characteristic_rows(
        "PyEIDORS/DOLFINx",
        "classic",
        currents,
        voltage_classic,
        labels,
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

    diagnostics = robin.get_backend_diagnostics()
    report = {
        "solver": "PyEIDORS/DOLFINx",
        "physical_config": asdict(config),
        "discretization": {
            **_mesh_counts(eit_mesh),
            "element_family": "Lagrange triangle",
            "potential_order": config.potential_order,
            "conductivity_order": int(classic.V_sigma.ufl_element().degree),
            "electrode_integration": "DOLFINx facet forms",
        },
        "linear_solver": {
            "classic": "SciPy SuperLU on augmented CEM matrix",
            "robin": "one SciPy SuperLU factorization of A_R plus reduced dense LU",
            "scalar_dtype": str(voltage_classic.dtype),
        },
        "within_solver": {
            "electrode_voltage_relative_l2": relative_l2(
                voltage_robin,
                voltage_classic,
            ),
            "body_potential_relative_l2": relative_l2(
                potential_robin_matrix,
                potential_classic_matrix,
            ),
            "classic_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(voltage_classic, axis=0)))
            ),
            "robin_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(voltage_robin, axis=0)))
            ),
            "classic_seconds": classic_seconds,
            "robin_seconds": robin_seconds,
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
        },
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
        "cross_solver_discretization": cross_solver,
    }


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
    combined_csv = output_dir / "cem_formulation_comparison.csv"
    write_rows(combined_csv, rows)
    plot_path = output_dir / "cem_formulation_comparison.png"
    plot_rows(rows, plot_path)
    metadata: list[dict[str, Any]] = []
    for path in metadata_paths:
        metadata.append(json.loads(path.read_text(encoding="utf-8")))
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "host": platform.platform(),
        "comparison_policy": {
            "within_solver": "same assembled FEM model; classic versus Robin",
            "cross_solver": "raw SI characteristic resistance; no fitted scaling",
            "gauge": "zero-mean electrode voltage",
            "transpose": "non-conjugate transpose for reciprocal complex systems",
        },
        "input_csvs": [str(path) for path in csv_paths],
        "solver_reports": metadata,
        "metrics": comparison_metrics(rows),
        "artifacts": {
            "combined_csv": combined_csv.name,
            "plot": plot_path.name,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [path.resolve() for path in args.external_csv]
    metadata_paths = [path.resolve() for path in args.external_report]
    if not args.skip_pyeidors:
        run_pyeidors(BenchmarkConfig(), output_dir)
        csv_paths.insert(0, output_dir / "pyeidors_characteristic_resistance.csv")
        metadata_paths.insert(0, output_dir / "pyeidors_report.json")
    aggregate(output_dir, csv_paths, metadata_paths)
    print(f"CEM comparison artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
