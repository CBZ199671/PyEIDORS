#!/usr/bin/env python3
"""Build Reviewer 1 Comment 5 assets for CEM-source PyEIDORS vs pyEIT comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pyeidors import EITSystem  # noqa: E402
from pyeidors.data.structures import EITImage, PatternConfig  # noqa: E402
from pyeidors.data.synthetic_data import create_custom_phantom  # noqa: E402
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh  # noqa: E402

from benchmark_difference_runtime import (  # noqa: E402
    build_single_step_namespace,
    compute_difference,
    run_single_step_benchmark,
)
from benchmark_reviewer_case import PYEIDORS_REFINEMENTS, PYEIT_H0, conductivity_metrics, get_git_commit, voltage_metrics  # noqa: E402


SOURCE_ZS = {
    "low_z": 1e-6,
    "high_z": 1e-2,
}


def parse_args() -> argparse.Namespace:
    fairness_dir = REPO_ROOT / "docs" / "benchmarks" / "reviewer_suite" / "fairness"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=fairness_dir / "r1c5_cem_vs_pyeit.json")
    parser.add_argument("--output-csv", type=Path, default=fairness_dir / "r1c5_cem_vs_pyeit.csv")
    parser.add_argument("--source-dir", type=Path, default=fairness_dir / "r1c5_sources")
    parser.add_argument("--mesh-dir", type=Path, default=REPO_ROOT / "eit_meshes")
    parser.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="medium")
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--background", type=float, default=1.0)
    parser.add_argument("--phantom-conductivity", type=float, default=2.0)
    parser.add_argument("--phantom-center-x", type=float, default=0.30)
    parser.add_argument("--phantom-center-y", type=float, default=0.20)
    parser.add_argument("--phantom-radius", type=float, default=0.20)
    parser.add_argument("--difference-hyperparameter", type=float, default=None)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--eidors-source-low-z", type=Path, default=None)
    parser.add_argument("--eidors-source-high-z", type=Path, default=None)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_forward_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in forward CSV: {path}")
    baseline = np.asarray([float(row["meas_homogeneous"]) for row in rows], dtype=float)
    phantom = np.asarray([float(row["meas_phantom"]) for row in rows], dtype=float)
    return baseline, phantom


def save_forward_csv(path: Path, baseline: np.ndarray, phantom: np.ndarray) -> None:
    ensure_parent(path)
    diff = phantom - baseline
    rows = [
        {
            "meas_homogeneous": float(vh),
            "meas_phantom": float(vi),
            "difference": float(vd),
        }
        for vh, vi, vd in zip(baseline, phantom, diff, strict=True)
    ]
    write_csv(path, ["meas_homogeneous", "meas_phantom", "difference"], rows)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def format_rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def make_pattern_config(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        rotate_meas=True,
    )


class PyEidorsSourceBuilder:
    def __init__(self, args: argparse.Namespace, contact_impedance: float):
        self.args = args
        self.contact_impedance = contact_impedance
        self.pattern = make_pattern_config(args.n_elec)
        self.mesh = load_or_create_mesh(
            mesh_dir=str(args.mesh_dir),
            n_elec=args.n_elec,
            refinement=PYEIDORS_REFINEMENTS[args.mesh_level],
            radius=args.radius,
            electrode_coverage=args.electrode_coverage,
        )
        self.system = EITSystem(
            n_elec=args.n_elec,
            pattern_config=self.pattern,
            contact_impedance=np.ones(args.n_elec) * contact_impedance,
            base_conductivity=args.background,
            regularization_type="noser",
            regularization_alpha=1.0,
            noser_exponent=0.5,
        )
        self.system.setup(mesh=self.mesh)
        self.baseline_image = self.system.create_homogeneous_image(conductivity=args.background)
        sigma = create_custom_phantom(
            self.system.fwd_model,
            background_conductivity=args.background,
            anomalies=[{
                "center": (args.phantom_center_x, args.phantom_center_y),
                "radius": args.phantom_radius,
                "conductivity": args.phantom_conductivity,
            }],
        )
        self.truth_image = EITImage(elem_data=sigma.vector()[:], fwd_model=self.system.fwd_model)

    def export(self, output_csv: Path) -> dict[str, object]:
        baseline_meas, _ = self.system.fwd_model.fwd_solve(self.baseline_image)
        phantom_meas, _ = self.system.fwd_model.fwd_solve(self.truth_image)
        baseline = np.asarray(baseline_meas.meas, dtype=float).ravel()
        phantom = np.asarray(phantom_meas.meas, dtype=float).ravel()
        save_forward_csv(output_csv, baseline, phantom)
        return {
            "source_framework": "pyeidors",
            "source_model": "CEM",
            "source_csv": format_rel_path(output_csv),
            "background": float(self.args.background),
            "contact_impedance": float(self.contact_impedance),
            "phantom_contrast": float(self.args.phantom_conductivity),
            "mesh_name": getattr(self.mesh, "mesh_name", f"ref{PYEIDORS_REFINEMENTS[self.args.mesh_level]}"),
            "n_nodes": int(self.mesh.num_vertices()),
            "n_elements": int(len(self.system.fwd_model.V_sigma.dofmap().dofs())),
        }


class PyEidorsReconstructor:
    def __init__(self, args: argparse.Namespace, reconstruction_contact_impedance: float):
        self.args = args
        self.reconstruction_contact_impedance = reconstruction_contact_impedance
        self.pattern = make_pattern_config(args.n_elec)
        self.mesh = load_or_create_mesh(
            mesh_dir=str(args.mesh_dir),
            n_elec=args.n_elec,
            refinement=PYEIDORS_REFINEMENTS[args.mesh_level],
            radius=args.radius,
            electrode_coverage=args.electrode_coverage,
        )
        self.system = EITSystem(
            n_elec=args.n_elec,
            pattern_config=self.pattern,
            contact_impedance=np.ones(args.n_elec) * reconstruction_contact_impedance,
            base_conductivity=args.background,
            regularization_type="noser",
            regularization_alpha=1.0,
            noser_exponent=0.5,
        )
        self.system.setup(mesh=self.mesh)
        self.baseline_image = self.system.create_homogeneous_image(conductivity=args.background)
        sigma = create_custom_phantom(
            self.system.fwd_model,
            background_conductivity=args.background,
            anomalies=[{
                "center": (args.phantom_center_x, args.phantom_center_y),
                "radius": args.phantom_radius,
                "conductivity": args.phantom_conductivity,
            }],
        )
        self.truth_image = EITImage(elem_data=sigma.vector()[:], fwd_model=self.system.fwd_model)
        self.truth_sigma = self.truth_image.elem_data.copy()

    @property
    def mesh_name(self) -> str:
        return getattr(self.mesh, "mesh_name", f"ref{PYEIDORS_REFINEMENTS[self.args.mesh_level]}")

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.num_vertices())

    @property
    def n_elements(self) -> int:
        return int(len(self.system.fwd_model.V_sigma.dofmap().dofs()))

    def reconstruct(self, baseline: np.ndarray, phantom: np.ndarray) -> dict[str, float]:
        diff_vector = compute_difference(phantom, baseline, False)
        single_step_kwargs: dict[str, float] = {}
        if self.args.difference_hyperparameter is not None:
            single_step_kwargs["difference_hyperparameter"] = self.args.difference_hyperparameter
        single_step_args = build_single_step_namespace(self.system, **single_step_kwargs)
        recon_image, predicted_diff, step_size = run_single_step_benchmark(
            self.system,
            self.baseline_image,
            baseline,
            phantom,
            single_step_args,
            target_diff=diff_vector,
        )
        metrics: dict[str, float] = {}
        metrics.update(voltage_metrics(phantom - baseline, predicted_diff))
        metrics.update(conductivity_metrics(self.truth_sigma, recon_image.elem_data))
        metrics["optimal_step_size"] = float(step_size)
        return metrics


class PyEitReconstructor:
    def __init__(self, args: argparse.Namespace):
        from pyeit.eit.fem import EITForward
        from pyeit.eit.jac import JAC
        from pyeit.eit.protocol import create as create_protocol
        from pyeit.mesh import create
        from pyeit.mesh.wrapper import PyEITAnomaly_Circle, set_perm

        self.args = args
        self._JAC = JAC
        h0 = PYEIT_H0[args.mesh_level]
        self.mesh = create(n_el=args.n_elec, h0=h0)
        self.protocol = create_protocol(n_el=args.n_elec, dist_exc=1, step_meas=1, parser_meas="std")
        self.forward_solver = EITForward(self.mesh, self.protocol)
        self.baseline_perm = np.ones(self.mesh.n_elems, dtype=float) * args.background
        anomaly = PyEITAnomaly_Circle(
            center=[args.phantom_center_x, args.phantom_center_y],
            r=args.phantom_radius,
            perm=args.phantom_conductivity,
        )
        self.truth_mesh = set_perm(self.mesh, anomaly=anomaly, background=args.background)
        self.truth_perm = np.asarray(self.truth_mesh.perm_array, dtype=float).copy()

    @property
    def mesh_name(self) -> str:
        return f"pyeit_{self.args.mesh_level}"

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.n_nodes)

    @property
    def n_elements(self) -> int:
        return int(self.mesh.n_elems)

    def reconstruct(self, baseline: np.ndarray, phantom: np.ndarray) -> dict[str, float]:
        jac = self._JAC(self.mesh, self.protocol)
        jac.setup(p=0.5, lamb=0.01, method="kotre", perm=self.baseline_perm)
        ds = np.asarray(jac.solve(phantom, baseline)).ravel()
        predicted_perm = self.baseline_perm + ds
        pred_v = np.asarray(self.forward_solver.solve_eit(perm=predicted_perm)).ravel()
        metrics: dict[str, float] = {}
        metrics.update(voltage_metrics(phantom - baseline, pred_v - baseline))
        metrics.update(conductivity_metrics(self.truth_perm, predicted_perm))
        return metrics


def build_row(
    *,
    source_framework: str,
    source_z: str,
    source_csv: Path,
    reconstructor: str,
    reconstruction_z: str | None,
    reconstructor_model: str,
    mesh_name: str,
    n_nodes: int,
    n_elements: int,
    runtime_sec: float,
    metrics: dict[str, float],
) -> dict[str, object]:
    matched_to_source = None
    if reconstruction_z is not None:
        matched_to_source = reconstruction_z == source_z
    reconstruction_contact_impedance = SOURCE_ZS[reconstruction_z] if reconstruction_z else None
    return {
        "study": "r1c5_cem_vs_pyeit",
        "source_framework": source_framework,
        "source_model": "CEM",
        "source_z": source_z,
        "source_background": args.background,
        "source_contact_impedance": SOURCE_ZS[source_z],
        "source_phantom_contrast": args.phantom_conductivity,
        "source_csv": format_rel_path(source_csv),
        "reconstructor": reconstructor,
        "reconstructor_model": reconstructor_model,
        "reconstruction_z": reconstruction_z or "",
        "reconstruction_contact_impedance": reconstruction_contact_impedance if reconstruction_contact_impedance is not None else "",
        "matched_to_source": matched_to_source if matched_to_source is not None else "",
        "mesh_level": "medium",
        "mesh_name": mesh_name,
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "runtime_sec": runtime_sec,
        "voltage_rmse": float(metrics["voltage_rmse"]),
        "voltage_mae": float(metrics["voltage_mae"]),
        "conductivity_relative_error_pct": float(metrics["conductivity_relative_error_pct"]),
        "conductivity_rmse": float(metrics["conductivity_rmse"]),
        "conductivity_mae": float(metrics["conductivity_mae"]),
        "optimal_step_size": float(metrics["optimal_step_size"]) if "optimal_step_size" in metrics else "",
        "commit": get_git_commit(),
    }


def summarise_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for source_z in ("low_z", "high_z"):
        subset = [row for row in rows if row["source_z"] == source_z]
        matched_rows = [row for row in subset if row["reconstructor"] == "PyEIDORS" and row["matched_to_source"] is True]
        mismatched_rows = [row for row in subset if row["reconstructor"] == "PyEIDORS" and row["matched_to_source"] is False]
        pyeit_rows = [row for row in subset if row["reconstructor"] == "pyEIT"]
        summaries[source_z] = {
            "matched_pyeidors_voltage_rmse_median": median([float(row["voltage_rmse"]) for row in matched_rows]),
            "mismatched_pyeidors_voltage_rmse_median": median([float(row["voltage_rmse"]) for row in mismatched_rows]),
            "pyeit_voltage_rmse_median": median([float(row["voltage_rmse"]) for row in pyeit_rows]),
            "matched_pyeidors_voltage_mae_median": median([float(row["voltage_mae"]) for row in matched_rows]),
            "mismatched_pyeidors_voltage_mae_median": median([float(row["voltage_mae"]) for row in mismatched_rows]),
            "pyeit_voltage_mae_median": median([float(row["voltage_mae"]) for row in pyeit_rows]),
            "matched_pyeidors_conductivity_relative_error_pct_median": median(
                [float(row["conductivity_relative_error_pct"]) for row in matched_rows]
            ),
            "mismatched_pyeidors_conductivity_relative_error_pct_median": median(
                [float(row["conductivity_relative_error_pct"]) for row in mismatched_rows]
            ),
            "pyeit_conductivity_relative_error_pct_median": median(
                [float(row["conductivity_relative_error_pct"]) for row in pyeit_rows]
            ),
        }
    return summaries


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)

    eidors_sources = {
        "low_z": args.eidors_source_low_z or (args.source_dir / "eidors_source_low_z.csv"),
        "high_z": args.eidors_source_high_z or (args.source_dir / "eidors_source_high_z.csv"),
    }
    for path in eidors_sources.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing EIDORS source CSV: {path}")

    pyeidors_sources: dict[str, Path] = {}
    source_metadata: dict[str, dict[str, object]] = {"eidors": {}, "pyeidors": {}}
    for source_z, z_value in SOURCE_ZS.items():
        output_csv = args.source_dir / f"pyeidors_source_{source_z}.csv"
        source_builder = PyEidorsSourceBuilder(args, z_value)
        source_metadata["pyeidors"][source_z] = source_builder.export(output_csv)
        pyeidors_sources[source_z] = output_csv
        source_metadata["eidors"][source_z] = {
            "source_framework": "eidors",
            "source_model": "CEM",
            "source_csv": format_rel_path(eidors_sources[source_z]),
            "background": args.background,
            "contact_impedance": z_value,
            "phantom_contrast": args.phantom_conductivity,
        }

    reconstructors = {
        "PyEIDORS-low_z": PyEidorsReconstructor(args, SOURCE_ZS["low_z"]),
        "PyEIDORS-high_z": PyEidorsReconstructor(args, SOURCE_ZS["high_z"]),
        "pyEIT": PyEitReconstructor(args),
    }

    rows: list[dict[str, object]] = []
    for source_framework, source_map in (("eidors", eidors_sources), ("pyeidors", pyeidors_sources)):
        for source_z, source_csv in source_map.items():
            baseline, phantom = load_forward_csv(source_csv)
            for label, reconstructor in reconstructors.items():
                t0 = time.perf_counter()
                metrics = reconstructor.reconstruct(baseline, phantom)
                runtime_sec = time.perf_counter() - t0
                if label.startswith("PyEIDORS-"):
                    reconstruction_z = label.split("-", 1)[1]
                    reconstructor_name = "PyEIDORS"
                    model_type = "CEM"
                else:
                    reconstruction_z = None
                    reconstructor_name = "pyEIT"
                    model_type = "PEM"
                rows.append(
                    build_row(
                        source_framework=source_framework,
                        source_z=source_z,
                        source_csv=source_csv,
                        reconstructor=reconstructor_name,
                        reconstruction_z=reconstruction_z,
                        reconstructor_model=model_type,
                        mesh_name=reconstructor.mesh_name,
                        n_nodes=reconstructor.n_nodes,
                        n_elements=reconstructor.n_elements,
                        runtime_sec=runtime_sec,
                        metrics=metrics,
                    )
                )

    rows = sorted(
        rows,
        key=lambda row: (
            0 if row["source_framework"] == "eidors" else 1,
            0 if row["source_z"] == "low_z" else 1,
            0 if row["reconstructor"] == "PyEIDORS" and row["reconstruction_z"] == "low_z" else
            1 if row["reconstructor"] == "PyEIDORS" and row["reconstruction_z"] == "high_z" else
            2,
        ),
    )
    summaries = summarise_rows(rows)

    fieldnames = [
        "study",
        "source_framework",
        "source_model",
        "source_z",
        "source_background",
        "source_contact_impedance",
        "source_phantom_contrast",
        "source_csv",
        "reconstructor",
        "reconstructor_model",
        "reconstruction_z",
        "reconstruction_contact_impedance",
        "matched_to_source",
        "mesh_level",
        "mesh_name",
        "n_nodes",
        "n_elements",
        "runtime_sec",
        "voltage_rmse",
        "voltage_mae",
        "conductivity_relative_error_pct",
        "conductivity_rmse",
        "conductivity_mae",
        "optimal_step_size",
        "commit",
    ]
    write_csv(args.output_csv, fieldnames, rows)

    payload = {
        "study": "r1c5_cem_vs_pyeit",
        "settings": {
            "mesh_level": args.mesh_level,
            "n_elec": args.n_elec,
            "background": args.background,
            "phantom_conductivity": args.phantom_conductivity,
            "phantom_center": [args.phantom_center_x, args.phantom_center_y],
            "phantom_radius": args.phantom_radius,
            "difference_hyperparameter": args.difference_hyperparameter,
            "electrode_coverage": args.electrode_coverage,
            "radius": args.radius,
            "source_contact_impedances": SOURCE_ZS,
        },
        "source_metadata": source_metadata,
        "scenario_summaries": summaries,
        "csv_path": format_rel_path(args.output_csv),
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
