#!/usr/bin/env python3
"""Run a single cross-generation difference-reconstruction case."""

from __future__ import annotations

import argparse
import csv
import json
import resource
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
from benchmark_reviewer_case import (  # noqa: E402
    PYEIDORS_REFINEMENTS,
    PYEIT_H0,
    SCENARIO_CONFIG,
    conductivity_metrics,
    get_git_commit,
    voltage_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["pyeidors", "pyeit"], required=True)
    parser.add_argument("--source-framework", choices=["pyeidors", "eidors"], required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-metrics-json", type=Path, default=None)
    parser.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="medium")
    parser.add_argument("--scenario", choices=["low_z", "high_z"], default="low_z")
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"))
    parser.add_argument("--difference-hyperparameter", type=float, default=None)
    return parser.parse_args()


def get_peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def load_forward_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in forward CSV: {path}")
    baseline = np.asarray([float(row["meas_homogeneous"]) for row in rows], dtype=float)
    phantom = np.asarray([float(row["meas_phantom"]) for row in rows], dtype=float)
    if "difference" in reader.fieldnames:
        target_diff = np.asarray([float(row["difference"]) for row in rows], dtype=float)
    else:
        target_diff = phantom - baseline
    return baseline, phantom, target_diff


def load_source_provenance(args: argparse.Namespace) -> dict[str, object]:
    cfg = SCENARIO_CONFIG[args.scenario]
    provenance: dict[str, object] = {
        "source_background": float(cfg["background"]),
        "source_contact_impedance": float(cfg["contact_impedance"]),
        "source_phantom_contrast": float(cfg["phantom_conductivity"]),
    }
    if args.source_metrics_json is None or not args.source_metrics_json.exists():
        return provenance

    try:
        payload = json.loads(args.source_metrics_json.read_text(encoding="utf-8"))
    except Exception:
        return provenance

    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(metadata, dict):
        return provenance

    if metadata.get("background") is not None:
        provenance["source_background"] = float(metadata["background"])
    if metadata.get("contact_impedance") is not None:
        provenance["source_contact_impedance"] = float(metadata["contact_impedance"])
    if metadata.get("phantom_contrast") is not None:
        provenance["source_phantom_contrast"] = float(metadata["phantom_contrast"])
    else:
        phantom = metadata.get("phantom", {})
        if isinstance(phantom, dict) and phantom.get("contrast") is not None:
            provenance["source_phantom_contrast"] = float(phantom["contrast"])

    try:
        provenance["source_metrics_json"] = str(args.source_metrics_json.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        provenance["source_metrics_json"] = str(args.source_metrics_json.resolve())
    return provenance


def make_pattern_config(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        rotate_meas=True,
    )


class PyEidorsCrossCase:
    def __init__(self, args: argparse.Namespace):
        cfg = SCENARIO_CONFIG[args.scenario]
        self.args = args
        self.cfg = cfg
        pattern = make_pattern_config(args.n_elec)
        self.mesh = load_or_create_mesh(
            mesh_dir=str(args.mesh_dir),
            n_elec=args.n_elec,
            refinement=PYEIDORS_REFINEMENTS[args.mesh_level],
            radius=1.0,
            electrode_coverage=0.5,
        )
        self.system = EITSystem(
            n_elec=args.n_elec,
            pattern_config=pattern,
            contact_impedance=np.ones(args.n_elec) * cfg["contact_impedance"],
            base_conductivity=cfg["background"],
            regularization_type="noser",
            regularization_alpha=1.0,
            noser_exponent=0.5,
        )
        self.system.setup(mesh=self.mesh)
        self.baseline_image = self.system.create_homogeneous_image(conductivity=cfg["background"])
        sigma = create_custom_phantom(
            self.system.fwd_model,
            background_conductivity=cfg["background"],
            anomalies=[{
                "center": tuple(cfg["phantom_center"]),
                "radius": cfg["phantom_radius"],
                "conductivity": cfg["phantom_conductivity"],
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
        return len(self.system.fwd_model.V_sigma.dofmap().dofs())

    def run(self, baseline: np.ndarray, phantom: np.ndarray, target_diff: np.ndarray) -> dict[str, float]:
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
            target_diff=target_diff,
        )
        result: dict[str, float] = {}
        result.update(voltage_metrics(target_diff, predicted_diff))
        result.update(conductivity_metrics(self.truth_sigma, recon_image.elem_data))
        result["optimal_step_size"] = float(step_size)
        return result


class PyEitCrossCase:
    def __init__(self, args: argparse.Namespace):
        from pyeit.mesh import create
        from pyeit.mesh.wrapper import PyEITAnomaly_Circle, set_perm
        from pyeit.eit.protocol import create as create_protocol
        from pyeit.eit.fem import EITForward
        from pyeit.eit.jac import JAC

        cfg = SCENARIO_CONFIG[args.scenario]
        self.args = args
        self.cfg = cfg
        self._JAC = JAC
        h0 = PYEIT_H0[args.mesh_level]
        self.mesh = create(n_el=args.n_elec, h0=h0)
        self.protocol = create_protocol(n_el=args.n_elec, dist_exc=1, step_meas=1, parser_meas="std")
        self.forward_solver = EITForward(self.mesh, self.protocol)
        self.baseline_perm = np.ones(self.mesh.n_elems, dtype=float) * cfg["background"]
        anomaly = PyEITAnomaly_Circle(
            center=list(cfg["phantom_center"]),
            r=cfg["phantom_radius"],
            perm=cfg["phantom_conductivity"],
        )
        self.truth_mesh = set_perm(self.mesh, anomaly=anomaly, background=cfg["background"])
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

    def run(self, baseline: np.ndarray, phantom: np.ndarray, target_diff: np.ndarray) -> dict[str, float]:
        jac = self._JAC(self.mesh, self.protocol)
        jac.setup(p=0.5, lamb=0.01, method="kotre", perm=self.baseline_perm)
        ds = np.asarray(jac.solve(phantom, baseline)).ravel()
        predicted_perm = self.baseline_perm + ds
        pred_v = np.asarray(self.forward_solver.solve_eit(perm=predicted_perm)).ravel()
        result: dict[str, float] = {}
        result.update(voltage_metrics(target_diff, pred_v - baseline))
        result.update(conductivity_metrics(self.truth_perm, predicted_perm))
        return result


def build_row(
    args: argparse.Namespace,
    case,
    metrics: dict[str, float],
    runtime: float,
    source_provenance: dict[str, object],
) -> dict[str, object]:
    row = {
        "study": "cross_generation",
        "source_framework": args.source_framework,
        "framework": args.framework,
        "task": "difference",
        "mesh_level": args.mesh_level,
        "mesh_name": case.mesh_name,
        "scenario": args.scenario,
        "n_nodes": int(case.n_nodes),
        "n_elements": int(case.n_elements),
        "n_frames": 1,
        "device": "cpu",
        "warmups": 0,
        "repeats": 1,
        "commit": get_git_commit(),
        "peak_rss_mb": get_peak_rss_mb(),
        "mean": float(runtime),
        "std": 0.0,
        "median": float(runtime),
        "iqr": 0.0,
        "mean_sec": float(runtime),
        "std_sec": 0.0,
        "median_sec": float(runtime),
        "iqr_sec": 0.0,
        "source_csv": str(args.input_csv),
    }
    row.update(source_provenance)
    row.update(metrics)
    return row


def main() -> None:
    args = parse_args()
    baseline, phantom, target_diff = load_forward_csv(args.input_csv)
    source_provenance = load_source_provenance(args)
    case = PyEidorsCrossCase(args) if args.framework == "pyeidors" else PyEitCrossCase(args)
    t0 = time.perf_counter()
    metrics = case.run(baseline, phantom, target_diff)
    runtime = time.perf_counter() - t0
    row = build_row(args, case, metrics, runtime, source_provenance)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(row, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
