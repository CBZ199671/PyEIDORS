#!/usr/bin/env python3
"""Run NGSolve P1 float64 CEM on one true-circle continuum-suite fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_fair_common import (
    _as_csc,
    _classic_state,
    _robin_state,
    _solve_classic,
    _solve_robin,
)
from scripts.benchmarks.ngsolve_cem_formulations import (
    Config,
    assemble_blocks,
    load_verified_mesh,
)


SUITE_SCHEMA = "cem-continuum-circle-suite-v1"


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(candidate - reference) / denominator)


def run_fixture(
    *,
    output_dir: Path,
    mesh_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("suite_schema") != SUITE_SCHEMA:
        raise ValueError("NGSolve continuum fixture schema mismatch")
    config = Config(
        n_electrodes=int(metadata["n_electrodes"]),
        radius_m=float(metadata["radius"]),
        conductivity_s_per_m=float(metadata["conductivity"]),
        contact_impedance=float(metadata["contact_impedance"]),
        electrode_coverage=float(metadata["electrode_coverage"]),
        potential_order=1,
        timing_repeats=3,
    )
    mesh, verification, import_seconds = load_verified_mesh(mesh_path, metadata_path)
    space, robin_matrix, coupling, electrode_matrix = assemble_blocks(config, mesh)
    currents = np.asarray(metadata["current_patterns"], dtype=np.float64)
    a_r = _as_csc(robin_matrix)
    c = _as_csc(coupling)
    d = _as_csc(electrode_matrix)
    classic_potential, classic_voltage = _solve_classic(
        _classic_state(a_r, c, d), currents
    )
    robin_potential, robin_voltage = _solve_robin(_robin_state(a_r, c, d), currents)
    report: dict[str, object] = {
        "solver": "NGSolve",
        "suite_schema": SUITE_SCHEMA,
        "case_id": metadata["case_id"],
        "mesh_level_id": metadata["mesh_level_id"],
        "physical_config": {
            "radius": config.radius_m,
            "n_electrodes": config.n_electrodes,
            "electrode_coverage": config.electrode_coverage,
            "conductivity": config.conductivity_s_per_m,
            "contact_impedance": config.contact_impedance,
            "drive_skip": int(metadata["drive_skip"]),
        },
        "discretization": {
            **verification,
            "degrees_of_freedom": int(space.ndof),
            "element_family": "NGSolve P1 H1 triangle",
            "potential_order": 1,
            "scalar_dtype": "float64",
            "mesh_import_verified": True,
            "mesh_import_seconds": import_seconds,
            "target_h": metadata["target_h"],
            "h_max": metadata["h_max"],
            "boundary_chord_max": metadata["boundary_chord_max"],
            "boundary_sagitta_max": metadata["boundary_sagitta_max"],
        },
        "linear_solver": {
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus dense reduced LU",
            "scalar_dtype": "float64",
        },
        "within_solver": {
            "electrode_voltage_relative_l2": _relative_l2(
                robin_voltage, classic_voltage
            ),
            "body_potential_relative_l2": _relative_l2(
                robin_potential, classic_potential
            ),
        },
        "raw_electrode_voltages": {
            "classic": np.asarray(classic_voltage, dtype=np.float64).tolist(),
            "robin_transconductance": np.asarray(
                robin_voltage, dtype=np.float64
            ).tolist(),
        },
        "implementation_note": (
            "NGSolve imports the canonical Gmsh P1 true-circle chord mesh, verifies "
            "its fingerprint, assembles the CEM blocks, and uses independent "
            "float64 Classic and Robin factor states."
        ),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    (output_path / "ngsolve_report.json").write_text(
        serialized + "\n", encoding="utf-8"
    )
    return report


def run_suite(suite_output: Path) -> int:
    """Run every fixture from one prepared continuum-suite manifest."""

    root = Path(suite_output).resolve()
    manifest = json.loads((root / "suite_manifest.json").read_text(encoding="utf-8"))
    fixtures = manifest.get("fixtures", [])
    if not fixtures:
        raise ValueError("continuum suite manifest contains no fixtures")
    for fixture in fixtures:
        run_fixture(
            output_dir=Path(fixture["case_dir"]),
            mesh_path=Path(fixture["msh"]),
            metadata_path=Path(fixture["metadata"]),
        )
    return len(fixtures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--mesh-metadata", type=Path)
    args = parser.parse_args()
    if args.suite_output is not None:
        if any(
            value is not None
            for value in (args.output_dir, args.mesh, args.mesh_metadata)
        ):
            parser.error(
                "--suite-output cannot be combined with single-fixture arguments"
            )
        count = run_suite(args.suite_output)
        print(
            f"NGSolve continuum reports: {count} fixtures in {args.suite_output.resolve()}"
        )
        return 0
    if any(value is None for value in (args.output_dir, args.mesh, args.mesh_metadata)):
        parser.error(
            "single-fixture mode requires --output-dir, --mesh, and --mesh-metadata"
        )
    run_fixture(
        output_dir=args.output_dir.resolve(),
        mesh_path=args.mesh.resolve(),
        metadata_path=args.mesh_metadata.resolve(),
    )
    print(f"NGSolve continuum report: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
