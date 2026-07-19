#!/usr/bin/env python3
"""Run one rational-circular exact CEM case with NGSolve P1 float64."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import ngsolve as ngs
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_fair_common import benchmark_preassembled_blocks
from scripts.benchmarks.ngsolve_cem_formulations import (
    Config,
    assemble_blocks,
    load_verified_mesh,
)


def run_case(
    mesh_path: Path,
    metadata_path: Path,
    output_path: Path,
    *,
    timing_repeats: int,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = Config(
        n_electrodes=int(metadata["n_electrodes"]),
        conductivity_s_per_m=float(metadata["conductivity"]),
        contact_impedance=float(metadata["contact_impedance"]),
        electrode_coverage=0.64,
        potential_order=1,
        timing_repeats=int(timing_repeats),
    )
    mesh, mesh_verification, mesh_import_seconds = load_verified_mesh(
        mesh_path,
        metadata_path,
    )
    started = time.perf_counter()
    space, robin_matrix, coupling, electrode_matrix = assemble_blocks(config, mesh)
    assembly_seconds = float(time.perf_counter() - started)
    currents = np.asarray(metadata["current_patterns"], dtype=np.float64)
    if currents.shape != (config.n_electrodes, config.n_electrodes):
        raise ValueError(f"unexpected exact-suite current shape: {currents.shape}")
    timing, voltages, parity = benchmark_preassembled_blocks(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
        repeats=config.timing_repeats,
    )
    timing.update(
        {
            "mesh_import_seconds": mesh_import_seconds,
            "assembly_seconds": assembly_seconds,
        }
    )
    report: dict[str, object] = {
        "solver": "NGSolve",
        "ngsolve_version": str(ngs.__version__),
        "suite_schema": metadata["suite_schema"],
        "case_id": metadata["case_id"],
        "physical_config": {
            "n_electrodes": config.n_electrodes,
            "conductivity": config.conductivity_s_per_m,
            "contact_impedance": config.contact_impedance,
            "drive_skip": int(metadata["drive_skip"]),
        },
        "discretization": {
            **mesh_verification,
            "degrees_of_freedom": int(space.ndof),
            "element_family": "NGSolve P1 H1 triangle",
            "potential_order": 1,
            "electrode_integration": "NGSolve boundary SymbolicBFI/SymbolicLFI",
            "mesh_import_verified": True,
            "common_mesh_role": "imported exact-suite Gmsh 2.2",
        },
        "linear_solver": {
            "assembly": "NGSolve",
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus dense reduced LU",
            "scalar_dtype": "float64",
        },
        "timing": timing,
        "within_solver": parity,
        "raw_electrode_voltages": {
            formulation: np.asarray(voltage, dtype=np.float64).tolist()
            for formulation, voltage in voltages.items()
        },
        "implementation_note": (
            "NGSolve imports and verifies the exact-suite Gmsh mesh, then both "
            "formulations use independent factor states and identical RHS matrices."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=11)
    args = parser.parse_args()
    report = run_case(
        args.mesh.resolve(),
        args.mesh_metadata.resolve(),
        args.output.resolve(),
        timing_repeats=int(args.timing_repeats),
    )
    print(f"NGSolve exact CEM case {report['case_id']}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
