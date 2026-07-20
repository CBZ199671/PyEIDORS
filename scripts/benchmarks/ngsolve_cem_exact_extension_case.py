#!/usr/bin/env python3
"""Run one preregistered rational extension case with NGSolve P1 float64."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import ngsolve as ngs
import numpy as np
from scipy.io import savemat
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_fair_common import benchmark_preassembled_blocks
from scripts.benchmarks.ngsolve_cem_formulations import (
    Config,
    load_verified_mesh,
    ngsolve_csr,
)


def conductivity_digest(values: np.ndarray) -> str:
    exact_strings = []
    for value in np.asarray(values, dtype=np.float64).reshape(-1):
        fraction = Fraction.from_float(float(value))
        exact_strings.append(f"{fraction.numerator}/{fraction.denominator}")
    return hashlib.sha256(
        json.dumps(exact_strings, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _update_array_digest(digest, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))


def block_payload_sha256(
    robin_matrix,
    coupling,
    electrode_matrix,
    currents: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"cem-low-z-block-payload-v1")
    for matrix in (robin_matrix, coupling, electrode_matrix):
        canonical = csc_matrix(matrix, dtype=np.float64)
        canonical.sum_duplicates()
        canonical.sort_indices()
        _update_array_digest(digest, np.asarray(canonical.shape, dtype="<i8"))
        _update_array_digest(digest, canonical.indptr.astype("<i8", copy=False))
        _update_array_digest(digest, canonical.indices.astype("<i8", copy=False))
        _update_array_digest(digest, canonical.data.astype("<f8", copy=False))
    _update_array_digest(digest, np.ascontiguousarray(currents, dtype="<f8"))
    return digest.hexdigest()


def _material_conductivity(
    metadata: dict[str, object],
    mesh: ngs.Mesh,
) -> tuple[ngs.CoefficientFunction, np.ndarray]:
    values = np.asarray(metadata["material_conductivities"], dtype=np.float64).reshape(
        -1
    )
    materials = tuple(str(name) for name in mesh.GetMaterials())
    expected = {f"sigma_{index}" for index in range(1, values.size + 1)}
    if set(materials) != expected:
        raise RuntimeError(f"NGSolve material names {materials} != {sorted(expected)}")
    material_values = {
        f"sigma_{index}": float(value) for index, value in enumerate(values, start=1)
    }
    ordered_values = tuple(material_values[name] for name in materials)
    coefficient = mesh.MaterialCF(material_values)
    imported = np.asarray(
        [
            ordered_values[int(element.index) - 1]
            for element in mesh.ngmesh.Elements2D()
        ],
        dtype=np.float64,
    )
    if conductivity_digest(imported) != str(metadata["conductivity_digest"]):
        raise RuntimeError("NGSolve per-cell conductivity digest mismatch")
    return coefficient, imported


def assemble_extension_blocks(
    config: Config,
    mesh: ngs.Mesh,
    conductivity: ngs.CoefficientFunction,
):
    space = ngs.H1(mesh, order=config.potential_order)
    trial, test = space.TnT()
    robin_form = ngs.BilinearForm(space)
    robin_form += ngs.SymbolicBFI(conductivity * ngs.grad(trial) * ngs.grad(test))
    coupling = np.zeros((space.ndof, config.n_electrodes), dtype=np.float64)
    electrode_diagonal = np.zeros(config.n_electrodes, dtype=np.float64)
    for electrode in range(config.n_electrodes):
        boundary = mesh.Boundaries(f"electrode_{electrode + 1}")
        robin_form += ngs.SymbolicBFI(
            trial * test / config.contact_impedance,
            definedon=boundary,
        )
    robin_form.Assemble()
    for electrode in range(config.n_electrodes):
        boundary = mesh.Boundaries(f"electrode_{electrode + 1}")
        linear_form = ngs.LinearForm(space)
        linear_form += ngs.SymbolicLFI(
            -test / config.contact_impedance,
            definedon=boundary,
        )
        linear_form.Assemble()
        coupling[:, electrode] = linear_form.vec.FV().NumPy()
        electrode_diagonal[electrode] = float(
            ngs.Integrate(1.0 / config.contact_impedance, mesh, definedon=boundary)
        )
    robin_matrix = ngsolve_csr(robin_form.mat, (space.ndof, space.ndof))
    return space, robin_matrix, coupling, np.diag(electrode_diagonal)


def run_case(
    mesh_path: Path,
    metadata_path: Path,
    output_path: Path,
    *,
    timing_repeats: int,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    count = int(metadata["n_electrodes"])
    config = Config(
        n_electrodes=count,
        conductivity_s_per_m=float(metadata["material_conductivities"][0]),
        contact_impedance=float(metadata["contact_impedance"]),
        electrode_coverage=0.5,
        potential_order=1,
        timing_repeats=int(timing_repeats),
    )
    mesh, mesh_verification, mesh_import_seconds = load_verified_mesh(
        mesh_path,
        metadata_path,
    )
    coefficient, _ = _material_conductivity(metadata, mesh)
    started = time.perf_counter()
    space, robin_matrix, coupling, electrode_matrix = assemble_extension_blocks(
        config,
        mesh,
        coefficient,
    )
    assembly_seconds = float(time.perf_counter() - started)
    currents = np.asarray(metadata["current_patterns"], dtype=np.float64)
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    block_path = output_path.parent / "ngsolve_assembled_blocks.mat"
    block_digest = block_payload_sha256(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
    )
    savemat(
        block_path,
        {
            "A_R": robin_matrix,
            "C": coupling,
            "D": electrode_matrix,
            "currents": currents,
            "case_id": metadata["case_id"],
            "assembly": "NGSolve",
            "assembled_blocks_sha256": block_digest,
        },
    )
    report: dict[str, object] = {
        "solver": "NGSolve",
        "ngsolve_version": str(ngs.__version__),
        "suite_schema": metadata["suite_schema"],
        "case_id": metadata["case_id"],
        "physical_config": {
            "n_electrodes": count,
            "conductivity_pattern": metadata["conductivity_pattern"],
            "conductivity_digest": metadata["conductivity_digest"],
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
            "cell_conductivity_order": "Gmsh physical material per triangle",
        },
        "linear_solver": {
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus dense reduced LU",
            "scalar_dtype": "float64",
        },
        "timing": timing,
        "within_solver": parity,
        "assembled_blocks": str(block_path.resolve()),
        "assembled_blocks_sha256": block_digest,
        "raw_electrode_voltages": {
            formulation: np.asarray(voltage, dtype=np.float64).tolist()
            for formulation, voltage in voltages.items()
        },
    }
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
    print(f"NGSolve extension case {report['case_id']}: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
