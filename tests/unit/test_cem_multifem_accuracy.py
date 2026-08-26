"""Fair multi-FEM Robin CEM experiment guardrails."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from scripts.benchmarks.cem_multifem_accuracy import (
    ENVIRONMENT_SCHEMA,
    build_environment_report,
    run_freefem_fixture,
    run_getfem_fixture,
    run_mfem_fixture,
    runtime_environment,
    runtime_paths,
)
from scripts.benchmarks.cem_multifem_common import (
    FORMULATION_ROBIN,
    PRIMARY_METHODS,
    REPORT_SCHEMA,
    solve_robin_from_blocks,
    validate_native_report,
)
from scripts.benchmarks.cem_block_audit import (
    assemble_analytic_blocks,
    build_nonuniform_fixture,
)


def test_v804_runtime_paths_are_isolated_and_deterministic(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path)
    assert paths.prefix == tmp_path.resolve()
    assert paths.mfem_prefix == tmp_path.resolve() / "mfem-4.9"
    assert paths.freefem == (tmp_path.resolve() / "ubuntu-jammy/usr/bin/FreeFem++-nw")
    assert paths.getfem_pythonpath == (
        tmp_path.resolve() / "ubuntu-jammy/usr/lib/python3/dist-packages"
    )

    env = runtime_environment(paths)
    assert env is not os.environ
    assert env["PATH"].split(os.pathsep)[0] == str(paths.mfem_prefix / "bin")
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(paths.getfem_pythonpath)
    assert str(paths.deb_root / "usr/lib/freefem++") in env["FF_LOADPATH"]


def test_v804_doctor_fails_closed_when_prefix_is_missing(tmp_path: Path) -> None:
    report = build_environment_report(tmp_path / "missing")
    assert report["schema"] == ENVIRONMENT_SCHEMA
    assert report["ok"] is False
    assert report["checks"]["metadata_schema"] is False
    assert report["checks"]["mfem_library"] is False
    assert report["checks"]["freefem"] is False
    assert report["checks"]["getfem"] is False


def test_v807_setup_rejects_invalid_mfem_build_jobs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts/benchmarks/setup_cem_multifem_env.sh"
    env = dict(os.environ)
    env["PYEIDORS_CEM_MULTIFEM_PREFIX"] = str(tmp_path / "runtime")
    env["PYEIDORS_CEM_MFEM_BUILD_JOBS"] = "0"
    completed = subprocess.run(
        [str(script), "install"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "must be a positive integer" in completed.stderr
    assert not (tmp_path / "runtime").exists()


def test_v808_runtime_environment_drops_nix_and_python_contamination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/nix/store/fake-lib")
    monkeypatch.setenv("LD_PRELOAD", "/nix/store/fake-preload.so")
    monkeypatch.setenv("PYTHONPATH", "/nix/store/fake-python")
    monkeypatch.setenv("PYTHONHOME", "/nix/store/fake-home")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/fake-venv")
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/fake-conda")
    env = runtime_environment(runtime_paths(tmp_path))
    assert "/nix/store" not in env["PATH"]
    assert "/nix/store" not in env["LD_LIBRARY_PATH"]
    assert "/nix/store" not in env["PYTHONPATH"]
    assert "LD_PRELOAD" not in env
    assert "PYTHONHOME" not in env
    assert "VIRTUAL_ENV" not in env
    assert "CONDA_PREFIX" not in env


def _valid_native_report() -> tuple[dict, dict]:
    fixture = build_nonuniform_fixture()
    blocks, _ = assemble_analytic_blocks(fixture)
    solution = solve_robin_from_blocks(
        K=blocks["K"],
        B=blocks["B"],
        C_plus=blocks["C_plus"],
        D=blocks["D"],
        currents=fixture.currents,
    )
    fixture_payload = {
        "mesh_fingerprint": fixture.mesh_fingerprint,
        "currents": fixture.currents,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "solver": "MFEM",
        "formulation": FORMULATION_ROBIN,
        "implementation": {"native_assembly": True},
        "discretization": {
            "mesh_fingerprint": fixture.mesh_fingerprint,
            "mesh_import_verified": True,
            "potential_order": 1,
            "geometry_order": 1,
            "scalar_dtype": "float64",
            "imported_nodes": fixture.nodes.tolist(),
            "imported_cells_zero_based": fixture.cells.tolist(),
            "imported_tagged_boundary_edges_zero_based": (
                fixture.tagged_edges.tolist()
            ),
        },
        "physical_config": {"currents": fixture.currents.tolist()},
        "blocks": {
            key: blocks[key].tolist() for key in ("K", "B", "C_plus", "D", "A_R")
        },
        "solution": {
            "T": solution.T.tolist(),
            "reduced_map": solution.reduced_map.tolist(),
            "body_potential": solution.body_potential.tolist(),
            "electrode_voltage": solution.electrode_voltage.tolist(),
        },
    }
    return report, fixture_payload


def test_v806_primary_method_set_is_exactly_the_registered_six() -> None:
    assert [(method.solver, method.formulation) for method in PRIMARY_METHODS] == [
        ("EIDORS", "classic_augmented"),
        ("PyEIDORS-DOLFINx", "robin_transconductance"),
        ("NGSolve", "robin_transconductance"),
        ("MFEM", "robin_transconductance"),
        ("FreeFEM", "robin_transconductance"),
        ("GetFEM", "robin_transconductance"),
    ]


def test_v805_native_report_rebuilds_robin_identities() -> None:
    report, fixture = _valid_native_report()
    metrics = validate_native_report(report, fixture, expected_solver="MFEM")
    assert max(metrics.values()) < 5.0e-12


def test_v805_native_report_rejects_copied_hash_or_non_p1() -> None:
    report, fixture = _valid_native_report()
    report["discretization"]["imported_nodes"][0][0] += 0.0625
    with pytest.raises(ValueError, match="declared fingerprint"):
        validate_native_report(report, fixture, expected_solver="MFEM")

    report, fixture = _valid_native_report()
    report["discretization"]["potential_order"] = 2
    with pytest.raises(ValueError, match="requires P1"):
        validate_native_report(report, fixture, expected_solver="MFEM")


def test_v806_accuracy_report_rejects_timing_and_wrong_formulation() -> None:
    report, fixture = _valid_native_report()
    report["diagnostics"] = {"elapsed_seconds": 0.01}
    with pytest.raises(ValueError, match="must not contain timing"):
        validate_native_report(report, fixture, expected_solver="MFEM")

    report, fixture = _valid_native_report()
    report["formulation"] = "classic_augmented"
    with pytest.raises(ValueError, match="formulation mismatch"):
        validate_native_report(report, fixture, expected_solver="MFEM")


def test_v683_multifem_derivation_contains_equivalence_and_energy_proof() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/benchmarks/cem_multifem_robin_derivation.md").read_text(
        encoding="utf-8"
    )
    for marker in (
        "T=D-C_+^TA_R^{-1}C_+",
        "U^TTU",
        "Q^TTQ",
        "symmetric positive definite",
        "augmented and Robin–transconductance methods produce identical",
    ):
        assert marker in text


def test_v805_v809_mfem_native_p1_assembly_and_solve(tmp_path: Path) -> None:
    prefix = Path.home() / ".local/share/pyeidors-cem-multifem"
    paths = runtime_paths(prefix)
    if not paths.mfem_library.is_file():
        pytest.skip("isolated MFEM runtime is not installed")
    result = run_mfem_fixture(tmp_path, prefix=prefix)
    assert result["all_pass"] is True
    assert max(result["analytic_block_relative_frobenius"].values()) < 5.0e-12
    assert max(result["native_identity_metrics"].values()) < 5.0e-11

    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/benchmarks/mfem_cem_robin.cpp").read_text(
        encoding="utf-8"
    )
    for native_marker in (
        "DiffusionIntegrator",
        "MassIntegrator",
        "BoundaryLFIntegrator",
        "UMFPackSolver",
    ):
        assert native_marker in source


def test_v805_v810_freefem_native_p1_assembly_and_solve(tmp_path: Path) -> None:
    prefix = Path.home() / ".local/share/pyeidors-cem-multifem"
    paths = runtime_paths(prefix)
    plugin = paths.deb_root / "usr/lib/freefem++/gmsh.so"
    if not paths.freefem.is_file() or not plugin.is_file():
        pytest.skip("isolated FreeFEM runtime with Gmsh plugin is not installed")
    result = run_freefem_fixture(tmp_path, prefix=prefix)
    assert result["all_pass"] is True
    assert max(result["analytic_block_relative_frobenius"].values()) < 5.0e-12
    assert max(result["native_identity_metrics"].values()) < 5.0e-11

    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/benchmarks/freefem_cem_robin.edp").read_text(
        encoding="utf-8"
    )
    for native_marker in (
        "gmshload(meshPath, renum=0)",
        "stiffnessForm(Vh, Vh)",
        "boundaryMassForm(Vh, Vh)",
        "AR^-1 * rhs",
        "reducedSparse^-1 * reducedRhs",
    ):
        assert native_marker in source

    setup = (root / "scripts/benchmarks/setup_cem_multifem_env.sh").read_text(
        encoding="utf-8"
    )
    assert "libfreefem++=4.9+dfsg1-2build1" in setup


def test_v805_getfem_native_p1_assembly_and_solve(tmp_path: Path) -> None:
    prefix = Path.home() / ".local/share/pyeidors-cem-multifem"
    paths = runtime_paths(prefix)
    if not paths.getfem_pythonpath.joinpath("getfem").is_dir():
        pytest.skip("isolated GetFEM runtime is not installed")
    result = run_getfem_fixture(tmp_path, prefix=prefix)
    assert result["all_pass"] is True
    assert max(result["analytic_block_relative_frobenius"].values()) < 5.0e-12
    assert max(result["native_identity_metrics"].values()) < 5.0e-11

    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/benchmarks/getfem_cem_robin.py").read_text(
        encoding="utf-8"
    )
    for native_marker in (
        'getfem.Mesh("import", "gmsh", mesh_path)',
        "mesh_fem.set_classical_fem(1)",
        "getfem.asm_generic(",
        "getfem.asm_mass_matrix(",
        "getfem.linsolve_mumps(",
    ):
        assert native_marker in source
