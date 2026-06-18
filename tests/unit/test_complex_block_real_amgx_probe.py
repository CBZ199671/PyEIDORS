"""Unit tests for the complex block-real AmgX diagnostic harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy import sparse


def _load_probe_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "diagnostics"
        / "complex_block_real_amgx_probe.py"
    )
    spec = importlib.util.spec_from_file_location(
        "complex_block_real_probe_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_block_jacobi_amgx_profile_omits_selector_rejected_by_aggregation() -> None:
    module = _load_probe_module()

    class FakePETSc:
        options: dict[str, str] = {}

        @classmethod
        def Options(cls):
            return cls.options

    module._apply_amgx_options(FakePETSc, "probe_", "block_jacobi")

    assert "probe_pc_amgx_amg_method" in FakePETSc.options
    assert "probe_pc_amgx_smoother" in FakePETSc.options
    assert "probe_pc_amgx_selector" not in FakePETSc.options


def test_solve_block_real_defaults_to_working_real_jacobi_profile(monkeypatch) -> None:
    module = _load_probe_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "complex_block_real_amgx_probe.py",
            "solve-block-real",
            "--input-dir",
            "out",
        ],
    )

    args = module._parse_args()

    assert args.amgx_profile == "real_jacobi_l1"


def test_reference_electrode_gauge_recenter_matches_zero_mean_electrodes() -> None:
    module = _load_probe_module()
    solution = np.array(
        [
            [10.0 + 1.0j],
            [20.0 + 1.0j],
            [5.0 + 2.0j],
            [7.0 + 4.0j],
            [0.0 + 0.0j],
        ]
    )

    recentered = module._recenter_reference_electrode_gauge(
        solution,
        potential_dofs=2,
        n_elec=2,
    )

    electrodes = recentered[2:4, :]
    np.testing.assert_allclose(electrodes.mean(axis=0), 0.0)
    np.testing.assert_allclose(recentered[4, :], 0.0)


def test_solve_block_real_recenters_runtime_reference_and_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_probe_module()
    input_dir = tmp_path / "case"
    input_dir.mkdir()
    sparse.save_npz(
        input_dir / "system_matrix_complex.npz", sparse.eye(4, dtype=complex)
    )
    np.save(input_dir / "rhs_complex.npy", np.zeros((4, 1), dtype=np.complex128))
    np.save(
        input_dir / "reference_solution_complex.npy",
        np.array([[10.0], [11.0], [12.0], [0.0]], dtype=np.complex128),
    )
    (input_dir / "metadata.json").write_text(
        json.dumps(
            {
                "gauge": "reference-electrode-row",
                "potential_dofs": 2,
                "n_elec": 1,
            }
        ),
        encoding="utf-8",
    )
    candidate = np.array([[0.0], [1.0], [2.0], [0.0]], dtype=np.complex128)

    def fake_solve(*_args, **_kwargs):
        return module.complex_rhs_to_block_real(candidate), {
            "solve_seconds": 0.1,
            "iterations_per_rhs": [1],
        }

    monkeypatch.setattr(module, "_solve_block_real_with_petsc_amgx", fake_solve)
    output_json = tmp_path / "report.json"

    module._command_solve_block_real(
        Namespace(
            input_dir=input_dir,
            output_json=output_json,
            mat_type="aijcusparse",
            amgx_profile="real_jacobi_l1",
            rtol=1.0e-8,
            atol=1.0e-10,
            max_it=10,
        )
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["reference_kind"] == "runtime_reference"
    assert report["solution_error_vs_reference"]["relative_l2"] == 0.0
    assert report["electrode_voltage_error_vs_reference"]["relative_l2"] == 0.0
