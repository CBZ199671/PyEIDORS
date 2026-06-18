from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from types import SimpleNamespace

from pyeidors.forward.block_real_amgx import solve_problem_files
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact
from pyeidors.utils.block_real import (
    absolute_error_summary,
    block_real_solution_to_complex,
    complex_csr_to_block_real,
    complex_rhs_to_block_real,
    relative_l2_error,
)
import pyeidors.forward.block_real_amgx as block_real_amgx


def test_complex_csr_to_block_real_matches_complex_matvec() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [2.0 + 1.0j, -0.5j],
                [3.0 - 2.0j, 4.0 + 0.25j],
            ],
            dtype=np.complex128,
        )
    )
    x = np.array([1.0 - 0.5j, -2.0 + 0.25j], dtype=np.complex128)

    block = complex_csr_to_block_real(matrix)
    got = block_real_solution_to_complex(
        block @ complex_rhs_to_block_real(x),
        original_size=matrix.shape[0],
    )

    np.testing.assert_allclose(got, matrix @ x)


def test_block_real_roundtrip_handles_multiple_rhs() -> None:
    values = np.array(
        [[1.0 + 2.0j, 3.0 - 4.0j], [5.0 + 0.5j, -1.0j]],
        dtype=np.complex128,
    )

    block = complex_rhs_to_block_real(values)
    restored = block_real_solution_to_complex(block, original_size=values.shape[0])

    np.testing.assert_allclose(restored, values)


def test_absolute_error_summary_is_json_safe() -> None:
    reference = np.array([1.0 + 1.0j, 2.0 - 1.0j])
    candidate = reference + np.array([0.0, 0.1j])

    summary = absolute_error_summary(reference, candidate)

    assert summary["relative_l2"] == relative_l2_error(reference, candidate)
    assert summary["max_abs"] > 0.0
    assert set(summary) == {"relative_l2", "max_abs", "mean_abs", "rms_abs"}


def test_block_real_amgx_cli_defaults_match_large_cem_tolerance(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "block_real_amgx.py",
            "solve-files",
            "--input-dir",
            "input",
            "--output-json",
            "out.json",
        ],
    )

    args = block_real_amgx._parse_args()

    assert args.rtol == block_real_amgx.BLOCK_REAL_AMGX_DEFAULT_RTOL
    assert args.rtol == 1.0e-6
    assert args.ksp_type == block_real_amgx.BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE
    assert args.ksp_type == "bcgs"
    assert args.max_it == block_real_amgx.BLOCK_REAL_AMGX_DEFAULT_MAX_IT


def test_block_real_amgx_worker_env_drops_complex_parent_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/complex-source")
    monkeypatch.setenv("PYTHONHOME", "/tmp/python-home")
    monkeypatch.setenv("PYEIDORS_ENV_PROFILE", "complex64-cuda")
    monkeypatch.setenv("PYEIDORS_PETSC_SCALAR_TYPE", "complex64")
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PRECISION", "complex64")

    env = block_real_amgx._external_worker_env()

    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYEIDORS_ENV_PROFILE" not in env
    assert "PYEIDORS_PETSC_SCALAR_TYPE" not in env
    assert "EIT_APP_GUI_RUNTIME_PROFILE" not in env
    assert "EIT_APP_GUI_PRECISION" not in env


def test_block_real_amgx_residual_uses_raw_gauge_solution(
    monkeypatch,
    tmp_path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    raw_solution = np.array([[10.0 + 0.0j], [3.0 + 0.0j], [0.0 + 0.0j]])
    sparse.save_npz(
        input_dir / "system_matrix_complex.npz", sparse.eye(3, dtype=complex)
    )
    np.save(input_dir / "rhs_complex.npy", raw_solution)
    (input_dir / "metadata.json").write_text(
        ('{"gauge":"reference-electrode-row","potential_dofs":1,"n_elec":1}'),
        encoding="utf-8",
    )

    def fake_solve(*_args, **_kwargs):
        return complex_rhs_to_block_real(raw_solution), {
            "rtol": 1.0e-6,
            "iterations_per_rhs": [1],
            "converged_reasons": [2],
            "true_relative_residual_max": 0.0,
        }

    monkeypatch.setattr(block_real_amgx, "solve_block_real_with_petsc_amgx", fake_solve)

    report = solve_problem_files(input_dir=input_dir, output_json=tmp_path / "out.json")

    assert report["complex_true_residual"]["relative_max"] == 0.0
    assert report["complex_residual_after_gauge_recenter"]["relative_max"] > 0.0
    saved = read_hdf5_artifact(report["candidate_solution_path"]).arrays["solution"]
    assert saved[1, 0] == 0.0


def test_external_block_real_amgx_uses_persistent_worker_by_default(
    monkeypatch,
    tmp_path,
) -> None:
    matrix = sparse.eye(2, dtype=np.complex64)
    rhs = np.array([[1.0 + 2.0j], [3.0 + 4.0j]], dtype=np.complex64)
    runtime_root = tmp_path / "runtime"
    calls: list[dict[str, object]] = []

    class FakeWorker:
        stdout_tail = ""
        stderr_tail = ""

        def solve(self, **kwargs):
            output_json = kwargs["output_json"]
            solution_path = output_json.with_suffix(".solution_complex.npy")
            np.save(solution_path, rhs)
            calls.append(dict(kwargs))
            block_real_amgx._write_json(
                output_json,
                {
                    "route": "complex_block_real_cuda_amgx",
                    "candidate_solution_path": str(solution_path),
                    "solver": {
                        "ksp_type": kwargs["ksp_type"],
                        "iterations_per_rhs": [2],
                        "converged_reasons": [2],
                        "true_relative_residual_max": 0.0,
                    },
                    "complex_true_residual": {"relative_max": 0.0},
                },
            )
            return {"ksp_type": kwargs["ksp_type"]}

    monkeypatch.setattr(block_real_amgx, "petsc_scalar_is_complex", lambda: True)
    monkeypatch.setattr(block_real_amgx, "petsc_scalar_dtype_name", lambda: "complex64")
    monkeypatch.setattr(block_real_amgx, "pyeidors_runtime_root", lambda: runtime_root)
    monkeypatch.setattr(
        block_real_amgx, "_find_repo_root", lambda _explicit=None: tmp_path
    )
    monkeypatch.setattr(block_real_amgx, "_persistent_worker_enabled", lambda: True)
    monkeypatch.setattr(
        block_real_amgx, "_persistent_worker", lambda _repo: FakeWorker()
    )

    def fail_subprocess_run(*_args, **_kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("persistent worker path should not use subprocess.run")

    monkeypatch.setattr(block_real_amgx.subprocess, "run", fail_subprocess_run)

    solution, report = (
        block_real_amgx.solve_complex_system_with_external_block_real_amgx(
            matrix,
            rhs,
            potential_dofs=1,
            n_elec=1,
        )
    )

    np.testing.assert_allclose(solution, rhs)
    assert report["external_worker_persistent"] is True
    assert report["external_worker_metadata"]["ksp_type"] == "bcgs"
    assert calls[0]["ksp_type"] == "bcgs"


def test_external_block_real_amgx_transport_error_falls_back_to_one_shot(
    monkeypatch,
    tmp_path,
) -> None:
    matrix = sparse.eye(1, dtype=np.complex64)
    rhs = np.array([[1.0 + 0.5j]], dtype=np.complex64)
    runtime_root = tmp_path / "runtime"

    monkeypatch.setattr(block_real_amgx, "petsc_scalar_is_complex", lambda: True)
    monkeypatch.setattr(block_real_amgx, "petsc_scalar_dtype_name", lambda: "complex64")
    monkeypatch.setattr(block_real_amgx, "pyeidors_runtime_root", lambda: runtime_root)
    monkeypatch.setattr(
        block_real_amgx, "_find_repo_root", lambda _explicit=None: tmp_path
    )
    monkeypatch.setattr(block_real_amgx, "_persistent_worker_enabled", lambda: True)

    def broken_worker(_repo):
        raise block_real_amgx._BlockRealAmgxWorkerTransportError("worker offline")

    monkeypatch.setattr(block_real_amgx, "_persistent_worker", broken_worker)

    def fake_run(cmd, cwd, text, stdout, stderr, timeout, check):
        output_json = Path(cmd[cmd.index("--output-json") + 1])
        solution_path = output_json.with_suffix(".solution_complex.npy")
        np.save(solution_path, rhs)
        block_real_amgx._write_json(
            output_json,
            {
                "route": "complex_block_real_cuda_amgx",
                "candidate_solution_path": str(solution_path),
                "solver": {
                    "ksp_type": "bcgs",
                    "iterations_per_rhs": [1],
                    "converged_reasons": [2],
                    "true_relative_residual_max": 0.0,
                },
                "complex_true_residual": {"relative_max": 0.0},
            },
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(block_real_amgx.subprocess, "run", fake_run)

    _solution, report = (
        block_real_amgx.solve_complex_system_with_external_block_real_amgx(
            matrix,
            rhs,
            potential_dofs=1,
            n_elec=1,
        )
    )

    assert report["external_worker_persistent"] is False
    assert "worker offline" in report["external_worker_transport_error"]
