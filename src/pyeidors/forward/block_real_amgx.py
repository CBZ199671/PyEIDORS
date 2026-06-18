"""Block-real PETSc/AmgX bridge for complex CEM linear systems."""

from __future__ import annotations

import argparse
import atexit
import contextlib
from collections import deque
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.forward.complex_support import (
    petsc_scalar_dtype_name,
    petsc_scalar_is_complex,
)
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact, write_hdf5_artifact
from pyeidors.runtime_paths import pyeidors_runtime_root
from pyeidors.utils.block_real import (
    block_real_solution_to_complex,
    complex_csr_to_block_real,
    complex_rhs_to_block_real,
)

BLOCK_REAL_AMGX_DEFAULT_RTOL = 1.0e-6
BLOCK_REAL_AMGX_DEFAULT_ATOL = 1.0e-12
BLOCK_REAL_AMGX_DEFAULT_MAX_IT = 4000
BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE = "bcgs"
BLOCK_REAL_AMGX_ARRAY_SCHEMA = "pyeidors-block-real-amgx-array-v1"
_WORKER_COMMAND_ENV = "PYEIDORS_BLOCK_REAL_AMGX_WORKER_COMMAND"
_ONE_SHOT_COMMAND_ENV = "PYEIDORS_BLOCK_REAL_AMGX_SOLVER_COMMAND"


class _BlockRealAmgxWorkerTransportError(RuntimeError):
    """The persistent block-real AmgX worker process or protocol failed."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _write_array_artifact(path: Path, name: str, values: np.ndarray) -> Path:
    return write_hdf5_artifact(
        path,
        {name: np.asarray(values)},
        {"array_name": name},
        schema=BLOCK_REAL_AMGX_ARRAY_SCHEMA,
    )


def _read_array_artifact(path: Path, name: str) -> np.ndarray:
    source = Path(path)
    if source.exists() and source.suffix.lower() in {".h5", ".hdf5"}:
        artifact = read_hdf5_artifact(source)
        return np.asarray(artifact.arrays[name])
    if source.exists():
        return np.asarray(np.load(source, allow_pickle=False))
    if source.suffix.lower() in {".h5", ".hdf5"}:
        legacy = source.with_suffix(".npy")
        if legacy.exists():
            return np.asarray(np.load(legacy, allow_pickle=False))
    raise FileNotFoundError(source)


def _complex_residual_summary(
    matrix: sparse.spmatrix,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> dict[str, Any]:
    residual = matrix.astype(np.complex128) @ np.asarray(
        solution, dtype=np.complex128
    ) - np.asarray(rhs, dtype=np.complex128)
    residual_norms = np.linalg.norm(residual, axis=0)
    rhs_norms = np.linalg.norm(np.asarray(rhs, dtype=np.complex128), axis=0)
    relative = np.divide(
        residual_norms,
        rhs_norms,
        out=np.zeros_like(residual_norms, dtype=np.float64),
        where=rhs_norms > 0,
    )
    return {
        "residual_norms": [float(value) for value in residual_norms],
        "rhs_norms": [float(value) for value in rhs_norms],
        "relative_residuals": [float(value) for value in relative],
        "relative_max": float(relative.max()) if relative.size else 0.0,
    }


def _recenter_reference_electrode_gauge(
    solution: np.ndarray,
    *,
    potential_dofs: int,
    n_elec: int,
) -> np.ndarray:
    sol = np.asarray(solution).copy()
    if potential_dofs <= 0 or n_elec <= 0:
        return sol
    electrode_block = sol[potential_dofs : potential_dofs + n_elec, :]
    offsets = electrode_block.mean(axis=0, keepdims=True)
    sol[:potential_dofs, :] -= offsets
    sol[potential_dofs : potential_dofs + n_elec, :] -= offsets
    sol[potential_dofs + n_elec, :] = 0.0
    return sol


def _petsc_mat_from_csr(matrix: sparse.csr_matrix, petsc_module: Any, mat_type: str):
    csr = matrix.tocsr()
    mat = petsc_module.Mat().createAIJ(
        size=csr.shape,
        csr=(
            csr.indptr.astype(np.int32, copy=False),
            csr.indices.astype(np.int32, copy=False),
            csr.data.astype(np.float64, copy=False),
        ),
        comm=petsc_module.COMM_SELF,
    )
    mat.assemble()
    if mat_type:
        try:
            converted = mat.convert(mat_type)
            if converted is not mat:
                mat.destroy()
                mat = converted
        except Exception:
            try:
                mat.setType(mat_type)
                mat.assemble()
            except Exception:
                pass
    return mat


def _apply_amgx_options(petsc_module: Any, prefix: str, profile: str) -> None:
    opts = petsc_module.Options()
    if profile == "real_jacobi_l1":
        values = {
            "pc_amgx_smoother": "JACOBI_L1",
            "pc_amgx_exact_coarse_solve": "0",
            "pc_amgx_presweeps": "2",
            "pc_amgx_postsweeps": "2",
            "pc_amgx_coarse_solver": "NOSOLVER",
        }
    else:
        values = {
            "pc_amgx_amg_method": "AGGREGATION",
            "pc_amgx_smoother": "BLOCK_JACOBI",
            "pc_amgx_exact_coarse_solve": "0",
            "pc_amgx_presweeps": "2",
            "pc_amgx_postsweeps": "2",
            "pc_amgx_coarse_solver": "NOSOLVER",
        }
    for key, value in values.items():
        opts[f"{prefix}{key}"] = str(value)


def solve_block_real_with_petsc_amgx(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    mat_type: str = "aijcusparse",
    amgx_profile: str = "real_jacobi_l1",
    ksp_type: str = BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE,
    rtol: float = BLOCK_REAL_AMGX_DEFAULT_RTOL,
    atol: float = BLOCK_REAL_AMGX_DEFAULT_ATOL,
    max_it: int = BLOCK_REAL_AMGX_DEFAULT_MAX_IT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a real block system with PETSc PCAMGX in a real PETSc runtime."""
    from petsc4py import PETSc

    if petsc_scalar_is_complex():
        raise RuntimeError(
            "block-real AmgX requires a real PETSc runtime; use the cuda-amgx "
            "Nix profile for the solve side."
        )

    prefix = "pyeidors_block_real_amgx_"
    A = _petsc_mat_from_csr(matrix, PETSc, mat_type)
    _apply_amgx_options(PETSc, prefix, amgx_profile)
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(A)
    ksp.setType(str(ksp_type or BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE))
    ksp.getPC().setType("amgx")
    ksp.setTolerances(rtol=float(rtol), atol=float(atol), max_it=int(max_it))
    norm_type = "unpreconditioned"
    try:
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    except Exception:
        norm_type = "default"
    ksp.setFromOptions()
    ksp.setUp()

    rhs_2d = np.asarray(rhs, dtype=np.float64)
    if rhs_2d.ndim == 1:
        rhs_2d = rhs_2d.reshape(-1, 1)
    solution = np.empty_like(rhs_2d)
    iterations: list[int] = []
    reasons: list[int] = []
    reported_residual_norms: list[float] = []
    solve_t0 = time.perf_counter()
    b = A.createVecRight()
    x = A.createVecRight()
    for col in range(rhs_2d.shape[1]):
        b.getArray(readonly=False)[:] = rhs_2d[:, col]
        ksp.solve(b, x)
        reason = int(ksp.getConvergedReason())
        reasons.append(reason)
        iterations.append(int(ksp.getIterationNumber()))
        reported_residual_norms.append(float(ksp.getResidualNorm()))
        if reason < 0:
            raise RuntimeError(
                "block-real PCAMGX solve failed for RHS "
                f"{col} with convergence reason {reason}"
            )
        solution[:, col] = x.getArray(readonly=True)
    solve_seconds = float(time.perf_counter() - solve_t0)
    true_residual = matrix @ solution - rhs_2d
    true_residual_norms = np.linalg.norm(true_residual, axis=0)
    rhs_norms = np.linalg.norm(rhs_2d, axis=0)
    true_relative_residuals = np.divide(
        true_residual_norms,
        rhs_norms,
        out=np.zeros_like(true_residual_norms, dtype=np.float64),
        where=rhs_norms > 0,
    )
    diagnostics = {
        "route": "complex_block_real_cuda_amgx",
        "petsc_scalar_type": petsc_scalar_dtype_name(),
        "mat_type": str(A.getType()) if hasattr(A, "getType") else str(mat_type),
        "vec_type": str(x.getType()) if hasattr(x, "getType") else "",
        "ksp_type": str(ksp.getType()) if hasattr(ksp, "getType") else str(ksp_type),
        "pc_type": str(ksp.getPC().getType())
        if hasattr(ksp.getPC(), "getType")
        else "amgx",
        "ksp_norm_type": norm_type,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_it": int(max_it),
        "amgx_profile": str(amgx_profile),
        "solve_seconds": solve_seconds,
        "iterations_per_rhs": iterations,
        "converged_reasons": reasons,
        "reported_residual_norms": reported_residual_norms,
        "true_residual_norms": [float(value) for value in true_residual_norms],
        "rhs_norms": [float(value) for value in rhs_norms],
        "true_relative_residuals": [float(value) for value in true_relative_residuals],
        "true_relative_residual_max": float(true_relative_residuals.max())
        if true_relative_residuals.size
        else 0.0,
    }
    x.destroy()
    b.destroy()
    ksp.destroy()
    A.destroy()
    return solution, diagnostics


def solve_problem_files(
    *,
    input_dir: Path,
    output_json: Path,
    mat_type: str = "aijcusparse",
    amgx_profile: str = "real_jacobi_l1",
    ksp_type: str = BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE,
    rtol: float = BLOCK_REAL_AMGX_DEFAULT_RTOL,
    atol: float = BLOCK_REAL_AMGX_DEFAULT_ATOL,
    max_it: int = BLOCK_REAL_AMGX_DEFAULT_MAX_IT,
) -> dict[str, Any]:
    """Solve a saved complex CEM system as a real block AmgX problem."""
    input_dir = Path(input_dir).resolve()
    output_json = Path(output_json).resolve()
    matrix = sparse.load_npz(input_dir / "system_matrix_complex.npz").tocsr()
    rhs = _read_array_artifact(input_dir / "rhs_complex.h5", "rhs")
    metadata_path = input_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )

    block_matrix = complex_csr_to_block_real(matrix)
    block_rhs = complex_rhs_to_block_real(rhs)
    block_solution, solver_diagnostics = solve_block_real_with_petsc_amgx(
        block_matrix,
        block_rhs,
        mat_type=str(mat_type),
        amgx_profile=str(amgx_profile),
        ksp_type=str(ksp_type),
        rtol=float(rtol),
        atol=float(atol),
        max_it=int(max_it),
    )
    raw_solution = block_real_solution_to_complex(
        block_solution,
        original_size=int(matrix.shape[0]),
    )
    complex_residual = _complex_residual_summary(matrix, raw_solution, rhs)
    candidate_solution = raw_solution

    gauge = str(metadata.get("gauge") or "").strip().lower()
    recentered_residual: dict[str, Any] | None = None
    if gauge == "reference-electrode-row":
        candidate_solution = _recenter_reference_electrode_gauge(
            candidate_solution,
            potential_dofs=int(metadata.get("potential_dofs") or 0),
            n_elec=int(metadata.get("n_elec") or 0),
        )
        recentered_residual = _complex_residual_summary(
            matrix,
            candidate_solution,
            rhs,
        )

    solution_path = output_json.with_suffix(".solution_complex.h5")
    _write_array_artifact(solution_path, "solution", candidate_solution)
    report = {
        "schema_version": 1,
        "route": "complex_block_real_cuda_amgx",
        "input_dir": str(input_dir),
        "source_petsc_scalar_type": metadata.get("petsc_scalar_type"),
        "source_gauge": metadata.get("gauge"),
        "block_matrix": {
            "shape": list(block_matrix.shape),
            "nnz": int(block_matrix.nnz),
            "dtype": str(block_matrix.dtype),
        },
        "solver": solver_diagnostics,
        "complex_true_residual": complex_residual,
        "complex_residual_after_gauge_recenter": recentered_residual,
        "candidate_solution_path": str(solution_path),
    }
    _write_json(output_json, report)
    return report


def _find_repo_root(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.getenv("PYEIDORS_REPO_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path(__file__).resolve().parents[3])
    for candidate in candidates:
        if (candidate / "flake.nix").exists():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot locate the source flake needed to launch the cuda-amgx "
        "block-real solver. Set "
        "PYEIDORS_REPO_ROOT."
    )


def _external_solver_command(
    *,
    input_dir: Path,
    output_json: Path,
    mat_type: str,
    amgx_profile: str,
    ksp_type: str,
    rtol: float,
    atol: float,
    max_it: int,
) -> list[str]:
    args = [
        "solve-files",
        "--input-dir",
        str(input_dir),
        "--output-json",
        str(output_json),
        "--mat-type",
        str(mat_type),
        "--amgx-profile",
        str(amgx_profile),
        "--ksp-type",
        str(ksp_type),
        "--rtol",
        str(float(rtol)),
        "--atol",
        str(float(atol)),
        "--max-it",
        str(int(max_it)),
    ]
    raw = os.getenv(_ONE_SHOT_COMMAND_ENV, "").strip()
    if raw:
        has_placeholders = "{" in raw and "}" in raw
        formatted = raw.format(
            input_dir=shlex.quote(str(input_dir)),
            output_json=shlex.quote(str(output_json)),
            mat_type=shlex.quote(str(mat_type)),
            amgx_profile=shlex.quote(str(amgx_profile)),
            ksp_type=shlex.quote(str(ksp_type)),
            rtol=float(rtol),
            atol=float(atol),
            max_it=int(max_it),
        )
        command = shlex.split(formatted)
        return command if has_placeholders else [*command, *args]
    return [
        "nix",
        "--option",
        "warn-dirty",
        "false",
        "develop",
        ".#cuda-amgx",
        "--command",
        "python",
        "-m",
        "pyeidors.forward.block_real_amgx",
        *args,
    ]


def _external_worker_command() -> list[str]:
    raw = os.getenv(_WORKER_COMMAND_ENV, "").strip()
    if raw:
        command = shlex.split(raw)
        return [*command, "serve-files"] if "serve-files" not in command else command
    return [
        "nix",
        "--option",
        "warn-dirty",
        "false",
        "develop",
        ".#cuda-amgx",
        "--command",
        "python",
        "-m",
        "pyeidors.forward.block_real_amgx",
        "serve-files",
    ]


def _persistent_worker_enabled() -> bool:
    raw = os.getenv("PYEIDORS_BLOCK_REAL_AMGX_PERSISTENT", "1").strip().lower()
    if raw in {"0", "false", "no", "off", "none", "disabled"}:
        return False
    return not bool(os.getenv(_ONE_SHOT_COMMAND_ENV, "").strip())


def _external_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYEIDORS_ENV_PROFILE",
        "PYEIDORS_PETSC_SCALAR_TYPE",
        "EIT_APP_GUI_RUNTIME_PROFILE",
        "EIT_APP_GUI_PRECISION",
    ):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


class _PersistentBlockRealAmgxWorker:
    def __init__(self, *, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._stdout_tail: deque[str] = deque(maxlen=80)
        self._stderr_thread: threading.Thread | None = None

    def solve(
        self,
        *,
        input_dir: Path,
        output_json: Path,
        mat_type: str,
        amgx_profile: str,
        ksp_type: str,
        rtol: float,
        atol: float,
        max_it: int,
        timeout: float,
    ) -> dict[str, Any]:
        with self._lock:
            if not self._is_running():
                self._start()
            proc = self._proc
            if proc is None:
                raise _BlockRealAmgxWorkerTransportError(
                    "block-real AmgX worker did not start"
                )
            request_id = uuid.uuid4().hex
            payload = {
                "id": request_id,
                "command": "solve-files",
                "input_dir": str(input_dir),
                "output_json": str(output_json),
                "mat_type": str(mat_type),
                "amgx_profile": str(amgx_profile),
                "ksp_type": str(ksp_type),
                "rtol": float(rtol),
                "atol": float(atol),
                "max_it": int(max_it),
            }
            message = self._send_payload(
                proc=proc,
                payload=payload,
                timeout=float(timeout),
            )
            if str(message.get("status", "")) != "ok":
                error = str(message.get("error", "block-real AmgX worker failed"))
                self.request_stop()
                raise RuntimeError(
                    "complex block-real AmgX persistent solve failed: "
                    f"{error}; stderr_tail={self.stderr_tail!r}"
                )
            metadata = message.get("metadata", {})
            return dict(metadata) if isinstance(metadata, dict) else {}

    def request_stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(
                    json.dumps(
                        {"id": uuid.uuid4().hex, "command": "shutdown"},
                        sort_keys=True,
                    )
                    + "\n"
                )
                proc.stdin.flush()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    @property
    def stdout_tail(self) -> str:
        return "\n".join(self._stdout_tail)

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self) -> None:
        cmd = _external_worker_command()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                env=_external_worker_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise _BlockRealAmgxWorkerTransportError(
                f"failed to start block-real AmgX worker: {exc}"
            ) from exc
        self._proc = proc
        self._stderr_tail.clear()
        self._stdout_tail.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(proc,),
            name="pyeidors-block-real-amgx-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            for raw in stream:
                line = raw.rstrip()
                if line:
                    self._stderr_tail.append(line)
        except ValueError:
            return

    def _send_payload(
        self,
        *,
        proc: subprocess.Popen[str],
        payload: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        request_id = str(payload.get("id", ""))
        if proc.stdin is None or proc.stdout is None:
            self.request_stop()
            raise _BlockRealAmgxWorkerTransportError(
                "block-real AmgX worker pipes are closed"
            )
        try:
            proc.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
            proc.stdin.flush()
        except OSError as exc:
            self.request_stop()
            raise _BlockRealAmgxWorkerTransportError(
                f"failed to write block-real AmgX worker request: {exc}"
            ) from exc

        deadline = time.monotonic() + float(timeout)
        while True:
            if timeout > 0 and time.monotonic() > deadline:
                self.request_stop()
                raise TimeoutError(
                    f"block-real AmgX worker request timed out after {timeout}s"
                )
            if timeout > 0:
                remaining = max(deadline - time.monotonic(), 0.0)
                if remaining <= 0:
                    self.request_stop()
                    raise TimeoutError(
                        f"block-real AmgX worker request timed out after {timeout}s"
                    )
                ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.25))
                if not ready:
                    continue
            line = proc.stdout.readline()
            if line == "":
                code = proc.poll()
                self.request_stop()
                raise _BlockRealAmgxWorkerTransportError(
                    "block-real AmgX worker exited before replying "
                    f"(code={code}): {self.stderr_tail}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._stdout_tail.append(line)
                continue
            if str(message.get("id", "")) != request_id:
                continue
            if str(message.get("type", "")) == "done":
                return message


_WORKER_LOCK = threading.RLock()
_WORKERS: dict[str, _PersistentBlockRealAmgxWorker] = {}


def _persistent_worker(repo_root: Path) -> _PersistentBlockRealAmgxWorker:
    key = str(Path(repo_root).resolve())
    with _WORKER_LOCK:
        worker = _WORKERS.get(key)
        if worker is None:
            worker = _PersistentBlockRealAmgxWorker(repo_root=Path(repo_root))
            _WORKERS[key] = worker
        return worker


def _shutdown_persistent_workers() -> None:
    with _WORKER_LOCK:
        workers = list(_WORKERS.values())
        _WORKERS.clear()
    for worker in workers:
        worker.request_stop()


atexit.register(_shutdown_persistent_workers)


def solve_complex_system_with_external_block_real_amgx(
    matrix: sparse.spmatrix,
    rhs: np.ndarray,
    *,
    potential_dofs: int,
    n_elec: int,
    gauge: str = "",
    mat_type: str = "aijcusparse",
    amgx_profile: str = "real_jacobi_l1",
    rtol: float = BLOCK_REAL_AMGX_DEFAULT_RTOL,
    atol: float = BLOCK_REAL_AMGX_DEFAULT_ATOL,
    max_it: int = BLOCK_REAL_AMGX_DEFAULT_MAX_IT,
    ksp_type: str = BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE,
    repo_root: str | Path | None = None,
    timeout_seconds: float | None = None,
    max_relative_residual: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a complex system through an external real PETSc/AmgX process."""
    if not petsc_scalar_is_complex():
        block_matrix = complex_csr_to_block_real(matrix)
        block_rhs = complex_rhs_to_block_real(rhs)
        block_solution, diagnostics = solve_block_real_with_petsc_amgx(
            block_matrix,
            block_rhs,
            mat_type=mat_type,
            amgx_profile=amgx_profile,
            ksp_type=ksp_type,
            rtol=rtol,
            atol=atol,
            max_it=max_it,
        )
        return (
            block_real_solution_to_complex(
                block_solution, original_size=int(matrix.shape[0])
            ),
            {"solver": diagnostics, "route": "complex_block_real_cuda_amgx"},
        )

    run_dir = (
        pyeidors_runtime_root()
        / "block_real_amgx"
        / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_json = run_dir / "block_real_amgx.json"
    matrix_csr = sparse.csr_matrix(matrix)
    rhs_array = np.asarray(rhs)
    if rhs_array.ndim == 1:
        rhs_array = rhs_array.reshape(-1, 1)
    sparse.save_npz(input_dir / "system_matrix_complex.npz", matrix_csr)
    _write_array_artifact(input_dir / "rhs_complex.h5", "rhs", rhs_array)
    _write_json(
        input_dir / "metadata.json",
        {
            "schema_version": 1,
            "petsc_scalar_type": petsc_scalar_dtype_name(),
            "potential_dofs": int(potential_dofs),
            "n_elec": int(n_elec),
            "gauge": str(gauge),
            "rhs_shape": list(rhs_array.shape),
            "matrix_shape": list(matrix_csr.shape),
        },
    )

    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(os.getenv("PYEIDORS_BLOCK_REAL_AMGX_TIMEOUT_SECONDS", "900"))
    )
    residual_limit = (
        float(max_relative_residual)
        if max_relative_residual is not None
        else float(os.getenv("PYEIDORS_BLOCK_REAL_AMGX_MAX_RELRES", "1e-6"))
    )
    repo_path = _find_repo_root(None if repo_root is None else str(repo_root))
    cmd: list[str] = []
    stdout_tail = ""
    stderr_tail = ""
    worker_metadata: dict[str, Any] = {}
    used_persistent_worker = False
    transport_error = ""
    if _persistent_worker_enabled():
        try:
            worker = _persistent_worker(repo_path)
            worker_metadata = worker.solve(
                input_dir=input_dir,
                output_json=output_json,
                mat_type=mat_type,
                amgx_profile=amgx_profile,
                ksp_type=ksp_type,
                rtol=rtol,
                atol=atol,
                max_it=max_it,
                timeout=timeout,
            )
            used_persistent_worker = True
            cmd = _external_worker_command()
            stdout_tail = worker.stdout_tail[-2000:]
            stderr_tail = worker.stderr_tail[-2000:]
        except _BlockRealAmgxWorkerTransportError as exc:
            transport_error = str(exc)
    if not used_persistent_worker:
        cmd = _external_solver_command(
            input_dir=input_dir,
            output_json=output_json,
            mat_type=mat_type,
            amgx_profile=amgx_profile,
            ksp_type=ksp_type,
            rtol=rtol,
            atol=atol,
            max_it=max_it,
        )
        completed = subprocess.run(
            cmd,
            cwd=str(repo_path),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stdout_tail = completed.stdout[-2000:]
        stderr_tail = completed.stderr[-4000:]
        if completed.returncode != 0:
            raise RuntimeError(
                "complex block-real AmgX external solve failed with exit code "
                f"{completed.returncode}. stderr={stderr_tail!r} "
                f"stdout={stdout_tail!r}"
            )
    report = json.loads(output_json.read_text(encoding="utf-8"))
    solution = _read_array_artifact(
        Path(str(report["candidate_solution_path"])), "solution"
    )
    residual = float(
        report.get("complex_true_residual", {}).get(
            "relative_max",
            report.get("solver", {}).get("true_relative_residual_max", 0.0),
        )
        or 0.0
    )
    if residual > residual_limit:
        raise RuntimeError(
            "complex block-real AmgX residual check failed: "
            f"{residual:.3e} > {residual_limit:.3e}"
        )
    report["run_dir"] = str(run_dir)
    report["command"] = cmd
    report["external_worker_persistent"] = bool(used_persistent_worker)
    report["external_worker_metadata"] = worker_metadata
    report["external_worker_transport_error"] = transport_error
    report["stdout_tail"] = stdout_tail[-2000:]
    report["stderr_tail"] = stderr_tail[-2000:]
    return solution, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve = subparsers.add_parser("solve-files")
    solve.add_argument("--input-dir", type=Path, required=True)
    solve.add_argument("--output-json", type=Path, required=True)
    solve.add_argument("--mat-type", default="aijcusparse")
    solve.add_argument(
        "--amgx-profile",
        choices=["block_jacobi", "real_jacobi_l1"],
        default="real_jacobi_l1",
    )
    solve.add_argument("--ksp-type", default=BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE)
    solve.add_argument("--rtol", type=float, default=BLOCK_REAL_AMGX_DEFAULT_RTOL)
    solve.add_argument("--atol", type=float, default=BLOCK_REAL_AMGX_DEFAULT_ATOL)
    solve.add_argument("--max-it", type=int, default=BLOCK_REAL_AMGX_DEFAULT_MAX_IT)
    subparsers.add_parser(
        "serve-files",
        help="Run a persistent JSON-lines block-real AmgX file solver.",
    )
    return parser.parse_args()


def _serve_files() -> int:
    protocol_out = sys.stdout

    def send(payload: dict[str, object]) -> None:
        print(json.dumps(payload, sort_keys=True), file=protocol_out, flush=True)

    for raw in sys.stdin:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            send({"id": "", "type": "done", "status": "error", "error": str(exc)})
            return 1
        request_id = str(message.get("id", ""))
        command = str(message.get("command", ""))
        if command == "shutdown":
            send({"id": request_id, "type": "done", "status": "ok"})
            return 0
        if command != "solve-files":
            send(
                {
                    "id": request_id,
                    "type": "done",
                    "status": "error",
                    "error": f"unknown block-real AmgX worker command: {command!r}",
                }
            )
            return 1
        try:
            with contextlib.redirect_stdout(sys.stderr):
                report = solve_problem_files(
                    input_dir=Path(str(message["input_dir"])),
                    output_json=Path(str(message["output_json"])),
                    mat_type=str(message.get("mat_type", "aijcusparse")),
                    amgx_profile=str(message.get("amgx_profile", "real_jacobi_l1")),
                    ksp_type=str(
                        message.get("ksp_type", BLOCK_REAL_AMGX_DEFAULT_KSP_TYPE)
                    ),
                    rtol=float(message.get("rtol", BLOCK_REAL_AMGX_DEFAULT_RTOL)),
                    atol=float(message.get("atol", BLOCK_REAL_AMGX_DEFAULT_ATOL)),
                    max_it=int(message.get("max_it", BLOCK_REAL_AMGX_DEFAULT_MAX_IT)),
                )
        except Exception as exc:
            send(
                {
                    "id": request_id,
                    "type": "done",
                    "status": "error",
                    "error": str(exc),
                }
            )
            return 1
        solver = report.get("solver", {}) if isinstance(report, dict) else {}
        send(
            {
                "id": request_id,
                "type": "done",
                "status": "ok",
                "metadata": {
                    "route": report.get("route", ""),
                    "output_json": str(message["output_json"]),
                    "ksp_type": solver.get("ksp_type", ""),
                    "iterations_per_rhs": solver.get("iterations_per_rhs", []),
                    "true_relative_residual_max": solver.get(
                        "true_relative_residual_max", 0.0
                    ),
                },
            }
        )
    return 0


def main() -> None:
    args = _parse_args()
    if args.command == "solve-files":
        report = solve_problem_files(
            input_dir=args.input_dir,
            output_json=args.output_json,
            mat_type=args.mat_type,
            amgx_profile=args.amgx_profile,
            ksp_type=args.ksp_type,
            rtol=args.rtol,
            atol=args.atol,
            max_it=args.max_it,
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    elif args.command == "serve-files":
        raise SystemExit(_serve_files())


if __name__ == "__main__":
    main()
