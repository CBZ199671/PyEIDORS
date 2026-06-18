#!/usr/bin/env python3
"""Probe complex CEM solves through a real block-real PETSc/AmgX system.

Workflow:

1. Run ``export-system`` in a complex PETSc runtime, for example
   ``nix develop .#complex64-cuda --command python ... export-system``.
2. Run ``solve-block-real`` in a real PETSc+AmgX runtime, for example
   ``nix develop .#cuda-amgx --command python ... solve-block-real``.

This keeps native-complex PETSc and real-scalar PCAMGX in separate processes,
which is required for a genuine block-real AmgX comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dolfinx import fem
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyeidors.data.structures import PatternConfig
from pyeidors.forward import (
    petsc_scalar_dtype,
    petsc_scalar_dtype_name,
    petsc_scalar_is_complex,
)
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact, write_hdf5_artifact
from pyeidors.runtime_paths import pyeidors_output_path
from pyeidors.utils.block_real import (
    absolute_error_summary,
    block_real_solution_to_complex,
    complex_csr_to_block_real,
    complex_rhs_to_block_real,
)

ARRAY_ARTIFACT_SCHEMA = "pyeidors-complex-block-real-amgx-probe-array-v1"


def _parse_complex(raw: str) -> complex:
    text = str(raw).strip().replace("i", "j")
    try:
        return complex(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"cannot parse complex value {raw!r}; examples: 1+0.25j, 1e-3+2e-4j"
        ) from exc


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
        schema=ARRAY_ARTIFACT_SCHEMA,
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


def _array_artifact_exists(path: Path) -> bool:
    source = Path(path)
    if source.exists():
        return True
    return (
        source.suffix.lower() in {".h5", ".hdf5"}
        and source.with_suffix(".npy").exists()
    )


def _unlink_array_artifact(path: Path) -> None:
    path.unlink(missing_ok=True)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        path.with_suffix(".npy").unlink(missing_ok=True)


def _make_pattern(n_elec: int) -> PatternConfig:
    return PatternConfig(
        n_elec=int(n_elec),
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )


def _apply_reference_gauge_to_csr(model: EITForwardModel, matrix: sparse.csr_matrix):
    gauge = matrix.tolil()
    constraint_row, reference_col = model._cuda_gauge_rows()
    gauge[constraint_row, :] = model._scalar_value(0.0)
    gauge[constraint_row, reference_col] = model._scalar_value(1.0)
    return gauge.tocsr()


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


def _full_rhs_for_model(
    model: EITForwardModel, pattern_matrix: np.ndarray
) -> np.ndarray:
    full_size = model.dofs + model.n_elec + 1
    rhs = np.zeros((full_size, pattern_matrix.shape[0]), dtype=petsc_scalar_dtype())
    rhs[model.dofs : model.dofs + model.n_elec, :] = pattern_matrix.T
    return model._apply_cuda_gauge_fix_rhs(rhs)


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


def _command_export_system(args: argparse.Namespace) -> None:
    if not petsc_scalar_is_complex():
        raise RuntimeError(
            "export-system requires a complex PETSc runtime; use complex64-cuda, "
            "complex-cuda, complex64, or complex."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = (
        args.mesh_dir.resolve() if args.mesh_dir is not None else output_dir / "mesh"
    )
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_or_create_mesh(
        mesh_dir=str(mesh_dir),
        mesh_name=None,
        n_elec=int(args.n_elec),
        dimension=3,
        radius=float(args.radius),
        refinement=int(args.refinement),
        height=float(args.height),
        electrode_height_ratio=0.2,
        z_center=0.0,
        electrode_coverage=0.5,
        mesh_family="tetra",
    )
    dtype = petsc_scalar_dtype()
    model = EITForwardModel(
        n_elec=int(args.n_elec),
        pattern_config=_make_pattern(int(args.n_elec)),
        z=np.full(int(args.n_elec), complex(args.contact_impedance), dtype=dtype),
        mesh=mesh,
        linear_backend="petsc",
        backend_config={
            "petsc_device": str(args.petsc_device),
            "solver_preset": str(args.reference_solver_preset),
            "mat_solve_mode": "off",
            "rtol": float(args.rtol),
            "atol": float(args.atol),
            "max_it": int(args.max_it),
        },
        forward_backend="dolfinx",
    )
    sigma = fem.Function(model.V_sigma)
    sigma.x.array[:] = np.full(
        sigma.x.array.shape, complex(args.background), dtype=dtype
    )

    pattern_matrix = model._resolve_pattern_matrix()
    matrix = model._create_full_matrix_scipy(sigma).tocsr()
    if model._gpu_gauge_fix_enabled():
        matrix = _apply_reference_gauge_to_csr(model, matrix)
    rhs = _full_rhs_for_model(model, pattern_matrix)

    cpu_direct_mode = str(args.cpu_direct_mode)
    direct_solution = None
    direct_electrode_solution = None
    direct_seconds = None
    if cpu_direct_mode in {"sparse", "sparse-and-dense"}:
        direct_t0 = time.perf_counter()
        direct_lu = sparse_linalg.splu(matrix.astype(np.complex128).tocsc())
        direct_solution = np.column_stack(
            [
                direct_lu.solve(np.asarray(rhs[:, idx], dtype=np.complex128))
                for idx in range(rhs.shape[1])
            ]
        )
        if model._gpu_gauge_fix_enabled():
            direct_solution = _recenter_reference_electrode_gauge(
                direct_solution,
                potential_dofs=int(model.dofs),
                n_elec=int(model.n_elec),
            )
        direct_seconds = float(time.perf_counter() - direct_t0)
        direct_electrode_solution = direct_solution[
            model.dofs : model.dofs + model.n_elec, :
        ].T

    dense_solution = None
    dense_electrode_solution = None
    dense_seconds = None
    dense_vs_sparse = None
    dense_electrode_vs_sparse = None
    dense_direct_max_dofs = int(args.dense_direct_max_dofs)
    dense_skip_reason = ""
    if cpu_direct_mode != "sparse-and-dense":
        dense_skip_reason = f"cpu_direct_mode={cpu_direct_mode}"
    elif direct_solution is None or direct_electrode_solution is None:
        dense_skip_reason = "sparse direct reference unavailable"
    elif dense_direct_max_dofs <= 0:
        dense_skip_reason = "dense_direct_max_dofs disabled"
    elif matrix.shape[0] > dense_direct_max_dofs:
        dense_skip_reason = (
            f"n_dofs {matrix.shape[0]} exceeds dense_direct_max_dofs "
            f"{dense_direct_max_dofs}"
        )
    if (
        cpu_direct_mode == "sparse-and-dense"
        and direct_solution is not None
        and direct_electrode_solution is not None
        and dense_direct_max_dofs > 0
        and matrix.shape[0] <= dense_direct_max_dofs
    ):
        dense_t0 = time.perf_counter()
        dense_solution = np.linalg.solve(
            matrix.toarray().astype(np.complex128),
            np.asarray(rhs, dtype=np.complex128),
        )
        if model._gpu_gauge_fix_enabled():
            dense_solution = _recenter_reference_electrode_gauge(
                dense_solution,
                potential_dofs=int(model.dofs),
                n_elec=int(model.n_elec),
            )
        dense_seconds = float(time.perf_counter() - dense_t0)
        dense_electrode_solution = dense_solution[
            model.dofs : model.dofs + model.n_elec, :
        ].T
        dense_vs_sparse = absolute_error_summary(direct_solution, dense_solution)
        dense_electrode_vs_sparse = absolute_error_summary(
            direct_electrode_solution,
            dense_electrode_solution,
        )

    solve_t0 = time.perf_counter()
    runtime_solution = model.solve_full_rhs(
        sigma,
        rhs,
        rhs_kind="complex_block_real_export_reference",
    )
    solve_seconds = float(time.perf_counter() - solve_t0)
    runtime_solution = np.asarray(runtime_solution, dtype=dtype)
    runtime_electrode_solution = runtime_solution[
        model.dofs : model.dofs + model.n_elec, :
    ].T
    runtime_true_residual = _complex_residual_summary(matrix, runtime_solution, rhs)
    runtime_vs_direct = (
        absolute_error_summary(direct_solution, runtime_solution)
        if direct_solution is not None
        else {}
    )
    runtime_electrode_vs_direct = (
        absolute_error_summary(direct_electrode_solution, runtime_electrode_solution)
        if direct_electrode_solution is not None
        else {}
    )

    sparse.save_npz(output_dir / "system_matrix_complex.npz", matrix)
    _write_array_artifact(output_dir / "rhs_complex.h5", "rhs", rhs)
    stale_direct_paths = [
        output_dir / "direct_reference_solution_complex.h5",
        output_dir / "direct_reference_electrode_voltages_complex.h5",
    ]
    stale_dense_paths = [
        output_dir / "dense_direct_solution_complex.h5",
        output_dir / "dense_direct_electrode_voltages_complex.h5",
    ]
    if direct_solution is None:
        for stale_path in stale_direct_paths:
            _unlink_array_artifact(stale_path)
    if dense_solution is None:
        for stale_path in stale_dense_paths:
            _unlink_array_artifact(stale_path)
    if direct_solution is not None and direct_electrode_solution is not None:
        _write_array_artifact(
            output_dir / "direct_reference_solution_complex.h5",
            "solution",
            direct_solution,
        )
        _write_array_artifact(
            output_dir / "direct_reference_electrode_voltages_complex.h5",
            "electrode_voltages",
            direct_electrode_solution,
        )
    if dense_solution is not None and dense_electrode_solution is not None:
        _write_array_artifact(
            output_dir / "dense_direct_solution_complex.h5",
            "solution",
            dense_solution,
        )
        _write_array_artifact(
            output_dir / "dense_direct_electrode_voltages_complex.h5",
            "electrode_voltages",
            dense_electrode_solution,
        )
    _write_array_artifact(
        output_dir / "reference_solution_complex.h5",
        "solution",
        runtime_solution,
    )
    _write_array_artifact(
        output_dir / "reference_electrode_voltages_complex.h5",
        "electrode_voltages",
        runtime_electrode_solution,
    )

    metadata = {
        "schema_version": 1,
        "route": "complex_system_export",
        "reference_route": str(args.reference_solver_preset),
        "petsc_scalar_type": petsc_scalar_dtype_name(),
        "petsc_device": str(args.petsc_device),
        "gauge": (
            "reference-electrode-row"
            if model._gpu_gauge_fix_enabled()
            else "sum-electrode-lagrange"
        ),
        "n_elec": int(args.n_elec),
        "n_patterns": int(pattern_matrix.shape[0]),
        "n_dofs": int(model.dofs + model.n_elec + 1),
        "potential_dofs": int(model.dofs),
        "mesh": {
            "nodes": int(mesh.num_vertices()),
            "elements": int(mesh.num_cells()),
            "family": getattr(mesh, "mesh_family", "tetra"),
            "radius": float(args.radius),
            "height": float(args.height),
            "refinement": int(args.refinement),
        },
        "background": complex(args.background),
        "contact_impedance": complex(args.contact_impedance),
        "matrix": {
            "path": "system_matrix_complex.npz",
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
            "dtype": str(matrix.dtype),
        },
        "rhs": {
            "path": "rhs_complex.h5",
            "shape": list(rhs.shape),
            "dtype": str(rhs.dtype),
        },
        "reference_solution": (
            {
                "path": "direct_reference_solution_complex.h5",
                "kind": "scipy_splu_direct",
                "shape": list(direct_solution.shape),
                "dtype": str(direct_solution.dtype),
                "solve_seconds": direct_seconds,
            }
            if direct_solution is not None
            else {
                "path": "reference_solution_complex.h5",
                "kind": str(args.reference_solver_preset),
                "shape": list(runtime_solution.shape),
                "dtype": str(runtime_solution.dtype),
                "solve_seconds": solve_seconds,
                "cpu_direct_skipped": True,
                "skip_reason": f"cpu_direct_mode={cpu_direct_mode}",
            }
        ),
        "dense_reference_solution": (
            {
                "path": "dense_direct_solution_complex.h5",
                "kind": "numpy_dense_direct",
                "shape": list(dense_solution.shape),
                "dtype": str(dense_solution.dtype),
                "solve_seconds": dense_seconds,
                "max_dofs": dense_direct_max_dofs,
            }
            if dense_solution is not None
            else {
                "kind": "numpy_dense_direct",
                "skipped": True,
                "reason": dense_skip_reason,
                "max_dofs": dense_direct_max_dofs,
            }
        ),
        "dense_direct_vs_sparse_direct": dense_vs_sparse or {},
        "dense_direct_electrode_voltage_vs_sparse_direct": (
            dense_electrode_vs_sparse or {}
        ),
        "runtime_reference_solution": {
            "path": "reference_solution_complex.h5",
            "kind": str(args.reference_solver_preset),
            "shape": list(runtime_solution.shape),
            "dtype": str(runtime_solution.dtype),
            "solve_seconds": solve_seconds,
            "backend_diagnostics": model.get_backend_diagnostics(),
            "true_residual": runtime_true_residual,
        },
        "runtime_true_residual": runtime_true_residual,
        "runtime_vs_direct": runtime_vs_direct,
        "runtime_electrode_voltage_vs_direct": runtime_electrode_vs_direct,
        "reference_electrode_voltages": (
            {
                "path": "direct_reference_electrode_voltages_complex.h5",
                "kind": "scipy_splu_direct",
                "shape": list(direct_electrode_solution.shape),
                "dtype": str(direct_electrode_solution.dtype),
            }
            if direct_electrode_solution is not None
            else {
                "path": "reference_electrode_voltages_complex.h5",
                "kind": str(args.reference_solver_preset),
                "shape": list(runtime_electrode_solution.shape),
                "dtype": str(runtime_electrode_solution.dtype),
                "cpu_direct_skipped": True,
            }
        ),
        "runtime_reference_electrode_voltages": {
            "path": "reference_electrode_voltages_complex.h5",
            "kind": str(args.reference_solver_preset),
            "shape": list(runtime_electrode_solution.shape),
            "dtype": str(runtime_electrode_solution.dtype),
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    print(
        json.dumps({"output_dir": str(output_dir), **metadata}, default=_json_default)
    )


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


def _solve_block_real_with_petsc_amgx(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    mat_type: str,
    amgx_profile: str,
    rtol: float,
    atol: float,
    max_it: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from petsc4py import PETSc

    if petsc_scalar_is_complex():
        raise RuntimeError(
            "solve-block-real requires a real PETSc runtime so PCAMGX sees a real "
            "2x2 block system; use nix develop .#cuda-amgx."
        )

    prefix = "pyeidors_block_real_amgx_"
    A = _petsc_mat_from_csr(matrix, PETSc, mat_type)
    _apply_amgx_options(PETSc, prefix, amgx_profile)
    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(A)
    ksp.setType("fgmres")
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
                f"block-real PCAMGX solve failed for RHS {col} with convergence reason {reason}"
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
        "ksp_type": str(ksp.getType()) if hasattr(ksp, "getType") else "fgmres",
        "pc_type": str(ksp.getPC().getType())
        if hasattr(ksp.getPC(), "getType")
        else "amgx",
        "ksp_norm_type": norm_type,
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


def _command_solve_block_real(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_json = args.output_json.resolve()
    matrix = sparse.load_npz(input_dir / "system_matrix_complex.npz").tocsr()
    rhs = _read_array_artifact(input_dir / "rhs_complex.h5", "rhs")
    reference_path = input_dir / "direct_reference_solution_complex.h5"
    reference_kind = "scipy_splu_direct"
    if not _array_artifact_exists(reference_path):
        reference_path = input_dir / "reference_solution_complex.h5"
        reference_kind = "runtime_reference"
    reference_solution = _read_array_artifact(reference_path, "solution")
    metadata_path = input_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    block_matrix = complex_csr_to_block_real(matrix)
    block_rhs = complex_rhs_to_block_real(rhs)
    block_solution, solver_diagnostics = _solve_block_real_with_petsc_amgx(
        block_matrix,
        block_rhs,
        mat_type=str(args.mat_type),
        amgx_profile=str(args.amgx_profile),
        rtol=float(args.rtol),
        atol=float(args.atol),
        max_it=int(args.max_it),
    )
    candidate_solution = block_real_solution_to_complex(
        block_solution,
        original_size=int(matrix.shape[0]),
    )
    gauge = str(metadata.get("gauge") or "").strip().lower()
    if not gauge and str(metadata.get("petsc_device") or "").strip().lower() == "cuda":
        gauge = "reference-electrode-row"
    if gauge == "reference-electrode-row":
        reference_solution = _recenter_reference_electrode_gauge(
            reference_solution,
            potential_dofs=int(metadata.get("potential_dofs") or 0),
            n_elec=int(metadata.get("n_elec") or 0),
        )
        candidate_solution = _recenter_reference_electrode_gauge(
            candidate_solution,
            potential_dofs=int(metadata.get("potential_dofs") or 0),
            n_elec=int(metadata.get("n_elec") or 0),
        )
    solution_error = absolute_error_summary(reference_solution, candidate_solution)

    potential_dofs = int(metadata.get("potential_dofs") or 0)
    n_elec = int(metadata.get("n_elec") or 0)
    electrode_error: dict[str, float] = {}
    if potential_dofs > 0 and n_elec > 0:
        ref_electrode = reference_solution[
            potential_dofs : potential_dofs + n_elec, :
        ].T
        got_electrode = candidate_solution[
            potential_dofs : potential_dofs + n_elec, :
        ].T
        electrode_error = absolute_error_summary(ref_electrode, got_electrode)

    solution_path = output_json.with_suffix(".solution_complex.h5")
    _write_array_artifact(solution_path, "solution", candidate_solution)
    report = {
        "schema_version": 1,
        "route": "complex_block_real_cuda_amgx",
        "input_dir": str(input_dir),
        "source_reference_route": metadata.get("reference_route"),
        "reference_kind": reference_kind,
        "reference_solution_path": str(reference_path),
        "source_petsc_scalar_type": metadata.get("petsc_scalar_type"),
        "source_gauge": metadata.get("gauge"),
        "block_matrix": {
            "shape": list(block_matrix.shape),
            "nnz": int(block_matrix.nnz),
            "dtype": str(block_matrix.dtype),
        },
        "solver": solver_diagnostics,
        "solution_error_vs_reference": solution_error,
        "electrode_voltage_error_vs_reference": electrode_error,
        "candidate_solution_path": str(solution_path),
        "known_limitations": [
            "This route solves an exported complex CEM linear system in a separate real PETSc process.",
            "It validates block-real PCAMGX quality but does not yet assemble CEM forms directly in a real PETSc runtime.",
        ],
    }
    _write_json(output_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-system")
    export.add_argument(
        "--output-dir",
        type=Path,
        default=pyeidors_output_path("complex_block_real_amgx", "export"),
    )
    export.add_argument("--mesh-dir", type=Path, default=None)
    export.add_argument("--n-elec", type=int, default=16)
    export.add_argument("--radius", type=float, default=0.18)
    export.add_argument("--height", type=float, default=0.16)
    export.add_argument("--refinement", type=int, default=2)
    export.add_argument("--background", type=_parse_complex, default=complex(1.0, 0.25))
    export.add_argument(
        "--contact-impedance",
        type=_parse_complex,
        default=complex(1.0e-3, 2.0e-4),
    )
    export.add_argument(
        "--reference-solver-preset",
        default="3d_gamg",
        help="Native complex reference preset, e.g. 3d_gamg or complex_cuda_amgx.",
    )
    export.add_argument(
        "--petsc-device", choices=["cpu", "cuda", "auto"], default="cuda"
    )
    export.add_argument("--rtol", type=float, default=1.0e-10)
    export.add_argument("--atol", type=float, default=1.0e-12)
    export.add_argument("--max-it", type=int, default=2000)
    export.add_argument(
        "--cpu-direct-mode",
        choices=["sparse-and-dense", "sparse", "none"],
        default="sparse-and-dense",
        help=(
            "Small correctness runs use sparse-and-dense. Large GPU-only scaling "
            "runs can use none to avoid CPU direct factorization."
        ),
    )
    export.add_argument(
        "--dense-direct-max-dofs",
        type=int,
        default=1500,
        help=(
            "Also run NumPy dense direct for small exported systems up to this "
            "DOF count; use 0 to skip."
        ),
    )
    export.set_defaults(func=_command_export_system)

    solve = subparsers.add_parser("solve-block-real")
    solve.add_argument(
        "--input-dir",
        type=Path,
        default=pyeidors_output_path("complex_block_real_amgx", "export"),
    )
    solve.add_argument(
        "--output-json",
        type=Path,
        default=pyeidors_output_path("complex_block_real_amgx", "block_real_amgx.json"),
    )
    solve.add_argument("--mat-type", default="aijcusparse")
    solve.add_argument(
        "--amgx-profile",
        choices=["block_jacobi", "real_jacobi_l1"],
        default="real_jacobi_l1",
    )
    solve.add_argument("--rtol", type=float, default=1.0e-10)
    solve.add_argument("--atol", type=float, default=1.0e-12)
    solve.add_argument("--max-it", type=int, default=4000)
    solve.set_defaults(func=_command_solve_block_real)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
