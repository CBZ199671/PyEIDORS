from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_top_level_pyeidors_facade_stays_narrow() -> None:
    import pyeidors

    assert pyeidors.__all__ == ["EITSystem", "check_environment", "__version__"]
    assert "EITSystem" in dir(pyeidors)
    assert callable(pyeidors.check_environment)

    for name in (
        "EITForwardModel",
        "LinearBackendConfig",
        "GaussNewtonReconstructor",
        "DirectJacobianCalculator",
    ):
        with pytest.raises(AttributeError):
            getattr(pyeidors, name)


def test_top_level_pyeidors_import_stays_lazy_light() -> None:
    script = """
import importlib
import sys

mod = importlib.import_module("pyeidors")
if "dolfinx" in sys.modules:
    raise SystemExit("top-level import loaded dolfinx")
if "torch" in sys.modules:
    raise SystemExit("top-level import loaded torch")
if "cuqi" in sys.modules:
    raise SystemExit("top-level import loaded cuqi")
if "pyeidors.core_system" in sys.modules:
    raise SystemExit("top-level import loaded EITSystem implementation")
if not callable(mod.check_environment):
    raise SystemExit("missing check_environment")
if "_TORCH_AVAILABLE" not in dir(mod):
    raise SystemExit("private env compatibility flag missing from dir()")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_claimed_subpackage_exports_are_declared() -> None:
    forward = importlib.import_module("pyeidors.forward")
    inverse = importlib.import_module("pyeidors.inverse")
    jacobian = importlib.import_module("pyeidors.inverse.jacobian")

    assert {"EITForwardModel", "LinearBackendConfig"}.issubset(forward.__all__)
    assert {
        "GaussNewtonReconstructor",
        "assemble_sigma_contact_normal_system",
        "build_sigma_contact_block_metadata",
        "build_electrode_movement_jacobian",
        "configure_petsc_fieldsplit_solver",
        "prior_movement",
        "solve_sigma_contact_fieldsplit",
    }.issubset(inverse.__all__)
    assert {
        "DirectJacobianCalculator",
        "JacobianLinearization",
        "compute_sigma_fingerprint",
    }.issubset(jacobian.__all__)


def test_forward_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

forward = importlib.import_module("pyeidors.forward")
required = {
    "CudaStructuredForwardBackend",
    "EITForwardModel",
    "LinearBackendConfig",
    "petsc_scalar_dtype",
    "petsc_scalar_is_complex",
}
missing = required.difference(forward.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "dolfinx",
        "petsc4py",
        "torch",
        "meshio",
        "mpi4py",
        "pyeidors.forward.complex_support",
        "pyeidors.forward.cuda_structured_backend",
        "pyeidors.forward.eit_forward_model",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager forward imports: {heavy_loaded}")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_forward_scalar_support_module_delays_petsc_import() -> None:
    script = """
import importlib
import numpy as np
import sys
from types import SimpleNamespace

mod = importlib.import_module("pyeidors.forward.complex_support")
if "petsc4py" in sys.modules:
    raise SystemExit("complex_support import loaded petsc4py")
if not callable(mod.petsc_scalar_dtype):
    raise SystemExit("missing scalar dtype helper")
mod.PETSc = SimpleNamespace(ScalarType=np.complex64)
if mod.petsc_scalar_dtype() != np.dtype(np.complex64):
    raise SystemExit("monkeypatched PETSc scalar dtype was not honored")
if "petsc4py" in sys.modules:
    raise SystemExit("fake PETSc scalar query loaded petsc4py")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_data_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

data = importlib.import_module("pyeidors.data")
required = {"PatternConfig", "EITData", "MeasurementDataset", "run_factor_sweep"}
missing = required.difference(data.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "pyeidors.data.eit_digit_metrics",
        "pyeidors.data.factor_sweep",
        "pyeidors.data.measurement_dataset",
        "pyeidors.data.synthetic_data",
        "pyeidors.data.voltage_digit_sweep",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager data imports: {heavy_loaded}")

from pyeidors.data import PatternConfig

if PatternConfig.__name__ != "PatternConfig":
    raise SystemExit("lazy core data export failed")
if "pyeidors.data.structures" not in sys.modules:
    raise SystemExit("lazy core data export did not import structures")
if "pyeidors.data.factor_sweep" in sys.modules:
    raise SystemExit("core data export loaded factor_sweep")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_perf_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

perf = importlib.import_module("pyeidors.perf")
required = {"DEFAULT_ACCELERATION_PROFILE", "RMMatmulHandle", "rm_matmul"}
missing = required.difference(perf.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {"scipy", "pyeidors.perf.gpu_kernels"}
]
if heavy_loaded:
    raise SystemExit(f"eager perf imports: {heavy_loaded}")

from pyeidors.perf import DEFAULT_ACCELERATION_PROFILE

if not DEFAULT_ACCELERATION_PROFILE:
    raise SystemExit("lazy policy export failed")
if "pyeidors.perf.policy" not in sys.modules:
    raise SystemExit("lazy policy export did not import policy")
if "pyeidors.perf.gpu_kernels" in sys.modules:
    raise SystemExit("policy export loaded gpu_kernels")

from pyeidors.perf import capabilities

if capabilities.__name__ != "pyeidors.perf.capabilities":
    raise SystemExit("submodule import failed")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_visualization_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

viz = importlib.import_module("pyeidors.visualization")
if {"EITVisualizer", "create_visualizer"}.difference(viz.__all__):
    raise SystemExit("missing visualization exports")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "dolfinx",
        "matplotlib",
        "mpi4py",
        "ufl",
        "pyeidors.visualization.eit_plots",
        "pyeidors.visualization.eit_plot_helpers",
        "pyeidors.visualization.eit_plot_renderers",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager visualization imports: {heavy_loaded}")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_io_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

io = importlib.import_module("pyeidors.io")
required = {"HDF5Artifact", "read_hdf5_artifact", "write_hdf5_artifact"}
missing = required.difference(io.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {"h5py", "pyeidors.io.hdf5_artifacts"}
]
if heavy_loaded:
    raise SystemExit(f"eager io imports: {heavy_loaded}")

from pyeidors.io import hdf5_artifacts

if hdf5_artifacts.__name__ != "pyeidors.io.hdf5_artifacts":
    raise SystemExit("io submodule import failed")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_femx_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

femx = importlib.import_module("pyeidors.femx")
required = {"build_eit_mesh", "function_get_array", "mesh_coordinates"}
missing = required.difference(femx.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {"dolfinx", "ufl", "pyeidors.femx.helpers"}
]
if heavy_loaded:
    raise SystemExit(f"eager femx imports: {heavy_loaded}")
if "helpers" not in dir(femx):
    raise SystemExit("femx helpers submodule missing from dir")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_interop_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

interop = importlib.import_module("pyeidors.interop")
required = {"STANDARD_INTEROP_FORMAT", "save_exchange_mat", "load_forward_csv"}
missing = required.difference(interop.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "dolfinx",
        "scipy",
        "pyeidors.femx.helpers",
        "pyeidors.interop.geometry_exchange",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager interop imports: {heavy_loaded}")
if "geometry_exchange" not in dir(interop):
    raise SystemExit("interop geometry_exchange submodule missing from dir")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cache_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

cache = importlib.import_module("pyeidors.cache")
required = {"CacheManager", "CachePolicy", "build_cache_key", "stable_signature_hash"}
missing = required.difference(cache.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "numpy",
        "pyeidors.cache.keys",
        "pyeidors.cache.manager",
        "pyeidors.cache.object_signature",
        "pyeidors.cache.store_disk",
        "pyeidors.cache.store_process",
        "pyeidors.cache.types",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager cache imports: {heavy_loaded}")
if "manager" not in dir(cache):
    raise SystemExit("cache manager submodule missing from dir")

from pyeidors.cache import CacheManager

if CacheManager.__name__ != "CacheManager":
    raise SystemExit("lazy cache export failed")
if "pyeidors.cache.manager" not in sys.modules:
    raise SystemExit("lazy cache export did not import manager")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_physics_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

physics = importlib.import_module("pyeidors.physics")
required = {"UnitCheckLevel", "build_stim_currents", "run_unit_consistency_checks"}
missing = required.difference(physics.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "numpy",
        "pyeidors.physics.current_drive",
        "pyeidors.physics.unit_consistency",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager physics imports: {heavy_loaded}")
if "unit_consistency" not in dir(physics):
    raise SystemExit("physics unit_consistency submodule missing from dir")

from pyeidors.physics import UnitCheckLevel

if UnitCheckLevel.__name__ != "UnitCheckLevel":
    raise SystemExit("lazy physics export failed")
if "pyeidors.physics.unit_consistency" not in sys.modules:
    raise SystemExit("lazy physics export did not import unit_consistency")
if "pyeidors.physics.current_drive" in sys.modules:
    raise SystemExit("unit consistency export loaded current_drive")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_electrodes_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

electrodes = importlib.import_module("pyeidors.electrodes")
if "StimMeasPatternManager" not in electrodes.__all__:
    raise SystemExit("missing electrodes export")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "numpy",
        "pyeidors.electrodes.patterns",
        "pyeidors.physics.current_drive",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager electrodes imports: {heavy_loaded}")
if "layout" not in dir(electrodes):
    raise SystemExit("electrodes layout submodule missing from dir")

from pyeidors.electrodes import layout

if layout.__name__ != "pyeidors.electrodes.layout":
    raise SystemExit("electrodes layout submodule import failed")
if "pyeidors.electrodes.patterns" in sys.modules:
    raise SystemExit("layout submodule loaded pattern manager")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

inverse = importlib.import_module("pyeidors.inverse")
jacobian = importlib.import_module("pyeidors.inverse.jacobian")
required = {"GaussNewtonReconstructor", "VoxelGrid", "calc_greit_rm"}
missing = required.difference(inverse.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")
if "DirectJacobianCalculator" not in jacobian.__all__:
    raise SystemExit("missing jacobian export")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "pyeidors.inverse.solvers.gauss_newton",
        "pyeidors.inverse.greit",
        "pyeidors.inverse.workflows",
        "pyeidors.inverse.jacobian.adjoint_jacobian",
        "pyeidors.inverse.jacobian.direct_jacobian",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager inverse imports: {heavy_loaded}")

from pyeidors.inverse import VoxelGrid
from pyeidors.inverse.jacobian import compute_sigma_fingerprint

if VoxelGrid.__name__ != "VoxelGrid":
    raise SystemExit("lazy export resolved wrong symbol")
if not compute_sigma_fingerprint([1.0, 2.0]):
    raise SystemExit("lazy jacobian export failed")
if "pyeidors.inverse.dual_mesh" not in sys.modules:
    raise SystemExit("lazy export did not import source module")
if "pyeidors.inverse.solvers.gauss_newton" in sys.modules:
    raise SystemExit("lightweight export loaded GN solver")
if "pyeidors.inverse.jacobian.direct_jacobian" in sys.modules:
    raise SystemExit("lightweight jacobian export loaded direct calculator")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_solvers_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

solvers = importlib.import_module("pyeidors.inverse.solvers")
required = {
    "GaussNewtonReconstructor",
    "MatrixFreeGNStepResult",
    "SparseBayesianReconstructor",
}
missing = required.difference(solvers.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "scipy",
        "pyeidors.inverse.solvers.gauss_newton",
        "pyeidors.inverse.solvers.gauss_newton_engine",
        "pyeidors.inverse.solvers.matrix_free_gn",
        "pyeidors.inverse.solvers.sparse_bayesian_engine",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager solver imports: {heavy_loaded}")

from pyeidors.inverse.solvers import sparse_projection

if sparse_projection.__name__ != "pyeidors.inverse.solvers.sparse_projection":
    raise SystemExit("solver submodule import failed")
if "pyeidors.inverse.solvers.gauss_newton_engine" in sys.modules:
    raise SystemExit("light solver submodule import loaded GN engine")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_regularization_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

regularization = importlib.import_module("pyeidors.inverse.regularization")
required = {
    "BaseRegularization",
    "CurvatureRegularization",
    "SmoothnessRegularization",
    "TikhonovRegularization",
    "TotalVariationRegularization",
}
missing = required.difference(regularization.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "dolfinx",
        "scipy",
        "pyeidors.inverse.jacobian.direct_jacobian",
        "pyeidors.inverse.regularization.base_regularization",
        "pyeidors.inverse.regularization.smoothness",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager regularization imports: {heavy_loaded}")

from pyeidors.inverse.regularization import base_regularization

if base_regularization.__name__ != "pyeidors.inverse.regularization.base_regularization":
    raise SystemExit("regularization submodule import failed")
if "pyeidors.inverse.regularization.smoothness" in sys.modules:
    raise SystemExit("base regularization submodule loaded smoothness")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_prior_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

prior = importlib.import_module("pyeidors.inverse.prior")
required = {
    "RtRPrior",
    "TVIRLSResult",
    "as_rtr_prior",
    "graph_laplacian",
    "solve_tv_irls_batch",
}
missing = required.difference(prior.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "scipy",
        "pyeidors.inverse.prior.laplace",
        "pyeidors.inverse.prior.rtr",
        "pyeidors.inverse.prior.tv_irls",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager prior imports: {heavy_loaded}")

from pyeidors.inverse.prior import laplace

if laplace.__name__ != "pyeidors.inverse.prior.laplace":
    raise SystemExit("prior submodule import failed")
if "pyeidors.inverse.prior.tv_irls" in sys.modules:
    raise SystemExit("laplace submodule import loaded tv_irls")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_postprocess_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

postprocess = importlib.import_module("pyeidors.inverse.postprocess")
required = {
    "TemporalTVPipelineResult",
    "TVRefinementResult",
    "moving_average_frames",
    "refine_tv_pdhg",
}
missing = required.difference(postprocess.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "scipy",
        "pyeidors.inverse.postprocess.temporal",
        "pyeidors.inverse.postprocess.tv",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager postprocess imports: {heavy_loaded}")

from pyeidors.inverse.postprocess import temporal

if temporal.__name__ != "pyeidors.inverse.postprocess.temporal":
    raise SystemExit("postprocess submodule import failed")
if "pyeidors.inverse.postprocess.tv" in sys.modules:
    raise SystemExit("temporal submodule import loaded tv")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_reduced_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

reduced = importlib.import_module("pyeidors.inverse.reduced")
required = {
    "InexactController",
    "SnapshotBank",
    "build_lowrank_subspace",
    "build_reduced_operator",
    "solve_reduced_step",
}
missing = required.difference(reduced.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "scipy",
        "pyeidors.inverse.reduced.lowrank_subspace",
        "pyeidors.inverse.reduced.reduced_gn_step",
        "pyeidors.inverse.reduced.snapshot_bank",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager reduced imports: {heavy_loaded}")

from pyeidors.inverse.reduced import inexact_controller

if inexact_controller.__name__ != "pyeidors.inverse.reduced.inexact_controller":
    raise SystemExit("reduced submodule import failed")
if "pyeidors.inverse.reduced.reduced_gn_step" in sys.modules:
    raise SystemExit("inexact controller import loaded reduced_gn_step")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_matrix_free_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

matrix_free = importlib.import_module("pyeidors.inverse.matrix_free")
if "DualMeshJacobianOperator" not in matrix_free.__all__:
    raise SystemExit("missing matrix-free export")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {"scipy", "pyeidors.inverse.matrix_free.dual_mesh"}
]
if heavy_loaded:
    raise SystemExit(f"eager matrix-free imports: {heavy_loaded}")

from pyeidors.inverse.matrix_free import dual_mesh

if dual_mesh.__name__ != "pyeidors.inverse.matrix_free.dual_mesh":
    raise SystemExit("matrix-free submodule import failed")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_inverse_workflows_package_exports_stay_lazy_until_accessed() -> None:
    script = """
import importlib
import sys

workflows = importlib.import_module("pyeidors.inverse.workflows")
required = {
    "ReconstructionResult",
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
}
missing = required.difference(workflows.__all__)
if missing:
    raise SystemExit(f"missing exports: {sorted(missing)}")

heavy_loaded = [
    name
    for name in sys.modules
    if name in {
        "pyeidors.inverse.workflows.absolute",
        "pyeidors.inverse.workflows.difference",
        "pyeidors.inverse.workflows.sparse_bayesian",
        "pyeidors.inverse.solvers.sparse_bayesian_engine",
    }
]
if heavy_loaded:
    raise SystemExit(f"eager workflow imports: {heavy_loaded}")

if "base" not in dir(workflows):
    raise SystemExit("workflow submodule name missing from dir")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_repository_root_cmd_wrappers_delegate_to_supported_gui_launcher() -> None:
    wrappers = {
        "EIT-GUI.cmd": "auto",
        "EIT-GUI-CPU.cmd": "cpu",
        "EIT-GUI-GPU.cmd": "gpu",
    }

    for filename, profile in wrappers.items():
        path = REPO_ROOT / filename
        text = path.read_text(encoding="utf-8")

        assert path.exists()
        assert r"%~dp0scripts\gui\run_eit_app.ps1" in text
        assert f"-Profile {profile}" in text
        assert "%*" in text
        assert "exit /b %EXIT_CODE%" in text


def test_repository_root_posix_wrapper_delegates_to_supported_gui_launcher() -> None:
    path = REPO_ROOT / "eit-gui"
    text = path.read_text(encoding="utf-8")

    assert path.exists()
    assert "scripts/gui/run_eit_app.sh" in text
    assert "PROFILE_ARG=(--auto)" in text
    assert "PROFILE_ARG=(--gpu)" in text
    assert "PROFILE_ARG=(--cpu)" in text
    assert "PROFILE_ARG=(--real-gpu)" in text
    assert "PROFILE_ARG=(--complex64-gpu)" in text
    assert 'exec bash "$ROOT_DIR/scripts/gui/run_eit_app.sh"' in text
