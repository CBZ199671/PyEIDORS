"""Shared pytest fixtures for DOLFINx + gmsh integration tests."""

from __future__ import annotations

import gc
import hashlib
import os
import sys
from pathlib import Path

# Pin BLAS/OpenMP pools to a single thread BEFORE NumPy/SciPy load
# OpenBLAS.  The GUI process does the same at startup via
# eit_app.runtime_threads.configure_realtime_thread_env(); without it the
# offscreen GUI tests can deadlock inside OpenBLAS blas_thread_init when
# the Qt main thread and a worker QThread race the pool bring-up
# (observed: pytest hung >15 min in exec_blas/exec_blas_async).
# setdefault keeps any explicitly exported value in charge.
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DEFAULT_SKIP_MARKERS = {
    "slow": "--run-slow",
    "gpu": "--run-gpu",
    "integration": "--run-integration",
    "gui": "--run-gui",
    "hardware": "--run-hardware",
}

GUI_TEST_FILES = {
    "test_conductivity_3d_widget_runtime.py",
    "test_database_backfill_shutdown.py",
    "test_eit_app_full_button_smoke.py",
    "test_eit_app_gui_smoke.py",
    "test_eit_app_interop_environment.py",
    "test_eit_app_interop_hub.py",
    "test_eit_app_measurement_layout.py",
    "test_eit_app_runtime_walk.py",
    "test_runtime_threads.py",
}

HARDWARE_TEST_FILES = {
    "test_eit_app_connection_preflight.py",
    "test_eit_app_protocol_legacy.py",
    "test_eit_app_relay_transport.py",
    "test_eit_app_serial_device.py",
    "test_eit_app_serial_port_discovery.py",
    "test_eit_app_simulator.py",
    "test_eit_app_windows_serial_transport.py",
    "test_frame_io_legacy_compat.py",
}

SLOW_TEST_FILES = {
    "test_bucket_dense_experiments.py",
    "test_dual_model_rm_benchmark.py",
    "test_dynamic_t65_t66_t67_sweep_benchmark.py",
    "test_dynamic_validation_benchmark.py",
    "test_gn_diff_3d_operator_cache.py",
    "test_mesh_generator_lazy_gmsh.py",
    "test_mesh_io_format_benchmark.py",
    "test_prior_travelling_wave_benchmark.py",
    "test_real_mesh_generation.py",
}

_TEST_STACK_IMPORT_ERROR: Exception | None = None
try:
    from dolfinx import mesh as dmesh
    from mpi4py import MPI

    from pyeidors.data.structures import PatternConfig
    from pyeidors.core_system import EITSystem
    from pyeidors.femx import build_eit_mesh
except Exception as exc:  # pragma: no cover - import guard for lean environments
    dmesh = None
    MPI = None
    PatternConfig = None
    EITSystem = None
    build_eit_mesh = None
    _TEST_STACK_IMPORT_ERROR = exc

# Darwin/OpenMP runtime stability guard for mixed PETSc/Torch test runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _populate_lazy_mesh_gmsh_imports() -> None:
    """T91/T92: tests monkeypatch ``module.gmshio`` /
    ``module.build_eit_mesh`` / ``module.estimate_radius`` /
    ``module.fem`` / ``module.ufl`` attributes. Ensure the lazy imports are
    resolved + back-bound on the geometry modules before any test file is
    collected so monkeypatch.setattr targets exist. Production code paths
    still hit the lazy path because the cold-start gate runs in a fresh
    subprocess.
    """
    geometry_modules: list = []
    for module_name in (
        "pyeidors.geometry.optimized_mesh_generator",
        "pyeidors.geometry.mesh3d_generator",
        "pyeidors.geometry.mesh_generator",
        "pyeidors.geometry.mesh_converter",
        "pyeidors.geometry._helpers",
    ):
        try:
            module = __import__(module_name, fromlist=["__name__"])
        except Exception:  # pragma: no cover - import guard
            continue
        geometry_modules.append(module)
        ensure = getattr(module, "_ensure_gmsh", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:  # pragma: no cover - lean env without gmsh
                pass

    try:
        import ufl as _ufl
        from dolfinx import fem as _fem
        from pyeidors.femx import build_eit_mesh as _build, estimate_radius as _estimate
    except Exception:  # pragma: no cover - lean env without dolfinx
        return
    for module in geometry_modules:
        if not hasattr(module, "build_eit_mesh"):
            module.build_eit_mesh = _build  # type: ignore[attr-defined]
        if not hasattr(module, "estimate_radius"):
            module.estimate_radius = _estimate  # type: ignore[attr-defined]
        if not hasattr(module, "fem"):
            module.fem = _fem  # type: ignore[attr-defined]
        if not hasattr(module, "ufl"):
            module.ufl = _ufl  # type: ignore[attr-defined]


_populate_lazy_mesh_gmsh_imports()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("pyeidors")
    group.addoption(
        "--run-slow",
        action="store_true",
        help="Run tests marked slow. Skipped by default to keep pytest under the local quick gate.",
    )
    group.addoption(
        "--run-gpu",
        action="store_true",
        help="Run tests marked gpu. Skipped by default unless a CUDA validation is requested.",
    )
    group.addoption(
        "--run-integration",
        action="store_true",
        help="Run tests under tests/integration. Skipped by default for the quick local suite.",
    )
    group.addoption(
        "--run-gui",
        action="store_true",
        help="Run GUI tests marked gui. Skipped by default to avoid long Qt smoke walks.",
    )
    group.addoption(
        "--run-hardware",
        action="store_true",
        help="Run hardware-facing tests. Skipped by default; use only when validating hardware paths.",
    )


def _item_path(item: pytest.Item) -> Path:
    path = getattr(item, "path", None)
    if path is not None:
        return Path(path)
    return Path(str(item.fspath))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_by_marker = {
        marker: pytest.mark.skip(
            reason=f"requires explicit {option} opt-in for this validation tier"
        )
        for marker, option in DEFAULT_SKIP_MARKERS.items()
        if not config.getoption(option)
    }

    for item in items:
        path = _item_path(item)
        try:
            rel_path = path.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = path
        filename = path.name

        if rel_path.parts[:2] == ("tests", "integration"):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        if filename in GUI_TEST_FILES:
            item.add_marker(pytest.mark.gui)
        if filename in HARDWARE_TEST_FILES:
            item.add_marker(pytest.mark.hardware)
        if filename in SLOW_TEST_FILES:
            item.add_marker(pytest.mark.slow)
        if "cuda" in filename:
            item.add_marker(pytest.mark.gpu)

        for marker, skip_marker in skip_by_marker.items():
            if item.get_closest_marker(marker):
                item.add_marker(skip_marker)
                break


def _cleanup_qt_runtime() -> None:
    try:
        from PySide6.QtCore import QCoreApplication, QThread
        from PySide6.QtWidgets import QApplication
    except Exception:
        return

    app = QApplication.instance() or QCoreApplication.instance()
    if app is not None:
        try:
            app.processEvents()
        except Exception:
            pass
        top_level_widgets = getattr(app, "topLevelWidgets", None)
        if callable(top_level_widgets):
            for widget in list(top_level_widgets()):
                try:
                    widget.close()
                except Exception:
                    pass
        try:
            app.processEvents()
        except Exception:
            pass

    for obj in gc.get_objects():
        try:
            if isinstance(obj, QThread) and obj.isRunning():
                obj.requestInterruption()
                obj.quit()
                obj.wait(3000)
        except Exception:
            pass

    if app is not None:
        try:
            app.processEvents()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def _isolate_app_persistence(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory, request
):
    """Keep GUI/database tests from writing into the developer's app data."""
    test_id = hashlib.sha1(request.node.nodeid.encode("utf-8")).hexdigest()
    app_data = tmp_path_factory.getbasetemp() / "app-data" / test_id
    monkeypatch.setenv("XDG_DATA_HOME", str(app_data))
    monkeypatch.delenv("EIT_APP_DB_PATH", raising=False)


@pytest.fixture(autouse=True)
def _qt_runtime_cleanup_after_each_test():
    yield
    _cleanup_qt_runtime()


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    _cleanup_qt_runtime()


@pytest.fixture(scope="session")
def gmsh_mesh_artifacts(tmp_path_factory: pytest.TempPathFactory):
    """Provide a stable .msh cache for most tests (generation is tested separately)."""
    repo_root = Path(__file__).resolve().parents[1]
    mesh_dir = repo_root / "eit_meshes"
    mesh_name = "mesh_102070"
    return {
        "mesh_dir": mesh_dir,
        "mesh_name": mesh_name,
        "msh_file": Path(mesh_dir) / f"{mesh_name}.msh",
        "association_file": Path(mesh_dir) / f"{mesh_name}_association_table.ini",
    }


@pytest.fixture(scope="session")
def eit_mesh(gmsh_mesh_artifacts):
    if _TEST_STACK_IMPORT_ERROR is not None:
        pytest.skip(f"requires DOLFINx test stack: {_TEST_STACK_IMPORT_ERROR}")
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    ).astype(np.int32)
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x[:, :2]
    centroids = np.zeros((boundary_facets.size, 2), dtype=float)
    for i, facet in enumerate(boundary_facets):
        vertices = f2v.links(int(facet))
        centroids[i, :] = coords[vertices].mean(axis=0)

    x = centroids[:, 0]
    y = centroids[:, 1]
    eps = 1e-10
    t = np.zeros_like(x)
    left = np.isclose(x, 0.0, atol=eps)
    top = (~left) & np.isclose(y, 1.0, atol=eps)
    right = (~left) & (~top) & np.isclose(x, 1.0, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, 0.0, atol=eps)
    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])
    seg_len = 4.0 / 16
    tags = (np.floor(np.clip(t, 0.0, 4.0 - eps) / seg_len).astype(np.int32) + 2).astype(
        np.int32
    )
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], tags[order])
    association = {f"electrode_{idx + 1}": idx + 2 for idx in range(16)}
    return build_eit_mesh(
        mesh, facet_tags=facet_tags, association_table=association, radius=1.0
    )


@pytest.fixture(scope="session")
def eit_system(eit_mesh):
    if _TEST_STACK_IMPORT_ERROR is not None:
        pytest.skip(f"requires DOLFINx test stack: {_TEST_STACK_IMPORT_ERROR}")
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    system = EITSystem(
        n_elec=16,
        pattern_config=pattern,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        cache_scope="off",
    )
    system.setup(mesh=eit_mesh)
    # Keep tests fast while still exercising full GN pipeline.
    system.reconstructor.max_iterations = 2
    system.reconstructor.min_iterations = 1
    system.reconstructor.verbose = False
    return system
