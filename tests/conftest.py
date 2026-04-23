"""Shared pytest fixtures for DOLFINx + gmsh integration tests."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

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
