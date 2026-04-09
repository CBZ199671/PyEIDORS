"""Low-risk branch tests that harvest small remaining coverage gaps."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib
import numpy as np
import pytest

import pyeidors.cache.lifecycle as lifecycle_mod
import pyeidors.cache.object_signature as sig_module
import pyeidors.perf.capabilities as perf_caps
from pyeidors.data.difference import _as_measurement_vector
from pyeidors.data.structures import EITMesh
from pyeidors.inverse.regularization.smoothness import TotalVariationRegularization
from pyeidors.physics import UnitCheckLevel, run_unit_consistency_checks
from pyeidors.visualization.eit_plot_helpers import format_colorbar, plot_electrodes

matplotlib.use("Agg")


def test_as_measurement_vector_rejects_3d_input():
    with pytest.raises(ValueError, match="3D"):
        _as_measurement_vector(np.zeros((1, 2, 3), dtype=float), name="test_values")


def test_eit_mesh_cells_handles_missing_connectivity_and_zero_cells():
    missing_topology = SimpleNamespace(
        dim=2,
        create_connectivity=lambda *_args: None,
        connectivity=lambda *_args: None,
        index_map=lambda dim: SimpleNamespace(size_local=4 if dim == 0 else 3),
    )
    missing_mesh = EITMesh(
        mesh=SimpleNamespace(
            topology=missing_topology,
            geometry=SimpleNamespace(dim=2, x=np.zeros((4, 2), dtype=float)),
            comm=None,
        ),
        facet_tags=None,
    )
    assert missing_mesh.cells().shape == (0, 0)

    class _Conn:
        @staticmethod
        def links(_idx):
            return np.array([0, 1, 2], dtype=np.int32)

    zero_topology = SimpleNamespace(
        dim=2,
        create_connectivity=lambda *_args: None,
        connectivity=lambda *_args: _Conn(),
        index_map=lambda dim: SimpleNamespace(size_local=4 if dim == 0 else 0),
    )
    zero_mesh = EITMesh(
        mesh=SimpleNamespace(
            topology=zero_topology,
            geometry=SimpleNamespace(dim=2, x=np.zeros((4, 2), dtype=float)),
            comm=None,
        ),
        facet_tags=None,
    )
    assert zero_mesh.cells().shape == (0, 0)


def test_total_variation_reference_vector_accepts_matching_array():
    fake_space = SimpleNamespace(dofmap=SimpleNamespace(index_map=SimpleNamespace(size_local=3), index_map_bs=1))
    fake_model = SimpleNamespace(mesh="mesh", V_sigma=fake_space)
    reg = TotalVariationRegularization(fake_model, reference_conductivity=np.array([1.0, 1.2, 0.8], dtype=float))
    np.testing.assert_allclose(reg._reference_vector(), np.array([1.0, 1.2, 0.8], dtype=float))


def test_model_signature_from_forward_model_handles_bad_coordinate_fallback():
    bad_mesh = SimpleNamespace(
        association_table={"domain": 1},
        mesh=SimpleNamespace(topology=SimpleNamespace(dim=2)),
        mesh_file=None,
        mesh_family="tetra",
        geometry_version="legacy",
        generator_revision="g0",
        structured_sidecar_file=None,
        structured_sidecar_version=None,
        coordinates=lambda: (_ for _ in ()).throw(RuntimeError("bad coordinates")),
        cells=lambda: (_ for _ in ()).throw(RuntimeError("bad cells")),
    )
    signature = sig_module.model_signature_from_forward_model(
        SimpleNamespace(n_elec=4, z=np.ones(4, dtype=float), geometry_scale_to_m=1.0, eit_mesh=bad_mesh)
    )
    assert isinstance(signature, str)
    assert signature


def test_cleanup_stale_session_caches_ignores_children_vanishing_before_stat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _VanishingChild:
        name = "stale_session"

        @staticmethod
        def is_dir() -> bool:
            return True

        @staticmethod
        def stat():
            raise FileNotFoundError("gone")

    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    original_iterdir = Path.iterdir

    def _fake_iterdir(self: Path):
        if self == tmp_path:
            return iter([_VanishingChild()])
        return original_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _fake_iterdir)
    assert lifecycle_mod.cleanup_stale_session_caches(tmp_path, max_age_seconds=0.0) == 0
    assert keep.exists()


def test_perf_capability_import_guards_and_fast_path_selection(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("eit_forward_model"):
            raise ImportError("blocked PETSc runtime")
        if name == "pyamg":
            raise ImportError("blocked pyamg")
        if name == "sksparse.cholmod":
            raise ImportError("blocked cholmod")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "pyeidors.forward.eit_forward_model", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert perf_caps._load_petsc_runtime() is None
    assert perf_caps._has_pyamg() is False
    assert perf_caps._has_cholmod() is False

    assert (
        perf_caps.select_fast_linear_path(
            "auto",
            regularization_is_diagonal=False,
            regularization_is_sparse_spd=True,
            capabilities={"cholmod": True},
        )
        == "pcg"
    )
    assert (
        perf_caps.select_fast_linear_path(
            "auto",
            regularization_is_diagonal=False,
            regularization_is_sparse_spd=False,
            capabilities={},
        )
        == "pcg"
    )


def test_perf_capability_positive_cholmod_detection(monkeypatch: pytest.MonkeyPatch):
    cholmod_module = SimpleNamespace(cholesky=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "sksparse", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", cholmod_module)
    assert perf_caps._has_cholmod() is True


def _fake_unit_model(
    coords: np.ndarray,
    *,
    drive_mode: str = "line_current_density",
    drive_value: float = 1.0,
    geometry_scale_to_m: float = 1.0,
    electrode_lengths: np.ndarray | list[float] | None = None,
    n_elec: int = 2,
) -> SimpleNamespace:
    coords = np.asarray(coords, dtype=float)
    gdim = int(coords.shape[1]) if coords.ndim == 2 and coords.size else 2
    mesh = SimpleNamespace(
        geometry=SimpleNamespace(x=coords, dim=gdim),
        topology=SimpleNamespace(dim=gdim),
    )
    cfg = SimpleNamespace(
        drive_mode=drive_mode,
        drive_value=drive_value,
        geometry_scale_to_m=geometry_scale_to_m,
    )
    stim_matrix = np.array([[1.0, -1.0]], dtype=float)
    return SimpleNamespace(
        pattern_manager=SimpleNamespace(
            config=cfg,
            stim_matrix=stim_matrix,
            _electrode_lengths_m=np.asarray(
                [0.1] * n_elec if electrode_lengths is None else electrode_lengths,
                dtype=float,
            ),
        ),
        mesh=mesh,
        n_elec=n_elec,
        electrode_lengths_m=np.asarray(
            [0.1] * n_elec if electrode_lengths is None else electrode_lengths,
            dtype=float,
        ),
    )


def test_unit_consistency_checks_cover_error_paths():
    bad_drive = _fake_unit_model(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float), drive_mode="invalid")
    report = run_unit_consistency_checks(bad_drive)
    assert len(report.items) == 1
    assert report.items[0].name == "drive_config_validity"
    assert report.items[0].level == UnitCheckLevel.ERROR

    empty_coords = _fake_unit_model(np.empty((0, 2), dtype=float), drive_mode="normalized")
    empty_report = run_unit_consistency_checks(empty_coords)
    empty_geom = next(item for item in empty_report.items if item.name == "geometry_scale_consistency")
    assert empty_geom.level == UnitCheckLevel.ERROR
    assert "Mesh has no coordinates" in empty_geom.message

    zero_extent = _fake_unit_model(
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        electrode_lengths=np.array([0.2], dtype=float),
        n_elec=2,
        drive_mode="normalized",
    )
    zero_report = run_unit_consistency_checks(zero_extent)
    geom_item = next(item for item in zero_report.items if item.name == "geometry_scale_consistency")
    length_item = next(item for item in zero_report.items if item.name == "electrode_length_physical_consistency")
    assert geom_item.level == UnitCheckLevel.ERROR
    assert length_item.level == UnitCheckLevel.ERROR


def test_plot_helper_branches_and_visualizer_import_guard(eit_mesh, monkeypatch: pytest.MonkeyPatch):
    ax = mock.MagicMock()
    plot_electrodes(
        ax,
        [
            np.empty((0, 2), dtype=float),
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            np.array([[1.0, 0.0], [1.0, 1.0]], dtype=float),
            np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float),
            np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float),
            np.array([[0.5, 0.0], [0.5, 1.0]], dtype=float),
        ],
    )
    assert ax.plot.call_count == 5
    ax.legend.assert_not_called()

    cbar = mock.MagicMock()
    cbar.ax.yaxis.get_offset_text.return_value = mock.MagicMock()
    format_colorbar(cbar, "scientific")
    assert cbar.formatter(99.99, 0).startswith("1e+02")
    assert cbar.formatter(12.34, 0).startswith("1.2e+01")

    import pyeidors.visualization.eit_plots as viz_module

    temp_name = "pyeidors.visualization._eit_plots_blocked_mpl"
    spec = importlib.util.spec_from_file_location(temp_name, viz_module.__file__)
    assert spec is not None and spec.loader is not None
    blocked_module = importlib.util.module_from_spec(spec)
    sys.modules[temp_name] = blocked_module
    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.pyplot":
            raise ImportError("blocked matplotlib")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    try:
        spec.loader.exec_module(blocked_module)
        assert blocked_module.MATPLOTLIB_AVAILABLE is False
        with pytest.raises(ImportError, match="matplotlib not available"):
            blocked_module.EITVisualizer()
    finally:
        sys.modules.pop(temp_name, None)

    viz = viz_module.EITVisualizer(style="default")
    viz._texts["demo"] = "value={value}"
    assert viz._text("demo", value="ok") == "value=ok"
    cbar2 = mock.MagicMock()
    cbar2.ax.yaxis.get_offset_text.return_value = mock.MagicMock()
    viz._apply_eidors_ticks(cbar2, -1.0, 1.0)
    viz._format_colorbar(cbar2, "plain")
    values = viz._interpolate_cell_to_node(eit_mesh, np.ones(eit_mesh.num_cells(), dtype=float))
    assert values.shape[0] == eit_mesh.num_vertices()
    vmin, vmax = viz._resolve_eidors_diff_limits(np.array([1.0, -2.0], dtype=float), None, None)
    assert vmin < 0 < vmax
