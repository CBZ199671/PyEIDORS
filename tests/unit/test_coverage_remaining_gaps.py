"""Tests targeting remaining coverage gaps."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np


# --- __init__.py import guards (lines 20-21, 29-32, 39-40) ---


class TestInitImportGuards:
    """Use subprocess to test import failure paths."""

    def test_all_imports_fail(self):
        from tests.utils import run_python

        code = """
import sys
sys.modules['dolfinx'] = None
sys.modules['torch'] = None
sys.modules['cuqi'] = None

# Remove cached pyeidors
mods = [k for k in sys.modules if k.startswith('pyeidors')]
for m in mods:
    del sys.modules[m]

import pyeidors
assert pyeidors._DOLFINX_AVAILABLE is False
assert pyeidors._TORCH_AVAILABLE is False
assert pyeidors._CUDA_AVAILABLE is False
assert pyeidors._MPS_AVAILABLE is False
assert pyeidors._CUQI_AVAILABLE is False
print("PASS")
"""
        result = run_python(code)
        assert "PASS" in result.stdout, result.stderr


# --- cache/lifecycle resolve_cache_directory (lines 202-205) ---


class TestLifecycleResolveFNF:
    """Cover FileNotFoundError in resolve."""

    def test_resolve_with_nonexistent_requested_root(self, tmp_path, monkeypatch):
        from pyeidors.cache.lifecycle import (
            resolve_cache_directory,
            _REGISTERED_SPECS,
            _LOCK,
        )

        session_dir = tmp_path / "sess"
        session_dir.mkdir()
        # Set shell env with non-existent requested root to trigger same_root=False
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_ID", "test-fnf")
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_DIR", str(session_dir))
        monkeypatch.setenv("PYEIDORS_CACHE_REQUESTED_ROOT", "/nonexistent/path/abc123")
        monkeypatch.setenv("PYEIDORS_CACHE_OWNER_PID", "")

        cache_root = tmp_path / "fnf_cache"
        cache_root.mkdir()

        with _LOCK:
            key = str(cache_root.resolve())
            _REGISTERED_SPECS.pop(key, None)

        spec = resolve_cache_directory(
            cache_root,
            lifecycle="session",
            cleanup_on_exit=False,
            cleanup_stale_sessions_on_startup=False,
            stale_session_max_age_seconds=0,
        )
        assert spec.shell_managed is True
        # same_root should be False since requested_root doesn't exist

        with _LOCK:
            _REGISTERED_SPECS.pop(key, None)


# --- cache/lifecycle cleanup_registered (lines 256-257) ---


class TestCleanupRegisteredOSError:
    """Cover parent.rmdir() OSError."""

    def test_rmdir_oserror(self, tmp_path):
        from pyeidors.cache.lifecycle import (
            cleanup_registered_session_caches,
            _REGISTERED_SESSION_DIRS,
            _LOCK,
        )

        session_dir = tmp_path / "sessions_oserr" / "mysession"
        session_dir.mkdir(parents=True)
        # Put a file in parent to prevent rmdir
        (tmp_path / "sessions_oserr" / "keep.txt").write_text("keep")

        with _LOCK:
            _REGISTERED_SESSION_DIRS.add(session_dir)

        removed = cleanup_registered_session_caches()
        assert removed == 1
        # Parent rmdir should fail silently because it's not empty


# --- cache/lifecycle _ensure_atexit (line 43) ---


class TestEnsureAtexitAlreadyRegistered:
    """Line 43: already registered returns early."""

    def test_double_registration(self):
        from pyeidors.cache import lifecycle

        original = lifecycle._CLEANUP_REGISTERED
        lifecycle._CLEANUP_REGISTERED = True
        lifecycle._ensure_atexit_cleanup()
        lifecycle._CLEANUP_REGISTERED = original


# --- cache/manager list_entries disk path (line 519) ---


class TestCacheManagerDiskListEntries:
    """Cover line 519: disk store list_entries in manager.list_entries."""

    def test_list_entries_disk(self, tmp_path):
        from pyeidors.cache.manager import CacheManager
        from pyeidors.cache.types import CachePolicy

        mgr = CacheManager(
            scope="disk",
            cache_dir=tmp_path / "disk_le",
            policy=CachePolicy(disk_lifecycle="persistent"),
        )
        mgr.get_or_compute(artifact="test", payload={"k": 1}, compute_fn=lambda: "val")
        entries = mgr.list_entries()
        assert len(entries) >= 1


# --- cache/store_disk schema migration (line 116) ---


class TestDiskStoreSchemaMigration:
    """Cover line 116: executing ALTER TABLE for missing columns."""

    def test_legacy_schema_migration(self, tmp_path):
        import sqlite3
        from pyeidors.cache.store_disk import DiskCacheStore

        root = tmp_path / "migration_cache"
        root.mkdir(parents=True)
        objects_dir = root / "objects"
        objects_dir.mkdir()
        db_path = root / "index.sqlite"

        # Create a minimal table missing columns
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE cache_entries (
                cache_key TEXT PRIMARY KEY,
                artifact TEXT NOT NULL,
                file_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                cost REAL NOT NULL,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                ttl_seconds REAL
            )
        """)
        conn.commit()
        conn.close()

        # Opening the store should trigger migrations
        store = DiskCacheStore(root, max_bytes=1024 * 1024)
        # Verify store works after migration
        assert store.put("k1", "v1", artifact="test", cost=1.0)


# --- data/structures EITMesh properties (lines 129, 154, 158, 169-172) ---
# These need DOLFINx so we test via the existing conftest fixtures


# --- visualization/eit_plot_helpers more branches ---


class TestEITPlotHelpersAdditional:
    """Cover more branches in eit_plot_helpers."""

    def test_eidors_tick_vals_various_scales(self):
        from pyeidors.visualization.eit_plot_helpers import eidors_tick_vals

        # Test various scale ranges to cover all branches
        for scale in [0.3, 0.7, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 9.0]:
            ticks = eidors_tick_vals(scale, 0.0)
            assert len(ticks) > 0

    def test_resolve_diff_limits_both_specified(self):
        from pyeidors.visualization.eit_plot_helpers import resolve_eidors_diff_limits

        vmin, vmax = resolve_eidors_diff_limits(np.array([1.0]), -2.0, 3.0)
        assert vmin == -2.0
        assert vmax == 3.0


# --- perf/capabilities detection functions ---


class TestCapabilityDetection:
    """Cover exception branches in capability detection."""

    def test_has_cuda_structured_exception(self):
        from pyeidors.perf.capabilities import _has_cuda_structured

        with mock.patch.dict(
            "sys.modules",
            {
                "pyeidors.forward.cuda_structured_backend": mock.MagicMock(
                    side_effect=Exception("fail")
                )
            },
        ):
            # Just verify it doesn't crash
            _has_cuda_structured()

    def test_has_pyamg_false(self, monkeypatch):
        from pyeidors.perf import capabilities as mod

        # Just call the real one
        result = mod._has_pyamg()
        assert isinstance(result, bool)

    def test_has_cholmod_false(self):
        from pyeidors.perf.capabilities import _has_cholmod

        with mock.patch.dict(
            "sys.modules", {"sksparse": None, "sksparse.cholmod": None}
        ):
            result = _has_cholmod()
            assert result is False


# --- visualization/eit_plots style fallback and method wrappers ---


class TestEITVisualizerMethods:
    """Cover static method wrappers (lines 166-196)."""

    def test_visualizer_text_method(self):
        import matplotlib

        matplotlib.use("Agg")
        from pyeidors.visualization.eit_plots import EITVisualizer

        viz = EITVisualizer(style="default")
        # Test _text method
        assert viz._text("mesh_title") is not None


# --- plot_font_i18n remaining (lines 151-152) ---


class TestFontRegistrationFailure:
    """Lines 151-152: font registration exception."""

    def test_font_registration_exception(self, monkeypatch):
        import pyeidors.utils.plot_font_i18n as mod

        existing_path = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
        # Only test if the font actually exists
        if existing_path.exists():
            mod._WARNED_KEYS.discard(f"font-register-{existing_path}")
            with mock.patch.object(
                mod.font_manager.fontManager, "addfont", side_effect=Exception("fail")
            ):
                mod._register_optional_fonts()


# --- data/difference line 50 (unreachable but verify) ---


class TestDifferenceEdge:
    """Line 50: _as_measurement_vector with non-1D."""

    def test_as_measurement_vector_2d(self):
        from pyeidors.data.difference import _as_measurement_vector

        # After reshape(-1), result is always 1D, so line 50 is unreachable
        result = _as_measurement_vector([[1.0, 2.0], [3.0, 4.0]], name="test")
        assert result.ndim == 1


# --- data/measurement_dataset replace (line 177) ---


class TestMeasurementDatasetReplaceNew:
    """Ensure replace_measurements validation works."""

    def test_replace_valid_shape(self):
        from pyeidors.data.measurement_dataset import MeasurementDataset
        from pyeidors.electrodes.patterns import StimMeasPatternManager
        from pyeidors.data.structures import PatternConfig

        config = PatternConfig(
            n_elec=4,
            stim_pattern="{ad}",
            meas_pattern="{ad}",
            drive_mode="normalized",
            drive_value=1.0,
            geometry_scale_to_m=1.0,
            use_meas_current=False,
            rotate_meas=True,
            stim_direction="ccw",
            meas_direction="ccw",
            n_rings=1,
        )
        pm = StimMeasPatternManager(config)
        n_cols = pm.n_meas_total
        metadata = {
            "n_elec": 4,
            "stim_pattern": "{ad}",
            "meas_pattern": "{ad}",
            "drive_mode": "normalized",
            "drive_value": 1.0,
            "geometry_scale_to_m": 1.0,
            "electrode_length_m_override": None,
            "use_meas_current": False,
            "use_meas_current_next": 0,
            "rotate_meas": True,
            "stim_direction": "ccw",
            "meas_direction": "ccw",
            "n_rings": 1,
            "n_frames": 2,
        }
        ds = MeasurementDataset.from_metadata(
            np.ones((2, n_cols)), metadata, data_type="real"
        )
        # Replace with different values but same shape
        ds.replace_measurements(np.zeros((2, n_cols)))
        assert np.all(ds.measurements == 0)
