"""Additional branch coverage for stimulation/measurement pattern helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pyeidors.electrodes.patterns as patterns_module
from pyeidors.data.structures import PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager


def _config(**kwargs) -> PatternConfig:
    payload = dict(
        n_elec=4,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=2.0,
        geometry_scale_to_m=1.0,
    )
    payload.update(kwargs)
    return PatternConfig(**payload)


def test_resolve_electrode_lengths_and_parse_error_paths(monkeypatch: pytest.MonkeyPatch):
    manager = StimMeasPatternManager.__new__(StimMeasPatternManager)
    manager.n_elec = 4
    manager.n_rings = 2
    manager.tn_elec = 8
    manager.drive_mode = "line_current_density"

    np.testing.assert_allclose(manager._resolve_electrode_lengths(None), np.ones(8, dtype=float))
    np.testing.assert_allclose(
        manager._resolve_electrode_lengths(np.array([1.0, 2.0, 3.0, 4.0], dtype=float)),
        np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
    )
    with pytest.raises(ValueError, match="size mismatch"):
        manager._resolve_electrode_lengths(np.array([1.0, 2.0], dtype=float))
    with pytest.raises(ValueError, match="must be positive"):
        manager._resolve_electrode_lengths(np.array([1.0, -1.0, 1.0, 1.0], dtype=float))

    monkeypatch.setattr(patterns_module, "validate_drive_config", lambda **kwargs: "total_current")
    with pytest.raises(ValueError, match="Unknown stimulation pattern"):
        StimMeasPatternManager(_config(stim_pattern="{bad}"))
    with pytest.raises(ValueError, match="Unknown measurement pattern"):
        StimMeasPatternManager(_config(meas_pattern="{bad}"))

    monkeypatch.setattr(
        patterns_module,
        "build_stim_currents",
        lambda **kwargs: np.asarray(kwargs["inj_weights"], dtype=float) * kwargs["drive_value"],
    )
    single = StimMeasPatternManager(
        _config(stim_pattern=[0], meas_pattern=[0], use_meas_current=True),
        mesh_tdim=2,
    )
    np.testing.assert_allclose(single.inj_weights, np.array([1], dtype=float))
    np.testing.assert_allclose(single.meas_weights, np.array([1], dtype=float))


def test_selector_hash_filter_and_getter_branches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(patterns_module, "validate_drive_config", lambda **kwargs: "total_current")
    monkeypatch.setattr(
        patterns_module,
        "build_stim_currents",
        lambda **kwargs: np.asarray(kwargs["inj_weights"], dtype=float) * kwargs["drive_value"],
    )
    manager = StimMeasPatternManager(
        _config(
            n_rings=2,
            use_meas_current=True,
            rotate_meas=False,
            stim_direction="cw",
            meas_direction="cw",
        ),
        electrode_lengths_m=np.arange(1, 9, dtype=float),
        mesh_tdim=2,
    )

    assert manager.meas_selector.shape == (manager.n_elec * manager.n_stim,)
    assert np.all(manager.meas_selector)
    assert manager.get_stim_matrix() is manager.stim_matrix

    empty_hash = manager._create_meas_hash(np.empty((0, manager.tn_elec), dtype=float))
    assert empty_hash.size == 0
    assert manager._finite_summary(np.array([np.nan, np.inf], dtype=float)) == "finite_count=0"

    meas_mat = manager._make_meas_matrix(elec=1, ring=0)
    assert meas_mat.shape == (manager.tn_elec, manager.tn_elec)
    assert np.any(meas_mat[0] != 0.0)


def test_opposite_patterns_and_positive_first_branch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(patterns_module, "validate_drive_config", lambda **kwargs: "total_current")
    monkeypatch.setattr(
        patterns_module,
        "build_stim_currents",
        lambda **kwargs: np.asarray(kwargs["inj_weights"], dtype=float) * kwargs["drive_value"],
    )
    manager = StimMeasPatternManager(
        _config(
            stim_pattern="{op}",
            meas_pattern="{op}",
            stim_first_positive=True,
            use_meas_current=True,
        ),
        mesh_tdim=2,
    )
    assert manager.inj_electrodes == [0, 2]
    np.testing.assert_allclose(manager.inj_weights, np.array([1, -1], dtype=float))
    assert manager.meas_electrodes == [0, 2]


def test_filter_measurements_with_neighbor_exclusion_branch():
    manager = StimMeasPatternManager.__new__(StimMeasPatternManager)
    manager.n_elec = 4
    manager.n_rings = 1
    manager.tn_elec = 4
    manager.inj_electrodes = [0, 1]
    manager.stim_direction = 1
    manager.config = SimpleNamespace(use_meas_current_next=1)

    meas_mat = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    filtered = manager._filter_measurements(meas_mat, elec=0, ring=0)
    assert filtered.shape == (0, 4)
