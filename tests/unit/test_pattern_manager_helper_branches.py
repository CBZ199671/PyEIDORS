"""Additional branch coverage for stimulation/measurement pattern helpers."""

from __future__ import annotations

import inspect
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


def test_resolve_electrode_lengths_and_parse_error_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    manager = StimMeasPatternManager.__new__(StimMeasPatternManager)
    manager.n_elec = 4
    manager.n_rings = 2
    manager.tn_elec = 8
    manager.drive_mode = "line_current_density"

    np.testing.assert_allclose(
        manager._resolve_electrode_lengths(None), np.ones(8, dtype=float)
    )
    np.testing.assert_allclose(
        manager._resolve_electrode_lengths(np.array([1.0, 2.0, 3.0, 4.0], dtype=float)),
        np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
    )
    with pytest.raises(ValueError, match="size mismatch"):
        manager._resolve_electrode_lengths(np.array([1.0, 2.0], dtype=float))
    with pytest.raises(ValueError, match="must be positive"):
        manager._resolve_electrode_lengths(np.array([1.0, -1.0, 1.0, 1.0], dtype=float))

    monkeypatch.setattr(
        patterns_module, "validate_drive_config", lambda **kwargs: "total_current"
    )
    with pytest.raises(ValueError, match="Unknown stimulation pattern"):
        StimMeasPatternManager(_config(stim_pattern="{bad}"))
    with pytest.raises(ValueError, match="Unknown measurement pattern"):
        StimMeasPatternManager(_config(meas_pattern="{bad}"))

    monkeypatch.setattr(
        patterns_module,
        "build_stim_currents",
        lambda **kwargs: (
            np.asarray(kwargs["inj_weights"], dtype=float) * kwargs["drive_value"]
        ),
    )
    single = StimMeasPatternManager(
        _config(stim_pattern=[0], meas_pattern=[0], use_meas_current=True),
        mesh_tdim=2,
    )
    np.testing.assert_allclose(single.inj_weights, np.array([1], dtype=float))
    np.testing.assert_allclose(single.meas_weights, np.array([1], dtype=float))


def test_selector_hash_filter_and_getter_branches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        patterns_module, "validate_drive_config", lambda **kwargs: "total_current"
    )
    monkeypatch.setattr(
        patterns_module,
        "build_stim_currents",
        lambda **kwargs: (
            np.asarray(kwargs["inj_weights"], dtype=float) * kwargs["drive_value"]
        ),
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

    assert manager.meas_selector.shape == (manager.tn_elec * manager.n_stim,)
    assert np.all(manager.meas_selector)
    assert manager.get_stim_matrix() is manager.stim_matrix

    empty_hash = manager._create_meas_hash(np.empty((0, manager.tn_elec), dtype=float))
    assert empty_hash.size == 0
    assert (
        manager._finite_summary(np.array([np.nan, np.inf], dtype=float))
        == "finite_count=0"
    )
    complex_summary = manager._finite_summary(
        np.array([1.0 + 1.0j, np.nan + 0.0j, 3.0 + 4.0j], dtype=np.complex128)
    )
    assert "finite_count=2" in complex_summary
    assert "max=5.000000e+00" in complex_summary
    source = inspect.getsource(StimMeasPatternManager._finite_summary)
    assert "[np.isfinite" not in source
    assert "np.abs(finite)" not in source

    hash_source = inspect.getsource(StimMeasPatternManager._create_meas_hash)
    assert "pos_hits = meas_mat > 0" in hash_source
    assert "neg_hits = meas_mat < 0" in hash_source
    assert "np.any(meas_mat > 0" not in hash_source
    assert "np.any(meas_mat < 0" not in hash_source

    meas_mat = manager._make_meas_matrix(elec=1, ring=0)
    assert meas_mat.shape == (manager.tn_elec, manager.tn_elec)
    assert np.any(meas_mat[0] != 0.0)


def test_v548_measurement_current_filter_direct_fills_row_mask() -> None:
    source = inspect.getsource(StimMeasPatternManager._filter_measurements)
    helper_source = inspect.getsource(patterns_module._rows_zero_at_columns)

    assert "_rows_zero_at_columns(meas_mat, stim_indices)" in source
    assert "_select_rows_by_mask(meas_mat, mask)" in source
    assert "meas_mat[mask]" not in source
    assert "np.any(meas_mat[:, stim_indices] != 0, axis=1)" not in source
    assert "np.equal(mat[:, int(column)], 0, out=work)" in helper_source
    assert "np.logical_and(mask, work, out=mask)" in helper_source

    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=float)
    np.testing.assert_array_equal(
        patterns_module._rows_zero_at_columns(matrix, [0, 1]),
        np.array([True, False, False, True]),
    )
    np.testing.assert_allclose(
        patterns_module._select_rows_by_mask(
            matrix,
            np.array([True, False, False, True]),
        ),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
    )


def test_v498_apply_meas_pattern_uses_bounded_finite_scan() -> None:
    source = inspect.getsource(StimMeasPatternManager.apply_meas_pattern)

    assert "all_finite_values(voltages)" in source
    assert "all_finite_values(measured)" in source
    assert "np.isfinite(voltages).all()" not in source
    assert "np.isfinite(measured).all()" not in source


def test_v304_multiring_lengths_and_selector_direct_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_tile(*_args, **_kwargs):
        raise AssertionError("multi-ring electrode lengths must not call np.tile")

    manager = StimMeasPatternManager.__new__(StimMeasPatternManager)
    manager.n_elec = 4
    manager.n_rings = 2
    manager.tn_elec = 8
    manager.drive_mode = "line_current_density"

    with monkeypatch.context() as patch_ctx:
        patch_ctx.setattr(patterns_module.np, "tile", _fail_tile)
        lengths = manager._resolve_electrode_lengths(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        )

    np.testing.assert_allclose(
        lengths,
        np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], dtype=float),
    )
    assert lengths.dtype == np.float64

    manager = StimMeasPatternManager.__new__(StimMeasPatternManager)
    manager.config = SimpleNamespace(use_meas_current=False)
    manager.n_stim = 2
    manager._full_meas_matrices = [
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=float,
        ),
        np.array(
            [
                [0.0, 1.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
    ]
    manager.meas_matrices = [
        manager._full_meas_matrices[0][[0, 2]],
        manager._full_meas_matrices[1][[0]],
    ]

    manager._compute_measurement_selector()

    np.testing.assert_array_equal(
        manager.meas_selector,
        np.array([True, False, True, True, False], dtype=bool),
    )
    assert "np.tile" not in inspect.getsource(
        StimMeasPatternManager._resolve_electrode_lengths
    )
    assert "np.any(lengths <= 0.0)" not in inspect.getsource(
        StimMeasPatternManager._resolve_electrode_lengths
    )
    assert "np.nonzero(lengths <= 0.0)" not in inspect.getsource(
        StimMeasPatternManager._resolve_electrode_lengths
    )
    assert "np.concatenate" not in inspect.getsource(
        StimMeasPatternManager._compute_measurement_selector
    )


def test_v297_cross_layer_measurement_matrix_direct_fills(monkeypatch) -> None:
    def _fail_vstack(*_args, **_kwargs):
        raise AssertionError("cross-layer measurement matrix must not call np.vstack")

    monkeypatch.setattr(patterns_module.np, "vstack", _fail_vstack)
    manager = StimMeasPatternManager.__new__(StimMeasPatternManager)
    manager.measurement_protocol = "cross_layer_full"
    manager.n_rings = 2
    manager.tn_elec = 4
    same_layer = np.array([[1.0, -1.0, 0.0, 0.0]], dtype=np.float32)
    cross_layer = np.array(
        [[0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, -1.0]],
        dtype=np.float64,
    )
    manager._make_meas_matrix_for_rings = lambda _elec, _rings: same_layer
    manager._make_cross_layer_meas_matrix = lambda _elec: cross_layer

    out = manager._make_meas_matrix_for_protocol(elec=0, ring=0)

    np.testing.assert_allclose(out[:1], same_layer)
    np.testing.assert_allclose(out[1:], cross_layer)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous
    assert "np.vstack" not in inspect.getsource(
        StimMeasPatternManager._make_meas_matrix_for_protocol
    )


def test_opposite_patterns_and_positive_first_branch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        patterns_module, "validate_drive_config", lambda **kwargs: "total_current"
    )
    monkeypatch.setattr(
        patterns_module,
        "build_stim_currents",
        lambda **kwargs: (
            np.asarray(kwargs["inj_weights"], dtype=float) * kwargs["drive_value"]
        ),
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


def test_custom_protocol_accepts_ragged_per_stim_measurement_matrices():
    stim_matrix = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
        ],
        dtype=float,
    )
    meas_matrices = [
        np.array([[1.0, -1.0, 0.0, 0.0]], dtype=float),
        np.array(
            [
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=float,
        ),
    ]
    manager = StimMeasPatternManager(
        _config(
            measurement_protocol="custom",
            custom_stim_matrix=stim_matrix,
            custom_meas_matrices=meas_matrices,
        ),
        mesh_tdim=2,
    )

    assert manager.n_meas_per_stim == [1, 2]
    assert manager.n_meas_total == 3
    np.testing.assert_allclose(manager.stim_matrix, stim_matrix)
    voltages = np.array(
        [
            [3.0, 1.0, 0.0, 0.0],
            [0.0, 4.0, 1.0, -2.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        manager.apply_meas_pattern(voltages),
        np.array([2.0, 3.0, 3.0], dtype=float),
    )
