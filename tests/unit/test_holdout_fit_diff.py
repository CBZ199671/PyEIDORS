from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np

import pyeidors.data.holdout_fit_diff as holdout_module
from pyeidors.data.eit_digit_metrics import build_surrogate_linearized_model
from pyeidors.data.holdout_fit_diff import (
    FIT_CURVE_LEGEND_LABELS,
    _weighted_centroid_covariance_2d,
    _prediction_marker_offsets,
    format_holdout_fit_report,
    plot_holdout_fit_curves,
    plot_holdout_fit_summary,
    plot_holdout_recon_compare,
    run_holdout_fit_diff,
)


def _surrogate_model():
    return build_surrogate_linearized_model(
        n_measurements=208,
        n_parameters=6,
        seed=123,
    )


def test_holdout_fit_diff_compares_raw_160_and_three_fit_methods() -> None:
    case = run_holdout_fit_diff(
        model=_surrogate_model(),
        fit_methods=["poly2", "poly3", "spline"],
        raw_160_baseline=True,
        inverse_backend="least-squares",
        ridge=1e-4,
    )

    assert {row.recon_method for row in case.summaries} == {
        "raw_160",
        "poly2_208",
        "poly3_208",
        "spline_208",
    }
    inverse_points = {row.recon_method: row.n_inverse_points for row in case.summaries}
    assert inverse_points["raw_160"] == 160
    assert inverse_points["poly2_208"] == 208
    assert inverse_points["poly3_208"] == 208
    assert inverse_points["spline_208"] == 208
    assert len(case.field_rows) == 4 * case.model.sigma_true.size
    assert {row.recon_kind for row in case.structure_rows} == {
        "truth",
        "full_208",
        "raw_160",
        "poly2_208",
        "poly3_208",
        "spline_208",
    }
    for row in case.summaries:
        assert np.isfinite(row.recon_sigma_relative_rmse)
        assert np.isfinite(row.delta_sigma_effective_digits)


def test_v241_holdout_weighted_structure_uses_shared_moment_helper() -> None:
    structure_source = inspect.getsource(holdout_module._weighted_structure)
    helper_source = inspect.getsource(_weighted_centroid_covariance_2d)

    assert "_masked_weighted_structure_stats_2d" in structure_source
    assert "weights[:, None]" not in structure_source
    assert "weights[:, None]" not in helper_source
    assert "coords * weights" not in helper_source
    assert "centered * weights" not in helper_source

    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=float)
    weights = np.array([1.0, 2.0, 1.0], dtype=float)
    centroid, covariance = _weighted_centroid_covariance_2d(coords, weights)

    np.testing.assert_allclose(centroid, [1.5, 0.5])
    np.testing.assert_allclose(covariance, [[0.75, 0.25], [0.25, 0.75]])


def test_v551_holdout_structure_metrics_stream_masked_reductions() -> None:
    structure_source = inspect.getsource(holdout_module._weighted_structure)
    rows_source = inspect.getsource(holdout_module._structure_metric_rows)
    helper_source = inspect.getsource(
        holdout_module._masked_weighted_structure_stats_2d
    )

    for legacy in (
        "weights_raw[mask]",
        "areas[mask]",
        "points[mask",
        "contrast[outside]",
        "areas[outside]",
        "artifact_active = mask & outside",
    ):
        assert legacy not in structure_source
        assert legacy not in rows_source
    assert "_masked_area_sum(areas, mask, exclude_mask=truth_mask)" in rows_source
    assert "_masked_square_area_sum(" in rows_source
    assert "_masked_abs_peak(" in rows_source
    assert "np.divide(margin, ratios, out=ratios, where=negative)" not in helper_source

    mask, cx, cy, area, eccentricity, major_axis, minor_axis = (
        holdout_module._weighted_structure(
            values=np.array([0.0, 2.0, 1.0], dtype=float),
            points=np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=float),
            areas=np.ones(3, dtype=float),
            threshold=0.5,
        )
    )
    np.testing.assert_array_equal(mask, [False, True, True])
    np.testing.assert_allclose([cx, cy], [2.0, 2.0 / 3.0])
    assert area == 2.0
    assert eccentricity >= 0.0
    assert major_axis >= minor_axis


def test_v496_holdout_fit_validators_use_bounded_finite_scan() -> None:
    vector_source = inspect.getsource(holdout_module._as_float_vector)
    matrix_source = inspect.getsource(holdout_module._as_float_matrix)
    fit_source = inspect.getsource(holdout_module._fit_values)
    areas_source = inspect.getsource(holdout_module._cell_areas)

    assert "all_finite_values(arr)" in vector_source
    assert "all_finite_values(arr)" in matrix_source
    assert "all_finite_values(predicted)" in fit_source
    assert "np.min(areas, initial=np.inf)" in areas_source
    assert "np.all(np.isfinite(arr))" not in vector_source
    assert "np.all(np.isfinite(arr))" not in matrix_source
    assert "np.all(np.isfinite(predicted))" not in fit_source
    assert "np.any(areas <= 0.0)" not in areas_source


def test_v528_holdout_rmse_at_indices_uses_chunked_take() -> None:
    run_source = inspect.getsource(holdout_module.run_holdout_fit_diff)
    value_helper_source = inspect.getsource(holdout_module._rmse_values_at_indices)
    pair_helper_source = inspect.getsource(holdout_module._rmse_pair_at_indices)

    assert "full_diff[holdout_indices]" not in run_source
    assert "v_true[holdout_indices]" not in run_source
    assert "fit_true[holdout_indices]" not in run_source
    assert "_rmse_values_at_indices(" in run_source
    assert "_rmse_pair_at_indices(" in run_source
    assert "np.take(arr, chunk, out=target)" in value_helper_source
    assert "np.take(obs, chunk, out=target)" in pair_helper_source
    assert "np.take(ref, chunk, out=ref_target)" in pair_helper_source

    values = np.array([0.0, 3.0, 4.0, 12.0], dtype=np.float64)
    indices = np.array([1, 2], dtype=np.int64)
    np.testing.assert_allclose(
        holdout_module._rmse_values_at_indices(values, indices, chunk_size=1),
        np.sqrt((3.0**2 + 4.0**2) / 2.0),
    )
    np.testing.assert_allclose(
        holdout_module._rmse_pair_at_indices(
            np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float64),
            np.array([1.0, 5.0, 9.0, 10.0], dtype=np.float64),
            indices,
            chunk_size=1,
        ),
        np.sqrt((3.0**2 + 4.0**2) / 2.0),
    )


def test_holdout_fit_diff_plots_and_report(tmp_path) -> None:
    case = run_holdout_fit_diff(
        model=_surrogate_model(),
        fit_methods=["poly2"],
        raw_160_baseline=True,
        inverse_backend="least-squares",
        ridge=1e-4,
    )

    report = format_holdout_fit_report(case)
    curve = plot_holdout_fit_curves(case, tmp_path / "curves.png", dpi=80)
    recon = plot_holdout_recon_compare(case, tmp_path / "recon.png", dpi=80)
    summary = plot_holdout_fit_summary(case, tmp_path / "summary.png", dpi=80)

    assert "拟合" in report
    for path in [curve, recon, summary]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000


def test_v36_fit_curves_keep_absolute_voltage_separate_from_diff() -> None:
    model = _surrogate_model()
    u_curve = np.array(
        [
            0.045,
            0.02,
            0.012,
            0.009,
            0.007,
            0.006,
            0.0058,
            0.006,
            0.007,
            0.009,
            0.012,
            0.02,
            0.045,
        ],
        dtype=float,
    )
    reference = np.tile(u_curve, 16)
    absolute_model = replace(
        model,
        voltage_reference=reference,
        voltage_true=reference + model.voltage_true,
    )

    case = run_holdout_fit_diff(
        model=absolute_model,
        fit_methods=["poly2"],
        raw_160_baseline=True,
        inverse_backend="least-squares",
        ridge=1e-4,
    )
    curve = case.frame_curves[0]

    np.testing.assert_allclose(curve.voltage_reference_full, reference[:13])
    np.testing.assert_allclose(
        curve.voltage_anomaly_full, reference[:13] + model.voltage_true[:13]
    )
    assert not np.allclose(curve.voltage_anomaly_full, curve.diff_full)
    assert "poly2" in curve.fitted_anomaly_by_method


def test_v38_fit_curve_prediction_marker_offsets_make_overlaps_visible() -> None:
    offsets = _prediction_marker_offsets(["poly2", "poly3", "spline"])

    assert set(offsets) == {"poly2", "poly3", "spline"}
    assert offsets["poly2"] < offsets["poly3"] < offsets["spline"]
    assert (
        min(
            abs(left - right)
            for idx, left in enumerate(offsets.values())
            for right in list(offsets.values())[idx + 1 :]
        )
        >= 0.15
    )


def test_v39_fit_curve_legend_labels_explain_original_fit_points() -> None:
    labels = set(FIT_CURVE_LEGEND_LABELS.values())

    assert "train 10" not in labels
    assert "holdout true" not in labels
    assert FIT_CURVE_LEGEND_LABELS["fit_input"] == "fit input: 10 original pts"
    assert FIT_CURVE_LEGEND_LABELS["withheld_true"] == "withheld true: 3 original pts"
    assert "original" in FIT_CURVE_LEGEND_LABELS["target_full"]
