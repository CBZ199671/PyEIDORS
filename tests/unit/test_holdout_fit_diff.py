from __future__ import annotations

from dataclasses import replace

import numpy as np

from pyeidors.data.eit_digit_metrics import build_surrogate_linearized_model
from pyeidors.data.holdout_fit_diff import (
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
