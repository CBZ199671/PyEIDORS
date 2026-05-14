from __future__ import annotations

import numpy as np
import pytest

from pyeidors.inverse.greit import (
    GREITWeightSearchResult,
    build_3d_greit_rm,
    optimize_greit_weight_eidors_nf,
    optimize_greit_weight_for_metric,
    search_greit_weight_for_metric,
)


def test_tiny_bracket_search_fixture_has_no_rm_core_coupling() -> None:
    calls: list[float] = []
    optimum = -0.65
    target_metric = 2.5

    def metric_fn(log10_weight: float) -> float:
        calls.append(float(log10_weight))
        return target_metric + (log10_weight - optimum)

    result = search_greit_weight_for_metric(
        metric_fn,
        target_metric=target_metric,
        bracket=(-2.0, 1.0),
        tolerance=1.0e-5,
        maxiter=72,
    )

    assert isinstance(result, GREITWeightSearchResult)
    assert result.log10_weight == pytest.approx(optimum, abs=2.0e-4)
    assert result.weight == pytest.approx(10.0**optimum, rel=5.0e-4)
    assert result.achieved_metric == pytest.approx(target_metric, abs=2.0e-4)
    assert result.metadata["search_variable"] == "log10_weight"
    assert result.metadata["initial_bracket"] == (-2.0, 1.0)
    assert result.metadata["bracket_expansions"] == 0
    assert result.evaluations <= 80
    assert len(calls) == result.evaluations


def test_bracket_search_expands_when_optimum_starts_outside_bounds() -> None:
    optimum = 2.4
    target_metric = 1.25

    def metric_fn(log10_weight: float) -> float:
        return target_metric + (log10_weight - optimum)

    result = search_greit_weight_for_metric(
        metric_fn,
        target_metric=target_metric,
        bracket=(-1.0, 1.0),
        tolerance=1.0e-4,
        maxiter=80,
        max_expand=3,
    )

    assert result.log10_weight == pytest.approx(optimum, abs=2.0e-3)
    assert result.metadata["bracket_expansions"] >= 1
    lo, hi = result.bracket
    assert lo < optimum < hi
    assert result.evaluations <= 160


def test_metric_optimizer_uses_calc_greit_rm_as_black_box() -> None:
    y = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.4, 0.9, -0.3],
            [-0.2, 0.5, 0.8],
        ],
        dtype=float,
    )
    d = np.eye(3, dtype=float)
    noise = np.array(
        [
            [0.01, -0.02, 0.03],
            [-0.03, 0.01, -0.02],
            [0.02, 0.03, -0.01],
        ],
        dtype=float,
    )

    reference = optimize_greit_weight_for_metric(
        y,
        d,
        target_metric=1.0,
        metric="noise_figure",
        measurement_noise=noise,
        bracket=(-4.0, 2.0),
        tolerance=1.0e-3,
    )
    result = optimize_greit_weight_for_metric(
        y,
        d,
        target_metric=reference.achieved_metric,
        metric="noise_figure",
        measurement_noise=noise,
        bracket=(-4.0, 2.0),
        tolerance=1.0e-3,
    )

    assert result.objective_value <= 1.0e-6
    assert result.metadata["algorithm"] == "greit_weight_metric_search"
    assert result.metadata["metric"] == "noise_figure"
    assert result.metadata["uses_calc_greit_rm_as_black_box"] is True
    assert result.metadata["pjt_cache_source"] == "computed_once"
    assert result.metadata["pjt_cache_reused_across_weight_search"] is True
    assert result.metadata["noise_source"] == "provided"


def test_eidors_nf_optimizer_matches_vi_nf_and_signal_y_paths() -> None:
    y = np.array(
        [
            [0.9, 0.1, -0.2],
            [0.3, 0.7, 0.4],
            [-0.1, 0.2, 0.8],
            [0.5, -0.4, 0.2],
        ],
        dtype=float,
    )
    d = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.2, 0.8, 0.1],
            [0.0, 0.3, 0.9],
        ],
        dtype=float,
    )
    vh = np.array([1.1, -0.3, 0.6, -0.8], dtype=float)
    signal_y = np.array([0.03, -0.015, 0.022, -0.011], dtype=float)
    vi_nf = vh + signal_y
    volumes = np.array([0.7, 1.2, 1.0], dtype=float)

    reference = optimize_greit_weight_eidors_nf(
        y,
        d,
        vh=vh,
        signal_y=signal_y,
        volume_weights=volumes,
        target_noise_figure=2.5,
        bracket=(-3.0, 2.0),
        tolerance=1.0e-3,
    )
    result = optimize_greit_weight_eidors_nf(
        y,
        d,
        vh=vh,
        vi_nf=vi_nf,
        volume_weights=volumes,
        target_noise_figure=2.5,
        bracket=(-3.0, 2.0),
        tolerance=1.0e-3,
    )

    assert result.objective_value <= 1.0e-6
    assert result.weight == pytest.approx(reference.weight, rel=2.0e-3)
    assert result.metadata["algorithm"] == "eidors_greit_noise_figure_search"
    assert result.metadata["metric"] == "noise_figure"
    assert result.metadata["signal_source"] == "computed_from_vi_nf"
    assert result.metadata["volume_weight_source"] == "provided"
    assert result.metadata["vh_std_ddof"] == 1
    assert result.metadata["boundary_margin"] == pytest.approx(0.1)
    assert result.metadata["pjt_cache_reused_across_weight_search"] is True


def test_build_3d_greit_rm_can_choose_weight_from_metric_search() -> None:
    jacobian = np.eye(3, dtype=float)
    targets = np.eye(3, dtype=float)
    search_noise = np.array(
        [
            [0.01, -0.02, 0.03],
            [-0.03, 0.01, -0.02],
            [0.02, 0.03, -0.01],
        ],
        dtype=float,
    )

    reference = build_3d_greit_rm(
        jacobian=jacobian,
        targets=targets,
        noise_figure=0.05,
    )
    target_metric = optimize_greit_weight_for_metric(
        reference.training_responses.T,
        targets.T,
        target_metric=1.0,
        metric="image_snr",
        measurement_noise=search_noise,
        bracket=(-4.0, 1.0),
    ).achieved_metric

    greit = build_3d_greit_rm(
        jacobian=jacobian,
        targets=targets,
        noise_figure=None,
        image_snr=target_metric,
        weight_search_bracket=(-4.0, 1.0),
        weight_search_tolerance=1.0e-3,
        weight_search_noise=search_noise,
    )

    assert greit.metadata["weight_source"] == "metric_search"
    assert greit.metadata["target_image_snr"] == pytest.approx(target_metric)
    assert greit.metadata["weight_search"]["metric"] == "image_snr"
    assert greit.metadata["weight_search"]["objective_value"] <= 1.0e-6
    assert greit.metadata["weight_search"]["pjt_cache_source"] == "provided"
    assert (
        greit.metadata["weight_search"]["pjt_cache_reused_across_weight_search"] is True
    )
    assert greit.metadata["pjt_cache_source"] == "provided_cache"
    assert greit.metadata["pjt_cache_reused_across_weight_search"] is True
    assert greit.metadata["cache_signature_hash"]
    assert greit.metadata["weight"] > 0.0


def test_metric_search_rejects_explicit_weight_mix() -> None:
    with pytest.raises(ValueError, match="noise_figure=None"):
        build_3d_greit_rm(
            jacobian=np.eye(2, dtype=float),
            targets=np.eye(2, dtype=float),
            noise_figure=0.5,
            target_noise_figure=1.0,
        )
