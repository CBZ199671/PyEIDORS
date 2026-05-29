"""Regression tests for script-side array hash helpers."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_large_payload_script_hashes_use_streaming_helpers() -> None:
    expected_helpers = {
        "scripts/run_synthetic_parity.py": "hash_array_payload",
        "scripts/benchmarks/benchmark_difference_runtime.py": "hash_array_payload",
        "scripts/benchmarks/benchmark_forward_ksp_session_reuse.py": (
            "update_digest_with_array_payload"
        ),
        "scripts/benchmarks/benchmark_mesh_io_formats.py": "hash_array_payload",
    }
    for relpath, helper in expected_helpers.items():
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert helper in text
        assert ".tobytes(" not in text
        assert "np.ascontiguousarray(np.asarray" not in text


def test_v547_remaining_script_finite_guards_use_bounded_helpers() -> None:
    greit_text = (
        REPO_ROOT / "scripts/diagnostics/compare_greit_eidors_parity.py"
    ).read_text(encoding="utf-8")
    assert "all_finite_values(array)" in greit_text
    assert "np.isfinite(array).all()" not in greit_text

    ksp_text = (
        REPO_ROOT / "scripts/benchmarks/benchmark_forward_ksp_session_reuse.py"
    ).read_text(encoding="utf-8")
    assert "all_finite_values(sigma_sequence)" in ksp_text
    assert "_any_value_below(sigma_sequence" in ksp_text
    assert "np.isfinite(sigma_sequence).all()" not in ksp_text
    assert "np.any(sigma_sequence <" not in ksp_text

    dynamic_text = (
        REPO_ROOT / "scripts/benchmarks/benchmark_dynamic_validation.py"
    ).read_text(encoding="utf-8")
    assert "all_finite_values(arr)" in dynamic_text
    assert "np.isfinite(arr).all()" not in dynamic_text


def test_v548_remaining_script_comparison_guards_use_bounded_helpers() -> None:
    greit_text = (
        REPO_ROOT / "scripts/diagnostics/compare_greit_eidors_parity.py"
    ).read_text(encoding="utf-8")
    assert "any_abs_less_equal_values(vh" in greit_text
    assert "np.any(np.abs(denom)" not in greit_text

    dynamic_sweep_text = (
        REPO_ROOT / "scripts/benchmarks/benchmark_dynamic_tv_huber_sweep.py"
    ).read_text(encoding="utf-8")
    assert "_strictly_increasing(pos)" in dynamic_sweep_text
    assert "np.any(np.diff(pos) <= 0.0)" not in dynamic_sweep_text

    gn_text = (REPO_ROOT / "scripts/common/gn_difference_runner.py").read_text(
        encoding="utf-8"
    )
    assert "all_finite_values(raw_sigma_est)" in gn_text
    assert "any_not_equal_values(sigma_est, raw_sigma_est)" in gn_text
    assert "np.all(np.isfinite(raw_sigma_est))" not in gn_text
    assert "np.any(sigma_est != raw_sigma_est)" not in gn_text


def test_v549_remaining_np_all_isfinite_guards_use_bounded_helpers() -> None:
    gn_text = (REPO_ROOT / "scripts/common/gn_difference_runner.py").read_text(
        encoding="utf-8"
    )
    assert "all_finite_values(arr)" in gn_text
    assert "all_finite_values(x)" in gn_text
    assert "np.all(np.isfinite(" not in gn_text

    tank_text = (REPO_ROOT / "scripts/run_tank_realdata_holdout_compare.py").read_text(
        encoding="utf-8"
    )
    assert "all_finite_values(predicted)" in tank_text
    assert "np.all(np.isfinite(predicted))" not in tank_text

    runtime_text = (REPO_ROOT / "scripts/benchmarks/benchmark_3d_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "all_finite_values(electrode_voltages)" in runtime_text
    assert "np.all(np.isfinite(electrode_voltages))" not in runtime_text


def test_v550_gn_difference_sigma_floor_step_uses_chunked_limit_helper() -> None:
    gn_text = (REPO_ROOT / "scripts/common/gn_difference_runner.py").read_text(
        encoding="utf-8"
    )
    assert "min_alpha_for_value_floor(sigma, delta, floor)" in gn_text
    assert "sigma[negative_update]" not in gn_text
    assert "delta[negative_update]" not in gn_text
    assert "np.any(negative_update)" not in gn_text


def test_difference_scripts_build_measurement_system_without_scaled_j_temp() -> None:
    for relpath in (
        "scripts/run_synthetic_parity.py",
        "scripts/benchmarks/benchmark_difference_runtime.py",
    ):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "_column_scaled_jjt" in text
        assert "jw_scaled =" not in text
        assert "inv_noser[None, :]" not in text


def test_difference_scripts_avoid_full_weighted_jacobian_temp() -> None:
    for relpath in (
        "scripts/run_synthetic_parity.py",
        "scripts/benchmarks/benchmark_difference_runtime.py",
    ):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "_row_weighted_column_sumsq" in text
        assert "_row_weighted_jtj" in text
        assert "_hash_row_scaled_array" in text
        assert "jacobian_weighted =" not in text
        assert "jacobian * sqrt_weights[:, None]" not in text


@pytest.mark.parametrize(
    "module_name",
    (
        "scripts.run_synthetic_parity",
        "scripts.benchmarks.benchmark_difference_runtime",
    ),
)
def test_difference_script_row_weighted_helpers_match_legacy_weighted_j(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)
    jacobian = np.arange(1, 21, dtype=np.float64).reshape(5, 4) / 10.0
    weights = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    sqrt_weights = np.sqrt(weights)
    weighted = jacobian * sqrt_weights[:, None]
    column_scale = np.array([1.0, 0.5, 2.0, 1.5], dtype=np.float64)
    vector = np.linspace(-0.2, 0.3, jacobian.shape[0], dtype=np.float64)

    np.testing.assert_allclose(
        module._row_weighted_column_sumsq(jacobian, weights, chunk_target_bytes=32),
        np.sum(weighted**2, axis=0),
    )
    lhs = module._column_scaled_jjt(jacobian, column_scale, chunk_target_bytes=32)
    lhs *= sqrt_weights[:, None]
    lhs *= sqrt_weights[None, :]
    np.testing.assert_allclose(lhs, weighted @ np.diag(column_scale) @ weighted.T)
    np.testing.assert_allclose(
        module._row_weighted_jtj(jacobian, weights, chunk_target_bytes=32),
        weighted.T @ weighted,
    )
    np.testing.assert_allclose(
        module._weighted_jt_vec(jacobian, vector, sqrt_weights),
        weighted.T @ vector,
    )

    expected_hash = hashlib.sha256()
    expected_hash.update(f"{np.dtype(np.float64)}:{weighted.shape}:".encode("utf-8"))
    expected_hash.update(np.ascontiguousarray(weighted, dtype=np.float64).tobytes())
    assert (
        module._hash_row_scaled_array(jacobian, sqrt_weights, chunk_target_bytes=32)
        == expected_hash.hexdigest()
    )


def test_3d_scripts_use_streaming_point_distance_helper() -> None:
    for relpath in (
        "scripts/benchmarks/benchmark_3d_runtime.py",
        "scripts/run_cem_16e_cylinder_3d_test.py",
        "scripts/diagnostics/render_3d_inverse_reconstruction_overview.py",
        "scripts/diagnostics/gallery_shared.py",
        "scripts/diagnostics/run_real_reconstruction_gallery_worker.py",
        "scripts/benchmarks/benchmark_p_refinement_forward.py",
    ):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "squared_distances_to_point" in text
        assert "center[None, :]" not in text
        assert "coords[:, :3] -" not in text
        assert "coords[:, :2] -" not in text


def test_v522_render_3d_overview_uses_in_place_masks_and_reductions() -> None:
    module = importlib.import_module(
        "scripts.diagnostics.render_3d_inverse_reconstruction_overview"
    )

    shape_source = inspect.getsource(module._compute_shape_metrics)
    payload_source = inspect.getsource(module._compute_regular_volume_payload)
    voxel_source = inspect.getsource(module._add_voxel_volume)
    run_source = inspect.getsource(module.run_case)

    assert "coords[mask]" not in shape_source
    assert "where=mask" in shape_source
    assert "np.where" not in payload_source
    assert "_apply_cylindrical_nan_mask_in_place" in payload_source
    assert "np.where" not in voxel_source
    assert "np.corrcoef" not in run_source
    assert "recon_sigma[target_mask]" not in run_source
    assert "recon_sigma[background_mask]" not in run_source
    assert "_mean_where(recon_sigma, target_mask)" in run_source

    volume = np.zeros((3, 3, 2), dtype=np.float64)
    module._apply_cylindrical_nan_mask_in_place(
        volume,
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        radius=1.0,
    )
    assert np.isfinite(volume[1, 1, 0])
    assert np.isnan(volume[0, 1, 0])
    assert np.isnan(volume[1, 2, 1])

    np.testing.assert_allclose(
        module._pearson_correlation(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
        ),
        1.0,
    )
    np.testing.assert_allclose(
        module._rmse(
            np.array([1.0, 3.0, 6.0]),
            np.array([1.0, 1.0, 3.0]),
        ),
        np.sqrt(13.0 / 3.0),
    )
    assert (
        module._mean_where(
            np.array([1.0, 2.0, 5.0]),
            np.array([True, False, True]),
        )
        == 3.0
    )

    metrics = module._compute_shape_metrics(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.5, 1.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        ),
        np.array([3.0, 4.0, 1.0], dtype=np.float64),
        threshold=2.0,
    )
    assert metrics["selected_count"] == 2
    np.testing.assert_allclose(metrics["extent_x"], 2.0)
    np.testing.assert_allclose(metrics["extent_y"], 0.5)
    np.testing.assert_allclose(metrics["extent_z"], 1.0)


def test_v539_render_3d_overview_direct_fills_wireframe_points() -> None:
    module = importlib.import_module(
        "scripts.diagnostics.render_3d_inverse_reconstruction_overview"
    )
    wire_source = inspect.getsource(module._build_cylinder_wireframe)
    marker_source = inspect.getsource(module._build_electrode_markers)
    assert "np.column_stack" not in wire_source
    assert "np.full_like" not in wire_source
    assert "np.column_stack" not in marker_source
    assert "_three_column_points" in wire_source
    assert "_three_column_points" in marker_source

    points = module._three_column_points(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        5.0,
    )
    np.testing.assert_allclose(
        points,
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 5.0]]),
    )
    markers = module._build_electrode_markers(
        n_elec=4,
        radius=1.0,
        height=2.0,
        z_center=0.0,
        electrode_level_fractions=(0.25, 0.75),
    )
    assert markers.shape == (4, 3)


def test_v523_common_script_correlation_avoids_corrcoef() -> None:
    module = importlib.import_module("scripts.common.array_metrics")
    absolute_source = (REPO_ROOT / "scripts/common/gn_absolute_runner.py").read_text(
        encoding="utf-8"
    )
    difference_source = (
        REPO_ROOT / "scripts/common/gn_difference_runner.py"
    ).read_text(encoding="utf-8")

    assert "np.corrcoef" not in absolute_source
    assert "np.corrcoef" not in difference_source
    assert "pearson_correlation(measured_vec, predicted_vec)" in absolute_source
    assert "pearson_correlation(meas_diff, pred_diff)" in difference_source

    np.testing.assert_allclose(
        module.pearson_correlation(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
        ),
        1.0,
    )
    np.testing.assert_allclose(
        module.pearson_correlation(
            np.array([1.0, 2.0, 3.0]),
            np.array([6.0, 4.0, 2.0]),
        ),
        -1.0,
    )
    assert np.isnan(
        module.pearson_correlation(
            np.array([1.0, 1.0]),
            np.array([2.0, 3.0]),
        )
    )


def test_v524_gallery_shared_metrics_use_reductions_without_mask_subsets() -> None:
    module = importlib.import_module("scripts.diagnostics.gallery_shared")
    pearson_source = inspect.getsource(module.safe_pearson)
    truth_metrics_source = inspect.getsource(module.truth_metrics)

    assert "np.corrcoef" not in pearson_source
    assert "pearson_correlation(left_arr, right_arr)" in pearson_source
    assert "truth[roi]" not in truth_metrics_source
    assert "recon[roi]" not in truth_metrics_source
    assert "recon[background_mask]" not in truth_metrics_source
    assert "background_mask &= ~roi" not in truth_metrics_source
    assert "np.logical_and(background_mask, roi, out=background_mask)" in (
        truth_metrics_source
    )

    metrics = module.truth_metrics(
        truth=np.array([2.0, 1.0, 1.0], dtype=np.float64),
        recon=np.array([1.8, 1.2, 1.0], dtype=np.float64),
        coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        anomalies=[{"center": (0.0, 0.0, 0.0), "radius": 0.25, "label": "hot"}],
        background_conductivity=1.0,
    )
    np.testing.assert_allclose(metrics["roi_mean_hot"], 1.8)
    np.testing.assert_allclose(metrics["background_bias"], 0.1)
    np.testing.assert_allclose(metrics["contrast_recovery_hot"], 0.7)


def test_v525_diagnostic_correlation_scripts_use_finite_reducer() -> None:
    checked_scripts = {
        "scripts/diagnostics/compare_linear_systems.py": (
            "finite_pearson_correlation(predicted, measured)"
        ),
        "scripts/diagnostics/fair_eidors_pyeidors_8e_compare.py": (
            "finite_pearson_correlation(a, b, min_count=3)"
        ),
        "scripts/diagnostics/eidors_forward_parity_gate.py": (
            "finite_pearson_correlation(a, b, min_count=3)"
        ),
        "scripts/diagnostics/compare_8e_16e_small_domain.py": (
            "safe_finite_pearson_correlation(left, right)"
        ),
        "scripts/diagnostics/scaled_boundary_voltage_circle_experiment.py": (
            "safe_finite_pearson_correlation(left, right)"
        ),
    }
    for relpath, expected_call in checked_scripts.items():
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert "np.corrcoef" not in text
        assert expected_call in text

    module = importlib.import_module("scripts.common.array_metrics")
    np.testing.assert_allclose(
        module.finite_pearson_correlation(
            np.array([1.0, np.nan, 2.0, 3.0]),
            np.array([2.0, 99.0, 4.0, 6.0]),
        ),
        1.0,
    )
    assert np.isnan(
        module.finite_pearson_correlation(
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 3.0, 4.0]),
        )
    )
    assert (
        module.safe_finite_pearson_correlation(
            np.array([1.0, 1.0, np.nan]),
            np.array([1.0, 1.0, 5.0]),
        )
        == 1.0
    )
    assert (
        module.safe_finite_pearson_correlation(
            np.array([1.0, 1.0, np.nan]),
            np.array([2.0, 2.0, 5.0]),
        )
        == 0.0
    )


def test_v540_fair_eidors_exports_direct_fill_measurement_blocks() -> None:
    module = importlib.import_module(
        "scripts.diagnostics.fair_eidors_pyeidors_8e_compare"
    )
    boundary_source = inspect.getsource(module.boundary_facets_3d)
    export_source = inspect.getsource(module.export_case)
    assert "np.vstack(rows)" not in boundary_source
    assert "np.concatenate([[0], np.cumsum(meas_counts[:-1])])" not in export_source
    assert "np.vstack(pm.meas_matrices)" not in export_source
    assert "_measurement_starts(meas_counts)" in export_source
    assert "_stack_measurement_matrices(pm.meas_matrices)" in export_source

    np.testing.assert_array_equal(
        module._measurement_starts(np.array([3, 4, 5], dtype=np.int64)),
        np.array([0, 3, 7], dtype=np.int64),
    )
    stacked = module._stack_measurement_matrices(
        [
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0], [5.0, 6.0]]),
        ]
    )
    np.testing.assert_allclose(
        stacked,
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )


def test_v541_eidors_forward_parity_gate_direct_fills_measurement_blocks() -> None:
    module = importlib.import_module("scripts.diagnostics.eidors_forward_parity_gate")
    verify_source = inspect.getsource(module.verify_pattern_manager)
    assert "np.vstack(manager.meas_matrices)" not in verify_source
    assert "_stack_measurement_matrices(manager.meas_matrices)" in verify_source

    stacked = module._stack_measurement_matrices(
        [
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[2.0, -2.0]]),
        ]
    )
    np.testing.assert_allclose(
        stacked,
        np.array([[1.0, 0.0], [0.0, 1.0], [2.0, -2.0]]),
    )


def test_v544_small_diagnostic_helpers_avoid_stack_concat_builders() -> None:
    points_module = importlib.import_module(
        "scripts.make_all_modes_16stim_point_status"
    )
    fair_module = importlib.import_module(
        "scripts.diagnostics.fair_eidors_pyeidors_8e_compare"
    )
    plot_text = (REPO_ROOT / "scripts/diagnostics/plot_electrode_tags.py").read_text(
        encoding="utf-8"
    )
    fair_render_source = inspect.getsource(fair_module.render_case)
    assert "np.column_stack" not in inspect.getsource(points_module._electrode_points)
    assert "np.vstack(segs)" not in plot_text
    assert 'np.concatenate([eidors["sigma"], cpu["sigma"], cuda["sigma"]])' not in (
        fair_render_source
    )
    assert "_concatenate_vectors_direct(" in fair_render_source

    points = points_module._electrode_points(4)
    np.testing.assert_allclose(
        points,
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [-1.0, 0.0],
            ]
        ),
        atol=1.0e-12,
    )
    combined = fair_module._concatenate_vectors_direct(
        [np.array([1.0, 2.0]), np.array([3.0])]
    )
    np.testing.assert_allclose(combined, np.array([1.0, 2.0, 3.0]))


def test_v526_small_domain_diagnostics_use_masked_mean_reducer() -> None:
    checked_scripts = {
        "scripts/diagnostics/compare_8e_16e_small_domain.py": (
            "mean_where(recon_delta, target_mask)",
            "mean_where(recon_delta, background_mask)",
        ),
        "scripts/diagnostics/scaled_boundary_voltage_circle_experiment.py": (
            "mean_where(recon_delta, target_mask)",
            "mean_where(recon_delta, background_mask)",
            "mean_where(original_delta, target_mask)",
            "mean_where(scaled_delta, target_mask)",
        ),
        "scripts/diagnostics/gallery_shared.py": (
            "mean_where(truth, roi",
            "mean_where(recon, roi",
            "mean_where(\n            recon,",
        ),
    }
    forbidden = (
        "np.mean(recon_delta[target_mask])",
        "np.mean(recon_delta[background_mask])",
        "np.mean(original_delta[target_mask])",
        "np.mean(scaled_delta[target_mask])",
        "np.mean(recon[background_mask])",
        "np.mean(recon[roi])",
        "np.mean(truth[roi])",
    )
    for relpath, expected_calls in checked_scripts.items():
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in text
        for expected_call in expected_calls:
            assert expected_call in text

    module = importlib.import_module("scripts.common.array_metrics")
    np.testing.assert_allclose(
        module.mean_where(
            np.array([1.0, 2.0, 5.0]),
            np.array([True, False, True]),
        ),
        3.0,
    )
    assert np.isnan(
        module.mean_where(
            np.array([1.0, 2.0, 5.0]),
            np.array([False, False, False]),
        )
    )


def test_v527_benchmark_difference_weights_clean_in_place() -> None:
    module = importlib.import_module("scripts.benchmarks.benchmark_difference_runtime")
    source = inspect.getsource(module.build_measurement_weights)

    assert "np.where(np.isfinite(weights), weights, 0.0)" not in source
    assert "np.square(weights, out=weights)" in source
    assert "np.nan_to_num(weights, copy=False" in source
    assert "np.maximum(weights, float(floor), out=weights)" in source

    weights = module.build_measurement_weights(
        baseline_vector=np.array([2.0, np.nan, np.inf], dtype=np.float64),
        diff_vector=np.array([0.5, 3.0, 4.0], dtype=np.float64),
        strategy="baseline",
        floor=0.25,
    )
    np.testing.assert_allclose(weights, np.array([4.0, 0.25, 0.25]))


def test_v534_benchmark_difference_system_diagonal_adds_in_place() -> None:
    module = importlib.import_module("scripts.benchmarks.benchmark_difference_runtime")
    text = (REPO_ROOT / "scripts/benchmarks/benchmark_difference_runtime.py").read_text(
        encoding="utf-8"
    )

    assert "(hp**2) * np.eye" not in text
    assert "RtR = np.diag" not in text
    assert "_add_diagonal_in_place(lhs, hp**2)" in text
    assert "_add_diagonal_in_place(lhs, (hp**2) * noser_diag)" in text

    system = np.zeros((2, 2), dtype=np.float64)
    out = module._add_diagonal_in_place(system, np.array([2.0, 4.0]))
    assert out is system
    np.testing.assert_allclose(np.diag(system), np.array([2.0, 4.0]))
    module._add_diagonal_in_place(system, 1.0)
    np.testing.assert_allclose(np.diag(system), np.array([3.0, 5.0]))


def test_v531_tank_realdata_holdout_script_uses_streaming_metrics() -> None:
    module = importlib.import_module("scripts.run_tank_realdata_holdout_compare")
    text = (REPO_ROOT / "scripts/run_tank_realdata_holdout_compare.py").read_text(
        encoding="utf-8"
    )

    assert "np.corrcoef" not in text
    assert "np.concatenate([field for _, field in fields])" not in text
    assert "fit_residual[holdout_indices]" not in text
    assert "np.vstack([vh, vi])" not in text
    assert "np.vstack([item.delta_sigma for item in variants])" not in text
    assert "np.vstack([item.pred_diff for item in variants])" not in text
    assert "np.vstack([item.fit_diff for item in variants])" not in text

    np.testing.assert_allclose(
        module._rmse_at_indices(
            np.array([1.0, -2.0, 3.0, 10.0]),
            np.array([0, 2], dtype=np.int64),
        ),
        np.sqrt(5.0),
    )
    np.testing.assert_allclose(
        module._max_abs_value(np.array([-1.0, 3.5, -2.0])),
        3.5,
    )
    np.testing.assert_allclose(
        module._stack_two_frames(
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )
    variants = [
        SimpleNamespace(name="a", pred_diff=np.array([1.0, 2.0])),
        SimpleNamespace(name="b", pred_diff=np.array([3.0, 4.0])),
    ]
    np.testing.assert_allclose(
        module._stack_variant_vectors(variants, "pred_diff"),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_v532_small_domain_grid_sampling_direct_fills_query_and_mask() -> None:
    for module_name in (
        "scripts.diagnostics.compare_8e_16e_small_domain",
        "scripts.diagnostics.scaled_boundary_voltage_circle_experiment",
    ):
        module = importlib.import_module(module_name)
        sample_source = inspect.getsource(module.sample_to_grid)
        assert "np.column_stack" not in sample_source
        assert "xg**2 + yg**2" not in sample_source
        assert "_query_points_2d(xg, yg)" in sample_source
        assert "_apply_outside_radius_nan(image, xg, yg, radius)" in sample_source

        query = module._query_points_2d(
            np.array([[0.0, 1.0], [2.0, 3.0]]),
            np.array([[4.0, 5.0], [6.0, 7.0]]),
        )
        np.testing.assert_allclose(
            query,
            np.array(
                [
                    [0.0, 4.0],
                    [1.0, 5.0],
                    [2.0, 6.0],
                    [3.0, 7.0],
                ]
            ),
        )

        image = np.arange(4, dtype=np.float64).reshape(2, 2)
        module._apply_outside_radius_nan(
            image,
            np.array([[0.0, 2.0], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            1.0,
        )
        assert np.isfinite(image[0, 0])
        assert np.isnan(image[0, 1])
        assert np.isnan(image[1, 0])
        assert np.isfinite(image[1, 1])


def test_v533_run_synthetic_parity_avoids_remaining_dense_temporaries() -> None:
    module = importlib.import_module("scripts.run_synthetic_parity")
    text = (REPO_ROOT / "scripts/run_synthetic_parity.py").read_text(encoding="utf-8")

    assert "np.corrcoef" not in text
    assert "np.where(np.isfinite(weights), weights, 0.0)" not in text
    assert "(hp**2) * np.eye" not in text
    assert "RtR = np.diag" not in text
    assert "np.column_stack" not in text

    source = inspect.getsource(module.build_measurement_weights)
    assert "np.square(weights, out=weights)" in source
    assert "np.nan_to_num(weights, copy=False" in source
    assert "np.maximum(weights, float(floor), out=weights)" in source

    weights = module.build_measurement_weights(
        baseline_vector=np.array([2.0, np.nan, np.inf], dtype=np.float64),
        diff_vector=np.array([0.5, 3.0, 4.0], dtype=np.float64),
        strategy="baseline",
        floor=0.25,
    )
    np.testing.assert_allclose(weights, np.array([4.0, 0.25, 0.25]))

    system = np.zeros((3, 3), dtype=np.float64)
    out = module._add_diagonal_in_place(system, np.array([1.0, 2.0, 3.0]))
    assert out is system
    np.testing.assert_allclose(np.diag(system), np.array([1.0, 2.0, 3.0]))
    module._add_diagonal_in_place(system, 0.5)
    np.testing.assert_allclose(np.diag(system), np.array([1.5, 2.5, 3.5]))

    np.testing.assert_allclose(
        module._three_column_matrix(
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ),
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
    )
