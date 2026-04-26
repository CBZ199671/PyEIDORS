"""Contract tests for the T49 48e/5936 GREIT parity benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from pyeidors.inverse import GREIT_EIDORS_HDF5_SCHEMA, calc_greit_rm
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "benchmark_greit_eidors_parity_48e.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_greit_t49", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_t49_woodbury_large_path_matches_dense_calc_greit_rm() -> None:
    module = _load_module()
    rng = np.random.default_rng(20260426)
    y = rng.normal(size=(7, 3))
    d = rng.normal(size=(5, 3))
    weight = 0.13

    fast = module._calc_large_scalar_noise_components(y, d, weight=weight)
    dense = calc_greit_rm(y, d, weight=weight, noise_covar=1.0)

    np.testing.assert_allclose(fast["PJt"], dense.pjt, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(fast["M"], dense.m, rtol=1.0e-11, atol=1.0e-11)
    np.testing.assert_allclose(
        fast["noiselev"], dense.noiselev, rtol=1.0e-11, atol=1.0e-11
    )
    np.testing.assert_allclose(fast["RM"], dense.rm, rtol=1.0e-10, atol=1.0e-10)


def test_t49_benchmark_writes_parity_report_and_hot_path_summary(
    tmp_path: Path,
) -> None:
    module = _load_module()

    payload = module.run_benchmark(
        output_dir=tmp_path / "bench",
        n_measurements=16,
        voxel_shape=(2, 2, 1),
        n_frames=4,
        n_elec=8,
        n_rings=2,
        target_radius=0.2,
        target_contrast=0.05,
        devices=("cpu",),
        cases=("nominal", "bad_weighted"),
    )

    assert payload["schema"] == module.REPORT_SCHEMA
    assert payload["scope"] == "48e/5936 EIDORS-parity GREIT RM benchmark"
    assert payload["config"]["n_measurements"] == 16
    assert payload["config"]["n_frames"] == 4
    assert "common_config_reference" in payload["config"]
    assert "surrogate_runtime_config_reference" not in payload["config"]
    assert payload["gate"]["parity_components_passed"] is True
    assert payload["gate"]["official_equivalence_claim_allowed"] is False
    assert payload["invariants"]["V55_target_distribution"] is True
    assert payload["invariants"]["V64_online_hot_path"] is True
    assert set(payload["cases"]) == {"nominal", "bad_weighted"}

    weighted = payload["cases"]["bad_weighted"]
    assert weighted["measurement_contract"]["bad_channel_count"] > 0
    assert weighted["measurement_contract"]["measurement_weight_kind"] == "diagonal"
    assert weighted["offline_counts"]["forward_solve_count"] == 1
    assert weighted["offline_counts"]["jacobian_rebuild_count"] == 0
    assert weighted["parity_report"]["all_passed"] is True
    assert set(weighted["parity_report"]["comparison_names"]) == {
        "Y",
        "D",
        "PJt",
        "M",
        "noiselev",
        "RM",
        "RM@dv",
        "metrics",
    }

    cpu = weighted["online_apply"]["cpu"]
    assert cpu["apply_batch_n_frames"] == 4
    assert cpu["metadata_batch"]["online_hot_path"] == "rm_matmul"
    assert cpu["metadata_batch"]["forward_solve_count"] == 0
    assert cpu["metadata_batch"]["jacobian_rebuild_count"] == 0
    assert cpu["metadata_batch"]["ksp_solve_count"] == 0
    assert cpu["metadata_batch"]["rm_prepare_mode"] == "reused_handle"

    summary_path = tmp_path / "bench" / "summary.json"
    report_path = tmp_path / "bench" / "README.md"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["schema"] == (
        module.REPORT_SCHEMA
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "48e/5936 EIDORS-Parity GREIT Runtime Gate" in report_text
    assert "- scope: `48e/5936 EIDORS-parity GREIT RM benchmark`" in report_text

    artifact = read_hdf5_artifact(
        tmp_path / "bench" / "bad_weighted_greit_eidors_rm.h5"
    )
    assert artifact.schema == GREIT_EIDORS_HDF5_SCHEMA
    assert artifact.metadata["artifact_schema"] == GREIT_EIDORS_HDF5_SCHEMA
    assert artifact.metadata["large_cache"] is True
    assert artifact.metadata["checksum_algorithm"] == "sha256"
    assert artifact.metadata["online_hot_path"] == "rm_matmul"
    assert artifact.metadata["cache_signature_schema"]
    for name in ("RM", "PJt", "M", "Sn", "vh", "vi", "xyzr", "D", "Y", "rec_model"):
        assert name in artifact.arrays
