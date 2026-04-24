"""Contract tests for the travelling-wave prior benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "benchmark_prior_travelling_wave.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_prior_travelling_wave", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prior_travelling_wave_benchmark_reports_fidelity_and_signature_delta(
    tmp_path: Path,
) -> None:
    module = _load_module()

    payload = module.run_benchmark(
        n_cells=12,
        n_frames=8,
        n_measurements=9,
        lambda_=0.12,
        ridge=1.0e-8,
        seed=123,
    )
    output = module.write_payload(tmp_path / "prior_wave.json", payload)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["schema"] == module.SCHEMA
    assert saved["fixture"]["truth_shape"] == [8, 12]
    assert saved["fixture"]["measurement_shape"] == [8, 9]
    assert set(saved["methods"]) == {"laplace", "graph_ltl", "curvature", "tv_irls"}
    assert saved["summary"]["best_rmse_method"] in saved["methods"]
    assert saved["summary"]["signatures_distinct_from_laplace"]["graph_ltl"] is True
    assert saved["summary"]["signatures_distinct_from_laplace"]["curvature"] is True
    assert saved["summary"]["signatures_distinct_from_laplace"]["tv_irls"] is True
    assert saved["summary"]["matches_laplace_reconstruction"]["laplace"] is True
    assert saved["summary"]["matches_laplace_reconstruction"]["graph_ltl"] is True
    assert saved["summary"]["matches_laplace_reconstruction"]["curvature"] is True

    laplace = saved["methods"]["laplace"]
    graph_ltl = saved["methods"]["graph_ltl"]
    curvature = saved["methods"]["curvature"]
    tv_irls = saved["methods"]["tv_irls"]
    assert laplace["RtR_signature_hash"] != graph_ltl["RtR_signature_hash"]
    assert graph_ltl["RtR_signature_hash"] == curvature["RtR_signature_hash"]
    assert graph_ltl["matrix_delta_fro_vs_laplace"] == 0.0
    assert curvature["matrix_delta_fro_vs_laplace"] == 0.0
    assert tv_irls["RtR_signature_hash"] != laplace["RtR_signature_hash"]
    assert tv_irls["matrix_delta_fro_vs_laplace"] is None
    assert tv_irls["tv_irls_metadata"]["objective_monotone_all"] is True
    for method in saved["methods"].values():
        fidelity = method["fidelity"]
        assert np.isfinite(list(fidelity.values())).all()
        assert fidelity["rmse"] >= 0.0
        assert fidelity["center_rmse"] >= 0.0
        assert method["online_metadata"]["forward_solve_count"] == 0


def test_prior_travelling_wave_benchmark_validates_fixture_inputs() -> None:
    module = _load_module()
    for kwargs, message in (
        ({"n_cells": 3, "n_frames": 4, "n_measurements": 4}, "n_cells"),
        ({"n_cells": 4, "n_frames": 1, "n_measurements": 4}, "n_frames"),
        ({"n_cells": 4, "n_frames": 4, "n_measurements": 1}, "n_measurements"),
    ):
        try:
            module.build_travelling_wave_fixture(
                **kwargs,
                noise_std=0.0,
                seed=1,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - assertion clarity
            raise AssertionError(f"expected ValueError containing {message!r}")
