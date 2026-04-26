from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np

from pyeidors.inverse.greit import GREIT_METRIC_KEYS, calc_greit_rm


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = REPO_ROOT / "scripts" / "diagnostics" / "compare_greit_eidors_parity.py"


def _load_diagnostic():
    spec = importlib.util.spec_from_file_location(
        "compare_greit_eidors_parity", DIAGNOSTIC
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tiny_fixture(path: Path, *, rm_drift: float = 0.0) -> dict[str, np.ndarray]:
    vh = np.array([2.0, 4.0, 5.0], dtype=np.float64)
    y = np.array(
        [
            [0.10, -0.05],
            [0.00, 0.12],
            [-0.04, 0.08],
        ],
        dtype=np.float64,
    )
    vi = vh.reshape(-1, 1) * (1.0 + y)
    d = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.0],
            [0.0, 1.0],
            [0.0, 0.25],
        ],
        dtype=np.float64,
    )
    weight = 0.2
    components = calc_greit_rm(y, d, weight=weight, noise_covar=1.0)
    rm = components.rm.copy()
    if rm_drift:
        rm[0, 0] += float(rm_drift)
    rec_centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    payload = {
        "vh": vh,
        "vi": vi,
        "xyzr": np.array(
            [[0.25, 0.75], [0.25, 0.75], [0.0, 0.0], [0.2, 0.2]],
            dtype=np.float64,
        ),
        "D": d,
        "Y": y,
        "PJt": components.pjt,
        "M": components.m,
        "noiselev": np.asarray([components.noiselev], dtype=np.float64),
        "RM": rm,
        "weight": np.asarray([weight], dtype=np.float64),
        "rec_centers": rec_centers,
        "normalize": np.asarray([1], dtype=np.int64),
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "pyeidors-eidors-greit-fixture-v1"
        for name, value in payload.items():
            handle.create_dataset(name, data=value)
    return payload


def test_compare_greit_eidors_parity_report_passes_tiny_fixture(tmp_path) -> None:
    module = _load_diagnostic()
    fixture = tmp_path / "tiny_eidors_greit_fixture.mat"
    _write_tiny_fixture(fixture)
    report_path = tmp_path / "parity_report.json"

    report = module.compare_greit_eidors_parity(
        fixture,
        report_out=report_path,
        abs_tol=1.0e-11,
        rel_tol=1.0e-11,
    )

    assert report_path.exists()
    assert report["schema"] == module.REPORT_SCHEMA
    assert report["all_passed"] is True
    assert report["pyeidors_source"] == "computed_from_fixture_components"
    assert report["tolerances"]["Y"]["abs"] == 1.0e-11
    assert report["tolerances"]["RM@dv"]["rel"] == 1.0e-11
    comparison_names = {item["name"] for item in report["comparisons"]}
    assert set(module.DEFAULT_COMPONENTS) == comparison_names
    metric_result = next(
        item for item in report["comparisons"] if item["name"] == "metrics"
    )
    assert set(metric_result["shape_eidors"]) == set(GREIT_METRIC_KEYS)


def test_compare_greit_eidors_parity_report_flags_rm_drift(tmp_path) -> None:
    module = _load_diagnostic()
    fixture = tmp_path / "drifted_eidors_greit_fixture.mat"
    _write_tiny_fixture(fixture, rm_drift=0.05)

    report = module.compare_greit_eidors_parity(
        fixture,
        abs_tol=1.0e-12,
        rel_tol=1.0e-12,
    )

    assert report["all_passed"] is False
    rm = next(item for item in report["comparisons"] if item["name"] == "RM")
    recon = next(item for item in report["comparisons"] if item["name"] == "RM@dv")
    assert rm["passed"] is False
    assert recon["passed"] is False
    assert rm["max_abs_error"] > 0.0
    assert recon["relative_error"] > 0.0
