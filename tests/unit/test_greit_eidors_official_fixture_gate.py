"""Official-fixture gate tests for the T49 GREIT benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from pyeidors.inverse import calc_greit_rm
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact
from scripts.diagnostics.eidors_greit_fixture import FIXTURE_SCHEMA


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "run_greit_eidors_official_fixture_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("official_greit_gate", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_official_fixture_gate_blocks_without_real_fixture(tmp_path: Path) -> None:
    module = _load_module()
    output_dir = tmp_path / "gate"
    missing_fixture = tmp_path / "missing_reduced_48e_fixture.mat"

    payload = module.run_official_fixture_gate(
        output_dir=output_dir,
        fixture=missing_fixture,
        devices=("cpu",),
    )

    assert payload["status"] == "blocked"
    assert payload["blocked_reason"] == "missing_official_fixture"
    assert payload["official_eidors_fixture"] is False
    assert payload["official_equivalence_claim_allowed"] is False
    assert payload["t49_rerun_status"] == "not_run"
    assert "capture_eidors_greit_fixture" in payload["capture"]["matlab_batch_command"]

    summary = json.loads(
        (output_dir / "official_gate_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "blocked"
    assert "missing_official_fixture" in (output_dir / "README.md").read_text(
        encoding="utf-8"
    )


def test_official_fixture_gate_reruns_t49_with_fixture(tmp_path: Path) -> None:
    module = _load_module()
    fixture = tmp_path / "reduced_48e_5936_eidors_greit_fixture.mat"
    _write_tiny_official_fixture(fixture)

    payload = module.run_official_fixture_gate(
        output_dir=tmp_path / "gate",
        fixture=fixture,
        case_id="reduced_48e_5936",
        n_frames=3,
        voxel_shape=(3, 1, 1),
        devices=("cpu",),
    )

    assert payload["status"] == "passed"
    assert payload["official_fixture_available"] is True
    assert payload["official_eidors_fixture"] is True
    assert payload["official_equivalence_claim_allowed"] is True
    assert payload["benchmark_gate"]["official_eidors_fixture"] is True
    assert payload["benchmark_gate"]["official_equivalence_claim_allowed"] is True
    assert payload["computed_from_fixture_parity_passed"] is True
    assert set(payload["computed_from_fixture_comparison_names"]) == {
        "Y",
        "D",
        "PJt",
        "M",
        "noiselev",
        "RM",
        "RM@dv",
        "metrics",
    }

    t49_summary = json.loads(
        Path(payload["t49_summary_path"]).read_text(encoding="utf-8")
    )
    assert t49_summary["config"]["n_measurements"] == 4
    case = t49_summary["cases"]["reduced_48e_5936"]
    assert case["official_eidors_fixture"] is True
    artifact = read_hdf5_artifact(case["greit_artifact_path"])
    assert "official-eidors-fixture" in artifact.metadata["fwd_model_signature"]
    assert "surrogate" not in artifact.metadata["fwd_model_signature"]


def _write_tiny_official_fixture(path: Path) -> None:
    vh = np.asarray([1.0, 1.1, 0.9, 1.2], dtype=np.float64)
    y = np.asarray(
        [
            [0.03, -0.02],
            [0.01, 0.04],
            [-0.025, 0.015],
            [0.02, -0.01],
        ],
        dtype=np.float64,
    )
    vi = vh.reshape(-1, 1) * (1.0 + y)
    d = np.asarray(
        [
            [1.0, 0.2],
            [0.4, 0.8],
            [0.1, 1.0],
        ],
        dtype=np.float64,
    )
    weight = 0.02
    components = calc_greit_rm(y, d, weight=weight, noise_covar=1.0)
    rec_model = np.asarray(
        [
            [-0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        # MATLAB v7.3 commonly stores char arrays as integer code units.
        handle.create_dataset(
            "schema",
            data=np.asarray([ord(ch) for ch in FIXTURE_SCHEMA], dtype=np.uint16),
        )
        handle.create_dataset("vh", data=vh)
        handle.create_dataset("vi", data=vi.T)
        handle.create_dataset("xyzr", data=np.ones((4, y.shape[1]), dtype=np.float64).T)
        _write_matlab_sparse_group(handle, "D", d)
        handle.create_dataset("Y", data=y.T)
        handle.create_dataset("PJt", data=components.pjt.T)
        handle.create_dataset("M", data=components.m)
        handle.create_dataset(
            "noiselev", data=np.asarray([components.noiselev], dtype=np.float64)
        )
        handle.create_dataset("RM", data=components.rm.T)
        handle.create_dataset("weight", data=np.asarray([weight], dtype=np.float64))
        handle.create_dataset("Sn", data=np.eye(vh.size, dtype=np.float64))
        handle.create_dataset("rec_model", data=rec_model)
        handle.create_dataset("normalize", data=np.asarray([1], dtype=np.int64))


def _write_matlab_sparse_group(
    handle: h5py.File, name: str, values: np.ndarray
) -> None:
    values = np.asarray(values, dtype=np.float64)
    group = handle.create_group(name)
    group.attrs["MATLAB_class"] = np.bytes_("double")
    group.attrs["MATLAB_sparse"] = np.uint64(values.shape[0])
    ir = []
    data = []
    jc = [0]
    for col in range(values.shape[1]):
        column = values[:, col]
        nz = np.flatnonzero(column)
        ir.extend(int(idx) for idx in nz)
        data.extend(float(column[idx]) for idx in nz)
        jc.append(len(ir))
    group.create_dataset("ir", data=np.asarray(ir, dtype=np.uint64))
    group.create_dataset("jc", data=np.asarray(jc, dtype=np.uint64))
    group.create_dataset("data", data=np.asarray(data, dtype=np.float64))
