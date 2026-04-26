from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "diagnostics"
SOURCE_MAP = SCRIPT_DIR / "eidors_greit_source_map.json"
MATLAB_SCRIPT = SCRIPT_DIR / "capture_eidors_greit_fixture.m"
VALIDATOR = SCRIPT_DIR / "eidors_greit_fixture.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("eidors_greit_fixture", VALIDATOR)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_eidors_greit_source_map_covers_official_pipeline() -> None:
    module = _load_validator()
    payload = module.load_source_map(SOURCE_MAP)

    module.validate_source_map(payload)

    assert payload["schema"] == module.SOURCE_MAP_SCHEMA
    assert set(module.REQUIRED_EXPORTS).issubset(payload["required_exports"])
    official_ids = {entry["id"] for entry in payload["official_functions"]}
    assert set(module.REQUIRED_OFFICIAL_FUNCTIONS).issubset(official_ids)
    case_ids = {case["case_id"] for case in payload["fixture_cases"]}
    assert set(module.REQUIRED_CASE_IDS).issubset(case_ids)
    assert payload["parity_status"]["current_pyeidors_greit"] == (
        "eidors-component-path-t40-t50-complete; "
        "linearized-rm-v0-explicit-non-parity-mode"
    )
    assert payload["parity_status"]["eidors_complete_claim_allowed"] is False
    assert payload["parity_status"]["official_equivalence_claim_allowed"] is True
    assert (
        "n_measurements=2160" in payload["parity_status"]["official_equivalence_scope"]
    )
    assert "T49" in payload["parity_status"]["completed_tasks"]
    assert (
        "external MATLAB/EIDORS 48e fixture captured"
        in (payload["parity_status"]["completed_claim_gates"])
    )
    assert (
        "separate 5936 measurement protocol MATLAB/EIDORS fixture captured"
        in (payload["parity_status"]["remaining_claim_gates"])
    )


def test_matlab_capture_script_exports_required_component_fields() -> None:
    text = MATLAB_SCRIPT.read_text(encoding="utf-8")

    for token in (
        "GREIT3D_distribution",
        "mk_GREIT_model",
        "simulate_movement",
        "calc_GREIT_RM",
        "bsxfun(@rdivide, vi, vh) - 1",
        "eidors_default",
        "GREIT_desired_img",
        "-v7.3",
        "tiny_3d_cylinder",
        "reduced_48e_5936",
        "scaled_ring_z_levels(cases(2).cyl_shape(1), [0.15, 0.50, 0.85])",
        "cases(2).pattern_n_elec = 48",
        "cases(2).pattern_n_rings = 1",
    ):
        assert token in text
    for field in (
        "payload.vh",
        "payload.vi",
        "payload.xyzr",
        "payload.D",
        "payload.Y",
        "payload.PJt",
        "payload.M",
        "payload.noiselev",
        "payload.RM",
        "payload.weight",
        "payload.rec_model",
        "payload.normalize",
    ):
        assert field in text


def test_fixture_validator_accepts_v7p3_hdf5_required_fields(tmp_path: Path) -> None:
    module = _load_validator()
    fixture = tmp_path / "tiny_3d_cylinder_eidors_greit_fixture.mat"
    with h5py.File(fixture, "w") as handle:
        handle.attrs["schema"] = module.FIXTURE_SCHEMA
        handle.create_dataset("vh", data=np.ones(4))
        handle.create_dataset("vi", data=np.ones((4, 2)))
        handle.create_dataset("xyzr", data=np.ones((4, 2)))
        handle.create_dataset("D", data=np.ones((3, 2)))
        handle.create_dataset("Y", data=np.ones((4, 2)))
        handle.create_dataset("PJt", data=np.ones((3, 4)))
        handle.create_dataset("M", data=np.eye(4))
        handle.create_dataset("noiselev", data=np.array([0.1]))
        handle.create_dataset("RM", data=np.ones((3, 4)))
        handle.create_dataset("weight", data=np.array([0.02]))

    summary = module.validate_fixture_hdf5(fixture)

    assert summary["schema"] == module.FIXTURE_SCHEMA
    assert summary["shapes"]["RM"] == (3, 4)
    assert summary["shapes"]["xyzr"] == (4, 2)


def test_fixture_validator_rejects_missing_required_field(tmp_path: Path) -> None:
    module = _load_validator()
    fixture = tmp_path / "broken_fixture.mat"
    with h5py.File(fixture, "w") as handle:
        handle.attrs["schema"] = module.FIXTURE_SCHEMA
        for name in module.REQUIRED_EXPORTS:
            if name != "PJt":
                handle.create_dataset(name, data=np.ones(1))

    with pytest.raises(ValueError, match="PJt"):
        module.validate_fixture_hdf5(fixture)
