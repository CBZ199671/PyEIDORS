"""Forward-parity gate for EIDORS-exported 3D protocol data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pyeidors.electrodes.patterns import StimMeasPatternManager

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "results" / "eidors_same_pyeidors_mesh"
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "eidors_forward_parity_gate.py"
REQUIRED_DATA = (
    DATA_DIR / "pyeidors_same_tetra_mesh.mat",
    DATA_DIR / "same_mesh_vh_background.csv",
    DATA_DIR / "same_mesh_vi_sphere.csv",
    DATA_DIR / "same_mesh_dv_measured_normalized.csv",
)

SCRIPTS_PATH = REPO_ROOT / "scripts" / "diagnostics"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

import eidors_forward_parity_gate as parity


def _require_eidors_fixture() -> None:
    missing = [str(path) for path in REQUIRED_DATA if not path.exists()]
    if missing:
        pytest.skip("requires EIDORS same-mesh fixture files: " + ", ".join(missing))


def test_eidors_exported_stim_meas_payload_replays_exactly():
    _require_eidors_fixture()
    payload = loadmat(DATA_DIR / "pyeidors_same_tetra_mesh.mat")
    stim, _, meas_concat, starts, counts = parity.payload_measurement_matrices(payload)
    manager = StimMeasPatternManager(parity.build_custom_pattern(payload), mesh_tdim=3)

    actual_concat = np.vstack(manager.meas_matrices)
    assert np.allclose(manager.stim_matrix, stim)
    assert np.allclose(actual_concat, meas_concat)
    assert np.array_equal(np.asarray(manager.n_meas_per_stim, dtype=np.int64), counts)
    assert manager.n_stim == stim.shape[0]
    assert manager.n_meas_total == meas_concat.shape[0]
    assert int(starts[0]) == 0
    assert int(starts[-1] + counts[-1]) == meas_concat.shape[0]


@pytest.mark.slow
@pytest.mark.fenicsx
def test_eidors_forward_parity_gate_corr_gt_099(tmp_path: Path):
    _require_eidors_fixture()
    if os.environ.get("PYEIDORS_RUN_EIDORS_FORWARD_PARITY") != "1":
        pytest.skip("set PYEIDORS_RUN_EIDORS_FORWARD_PARITY=1 to run the slow gate")

    env = dict(os.environ)
    env.update(
        {
            "PYEIDORS_PARITY_DATA_DIR": str(DATA_DIR),
            "PYEIDORS_PARITY_OUTPUT_DIR": str(tmp_path),
            "PYEIDORS_PARITY_DEVICE": "cpu",
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr

    metrics = json.loads(
        (tmp_path / "forward_parity_exact_protocol_metrics_cpu.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["pattern_exact"] is True
    assert metrics["gate_status"] == "PASS"
    assert metrics["target_minus_reference"]["corr"] > 0.99
