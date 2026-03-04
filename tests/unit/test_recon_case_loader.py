"""Tests for unified case discovery and frame loading utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common.case_discovery import build_cases, collect_csv_files
from common.case_loader import load_frame_csv
from common.recon_cli_models import InputMode


def test_collect_csv_files_deduplicates_and_filters(tmp_path: Path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b_AD.csv"
    c = tmp_path / "c.txt"
    a.write_text("1,2,3,4\n", encoding="utf-8")
    b.write_text("1,2,3,4\n", encoding="utf-8")
    c.write_text("x", encoding="utf-8")

    files = collect_csv_files(
        input_dir=tmp_path,
        glob_pattern="*",
        csv_files=[a, a],
        include_ad=False,
    )
    assert files == [a]


def test_build_cases_frame_with_reference_index(tmp_path: Path):
    ref = tmp_path / "ref.csv"
    tar1 = tmp_path / "tar1.csv"
    tar2 = tmp_path / "tar2.csv"
    for path in [ref, tar1, tar2]:
        path.write_text("1\n2\n", encoding="utf-8")

    cases = build_cases(
        input_mode=InputMode.FRAME,
        input_files=[ref, tar1, tar2],
        require_reference=True,
        reference_csv=None,
        reference_index=0,
    )
    assert [c.case_name for c in cases] == ["tar1", "tar2"]
    assert all(c.reference_csv == ref for c in cases)


def test_load_frame_csv_complex_two_column(tmp_path: Path):
    csv_path = tmp_path / "complex.csv"
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.savetxt(csv_path, data, delimiter=",")

    frame_real = load_frame_csv(
        csv_path,
        measurement_gain=1.0,
        layout="auto",
        use_part="real",
    )
    frame_mag = load_frame_csv(
        csv_path,
        measurement_gain=1.0,
        layout="auto",
        use_part="mag",
    )

    assert np.allclose(frame_real, np.array([1.0, 3.0]))
    assert np.allclose(frame_mag, np.array([np.hypot(1.0, 2.0), np.hypot(3.0, 4.0)]))


def test_load_frame_csv_stim_meas_layout(tmp_path: Path):
    csv_path = tmp_path / "frame_matrix.csv"
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.savetxt(csv_path, matrix, delimiter=",")

    frame = load_frame_csv(
        csv_path,
        measurement_gain=2.0,
        layout="stim-meas",
        use_part="real",
        expected_len=4,
        n_stim=2,
        n_meas_per_stim=2,
    )
    assert np.allclose(frame, np.array([0.5, 1.0, 1.5, 2.0]))
