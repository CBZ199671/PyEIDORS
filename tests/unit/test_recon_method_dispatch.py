"""Tests for unified method dispatch registry."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common.method_runners import (
    _stack_frame_rows,
    get_method_runner,
    run_gn_absolute_cases,
    run_gn_difference_cases,
    run_sparse_bayes_difference_cases,
)
import common.method_runners as method_runners
from common.recon_cli_models import ReconstructionMethod


def test_dispatch_maps_methods_to_expected_runners():
    assert get_method_runner(ReconstructionMethod.GN_ABSOLUTE) is run_gn_absolute_cases
    assert (
        get_method_runner(ReconstructionMethod.GN_DIFFERENCE) is run_gn_difference_cases
    )
    assert (
        get_method_runner(ReconstructionMethod.SPARSE_BAYES)
        is run_sparse_bayes_difference_cases
    )


def test_dispatch_rejects_unknown_method():
    with pytest.raises(ValueError):
        get_method_runner("unknown")  # type: ignore[arg-type]


def test_v537_method_runners_stack_frames_by_direct_fill():
    source = inspect.getsource(method_runners)
    assert "np.vstack([vh, vi])" not in source
    assert "np.vstack([raw_measurements[:, c] for c in unique_cols])" not in source
    assert "np.vstack([ref_frame, target_frame])" not in source

    stacked = _stack_frame_rows(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
    )
    np.testing.assert_allclose(
        stacked,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
    )
    with pytest.raises(ValueError, match="measurements"):
        _stack_frame_rows(np.array([1.0]), np.array([2.0, 3.0]))
