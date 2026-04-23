"""Tests for unified method dispatch registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = REPO_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from common.method_runners import (
    get_method_runner,
    run_gn_absolute_cases,
    run_gn_difference_cases,
    run_sparse_bayes_difference_cases,
)
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
