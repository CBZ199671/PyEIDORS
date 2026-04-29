from __future__ import annotations

import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_top_level_pyeidors_facade_stays_narrow() -> None:
    import pyeidors

    assert pyeidors.__all__ == ["EITSystem", "check_environment", "__version__"]
    assert "EITSystem" in dir(pyeidors)
    assert callable(pyeidors.check_environment)

    for name in (
        "EITForwardModel",
        "LinearBackendConfig",
        "GaussNewtonReconstructor",
        "DirectJacobianCalculator",
    ):
        with pytest.raises(AttributeError):
            getattr(pyeidors, name)


def test_claimed_subpackage_exports_are_declared() -> None:
    forward = importlib.import_module("pyeidors.forward")
    inverse = importlib.import_module("pyeidors.inverse")
    jacobian = importlib.import_module("pyeidors.inverse.jacobian")

    assert {"EITForwardModel", "LinearBackendConfig"}.issubset(forward.__all__)
    assert {
        "GaussNewtonReconstructor",
        "assemble_sigma_contact_normal_system",
        "build_sigma_contact_block_metadata",
        "build_electrode_movement_jacobian",
        "configure_petsc_fieldsplit_solver",
        "prior_movement",
        "solve_sigma_contact_fieldsplit",
    }.issubset(inverse.__all__)
    assert {
        "DirectJacobianCalculator",
        "JacobianLinearization",
        "compute_sigma_fingerprint",
    }.issubset(jacobian.__all__)


def test_repository_root_cmd_wrappers_delegate_to_supported_gui_launcher() -> None:
    wrappers = {
        "EIT-GUI-CPU.cmd": "cpu",
        "EIT-GUI-GPU.cmd": "gpu",
    }

    for filename, profile in wrappers.items():
        path = REPO_ROOT / filename
        text = path.read_text(encoding="utf-8")

        assert path.exists()
        assert r"%~dp0scripts\gui\run_eit_app.ps1" in text
        assert f"-Profile {profile}" in text
        assert "%*" in text
        assert "exit /b %EXIT_CODE%" in text
