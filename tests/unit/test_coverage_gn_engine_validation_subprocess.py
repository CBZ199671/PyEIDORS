"""Subprocess-based tests for GN engine parameter validation (lines 216-279)."""

from __future__ import annotations


from tests.utils import run_python


def _validation_test(param_name, param_value, expected_match):
    """Helper to test a single validation error via subprocess."""
    code = f"""
from unittest import mock
fm = mock.MagicMock()
fm.n_elec = 4
fm.mesh.topology.dim = 2
try:
    from pyeidors.inverse.solvers.gauss_newton_engine import GaussNewtonReconstructor
    GaussNewtonReconstructor(fwd_model=fm, {param_name}={param_value!r})
    print("NO_ERROR")
except ValueError as e:
    if {expected_match!r} in str(e):
        print("PASS")
    else:
        print(f"WRONG: {{e}}")
except Exception as e:
    if {expected_match!r} in str(e):
        print("PASS")
    else:
        print(f"OTHER: {{e}}")
"""
    result = run_python(code)
    return "PASS" in result.stdout


class TestGNEngineValidationSubprocess:
    """Cover lines 216-279 via subprocess to avoid UFL issues."""

    def test_invalid_performance_mode(self):
        assert _validation_test("performance_mode", "bad", "performance_mode")

    def test_invalid_solver_mode(self):
        assert _validation_test("solver_mode", "bad", "solver_mode")

    def test_invalid_linear_solver(self):
        assert _validation_test("linear_solver", "bad", "linear_solver")

    def test_invalid_line_search_mode(self):
        assert _validation_test("line_search_mode", "bad", "line_search_mode")

    def test_invalid_preconditioner(self):
        assert _validation_test("preconditioner", "bad", "preconditioner")

    def test_invalid_fast_linear_path(self):
        assert _validation_test("fast_linear_path", "bad", "fast_linear_path")

    def test_invalid_rom_mode(self):
        assert _validation_test("rom_mode", "bad", "rom_mode")

    def test_invalid_rom_snapshot_source(self):
        assert _validation_test("rom_snapshot_source", "bad", "rom_snapshot_source")

    def test_invalid_inexact_mode(self):
        assert _validation_test("inexact_mode", "bad", "inexact_mode")

    def test_invalid_inexact_forcing(self):
        assert _validation_test("inexact_forcing", "bad", "inexact_forcing")

    def test_invalid_lowrank_mode(self):
        assert _validation_test("lowrank_mode", "bad", "lowrank_mode")

    def test_invalid_lowrank_method(self):
        assert _validation_test("lowrank_method", "bad", "lowrank_method")

    def test_invalid_lowrank_energy_zero(self):
        assert _validation_test("lowrank_energy", 0.0, "lowrank_energy")

    def test_invalid_lowrank_energy_high(self):
        assert _validation_test("lowrank_energy", 1.5, "lowrank_energy")

    def test_invalid_inexact_eta_negative(self):
        assert _validation_test("inexact_eta_min", -0.1, "inexact eta bounds")

    def test_invalid_eta_min_gt_max(self):
        code = """
from unittest import mock
fm = mock.MagicMock()
fm.n_elec = 4
fm.mesh.topology.dim = 2
try:
    from pyeidors.inverse.solvers.gauss_newton_engine import GaussNewtonReconstructor
    GaussNewtonReconstructor(fwd_model=fm, inexact_eta_min=0.9, inexact_eta_max=0.1)
    print("NO_ERROR")
except ValueError as e:
    if "inexact_eta_min" in str(e):
        print("PASS")
    else:
        print(f"WRONG: {e}")
except Exception as e:
    if "inexact_eta_min" in str(e):
        print("PASS")
    else:
        print(f"OTHER: {e}")
"""
        result = run_python(code)
        assert "PASS" in result.stdout

    def test_hyperparameter_path(self):
        """Line 144: hyperparameter init. Tested via eit_system in test_coverage_gn_engine_deep.py."""
        pass  # Covered by integration test

    def test_verbose_set_regularization(self):
        """Lines 522, 527. Tested via eit_system in test_coverage_gn_engine_deep.py."""
        pass  # Covered by integration test
