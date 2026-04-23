"""Tests for perf, physics, and geometry edge cases."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest


# --- perf/capabilities tests ---


class TestPerfCapabilities:
    """Cover lines in perf/capabilities.py."""

    def test_load_petsc_runtime_fails(self, monkeypatch):
        from pyeidors.perf import capabilities as mod

        monkeypatch.setattr(mod, "_load_petsc_runtime", lambda: None)
        assert mod._has_petsc_mat_solve() is False
        assert mod._has_petsc_gamg() is False

    def test_has_cuda_structured_fails(self, monkeypatch):
        from pyeidors.perf import capabilities as mod

        with mock.patch.dict(
            "sys.modules", {"pyeidors.forward.cuda_structured_backend": None}
        ):
            result = mod._has_cuda_structured()
        assert isinstance(result, bool)

    def test_has_pyamg_import_error(self, monkeypatch):

        with mock.patch.dict("sys.modules", {"pyamg": None}):
            # Force fresh check
            pass

    def test_has_cholmod_import_error(self, monkeypatch):

        with mock.patch.dict(
            "sys.modules", {"sksparse": None, "sksparse.cholmod": None}
        ):
            pass

    def test_select_preconditioner_pyamg_unavailable(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": False, "cholmod": False, "petsc_gamg": False}
        assert select_preconditioner("pyamg", capabilities=caps) == "diag"

    def test_select_preconditioner_cholmod_unavailable(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": False, "cholmod": False, "petsc_gamg": False}
        assert select_preconditioner("cholmod", capabilities=caps) == "diag"

    def test_select_preconditioner_petsc_gamg_unavailable(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": False, "cholmod": False, "petsc_gamg": False}
        assert select_preconditioner("petsc-gamg", capabilities=caps) == "diag"

    def test_select_preconditioner_auto_with_cholmod(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": True, "cholmod": True, "petsc_gamg": False}
        assert select_preconditioner("auto", capabilities=caps) == "cholmod"

    def test_select_preconditioner_auto_with_pyamg(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": True, "cholmod": False, "petsc_gamg": False}
        assert select_preconditioner("auto", capabilities=caps) == "pyamg"

    def test_select_preconditioner_auto_with_gamg(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": False, "cholmod": False, "petsc_gamg": True}
        assert select_preconditioner("auto", capabilities=caps) == "petsc-gamg"

    def test_select_preconditioner_auto_nothing(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": False, "cholmod": False, "petsc_gamg": False}
        assert select_preconditioner("auto", capabilities=caps) == "diag"

    def test_select_preconditioner_unknown_mode(self):
        from pyeidors.perf.capabilities import select_preconditioner

        caps = {"pyamg": False, "cholmod": False, "petsc_gamg": False}
        assert select_preconditioner("unknown", capabilities=caps) == "diag"

    def test_select_fast_linear_path_explicit(self):
        from pyeidors.perf.capabilities import select_fast_linear_path

        caps = {"cholmod": False}
        result = select_fast_linear_path(
            "woodbury",
            regularization_is_diagonal=False,
            regularization_is_sparse_spd=False,
            capabilities=caps,
        )
        assert result == "woodbury"

    def test_select_fast_linear_path_auto_diagonal(self):
        from pyeidors.perf.capabilities import select_fast_linear_path

        caps = {"cholmod": False}
        result = select_fast_linear_path(
            "auto",
            regularization_is_diagonal=True,
            regularization_is_sparse_spd=False,
            capabilities=caps,
        )
        assert result == "woodbury"

    def test_select_fast_linear_path_auto_spd_cholmod(self):
        from pyeidors.perf.capabilities import select_fast_linear_path

        caps = {"cholmod": True}
        result = select_fast_linear_path(
            "auto",
            regularization_is_diagonal=False,
            regularization_is_sparse_spd=True,
            capabilities=caps,
        )
        assert result == "pcg"

    def test_select_fast_linear_path_unknown(self):
        from pyeidors.perf.capabilities import select_fast_linear_path

        caps = {"cholmod": False}
        result = select_fast_linear_path(
            "unknown",
            regularization_is_diagonal=False,
            regularization_is_sparse_spd=False,
            capabilities=caps,
        )
        assert result == "pcg"

    def test_select_fused_strategy(self):
        from pyeidors.perf.capabilities import select_fused_strategy

        caps = {"cholmod": False, "pyamg": False, "petsc_gamg": False}
        result = select_fused_strategy(
            solver_mode="fast",
            mesh_dim=2,
            n_param=100,
            n_meas=50,
            rom_mode="off",
            inexact_mode="off",
            lowrank_mode="off",
            regularization_is_diagonal=True,
            capabilities=caps,
        )
        assert "enabled" in result

    def test_select_fused_strategy_rom_on(self):
        from pyeidors.perf.capabilities import select_fused_strategy

        caps = {"cholmod": True, "pyamg": False, "petsc_gamg": False}
        result = select_fused_strategy(
            solver_mode="fast",
            mesh_dim=3,  # Must be 3D for ROM to be enabled
            n_param=15000,
            n_meas=200,
            rom_mode="on",
            inexact_mode="on",
            lowrank_mode="on",
            regularization_is_diagonal=True,
            capabilities=caps,
        )
        assert "enabled" in result
        assert result["rom"] is True


# --- perf/policy tests ---


class TestPerfPolicy:
    """Cover lines 162-166, 179 in policy.py."""

    def test_parse_block_size_invalid(self):
        from pyeidors.perf.policy import parse_block_size_candidates

        with pytest.raises(ValueError, match="invalid block-size"):
            parse_block_size_candidates("abc")

    def test_parse_block_size_empty(self):
        from pyeidors.perf.policy import parse_block_size_candidates

        with pytest.raises(ValueError, match="at least one positive"):
            parse_block_size_candidates("")

    def test_parse_block_size_valid(self):
        from pyeidors.perf.policy import parse_block_size_candidates

        result = parse_block_size_candidates("10,5,20")
        assert result == [5, 10, 20]

    def test_parse_block_size_from_iterable(self):
        from pyeidors.perf.policy import parse_block_size_candidates

        result = parse_block_size_candidates([3, 7])
        assert result == [3, 7]


# --- physics/current_drive tests ---


class TestCurrentDrive:
    """Cover lines in physics/current_drive.py."""

    def test_invalid_drive_mode(self):
        from pyeidors.physics.current_drive import normalize_drive_mode

        with pytest.raises(ValueError, match="Unsupported drive_mode"):
            normalize_drive_mode("bad_mode")

    def test_positive_array_negative_value(self):
        from pyeidors.physics.current_drive import _as_positive_array

        with pytest.raises(ValueError, match="must be positive"):
            _as_positive_array([1.0, -0.5], n_elec=2, name="test")

    def test_resolve_electrode_lengths_zero_n_elec(self):
        from pyeidors.physics.current_drive import resolve_electrode_lengths_m

        with pytest.raises(ValueError, match="n_elec must be positive"):
            resolve_electrode_lengths_m(
                electrode_lengths_mesh=[1.0],
                geometry_scale_to_m=1.0,
                electrode_length_m_override=None,
                n_elec=0,
            )

    def test_resolve_electrode_lengths_negative_scale(self):
        from pyeidors.physics.current_drive import resolve_electrode_lengths_m

        with pytest.raises(ValueError, match="geometry_scale_to_m must be positive"):
            resolve_electrode_lengths_m(
                electrode_lengths_mesh=[1.0],
                geometry_scale_to_m=-1.0,
                electrode_length_m_override=None,
                n_elec=1,
            )

    def test_resolve_electrode_override_negative(self):
        from pyeidors.physics.current_drive import resolve_electrode_lengths_m

        with pytest.raises(ValueError, match="must be positive when scalar"):
            resolve_electrode_lengths_m(
                electrode_lengths_mesh=[1.0],
                geometry_scale_to_m=1.0,
                electrode_length_m_override=-0.5,
                n_elec=1,
            )

    def test_build_stim_currents_total_current(self):
        from pyeidors.physics.current_drive import build_stim_currents

        result = build_stim_currents(
            drive_mode="total_current",
            drive_value=1.0,
            inj_indices=[0, 1],
            inj_weights=[1.0, -1.0],
            electrode_lengths_m=None,
        )
        np.testing.assert_array_equal(result, [1.0, -1.0])

    def test_build_stim_currents_line_density_missing_lengths(self):
        from pyeidors.physics.current_drive import build_stim_currents

        with pytest.raises(ValueError, match="requires electrode_lengths_m"):
            build_stim_currents(
                drive_mode="line_current_density",
                drive_value=1.0,
                inj_indices=[0],
                inj_weights=[1.0],
                electrode_lengths_m=None,
            )

    def test_build_stim_currents_line_density_empty_lengths(self):
        from pyeidors.physics.current_drive import build_stim_currents

        with pytest.raises(ValueError, match="cannot be empty"):
            build_stim_currents(
                drive_mode="line_current_density",
                drive_value=1.0,
                inj_indices=[0],
                inj_weights=[1.0],
                electrode_lengths_m=[],
            )

    def test_build_stim_currents_line_density_out_of_range(self):
        from pyeidors.physics.current_drive import build_stim_currents

        with pytest.raises(ValueError, match="out of range"):
            build_stim_currents(
                drive_mode="line_current_density",
                drive_value=1.0,
                inj_indices=[5],
                inj_weights=[1.0],
                electrode_lengths_m=[0.1, 0.1],
            )

    def test_build_stim_currents_line_density_zero_length(self):
        from pyeidors.physics.current_drive import build_stim_currents

        with pytest.raises(ValueError, match="must be positive"):
            build_stim_currents(
                drive_mode="line_current_density",
                drive_value=1.0,
                inj_indices=[0],
                inj_weights=[1.0],
                electrode_lengths_m=[0.0],
            )

    def test_build_stim_currents_index_weight_mismatch(self):
        from pyeidors.physics.current_drive import build_stim_currents

        with pytest.raises(ValueError, match="length mismatch"):
            build_stim_currents(
                drive_mode="total_current",
                drive_value=1.0,
                inj_indices=[0, 1],
                inj_weights=[1.0],
                electrode_lengths_m=None,
            )


# --- geometry tests ---


class TestMeshConverter:
    """Cover line 28 in mesh_converter.py."""

    def test_invalid_gdim(self):
        from pyeidors.geometry.mesh_converter import MeshConverter

        with pytest.raises(ValueError, match="gdim must be 2 or 3"):
            MeshConverter(mesh_file="test.msh", output_dir="/tmp", gdim=4)


# --- regularization tests ---


class TestBaseRegularization:
    """Cover lines 27, 44 in base_regularization.py."""

    def test_as_linear_operator_sparse(self):
        from scipy.sparse import eye
        from pyeidors.inverse.regularization.base_regularization import (
            BaseRegularization,
        )

        mat = eye(5, format="csr")
        op = BaseRegularization.as_linear_operator(mat, shape=(5, 5))
        result = op.matvec(np.ones(5))
        np.testing.assert_array_equal(result, np.ones(5))

    def test_as_linear_operator_dense(self):
        from pyeidors.inverse.regularization.base_regularization import (
            BaseRegularization,
        )

        mat = np.eye(4)
        op = BaseRegularization.as_linear_operator(mat, shape=(4, 4))
        result = op.matvec(np.ones(4))
        np.testing.assert_array_equal(result, np.ones(4))

    def test_as_linear_operator_shape_mismatch(self):
        from pyeidors.inverse.regularization.base_regularization import (
            BaseRegularization,
        )

        mat = np.eye(3)
        with pytest.raises(ValueError, match="shape"):
            BaseRegularization.as_linear_operator(mat, shape=(4, 4))
