"""T77 phase 1 contract gate: GN engine + runtime invariants.

T77 is the highest-risk Path C row in the code-fusion tier (3694-line
``gauss_newton_runtime.py`` + 591-line ``gauss_newton_engine.py`` +
``matrix_free_gn`` / ``line_search`` / ``weights`` / ``device``
helpers). The SPEC mandates a V73-style characterization gate before
any source movement: lock the literal values + function signatures
that downstream consumers (cache disk artifacts, diagnostic dumps,
benchmark JSON readers, GUI status panels) parse, so a sub-module
split commit cannot silently rename a string and break tools we have
no test for.

What this test freezes (phase 1):

* V73 sign pairing — ``rhs = -jtr`` literal still appears in the
  runtime contract source. Combined with ``DirectJacobianCalculator``
  ``sign=+1.0`` this is what produces the EIDORS-direction δσ.
* V11 ``matrix_free_pc_source`` enum literals — the strings
  ``dense-sensitivity`` / ``explicit`` / ``matrix_free_hessian_diag``
  / ``hessian_diag`` / ``noser`` / ``prior`` / ``auto_linearization_diag``
  / ``pmat`` / ``coarse-pmat`` / ``custom-pcshell`` / ``identity``
  appear as string literals so the diagnostic surface is bytewise
  stable. T77 phase 2 may host these literals in extracted companions
  while keeping the runtime import path stable.
* V12 ``matrix_free_ksp_backend_fallback_reason`` literals —
  ``petsc_backend_unavailable`` etc.
* V10 ``matrix_free_pc`` fallback reason
  ``petsc_gamg_not_supported_in_matrix_free``.
* V14 forward PC refresh reason ``direct_factor_requires_rebuild``.
* ``_IterationLog`` field set + ``to_payload`` keys (cache disk
  artifacts read these).
* ``_JacobianActionBundle`` field set.
* ``run_reconstruction`` public entry signature.
* The four matrix-free PETSc context classes and their construction
  helpers exist (sub-module split must keep them importable from
  ``gauss_newton_runtime``).

When phase 2 lands (sub-module split + reconstructor wrapper +
dead-helper deletion), every assertion in this file MUST stay green.
The strings live in the SPEC §V invariants; this test is the
mechanical wall guarding them.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SOURCE = (
    REPO_ROOT / "src" / "pyeidors" / "inverse" / "solvers" / "gauss_newton_runtime.py"
).read_text(encoding="utf-8")
LINEAR_SYSTEM_SOURCE = (
    REPO_ROOT
    / "src"
    / "pyeidors"
    / "inverse"
    / "solvers"
    / "gauss_newton_linear_system.py"
).read_text(encoding="utf-8")
RUNTIME_CONTRACT_SOURCE = f"{RUNTIME_SOURCE}\n{LINEAR_SYSTEM_SOURCE}"


# ---------------------------------------------------------------------------
# V73 — ``rhs = -jtr`` sign pairing literal MUST stay in runtime contract source.
# ---------------------------------------------------------------------------


def test_v73_rhs_minus_jtr_literal_present_in_runtime_contract_source() -> None:
    """V73: matrix-free fast-PCG builds ``rhs = -jtr`` to honor Direct sign."""
    assert "rhs = -jtr" in RUNTIME_CONTRACT_SOURCE, (
        "V73 contract: gauss_newton_runtime or its linear-system companion must "
        "build the matrix-free fast-PCG right-hand side as ``rhs = -jtr`` so "
        "DirectJacobianCalculator "
        "(sign=+1.0) yields physical δσ in the EIDORS direction."
    )


# ---------------------------------------------------------------------------
# V11 — matrix_free_pc_source enum literal stability.
# ---------------------------------------------------------------------------


V11_PC_SOURCE_LITERALS = (
    "dense-sensitivity",
    "explicit",
    "matrix_free_hessian_diag",
    "hessian_diag",
    "noser",
    "prior",
    "auto_linearization_diag",
    "pmat",
    "coarse-pmat",
    "custom-pcshell",
    "identity",
)


def test_v11_matrix_free_pc_source_enum_literals_present() -> None:
    """V11: every documented PC source string still appears in the runtime."""
    missing = [
        literal
        for literal in V11_PC_SOURCE_LITERALS
        if literal not in RUNTIME_CONTRACT_SOURCE
    ]
    assert not missing, (
        f"V11 contract: matrix_free_pc_source literals missing from runtime contract source: {missing!r}. "
        "These are diagnostic strings consumed by cache artifacts + GUI panels; "
        "renaming requires a SPEC.md V11 update + downstream consumer migration."
    )


# ---------------------------------------------------------------------------
# V12 — matrix_free_ksp_backend fallback-reason literal stability.
# ---------------------------------------------------------------------------


def test_v12_petsc_backend_unavailable_fallback_reason_literal_present() -> None:
    """V12: ``petsc_backend_unavailable`` is the canonical fallback reason."""
    assert '"petsc_backend_unavailable"' in RUNTIME_CONTRACT_SOURCE, (
        "V12 contract: ``petsc_backend_unavailable`` fallback-reason literal "
        "must appear in the runtime contract source when petsc4py is missing "
        "and the matrix-free ksp backend falls back to scipy."
    )


# ---------------------------------------------------------------------------
# V10 — matrix-free PC fallback for unsupported petsc-gamg.
# ---------------------------------------------------------------------------


def test_v10_petsc_gamg_unsupported_fallback_reason_literal_present() -> None:
    """V10: matrix-free path falls back when petsc-gamg has no Pmat with a documented reason."""
    assert "petsc_gamg_not_supported_in_matrix_free" in RUNTIME_CONTRACT_SOURCE, (
        "V10 contract: ``petsc_gamg_not_supported_in_matrix_free`` fallback "
        "reason must appear in the runtime contract source so callers can "
        "detect the auto fallback to ``diag``."
    )


# ---------------------------------------------------------------------------
# V14 — direct PC refresh reason literal.
# ---------------------------------------------------------------------------


def test_v14_direct_factor_requires_rebuild_literal_present_in_forward() -> None:
    """V14: the ``direct_factor_requires_rebuild`` PC refresh reason lives in the forward path."""
    forward_source = (
        REPO_ROOT / "src" / "pyeidors" / "forward" / "eit_forward_model.py"
    ).read_text(encoding="utf-8")
    assert "direct_factor_requires_rebuild" in forward_source, (
        "V14 contract: ``ForwardKSPSession._decide_pc_reuse_for_session`` must "
        "stamp ``direct_factor_requires_rebuild`` on cross-sigma PC refresh "
        "when ksp_type=='preonly' and pc_type ∈ {lu, cholesky, qr}."
    )


# ---------------------------------------------------------------------------
# _IterationLog dataclass schema (consumed by cache disk artifacts).
# ---------------------------------------------------------------------------


EXPECTED_ITERATION_LOG_FIELDS = (
    "iteration",
    "residual",
    "residual_weighted",
    "relative_residual",
    "relative_residual_weighted",
    "residual_max",
    "meas_norm",
    "pred_norm",
    "meas_max",
    "pred_max",
    "jtr_norm",
    "delta_norm",
    "step",
    "lambda_eff",
    "relative_change",
    "res_drop",
    "meas_misfit",
    "prior_misfit",
    "total_objective",
)

EXPECTED_ITERATION_LOG_PAYLOAD_KEYS = (
    "iteration",
    "residual",
    "residual_weighted",
    "relative_residual",
    "relative_residual_weighted",
    "residual_max",
    "meas_norm",
    "pred_norm",
    "meas_max",
    "pred_max",
    "JTr_norm",
    "delta_norm",
    "step",
    "lambda_eff",
    "relative_change",
    "res_drop",
    "meas_misfit",
    "prior_misfit",
    "total_objective",
)


def test_iteration_log_dataclass_field_set_is_locked() -> None:
    """``_IterationLog`` field tuple stays fixed across the T77 refactor."""
    actual = tuple(f.name for f in fields(gn_runtime._IterationLog))
    assert actual == EXPECTED_ITERATION_LOG_FIELDS


def test_iteration_log_to_payload_keys_are_locked() -> None:
    """``_IterationLog.to_payload`` MUST emit the canonical key order.

    Several disk artifacts and diagnostic dumps key on the exact strings
    (``JTr_norm`` casing in particular); silently swapping field names
    breaks JSON consumers.
    """
    log = gn_runtime._IterationLog(
        iteration=0,
        residual=0.0,
        residual_weighted=0.0,
        relative_residual=0.0,
        relative_residual_weighted=None,
        residual_max=0.0,
        meas_norm=0.0,
        pred_norm=0.0,
        meas_max=0.0,
        pred_max=0.0,
        jtr_norm=0.0,
        delta_norm=0.0,
        step=0.0,
        lambda_eff=0.0,
        relative_change=0.0,
        res_drop=None,
        meas_misfit=0.0,
        prior_misfit=0.0,
        total_objective=0.0,
    )
    assert tuple(log.to_payload().keys()) == EXPECTED_ITERATION_LOG_PAYLOAD_KEYS


# ---------------------------------------------------------------------------
# _JacobianActionBundle dataclass schema (matrix-free dispatch surface).
# ---------------------------------------------------------------------------


def test_jacobian_action_bundle_field_set_is_locked() -> None:
    actual = tuple(f.name for f in fields(gn_runtime._JacobianActionBundle))
    assert actual == (
        "shape",
        "representation",
        "dense",
        "matvec",
        "rmatvec",
        "linearization",
    )


# ---------------------------------------------------------------------------
# ``run_reconstruction`` public signature.
# ---------------------------------------------------------------------------


def test_run_reconstruction_public_signature_is_locked() -> None:
    """Top-level GN entry point must keep its keyword set + defaults."""
    sig = inspect.signature(gn_runtime.run_reconstruction)
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == [
        "reconstructor",
        "measured_data",
        "initial_conductivity",
        "jacobian_method",
        "prior_data",
        "record_conductivity_history",
        "conductivity_history_stride",
    ]
    defaults = {
        p.name: p.default for p in params if p.default is not inspect.Parameter.empty
    }
    assert defaults["initial_conductivity"] == 1.0
    assert defaults["jacobian_method"] == "efficient"
    assert defaults["prior_data"] is None
    assert defaults["record_conductivity_history"] is False
    assert defaults["conductivity_history_stride"] == 1


# ---------------------------------------------------------------------------
# PETSc matrix-free context classes must remain importable from runtime.
# ---------------------------------------------------------------------------


def test_petsc_matrix_free_context_classes_exposed() -> None:
    """Sub-module split MUST keep these symbols re-exported from ``gauss_newton_runtime``.

    Down-stream tests + benchmark scripts reach in via
    ``gn_runtime._PETScMatrixFreeHessianContext`` etc.; if T77 phase 2
    moves the bodies to a sub-module, the runtime must still re-export
    them as module-level attributes for backward compatibility.
    """
    for symbol in (
        "_PETScMatrixFreeHessianContext",
        "_PETScMatrixFreePCContext",
        "_solve_matrix_free_hessian_via_petsc",
        "_build_matrix_free_custom_pc_operator",
        "_build_matrix_free_pmat_inverse_operator",
        "_build_matrix_free_explicit_pc_operator",
        "_as_jacobian_action_bundle",
        "_solve_linear_system_fast",
        "_difference_step_size_objective",
        "_apply_difference_step_size",
        "_select_step_size",
        "run_reconstruction",
    ):
        assert hasattr(gn_runtime, symbol), (
            f"gauss_newton_runtime must expose {symbol!r}; sub-module split "
            "in T77 phase 2 has to preserve the import path."
        )


# ---------------------------------------------------------------------------
# Sub-module imports companions must stay reachable.
# ---------------------------------------------------------------------------


def test_companion_modules_still_importable() -> None:
    """Engine + line_search + weights + device companions stay available."""
    for module_path in (
        "pyeidors.inverse.solvers.gauss_newton",
        "pyeidors.inverse.solvers.gauss_newton_engine",
        "pyeidors.inverse.solvers.gauss_newton_runtime",
        "pyeidors.inverse.solvers.gauss_newton_line_search",
        "pyeidors.inverse.solvers.gauss_newton_regularization",
        "pyeidors.inverse.solvers.gauss_newton_step_size",
        "pyeidors.inverse.solvers.gauss_newton_weights",
        "pyeidors.inverse.solvers.gauss_newton_device",
        "pyeidors.inverse.solvers.matrix_free_gn",
    ):
        __import__(module_path)
