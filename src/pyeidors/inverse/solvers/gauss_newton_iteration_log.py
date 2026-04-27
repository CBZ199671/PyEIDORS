"""Per-iteration Gauss-Newton log records.

T77 phase 2 commit #1 — first sub-module split lifted out of the
3694-line ``gauss_newton_runtime.py``. This module owns the
:class:`_IterationLog` dataclass that the GN runtime appends one of
per Gauss-Newton iteration plus the small ``_record_iteration_log``
factory that wraps the construction site.

The dataclass field tuple, ``to_payload`` key order (``JTr_norm``
casing matters — disk artifacts and benchmark JSON consumers parse
this name) and the ``_record_iteration_log`` keyword set are part of
the V73-style contract frozen by
:mod:`tests.unit.test_gn_runtime_contract_freeze`.
``gauss_newton_runtime`` re-exports both symbols at module scope so
downstream callers and the contract gate keep their existing import
paths.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class _IterationLog:
    iteration: int
    residual: float
    residual_weighted: float
    relative_residual: float
    relative_residual_weighted: float | None
    residual_max: float
    meas_norm: float
    pred_norm: float
    meas_max: float
    pred_max: float
    jtr_norm: float
    delta_norm: float
    step: float
    lambda_eff: float
    relative_change: float
    res_drop: float | None
    meas_misfit: float
    prior_misfit: float
    total_objective: float

    def to_payload(self) -> dict[str, float | int | None]:
        return {
            "iteration": self.iteration,
            "residual": self.residual,
            "residual_weighted": self.residual_weighted,
            "relative_residual": self.relative_residual,
            "relative_residual_weighted": self.relative_residual_weighted,
            "residual_max": self.residual_max,
            "meas_norm": self.meas_norm,
            "pred_norm": self.pred_norm,
            "meas_max": self.meas_max,
            "pred_max": self.pred_max,
            "JTr_norm": self.jtr_norm,
            "delta_norm": self.delta_norm,
            "step": self.step,
            "lambda_eff": self.lambda_eff,
            "relative_change": self.relative_change,
            "res_drop": self.res_drop,
            "meas_misfit": self.meas_misfit,
            "prior_misfit": self.prior_misfit,
            "total_objective": self.total_objective,
        }


def _record_iteration_log(
    iteration_logs: list[_IterationLog],
    *,
    iteration: int,
    residual_norm: float,
    residual_norm_weighted: float,
    rel_residual: float,
    rel_residual_weighted: float | None,
    residual_max: float,
    meas_norm: float,
    pred_norm: float,
    meas_max: float,
    pred_max: float,
    jtr_norm: float,
    delta_norm: float,
    optimal_step_size: float,
    lambda_eff: float,
    relative_change: float,
    res_drop: float | None,
    meas_misfit: float,
    prior_misfit: float,
    total_objective: float,
) -> None:
    iteration_logs.append(
        _IterationLog(
            iteration=iteration,
            residual=residual_norm,
            residual_weighted=residual_norm_weighted,
            relative_residual=rel_residual,
            relative_residual_weighted=rel_residual_weighted,
            residual_max=residual_max,
            meas_norm=meas_norm,
            pred_norm=pred_norm,
            meas_max=meas_max,
            pred_max=pred_max,
            jtr_norm=jtr_norm,
            delta_norm=delta_norm,
            step=optimal_step_size,
            lambda_eff=lambda_eff,
            relative_change=relative_change,
            res_drop=res_drop,
            meas_misfit=meas_misfit,
            prior_misfit=prior_misfit,
            total_objective=total_objective,
        )
    )
