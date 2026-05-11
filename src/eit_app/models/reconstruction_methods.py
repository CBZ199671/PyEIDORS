"""Shared reconstruction method options for database reconstruction flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CANONICAL_SINGLE_STEP_LAMBDA_EFF = 1.0e-2
CANONICAL_SINGLE_STEP_HP = CANONICAL_SINGLE_STEP_LAMBDA_EFF**0.5


@dataclass(frozen=True)
class ReconstructionMethodOption:
    label: str
    method: str
    requires_reference: bool
    uses_iterations: bool
    locked_lambda_eff: bool = False
    custom_lambda_eff: bool = False


DATABASE_RECONSTRUCTION_METHODS: tuple[ReconstructionMethodOption, ...] = (
    ReconstructionMethodOption(
        "NOSER RM · Difference (fast)",
        "noser_rm",
        requires_reference=True,
        uses_iterations=False,
        locked_lambda_eff=True,
        custom_lambda_eff=True,
    ),
    ReconstructionMethodOption(
        "Laplace RM · Difference (smooth)",
        "laplace_rm",
        requires_reference=True,
        uses_iterations=False,
        locked_lambda_eff=True,
        custom_lambda_eff=True,
    ),
    ReconstructionMethodOption(
        "Curvature RM · Difference (smooth)",
        "curvature_rm",
        requires_reference=True,
        uses_iterations=False,
        locked_lambda_eff=True,
        custom_lambda_eff=True,
    ),
    ReconstructionMethodOption(
        "Fine-mesh NOSER · Difference (baseline)",
        "debug_fine_mesh_noser",
        requires_reference=True,
        uses_iterations=False,
        locked_lambda_eff=True,
    ),
    ReconstructionMethodOption(
        "Gauss-Newton · Absolute (iterative)",
        "gn-absolute",
        requires_reference=False,
        uses_iterations=True,
    ),
    ReconstructionMethodOption(
        "Sparse Bayesian · Difference",
        "sparse-bayes-difference",
        requires_reference=True,
        uses_iterations=False,
    ),
    ReconstructionMethodOption(
        "Sparse Bayesian · Absolute",
        "sparse-bayes-absolute",
        requires_reference=False,
        uses_iterations=False,
    ),
)

_OPTIONS_BY_METHOD = {opt.method: opt for opt in DATABASE_RECONSTRUCTION_METHODS}
_ABSOLUTE_METHODS = {"gn-absolute", "sparse-bayes-absolute"}
_RM_METHODS = {"noser_rm", "laplace_rm", "curvature_rm"}
_LOCKED_LAMBDA_METHODS = _RM_METHODS | {"debug_fine_mesh_noser"}
_CUSTOM_LAMBDA_METHODS = _RM_METHODS
_RM_REGULARIZATION = {
    "noser_rm": "noser",
    "laplace_rm": "laplace",
    "curvature_rm": "curvature",
}
_RM_FORM = {
    "noser_rm": "measurement",
    "laplace_rm": "param",
    "curvature_rm": "param",
}


@dataclass(frozen=True)
class PreparedReconstructionMethod:
    method: str
    regularization_alpha: float
    max_iterations: int
    metadata: dict[str, Any]


def normalize_database_reconstruction_method(method: str) -> str:
    key = str(method or "").strip().lower()
    aliases = {
        "eidors_one_step_noser": "debug_fine_mesh_noser",
        "gn_absolute": "gn-absolute",
        "absolute_gn": "gn-absolute",
        "noser": "noser_rm",
        "laplace": "laplace_rm",
        "curvature": "curvature_rm",
    }
    return aliases.get(key, key)


def database_method_option(method: str) -> ReconstructionMethodOption | None:
    return _OPTIONS_BY_METHOD.get(normalize_database_reconstruction_method(method))


def database_method_requires_reference(method: str) -> bool:
    option = database_method_option(method)
    if option is not None:
        return option.requires_reference
    return normalize_database_reconstruction_method(method) not in _ABSOLUTE_METHODS


def database_method_uses_iterations(method: str) -> bool:
    option = database_method_option(method)
    if option is not None:
        return option.uses_iterations
    return normalize_database_reconstruction_method(method) == "gn-absolute"


def database_method_uses_locked_lambda_eff(method: str) -> bool:
    return normalize_database_reconstruction_method(method) in _LOCKED_LAMBDA_METHODS


def database_method_supports_custom_lambda_eff(method: str) -> bool:
    return normalize_database_reconstruction_method(method) in _CUSTOM_LAMBDA_METHODS


def prepare_database_reconstruction_method(
    method: str,
    *,
    regularization_alpha: float,
    max_iterations: int,
    custom_lambda_eff_enabled: bool = False,
    metadata: dict[str, Any] | None = None,
) -> PreparedReconstructionMethod:
    """Resolve a DB dialog method into a ReconstructionRequest method.

    The database dialogs expose user-facing route names, while
    ReconstructionController expects ``method="gn-difference"`` plus
    metadata for the cached RM routes.
    """

    route = normalize_database_reconstruction_method(method)
    meta = dict(metadata or {})
    alpha_input = float(regularization_alpha)
    max_iter = int(max_iterations)

    if route in _RM_METHODS:
        custom_lambda = bool(custom_lambda_eff_enabled)
        difference_lambda = (
            max(alpha_input, 1.0e-12)
            if custom_lambda
            else CANONICAL_SINGLE_STEP_LAMBDA_EFF
        )
        hp_eff = float(difference_lambda**0.5)
        hyperparameter_meta: dict[str, Any]
        if custom_lambda:
            hyperparameter_meta = {
                "hyperparameter_ui_name": "lambda_eff",
                "hyperparameter_ui_value": difference_lambda,
                "hyperparameter_ui_locked": False,
                "hyperparameter_effective_source": "custom_rm_rebuild",
                "hyperparameter_formula": "JtWJ_plus_hp2_RtR",
                "hyperparameter_diagnostic": "custom_lambda_eff_rebuilds_rm",
                "regularization_alpha_input": alpha_input,
                "regularization_alpha_applied": False,
                "lambda_eff": difference_lambda,
                "lambda_eff_custom_enabled": True,
                "rm_custom_lambda_eff": difference_lambda,
                "rm_rebuild_required_by_custom_lambda": True,
                "hp": hp_eff,
                "hp_squared": difference_lambda,
                "difference_lambda_semantics": "custom_lambda_eff_rebuilds_rm",
            }
        else:
            hyperparameter_meta = {
                "hyperparameter_ui_name": "lambda_eff",
                "hyperparameter_ui_value": CANONICAL_SINGLE_STEP_LAMBDA_EFF,
                "hyperparameter_ui_locked": True,
                "hyperparameter_effective_source": "canonical_single_step",
                "hyperparameter_formula": "JtWJ_plus_hp2_RtR",
                "hyperparameter_diagnostic": "locked_lambda_eff_1e-2_hp_0p1",
                "regularization_alpha_input": alpha_input,
                "regularization_alpha_applied": False,
                "lambda_eff": CANONICAL_SINGLE_STEP_LAMBDA_EFF,
                "lambda_eff_custom_enabled": False,
                "hp": CANONICAL_SINGLE_STEP_HP,
                "hp_squared": CANONICAL_SINGLE_STEP_LAMBDA_EFF,
                "difference_lambda_semantics": "lambda_eff_equals_hp_squared",
            }
        meta.update(
            {
                "difference_mode": "normalized",
                "difference_orientation": "target_minus_reference",
                "difference_preset": route,
                "absolute_preset": "eidors_abs_gn",
                "simulation_inverse_route": route,
                "simulation_inverse_route_kind": "rm",
                "simulation_inverse_debug_route": False,
                "rm_route_requires_artifact": True,
                "rm_auto_build": True,
                "rm_route_pending_task": "",
                "rm_regularization": _RM_REGULARIZATION[route],
                "rm_form": _RM_FORM[route],
                "rm_output_display_mode": "absolute_sigma",
                "rm_artifact_dir": ".pyeidors_cache/gui_rm",
                "reconstruction_runtime": "single_step_cached",
                "jacobian_representation": "auto",
                "linearized_solver_strategy": "auto",
                "linearized_maxiter": 0,
                "lazy_preconditioner_mode": "auto",
                "online_hot_path": "rm_matmul",
                "difference_lambda": difference_lambda,
                **hyperparameter_meta,
            }
        )
        return PreparedReconstructionMethod(
            method="gn-difference",
            regularization_alpha=difference_lambda,
            max_iterations=1,
            metadata=meta,
        )

    if route == "debug_fine_mesh_noser":
        meta.update(
            {
                "difference_orientation": "target_minus_reference",
                "difference_preset": "eidors_one_step_noser",
                "absolute_preset": "eidors_abs_gn",
                "simulation_inverse_route": route,
                "simulation_inverse_route_kind": "debug",
                "simulation_inverse_debug_route": True,
                "rm_route_requires_artifact": False,
                "rm_auto_build": False,
                "rm_route_pending_task": "",
                "rm_regularization": "",
                "rm_form": "",
                "rm_output_display_mode": "",
                "rm_artifact_dir": ".pyeidors_cache/gui_rm",
                "reconstruction_runtime": "single_step_cached",
                "jacobian_representation": "auto",
                "linearized_solver_strategy": "auto",
                "linearized_maxiter": 0,
                "lazy_preconditioner_mode": "auto",
                "hyperparameter_ui_name": "lambda_eff",
                "hyperparameter_ui_value": CANONICAL_SINGLE_STEP_LAMBDA_EFF,
                "hyperparameter_ui_locked": True,
                "hyperparameter_effective_source": "canonical_single_step",
                "hyperparameter_formula": "JtWJ_plus_hp2_RtR",
                "hyperparameter_diagnostic": "locked_lambda_eff_1e-2_hp_0p1",
                "regularization_alpha_input": alpha_input,
                "regularization_alpha_applied": False,
                "lambda_eff": CANONICAL_SINGLE_STEP_LAMBDA_EFF,
                "lambda_eff_custom_enabled": False,
                "hp": CANONICAL_SINGLE_STEP_HP,
                "hp_squared": CANONICAL_SINGLE_STEP_LAMBDA_EFF,
                "difference_lambda_semantics": "lambda_eff_equals_hp_squared",
                "difference_lambda": CANONICAL_SINGLE_STEP_LAMBDA_EFF,
            }
        )
        return PreparedReconstructionMethod(
            method="gn-difference",
            regularization_alpha=CANONICAL_SINGLE_STEP_LAMBDA_EFF,
            max_iterations=1,
            metadata=meta,
        )

    if route == "gn-absolute":
        meta.update(
            {
                "absolute_preset": "eidors_abs_gn",
                "simulation_inverse_route": "absolute_gn",
                "simulation_inverse_route_kind": "absolute",
                "simulation_inverse_debug_route": False,
                "reconstruction_runtime": "full_gn",
                "hyperparameter_ui_name": "alpha",
                "hyperparameter_ui_value": alpha_input,
                "hyperparameter_ui_locked": False,
                "hyperparameter_effective_source": "user_input",
                "hyperparameter_formula": "iterative_gn_alpha",
                "hyperparameter_diagnostic": "",
                "regularization_alpha_input": alpha_input,
                "regularization_alpha_applied": True,
            }
        )
        return PreparedReconstructionMethod(
            method="gn-absolute",
            regularization_alpha=alpha_input,
            max_iterations=max(1, max_iter),
            metadata=meta,
        )

    meta.update(
        {
            "reconstruction_runtime": "full_gn",
            "hyperparameter_ui_name": "alpha",
            "hyperparameter_ui_value": alpha_input,
            "hyperparameter_ui_locked": False,
            "hyperparameter_effective_source": "user_input",
            "hyperparameter_formula": "method_default_alpha",
            "hyperparameter_diagnostic": "",
            "regularization_alpha_input": alpha_input,
            "regularization_alpha_applied": True,
        }
    )
    return PreparedReconstructionMethod(
        method=route,
        regularization_alpha=alpha_input,
        max_iterations=max(1, max_iter)
        if database_method_uses_iterations(route)
        else 1,
        metadata=meta,
    )
