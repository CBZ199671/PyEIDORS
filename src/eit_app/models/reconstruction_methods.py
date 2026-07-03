"""Shared reconstruction method options for database reconstruction flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyeidors.runtime_paths import pyeidors_cache_path


CANONICAL_SINGLE_STEP_LAMBDA_EFF = 1.0e-2
CANONICAL_SINGLE_STEP_HP = CANONICAL_SINGLE_STEP_LAMBDA_EFF**0.5
NOSER_SPARSE_METHOD = "noser_sparse"
PSEUDO3D_NOSER_RM_METHOD = "pseudo3d_noser_rm"


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
        "NOSER Sparse · Difference (matrix-free)",
        NOSER_SPARSE_METHOD,
        requires_reference=True,
        uses_iterations=False,
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
        "Pseudo-3D NOSER RM · 2D→3D",
        PSEUDO3D_NOSER_RM_METHOD,
        requires_reference=True,
        uses_iterations=False,
        locked_lambda_eff=True,
        custom_lambda_eff=True,
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
_RM_METHODS = {"noser_rm", "laplace_rm", "curvature_rm", PSEUDO3D_NOSER_RM_METHOD}
_SPARSE_SINGLE_STEP_METHODS = {NOSER_SPARSE_METHOD}
_LOCKED_LAMBDA_METHODS = _RM_METHODS | {"debug_fine_mesh_noser"}
_CUSTOM_LAMBDA_METHODS = _RM_METHODS
_RM_REGULARIZATION = {
    "noser_rm": "noser",
    "laplace_rm": "laplace",
    "curvature_rm": "curvature",
    PSEUDO3D_NOSER_RM_METHOD: "noser",
}
_RM_FORM = {
    "noser_rm": "measurement",
    "laplace_rm": "measurement",
    "curvature_rm": "measurement",
    PSEUDO3D_NOSER_RM_METHOD: "measurement",
}


def _gui_rm_artifact_dir() -> str:
    return str(pyeidors_cache_path("gui_rm"))


@dataclass(frozen=True)
class PreparedReconstructionMethod:
    method: str
    regularization_alpha: float
    max_iterations: int
    metadata: dict[str, Any]


def normalize_database_reconstruction_method(method: str) -> str:
    key = str(method or "").strip().lower()
    aliases = {
        "eidors_one_step_noser": "noser_rm",
        "gn_absolute": "gn-absolute",
        "absolute_gn": "gn-absolute",
        "noser": "noser_rm",
        "noser_sparse": NOSER_SPARSE_METHOD,
        "noser_matrix_free": NOSER_SPARSE_METHOD,
        "matrix_free_noser": NOSER_SPARSE_METHOD,
        "laplace": "laplace_rm",
        "curvature": "curvature_rm",
        "pseudo3d": PSEUDO3D_NOSER_RM_METHOD,
        "pseudo_3d": PSEUDO3D_NOSER_RM_METHOD,
        "pseudo3d_noser": PSEUDO3D_NOSER_RM_METHOD,
        "pseudo_3d_noser": PSEUDO3D_NOSER_RM_METHOD,
        "pseudo3d_noser_rm": PSEUDO3D_NOSER_RM_METHOD,
        "pseudo_3d_noser_rm": PSEUDO3D_NOSER_RM_METHOD,
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


def _metadata_int(metadata: dict[str, Any], key: str, default: int) -> int:
    try:
        value = int(metadata.get(key, default))
    except (TypeError, ValueError):
        return int(default)
    return max(int(value), 1)


def _metadata_float(metadata: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value if value == value else float(default)


def pseudo3d_noser_rm_metadata(
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mark a 3D acquisition for layered 2D inverse solves and 3D display.

    The pseudo-3D route reconstructs one 2D circular inverse model per source
    electrode ring, then interpolates those layer results into a shallow 3D
    tetrahedral display mesh. It is deliberately metadata-explicit so downstream
    reports cannot confuse it with a true 3D CEM inverse solve.
    """

    meta = dict(metadata or {})
    source_mesh_dimension = _metadata_int(meta, "mesh_dimension", 2)
    source_n_elec = _metadata_int(
        meta, "n_elec", _metadata_int(meta, "n_electrodes", 16)
    )
    source_n_rings = _metadata_int(meta, "n_rings", 1)
    layered_output = source_mesh_dimension == 3 and source_n_rings > 1
    radius = max(_metadata_float(meta, "radius", 1.0), 1.0e-9)
    height = _metadata_float(
        meta,
        "height",
        _metadata_float(meta, "mesh_height", 2.0 * radius),
    )
    height = max(height, 1.0e-9)
    layers = _metadata_int(
        meta,
        "pseudo3d_layers",
        max(5, source_n_rings if source_mesh_dimension == 3 else 1),
    )
    layers = max(layers, 2)
    original = {
        "mesh_dimension": source_mesh_dimension,
        "n_elec": source_n_elec,
        "n_rings": source_n_rings,
        "electrode_layout": str(meta.get("electrode_layout", "ring_major")),
        "measurement_protocol": str(meta.get("measurement_protocol", "eidors_full_3d")),
        "drive_mode": str(meta.get("drive_mode", "")),
        "drive_value": meta.get("drive_value"),
        "petsc_device": meta.get("petsc_device"),
        "device": meta.get("device"),
        "forward_backend": meta.get("forward_backend"),
        "acceleration_profile": meta.get("acceleration_profile"),
    }
    meta.update(
        {
            "pseudo3d_output": True,
            "pseudo3d_layered_output": layered_output,
            "pseudo3d_algorithm": (
                "layered_2d_noser_rm_z_interpolated_tetra_v1"
                if layered_output
                else "2d_noser_rm_extruded_tetra_v1"
            ),
            "pseudo3d_source_mesh_dimension": source_mesh_dimension,
            "pseudo3d_source_n_elec": source_n_elec,
            "pseudo3d_source_n_rings": source_n_rings,
            "pseudo3d_source_geometry": original,
            "pseudo3d_inverse_mesh_dimension": 2,
            "pseudo3d_display_mesh_dimension": 3,
            "pseudo3d_display_layers": layers,
            "pseudo3d_display_height": height,
            "pseudo3d_layer_count": source_n_rings,
            "pseudo3d_layer_n_elec": source_n_elec,
            "pseudo3d_layer_measurement_source": "same_ring_stim_and_meas",
            "pseudo3d_legacy_collapsed_n_elec": source_n_elec * source_n_rings,
            "mesh_dimension": 2,
            "n_elec": source_n_elec,
            "n_rings": 1,
            "electrode_layout": "ring_major",
            "measurement_protocol": "eidors_full_3d",
            "drive_mode": "line_current_density",
            "drive_value": 1.0,
            "petsc_device": "cpu",
            "device": "cpu",
            "rm_device": "cpu",
            "forward_backend": "dolfinx",
            "forward_solver_preset": "auto",
            "forward_mat_solve": "off",
            "acceleration_profile": "default",
        }
    )
    return meta


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
    if route == PSEUDO3D_NOSER_RM_METHOD:
        meta = pseudo3d_noser_rm_metadata(meta)
    alpha_input = float(regularization_alpha)
    max_iter = int(max_iterations)

    if route in _SPARSE_SINGLE_STEP_METHODS:
        difference_lambda = max(alpha_input, 1.0e-12)
        meta.update(
            {
                "difference_mode": "normalized",
                "difference_orientation": "target_minus_reference",
                "difference_preset": "eidors_one_step_noser",
                "absolute_preset": "eidors_abs_gn",
                "simulation_inverse_route": route,
                "simulation_inverse_route_kind": "sparse",
                "simulation_inverse_debug_route": False,
                "rm_route_requires_artifact": False,
                "rm_auto_build": False,
                "rm_route_pending_task": "",
                "rm_regularization": "noser",
                "rm_form": "",
                "rm_output_display_mode": "absolute_sigma",
                "rm_artifact_dir": _gui_rm_artifact_dir(),
                "reconstruction_runtime": "single_step_cached",
                "jacobian_representation": "linearized",
                "linearized_solver_strategy": "auto",
                "linearized_maxiter": 0,
                "lazy_preconditioner_mode": "auto",
                "online_hot_path": "single_step_sparse_linearized_solver",
                "difference_lambda": difference_lambda,
                "hyperparameter_ui_name": "lambda_eff",
                "hyperparameter_ui_value": difference_lambda,
                "hyperparameter_ui_locked": False,
                "hyperparameter_effective_source": "single_step_sparse",
                "hyperparameter_formula": "JtJ_plus_lambda_noser_diag",
                "hyperparameter_diagnostic": "single_step_sparse_linearized_solve",
                "regularization_alpha_input": alpha_input,
                "regularization_alpha_applied": True,
                "lambda_eff": difference_lambda,
                "lambda_eff_custom_enabled": False,
                "hp": float(difference_lambda**0.5),
                "hp_squared": difference_lambda,
                "difference_lambda_semantics": "single_step_sparse_lambda_eff",
            }
        )
        return PreparedReconstructionMethod(
            method="gn-difference",
            regularization_alpha=difference_lambda,
            max_iterations=1,
            metadata=meta,
        )

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
                "difference_preset": "noser_rm"
                if route == PSEUDO3D_NOSER_RM_METHOD
                else route,
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
                "rm_artifact_dir": _gui_rm_artifact_dir(),
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
                "rm_artifact_dir": _gui_rm_artifact_dir(),
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
