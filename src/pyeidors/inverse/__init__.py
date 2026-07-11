"""PyEIDORS inverse problem solver module.

The inverse package exports many high-level reconstruction helpers, but most of
them pull substantial optional runtime stacks.  Keep the package import light and
resolve public symbols on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".solvers.gauss_newton": ("GaussNewtonReconstructor",),
    ".solvers.matrix_free_gn": (
        "MatrixFreeGNStepResult",
        "solve_matrix_free_gn_step",
    ),
    ".solvers.sparse_bayesian_engine": (
        "SparseBayesianConfig",
        "SparseBayesianReconstructor",
    ),
    ".contracts": ("SolverOutput",),
    ".block_system": (
        "BlockCoupling",
        "JointInverseBlockMetadata",
        "JointFieldSplitSolveResult",
        "ParameterBlock",
        "SigmaContactNormalSystem",
        "assemble_sigma_contact_normal_system",
        "build_electrode_movement_jacobian",
        "build_sigma_contact_block_metadata",
        "configure_petsc_fieldsplit_solver",
        "make_block_diagonal_inverse_action",
        "prior_movement",
        "scale_contact_impedance_update",
        "solve_sigma_contact_fieldsplit",
    ),
    ".dual_mesh": (
        "CellMesh",
        "DualMesh",
        "VoxelGrid",
        "coarse2fine",
    ),
    ".dynamic": (
        "DYNAMIC_KALMAN_SCHEMA",
        "SPATIOTEMPORAL_GN_SCHEMA",
        "SPATIOTEMPORAL_TV_HUBER_SCHEMA",
        "DynamicKalmanResult",
        "SpatiotemporalGNResult",
        "SpatiotemporalTVHuberResult",
        "run_dynamic_kalman_filter",
        "solve_batch_spatiotemporal_gn",
        "solve_spatiotemporal_tv_huber",
        "temporal_difference_operator",
    ),
    ".dynamic_session": (
        "DYNAMIC_DIAGONAL_SESSION_SCHEMA",
        "DYNAMIC_MEASUREMENT_DIAGONAL_SESSION_SCHEMA",
        "DiagonalKalmanConfig",
        "DiagonalKalmanUpdate",
        "PersistentDiagonalKalmanRegistry",
        "PersistentDiagonalKalmanSession",
        "PersistentMeasurementDiagonalKalmanSession",
    ),
    ".greit": (
        "GREIT3DDistribution",
        "GREIT_CACHE_SIGNATURE_SCHEMA",
        "GREITDesiredImages",
        "GREIT_EIDORS_HDF5_SCHEMA",
        "GREITFiniteTargetResponses",
        "GREIT_METRIC_KEYS",
        "GREIT_RM_HDF5_SCHEMA",
        "GREITRM",
        "GREITRMComponents",
        "GREITNativeTrainingPipeline",
        "GREITTrainingTargets",
        "GREITWeightSearchResult",
        "build_3d_greit_rm",
        "build_greit_desired_images",
        "build_greit_finite_target_responses",
        "build_greit_rm_from_eidors_components",
        "build_greit3d_distribution",
        "build_native_greit_training_pipeline",
        "calc_greit_rm",
        "generate_spherical_targets",
        "greit_cache_signature",
        "greit_cache_signature_payload",
        "greit_metrics",
        "greit_desired_image_sigmoid",
        "load_greit_rm",
        "migrate_greit_rm_to_hdf5",
        "optimize_greit_weight_eidors_nf",
        "optimize_greit_weight_for_metric",
        "search_greit_weight_for_metric",
        "write_greit_metrics_artifact",
    ),
    ".greit_warmup": (
        "GREIT_COMMON_CONFIG_ENV",
        "GREIT_COMMON_CONFIG_WARMUP_SCHEMA",
        "GREITCommonConfig",
        "GREITCommonWarmupResult",
        "common_config_runtime_metadata",
        "greit_common_config",
        "greit_common_config_artifact_path",
        "greit_common_config_dir",
        "greit_common_config_ids",
        "load_greit_common_config",
        "normalize_greit_common_config_id",
        "precompute_greit_common_config",
        "register_greit_common_config_artifact",
        "resolve_greit_common_config_artifact_path",
        "resolve_greit_common_config_artifact_path_from_meta",
    ),
    ".greit_registry": (
        "GREIT_ARTIFACT_REGISTRY_SCHEMA",
        "GREIT_ARTIFACT_SIGNATURE_SCHEMA",
        "GREIT_NATIVE_BUILDER_VERSION",
        "GREIT_REGISTRY_ENV",
        "GREITRegistryLookup",
        "build_native_greit_artifact",
        "greit_artifact_path_for_signature",
        "greit_artifact_signature",
        "greit_artifact_signature_payload",
        "greit_registry_dir",
        "greit_registry_manifest_path",
        "load_greit_registry_manifest",
        "register_greit_artifact",
        "resolve_greit_artifact",
        "resolve_or_build_greit_artifact",
        "write_greit_registry_manifest",
    ),
    ".prior": (
        "RtRPrior",
        "TVIRLSResult",
        "as_rtr_prior",
        "graph_curvature_prior",
        "graph_difference_operator",
        "graph_laplacian",
        "graph_ltl",
        "graph_ltl_prior",
        "load_rtr_prior_artifact",
        "solve_tv_irls_batch",
        "solve_tv_irls_frame",
        "tv_irls_objective",
        "tv_irls_prior_from_state",
        "write_rtr_prior_artifact",
    ),
    ".postprocess": (
        "TemporalTVPipelineResult",
        "TVRefinementResult",
        "exponential_smooth_frames",
        "moving_average_frames",
        "postprocess_rm_frames",
        "refine_tv_pdhg",
        "total_variation_norm",
    ),
    ".matrix_free": ("DualMeshJacobianOperator",),
    ".reconstruction_matrix": (
        "OneStepRMResult",
        "RMArtifact",
        "build_one_step_rm",
        "load_rm_artifact",
        "migrate_rm_artifact_to_hdf5",
        "reconstruct_difference",
        "reconstruct_difference_batch",
        "reconstruct_temporal_difference_batch",
        "rm_signature",
        "rm_signature_payload",
        "write_forward_rm_benchmark_artifact",
        "write_rm_artifact",
    ),
    ".workflows": (
        "ReconstructionResult",
        "perform_absolute_reconstruction",
        "perform_difference_reconstruction",
        "perform_sparse_absolute_reconstruction",
        "perform_sparse_difference_reconstruction",
    ),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
