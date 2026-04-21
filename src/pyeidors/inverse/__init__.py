"""PyEIDORS inverse problem solver module."""

from .solvers.gauss_newton import GaussNewtonReconstructor
from .solvers.matrix_free_gn import MatrixFreeGNStepResult, solve_matrix_free_gn_step
from .solvers.sparse_bayesian import SparseBayesianReconstructor, SparseBayesianConfig
from .contracts import SolverOutput
from .block_system import (
    BlockCoupling,
    JointInverseBlockMetadata,
    ParameterBlock,
    build_electrode_movement_jacobian,
    build_sigma_contact_block_metadata,
    make_block_diagonal_inverse_action,
    prior_movement,
    scale_contact_impedance_update,
)
from .dual_mesh import CellMesh, DualMesh, VoxelGrid, coarse2fine
from .greit import (
    GREIT_METRIC_KEYS,
    GREITRM,
    GREITTrainingTargets,
    build_3d_greit_rm,
    generate_spherical_targets,
    greit_metrics,
    load_greit_rm,
    write_greit_metrics_artifact,
)
from .prior import graph_difference_operator, graph_laplacian
from .postprocess import (
    TemporalTVPipelineResult,
    TVRefinementResult,
    exponential_smooth_frames,
    moving_average_frames,
    postprocess_rm_frames,
    refine_tv_pdhg,
    total_variation_norm,
)
from .matrix_free import DualMeshJacobianOperator
from .reconstruction_matrix import (
    OneStepRMResult,
    build_one_step_rm,
    reconstruct_difference,
    reconstruct_difference_batch,
    rm_signature,
    rm_signature_payload,
    write_forward_rm_benchmark_artifact,
)
from .workflows import (
    perform_absolute_reconstruction,
    perform_difference_reconstruction,
    perform_sparse_absolute_reconstruction,
    perform_sparse_difference_reconstruction,
    ReconstructionResult,
)

__all__ = [
    "GaussNewtonReconstructor",
    "MatrixFreeGNStepResult",
    "solve_matrix_free_gn_step",
    "SparseBayesianReconstructor",
    "SparseBayesianConfig",
    "SolverOutput",
    "BlockCoupling",
    "JointInverseBlockMetadata",
    "ParameterBlock",
    "build_electrode_movement_jacobian",
    "build_sigma_contact_block_metadata",
    "make_block_diagonal_inverse_action",
    "prior_movement",
    "scale_contact_impedance_update",
    "CellMesh",
    "DualMesh",
    "VoxelGrid",
    "coarse2fine",
    "GREIT_METRIC_KEYS",
    "GREITRM",
    "GREITTrainingTargets",
    "build_3d_greit_rm",
    "generate_spherical_targets",
    "greit_metrics",
    "load_greit_rm",
    "write_greit_metrics_artifact",
    "graph_difference_operator",
    "graph_laplacian",
    "TemporalTVPipelineResult",
    "TVRefinementResult",
    "exponential_smooth_frames",
    "moving_average_frames",
    "postprocess_rm_frames",
    "refine_tv_pdhg",
    "total_variation_norm",
    "DualMeshJacobianOperator",
    "OneStepRMResult",
    "build_one_step_rm",
    "reconstruct_difference",
    "reconstruct_difference_batch",
    "rm_signature",
    "rm_signature_payload",
    "write_forward_rm_benchmark_artifact",
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
    "perform_sparse_difference_reconstruction",
    "ReconstructionResult",
]
