"""PyEIDORS inverse problem solver module."""

from .solvers.gauss_newton import GaussNewtonReconstructor
from .solvers.sparse_bayesian import SparseBayesianReconstructor, SparseBayesianConfig
from .contracts import SolverOutput
from .block_system import (
    BlockCoupling,
    JointInverseBlockMetadata,
    ParameterBlock,
    build_sigma_contact_block_metadata,
    make_block_diagonal_inverse_action,
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
from .prior import graph_laplacian
from .reconstruction_matrix import (
    OneStepRMResult,
    build_one_step_rm,
    reconstruct_difference,
    reconstruct_difference_batch,
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
    "SparseBayesianReconstructor",
    "SparseBayesianConfig",
    "SolverOutput",
    "BlockCoupling",
    "JointInverseBlockMetadata",
    "ParameterBlock",
    "build_sigma_contact_block_metadata",
    "make_block_diagonal_inverse_action",
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
    "graph_laplacian",
    "OneStepRMResult",
    "build_one_step_rm",
    "reconstruct_difference",
    "reconstruct_difference_batch",
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
    "perform_sparse_difference_reconstruction",
    "ReconstructionResult",
]
