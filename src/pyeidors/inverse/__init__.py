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
from .prior import graph_laplacian
from .reconstruction_matrix import (
    OneStepRMResult,
    build_one_step_rm,
    reconstruct_difference,
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
    "graph_laplacian",
    "OneStepRMResult",
    "build_one_step_rm",
    "reconstruct_difference",
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
    "perform_sparse_difference_reconstruction",
    "ReconstructionResult",
]
