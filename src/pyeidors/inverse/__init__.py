"""PyEIDORS inverse problem solver module."""

from .solvers.gauss_newton import GaussNewtonReconstructor
from .solvers.sparse_bayesian import SparseBayesianReconstructor, SparseBayesianConfig
from .contracts import SolverOutput
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
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
    "perform_sparse_difference_reconstruction",
    "ReconstructionResult",
]
