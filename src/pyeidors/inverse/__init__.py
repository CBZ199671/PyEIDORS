"""PyEidors inverse problem solver module."""

from .solvers.gauss_newton import ModularGaussNewtonReconstructor
from .solvers.sparse_bayesian import SparseBayesianReconstructor, SparseBayesianConfig
from .workflows import (
    perform_absolute_reconstruction,
    perform_difference_reconstruction,
    perform_sparse_absolute_reconstruction,
    perform_sparse_difference_reconstruction,
    ReconstructionResult,
)

__all__ = [
    "ModularGaussNewtonReconstructor",
    "SparseBayesianReconstructor",
    "SparseBayesianConfig",
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
    "perform_sparse_difference_reconstruction",
    "ReconstructionResult",
]
