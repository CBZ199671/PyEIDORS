"""EIT imaging workflow wrapper.

This subpackage provides high-level helper functions for difference and absolute imaging,
avoiding confusion between the two modes in higher-level applications.
"""

from .absolute import perform_absolute_reconstruction
from .difference import perform_difference_reconstruction
from .sparse_bayesian import (
    perform_sparse_absolute_reconstruction,
    perform_sparse_difference_reconstruction,
)
from .base import ReconstructionResult

__all__ = [
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
    "perform_sparse_difference_reconstruction",
    "ReconstructionResult",
]
