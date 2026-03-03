"""Sparse Bayesian public entrypoint.

Implementation lives in :mod:`pyeidors.inverse.solvers.sparse_bayesian_engine`.
"""

from .sparse_bayesian_engine import SparseBayesianConfig, SparseBayesianReconstructor

__all__ = ["SparseBayesianConfig", "SparseBayesianReconstructor"]
