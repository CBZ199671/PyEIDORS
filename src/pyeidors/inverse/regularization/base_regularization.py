"""Regularization base class and operator helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from scipy.sparse import isspmatrix
from scipy.sparse.linalg import LinearOperator, aslinearoperator

RegularizationMatrix: TypeAlias = np.ndarray | LinearOperator | object


class BaseRegularization(ABC):
    """Regularization base class."""

    def __init__(self, fwd_model):
        self.fwd_model = fwd_model
        self.mesh = fwd_model.mesh
        V_sigma = fwd_model.fwd_model.V_sigma if hasattr(fwd_model, "fwd_model") else fwd_model.V_sigma
        self.n_elements = int(V_sigma.dofmap.index_map.size_local * V_sigma.dofmap.index_map_bs)

    @abstractmethod
    def create_matrix(self) -> RegularizationMatrix:
        """Create regularization matrix."""
        pass

    def get_regularization_matrix(self, cache: bool = True) -> RegularizationMatrix:
        """Get regularization matrix (with caching support)."""
        if not hasattr(self, '_cached_matrix') or not cache:
            self._cached_matrix = self.create_matrix()
        return self._cached_matrix

    @staticmethod
    def as_linear_operator(matrix: RegularizationMatrix, *, shape: tuple[int, int] | None = None) -> LinearOperator:
        """Convert dense/sparse matrix-like payload to ``LinearOperator``."""
        if isinstance(matrix, LinearOperator):
            return matrix
        if isspmatrix(matrix):
            return aslinearoperator(matrix)
        dense = np.asarray(matrix, dtype=np.float64)
        if shape is not None and dense.shape != shape:
            raise ValueError(f"Regularization shape mismatch: expected {shape}, got {dense.shape}")
        return aslinearoperator(dense)
