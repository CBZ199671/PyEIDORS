"""Reduced-order helpers for fused 3D GN acceleration."""

from .inexact_controller import InexactController
from .lowrank_subspace import build_lowrank_subspace
from .pod_basis import compute_pod_basis, merge_orthonormal_bases
from .reduced_gn_step import build_reduced_operator, solve_reduced_step
from .snapshot_bank import SnapshotBank, select_snapshot_matrix

__all__ = [
    "InexactController",
    "SnapshotBank",
    "build_lowrank_subspace",
    "compute_pod_basis",
    "merge_orthonormal_bases",
    "build_reduced_operator",
    "select_snapshot_matrix",
    "solve_reduced_step",
]
