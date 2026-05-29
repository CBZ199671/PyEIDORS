"""Lazy runtime imports for geometry modules."""

from __future__ import annotations

from typing import Any


def mpi_comm_world() -> Any:
    """Return ``MPI.COMM_WORLD`` without importing mpi4py at module import time."""
    from mpi4py import MPI

    return MPI.COMM_WORLD


def mpi_sum_op() -> Any:
    """Return ``MPI.SUM`` without importing mpi4py at module import time."""
    from mpi4py import MPI

    return MPI.SUM


def xdmf_file_cls() -> Any:
    """Return DOLFINx ``XDMFFile`` without importing DOLFINx eagerly."""
    from dolfinx.io import XDMFFile

    return XDMFFile
