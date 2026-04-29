"""PyEIDORS Jacobian calculation module"""

from .adjoint_jacobian import EidorsJacobianAdapter
from .base_jacobian import BaseJacobianCalculator
from .direct_jacobian import DirectJacobianCalculator
from .linearized import JacobianLinearization, compute_sigma_fingerprint
from .process_jacobian_cache import (
    build_process_jacobian_key,
    clear_process_jacobian_cache,
    get_process_cached_jacobian,
    process_jacobian_cache_stats,
    put_process_cached_jacobian,
)

__all__ = [
    "BaseJacobianCalculator",
    "DirectJacobianCalculator",
    "EidorsJacobianAdapter",
    "JacobianLinearization",
    "build_process_jacobian_key",
    "clear_process_jacobian_cache",
    "compute_sigma_fingerprint",
    "get_process_cached_jacobian",
    "process_jacobian_cache_stats",
    "put_process_cached_jacobian",
]
