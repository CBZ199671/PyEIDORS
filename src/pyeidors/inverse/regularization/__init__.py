"""PyEidors正则化模块"""

from .base_regularization import BaseRegularization
from .smoothness import (
    SmoothnessRegularization,
    TikhonovRegularization, 
    TotalVariationRegularization
)

__all__ = [
    'BaseRegularization',
    'SmoothnessRegularization',
    'TikhonovRegularization',
    'TotalVariationRegularization'
]