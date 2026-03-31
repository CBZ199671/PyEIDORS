"""PyEIDORS data processing module"""

from .structures import (
    PatternConfig,
    EITData, 
    EITImage,
    EITMesh,
    MeshConfig,
    ElectrodePosition
)
from .difference import (
    DEFAULT_DIFFERENCE_MODE,
    DEFAULT_DIFFERENCE_ORIENTATION,
    build_difference_vector,
    normalize_difference_mode,
    normalize_difference_orientation,
    project_measurement_jacobian,
    project_measurement_vector,
)
from .synthetic_data import create_synthetic_data, create_custom_phantom
from .measurement_dataset import MeasurementDataset

__all__ = [
    'PatternConfig',
    'EITData',
    'EITImage', 
    'EITMesh',
    'MeshConfig',
    'ElectrodePosition',
    'DEFAULT_DIFFERENCE_MODE',
    'DEFAULT_DIFFERENCE_ORIENTATION',
    'build_difference_vector',
    'normalize_difference_mode',
    'normalize_difference_orientation',
    'project_measurement_jacobian',
    'project_measurement_vector',
    'create_synthetic_data',
    'create_custom_phantom',
    'MeasurementDataset'
]
