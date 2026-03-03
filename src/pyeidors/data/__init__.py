"""PyEIDORS data processing module"""

from .structures import (
    PatternConfig,
    EITData, 
    EITImage,
    EITMesh,
    MeshConfig,
    ElectrodePosition
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
    'create_synthetic_data',
    'create_custom_phantom',
    'MeasurementDataset'
]
