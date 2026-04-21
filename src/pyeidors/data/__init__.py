"""PyEIDORS data processing module.

Keep this package import lightweight: pure I/O helpers such as
``pyeidors.data.frame_io`` should not eagerly pull in heavy FEniCSx
dependencies through ``synthetic_data``.
"""

from .structures import (
    PatternConfig,
    EITData,
    EITImage,
    EITMesh,
    MeshConfig,
    ElectrodePosition,
    FrameMetadata,
)
from .difference import (
    DEFAULT_DIFFERENCE_MODE,
    DEFAULT_DIFFERENCE_ORIENTATION,
    build_difference_vector,
    normalize_difference_mode,
    normalize_difference_orientation,
    normalize_time_difference,
    project_measurement_jacobian,
    project_measurement_vector,
)

__all__ = [
    "PatternConfig",
    "EITData",
    "EITImage",
    "EITMesh",
    "MeshConfig",
    "ElectrodePosition",
    "DEFAULT_DIFFERENCE_MODE",
    "DEFAULT_DIFFERENCE_ORIENTATION",
    "build_difference_vector",
    "normalize_difference_mode",
    "normalize_difference_orientation",
    "normalize_time_difference",
    "project_measurement_jacobian",
    "project_measurement_vector",
    "create_synthetic_data",
    "create_custom_phantom",
    "MeasurementDataset",
    "FrameMetadata",
]


def __getattr__(name: str):
    if name == "MeasurementDataset":
        from .measurement_dataset import MeasurementDataset

        globals()["MeasurementDataset"] = MeasurementDataset
        return MeasurementDataset
    if name in {"create_synthetic_data", "create_custom_phantom"}:
        from .synthetic_data import create_custom_phantom, create_synthetic_data

        globals()["create_synthetic_data"] = create_synthetic_data
        globals()["create_custom_phantom"] = create_custom_phantom
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
