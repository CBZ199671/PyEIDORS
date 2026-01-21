"""PyEIDORS Data Structure Definitions."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Any


@dataclass
class PatternConfig:
    """Stimulation and measurement pattern configuration."""
    n_elec: int
    n_rings: int = 1
    stim_pattern: Union[str, List[int]] = '{ad}'
    meas_pattern: Union[str, List[int]] = '{ad}'
    amplitude: float = 1.0
    use_meas_current: bool = False
    use_meas_current_next: int = 0
    rotate_meas: bool = True
    stim_direction: str = 'ccw'  # 'ccw' or 'cw'
    meas_direction: str = 'ccw'
    stim_first_positive: bool = False


@dataclass
class EITData:
    """EIT data container."""
    meas: np.ndarray
    stim_pattern: np.ndarray
    n_elec: int
    n_stim: int
    n_meas: int
    type: str = 'real'


@dataclass
class EITImage:
    """EIT image container."""
    elem_data: np.ndarray
    fwd_model: Any
    type: str = 'conductivity'
    name: str = ''

    def get_conductivity(self) -> np.ndarray:
        if self.type == 'resistivity':
            return 1.0 / self.elem_data
        return self.elem_data


@dataclass
class MeshConfig:
    """Mesh configuration parameters."""
    radius: float = 1.0
    refinement: int = 8
    electrode_vertices: int = 8
    gap_vertices: int = 4
    mesh_size: float = 0.1


@dataclass
class ElectrodePosition:
    """Electrode position information."""
    L: int  # Number of electrodes
    positions: List[Tuple[float, float]]  # Electrode angular positions (start, end)

    @classmethod
    def create_circular(cls, n_elec: int = 16, radius: float = 1.0) -> 'ElectrodePosition':
        """Create circular electrode positions."""
        import math

        # Calculate electrode coverage angle
        electrode_width = 2 * math.pi / n_elec / 4  # Each electrode covers 1/4 of segment
        gap_width = 2 * math.pi / n_elec * 3 / 4    # Gap covers 3/4 of segment

        positions = []
        for i in range(n_elec):
            center_angle = 2 * math.pi * i / n_elec
            start_angle = center_angle - electrode_width / 2
            end_angle = center_angle + electrode_width / 2
            positions.append((start_angle, end_angle))

        return cls(L=n_elec, positions=positions)
