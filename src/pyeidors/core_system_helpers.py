"""Helper routines for :mod:`pyeidors.core_system`.

These functions keep :class:`EITSystem` focused on orchestration while
preserving runtime behavior.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from dolfinx import fem

from .data.difference import (
    build_difference_vector,
    normalize_difference_mode,
    normalize_difference_orientation,
)
from .data.structures import EITData, EITImage


def conductivity_to_image(fwd_model, conductivity: Union[np.ndarray, fem.Function, EITImage]) -> EITImage:
    """Normalize conductivity inputs to ``EITImage`` for forward solving."""
    if isinstance(conductivity, EITImage):
        return conductivity
    if isinstance(conductivity, fem.Function):
        return EITImage(elem_data=conductivity.x.array.copy(), fwd_model=fwd_model)
    if isinstance(conductivity, np.ndarray):
        return EITImage(elem_data=conductivity, fwd_model=fwd_model)
    raise ValueError("Unsupported conductivity input type")


def difference_measurement(
    data: EITData,
    reference_data: Optional[EITData],
    *,
    mode: str = "raw",
    orientation: str = "target_minus_reference",
) -> EITData:
    """Build difference-mode measurements when reference data is provided."""
    if reference_data is None:
        return data
    resolved_mode = normalize_difference_mode(mode)
    resolved_orientation = normalize_difference_orientation(orientation)
    return EITData(
        meas=build_difference_vector(
            data.meas,
            reference_data.meas,
            mode=resolved_mode,
            orientation=resolved_orientation,
        ),
        stim_pattern=data.stim_pattern,
        n_elec=data.n_elec,
        n_stim=data.n_stim,
        n_meas=data.n_meas,
        type="difference",
        reference_meas=np.asarray(reference_data.meas, dtype=np.float64).copy(),
        target_meas=np.asarray(data.meas, dtype=np.float64).copy(),
        difference_mode=resolved_mode,
        difference_orientation=resolved_orientation,
    )


def create_homogeneous_image(eit_system, conductivity: Optional[float] = None) -> EITImage:
    """Create a homogeneous conductivity image for an initialized system."""
    value = eit_system.base_conductivity if conductivity is None else float(conductivity)
    n_elements = int(fem.Function(eit_system.fwd_model.V_sigma).x.array.size)
    elem_data = np.full(n_elements, value, dtype=float)
    return EITImage(elem_data=elem_data, fwd_model=eit_system.fwd_model)


def add_circular_phantom(
    eit_system,
    *,
    base_conductivity: float,
    phantom_conductivity: float,
    phantom_center: Tuple[float, float],
    phantom_radius: float,
) -> EITImage:
    """Create an image with a circular phantom anomaly."""
    dof_coordinates = eit_system.fwd_model.V_sigma.tabulate_dof_coordinates()
    elem_data = np.full(len(dof_coordinates), base_conductivity, dtype=float)

    center = np.asarray(phantom_center, dtype=float)
    distances = np.linalg.norm(dof_coordinates[:, :2] - center[None, :], axis=1)
    elem_data[distances <= phantom_radius] = phantom_conductivity
    return EITImage(elem_data=elem_data, fwd_model=eit_system.fwd_model)


def collect_system_info(eit_system) -> Dict[str, Any]:
    """Return stable, structured runtime diagnostics for ``EITSystem``."""
    info: Dict[str, Any] = {
        "n_elec": eit_system.n_elec,
        "pattern_config": eit_system.pattern_config,
        "mesh_config": eit_system.mesh_config,
        "difference_mode": getattr(eit_system, "difference_mode", "raw"),
        "difference_orientation": getattr(
            eit_system,
            "difference_orientation",
            "target_minus_reference",
        ),
        "difference_preset": getattr(eit_system, "difference_preset", "eidors_one_step_noser"),
        "absolute_preset": getattr(eit_system, "absolute_preset", "eidors_abs_gn"),
        "hyperparameter": getattr(eit_system, "hyperparameter", None),
        "jacobian_background_conductivity": getattr(
            eit_system,
            "jacobian_background_conductivity",
            getattr(eit_system, "base_conductivity", 1.0),
        ),
        "performance_mode": getattr(eit_system, "performance_mode", "aggressive"),
        "linear_backend": getattr(eit_system, "linear_backend", "scipy"),
        "cache_scope": getattr(eit_system, "cache_scope", "off"),
        "cache_stats": eit_system.get_cache_stats() if hasattr(eit_system, "get_cache_stats") else {},
        "initialized": eit_system._is_initialized,
    }
    if not eit_system._is_initialized:
        return info

    info.update(
        {
            "n_elements": int(fem.Function(eit_system.fwd_model.V_sigma).x.array.size),
            "n_nodes": int(fem.Function(eit_system.fwd_model.V).x.array.size),
            "n_measurements": eit_system.fwd_model.pattern_manager.n_meas_total,
            "n_stimulation_patterns": eit_system.fwd_model.pattern_manager.n_stim,
        }
    )
    return info
