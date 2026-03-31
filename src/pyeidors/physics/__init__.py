"""Physics helpers for DOLFINx-based EIT modeling."""

from .current_drive import (
    build_stim_currents,
    normalize_drive_mode,
    normalize_pattern_config_for_mesh,
    resolve_electrode_lengths_m,
    validate_drive_config,
)
from .unit_consistency import (
    UnitCheckItem,
    UnitCheckLevel,
    UnitCheckReport,
    run_unit_consistency_checks,
)

__all__ = [
    "build_stim_currents",
    "normalize_drive_mode",
    "normalize_pattern_config_for_mesh",
    "resolve_electrode_lengths_m",
    "validate_drive_config",
    "UnitCheckItem",
    "UnitCheckLevel",
    "UnitCheckReport",
    "run_unit_consistency_checks",
]
