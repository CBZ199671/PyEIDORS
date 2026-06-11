"""Global compute-precision toggle (float32 vs float64).

The hardware ADC delivers ~7 effective bits of precision, so float32
(~24 mantissa bits) is more than sufficient to represent measurement
samples without loss.  float64 doubles memory and slows vectorised
math without buying any extra accuracy at the input boundary.

The setting is persisted via QSettings under ``compute/precision`` and
is wired into the ADC parser, the forward solver result, and the
reconstruction result.  Internal solver stages (FEM assembly, linear
solves, Jacobian) keep their native dtypes — downcasting those would
hurt reconstruction quality on a problem that is already
ill-conditioned.

Use :func:`compute_dtype` everywhere boundary-data arrays are sized.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSettings


_SETTINGS_KEY = "compute/precision"
_DEFAULT_PRECISION = "float32"
_VALID = ("float32", "float64")

_current: str = _DEFAULT_PRECISION


def current_precision() -> str:
    """Return the active precision label ('float32' or 'float64')."""
    return _current


def compute_dtype() -> np.dtype:
    """Return the numpy dtype matching the active precision."""
    return np.dtype(np.float32) if _current == "float32" else np.dtype(np.float64)


def set_precision(mode: str, *, persist: bool = True) -> None:
    """Switch the active precision."""
    global _current
    mode = mode if mode in _VALID else _DEFAULT_PRECISION
    if mode == _current:
        return
    _current = mode
    if persist:
        QSettings("PyEIDORS", "EITWorkstation").setValue(_SETTINGS_KEY, mode)


def init_precision_from_settings() -> str:
    """Resolve the persisted precision without firing listeners."""
    global _current
    stored = QSettings("PyEIDORS", "EITWorkstation").value(
        _SETTINGS_KEY, _DEFAULT_PRECISION
    )
    text = str(stored) if stored in _VALID else _DEFAULT_PRECISION
    _current = text
    return text
