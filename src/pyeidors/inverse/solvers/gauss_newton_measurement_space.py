"""Measurement-space configuration + projection helpers for the GN runtime.

T77 phase 2 commit #3 — third sub-module split lifted out of
``gauss_newton_runtime.py``. Owns the small bundle of helpers that
dispatch between absolute (``"real"``) and difference EIT
measurement spaces:

* ``_extract_measured_vector`` — coerce ``EITData`` / ndarray input
  into a 1-D measurement vector, preserving native complex phasors.
* ``_configure_measurement_space`` — read ``measured_data`` and
  populate the per-iteration measurement-space state on the
  reconstructor (``_measurement_space_type`` / ``_difference_*_meas``
  / ``_difference_mode_effective`` / ``_difference_orientation_effective``).
* ``_measurement_space_kwargs`` — assemble the keyword bundle that
  the difference projection functions consume.
* ``_project_simulated_measurements`` / ``_project_measurement_jacobian``
  — thin wrappers around ``data.difference.project_measurement_vector``
  / ``project_measurement_jacobian`` using the kwargs bundle.

The helpers mutate reconstructor state via attribute writes; their
attribute names + the difference-space dispatch are part of the
V73-style contract frozen by
``test_gn_runtime_contract_freeze``. ``gauss_newton_runtime``
re-exports all five symbols at module scope so existing call sites
and any external monkeypatch consumer keep their import paths.
"""

from __future__ import annotations

import numpy as np

from ...data.difference import (
    normalize_difference_mode,
    normalize_difference_orientation,
    project_measurement_jacobian,
    project_measurement_vector,
)


def _measurement_vector_dtype(values) -> np.dtype:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return np.dtype(np.complex64 if arr.dtype == np.complex64 else np.complex128)
    return np.dtype(np.float64)


def _extract_measured_vector(measured_data) -> np.ndarray:
    if hasattr(measured_data, "meas"):
        values = measured_data.meas
    else:
        values = measured_data
    return np.asarray(values, dtype=_measurement_vector_dtype(values)).reshape(-1)


def _configure_measurement_space(reconstructor, measured_data) -> None:
    measurement_type = str(getattr(measured_data, "type", "real")).strip().lower()
    reference_meas_raw = getattr(measured_data, "reference_meas", None)
    target_meas_raw = getattr(measured_data, "target_meas", None)
    reference_meas = (
        np.asarray(
            reference_meas_raw,
            dtype=_measurement_vector_dtype(reference_meas_raw),
        ).reshape(-1)
        if reference_meas_raw is not None
        else None
    )
    target_meas = (
        np.asarray(
            target_meas_raw, dtype=_measurement_vector_dtype(target_meas_raw)
        ).reshape(-1)
        if target_meas_raw is not None
        else None
    )

    if measurement_type == "difference" and reference_meas is not None:
        reconstructor._measurement_space_type = "difference"
        reconstructor._difference_reference_meas = reference_meas.copy()
        reconstructor._difference_target_meas = (
            target_meas.copy() if target_meas is not None else None
        )
        reconstructor._difference_mode_effective = normalize_difference_mode(
            getattr(measured_data, "difference_mode", reconstructor.difference_mode),
            default=reconstructor.difference_mode,
        )
        reconstructor._difference_orientation_effective = (
            normalize_difference_orientation(
                getattr(
                    measured_data,
                    "difference_orientation",
                    reconstructor.difference_orientation,
                ),
                default=reconstructor.difference_orientation,
            )
        )
        return

    reconstructor._measurement_space_type = "real"
    reconstructor._difference_reference_meas = None
    reconstructor._difference_target_meas = None
    reconstructor._difference_mode_effective = reconstructor.difference_mode
    reconstructor._difference_orientation_effective = (
        reconstructor.difference_orientation
    )


def _measurement_space_kwargs(reconstructor) -> dict[str, object]:
    """Common keyword arguments for measurement projection functions."""
    return {
        "measurement_type": getattr(reconstructor, "_measurement_space_type", "real"),
        "reference_meas": getattr(reconstructor, "_difference_reference_meas", None),
        "difference_mode": getattr(
            reconstructor, "_difference_mode_effective", reconstructor.difference_mode
        ),
        "difference_orientation": getattr(
            reconstructor,
            "_difference_orientation_effective",
            reconstructor.difference_orientation,
        ),
    }


def _project_simulated_measurements(
    reconstructor, simulated_meas: np.ndarray
) -> np.ndarray:
    return project_measurement_vector(
        simulated_meas, **_measurement_space_kwargs(reconstructor)
    )


def _project_measurement_jacobian(reconstructor, jacobian: np.ndarray) -> np.ndarray:
    return project_measurement_jacobian(
        jacobian, **_measurement_space_kwargs(reconstructor)
    )
