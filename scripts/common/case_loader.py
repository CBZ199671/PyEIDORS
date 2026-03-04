"""Input frame loaders used by unified reconstruction runners."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .io_utils import load_csv_measurements, load_single_frame


def _select_complex_part(real: np.ndarray, imag: np.ndarray, use_part: str) -> np.ndarray:
    if use_part == "real":
        return real
    if use_part == "imag":
        return imag
    if use_part == "mag":
        return np.abs(real + 1j * imag)
    raise ValueError(f"Unsupported use_part={use_part}")


def _reshape_frame_matrix(
    arr: np.ndarray,
    *,
    layout: str,
    expected_len: Optional[int],
    n_stim: Optional[int],
    n_meas_per_stim: Optional[int],
) -> np.ndarray:
    if layout == "vector":
        return arr.reshape(-1)

    if layout in {"stim-meas", "meas-stim"}:
        if n_stim is None or n_meas_per_stim is None:
            raise ValueError(
                "stim-meas/meas-stim layout requires n_stim and n_meas_per_stim context."
            )
        expected_shape = (
            (n_stim, n_meas_per_stim)
            if layout == "stim-meas"
            else (n_meas_per_stim, n_stim)
        )
        if arr.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape} for {layout}, got {arr.shape}")
        return arr.reshape(-1) if layout == "stim-meas" else arr.T.reshape(-1)

    if layout != "auto":
        raise ValueError(f"Unsupported frame layout: {layout}")

    if expected_len is not None and arr.size == expected_len:
        if n_stim is not None and n_meas_per_stim is not None:
            if arr.shape == (n_stim, n_meas_per_stim):
                return arr.reshape(-1)
            if arr.shape == (n_meas_per_stim, n_stim):
                return arr.T.reshape(-1)
        return arr.reshape(-1)

    if 1 in arr.shape:
        return arr.reshape(-1)

    raise ValueError(
        f"Cannot infer frame layout from shape {arr.shape}. "
        "Use --frame-layout to specify explicit interpretation."
    )


def load_frame_csv(
    csv_path: Path,
    *,
    measurement_gain: float,
    layout: str,
    use_part: str,
    expected_len: Optional[int] = None,
    n_stim: Optional[int] = None,
    n_meas_per_stim: Optional[int] = None,
) -> np.ndarray:
    """Load one measurement frame from CSV.

    Supported CSV representations:
    - 1D vector
    - 2D matrix interpreted by layout
    - two-column/two-row real-imag matrix for `use_part` selection
    """
    arr = np.loadtxt(csv_path, delimiter=",")
    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 0:
        raise ValueError(f"CSV {csv_path.name} contains a single value.")

    if arr.ndim == 1:
        frame = arr
    elif arr.ndim == 2:
        # Explicit frame layouts always take precedence over complex-part parsing.
        if layout != "auto":
            frame = _reshape_frame_matrix(
                arr,
                layout=layout,
                expected_len=expected_len,
                n_stim=n_stim,
                n_meas_per_stim=n_meas_per_stim,
            )
        elif arr.shape[1] == 2:
            frame = _select_complex_part(arr[:, 0], arr[:, 1], use_part)
        elif arr.shape[0] == 2:
            frame = _select_complex_part(arr[0], arr[1], use_part)
        else:
            frame = _reshape_frame_matrix(
                arr,
                layout=layout,
                expected_len=expected_len,
                n_stim=n_stim,
                n_meas_per_stim=n_meas_per_stim,
            )
    else:
        raise ValueError(f"Unsupported CSV shape {arr.shape} in {csv_path.name}")

    if expected_len is not None and frame.shape[0] != expected_len:
        raise ValueError(
            f"Frame length {frame.shape[0]} does not match expected {expected_len}."
        )

    if measurement_gain != 1.0:
        frame = frame / measurement_gain

    return frame.reshape(-1)


def load_paired_frames(
    csv_path: Path,
    *,
    use_part: str,
    measurement_gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load reference/target frames from a paired CSV."""
    return load_csv_measurements(
        csv_path=csv_path,
        use_part=use_part,
        measurement_gain=measurement_gain,
    )


def load_absolute_frame_from_paired_csv(
    csv_path: Path,
    *,
    col_idx: int,
    measurement_gain: float,
) -> np.ndarray:
    """Load one frame from a multi-column CSV used in absolute reconstruction."""
    return load_single_frame(
        csv_path=csv_path,
        col_idx=col_idx,
        measurement_gain=measurement_gain,
    )
