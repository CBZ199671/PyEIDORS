"""Data input/output utility functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML required, install with: pip install pyyaml") from exc


def load_metadata(path: Path) -> Dict[str, Any]:
    """Load YAML or JSON format metadata file.

    Args:
        path: Metadata file path (.yaml, .yml, .json).

    Returns:
        Parsed metadata dictionary.
    """
    with path.open("r", encoding="utf-8") as fh:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        if suffix == ".json":
            return json.load(fh)
        raise ValueError(f"Unsupported metadata format: {suffix}")


def load_csv_measurements(
    csv_path: Path,
    use_part: str = "real",
    measurement_gain: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load EIT measurement CSV file.

    Expected CSV format: 4 columns (ref_real, ref_imag, target_real, target_imag).

    Args:
        csv_path: CSV file path.
        use_part: Which part of data to use ("real", "imag", "mag").
        measurement_gain: Measurement amplifier gain, data will be divided by this value.

    Returns:
        (reference_frame, target_frame) two frames of measurement data.
    """
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected 4-column CSV, actual shape: {arr.shape}")

    ref_re, ref_im, tgt_re, tgt_im = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    if use_part == "real":
        ref, tgt = ref_re, tgt_re
    elif use_part == "imag":
        ref, tgt = ref_im, tgt_im
    elif use_part == "mag":
        ref = np.abs(ref_re + 1j * ref_im)
        tgt = np.abs(tgt_re + 1j * tgt_im)
    else:
        raise ValueError(f"Unknown use_part={use_part}, options: real, imag, mag")

    if measurement_gain != 1.0:
        ref = ref / measurement_gain
        tgt = tgt / measurement_gain

    return ref, tgt


def load_single_frame(
    csv_path: Path,
    col_idx: int,
    measurement_gain: float = 1.0,
) -> np.ndarray:
    """Load single frame measurement data from CSV.

    Args:
        csv_path: CSV file path.
        col_idx: Column index (0-based).
        measurement_gain: Measurement amplifier gain.

    Returns:
        Single frame measurement vector.
    """
    raw = np.loadtxt(csv_path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] <= col_idx:
        raise ValueError(f"CSV shape {raw.shape} cannot extract column {col_idx}")

    frame = raw[:, col_idx].astype(float)
    if measurement_gain != 1.0:
        frame = frame / measurement_gain
    return frame


def align_measurement_polarity(
    measurement: np.ndarray,
    baseline_vector: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Correct measurement polarity based on baseline voltage direction (U-shape detection).

    With adjacent stimulation patterns, voltage should be "normal U-shaped" (high at ends, low in middle).
    If measurement dot product with baseline is negative, it's "inverted U-shaped" and needs to be flipped.

    Args:
        measurement: Single frame measurement vector.
        baseline_vector: Baseline voltage vector (typically model-predicted uniform field voltage).

    Returns:
        (corrected measurement, whether flipped).
    """
    baseline = np.asarray(baseline_vector, dtype=float)
    frame = np.asarray(measurement, dtype=float)

    if np.linalg.norm(baseline) < np.finfo(float).eps:
        return frame, False

    # Negative dot product means opposite direction, need to flip
    if float(np.dot(frame, baseline)) < 0:
        return -frame, True
    return frame, False


def align_frames_polarity(frames: np.ndarray, baseline_vector: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Batch align polarity for multiple frames.

    Args:
        frames: (n_frames, n_meas) or (n_meas, n_frames) both accepted.
        baseline_vector: Baseline voltage vector.

    Returns:
        (aligned frames, list of flipped frame indices).
    """
    arr = np.asarray(frames, dtype=float)
    baseline_len = np.asarray(baseline_vector, dtype=float).shape[0]
    # Flatten to at least 2D for uniform processing
    if arr.ndim == 1:
        arr = arr[None, :]
    flipped: list[int] = []
    # Keep internal format as (n_frames, n_meas)
    transposed = False
    if arr.shape[1] == baseline_len:
        pass  # already (n_frames, n_meas)
    elif arr.shape[0] == baseline_len:
        arr = arr.T  # Input is (n_meas, n_frames), transpose to (n_frames, n_meas)
        transposed = True
    else:
        raise ValueError(f"frames shape {arr.shape} does not match baseline length {baseline_len}")
    aligned = arr.copy()
    for i in range(aligned.shape[0]):
        aligned[i], was_flipped = align_measurement_polarity(aligned[i], baseline_vector)
        if was_flipped:
            flipped.append(i)
    if transposed:
        aligned = aligned.T
    return aligned, flipped


def save_reconstruction_results(
    output_dir: Path,
    conductivity: np.ndarray,
    measured: np.ndarray,
    predicted: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save reconstruction results to files.

    Args:
        output_dir: Output directory.
        conductivity: Reconstructed conductivity vector.
        measured: Measured voltage.
        predicted: Predicted voltage.
        metadata: Optional run metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save numerical data
    np.savez(
        output_dir / "result_arrays.npz",
        conductivity=conductivity,
        measured=measured,
        predicted=predicted,
        residual=np.asarray(predicted) - np.asarray(measured),
    )

    # Save metadata
    if metadata is not None:
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)
