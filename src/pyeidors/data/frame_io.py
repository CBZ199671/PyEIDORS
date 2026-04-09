"""Per-frame CSV + YAML I/O for the streaming acquisition format.

File naming convention::

    session_20260409_143000/
      session_metadata.yaml            # shared session metadata (n_elec, patterns, etc.)
      20260409_143000_frame_000.csv    # 208 rows, 2 columns (real, imag)
      20260409_143000_frame_000.yaml   # per-frame metadata
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def write_frame_csv(path: Path, real: np.ndarray, imag: np.ndarray) -> None:
    """Write a 2-column (real, imag) CSV with no header.

    Arrays must be 1-D with the same length.  Values are written in
    ``%.15e`` format for full float64 round-trip fidelity.
    """
    real = np.asarray(real, dtype=np.float64).ravel()
    imag = np.asarray(imag, dtype=np.float64).ravel()
    if real.shape != imag.shape:
        raise ValueError(
            f"real and imag shapes must match: {real.shape} vs {imag.shape}"
        )
    data = np.column_stack([real, imag])
    np.savetxt(Path(path), data, delimiter=",", fmt="%.15e")


def read_frame_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a frame CSV and return ``(real, imag)`` float64 arrays.

    Supports:
    - New format: 2 columns ``(real, imag)``
    - Legacy format: 4 columns ``(real_v0, imag_v0, real, imag)``
      In this case the target-frame columns are returned.
    """
    data = np.loadtxt(Path(path), delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] == 4:
        return data[:, 2].copy(), data[:, 3].copy()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2 or 4 columns, got {data.shape[1]}")
    return data[:, 0].copy(), data[:, 1].copy()


def read_legacy_frame_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a legacy 4-column CSV.

    Returns ``(real_v0, imag_v0, real, imag)``.
    """
    data = np.loadtxt(Path(path), delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(f"Expected 4 columns, got {data.shape[1]}")
    return (
        data[:, 0].copy(),
        data[:, 1].copy(),
        data[:, 2].copy(),
        data[:, 3].copy(),
    )


def write_frame_yaml(path: Path, metadata: dict[str, Any]) -> None:
    """Write per-frame YAML sidecar."""
    Path(path).write_text(
        yaml.dump(metadata, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def read_frame_yaml(path: Path) -> dict[str, Any]:
    """Read per-frame YAML sidecar."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def write_session_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write shared ``session_metadata.yaml``."""
    write_frame_yaml(path, metadata)


def read_session_metadata(path: Path) -> dict[str, Any]:
    """Read ``session_metadata.yaml``."""
    return read_frame_yaml(path)


_FRAME_RE = re.compile(r"_frame_(\d+)\.csv$")


def scan_frame_dir(dir_path: Path) -> list[tuple[Path, Path]]:
    """Find all ``(csv, yaml)`` frame pairs in a session directory.

    Looks for files matching ``*_frame_NNN.csv`` with a corresponding
    ``*_frame_NNN.yaml`` sidecar.  Returns pairs sorted by frame index.
    """
    dir_path = Path(dir_path)
    pairs: list[tuple[int, Path, Path]] = []
    for csv_path in sorted(dir_path.glob("*_frame_*.csv")):
        m = _FRAME_RE.search(csv_path.name)
        if m is None:
            continue
        yaml_path = csv_path.with_suffix(".yaml")
        if yaml_path.exists():
            pairs.append((int(m.group(1)), csv_path, yaml_path))
    pairs.sort(key=lambda t: t[0])
    return [(csv, yml) for _, csv, yml in pairs]
