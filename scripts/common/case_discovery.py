"""Case discovery utilities for unified reconstruction CLI."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from .recon_cli_models import InputMode, ReconstructionCase


def collect_csv_files(
    *,
    input_dir: Optional[Path],
    glob_pattern: str,
    csv_files: Optional[Sequence[Path]],
    include_ad: bool,
) -> List[Path]:
    """Collect CSV files from explicit paths and an optional directory."""
    candidates: List[Path] = []
    if csv_files:
        candidates.extend(csv_files)
    if input_dir is not None:
        candidates.extend(sorted(input_dir.expanduser().glob(glob_pattern)))

    resolved: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        csv_path = path.expanduser()
        if csv_path.suffix.lower() != ".csv":
            continue
        if not include_ad and csv_path.stem.endswith("_AD"):
            continue
        key = str(csv_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(csv_path)
    return sorted(resolved, key=lambda p: p.name)


def _resolve_reference_path(
    input_files: Sequence[Path],
    reference_csv: Optional[Path],
    reference_index: Optional[int],
) -> Path:
    if reference_csv is not None and reference_index is not None:
        raise ValueError("Use only one of reference_csv or reference_index in frame mode.")
    if reference_csv is None and reference_index is None:
        raise ValueError(
            "Frame mode requires reference_csv or reference_index for difference reconstruction."
        )

    if reference_index is not None:
        if reference_index < 0 or reference_index >= len(input_files):
            raise IndexError("reference_index is out of range for discovered input files.")
        return input_files[reference_index]

    resolved = reference_csv.expanduser() if reference_csv is not None else None
    if resolved is None or not resolved.exists():
        raise FileNotFoundError("reference_csv does not exist.")
    return resolved


def build_cases(
    *,
    input_mode: InputMode,
    input_files: Sequence[Path],
    require_reference: bool,
    reference_csv: Optional[Path],
    reference_index: Optional[int],
) -> List[ReconstructionCase]:
    """Build per-case descriptors from resolved input files."""
    if not input_files:
        return []

    if input_mode == InputMode.PAIRED:
        return [
            ReconstructionCase(
                case_name=path.stem,
                input_mode=InputMode.PAIRED,
                paired_csv=path,
            )
            for path in input_files
        ]

    if not require_reference:
        return [
            ReconstructionCase(
                case_name=path.stem,
                input_mode=InputMode.FRAME,
                target_csv=path,
            )
            for path in input_files
        ]

    reference_path = _resolve_reference_path(
        input_files=input_files,
        reference_csv=reference_csv,
        reference_index=reference_index,
    )

    targets = [p for p in input_files if p.resolve() != reference_path.resolve()]
    if not targets:
        raise ValueError("No target CSV files found after removing reference frame.")

    return [
        ReconstructionCase(
            case_name=path.stem,
            input_mode=InputMode.FRAME,
            target_csv=path,
            reference_csv=reference_path,
        )
        for path in targets
    ]
