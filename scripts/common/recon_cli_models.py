"""Shared data models for the unified reconstruction CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ReconstructionMethod(str, Enum):
    """Supported reconstruction methods."""

    GN_ABSOLUTE = "gn-absolute"
    GN_DIFFERENCE = "gn-difference"
    SPARSE_BAYES = "sparse-bayes"


class InputMode(str, Enum):
    """Input organization mode."""

    PAIRED = "paired"
    FRAME = "frame"


@dataclass(frozen=True)
class ReconstructionCase:
    """Resolved reconstruction case for one output folder."""

    case_name: str
    input_mode: InputMode
    paired_csv: Optional[Path] = None
    target_csv: Optional[Path] = None
    reference_csv: Optional[Path] = None

    def primary_path(self) -> Path:
        """Return the primary input path used to identify this case."""
        if self.input_mode == InputMode.PAIRED:
            if self.paired_csv is None:
                raise ValueError("paired case missing paired_csv")
            return self.paired_csv
        if self.target_csv is None:
            raise ValueError("frame case missing target_csv")
        return self.target_csv

    def to_dict(self) -> Dict[str, Any]:
        """Serialize case metadata for summaries and dry-runs."""
        data = {
            "case_name": self.case_name,
            "input_mode": self.input_mode.value,
            "paired_csv": str(self.paired_csv) if self.paired_csv else None,
            "target_csv": str(self.target_csv) if self.target_csv else None,
            "reference_csv": str(self.reference_csv) if self.reference_csv else None,
        }
        return data


@dataclass
class CaseResult:
    """Per-case execution result."""

    case_name: str
    status: str
    output_dir: Optional[Path] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result payload to JSON-friendly dict."""
        data = asdict(self)
        if self.output_dir is not None:
            data["output_dir"] = str(self.output_dir)
        return data
