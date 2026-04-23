"""Data models for the EIDORS <-> PyEIDORS interop workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from eit_app.models.forward_model_config import ForwardModelConfig


BRIDGE_PACKAGE_FORMAT_V2 = "eidors_pyeidors_bridge_v2"


@dataclass
class EidorsEnvironment:
    """Resolved MATLAB + EIDORS runtime environment."""

    name: str
    matlab_command: str = ""
    matlab_root: str = ""
    eidors_startup: str = ""
    source: str = "detected"
    last_script_path: str = ""
    last_output_dir: str = ""
    matlab_host_os: str = ""
    startup_host_os: str = ""
    runtime_kind: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "EidorsEnvironment":
        return cls(**{key: mapping.get(key, "") for key in cls.__dataclass_fields__})


@dataclass
class ReconstructionPreset:
    """Optional reconstruction defaults exchanged across frameworks."""

    method: str = "gn-difference"
    regularization_alpha: float = 1.0
    max_iterations: int = 10
    difference_mode: str = "raw"
    difference_orientation: str = "target_minus_reference"

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InteropCapabilityReport:
    """Environment or package capability report."""

    can_launch_matlab: bool = False
    has_eidors_startup: bool = False
    can_capture_script: bool = False
    can_import_geometry: bool = False
    can_import_measurements: bool = False
    matlab_found_count: int = 0
    startup_found_count: int = 0
    toolbox_startup_found: bool = False
    broadened_search_used: bool = False
    broadened_startup_found: bool = False
    manual_browse_required: bool = False
    issues: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InteropBridgeManifest:
    """High-level bridge package manifest."""

    exchange_format: str = BRIDGE_PACKAGE_FORMAT_V2
    source_framework: str = "pyeidors"
    package_kind: str = "bridge"
    created_at_utc: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    script_path: str = ""
    script_kind: str = ""
    script_hints: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, Any] | None = None
    ) -> "InteropBridgeManifest":
        raw = dict(mapping or {})
        return cls(
            exchange_format=str(raw.get("exchange_format", BRIDGE_PACKAGE_FORMAT_V2)),
            source_framework=str(raw.get("source_framework", "pyeidors")),
            package_kind=str(raw.get("package_kind", "bridge")),
            created_at_utc=str(raw.get("created_at_utc", "")),
            environment=dict(raw.get("environment", {})),
            capabilities=dict(raw.get("capabilities", {})),
            files=dict(raw.get("files", {})),
            script_path=str(raw.get("script_path", "")),
            script_kind=str(raw.get("script_kind", "")),
            script_hints=dict(raw.get("script_hints", {})),
            notes=[str(item) for item in raw.get("notes", [])],
        )


@dataclass
class EidorsImportPreview:
    """Normalized import preview shown before applying bridge data."""

    forward_model_config: ForwardModelConfig
    capability_report: InteropCapabilityReport
    geometry_summary: dict[str, str] = field(default_factory=dict)
    measurement_summary: dict[str, str] = field(default_factory=dict)
    recognized_fields: dict[str, Any] = field(default_factory=dict)
    inferred_fields: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class EidorsExportJob:
    """Export request for generating an EIDORS bridge project."""

    source_kind: str = "simulation"
    output_dir: str = ""
    include_geometry: bool = True
    include_measurements: bool = True
    include_scripts: bool = True
    source_name: str = ""
