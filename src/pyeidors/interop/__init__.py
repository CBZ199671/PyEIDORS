"""Interop helpers for exchanging native geometries with EIDORS.

MATLAB exchange helpers import SciPy I/O and FEM utilities.  Keep package import
light and resolve helpers only when interop operations are requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "BRIDGE_PACKAGE_FORMAT_V3": ".bridge_v3",
    "BRIDGE_SCHEMA_VERSION": ".bridge_v3",
    "BridgeV3Package": ".bridge_v3",
    "ElectrodeSpec": ".bridge_v3",
    "GEOMETRY_FORMAT_V3": ".bridge_v3",
    "ProtocolSpec": ".bridge_v3",
    "ActualCurrentResolution": ".protocol_mapping",
    "ProtocolChannelMapping": ".protocol_mapping",
    "build_bridge_fingerprints": ".bridge_v3",
    "validate_bridge_v3_package": ".bridge_v3",
    "MODEL_BINDING_FLOWS": ".model_registry",
    "MODEL_REGISTRY_SCHEMA_VERSION": ".model_registry",
    "ModelContext": ".model_registry",
    "ModelContextFactory": ".model_registry",
    "ModelRegistry": ".model_registry",
    "RegisteredModel": ".model_registry",
    "prove_protocol_mapping": ".protocol_mapping",
    "resolve_actual_stimulation": ".protocol_mapping",
    "LEGACY_INTEROP_FORMAT": ".geometry_exchange",
    "STANDARD_INTEROP_FORMAT": ".geometry_exchange",
    "STANDARD_INTEROP_FORMAT_V2": ".geometry_exchange",
    "STANDARD_INTEROP_FORMAT_V3": ".geometry_exchange",
    "SUPPORTED_INTEROP_FORMATS": ".geometry_exchange",
    "build_boundary_facets": ".geometry_exchange",
    "build_boundary_edges": ".geometry_exchange",
    "build_electrode_arrays": ".geometry_exchange",
    "electrode_specs_from_exchange_payload": ".geometry_exchange",
    "build_mesh_from_exchange_mat": ".geometry_exchange",
    "export_forward_csv": ".geometry_exchange",
    "load_forward_csv": ".geometry_exchange",
    "save_exchange_mat": ".geometry_exchange",
    "source_cell_data_to_local": ".geometry_exchange",
    "validate_exchange_payload": ".geometry_exchange",
}

__all__ = [
    "BRIDGE_PACKAGE_FORMAT_V3",
    "BRIDGE_SCHEMA_VERSION",
    "BridgeV3Package",
    "ElectrodeSpec",
    "GEOMETRY_FORMAT_V3",
    "ProtocolSpec",
    "ActualCurrentResolution",
    "ProtocolChannelMapping",
    "build_bridge_fingerprints",
    "validate_bridge_v3_package",
    "MODEL_BINDING_FLOWS",
    "MODEL_REGISTRY_SCHEMA_VERSION",
    "ModelContext",
    "ModelContextFactory",
    "ModelRegistry",
    "RegisteredModel",
    "prove_protocol_mapping",
    "resolve_actual_stimulation",
    "LEGACY_INTEROP_FORMAT",
    "STANDARD_INTEROP_FORMAT",
    "STANDARD_INTEROP_FORMAT_V2",
    "STANDARD_INTEROP_FORMAT_V3",
    "SUPPORTED_INTEROP_FORMATS",
    "build_boundary_facets",
    "build_boundary_edges",
    "build_electrode_arrays",
    "electrode_specs_from_exchange_payload",
    "build_mesh_from_exchange_mat",
    "export_forward_csv",
    "load_forward_csv",
    "save_exchange_mat",
    "source_cell_data_to_local",
    "validate_exchange_payload",
]

_SUBMODULE_NAMES = frozenset(
    {"bridge_v3", "geometry_exchange", "model_registry", "protocol_mapping"}
)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        module = import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULE_NAMES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULE_NAMES))
