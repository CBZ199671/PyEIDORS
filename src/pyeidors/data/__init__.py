"""PyEIDORS data processing module.

Keep this package import lightweight: pure I/O helpers and simple data classes
should not eagerly pull in digit-sweep, reporting, or synthetic-data helpers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".structures": (
        "PatternConfig",
        "EITData",
        "EITImage",
        "EITMesh",
        "MeshConfig",
        "ElectrodePosition",
        "FrameMetadata",
    ),
    ".difference": (
        "DEFAULT_DIFFERENCE_MODE",
        "DEFAULT_DIFFERENCE_ORIENTATION",
        "build_difference_vector",
        "normalize_difference_mode",
        "normalize_difference_orientation",
        "normalize_time_difference",
        "project_measurement_jacobian",
        "project_measurement_vector",
    ),
    ".noise": ("EIDORS_NOISE_NORM_OPTION", "add_noise"),
    ".channels": (
        "MeasurementContract",
        "apply_measurement_contract_to_jacobian",
        "apply_measurement_contract_to_vector",
        "bad_channel_mask",
        "normalize_bad_channel_mask",
        "prepare_measurement_contract",
        "zero_bad_channel_rows",
        "zero_bad_channel_vector",
        "zero_bad_channel_weights",
    ),
    ".adc_quantization": (
        "DEFAULT_BOUNDARY_VOLTAGES",
        "ADCInjectionConfig",
        "ADCQuantizationSummary",
        "adc_lsb",
        "add_voltage_noise",
        "effective_digits_from_rmse",
        "effective_adc_bits",
        "ideal_decimal_digits",
        "inject_adc_measurement",
        "noise_standard_deviation",
        "pointwise_effective_digits",
        "quantize_voltages",
        "rmse",
        "summarize_adc_quantization",
        "summarize_adc_sweep",
    ),
    ".eit_digit_metrics": (
        "EITDigitSummary",
        "EITLinearizedModel",
        "adjacent_measurement_count",
        "build_pyeidors_fem_linearized_model",
        "build_surrogate_sensitivity",
        "build_surrogate_linearized_model",
        "default_sigma_true",
        "forward_surrogate",
        "inverse_pyeidors_rm",
        "inverse_surrogate",
        "reconstruct_linearized_sigma",
        "sigma_true_from_anomaly_rule",
        "summarize_eit_digit_run",
        "summarize_eit_digit_sweep",
    ),
    ".voltage_digit_sweep": (
        "VoltageDigitFieldRow",
        "VoltageDigitSweepSummary",
        "keep_significant_digits",
        "run_voltage_digit_sweep",
        "run_voltage_digit_sweep_from_backend",
    ),
    ".digit_report": (
        "DigitReportCase",
        "DigitReportRow",
        "format_markdown_report",
        "format_markdown_table",
        "read_eit_digit_case",
        "read_eit_digit_cases",
        "write_report_files",
    ),
    ".factor_sweep": (
        "FactorSweepRow",
        "format_factor_sweep_report",
        "normalize_enob_level",
        "run_factor_sweep",
    ),
    ".synthetic_data": ("create_synthetic_data", "create_custom_phantom"),
    ".measurement_dataset": ("MeasurementDataset",),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "PatternConfig",
    "EITData",
    "EITImage",
    "EITMesh",
    "MeshConfig",
    "ElectrodePosition",
    "DEFAULT_DIFFERENCE_MODE",
    "DEFAULT_DIFFERENCE_ORIENTATION",
    "build_difference_vector",
    "normalize_difference_mode",
    "normalize_difference_orientation",
    "normalize_time_difference",
    "project_measurement_jacobian",
    "project_measurement_vector",
    "EIDORS_NOISE_NORM_OPTION",
    "add_noise",
    "MeasurementContract",
    "apply_measurement_contract_to_jacobian",
    "apply_measurement_contract_to_vector",
    "bad_channel_mask",
    "normalize_bad_channel_mask",
    "prepare_measurement_contract",
    "zero_bad_channel_rows",
    "zero_bad_channel_vector",
    "zero_bad_channel_weights",
    "DEFAULT_BOUNDARY_VOLTAGES",
    "ADCInjectionConfig",
    "ADCQuantizationSummary",
    "adc_lsb",
    "add_voltage_noise",
    "effective_digits_from_rmse",
    "effective_adc_bits",
    "ideal_decimal_digits",
    "inject_adc_measurement",
    "noise_standard_deviation",
    "pointwise_effective_digits",
    "quantize_voltages",
    "rmse",
    "summarize_adc_quantization",
    "summarize_adc_sweep",
    "EITDigitSummary",
    "EITLinearizedModel",
    "adjacent_measurement_count",
    "build_pyeidors_fem_linearized_model",
    "build_surrogate_sensitivity",
    "build_surrogate_linearized_model",
    "default_sigma_true",
    "forward_surrogate",
    "inverse_pyeidors_rm",
    "inverse_surrogate",
    "reconstruct_linearized_sigma",
    "sigma_true_from_anomaly_rule",
    "summarize_eit_digit_run",
    "summarize_eit_digit_sweep",
    "VoltageDigitFieldRow",
    "VoltageDigitSweepSummary",
    "keep_significant_digits",
    "run_voltage_digit_sweep",
    "run_voltage_digit_sweep_from_backend",
    "DigitReportCase",
    "DigitReportRow",
    "format_markdown_report",
    "format_markdown_table",
    "read_eit_digit_case",
    "read_eit_digit_cases",
    "write_report_files",
    "FactorSweepRow",
    "format_factor_sweep_report",
    "normalize_enob_level",
    "run_factor_sweep",
    "create_synthetic_data",
    "create_custom_phantom",
    "MeasurementDataset",
    "FrameMetadata",
]

_SUBMODULE_NAMES = frozenset(
    {
        "_sweep_core",
        "_temporal_core",
        "adc_quantization",
        "bucket_dense_experiments",
        "bucket_domain_audit",
        "channels",
        "difference",
        "digit_plot",
        "digit_report",
        "dynamic_sequence",
        "factor_sweep",
        "frame_io",
        "holdout_fit_diff",
        "holdout_point_audit",
        "measurement_dataset",
        "noise",
        "structures",
        "synthetic_data",
        "temporal_filtering",
        "visual_audit",
        "voltage_digit_sweep",
    }
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
