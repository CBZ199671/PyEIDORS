"""ADC quantization helpers for boundary-voltage precision studies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from pyeidors.utils.numeric_ops import all_finite_values


DEFAULT_BOUNDARY_VOLTAGES = np.array(
    [
        473.345698734,
        42.3456987378,
        4.32918985497,
        4273.34569873,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ADCQuantizationSummary:
    """Summary metrics for one ADC bit depth."""

    bit: int
    ideal_decimal_digits: float
    full_scale: float
    lsb: float
    voltage_rmse: float
    voltage_effective_digits: float

    def as_csv_row(self) -> dict[str, float | int]:
        return {
            "bit": self.bit,
            "ideal_decimal_digits": self.ideal_decimal_digits,
            "full_scale": self.full_scale,
            "lsb": self.lsb,
            "voltage_rmse": self.voltage_rmse,
            "voltage_effective_digits": self.voltage_effective_digits,
        }


@dataclass(frozen=True)
class ADCInjectionConfig:
    """ADC front-end perturbation config for one measurement vector."""

    bit: int
    full_scale_range: float
    enob: float | None = None
    noise_std: float = 0.0
    noise_relative: float = 0.0
    seed: int | None = None


def _validate_bit(bit: int) -> int:
    bit_int = int(bit)
    if bit_int <= 0:
        raise ValueError("ADC bit depth must be positive")
    return bit_int


def _validate_non_negative(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return number


def _as_float_vector(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if not all_finite_values(arr):
        raise ValueError("voltages must be finite")
    return arr


def ideal_decimal_digits(bit: int) -> float:
    """Return ideal decimal resolution estimate, N * log10(2)."""

    return _validate_bit(bit) * math.log10(2.0)


def effective_adc_bits(bit: int, enob: float | None = None) -> float:
    """Return effective bit count after optional ENOB substitution."""

    nominal_bit = _validate_bit(bit)
    if enob is None:
        return float(nominal_bit)
    effective_bit = float(enob)
    if not math.isfinite(effective_bit) or effective_bit <= 0.0:
        raise ValueError("enob must be positive and finite")
    if effective_bit > nominal_bit:
        raise ValueError("enob must be less than or equal to ADC bit depth")
    return effective_bit


def adc_lsb(
    full_scale_range: float,
    bit: int,
    *,
    enob: float | None = None,
) -> float:
    """Return ADC least-significant-bit width for a full-scale range."""

    full_scale = float(full_scale_range)
    if not math.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError("full_scale_range must be positive and finite")
    return full_scale / float(2 ** effective_adc_bits(bit, enob))


def noise_standard_deviation(
    reference: Iterable[float] | np.ndarray,
    *,
    noise_std: float = 0.0,
    noise_relative: float = 0.0,
) -> float:
    """Return combined absolute and RMS-relative Gaussian noise sigma."""

    ref = _as_float_vector(reference)
    absolute_std = _validate_non_negative(noise_std, name="noise_std")
    relative = _validate_non_negative(noise_relative, name="noise_relative")
    reference_rms = float(np.sqrt(np.mean(ref**2)))
    relative_std = relative * reference_rms
    return float(math.hypot(absolute_std, relative_std))


def add_voltage_noise(
    voltages: Iterable[float] | np.ndarray,
    *,
    noise_std: float = 0.0,
    noise_relative: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian front-end noise to voltages."""

    arr = _as_float_vector(voltages)
    sigma = noise_standard_deviation(
        arr,
        noise_std=noise_std,
        noise_relative=noise_relative,
    )
    if sigma == 0.0:
        return arr.copy()
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)
    return arr + generator.normal(loc=0.0, scale=sigma, size=arr.shape)


def quantize_voltages(
    voltages: Iterable[float] | np.ndarray,
    *,
    bit: int,
    full_scale_range: float,
    enob: float | None = None,
) -> np.ndarray:
    """Quantize voltages with ideal round-to-nearest ADC bins."""

    arr = _as_float_vector(voltages)
    step = adc_lsb(full_scale_range, bit, enob=enob)
    return np.round(arr / step) * step


def inject_adc_measurement(
    voltages: Iterable[float] | np.ndarray,
    config: ADCInjectionConfig,
) -> np.ndarray:
    """Apply front-end noise and ADC quantization to boundary voltages."""

    noisy = add_voltage_noise(
        voltages,
        noise_std=config.noise_std,
        noise_relative=config.noise_relative,
        seed=config.seed,
    )
    return quantize_voltages(
        noisy,
        bit=config.bit,
        full_scale_range=config.full_scale_range,
        enob=config.enob,
    )


def pointwise_effective_digits(
    reference: Iterable[float] | np.ndarray,
    observed: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return per-sample effective decimal digits from relative absolute error."""

    ref = _as_float_vector(reference)
    obs = _as_float_vector(observed)
    if ref.shape != obs.shape:
        raise ValueError("reference and observed must have same shape")

    error = np.abs(obs - ref)
    ref_abs = np.abs(ref)
    digits = np.full(ref.shape, np.nan, dtype=float)

    exact = error == 0.0
    digits[exact] = math.inf

    nonzero = (~exact) & (ref_abs > 0.0)
    digits[nonzero] = -np.log10(error[nonzero] / ref_abs[nonzero])
    return digits


def rmse(
    reference: Iterable[float] | np.ndarray, observed: Iterable[float] | np.ndarray
) -> float:
    """Return root-mean-square error."""

    ref = _as_float_vector(reference)
    obs = _as_float_vector(observed)
    if ref.shape != obs.shape:
        raise ValueError("reference and observed must have same shape")
    return float(np.sqrt(np.mean((obs - ref) ** 2)))


def effective_digits_from_rmse(
    reference: Iterable[float] | np.ndarray,
    observed: Iterable[float] | np.ndarray,
) -> float:
    """Return effective digits from RMSE divided by RMS reference magnitude."""

    ref = _as_float_vector(reference)
    obs = _as_float_vector(observed)
    if ref.shape != obs.shape:
        raise ValueError("reference and observed must have same shape")

    error_rms = rmse(ref, obs)
    if error_rms == 0.0:
        return math.inf

    reference_rms = float(np.sqrt(np.mean(ref**2)))
    if reference_rms == 0.0:
        return math.nan
    return -math.log10(error_rms / reference_rms)


def summarize_adc_quantization(
    voltages: Iterable[float] | np.ndarray,
    *,
    bit: int,
    full_scale_range: float,
    enob: float | None = None,
    noise_std: float = 0.0,
    noise_relative: float = 0.0,
    seed: int | None = None,
) -> ADCQuantizationSummary:
    """Quantize one voltage vector and return summary metrics."""

    ref = _as_float_vector(voltages)
    config = ADCInjectionConfig(
        bit=bit,
        full_scale_range=full_scale_range,
        enob=enob,
        noise_std=noise_std,
        noise_relative=noise_relative,
        seed=seed,
    )
    quantized = inject_adc_measurement(ref, config)
    return ADCQuantizationSummary(
        bit=_validate_bit(bit),
        ideal_decimal_digits=ideal_decimal_digits(bit),
        full_scale=float(full_scale_range),
        lsb=adc_lsb(full_scale_range, bit, enob=enob),
        voltage_rmse=rmse(ref, quantized),
        voltage_effective_digits=effective_digits_from_rmse(ref, quantized),
    )


def summarize_adc_sweep(
    voltages: Iterable[float] | np.ndarray,
    *,
    bits: Iterable[int],
    full_scale_range: float,
    enob: float | None = None,
    noise_std: float = 0.0,
    noise_relative: float = 0.0,
    seed: int | None = None,
) -> list[ADCQuantizationSummary]:
    """Return quantization summaries for multiple bit depths."""

    ref = _as_float_vector(voltages)
    return [
        summarize_adc_quantization(
            ref,
            bit=bit,
            full_scale_range=full_scale_range,
            enob=enob,
            noise_std=noise_std,
            noise_relative=noise_relative,
            seed=seed,
        )
        for bit in bits
    ]
