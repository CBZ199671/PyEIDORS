"""Hardware constants, command enums, and ADC conversion parameters."""

from dataclasses import dataclass
from enum import IntEnum


class AcquisitionMode(IntEnum):
    """Device-side acquisition modes."""

    LEGACY_ONE_SHOT = 0
    STREAMING = 1


class Command(IntEnum):
    POWER_CONTROL = 0x01
    FREQ_SET = 0x02
    STI_AMP_SET = 0x03
    VOLTAGE_AMP_SET = 0x04
    SWEEP_SET = 0x07
    SINGLE_POINT_TEST = 0x08
    START_MEA = 0x09
    CONTACT_IMPEDANCE_MEA = 0x0A
    CAPABILITY_QUERY = 0x30
    STREAM_START = 0x31
    STREAM_STOP = 0x32


class RelayCommand(IntEnum):
    """Legacy 4G server envelope commands."""

    REGISTER = 0x01
    TRANSMIT = 0x02
    TO_SERVER = 0x03


class RelayStatus(IntEnum):
    """Legacy 4G server status codes."""

    SUCCESS = 0x00
    CMD_ERROR = 0x01
    NO_BOARD = 0x02
    ID_ERROR = 0x03
    DEFEATED = 0xFF


# Frame delimiters
FRAME_HEAD = bytes([0x88, 0xFB, 0xFA])
FRAME_END = bytes([0xFD, 0xFC])


@dataclass(frozen=True)
class ADCParams:
    """ADC-to-voltage conversion parameters for the C8051F060 board."""

    vref: float = 2.412
    bits: int = 16
    max_value: int = 65535
    offset_v: float = 0.988
    real_offset: float = 0.054
    imag_offset: float = 0.04
    amplitude_scale: float = 1.10
    component_scale: float = 2.0


DEFAULT_ADC_PARAMS = ADCParams()


@dataclass(frozen=True)
class FrameSpec:
    """Per-frame measurement layout for one acquisition pattern."""

    n_electrodes: int = 16
    points_per_frame: int = 208
    bytes_per_point: int = 4  # ADC0_H, ADC0_L, ADC1_H, ADC1_L


DEFAULT_FRAME_SPEC = FrameSpec()

# Legacy stimulation current selections from the old host software.
STIM_AMP_VALUES_UA: dict[int, int] = {
    0: 50,
    1: 100,
    2: 200,
    3: 500,
    4: 1000,
    5: 2000,
    6: 5000,
    7: 10000,
}

STIM_AMP_LEVELS: dict[int, str] = {
    level: f"{value}uA" if value < 1000 else f"{value // 1000}mA"
    for level, value in STIM_AMP_VALUES_UA.items()
}

# Legacy voltage amplifier gain factors used in the C# host application.
VOLTAGE_AMP_FACTORS: tuple[float, ...] = (
    0.097,
    0.175,
    0.327,
    0.623,
    1.238,
    2.46,
    4.88,
    9.0,
)

DEFAULT_SERVER_PORT = 4555
DEFAULT_BOARD_ID = 1
DEFAULT_USER_ID = 1
DEFAULT_MEA_MODE = 2
