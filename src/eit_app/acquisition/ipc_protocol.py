"""IPC message types for acquisition process communication."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AcquisitionCommand(Enum):
    START = "start"
    CAPTURE_ONE = "capture_one"
    STOP = "stop"
    CONFIGURE = "configure"
    SHUTDOWN = "shutdown"


class AcquisitionStatus(Enum):
    IDLE = 0
    CONNECTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4
    SHUTDOWN = 5


@dataclass
class IPCMessage:
    command: AcquisitionCommand
    payload: dict[str, Any] = field(default_factory=dict)
