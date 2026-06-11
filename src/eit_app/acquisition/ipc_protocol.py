"""IPC message types for acquisition process communication."""

from enum import Enum


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
