"""Lock-free shared-memory ring buffer for EIT measurement frames.

Layout in shared memory::

    [write_count: 8 bytes (int64)]
    [slot_0: real(n_meas*8) + imag(n_meas*8) + timestamp(8) + frame_index(8)]
    [slot_1: ...]
    ...

Total per slot: ``n_meas * 8 * 2 + 8 + 8`` bytes.
Single-writer, multiple-reader design (lock-free via monotonic write_count).
"""

from __future__ import annotations

import multiprocessing.shared_memory as shm
import struct

import numpy as np

HEADER_SIZE = 8  # write_count (int64)


class FrameRingBuffer:
    """Shared-memory ring buffer for passing frames between processes.

    Args:
        capacity: Max number of frames in the ring.
        n_meas: Measurements per frame.
        name: SharedMemory name. Auto-generated on create if ``None``.
        create: ``True`` to allocate new memory, ``False`` to attach.
    """

    def __init__(
        self,
        capacity: int = 256,
        n_meas: int = 208,
        name: str | None = None,
        create: bool = True,
    ) -> None:
        self._capacity = capacity
        self._n_meas = n_meas
        self._slot_size = n_meas * 8 * 2 + 8 + 8  # real + imag + timestamp + index
        total = HEADER_SIZE + capacity * self._slot_size

        if create:
            self._shm = shm.SharedMemory(name=name, create=True, size=total)
            struct.pack_into("<q", self._shm.buf, 0, 0)
        else:
            self._shm = shm.SharedMemory(name=name, create=False)

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def write_count(self) -> int:
        """Total frames written (monotonic counter, not ring position)."""
        return struct.unpack_from("<q", self._shm.buf, 0)[0]

    def write(
        self,
        real: np.ndarray,
        imag: np.ndarray,
        timestamp: float,
        frame_index: int,
    ) -> None:
        """Write one frame to the next slot. Single-writer only."""
        wc = self.write_count
        slot = wc % self._capacity
        offset = HEADER_SIZE + slot * self._slot_size

        buf = self._shm.buf
        real_bytes = np.asarray(real, dtype=np.float64).tobytes()
        imag_bytes = np.asarray(imag, dtype=np.float64).tobytes()
        n = self._n_meas * 8

        buf[offset : offset + n] = real_bytes
        offset += n
        buf[offset : offset + n] = imag_bytes
        offset += n
        struct.pack_into("<d", buf, offset, timestamp)
        offset += 8
        struct.pack_into("<q", buf, offset, frame_index)

        # Increment write_count AFTER data is written (store-release)
        struct.pack_into("<q", buf, 0, wc + 1)

    def read_latest(self) -> tuple[np.ndarray, np.ndarray, float, int] | None:
        """Read the most recently written frame.

        Returns ``(real, imag, timestamp, frame_index)`` or ``None`` if empty.
        """
        wc = self.write_count
        if wc == 0:
            return None
        return self._read_slot((wc - 1) % self._capacity)

    def read_at(
        self, write_count: int
    ) -> tuple[np.ndarray, np.ndarray, float, int] | None:
        """Read frame at a specific write_count value.

        Returns ``None`` if the slot has been overwritten.
        """
        current_wc = self.write_count
        if write_count <= 0 or write_count > current_wc:
            return None
        if current_wc - write_count >= self._capacity:
            return None  # overwritten
        return self._read_slot((write_count - 1) % self._capacity)

    def _read_slot(self, slot: int) -> tuple[np.ndarray, np.ndarray, float, int]:
        offset = HEADER_SIZE + slot * self._slot_size
        n = self._n_meas * 8
        buf = self._shm.buf

        real = np.frombuffer(bytes(buf[offset : offset + n]), dtype=np.float64).copy()
        offset += n
        imag = np.frombuffer(bytes(buf[offset : offset + n]), dtype=np.float64).copy()
        offset += n
        timestamp = struct.unpack_from("<d", buf, offset)[0]
        offset += 8
        frame_index = struct.unpack_from("<q", buf, offset)[0]

        return real, imag, timestamp, frame_index

    def close(self) -> None:
        """Close the shared memory handle (does not unlink)."""
        self._shm.close()

    def unlink(self) -> None:
        """Unlink the shared memory (call only from the creator process)."""
        self._shm.unlink()
