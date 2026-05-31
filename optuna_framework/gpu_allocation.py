"""Thread-safe GPU lease helpers for parallel Optuna trials."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue


@dataclass(frozen=True)
class GpuLease:
    """One acquired GPU device that must be released after use."""

    device: str
    _queue: Queue[str]
    _released: bool = False

    def release(self) -> None:
        """Return the leased device to the allocator."""

        if self._released:
            return
        object.__setattr__(self, "_released", True)
        self._queue.put(self.device)


class GpuAllocator:
    """A blocking FIFO allocator for a fixed set of CUDA devices."""

    def __init__(self, devices: tuple[str, ...]):
        if not devices:
            raise ValueError("devices must not be empty")
        self.devices = tuple(devices)
        self.n_jobs = len(self.devices)
        self._queue: Queue[str] = Queue()
        for device in self.devices:
            self._queue.put(device)

    def acquire(self) -> GpuLease:
        """Block until a GPU is available and return its lease."""

        return GpuLease(device=self._queue.get(), _queue=self._queue)
