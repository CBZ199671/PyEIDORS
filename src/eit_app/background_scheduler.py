"""Small priority scheduler for GUI background maintenance jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import heapq
import itertools
import threading
import time
from typing import Any


class BackgroundTaskPriority:
    """Conventional priority bands. Lower numbers run first."""

    FOREGROUND = 0
    RECONSTRUCTION = 20
    PREWARM = 60
    GC = 90


@dataclass
class BackgroundTaskHandle:
    key: str
    name: str
    priority: int
    accepted: bool
    replaced: bool = False
    reason: str = ""
    submitted_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    status: str = "pending"
    error: str = ""

    @property
    def done(self) -> bool:
        return self.status in {"done", "failed", "cancelled", "rejected"}


@dataclass(order=True)
class _QueuedTask:
    priority: int
    sequence: int
    handle: BackgroundTaskHandle = field(compare=False)
    fn: Callable[[], Any] = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


class BackgroundTaskScheduler:
    """Single-process priority scheduler with pending-task coalescing."""

    def __init__(self, *, name: str = "background", max_workers: int = 1) -> None:
        self.name = str(name)
        self.max_workers = max(1, int(max_workers))
        self._condition = threading.Condition()
        self._queue: list[_QueuedTask] = []
        self._pending_by_key: dict[str, _QueuedTask] = {}
        self._active_keys: set[str] = set()
        self._sequence = itertools.count()
        self._shutdown = False
        self._workers: list[threading.Thread] = []
        for index in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"{self.name}-worker-{index}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def submit(
        self,
        *,
        key: str,
        name: str,
        priority: int,
        fn: Callable[[], Any],
        coalesce: bool = True,
    ) -> BackgroundTaskHandle:
        task_key = str(key)
        handle = BackgroundTaskHandle(
            key=task_key,
            name=str(name or task_key),
            priority=int(priority),
            accepted=True,
        )
        with self._condition:
            if self._shutdown:
                handle.accepted = False
                handle.status = "rejected"
                handle.reason = "scheduler_shutdown"
                return handle
            if coalesce and task_key in self._active_keys:
                handle.accepted = False
                handle.status = "rejected"
                handle.reason = "task_active"
                return handle
            if coalesce:
                previous = self._pending_by_key.pop(task_key, None)
                if previous is not None:
                    previous.cancelled = True
                    previous.handle.status = "cancelled"
                    previous.handle.reason = "replaced_by_newer_pending_task"
                    handle.replaced = True
            queued = _QueuedTask(
                priority=int(priority),
                sequence=next(self._sequence),
                handle=handle,
                fn=fn,
            )
            self._pending_by_key[task_key] = queued
            heapq.heappush(self._queue, queued)
            self._condition.notify()
            return handle

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "name": self.name,
                "max_workers": self.max_workers,
                "pending": len(self._pending_by_key),
                "queued": len([task for task in self._queue if not task.cancelled]),
                "active": len(self._active_keys),
                "active_keys": sorted(self._active_keys),
                "shutdown": self._shutdown,
            }

    def cancel_pending(
        self,
        *,
        min_priority: int | None = None,
        key_prefix: str | None = None,
        reason: str = "cancelled_by_policy",
    ) -> int:
        """Cancel pending jobs matching a priority/key filter."""

        cancelled = 0
        with self._condition:
            for task in list(self._pending_by_key.values()):
                if task.cancelled:
                    continue
                if min_priority is not None and int(task.priority) < int(min_priority):
                    continue
                if key_prefix is not None and not task.handle.key.startswith(
                    str(key_prefix)
                ):
                    continue
                task.cancelled = True
                task.handle.status = "cancelled"
                task.handle.reason = str(reason)
                self._pending_by_key.pop(task.handle.key, None)
                cancelled += 1
            if cancelled:
                self._condition.notify_all()
        return cancelled

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._condition:
            while self._pending_by_key or self._active_keys:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                else:
                    remaining = None
                self._condition.wait(timeout=remaining)
            return True

    def shutdown(self, *, wait: bool = True, timeout: float | None = 2.0) -> None:
        with self._condition:
            self._shutdown = True
            for task in self._queue:
                if not task.cancelled:
                    task.cancelled = True
                    task.handle.status = "cancelled"
                    task.handle.reason = "scheduler_shutdown"
            self._pending_by_key.clear()
            self._condition.notify_all()
        if wait:
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            for worker in self._workers:
                join_timeout = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                worker.join(timeout=join_timeout)

    def _worker_loop(self) -> None:
        while True:
            task = self._next_task()
            if task is None:
                return
            handle = task.handle
            with self._condition:
                if task.cancelled:
                    self._condition.notify_all()
                    continue
                self._pending_by_key.pop(handle.key, None)
                self._active_keys.add(handle.key)
                handle.status = "running"
                handle.started_at = time.time()
            try:
                task.fn()
            except Exception as exc:  # pragma: no cover - callers usually catch/log
                handle.status = "failed"
                handle.error = str(exc)
            else:
                handle.status = "done"
            finally:
                with self._condition:
                    handle.finished_at = time.time()
                    self._active_keys.discard(handle.key)
                    self._condition.notify_all()

    def _next_task(self) -> _QueuedTask | None:
        with self._condition:
            while True:
                if self._shutdown:
                    return None
                while self._queue:
                    task = heapq.heappop(self._queue)
                    if task.cancelled:
                        self._condition.notify_all()
                        continue
                    return task
                self._condition.wait()
