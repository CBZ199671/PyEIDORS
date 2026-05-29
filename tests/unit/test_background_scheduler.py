"""Tests for GUI background task scheduling."""

from __future__ import annotations

import threading
import time

from eit_app.background_scheduler import BackgroundTaskPriority, BackgroundTaskScheduler


def test_background_scheduler_runs_pending_tasks_by_priority() -> None:
    scheduler = BackgroundTaskScheduler(name="unit-priority")
    gate = threading.Event()
    first_started = threading.Event()
    order: list[str] = []

    def _blocking() -> None:
        first_started.set()
        assert gate.wait(timeout=2.0)
        order.append("blocking")

    scheduler.submit(
        key="blocking",
        name="blocking",
        priority=BackgroundTaskPriority.PREWARM,
        fn=_blocking,
    )
    assert first_started.wait(timeout=2.0)
    scheduler.submit(
        key="low",
        name="low",
        priority=BackgroundTaskPriority.GC,
        fn=lambda: order.append("low"),
    )
    scheduler.submit(
        key="high",
        name="high",
        priority=BackgroundTaskPriority.RECONSTRUCTION,
        fn=lambda: order.append("high"),
    )

    gate.set()

    assert scheduler.wait_for_idle(timeout=2.0)
    assert order == ["blocking", "high", "low"]
    scheduler.shutdown()


def test_background_scheduler_coalesces_pending_duplicate_keys() -> None:
    scheduler = BackgroundTaskScheduler(name="unit-coalesce")
    gate = threading.Event()
    first_started = threading.Event()
    ran: list[str] = []

    scheduler.submit(
        key="blocking",
        name="blocking",
        priority=BackgroundTaskPriority.FOREGROUND,
        fn=lambda: (first_started.set(), gate.wait(timeout=2.0), ran.append("block")),
    )
    assert first_started.wait(timeout=2.0)
    older = scheduler.submit(
        key="same",
        name="same-old",
        priority=BackgroundTaskPriority.PREWARM,
        fn=lambda: ran.append("old"),
    )
    newer = scheduler.submit(
        key="same",
        name="same-new",
        priority=BackgroundTaskPriority.PREWARM,
        fn=lambda: ran.append("new"),
    )

    gate.set()

    assert scheduler.wait_for_idle(timeout=2.0)
    assert older.status == "cancelled"
    assert older.reason == "replaced_by_newer_pending_task"
    assert newer.accepted is True
    assert newer.replaced is True
    assert ran == ["block", "new"]
    scheduler.shutdown()


def test_background_scheduler_rejects_duplicate_active_key() -> None:
    scheduler = BackgroundTaskScheduler(name="unit-active")
    gate = threading.Event()
    started = threading.Event()

    scheduler.submit(
        key="active",
        name="active",
        priority=BackgroundTaskPriority.PREWARM,
        fn=lambda: (started.set(), gate.wait(timeout=2.0)),
    )
    assert started.wait(timeout=2.0)
    duplicate = scheduler.submit(
        key="active",
        name="duplicate",
        priority=BackgroundTaskPriority.PREWARM,
        fn=lambda: None,
    )

    gate.set()

    assert duplicate.accepted is False
    assert duplicate.reason == "task_active"
    assert scheduler.wait_for_idle(timeout=2.0)
    scheduler.shutdown()


def test_background_scheduler_cancels_pending_by_priority() -> None:
    scheduler = BackgroundTaskScheduler(name="unit-cancel")
    gate = threading.Event()
    started = threading.Event()
    ran: list[str] = []

    scheduler.submit(
        key="foreground-active",
        name="foreground-active",
        priority=BackgroundTaskPriority.FOREGROUND,
        fn=lambda: (started.set(), gate.wait(timeout=2.0), ran.append("active")),
    )
    assert started.wait(timeout=2.0)
    scheduler.submit(
        key="prewarm:a",
        name="prewarm-a",
        priority=BackgroundTaskPriority.PREWARM,
        fn=lambda: ran.append("prewarm"),
    )
    scheduler.submit(
        key="gc:a",
        name="gc-a",
        priority=BackgroundTaskPriority.GC,
        fn=lambda: ran.append("gc"),
    )

    cancelled = scheduler.cancel_pending(
        min_priority=BackgroundTaskPriority.PREWARM,
        reason="foreground_requested",
    )
    gate.set()

    assert cancelled == 2
    assert scheduler.wait_for_idle(timeout=2.0)
    assert ran == ["active"]
    scheduler.shutdown()


def test_background_scheduler_shutdown_cancels_pending_tasks() -> None:
    scheduler = BackgroundTaskScheduler(name="unit-shutdown")
    gate = threading.Event()
    started = threading.Event()

    scheduler.submit(
        key="blocking",
        name="blocking",
        priority=BackgroundTaskPriority.FOREGROUND,
        fn=lambda: (started.set(), gate.wait(timeout=2.0)),
    )
    assert started.wait(timeout=2.0)
    pending = scheduler.submit(
        key="pending",
        name="pending",
        priority=BackgroundTaskPriority.GC,
        fn=lambda: time.sleep(0.1),
    )

    scheduler.shutdown(wait=False)
    gate.set()

    assert pending.status == "cancelled"
    assert pending.reason == "scheduler_shutdown"
