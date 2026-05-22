from __future__ import annotations

from pathlib import Path

import pytest

from eit_app.controllers.database_controller import (
    DatabaseController,
    _BackfillWorker,
)
from eit_app.ui.database.database_tab import DatabaseTab
from pyeidors.data.frame_io import frame_index_from_csv_name, scan_frame_dir


def test_backfill_worker_request_cancel_stops_after_current_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _BackfillWorker(db=object(), root_dir=Path("/tmp"))
    discovered = [
        (Path("/tmp/session_a"), None),
        (Path("/tmp/session_b"), None),
        (Path("/tmp/session_c"), None),
    ]
    imported_sessions: list[Path] = []
    progress_updates: list[tuple[int, int]] = []
    finished_counts: list[int] = []

    monkeypatch.setattr(worker, "_discover_sessions", lambda root: discovered)

    def _fake_ingest(session_dir: Path, metadata_path: Path | None) -> None:
        imported_sessions.append(session_dir)
        if len(imported_sessions) == 1:
            worker.request_cancel()

    monkeypatch.setattr(worker, "_ingest_session", _fake_ingest)
    worker.progress.connect(
        lambda current, total: progress_updates.append((current, total))
    )
    worker.finished.connect(finished_counts.append)

    worker.run()

    assert imported_sessions == [Path("/tmp/session_a")]
    assert progress_updates == [(1, 3)]
    assert finished_counts == [1]


def test_database_controller_shutdown_cancels_backfill_before_closing_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = DatabaseController(tmp_path / "frames.sqlite")
    controller.db.close()
    calls: list[str] = []

    class _FakeWorker:
        def request_cancel(self) -> None:
            calls.append("cancel")

        def deleteLater(self) -> None:
            calls.append("worker_delete")

    class _FakeThread:
        def __init__(self) -> None:
            self._running = True

        def isRunning(self) -> bool:
            return self._running

        def requestInterruption(self) -> None:
            calls.append("interrupt")

        def quit(self) -> None:
            calls.append("quit")
            self._running = False

        def wait(self, timeout_ms: int | None = None) -> bool:
            calls.append(f"wait:{timeout_ms}")
            self._running = False
            return True

        def deleteLater(self) -> None:
            calls.append("thread_delete")

    class _FakeDb:
        def close(self) -> None:
            calls.append("close")

    fake_worker = _FakeWorker()
    fake_thread = _FakeThread()

    monkeypatch.setattr(controller, "_backfill_worker", fake_worker)
    monkeypatch.setattr(controller, "_backfill_thread", fake_thread)
    monkeypatch.setattr(controller, "_db", _FakeDb())

    controller.shutdown()

    assert controller.is_shutting_down is True
    assert "cancel" in calls
    assert "interrupt" in calls
    assert any(entry.startswith("wait:") for entry in calls)
    assert "close" in calls
    assert calls.index("cancel") < calls.index("interrupt") < calls.index("close")


def test_v126_database_controller_hides_unavailable_sessions_and_frames(
    tmp_path: Path,
) -> None:
    controller = DatabaseController(tmp_path / "frames.sqlite")
    existing_dir = tmp_path / "session_existing"
    existing_dir.mkdir()
    missing_dir = tmp_path / "session_missing"

    existing_id = controller.db.add_session(
        existing_dir,
        {"session_start": "2026-05-22T12:00:00"},
        name="session_existing",
    )
    missing_id = controller.db.add_session(
        missing_dir,
        {"session_start": "2026-05-22T12:01:00"},
        name="session_missing",
    )

    available_csv = existing_dir / "20260522_120000_frame_0000_1000Hz.csv"
    available_csv.write_text("1.0,0.0\n", encoding="utf-8")
    missing_csv = existing_dir / "20260522_120000_frame_0001_1000Hz.csv"
    controller.db.add_frame(existing_id, 0, 1.0, available_csv)
    controller.db.add_frame(existing_id, 1, 2.0, missing_csv)
    controller.db.add_frame(
        missing_id,
        0,
        3.0,
        missing_dir / "20260522_120100_frame_0000_1000Hz.csv",
    )

    assert [row["id"] for row in controller.query_sessions()] == [existing_id]
    frames = controller.query_frames(existing_id)
    assert [row["frame_index"] for row in frames] == [0]
    assert frames[0]["csv_path"] == str(available_csv.resolve())

    controller.shutdown()


def test_v126_frequency_suffixed_frame_csvs_are_indexed(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_20260522_120000"
    session_dir.mkdir()
    metadata_path = session_dir / "session_metadata.yaml"
    metadata_path.write_text(
        "session_start: '2026-05-22T12:00:00'\nfrequency_hz: 1000\n",
        encoding="utf-8",
    )
    csv_path = session_dir / "20260522_120000_frame_0007_1000Hz.csv"
    yaml_path = csv_path.with_suffix(".yaml")
    csv_path.write_text("1.0,0.0\n", encoding="utf-8")
    yaml_path.write_text("timestamp: 1.0\nfrequency_hz: 1000\n", encoding="utf-8")

    assert frame_index_from_csv_name(csv_path.name) == 7
    assert scan_frame_dir(session_dir) == [(csv_path, yaml_path)]

    controller = DatabaseController(tmp_path / "frames.sqlite")
    worker = _BackfillWorker(controller.db, tmp_path)
    session_id = controller.db.add_session(
        session_dir,
        {"session_start": "2026-05-22T12:00:00", "frequency_hz": 1000},
        name=session_dir.name,
    )

    worker._ingest_frames(session_id, session_dir)

    frames = controller.query_frames(session_id)
    assert len(frames) == 1
    assert frames[0]["frame_index"] == 7
    assert frames[0]["csv_path"] == str(csv_path.resolve())

    controller.shutdown()


def test_database_tab_skips_refresh_after_prepare_for_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeDbController:
        def __init__(self) -> None:
            self.is_shutting_down = False

    class _FakeStatus:
        def __init__(self) -> None:
            self.texts: list[str] = []

        def setText(self, text: str) -> None:
            self.texts.append(text)

    class _Harness:
        def __init__(self) -> None:
            self._db_ctrl = _FakeDbController()
            self._is_shutting_down = False
            self._backfill_status = _FakeStatus()

        def _should_skip_database_refresh(self) -> bool:
            return DatabaseTab._should_skip_database_refresh(self)

        def refresh_sessions(self) -> None:
            raise AssertionError("refresh_sessions should be skipped during shutdown")

    tab = _Harness()
    monkeypatch.setattr(
        tab,
        "refresh_sessions",
        lambda: (_ for _ in ()).throw(
            AssertionError("refresh_sessions should be skipped during shutdown")
        ),
    )

    DatabaseTab.prepare_for_shutdown(tab)
    assert DatabaseTab._should_skip_database_refresh(tab) is True
    DatabaseTab._on_backfill_done(tab, 3)

    assert tab._backfill_status.texts == []
