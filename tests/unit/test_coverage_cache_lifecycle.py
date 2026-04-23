"""Tests for cache lifecycle edge cases to achieve 100% coverage."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from unittest import mock

import pytest

from pyeidors.cache.lifecycle import (
    _parse_session_pid,
    _pid_is_alive,
    _shell_session_env,
    _register_shell_session_dir,
    cleanup_stale_session_caches,
    cleanup_registered_session_caches,
    resolve_cache_directory,
    _REGISTERED_SESSION_DIRS,
    _REGISTERED_SPECS,
    _LOCK,
    SHELL_SESSION_PREFIX,
    PROCESS_SESSION_PREFIX,
)


class TestParseSessionPid:
    """Cover lines 53-58: empty/invalid pid text."""

    def test_shell_prefix_empty_pid(self):
        result = _parse_session_pid(f"{SHELL_SESSION_PREFIX}")
        assert result is None

    def test_shell_prefix_only_dash(self):
        result = _parse_session_pid(f"{SHELL_SESSION_PREFIX}-abc")
        assert result is None

    def test_process_prefix_non_integer(self):
        result = _parse_session_pid(f"{PROCESS_SESSION_PREFIX}notanumber-rest")
        assert result is None

    def test_process_prefix_empty_after_strip(self):
        result = _parse_session_pid(f"{PROCESS_SESSION_PREFIX} -rest")
        assert result is None

    def test_no_matching_prefix(self):
        result = _parse_session_pid("some-other-session")
        assert result is None

    def test_valid_pid_parsed(self):
        result = _parse_session_pid(f"{SHELL_SESSION_PREFIX}1234-rest")
        assert result == 1234


class TestPidIsAlive:
    """Cover lines 64, 69-72: PermissionError and OSError branches."""

    def test_permission_error_returns_true(self):
        with mock.patch("os.kill", side_effect=PermissionError("no perms")):
            assert _pid_is_alive(1) is True

    def test_generic_oserror_returns_false(self):
        with mock.patch("os.kill", side_effect=OSError("unexpected")):
            assert _pid_is_alive(1) is False

    def test_negative_pid_returns_false(self):
        assert _pid_is_alive(-1) is False

    def test_zero_pid_returns_false(self):
        assert _pid_is_alive(0) is False


class TestShellSessionEnv:
    """Cover lines 89-90: invalid PYEIDORS_CACHE_OWNER_PID."""

    def test_no_session_id_returns_none(self, monkeypatch):
        monkeypatch.delenv("PYEIDORS_CACHE_SESSION_ID", raising=False)
        assert _shell_session_env() is None

    def test_invalid_owner_pid(self, monkeypatch):
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_ID", "test-session")
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_DIR", "")
        monkeypatch.setenv("PYEIDORS_CACHE_REQUESTED_ROOT", "")
        monkeypatch.setenv("PYEIDORS_CACHE_OWNER_PID", "not_a_number")
        result = _shell_session_env()
        assert result is not None
        session_id, session_dir, requested_root, owner_pid = result
        assert session_id == "test-session"
        assert owner_pid is None


class TestRegisterShellSessionDir:
    """Cover line 99-100: default_session_dir is None."""

    def test_none_default_session_dir_returns_early(self):
        _register_shell_session_dir(None, Path("/some/dir"))


class TestCleanupStaleSessions:
    """Cover lines 138, 144-145, 147."""

    def test_skips_non_dir_entries(self, tmp_path):
        session_root = tmp_path / ".sessions"
        session_root.mkdir()
        (session_root / "somefile.txt").write_text("x")
        removed = cleanup_stale_session_caches(tmp_path)
        assert removed == 0

    def test_file_not_found_during_stat(self, tmp_path):
        """Line 144-145: FileNotFoundError during stat of session dir."""
        session_root = tmp_path / ".sessions"
        session_root.mkdir()
        child = session_root / "unknown-session"
        child.mkdir()
        # Remove after listing to trigger FileNotFoundError on stat
        import shutil

        original_iterdir = Path.iterdir

        def mock_iterdir(self_path):
            result = list(original_iterdir(self_path))
            # Remove child so stat fails
            if child.exists():
                shutil.rmtree(child)
            return iter(result)

        with mock.patch.object(Path, "iterdir", mock_iterdir):
            removed = cleanup_stale_session_caches(tmp_path, max_age_seconds=0)
        assert removed == 0

    def test_unknown_session_below_max_age_kept(self, tmp_path):
        session_root = tmp_path / ".sessions"
        session_root.mkdir()
        child = session_root / "unknown-session"
        child.mkdir()
        removed = cleanup_stale_session_caches(tmp_path, max_age_seconds=999999)
        assert removed == 0

    def test_file_not_found_during_real_cleanup_stat(self, tmp_path, monkeypatch):
        session_root = tmp_path / ".sessions"
        session_root.mkdir()
        child = session_root / "stale-shell-123"
        child.mkdir()

        class _VanishingDir:
            name = child.name

            @staticmethod
            def is_dir() -> bool:
                return True

            @staticmethod
            def stat():
                raise FileNotFoundError("gone")

        original_iterdir = Path.iterdir

        def _iterdir_with_delete(self_path: Path):
            if self_path == session_root:
                return iter([_VanishingDir()])
            return original_iterdir(self_path)

        monkeypatch.setattr(Path, "iterdir", _iterdir_with_delete)
        removed = cleanup_stale_session_caches(tmp_path, max_age_seconds=0)
        assert removed == 0


class TestResolveCacheDirectory:
    """Cover lines 202-205: FileNotFoundError in resolve check."""

    def test_file_not_found_in_resolve(self, tmp_path, monkeypatch):
        """Lines 202-205: FileNotFoundError when comparing roots."""
        session_dir = tmp_path / "session_dir"
        session_dir.mkdir()
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_ID", "test-id")
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_DIR", str(session_dir))
        monkeypatch.setenv(
            "PYEIDORS_CACHE_REQUESTED_ROOT", str(tmp_path / "nonexistent_root")
        )
        monkeypatch.setenv("PYEIDORS_CACHE_OWNER_PID", "")

        cache_root = tmp_path / "cache_resolve_test"
        cache_root.mkdir()

        with _LOCK:
            key = str(cache_root.resolve())
            _REGISTERED_SPECS.pop(key, None)

        spec = resolve_cache_directory(
            cache_root,
            lifecycle="session",
            cleanup_on_exit=False,
            cleanup_stale_sessions_on_startup=False,
            stale_session_max_age_seconds=0,
        )
        assert spec.lifecycle == "session"
        assert spec.shell_managed is True

        with _LOCK:
            _REGISTERED_SPECS.pop(key, None)

    def test_shell_root_resolve_file_not_found_falls_back(self, tmp_path, monkeypatch):
        session_dir = tmp_path / "session_dir_2"
        session_dir.mkdir()
        requested_root = tmp_path / "requested_root"
        requested_root.mkdir()
        cache_root = tmp_path / "cache_resolve_test_2"
        cache_root.mkdir()

        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_ID", "test-id-2")
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_DIR", str(session_dir))
        monkeypatch.setenv("PYEIDORS_CACHE_REQUESTED_ROOT", str(requested_root))
        monkeypatch.setenv("PYEIDORS_CACHE_OWNER_PID", "")

        original_resolve = Path.resolve

        def _resolve_with_fnf(self_path: Path, *args, **kwargs):
            if self_path == requested_root:
                raise FileNotFoundError("gone")
            return original_resolve(self_path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", _resolve_with_fnf)

        with _LOCK:
            key = str(original_resolve(cache_root))
            _REGISTERED_SPECS.pop(key, None)

        spec = resolve_cache_directory(
            cache_root,
            lifecycle="session",
            cleanup_on_exit=False,
            cleanup_stale_sessions_on_startup=False,
            stale_session_max_age_seconds=0,
        )
        assert spec.shell_managed is True
        assert spec.effective_dir != session_dir

        with _LOCK:
            _REGISTERED_SPECS.pop(key, None)

    def test_relative_shell_session_path_forces_same_root_false(
        self, tmp_path, monkeypatch
    ):
        cache_root = tmp_path / "cache_relative_test"
        cache_root.mkdir()
        relative_session_dir = Path("relative-session")

        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_ID", "test-relative")
        monkeypatch.setenv("PYEIDORS_CACHE_SESSION_DIR", str(relative_session_dir))
        monkeypatch.setenv("PYEIDORS_CACHE_REQUESTED_ROOT", str(cache_root))
        monkeypatch.setenv("PYEIDORS_CACHE_OWNER_PID", "")

        with _LOCK:
            key = str(cache_root.resolve())
            _REGISTERED_SPECS.pop(key, None)

        spec = resolve_cache_directory(
            cache_root,
            lifecycle="session",
            cleanup_on_exit=False,
            cleanup_stale_sessions_on_startup=False,
            stale_session_max_age_seconds=0,
        )
        assert spec.shell_managed is True
        assert spec.effective_dir != relative_session_dir

        with _LOCK:
            _REGISTERED_SPECS.pop(key, None)


class TestCleanupRegisteredSessionCaches:
    """Cover lines 256-257: parent.rmdir raises OSError."""

    def test_cleanup_with_oserror_on_rmdir(self, tmp_path):
        session_dir = tmp_path / "sessions" / "test-session"
        session_dir.mkdir(parents=True)

        with _LOCK:
            _REGISTERED_SESSION_DIRS.add(session_dir)

        removed = cleanup_registered_session_caches()
        assert removed == 1
