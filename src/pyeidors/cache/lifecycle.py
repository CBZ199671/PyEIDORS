"""Session-scoped disk cache lifecycle helpers."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

SESSION_ROOT_NAME = ".sessions"
SHELL_SESSION_PREFIX = "session-shellpid"
PROCESS_SESSION_PREFIX = "session-pid"
SHELL_SESSION_REGISTRY = ".session-dirs"
DEFAULT_STALE_SESSION_MAX_AGE_SECONDS = float(7 * 24 * 60 * 60)


@dataclass(frozen=True)
class CacheDirectorySpec:
    """Resolved cache directory metadata."""

    requested_dir: Path
    effective_dir: Path
    lifecycle: str
    session_root: Path | None = None
    session_id: str | None = None
    shell_managed: bool = False


_PROCESS_SESSION_ID = f"{PROCESS_SESSION_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:10]}"
_LOCK = threading.Lock()
_REGISTERED_SPECS: dict[str, CacheDirectorySpec] = {}
_REGISTERED_SESSION_DIRS: set[Path] = set()
_CLEANUP_REGISTERED = False


def _ensure_atexit_cleanup() -> None:
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    atexit.register(cleanup_registered_session_caches)
    _CLEANUP_REGISTERED = True


def _parse_session_pid(name: str) -> int | None:
    for prefix in (SHELL_SESSION_PREFIX, PROCESS_SESSION_PREFIX):
        if name.startswith(prefix):
            pid_text = name[len(prefix) :].split("-", 1)[0].strip()
            if not pid_text:
                return None
            try:
                return int(pid_text)
            except ValueError:
                return None
    return None


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _shell_session_env() -> tuple[str, Path | None, Path | None, int | None] | None:
    session_id = os.environ.get("PYEIDORS_CACHE_SESSION_ID", "").strip()
    if not session_id:
        return None
    session_dir_raw = os.environ.get("PYEIDORS_CACHE_SESSION_DIR", "").strip()
    requested_root_raw = os.environ.get("PYEIDORS_CACHE_REQUESTED_ROOT", "").strip()
    owner_pid_raw = os.environ.get("PYEIDORS_CACHE_OWNER_PID", "").strip()
    session_dir = Path(session_dir_raw) if session_dir_raw else None
    requested_root = Path(requested_root_raw) if requested_root_raw else None
    owner_pid = None
    if owner_pid_raw:
        try:
            owner_pid = int(owner_pid_raw)
        except ValueError:
            owner_pid = None
    return session_id, session_dir, requested_root, owner_pid


def _shell_session_registry_path(default_session_dir: Path) -> Path:
    return default_session_dir / SHELL_SESSION_REGISTRY


def _register_shell_session_dir(
    default_session_dir: Path | None, effective_dir: Path
) -> None:
    if default_session_dir is None:
        return
    registry_path = _shell_session_registry_path(default_session_dir)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    effective_text = str(effective_dir)
    default_text = str(default_session_dir)
    with _LOCK:
        existing: set[str] = set()
        if registry_path.exists():
            existing = {
                line.strip()
                for line in registry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        if effective_text in existing and default_text in existing:
            return
        existing.add(default_text)
        existing.add(effective_text)
        registry_path.write_text(
            "\n".join(sorted(existing)) + "\n",
            encoding="utf-8",
        )


def cleanup_stale_session_caches(
    cache_root: str | Path,
    *,
    session_root_name: str = SESSION_ROOT_NAME,
    max_age_seconds: float = DEFAULT_STALE_SESSION_MAX_AGE_SECONDS,
) -> int:
    requested_dir = Path(cache_root)
    session_root = requested_dir / session_root_name
    if not session_root.exists():
        return 0

    removed = 0
    now = time.time()
    for child in session_root.iterdir():
        if not child.is_dir():
            continue
        pid = _parse_session_pid(child.name)
        if pid is not None and _pid_is_alive(pid):
            continue
        try:
            age_seconds = max(0.0, now - float(child.stat().st_mtime))
        except FileNotFoundError:
            continue
        if pid is None and age_seconds < float(max_age_seconds):
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
    try:
        session_root.rmdir()
    except OSError:
        pass
    return removed


def resolve_cache_directory(
    cache_root: str | Path,
    *,
    lifecycle: str,
    cleanup_on_exit: bool,
    cleanup_stale_sessions_on_startup: bool,
    stale_session_max_age_seconds: float,
    session_root_name: str = SESSION_ROOT_NAME,
) -> CacheDirectorySpec:
    requested_dir = Path(cache_root)
    if lifecycle != "session":
        return CacheDirectorySpec(
            requested_dir=requested_dir,
            effective_dir=requested_dir,
            lifecycle=lifecycle,
        )

    key = str(requested_dir.resolve())
    with _LOCK:
        existing = _REGISTERED_SPECS.get(key)
        if existing is not None:
            return existing

    requested_dir.mkdir(parents=True, exist_ok=True)
    if cleanup_stale_sessions_on_startup:
        cleanup_stale_session_caches(
            requested_dir,
            session_root_name=session_root_name,
            max_age_seconds=float(stale_session_max_age_seconds),
        )

    shell_session = _shell_session_env()
    shell_managed = shell_session is not None
    if shell_session is not None:
        session_id, default_session_dir, shell_requested_root, _owner_pid = (
            shell_session
        )
        session_root = requested_dir / session_root_name
        session_root.mkdir(parents=True, exist_ok=True)
        if (
            default_session_dir is not None
            and shell_requested_root is not None
            and default_session_dir.is_absolute()
            and shell_requested_root.is_absolute()
        ):
            try:
                same_root = shell_requested_root.resolve() == requested_dir.resolve()
            except FileNotFoundError:
                same_root = False
        else:
            same_root = False
        effective_dir = (
            default_session_dir
            if same_root and default_session_dir is not None
            else session_root / session_id
        )
        effective_dir.mkdir(parents=True, exist_ok=True)
        _register_shell_session_dir(default_session_dir, effective_dir)
        spec = CacheDirectorySpec(
            requested_dir=requested_dir,
            effective_dir=effective_dir,
            lifecycle="session",
            session_root=session_root,
            session_id=session_id,
            shell_managed=True,
        )
    else:
        session_root = requested_dir / session_root_name
        session_root.mkdir(parents=True, exist_ok=True)
        effective_dir = session_root / _PROCESS_SESSION_ID
        effective_dir.mkdir(parents=True, exist_ok=True)
        spec = CacheDirectorySpec(
            requested_dir=requested_dir,
            effective_dir=effective_dir,
            lifecycle="session",
            session_root=session_root,
            session_id=_PROCESS_SESSION_ID,
            shell_managed=False,
        )

    with _LOCK:
        _REGISTERED_SPECS[key] = spec
        if cleanup_on_exit and not shell_managed:
            _REGISTERED_SESSION_DIRS.add(spec.effective_dir)
            _ensure_atexit_cleanup()
    return spec


def cleanup_registered_session_caches() -> int:
    with _LOCK:
        session_dirs = sorted(
            (Path(path) for path in _REGISTERED_SESSION_DIRS),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        _REGISTERED_SESSION_DIRS.clear()
        _REGISTERED_SPECS.clear()
    removed = 0
    for session_dir in session_dirs:
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
            removed += 1
        parent = session_dir.parent
        try:
            parent.rmdir()
        except OSError:
            pass
    return removed
