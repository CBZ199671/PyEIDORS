"""User-writable runtime paths for installed and source-tree runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

PYEIDORS_RUNTIME_ROOT_ENV = "PYEIDORS_RUNTIME_ROOT"
PYEIDORS_CACHE_ROOT_ENV = "PYEIDORS_CACHE_ROOT"
PYEIDORS_CACHE_REQUESTED_ROOT_ENV = "PYEIDORS_CACHE_REQUESTED_ROOT"
PYEIDORS_DATA_ROOT_ENV = "PYEIDORS_DATA_ROOT"
PYEIDORS_OUTPUT_ROOT_ENV = "PYEIDORS_OUTPUT_ROOT"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def xdg_cache_home() -> Path:
    """Return the XDG cache directory, with a home-directory fallback."""

    raw = os.environ.get("XDG_CACHE_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache"


def xdg_data_home() -> Path:
    """Return the user data directory, with platform-aware fallbacks."""

    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        if local_appdata:
            return Path(local_appdata).expanduser()
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return Path(appdata).expanduser()
    return Path.home() / ".local" / "share"


def pyeidors_runtime_root() -> Path:
    """Return the user-writable root for PyEIDORS runtime state."""

    override = _env_path(PYEIDORS_RUNTIME_ROOT_ENV)
    if override is not None:
        return override
    return xdg_cache_home() / "pyeidors"


def _cache_root_from_session_env() -> Path | None:
    requested = _env_path(PYEIDORS_CACHE_REQUESTED_ROOT_ENV)
    if requested is None:
        return None
    if requested.name == "v2":
        return requested.parent
    return requested


def pyeidors_cache_root() -> Path:
    """Return the root for PyEIDORS cache/artifact files.

    Packaged applications should set ``PYEIDORS_CACHE_ROOT`` explicitly.  Source
    dev shells can keep their existing session-scoped ``.pyeidors_cache/v2``
    behavior through ``PYEIDORS_CACHE_REQUESTED_ROOT``.
    """

    override = _env_path(PYEIDORS_CACHE_ROOT_ENV)
    if override is not None:
        return override
    session_root = _cache_root_from_session_env()
    if session_root is not None:
        return session_root
    return pyeidors_runtime_root() / ".pyeidors_cache"


def pyeidors_data_root() -> Path:
    """Return the user-writable root for durable PyEIDORS data files."""

    override = _env_path(PYEIDORS_DATA_ROOT_ENV)
    if override is not None:
        return override
    return xdg_data_home() / "pyeidors"


def pyeidors_output_root() -> Path:
    """Return the user-writable root for generated artifacts."""

    override = _env_path(PYEIDORS_OUTPUT_ROOT_ENV)
    if override is not None:
        return override
    return pyeidors_data_root() / "outputs"


def pyeidors_cache_path(*parts: str | Path) -> Path:
    """Return a path below the PyEIDORS cache root."""

    root = pyeidors_cache_root()
    if not parts:
        return root
    return root.joinpath(*(Path(part) for part in parts))


def pyeidors_data_path(*parts: str | Path) -> Path:
    """Return a path below the durable PyEIDORS data root."""

    root = pyeidors_data_root()
    if not parts:
        return root
    return root.joinpath(*(Path(part) for part in parts))


def pyeidors_output_path(*parts: str | Path) -> Path:
    """Return a path below the generated-output root."""

    root = pyeidors_output_root()
    if not parts:
        return root
    return root.joinpath(*(Path(part) for part in parts))


def resolve_pyeidors_mesh_dir(path: str | Path | None = None) -> Path:
    """Resolve the default mesh-cache directory to the user-writable cache root."""

    if path is None:
        return pyeidors_cache_path("eit_meshes")
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    parts = _path_parts(candidate)
    if not parts:
        return pyeidors_cache_path("eit_meshes")
    if parts[0] == "eit_meshes":
        return pyeidors_cache_path(*parts)
    if parts[0] == ".pyeidors_cache":
        return pyeidors_cache_path(*parts[1:])
    return candidate


def _path_parts(path: Path) -> tuple[str, ...]:
    return tuple(str(part) for part in path.parts)


def resolve_pyeidors_cache_dir(
    path: str | Path | None = None,
    *,
    default_parts: Iterable[str | Path] = (),
) -> Path:
    """Resolve cache-like paths, mapping ``.pyeidors_cache`` to the user root."""

    if path is None:
        return pyeidors_cache_path(*default_parts)
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    parts = _path_parts(candidate)
    if parts and parts[0] == ".pyeidors_cache":
        return pyeidors_cache_path(*parts[1:])
    return candidate


__all__ = [
    "PYEIDORS_CACHE_REQUESTED_ROOT_ENV",
    "PYEIDORS_CACHE_ROOT_ENV",
    "PYEIDORS_DATA_ROOT_ENV",
    "PYEIDORS_OUTPUT_ROOT_ENV",
    "PYEIDORS_RUNTIME_ROOT_ENV",
    "pyeidors_cache_path",
    "pyeidors_cache_root",
    "pyeidors_data_path",
    "pyeidors_data_root",
    "pyeidors_output_path",
    "pyeidors_output_root",
    "pyeidors_runtime_root",
    "resolve_pyeidors_cache_dir",
    "resolve_pyeidors_mesh_dir",
    "xdg_cache_home",
    "xdg_data_home",
]
