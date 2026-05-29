"""Runtime helpers for profile-isolated GUI backend workers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Iterator


@dataclass(frozen=True)
class BackendWorkerCache:
    """Resolved persistent cache paths for one backend profile."""

    profile: str
    profile_root: Path
    xdg_cache_home: Path
    removed_stale_jit_locks: tuple[Path, ...]


def _safe_profile_key(profile: str) -> str:
    raw = str(profile or "default").strip().lower() or "default"
    key = re.sub(r"[^a-z0-9_.+-]+", "_", raw)
    return key.strip("._") or "default"


def backend_worker_cache_root(repo: Path) -> Path:
    override = os.getenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(repo) / ".pyeidors_cache" / "gui_backend_worker"


def backend_worker_profile_root(repo: Path, profile: str) -> Path:
    return backend_worker_cache_root(repo) / "v1" / _safe_profile_key(profile)


def _stale_jit_lock_seconds() -> float:
    raw = os.getenv("EIT_APP_BACKEND_WORKER_STALE_JIT_LOCK_SECONDS", "60").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 60.0
    return max(value, 0.0)


def _compiled_ffcx_module_exists(fenics_cache: Path, c_file: Path) -> bool:
    stem = c_file.with_suffix("").name
    return any(fenics_cache.glob(f"{stem}*.so"))


def _compiled_ffcx_module_stems(fenics_cache: Path) -> set[str]:
    stems: set[str] = set()
    if not fenics_cache.exists():
        return stems
    for module in fenics_cache.glob("libffcx_*.so"):
        stem = module.name.removesuffix(".so").split(".", 1)[0]
        if stem:
            stems.add(stem)
    return stems


def _compiled_ffcx_module_exists_in(stems: set[str], c_file: Path) -> bool:
    return c_file.with_suffix("").name in stems


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _is_stale(path: Path, *, now: float, stale_after: float) -> bool:
    try:
        age = now - path.stat().st_mtime
    except OSError:
        return False
    return age >= stale_after


def cleanup_stale_ffcx_jit_locks(xdg_cache_home: Path) -> tuple[Path, ...]:
    """Repair/remove old FFCx lock-like files with no compiled module.

    FFCx creates ``libffcx_*.c`` with exclusive-open semantics while compiling.
    If a previous worker dies between creating that file and building the
    extension, later workers can time out before doing useful work. The GUI
    worker cache is project/profile scoped, so pruning orphaned old lock files
    keeps reuse fast without depending on the user's global ``~/.cache/fenics``.
    """

    fenics_cache = Path(xdg_cache_home) / "fenics"
    if not fenics_cache.exists():
        return ()
    stale_after = _stale_jit_lock_seconds()
    now = time.time()
    removed: list[Path] = []
    compiled_stems = _compiled_ffcx_module_stems(fenics_cache)
    for source in fenics_cache.glob("libffcx_*.c"):
        ready = source.with_suffix(".c.cached")
        if _compiled_ffcx_module_exists_in(compiled_stems, source):
            if not ready.exists():
                try:
                    ready.touch(exist_ok=True)
                except OSError:
                    pass
                else:
                    removed.append(ready)
            continue
        for artifact in (source, source.with_suffix(".o"), ready):
            if not artifact.exists():
                continue
            if artifact == source and not _is_stale(
                source, now=now, stale_after=stale_after
            ):
                continue
            try:
                artifact.unlink()
            except FileNotFoundError:
                pass
            else:
                removed.append(artifact)
    for ready in fenics_cache.glob("libffcx_*.c.cached"):
        source = ready.with_suffix("")
        if source.exists():
            continue
        if _compiled_ffcx_module_exists_in(compiled_stems, source):
            try:
                source.touch(exist_ok=True)
            except OSError:
                pass
            continue
        if not _is_stale(ready, now=now, stale_after=stale_after):
            continue
        try:
            ready.unlink()
        except FileNotFoundError:
            pass
        else:
            removed.append(ready)
    return tuple(removed)


_FFCX_SOURCE_PATH_RE = re.compile(r"(/[^\s)]+/libffcx_[^\s)]+\.c)")


def looks_like_ffcx_jit_timeout(error: str) -> bool:
    """Return true for FFCx cache-lock timeout errors."""

    text = str(error or "")
    return "JIT compilation timed out" in text and "libffcx_" in text


def repair_ffcx_jit_timeout_cache(
    xdg_cache_home: Path,
    error: str,
) -> tuple[Path, ...]:
    """Repair exact FFCx lock files named by a JIT timeout message."""

    fenics_cache = Path(xdg_cache_home) / "fenics"
    if not fenics_cache.exists():
        return ()
    repaired: list[Path] = []
    for match in _FFCX_SOURCE_PATH_RE.finditer(str(error or "")):
        source = Path(match.group(1))
        if (
            source.name.startswith("libffcx_")
            and source.suffix == ".c"
            and _path_is_relative_to(source, fenics_cache)
        ):
            ready = source.with_suffix(".c.cached")
            if _compiled_ffcx_module_exists(fenics_cache, source):
                try:
                    ready.touch(exist_ok=True)
                except OSError:
                    pass
                else:
                    repaired.append(ready)
                continue
            for artifact in (source, source.with_suffix(".o"), ready):
                if not artifact.exists():
                    continue
                try:
                    artifact.unlink()
                except FileNotFoundError:
                    pass
                else:
                    repaired.append(artifact)
    return tuple(repaired)


def prepare_backend_worker_cache(repo: Path, profile: str) -> BackendWorkerCache:
    profile_name = str(profile or "default").strip() or "default"
    profile_root = backend_worker_profile_root(repo, profile_name)
    xdg_cache_home = profile_root / "xdg-cache"
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    removed = cleanup_stale_ffcx_jit_locks(xdg_cache_home)
    return BackendWorkerCache(
        profile=profile_name,
        profile_root=profile_root,
        xdg_cache_home=xdg_cache_home,
        removed_stale_jit_locks=removed,
    )


def prepare_inprocess_backend_runtime(
    *, repo: Path, profile: str | None = None
) -> BackendWorkerCache:
    """Attach the current process to the project/profile JIT cache."""

    profile_name = (
        str(profile or os.getenv("EIT_APP_GUI_RUNTIME_PROFILE", "default")).strip()
        or "default"
    )
    cache = prepare_backend_worker_cache(repo, profile_name)
    os.environ["XDG_CACHE_HOME"] = str(cache.xdg_cache_home)
    os.environ.setdefault("PYEIDORS_PETSC_CUDA_PROBE_CACHE", "1")
    os.environ.setdefault(
        "PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR",
        str(cache.xdg_cache_home / "pyeidors-capabilities"),
    )
    module = sys.modules.get("dolfinx.jit")
    if module is not None:
        options = getattr(module, "DOLFINX_DEFAULT_JIT_OPTIONS", None)
        if isinstance(options, dict) and "cache_dir" in options:
            _old_value, description = options["cache_dir"]
            options["cache_dir"] = (cache.xdg_cache_home / "fenics", description)
    return cache


@contextmanager
def backend_worker_profile_lock(repo: Path, profile: str) -> Iterator[None]:
    """Serialize workers sharing one profile cache to avoid FFCx cache races."""

    profile_root = backend_worker_profile_root(repo, profile)
    profile_root.mkdir(parents=True, exist_ok=True)
    lock_path = profile_root / ".profile.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        handle.close()


def backend_worker_env(
    *, repo: Path, profile: str
) -> tuple[dict[str, str], BackendWorkerCache]:
    """Build subprocess environment and attach the persistent profile cache."""

    cache = prepare_backend_worker_cache(repo, profile)
    py_path = f"{Path(repo) / 'src'}:{Path(repo)}"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{py_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else py_path
    )
    env["UV_NO_PROGRESS"] = env.get("UV_NO_PROGRESS", "1")
    env["PYEIDORS_ENV_SYNC_CACHE"] = env.get("PYEIDORS_ENV_SYNC_CACHE", "1")
    env["PYEIDORS_ENV_SYNC_CACHE_TTL_SECONDS"] = env.get(
        "PYEIDORS_ENV_SYNC_CACHE_TTL_SECONDS",
        "43200",
    )
    env["PYEIDORS_GUI_LAUNCH"] = env.get("PYEIDORS_GUI_LAUNCH", "1")
    env["PYEIDORS_ENV_SYNC_QUIET_DRIFT"] = env.get(
        "PYEIDORS_ENV_SYNC_QUIET_DRIFT",
        "1",
    )
    env["PYEIDORS_ENV_SYNC_QUIET_REPAIR"] = env.get(
        "PYEIDORS_ENV_SYNC_QUIET_REPAIR",
        "1",
    )
    env["XDG_CACHE_HOME"] = str(cache.xdg_cache_home)
    env["PYEIDORS_PETSC_CUDA_PROBE_CACHE"] = env.get(
        "PYEIDORS_PETSC_CUDA_PROBE_CACHE",
        "1",
    )
    env["PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR"] = env.get(
        "PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR",
        str(cache.xdg_cache_home / "pyeidors-capabilities"),
    )
    env["EIT_APP_GUI_RUNTIME_PROFILE"] = cache.profile
    env["EIT_APP_GUI_PROFILE"] = (
        "gpu" if cache.profile.endswith("-cuda") or cache.profile == "cuda" else "cpu"
    )
    if cache.profile in {"complex", "complex-cuda"}:
        env["EIT_APP_GUI_PRECISION"] = "complex128"
    elif cache.profile in {"complex64", "complex64-cuda"}:
        env["EIT_APP_GUI_PRECISION"] = "complex64"
    return env, cache


def backend_worker_command(
    *,
    profile: str,
    worker_args: list[str],
) -> tuple[list[str], str]:
    """Resolve how to launch one worker request."""

    profile_name = str(profile or "default").strip() or "default"
    launch_mode = (
        os.getenv("EIT_APP_BACKEND_WORKER_LAUNCH_MODE", "auto").strip().lower()
    )
    current_profile = (
        os.getenv("EIT_APP_GUI_RUNTIME_PROFILE", "default").strip() or "default"
    )
    if launch_mode in {"direct", "current-python"} or (
        launch_mode in {"", "auto"} and profile_name == current_profile
    ):
        return (
            [sys.executable, "-m", "eit_app.backend_worker", *worker_args],
            "current_python",
        )

    worker_cmd = "uv run python -m eit_app.backend_worker " + " ".join(
        shlex.quote(str(arg)) for arg in worker_args
    )
    cmd = ["nix", "--option", "warn-dirty", "false", "develop"]
    if profile_name != "default":
        cmd.append(f".#{profile_name}")
    cmd.extend(["--command", "bash", "-lc", worker_cmd])
    return cmd, "nix_develop"
