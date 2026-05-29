"""Operational cache inspection and maintenance helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from .manager import CacheManager
from .types import CachePolicy
from .index_fields import CACHE_INDEX_FIELD_NAMES


_SIZE_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[kmgt]?i?b?)?\s*$", re.I)
_DEFAULT_PROGRESS_MESSAGE_LIMIT = 200
_IMPORT_HEALTH_MODULES = (
    "dolfinx",
    "petsc4py",
    "torch",
    "cuqi",
    "mpi4py",
    "gmsh",
    "meshio",
    "h5py",
    "scipy",
    "matplotlib",
    "pyeidors.core_system",
    "pyeidors.cache.manager",
    "pyeidors.cache.object_signature",
    "pyeidors.cache.store_disk",
    "pyeidors.cache.store_process",
    "pyeidors.cache.types",
    "pyeidors.data.eit_digit_metrics",
    "pyeidors.data.factor_sweep",
    "pyeidors.data.measurement_dataset",
    "pyeidors.data.synthetic_data",
    "pyeidors.data.voltage_digit_sweep",
    "pyeidors.electrodes.patterns",
    "pyeidors.forward.complex_support",
    "pyeidors.forward.cuda_structured_backend",
    "pyeidors.forward.eit_forward_model",
    "pyeidors.femx.helpers",
    "pyeidors.interop.geometry_exchange",
    "pyeidors.physics.current_drive",
    "pyeidors.physics.unit_consistency",
    "pyeidors.perf.gpu_kernels",
    "pyeidors.io.hdf5_artifacts",
    "pyeidors.visualization.eit_plots",
    "pyeidors.visualization.eit_plot_helpers",
    "pyeidors.visualization.eit_plot_renderers",
    "pyeidors.geometry.mesh_generator",
    "pyeidors.geometry.mesh_converter",
    "pyeidors.inverse.solvers.gauss_newton",
    "pyeidors.inverse.solvers.matrix_free_gn",
    "pyeidors.inverse.solvers.sparse_bayesian_engine",
    "pyeidors.inverse.regularization.base_regularization",
    "pyeidors.inverse.regularization.smoothness",
    "pyeidors.inverse.prior.laplace",
    "pyeidors.inverse.prior.rtr",
    "pyeidors.inverse.prior.tv_irls",
    "pyeidors.inverse.postprocess.temporal",
    "pyeidors.inverse.postprocess.tv",
    "pyeidors.inverse.reduced.lowrank_subspace",
    "pyeidors.inverse.reduced.reduced_gn_step",
    "pyeidors.inverse.reduced.snapshot_bank",
    "pyeidors.inverse.matrix_free.dual_mesh",
    "pyeidors.inverse.workflows.absolute",
    "pyeidors.inverse.workflows.difference",
    "pyeidors.inverse.workflows.sparse_bayesian",
    "pyeidors.inverse.greit",
    "pyeidors.inverse.jacobian.direct_jacobian",
    "pyeidors.inverse.jacobian.adjoint_jacobian",
)


@dataclass(frozen=True)
class WorkerCacheProfileSummary:
    """Summary for one persistent GUI backend worker cache profile."""

    profile: str
    path: str
    xdg_cache_home: str
    fenics_cache: str
    size_bytes: int
    ffcx_sources: int
    ffcx_ready_markers: int
    ffcx_shared_objects: int
    stale_ffcx_sources: int
    stale_ffcx_ready_markers: int
    removed_stale_jit_locks: int = 0
    capability_probe_cache: dict[str, Any] | None = None


def parse_size_bytes(value: str | int | float | None) -> int:
    """Parse a human byte size such as ``20GB`` or ``512MiB``."""

    if value is None:
        raise ValueError("size is required")
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    text = str(value).strip()
    match = _SIZE_RE.match(text)
    if match is None:
        raise ValueError(f"Invalid size value: {value!r}")
    number = float(match.group("num"))
    unit = (match.group("unit") or "b").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "ki": 1024,
        "kib": 1024,
        "m": 1000**2,
        "mb": 1000**2,
        "mi": 1024**2,
        "mib": 1024**2,
        "g": 1000**3,
        "gb": 1000**3,
        "gi": 1024**3,
        "gib": 1024**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise ValueError(f"Invalid size unit: {unit!r}")
    return max(0, int(number * multipliers[unit]))


def directory_size_bytes(path: str | Path) -> int:
    """Return recursive byte size for an existing directory or file."""

    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        try:
            return int(root.stat().st_size)
        except OSError:
            return 0
    total = 0
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += int(item.stat().st_size)
        except OSError:
            continue
    return total


def _compiled_ffcx_module_exists(fenics_cache: Path, source: Path) -> bool:
    stem = source.with_suffix("").name
    return any(fenics_cache.glob(f"{stem}*.so"))


def _is_stale(path: Path, *, now: float, stale_after: float) -> bool:
    try:
        return (now - path.stat().st_mtime) >= stale_after
    except OSError:
        return False


def _summarize_ffcx_cache(
    fenics_cache: Path,
    *,
    stale_after_seconds: float,
) -> dict[str, int]:
    now = time.time()
    sources = list(fenics_cache.glob("libffcx_*.c")) if fenics_cache.exists() else []
    ready_markers = (
        list(fenics_cache.glob("libffcx_*.c.cached")) if fenics_cache.exists() else []
    )
    shared_objects = (
        list(fenics_cache.glob("libffcx_*.so")) if fenics_cache.exists() else []
    )
    stale_sources = 0
    for source in sources:
        if _compiled_ffcx_module_exists(fenics_cache, source):
            continue
        if _is_stale(source, now=now, stale_after=stale_after_seconds):
            stale_sources += 1
    stale_ready = 0
    for ready in ready_markers:
        source = ready.with_suffix("")
        if source.exists():
            continue
        if _compiled_ffcx_module_exists(fenics_cache, source):
            continue
        if _is_stale(ready, now=now, stale_after=stale_after_seconds):
            stale_ready += 1
    return {
        "ffcx_sources": len(sources),
        "ffcx_ready_markers": len(ready_markers),
        "ffcx_shared_objects": len(shared_objects),
        "stale_ffcx_sources": stale_sources,
        "stale_ffcx_ready_markers": stale_ready,
    }


def _summarize_capability_probe_cache(xdg_cache_home: Path) -> dict[str, Any]:
    root = xdg_cache_home / "pyeidors-capabilities"
    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "count": 0,
            "size_bytes": 0,
            "latest": None,
        }
    files = sorted(root.glob("petsc_cuda_*.json"))
    latest_payload: dict[str, Any] | None = None
    latest_mtime = -1.0
    latest_path: Path | None = None
    for path in files:
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            continue
        if mtime < latest_mtime:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"schema": "invalid", "key": path.stem}
        latest_payload = payload
        latest_mtime = mtime
        latest_path = path
    latest = None
    if latest_payload is not None and latest_path is not None:
        result = latest_payload.get("result")
        result_map = result if isinstance(result, dict) else {}
        latest = {
            "path": str(latest_path),
            "schema": str(latest_payload.get("schema", "")),
            "key": str(latest_payload.get("key", "")),
            "mtime": latest_mtime,
            "petsc_cuda": bool(result_map.get("petsc_cuda", False)),
            "petsc_cuda_mat": bool(result_map.get("petsc_cuda_mat", False)),
            "petsc_cuda_vec": bool(result_map.get("petsc_cuda_vec", False)),
            "petsc_amgx": bool(result_map.get("petsc_amgx", False)),
            "petsc_hypre": bool(result_map.get("petsc_hypre", False)),
        }
    return {
        "root": str(root),
        "exists": True,
        "count": len(files),
        "size_bytes": sum(directory_size_bytes(path) for path in files),
        "latest": latest,
    }


def _backend_worker_v1_root(repo: Path) -> Path:
    override = os.getenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", "").strip()
    root = (
        Path(override).expanduser()
        if override
        else repo / ".pyeidors_cache" / "gui_backend_worker"
    )
    return root / "v1"


def summarize_backend_worker_caches(
    *,
    repo: str | Path,
    repair_jit: bool = False,
    stale_after_seconds: float = 60.0,
) -> dict[str, Any]:
    """Inspect profile-scoped GUI backend worker caches."""

    repo_path = Path(repo)
    root = _backend_worker_v1_root(repo_path)
    profiles: list[WorkerCacheProfileSummary] = []
    removed_total = 0
    if root.exists():
        for profile_root in sorted(path for path in root.iterdir() if path.is_dir()):
            xdg_cache_home = profile_root / "xdg-cache"
            fenics_cache = xdg_cache_home / "fenics"
            removed = 0
            if repair_jit:
                try:
                    from eit_app.backend_worker_runtime import (
                        cleanup_stale_ffcx_jit_locks,
                    )

                    removed = len(cleanup_stale_ffcx_jit_locks(xdg_cache_home))
                except Exception:
                    removed = 0
            removed_total += removed
            ffcx = _summarize_ffcx_cache(
                fenics_cache,
                stale_after_seconds=float(max(0.0, stale_after_seconds)),
            )
            profiles.append(
                WorkerCacheProfileSummary(
                    profile=profile_root.name,
                    path=str(profile_root),
                    xdg_cache_home=str(xdg_cache_home),
                    fenics_cache=str(fenics_cache),
                    size_bytes=directory_size_bytes(profile_root),
                    removed_stale_jit_locks=removed,
                    capability_probe_cache=_summarize_capability_probe_cache(
                        xdg_cache_home
                    ),
                    **ffcx,
                )
            )
    profile_dicts = [asdict(item) for item in profiles]
    return {
        "root": str(root),
        "exists": root.exists(),
        "profile_count": len(profile_dicts),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in profile_dicts),
        "total_stale_ffcx_locks": sum(
            int(item["stale_ffcx_sources"]) + int(item["stale_ffcx_ready_markers"])
            for item in profile_dicts
        ),
        "removed_stale_jit_locks": removed_total,
        "total_capability_probe_files": sum(
            int((item.get("capability_probe_cache") or {}).get("count", 0))
            for item in profile_dicts
        ),
        "total_capability_probe_size_bytes": sum(
            int((item.get("capability_probe_cache") or {}).get("size_bytes", 0))
            for item in profile_dicts
        ),
        "profiles": profile_dicts,
    }


def _count_legacy_arrays(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"count": 0, "size_bytes": 0, "examples": []}
    files = [
        path
        for pattern in ("*.npz", "*.npy")
        for path in root.rglob(pattern)
        if path.is_file()
    ]
    examples = [str(path) for path in sorted(files)[:10]]
    return {
        "count": len(files),
        "size_bytes": sum(directory_size_bytes(path) for path in files),
        "examples": examples,
    }


def _remove_legacy_arrays(root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    before = _count_legacy_arrays(root)
    removed: list[str] = []
    if root.exists():
        files = [
            path
            for pattern in ("*.npz", "*.npy")
            for path in root.rglob(pattern)
            if path.is_file()
        ]
        for path in sorted(files):
            removed.append(str(path))
            if not dry_run:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
    after = _count_legacy_arrays(root)
    return {
        "dry_run": bool(dry_run),
        "before": before,
        "after": after,
        "removed_count": len(removed),
        "removed_examples": removed[:10],
    }


def _count_hdf5_artifacts(root: Path, *, subdir: str) -> dict[str, Any]:
    target = root / subdir
    if not target.exists():
        return {"root": str(target), "count": 0, "size_bytes": 0, "examples": []}
    files = [
        path
        for pattern in ("*.h5", "*.hdf5")
        for path in target.rglob(pattern)
        if path.is_file()
    ]
    return {
        "root": str(target),
        "count": len(files),
        "size_bytes": sum(directory_size_bytes(path) for path in files),
        "examples": [str(path) for path in sorted(files)[:10]],
    }


def cache_manager_status(*, cache_dir: str | Path) -> dict[str, Any]:
    """Return persistent CacheManager status without touching session caches."""

    manager = CacheManager(
        scope="both",
        cache_dir=cache_dir,
        policy=CachePolicy(disk_lifecycle="persistent", cleanup_on_exit=False),
    )
    return {
        "cache_status": manager.status(),
        "debug_status": manager.debug_status(),
        "stats": manager.stats(),
        "index": cache_index_summary(manager.list_entries(limit=None)),
    }


def cache_index_summary(
    entries: list[dict[str, Any]], *, max_combinations: int = 50
) -> dict[str, Any]:
    """Summarize queryable CacheManager index fields for telemetry."""

    by_field: dict[str, dict[str, int]] = {
        field: {} for field in CACHE_INDEX_FIELD_NAMES
    }
    missing: dict[str, int] = {field: 0 for field in CACHE_INDEX_FIELD_NAMES}
    combinations: dict[tuple[str, ...], dict[str, Any]] = {}
    for entry in entries:
        values: list[str] = []
        for field in CACHE_INDEX_FIELD_NAMES:
            raw = entry.get(field)
            text = "" if raw is None else str(raw)
            if not text:
                missing[field] += 1
                text = "unknown"
            else:
                by_field[field][text] = by_field[field].get(text, 0) + 1
            values.append(text)
        key = tuple(values)
        combo = combinations.setdefault(
            key,
            {
                "count": 0,
                "size_bytes": 0,
                **{
                    field: values[index]
                    for index, field in enumerate(CACHE_INDEX_FIELD_NAMES)
                },
            },
        )
        combo["count"] = int(combo["count"]) + 1
        combo["size_bytes"] = int(combo["size_bytes"]) + int(
            entry.get("size_bytes", 0) or 0
        )
    combination_rows = sorted(
        combinations.values(),
        key=lambda row: (int(row["count"]), int(row["size_bytes"])),
        reverse=True,
    )
    return {
        "entry_count": len(entries),
        "indexed_entry_count": sum(
            1
            for entry in entries
            if any(
                entry.get(field) not in {None, ""} for field in CACHE_INDEX_FIELD_NAMES
            )
        ),
        "missing": missing,
        "by_field": {
            field: dict(
                sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
            )
            for field, counts in by_field.items()
        },
        "combinations": combination_rows[: max(0, int(max_combinations))],
        "combinations_truncated": max(
            0, len(combination_rows) - max(0, int(max_combinations))
        ),
    }


def summarize_import_health(
    *,
    repo: str | Path,
    python_executable: str | Path | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Probe whether lightweight public imports pull heavy runtime modules."""

    repo_path = Path(repo)
    python = str(python_executable or sys.executable)
    script = f"""
import importlib
import json
import sys
import time

targets = [
    "pyeidors",
    "pyeidors.cache",
    "pyeidors.data",
    "pyeidors.electrodes",
    "pyeidors.femx",
    "pyeidors.forward",
    "pyeidors.geometry",
    "pyeidors.interop",
    "pyeidors.io",
    "pyeidors.perf",
    "pyeidors.physics",
    "pyeidors.visualization",
    "pyeidors.inverse",
    "pyeidors.inverse.solvers",
    "pyeidors.inverse.regularization",
    "pyeidors.inverse.prior",
    "pyeidors.inverse.postprocess",
    "pyeidors.inverse.reduced",
    "pyeidors.inverse.matrix_free",
    "pyeidors.inverse.workflows",
    "pyeidors.inverse.jacobian",
]
timings = {{}}
for target in targets:
    t0 = time.perf_counter()
    importlib.import_module(target)
    timings[target] = time.perf_counter() - t0
heavy = {list(_IMPORT_HEALTH_MODULES)!r}
loaded = [name for name in heavy if name in sys.modules]
print(json.dumps({{
    "ok": not loaded,
    "loaded_heavy_modules": loaded,
    "timings_seconds": timings,
    "targets": targets,
}}, sort_keys=True))
"""
    env = os.environ.copy()
    src_path = str(repo_path / "src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else src_path + os.pathsep + str(env["PYTHONPATH"])
    )
    try:
        result = subprocess.run(
            [python, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(max(timeout_seconds, 1.0)),
            env=env,
            cwd=repo_path if repo_path.exists() else None,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "python": python,
            "repo": str(repo_path),
        }

    payload: dict[str, Any]
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {
            "ok": False,
            "error": "invalid_probe_output",
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    payload["python"] = python
    payload["repo"] = str(repo_path)
    payload["returncode"] = int(result.returncode)
    if result.returncode != 0:
        payload["ok"] = False
        payload.setdefault("stderr", result.stderr)
    return payload


def summarize_gui_array_geometry_cache(*, limit: int = 16) -> dict[str, Any]:
    """Return process-local GUI array geometry cache stats when importable."""

    try:
        from eit_app.ui.array_geometry_cache import array_geometry_cache_snapshot

        payload = array_geometry_cache_snapshot(limit=limit)
    except Exception as exc:
        return {
            "available": False,
            "process_local": True,
            "error": str(exc),
        }
    return {"available": True, **payload}


def doctor_cache(
    *,
    repo: str | Path,
    cache_dir: str | Path,
    repair_jit: bool = False,
) -> dict[str, Any]:
    """Return a high-level cache health report."""

    repo_path = Path(repo)
    manager = cache_manager_status(cache_dir=cache_dir)
    worker = summarize_backend_worker_caches(
        repo=repo_path,
        repair_jit=repair_jit,
        stale_after_seconds=float(
            os.getenv("EIT_APP_BACKEND_WORKER_STALE_JIT_LOCK_SECONDS", "60")
        ),
    )
    mesh_derived = _count_hdf5_artifacts(Path(cache_dir), subdir="mesh_derived")
    legacy = _count_legacy_arrays(repo_path / ".pyeidors_cache")
    import_health = summarize_import_health(repo=repo_path)
    gui_array_geometry = summarize_gui_array_geometry_cache()
    warnings: list[str] = []
    if int(worker["total_stale_ffcx_locks"]) > 0:
        warnings.append(
            "stale_ffcx_jit_locks: run `eit-cache doctor --repair-jit` or retry warm."
        )
    if int(worker["profile_count"]) == 0:
        warnings.append(
            "no_backend_worker_profiles: run `eit-cache warm --profile ...`."
        )
    if int(legacy["count"]) > 0:
        warnings.append("legacy_npz_npy_cache_files_present: prefer HDF5 artifacts.")
    if not bool(import_health.get("ok", False)):
        warnings.append("lightweight_import_health_failed: inspect import_health.")
    status = "ok" if not warnings else "warning"
    return {
        "status": status,
        "repo": str(repo_path),
        "cache_manager": manager,
        "backend_workers": worker,
        "import_health": import_health,
        "gui_array_geometry_cache": gui_array_geometry,
        "mesh_derived_artifacts": mesh_derived,
        "legacy_array_cache_files": legacy,
        "warnings": warnings,
    }


def gc_cache(
    *,
    repo: str | Path,
    cache_dir: str | Path,
    max_bytes: int,
    include_worker_cache: bool = False,
    include_legacy_arrays: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Trim persistent CacheManager disk cache and optionally worker profiles."""

    before = cache_manager_status(cache_dir=cache_dir)
    removed = 0
    if not dry_run:
        manager = CacheManager(
            scope="both",
            cache_dir=cache_dir,
            policy=CachePolicy(disk_lifecycle="persistent", cleanup_on_exit=False),
        )
        removed = manager.clear_max(max_bytes=max_bytes)
    after = cache_manager_status(cache_dir=cache_dir)
    worker_before = summarize_backend_worker_caches(repo=repo)
    worker_removed: list[str] = []
    worker_after = worker_before
    if include_worker_cache:
        worker_root = Path(worker_before["root"])
        profile_rows = sorted(
            worker_before["profiles"],
            key=lambda row: (
                Path(row["path"]).stat().st_mtime if Path(row["path"]).exists() else 0
            ),
        )
        total = int(worker_before["total_size_bytes"])
        for row in profile_rows:
            if total <= max_bytes:
                break
            path = Path(row["path"])
            worker_removed.append(str(path))
            total -= int(row["size_bytes"])
            if not dry_run and path.exists():
                shutil.rmtree(path, ignore_errors=True)
        worker_after = summarize_backend_worker_caches(repo=repo)
        if not worker_root.exists() and not dry_run:
            worker_root.mkdir(parents=True, exist_ok=True)
    legacy_cleanup = None
    if include_legacy_arrays:
        legacy_cleanup = _remove_legacy_arrays(
            Path(repo) / ".pyeidors_cache",
            dry_run=bool(dry_run),
        )
    return {
        "dry_run": bool(dry_run),
        "max_bytes": int(max_bytes),
        "cache_manager_removed": int(removed),
        "cache_manager_before": before,
        "cache_manager_after": after,
        "worker_cache_included": bool(include_worker_cache),
        "worker_profiles_removed": worker_removed,
        "worker_before": worker_before,
        "worker_after": worker_after,
        "legacy_arrays_included": bool(include_legacy_arrays),
        "legacy_array_cleanup": legacy_cleanup,
    }


def _profile_summary_from_worker_report(
    report: dict[str, Any],
    profile: str,
) -> dict[str, Any] | None:
    wanted = str(profile or "default").strip().lower() or "default"
    for row in report.get("profiles", []):
        if str(row.get("profile", "")).strip().lower() == wanted:
            return dict(row)
    return None


def _progress_message_limit_from_env(
    env_name: str = "PYEIDORS_CACHE_WARM_MESSAGE_LIMIT",
    default: int = _DEFAULT_PROGRESS_MESSAGE_LIMIT,
) -> int:
    raw = os.environ.get(env_name)
    if raw is None:
        return int(default)
    try:
        return max(0, int(raw))
    except ValueError:
        return int(default)


class BoundedProgressMessageCollector:
    """Collect a bounded preview of progress messages while counting all events."""

    def __init__(self, *, limit: int = _DEFAULT_PROGRESS_MESSAGE_LIMIT) -> None:
        self.limit = max(0, int(limit))
        self.messages: list[str] = []
        self.count = 0

    def append(self, message: object) -> None:
        self.count += 1
        if len(self.messages) < self.limit:
            self.messages.append(str(message))

    @property
    def truncated(self) -> int:
        return max(0, self.count - len(self.messages))

    def report_fields(self) -> dict[str, Any]:
        return {
            "messages": list(self.messages),
            "message_count": int(self.count),
            "message_limit": int(self.limit),
            "messages_truncated": int(self.truncated),
        }


def warm_backend_worker(
    *,
    repo: str | Path,
    profile: str,
    repair_jit: bool = False,
    forward_request: str | Path | None = None,
) -> dict[str, Any]:
    """Start or reuse a persistent GUI backend worker for one profile."""

    from eit_app.backend_worker_pool import (
        prime_persistent_backend_worker_forward_setup,
        warm_persistent_backend_worker,
    )

    repo_path = Path(repo)
    profile_name = str(profile or "default").strip() or "default"
    prewarm_repair = (
        summarize_backend_worker_caches(repo=repo_path, repair_jit=True)
        if repair_jit
        else None
    )
    messages = BoundedProgressMessageCollector(
        limit=_progress_message_limit_from_env(),
    )
    if forward_request is not None:
        request_path = Path(forward_request)
        meta = prime_persistent_backend_worker_forward_setup(
            repo=repo_path,
            profile=profile_name,
            input_path=request_path,
            progress_cb=messages.append,
        )
        warm_mode = "forward_setup"
    else:
        request_path = None
        meta = warm_persistent_backend_worker(
            repo=repo_path,
            profile=profile_name,
            progress_cb=messages.append,
        )
        warm_mode = "worker"
    if meta is None:
        raise RuntimeError("backend worker did not return warmup metadata")
    postwarm = summarize_backend_worker_caches(repo=repo_path)
    return {
        "warm_mode": warm_mode,
        "profile": meta.profile,
        "cache_home": str(meta.cache_home),
        "launch_mode": meta.launch_mode,
        "pid": meta.pid,
        "reused_process": meta.reused_process,
        "stale_jit_locks_removed": meta.stale_jit_locks_removed,
        "rss_bytes": getattr(meta, "rss_bytes", 0),
        "rss_limit_bytes": getattr(meta, "rss_limit_bytes", 0),
        "recycled_after_request": getattr(meta, "recycled_after_request", False),
        "recycle_reason": getattr(meta, "recycle_reason", ""),
        "primed_runtime": getattr(meta, "primed_runtime", False),
        "prime_command": getattr(meta, "prime_command", ""),
        "prime_duration_ms": getattr(meta, "prime_duration_ms", 0.0),
        "prime_metadata": getattr(meta, "prime_metadata", {}),
        "request_duration_ms": getattr(meta, "request_duration_ms", 0.0),
        "forward_request": str(request_path) if request_path is not None else None,
        **messages.report_fields(),
        "prewarm_repair": prewarm_repair,
        "backend_worker_cache": _profile_summary_from_worker_report(
            postwarm,
            meta.profile,
        ),
        "backend_workers": postwarm,
    }


def build_forward_setup_warm_request(
    *,
    dim: int = 3,
    mesh_refinement: float | None = None,
    n_elec: int = 16,
    n_rings: int | None = None,
    radius: float = 1.0,
    height: float | None = None,
    electrode_coverage: float = 0.5,
    electrode_height_ratio: float = 0.2,
    electrode_level_fractions: list[float] | None = None,
    electrode_layout: str | None = None,
    measurement_protocol: str | None = None,
    stim_pattern: str = "{ad}",
    meas_pattern: str = "{ad}",
    mesh_family: str | None = None,
    geometry_version: str | None = None,
    forward_backend: str = "dolfinx",
    acceleration_profile: str | None = None,
    forward_solver_preset: str = "auto",
    forward_mat_solve: str = "auto",
    petsc_device: str = "auto",
    background_conductivity: float = 1.0,
    noise_level: float = 0.0,
) -> Any:
    """Build a GUI-style forward setup request for cache warmup."""

    from eit_app.controllers.forward_solver_controller import ForwardSolverRequest

    use_dim = int(dim)
    if use_dim not in {2, 3}:
        raise ValueError("dim must be 2 or 3")
    use_n_rings = int(n_rings if n_rings is not None else (2 if use_dim == 3 else 1))
    use_refinement = float(
        mesh_refinement
        if mesh_refinement is not None
        else (0.25 if use_dim == 3 else 0.1)
    )
    use_height = float(height if height is not None else (1.0 if use_dim == 3 else 2.0))
    levels = (
        list(float(v) for v in electrode_level_fractions)
        if electrode_level_fractions
        else _default_electrode_level_fractions(dim=use_dim, n_rings=use_n_rings)
    )
    config = {
        "mesh_dimension": use_dim,
        "mesh_refinement": use_refinement,
        "n_elec": int(n_elec),
        "n_electrodes": int(n_elec),
        "n_rings": use_n_rings,
        "electrode_layout": str(
            electrode_layout or ("ring_major" if use_dim == 3 else "circular")
        ),
        "measurement_protocol": str(
            measurement_protocol or ("eidors_full_3d" if use_dim == 3 else "adjacent")
        ),
        "stim_pattern": str(stim_pattern),
        "meas_pattern": str(meas_pattern),
        "rotate_meas": True,
        "use_meas_current": False,
        "use_meas_current_next": 0,
        "background_conductivity": float(background_conductivity),
        "noise_level": float(noise_level),
        "radius": float(radius),
        "height": use_height,
        "electrode_coverage": float(electrode_coverage),
        "electrode_height_ratio": float(electrode_height_ratio),
        "electrode_level_fractions": levels,
        "z_center": 0.0,
        "mesh_family": str(mesh_family or ("tetra" if use_dim == 3 else "triangle")),
        "geometry_version": str(
            geometry_version or ("geomv2" if use_dim == 3 else "legacy")
        ),
        "forward_backend": str(forward_backend),
        "acceleration_profile": str(
            acceleration_profile or ("gpu3d" if use_dim == 3 else "auto")
        ),
        "forward_solver_preset": str(forward_solver_preset),
        "forward_mat_solve": str(forward_mat_solve),
        "petsc_device": str(petsc_device),
    }
    return ForwardSolverRequest(
        mesh_dimension=use_dim,
        mesh_refinement=use_refinement,
        n_electrodes=int(n_elec),
        background_conductivity=float(background_conductivity),
        inhomogeneities=[],
        noise_level=float(noise_level),
        forward_model_config=config,
    )


def _default_electrode_level_fractions(*, dim: int, n_rings: int) -> list[float]:
    if int(dim) != 3:
        return []
    rings = max(1, int(n_rings))
    if rings == 1:
        return [0.5]
    return [float((index + 0.5) / rings) for index in range(rings)]
