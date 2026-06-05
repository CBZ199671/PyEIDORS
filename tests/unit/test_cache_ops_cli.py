"""Cache operations CLI and doctor tests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.cache import CacheManager, CachePolicy
from pyeidors.cache import cli as cache_cli
from pyeidors.cache.ops import (
    cache_manager_status,
    doctor_cache,
    gc_cache,
    parse_size_bytes,
    summarize_gui_array_geometry_cache,
    summarize_import_health,
    summarize_backend_worker_caches,
    warm_backend_worker,
)


def _touch_old(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    old = time.time() - 3600.0
    os.utime(path, (old, old))


@pytest.fixture(autouse=True)
def _isolate_runtime_cache_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYEIDORS_CACHE_ROOT", str(tmp_path / ".pyeidors_cache"))
    monkeypatch.delenv("PYEIDORS_CACHE_REQUESTED_ROOT", raising=False)
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", raising=False)


def _set_runtime_cache_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / ".pyeidors_cache"
    monkeypatch.setenv("PYEIDORS_CACHE_ROOT", str(root))
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", raising=False)
    return root


def test_parse_size_bytes_accepts_decimal_and_binary_units() -> None:
    assert parse_size_bytes("20GB") == 20 * 1000**3
    assert parse_size_bytes("2GiB") == 2 * 1024**3
    assert parse_size_bytes("512k") == 512 * 1000
    assert parse_size_bytes(123) == 123


def test_backend_worker_cache_summary_detects_stale_ffcx_locks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root = _set_runtime_cache_root(monkeypatch, tmp_path)
    source = (
        cache_root
        / "gui_backend_worker"
        / "v1"
        / "complex64-cuda"
        / "xdg-cache"
        / "fenics"
        / "libffcx_forms_deadbeef.c"
    )
    _touch_old(source)
    probe = (
        cache_root
        / "gui_backend_worker"
        / "v1"
        / "complex64-cuda"
        / "xdg-cache"
        / "pyeidors-capabilities"
        / "petsc_cuda_deadbeef.json"
    )
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        json.dumps(
            {
                "schema": "petsc_cuda_runtime_probe_cache_v1",
                "key": "deadbeef",
                "result": {
                    "petsc_cuda": True,
                    "petsc_cuda_mat": True,
                    "petsc_cuda_vec": True,
                    "petsc_amgx": False,
                    "petsc_hypre": True,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_backend_worker_caches(
        repo=tmp_path,
        stale_after_seconds=0.0,
    )

    assert summary["profile_count"] == 1
    assert summary["total_stale_ffcx_locks"] == 1
    assert summary["total_capability_probe_files"] == 1
    assert summary["profiles"][0]["profile"] == "complex64-cuda"
    probe_summary = summary["profiles"][0]["capability_probe_cache"]
    assert probe_summary["count"] == 1
    assert probe_summary["latest"]["key"] == "deadbeef"
    assert probe_summary["latest"]["petsc_cuda"] is True
    assert probe_summary["latest"]["petsc_hypre"] is True


def test_cache_doctor_can_repair_stale_ffcx_locks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_STALE_JIT_LOCK_SECONDS", "0")
    cache_root = _set_runtime_cache_root(monkeypatch, tmp_path)
    source = (
        cache_root
        / "gui_backend_worker"
        / "v1"
        / "cuda"
        / "xdg-cache"
        / "fenics"
        / "libffcx_forms_orphan.c"
    )
    _touch_old(source)

    report = doctor_cache(
        repo=tmp_path,
        cache_dir=tmp_path / ".pyeidors_cache" / "v2",
        repair_jit=True,
    )

    assert report["backend_workers"]["removed_stale_jit_locks"] >= 1
    assert report["mesh_derived_artifacts"]["count"] == 0
    assert not source.exists()


def test_gc_cache_trims_persistent_cache_manager_store(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".pyeidors_cache" / "v2"
    manager = CacheManager(
        scope="both",
        cache_dir=cache_dir,
        policy=CachePolicy(
            disk_lifecycle="persistent",
            cleanup_on_exit=False,
            process_max_bytes=8 * 1024**2,
            disk_max_bytes=8 * 1024**2,
        ),
    )
    manager.get_or_compute(
        artifact="unit-array",
        payload={"id": "a"},
        compute_fn=lambda: np.ones(2048, dtype=np.float64),
        persist=True,
    )

    report = gc_cache(
        repo=tmp_path,
        cache_dir=cache_dir,
        max_bytes=1,
    )

    assert report["cache_manager_removed"] >= 1
    assert report["cache_manager_after"]["stats"]["disk_items"] == 0


def test_gc_cache_can_remove_legacy_array_cache_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".pyeidors_cache" / "v2"
    legacy = tmp_path / ".pyeidors_cache" / "legacy" / "old_rm.npz"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"legacy")

    dry_report = gc_cache(
        repo=tmp_path,
        cache_dir=cache_dir,
        max_bytes=1024,
        include_legacy_arrays=True,
        dry_run=True,
    )

    assert legacy.exists()
    assert dry_report["legacy_array_cleanup"]["removed_count"] == 1
    assert dry_report["legacy_array_cleanup"]["after"]["count"] == 1

    report = gc_cache(
        repo=tmp_path,
        cache_dir=cache_dir,
        max_bytes=1024,
        include_legacy_arrays=True,
    )

    assert not legacy.exists()
    assert report["legacy_arrays_included"] is True
    assert report["legacy_array_cleanup"]["removed_count"] == 1
    assert report["legacy_array_cleanup"]["after"]["count"] == 0


def test_cache_manager_status_reports_queryable_index_summary(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".pyeidors_cache" / "v2"
    manager = CacheManager(
        scope="both",
        cache_dir=cache_dir,
        policy=CachePolicy(disk_lifecycle="persistent", cleanup_on_exit=False),
    )
    manager.get_or_compute(
        artifact="forward_factor",
        name="forward_factor",
        namespace="forward",
        payload={
            "scalar_dtype": "complex64",
            "backend": "petsc",
            "petsc_backend": {"effective": "cuda"},
            "mesh": {"tdim": 3, "mesh_file_hash": "mesh-xyz"},
            "n_elec": 16,
        },
        compute_fn=lambda: {"factor": "ok"},
        persist=True,
    )

    report = cache_manager_status(cache_dir=cache_dir)

    index = report["index"]
    assert index["entry_count"] >= 1
    assert index["indexed_entry_count"] >= 1
    assert index["by_field"]["dtype"]["complex64"] >= 1
    assert index["by_field"]["backend"]["petsc"] >= 1
    assert index["by_field"]["device"]["cuda"] >= 1
    assert index["by_field"]["dim"]["3"] >= 1
    assert index["by_field"]["n_elec"]["16"] >= 1
    assert index["by_field"]["mesh_hash"]["mesh-xyz"] >= 1


def test_cache_cli_stats_writes_json_report(tmp_path: Path) -> None:
    output = tmp_path / "stats.json"

    cache_cli.main(
        [
            "stats",
            "--repo",
            str(tmp_path),
            "--cache-dir",
            str(tmp_path / ".pyeidors_cache" / "v2"),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "cache_manager" in payload
    assert "backend_workers" in payload
    assert "import_health" in payload
    assert "gui_array_geometry_cache" in payload


def test_cache_cli_list_filters_queryable_index_fields(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".pyeidors_cache" / "v2"
    manager = CacheManager(
        scope="both",
        cache_dir=cache_dir,
        policy=CachePolicy(disk_lifecycle="persistent", cleanup_on_exit=False),
    )
    manager.get_or_compute(
        artifact="rm_matrix",
        name="reconstruction_matrix",
        namespace="inverse",
        payload={
            "backend": "torch",
            "dtype": "float32",
            "device": "cuda",
            "n_elec": 16,
            "mesh": {"gdim": 3, "mesh_hash": "mesh-def"},
        },
        compute_fn=lambda: {"rm": [1.0, 2.0]},
        persist=True,
    )
    output = tmp_path / "cache-list.json"

    cache_cli.main(
        [
            "list",
            "--cache-dir",
            str(cache_dir),
            "--name",
            "reconstruction_matrix",
            "--namespace",
            "inverse",
            "--dtype",
            "float32",
            "--backend",
            "torch",
            "--device",
            "cuda",
            "--dim",
            "3",
            "--n-elec",
            "16",
            "--mesh-hash",
            "mesh-def",
            "--output",
            str(output),
        ]
    )

    entries = json.loads(output.read_text(encoding="utf-8"))
    assert entries
    assert all(entry["name"] == "reconstruction_matrix" for entry in entries)
    assert all(entry["dtype"] == "float32" for entry in entries)
    assert all(entry["backend"] == "torch" for entry in entries)
    assert all(entry["device"] == "cuda" for entry in entries)
    assert all(entry["dim"] == 3 for entry in entries)
    assert all(entry["n_elec"] == 16 for entry in entries)
    assert all(entry["mesh_hash"] == "mesh-def" for entry in entries)


def test_warm_backend_worker_can_repair_jit_and_report_profile_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root = _set_runtime_cache_root(monkeypatch, tmp_path)
    source = (
        cache_root
        / "gui_backend_worker"
        / "v1"
        / "cuda"
        / "xdg-cache"
        / "fenics"
        / "libffcx_forms_orphan.c"
    )
    _touch_old(source)

    def _fake_warm(*, repo, profile, progress_cb=None):
        profile_root = cache_root / "gui_backend_worker" / "v1" / str(profile)
        cache_home = profile_root / "xdg-cache"
        cache_home.mkdir(parents=True, exist_ok=True)
        if progress_cb is not None:
            progress_cb("fake worker warm")
        return SimpleNamespace(
            profile=str(profile),
            cache_home=cache_home,
            launch_mode="unit",
            pid=12345,
            reused_process=False,
            stale_jit_locks_removed=0,
            primed_runtime=True,
            prime_command="prime_runtime",
            prime_duration_ms=12.5,
            prime_metadata={"modules": ["pyeidors.core_system"]},
        )

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_STALE_JIT_LOCK_SECONDS", "0")
    monkeypatch.setattr(
        "eit_app.backend_worker_pool.warm_persistent_backend_worker",
        _fake_warm,
    )

    report = warm_backend_worker(repo=tmp_path, profile="cuda", repair_jit=True)

    assert not source.exists()
    assert report["profile"] == "cuda"
    assert report["pid"] == 12345
    assert report["prewarm_repair"]["removed_stale_jit_locks"] >= 1
    assert report["backend_worker_cache"]["profile"] == "cuda"
    assert report["backend_workers"]["profile_count"] == 1
    assert report["messages"] == ["fake worker warm"]
    assert report["primed_runtime"] is True
    assert report["prime_command"] == "prime_runtime"
    assert report["prime_duration_ms"] == 12.5
    assert report["prime_metadata"] == {"modules": ["pyeidors.core_system"]}


def test_v331_warm_backend_worker_caps_progress_messages(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root = _set_runtime_cache_root(monkeypatch, tmp_path)

    def _fake_warm(*, repo, profile, progress_cb=None):
        profile_root = cache_root / "gui_backend_worker" / "v1" / str(profile)
        cache_home = profile_root / "xdg-cache"
        cache_home.mkdir(parents=True, exist_ok=True)
        if progress_cb is not None:
            for index in range(5):
                progress_cb(f"warm message {index}")
        return SimpleNamespace(
            profile=str(profile),
            cache_home=cache_home,
            launch_mode="unit",
            pid=12345,
            reused_process=False,
            stale_jit_locks_removed=0,
        )

    monkeypatch.setenv("PYEIDORS_CACHE_WARM_MESSAGE_LIMIT", "2")
    monkeypatch.setattr(
        "eit_app.backend_worker_pool.warm_persistent_backend_worker",
        _fake_warm,
    )

    report = warm_backend_worker(repo=tmp_path, profile="cuda")

    assert report["messages"] == ["warm message 0", "warm message 1"]
    assert report["message_count"] == 5
    assert report["message_limit"] == 2
    assert report["messages_truncated"] == 3


def test_warm_backend_worker_can_setup_prime_forward_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request_path = tmp_path / "forward_request.h5"
    request_path.write_text("placeholder", encoding="utf-8")

    def _fake_prime(*, repo, profile, input_path, progress_cb=None):
        if progress_cb is not None:
            progress_cb("fake setup prime")
        return SimpleNamespace(
            profile=str(profile),
            cache_home=Path(repo) / ".cache" / str(profile),
            launch_mode="unit",
            pid=4321,
            reused_process=True,
            stale_jit_locks_removed=0,
            rss_bytes=1024,
            rss_limit_bytes=2048,
            recycled_after_request=False,
            recycle_reason="",
            primed_runtime=True,
            prime_command="prime_forward_setup",
            prime_duration_ms=44.0,
            prime_metadata={
                "forward_setup_prime": True,
                "forward_timing_ms": {"setup_mesh_and_forward_model": 33.0},
            },
            request_duration_ms=44.0,
        )

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.prime_persistent_backend_worker_forward_setup",
        _fake_prime,
    )

    report = warm_backend_worker(
        repo=tmp_path,
        profile="cuda",
        forward_request=request_path,
    )

    assert report["warm_mode"] == "forward_setup"
    assert report["forward_request"] == str(request_path)
    assert report["prime_command"] == "prime_forward_setup"
    assert report["prime_metadata"]["forward_setup_prime"] is True
    assert report["request_duration_ms"] == 44.0
    assert report["messages"] == ["fake setup prime"]


def test_cache_cli_warm_accepts_repair_jit_and_writes_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "warm.json"

    def _fake_warm_backend_worker(
        *, repo, profile, repair_jit=False, forward_request=None
    ):
        return {
            "repo": str(repo),
            "profile": profile,
            "repair_jit": repair_jit,
            "forward_request": str(forward_request) if forward_request else None,
            "pid": 7,
        }

    monkeypatch.setattr(cache_cli, "warm_backend_worker", _fake_warm_backend_worker)

    cache_cli.main(
        [
            "warm",
            "--repo",
            str(tmp_path),
            "--profile",
            "cuda",
            "--repair-jit",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["profile"] == "cuda"
    assert payload["repair_jit"] is True
    assert payload["forward_request"] is None
    assert payload["pid"] == 7


def test_cache_cli_warm_accepts_forward_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "warm.json"
    request_path = tmp_path / "forward_request.h5"
    request_path.write_text("placeholder", encoding="utf-8")

    def _fake_warm_backend_worker(
        *, repo, profile, repair_jit=False, forward_request=None
    ):
        return {
            "repo": str(repo),
            "profile": profile,
            "repair_jit": repair_jit,
            "forward_request": str(forward_request) if forward_request else None,
            "warm_mode": "forward_setup" if forward_request else "worker",
        }

    monkeypatch.setattr(cache_cli, "warm_backend_worker", _fake_warm_backend_worker)

    cache_cli.main(
        [
            "warm",
            "--repo",
            str(tmp_path),
            "--profile",
            "cuda",
            "--forward-request",
            str(request_path),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["warm_mode"] == "forward_setup"
    assert payload["forward_request"] == str(request_path)


def test_cache_cli_warm_can_generate_forward_setup_request(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "warm-generated.json"
    captured: dict[str, object] = {}

    def _fake_warm_backend_worker(
        *, repo, profile, repair_jit=False, forward_request=None
    ):
        from eit_app.backend_worker_protocol import read_forward_request

        assert forward_request is not None
        request = read_forward_request(forward_request)
        captured["repo"] = repo
        captured["profile"] = profile
        captured["repair_jit"] = repair_jit
        captured["mesh_dimension"] = request.mesh_dimension
        captured["n_electrodes"] = request.n_electrodes
        captured["config"] = dict(request.forward_model_config)
        return {
            "profile": profile,
            "warm_mode": "forward_setup",
            "forward_request": str(forward_request),
        }

    monkeypatch.setattr(cache_cli, "warm_backend_worker", _fake_warm_backend_worker)

    cache_cli.main(
        [
            "warm",
            "--repo",
            str(tmp_path),
            "--profile",
            "complex64-cuda",
            "--dim",
            "3",
            "--n-elec",
            "32",
            "--n-rings",
            "4",
            "--mesh-refinement",
            "0.2",
            "--petsc-device",
            "cuda",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generated_forward_request"] is True
    assert payload["warm_mode"] == "forward_setup"
    assert captured["profile"] == "complex64-cuda"
    assert captured["mesh_dimension"] == 3
    assert captured["n_electrodes"] == 32
    config = captured["config"]
    assert config["n_elec"] == 32
    assert config["n_rings"] == 4
    assert config["mesh_refinement"] == 0.2
    assert config["petsc_device"] == "cuda"
    assert payload["generated_forward_request_payload"]["mesh_dimension"] == 3


def test_import_health_probe_detects_lightweight_public_imports() -> None:
    report = summarize_import_health(repo=Path(__file__).resolve().parents[2])

    assert report["ok"] is True
    assert report["loaded_heavy_modules"] == []
    assert "pyeidors" in report["timings_seconds"]


def test_gui_array_geometry_cache_summary_reports_process_local_stats() -> None:
    from eit_app.ui.array_geometry_cache import (
        cached_cell_centers,
        clear_array_geometry_cache,
    )

    clear_array_geometry_cache()
    cached_cell_centers(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([[0, 1, 2]], dtype=np.int32),
    )

    report = summarize_gui_array_geometry_cache()

    assert report["available"] is True
    assert report["process_local"] is True
    assert report["stats"]["items"] == 1
    assert report["entries"][0]["cell_shape"] == (1, 3)
