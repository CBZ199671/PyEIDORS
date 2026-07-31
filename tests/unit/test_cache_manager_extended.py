"""Extended tests for the phase-2 cache manager."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np

from pyeidors.cache import (
    CacheKeyParts,
    CacheManager,
    CachePolicy,
    build_cache_key,
    cleanup_registered_session_caches,
    cleanup_stale_session_caches,
)


def test_cache_key_stable_for_semantically_equal_payload():
    parts_a = CacheKeyParts(
        artifact="jacobian",
        payload={"b": [3, 4], "a": {"x": 1, "y": 2}},
        namespace="unit",
        code_fingerprint="abc",
    )
    parts_b = CacheKeyParts(
        artifact="jacobian",
        payload={"a": {"y": 2, "x": 1}, "b": [3, 4]},
        namespace="unit",
        code_fingerprint="abc",
    )
    assert build_cache_key(parts_a) == build_cache_key(parts_b)


def test_v774_cache_key_canonicalizes_python_and_numpy_complex_scalars() -> None:
    def _key(value: object) -> str:
        return build_cache_key(
            CacheKeyParts(
                artifact="complex-scalar",
                payload={"value": value},
                namespace="unit",
                code_fingerprint="abc",
            )
        )

    python_key = _key(1.0 + 2.0j)
    numpy_key = _key(np.complex64(1.0 + 2.0j))

    assert python_key == numpy_key
    assert python_key == _key(1.0 + 2.0j)
    assert python_key != _key(1.0 + 3.0j)


def test_cache_manager_process_hit():
    manager = CacheManager(
        scope="process", policy=CachePolicy(process_max_bytes=2 * 1024**2)
    )
    manager.clear(scope="process")
    calls = {"count": 0}

    def _compute():
        calls["count"] += 1
        return np.arange(8, dtype=float)

    v1, lookup1 = manager.get_or_compute(
        artifact="jacobian",
        payload={"id": 1},
        compute_fn=_compute,
        persist=False,
    )
    v2, lookup2 = manager.get_or_compute(
        artifact="jacobian",
        payload={"id": 1},
        compute_fn=_compute,
        persist=False,
    )

    assert calls["count"] == 1
    assert lookup1.hit is False
    assert lookup2.hit is True
    np.testing.assert_allclose(v1, v2)
    stats = manager.stats()
    assert stats["process_hits"] >= 1


def test_cache_manager_process_store_shared_across_managers(tmp_path: Path):
    cache_dir = tmp_path / "shared-process-cache"
    policy = CachePolicy(process_max_bytes=2 * 1024**2)
    manager1 = CacheManager(scope="process", cache_dir=cache_dir, policy=policy)
    manager2 = CacheManager(scope="process", cache_dir=cache_dir, policy=policy)
    manager1.clear(scope="process")
    calls = {"count": 0}

    def _compute():
        calls["count"] += 1
        return {"value": np.arange(4, dtype=float)}

    _, lookup1 = manager1.get_or_compute(
        artifact="mesh_bundle",
        payload={"mesh": "shared", "gdim": 3},
        compute_fn=_compute,
        persist=False,
    )
    value2, lookup2 = manager2.get_or_compute(
        artifact="mesh_bundle",
        payload={"mesh": "shared", "gdim": 3},
        compute_fn=_compute,
        persist=False,
    )

    assert lookup1.hit is False
    assert lookup2.hit is True
    assert lookup2.layer == "process"
    assert calls["count"] == 1
    np.testing.assert_allclose(value2["value"], np.arange(4, dtype=float))


def test_cache_manager_singleflights_concurrent_process_miss(tmp_path: Path):
    manager = CacheManager(
        scope="process",
        cache_dir=tmp_path / "singleflight-cache",
        policy=CachePolicy(process_max_bytes=2 * 1024**2),
        code_fingerprint="singleflight-test",
    )
    manager.clear(scope="process")
    started = threading.Event()
    release = threading.Event()
    calls_lock = threading.Lock()
    calls = {"count": 0}
    results: list[tuple[dict[str, np.ndarray], object]] = []
    errors: list[BaseException] = []

    def _compute():
        with calls_lock:
            calls["count"] += 1
        started.set()
        if not release.wait(timeout=5.0):
            raise AssertionError("singleflight compute was not released")
        return {"value": np.arange(3, dtype=float)}

    def _run_lookup():
        try:
            results.append(
                manager.get_or_compute(
                    artifact="jacobian",
                    payload={"id": "concurrent"},
                    compute_fn=_compute,
                    persist=False,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first = threading.Thread(target=_run_lookup)
    second = threading.Thread(target=_run_lookup)
    first.start()
    assert started.wait(timeout=2.0)
    second.start()
    time.sleep(0.1)
    with calls_lock:
        assert calls["count"] == 1
    release.set()
    first.join(timeout=3.0)
    second.join(timeout=3.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(results) == 2
    assert calls["count"] == 1
    assert {lookup.layer for _, lookup in results} == {"compute", "process"}
    for value, _lookup in results:
        np.testing.assert_allclose(value["value"], np.arange(3, dtype=float))


def test_cache_manager_disk_persistence(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    policy = CachePolicy(process_max_bytes=1024, disk_max_bytes=8 * 1024**2)
    manager1 = CacheManager(scope="both", cache_dir=cache_dir, policy=policy)
    manager1.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 0.1, "shape": (4, 4)},
        compute_fn=lambda: {"lu": np.eye(4), "piv": np.arange(4)},
        persist=True,
    )

    manager2 = CacheManager(scope="both", cache_dir=cache_dir, policy=policy)
    calls = {"count": 0}

    def _compute():
        calls["count"] += 1
        return {"lu": np.eye(4), "piv": np.arange(4)}

    _, lookup = manager2.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 0.1, "shape": (4, 4)},
        compute_fn=_compute,
        persist=True,
    )

    assert lookup.hit is True
    assert lookup.layer in {"disk", "process"}
    assert calls["count"] == 0
    stats = manager2.stats()
    assert stats["disk_hits"] + stats["process_hits"] >= 1


def test_cache_manager_off_scope_and_invalidation_paths(tmp_path: Path):
    disabled = CacheManager(scope="off")
    calls = {"count": 0}

    def _compute_disabled():
        calls["count"] += 1
        return {"value": 1}

    value, lookup = disabled.get_or_compute(
        artifact="mesh_bundle",
        payload={"id": 1},
        compute_fn=_compute_disabled,
        persist=True,
    )
    assert value == {"value": 1}
    assert lookup.layer == "disabled"
    assert calls["count"] == 1
    assert disabled.enabled is False

    manager = CacheManager(
        scope="both", cache_dir=tmp_path / "cache-two", policy=CachePolicy()
    )
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 1},
        compute_fn=lambda: np.eye(2),
        persist=True,
        cost=3.5,
    )
    # Explicit cost branch and both-layer invalidation/clear.
    removed = manager.invalidate(reason="test-clean")
    assert removed >= 1
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 2},
        compute_fn=lambda: np.eye(2),
        persist=True,
    )
    manager.clear(scope="process")
    stats_after_process_clear = manager.stats()
    assert stats_after_process_clear["process_items"] == 0
    manager.clear(scope="disk")
    stats_after_disk_clear = manager.stats()
    assert stats_after_disk_clear["disk_items"] == 0


def test_cache_manager_semantic_helpers_and_name_controls(tmp_path: Path):
    manager = CacheManager(
        scope="both",
        cache_dir=tmp_path / "semantic-cache",
        policy=CachePolicy(process_max_bytes=8 * 1024**2, disk_max_bytes=8 * 1024**2),
    )
    calls = {"count": 0}

    def _compute():
        calls["count"] += 1
        return {"value": 42}

    payload = {"model": {"n_elec": 16}, "sigma_hash": "abc"}
    value1, lookup1 = manager.get_or_compute_semantic(
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cache_obj=payload,
        payload={"mode": "measurement"},
        compute_fn=_compute,
        persist=True,
    )
    value2, lookup2 = manager.get_or_compute_semantic(
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cache_obj=payload,
        payload={"mode": "measurement"},
        compute_fn=_compute,
        persist=True,
    )

    assert value1 == {"value": 42}
    assert value2 == {"value": 42}
    assert lookup1.hit is False
    assert lookup2.hit is True
    assert calls["count"] == 1

    collected = manager.collect_recent(
        names=["inv_solve_diff_GN_one_step"],
        limit_per_name=1,
        namespace="difference",
    )
    assert collected["inv_solve_diff_GN_one_step"]

    removed = manager.clear_name("inv_solve_diff_GN_one_step", namespace="difference")
    assert removed >= 1


def test_cache_manager_list_entries_filters_queryable_index_fields(tmp_path: Path):
    cache_dir = tmp_path / "indexed-cache"
    policy = CachePolicy(
        disk_lifecycle="persistent",
        cleanup_on_exit=False,
        process_max_bytes=8 * 1024**2,
        disk_max_bytes=8 * 1024**2,
    )
    manager = CacheManager(scope="both", cache_dir=cache_dir, policy=policy)
    payload = {
        "backend": "petsc",
        "scalar_dtype": "complex64",
        "n_elec": 16,
        "petsc_backend": {"effective": "cuda"},
        "mesh": {"tdim": 3, "mesh_file_hash": "mesh-abc"},
    }

    manager.get_or_compute(
        artifact="forward_factor",
        name="forward_factor",
        namespace="forward",
        payload=payload,
        compute_fn=lambda: {"factor": "ok"},
        persist=True,
    )

    entries = manager.list_entries(
        name="forward_factor",
        namespace="forward",
        dtype="complex64",
        backend="petsc",
        device="cuda",
        dim=3,
        n_elec=16,
        mesh_hash="mesh-abc",
    )

    assert entries
    assert {entry["layer"] for entry in entries} == {"process", "disk"}
    for entry in entries:
        assert entry["dtype"] == "complex64"
        assert entry["backend"] == "petsc"
        assert entry["device"] == "cuda"
        assert entry["dim"] == 3
        assert entry["n_elec"] == 16
        assert entry["mesh_hash"] == "mesh-abc"

    assert manager.list_entries(name="forward_factor", dim=2) == []

    disk_only = CacheManager(scope="disk", cache_dir=cache_dir, policy=policy)
    disk_entries = disk_only.list_entries(
        dtype="complex64",
        backend="petsc",
        device="cuda",
        dim=3,
        n_elec=16,
        mesh_hash="mesh-abc",
    )
    assert len(disk_entries) == 1
    assert disk_entries[0]["layer"] == "disk"


def test_cache_manager_uses_session_disk_cache_by_default(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    manager1 = CacheManager(scope="both", cache_dir=cache_dir, policy=CachePolicy())
    manager2 = CacheManager(scope="both", cache_dir=cache_dir, policy=CachePolicy())

    assert manager1.disk_lifecycle == "session"
    assert manager1.session_cache_enabled is True
    assert manager1.requested_cache_dir == cache_dir
    assert manager1.cache_dir != cache_dir
    assert manager1.cache_dir.parent == cache_dir / ".sessions"
    assert manager1.cache_dir == manager2.cache_dir

    stats = manager1.stats()
    assert stats["disk_cache_lifecycle"] == "session"
    assert stats["disk_cache_requested_dir"] == str(cache_dir)
    assert stats["disk_cache_effective_dir"] == str(manager1.cache_dir)
    assert stats["disk_cache_cleanup_on_exit"] is True


def test_cache_manager_can_opt_into_persistent_disk_cache(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    policy = CachePolicy(disk_lifecycle="persistent", cleanup_on_exit=False)
    manager = CacheManager(scope="both", cache_dir=cache_dir, policy=policy)

    assert manager.disk_lifecycle == "persistent"
    assert manager.session_cache_enabled is False
    assert manager.requested_cache_dir == cache_dir
    assert manager.cache_dir == cache_dir


def test_cleanup_registered_session_caches_removes_effective_session_dir(
    tmp_path: Path, monkeypatch
):
    from pyeidors.cache import cleanup_registered_session_caches

    for name in (
        "PYEIDORS_CACHE_SESSION_ID",
        "PYEIDORS_CACHE_SESSION_DIR",
        "PYEIDORS_CACHE_REQUESTED_ROOT",
        "PYEIDORS_CACHE_OWNER_PID",
    ):
        monkeypatch.delenv(name, raising=False)

    cache_dir = tmp_path / "cache"
    manager = CacheManager(scope="both", cache_dir=cache_dir, policy=CachePolicy())
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 1},
        compute_fn=lambda: np.eye(2),
        persist=True,
    )
    session_dir = manager.cache_dir
    assert session_dir.exists()

    removed = cleanup_registered_session_caches()
    assert removed >= 1
    assert not session_dir.exists()


def test_cache_manager_prefers_shell_session_environment(tmp_path: Path, monkeypatch):
    cleanup_registered_session_caches()
    cache_root = tmp_path / "cache-root"
    session_id = "session-shellpid424242-demo"
    session_dir = cache_root / ".sessions" / session_id
    session_dir.mkdir(parents=True)

    monkeypatch.setenv("PYEIDORS_CACHE_SESSION_ID", session_id)
    monkeypatch.setenv("PYEIDORS_CACHE_SESSION_DIR", str(session_dir))
    monkeypatch.setenv("PYEIDORS_CACHE_REQUESTED_ROOT", str(cache_root))
    monkeypatch.setenv("PYEIDORS_CACHE_OWNER_PID", "424242")

    manager1 = CacheManager(scope="both", cache_dir=cache_root, policy=CachePolicy())
    manager2 = CacheManager(scope="both", cache_dir=cache_root, policy=CachePolicy())

    assert manager1.cache_dir == session_dir
    assert manager2.cache_dir == session_dir
    assert manager1.stats()["disk_cache_effective_dir"] == str(session_dir)

    removed = cleanup_registered_session_caches()
    assert removed == 0
    assert session_dir.exists()


def test_cache_manager_uses_shell_session_id_for_custom_root(
    tmp_path: Path, monkeypatch
):
    cleanup_registered_session_caches()
    default_root = tmp_path / "default-cache"
    custom_root = tmp_path / "custom-cache"
    session_id = "session-shellpid515151-demo"
    default_session_dir = default_root / ".sessions" / session_id
    default_session_dir.mkdir(parents=True)

    monkeypatch.setenv("PYEIDORS_CACHE_SESSION_ID", session_id)
    monkeypatch.setenv("PYEIDORS_CACHE_SESSION_DIR", str(default_session_dir))
    monkeypatch.setenv("PYEIDORS_CACHE_REQUESTED_ROOT", str(default_root))
    monkeypatch.setenv("PYEIDORS_CACHE_OWNER_PID", "515151")

    manager = CacheManager(scope="both", cache_dir=custom_root, policy=CachePolicy())
    expected_dir = custom_root / ".sessions" / session_id
    registry_path = default_session_dir / ".session-dirs"

    assert manager.cache_dir == expected_dir
    assert expected_dir.exists()
    assert registry_path.exists()
    entries = {
        line.strip()
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert str(default_session_dir) in entries
    assert str(expected_dir) in entries


def test_cleanup_stale_session_caches_removes_dead_shellpid_dirs(tmp_path: Path):
    cache_root = tmp_path / "cache-root"
    stale_dir = cache_root / ".sessions" / "session-shellpid999999-dead"
    stale_dir.mkdir(parents=True)
    (stale_dir / "payload.bin").write_text("x", encoding="utf-8")

    removed = cleanup_stale_session_caches(cache_root)

    assert removed == 1
    assert not stale_dir.exists()
