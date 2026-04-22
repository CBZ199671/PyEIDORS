from __future__ import annotations

import sys

from eit_app.runtime_threads import (
    configure_realtime_compute_threads,
    configure_realtime_thread_env,
    get_realtime_thread_count,
)


def test_get_realtime_thread_count_defaults_to_one(monkeypatch):
    monkeypatch.delenv("EIT_APP_NUM_THREADS", raising=False)
    assert get_realtime_thread_count() == 1


def test_configure_realtime_thread_env_sets_common_thread_vars(monkeypatch):
    monkeypatch.setenv("EIT_APP_NUM_THREADS", "1")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    configure_realtime_thread_env(force=False)

    assert __import__("os").environ["OMP_NUM_THREADS"] == "1"
    assert __import__("os").environ["OPENBLAS_NUM_THREADS"] == "1"
    assert __import__("os").environ["MKL_NUM_THREADS"] == "1"


def test_configure_realtime_compute_threads_defers_torch_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    info = configure_realtime_compute_threads()

    assert info["torch"] == "deferred_until_import"


def test_configure_realtime_compute_threads_updates_loaded_torch(monkeypatch):
    class _FakeTorch:
        def __init__(self) -> None:
            self.num_threads = None
            self.num_interop_threads = None

        def set_num_threads(self, value: int) -> None:
            self.num_threads = value

        def set_num_interop_threads(self, value: int) -> None:
            self.num_interop_threads = value

    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    info = configure_realtime_compute_threads()

    assert info["torch_num_threads"] == 1
    assert info["torch_num_interop_threads"] == 1
    assert fake_torch.num_threads == 1
    assert fake_torch.num_interop_threads == 1
