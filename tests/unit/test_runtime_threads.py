from __future__ import annotations

from eit_app.runtime_threads import configure_realtime_thread_env, get_realtime_thread_count


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
