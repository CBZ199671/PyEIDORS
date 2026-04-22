"""Realtime-friendly thread configuration for the GUI application."""

from __future__ import annotations

import os
import sys
from typing import Any

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def get_realtime_thread_count(default: int = 1) -> int:
    """Return the desired realtime thread count for the GUI process."""
    raw = os.getenv("EIT_APP_NUM_THREADS", "").strip()
    if not raw:
        return max(int(default), 1)
    try:
        return max(int(raw), 1)
    except ValueError:
        return max(int(default), 1)


def configure_realtime_thread_env(*, force: bool = False, default: int = 1) -> int:
    """Seed common BLAS/OpenMP env vars for realtime GUI responsiveness.

    This should be called before importing heavy numerical stacks so that
    NumPy / SciPy / PETSc backed components pick up the intended limits.
    """
    threads = get_realtime_thread_count(default=default)
    value = str(threads)
    for key in _THREAD_ENV_KEYS:
        if force:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return threads


def configure_realtime_compute_threads(
    *, default: int = 1, import_torch: bool = False
) -> dict[str, Any]:
    """Apply best-effort runtime thread limits for already-imported backends."""
    threads = get_realtime_thread_count(default=default)
    applied: dict[str, Any] = {"threads": threads}

    if "torch" not in sys.modules and not import_torch:
        applied["torch"] = "deferred_until_import"
        return applied

    try:
        import torch

        torch.set_num_threads(threads)
        applied["torch_num_threads"] = threads
        try:
            torch.set_num_interop_threads(1)
            applied["torch_num_interop_threads"] = 1
        except RuntimeError:
            applied["torch_num_interop_threads"] = "locked"
    except Exception as exc:  # pragma: no cover - optional backend
        applied["torch"] = f"unavailable: {exc}"

    return applied
