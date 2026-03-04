"""Shared test helpers for running isolated Python subprocesses."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Mapping


def run_python(code: str, *, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = _merge_env(env)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


def _merge_env(env: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(os.environ)
    merged.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    merged.setdefault("OMP_NUM_THREADS", "1")
    if env:
        merged.update(env)
    return merged
