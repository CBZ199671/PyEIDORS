"""Ensure known CUQI import warnings are suppressed by project wrappers."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _skip_if_cuqi_missing() -> None:
    if importlib.util.find_spec("cuqi") is None:
        pytest.skip("CUQI is not installed")


def _run_python_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    python_path = env.get("PYTHONPATH", "")
    repo_root = str(REPO_ROOT)
    src_root = str(SRC_ROOT)
    injected = os.pathsep.join([src_root, repo_root])
    env["PYTHONPATH"] = (
        os.pathsep.join([injected, python_path]) if python_path else injected
    )
    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_import_pyeidors_suppresses_known_cuqi_warnings() -> None:
    _skip_if_cuqi_missing()
    proc = _run_python_snippet(
        "import warnings, importlib\n"
        "warnings.filterwarnings('error', message=r'pkg_resources is deprecated as an API', category=UserWarning)\n"
        "warnings.filterwarnings('error', message=r'Importing from numpy\\\\.matlib is deprecated', category=PendingDeprecationWarning)\n"
        "importlib.import_module('pyeidors')\n"
    )
    assert proc.returncode == 0, proc.stderr


def test_import_sparse_modules_suppress_known_cuqi_warnings() -> None:
    _skip_if_cuqi_missing()
    proc = _run_python_snippet(
        "import warnings, importlib\n"
        "warnings.filterwarnings('error', message=r'pkg_resources is deprecated as an API', category=UserWarning)\n"
        "warnings.filterwarnings('error', message=r'Importing from numpy\\\\.matlib is deprecated', category=PendingDeprecationWarning)\n"
        "importlib.import_module('pyeidors.inverse.solvers.eit_pde')\n"
        "importlib.import_module('pyeidors.inverse.solvers.sparse_bayesian_engine')\n"
    )
    assert proc.returncode == 0, proc.stderr
