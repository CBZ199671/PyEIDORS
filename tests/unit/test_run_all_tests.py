"""Regression tests for the advertised comprehensive test runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "tests" / "run_all_tests.py"


def _load_runner_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_pyeidors_run_all_tests",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v773_default_runner_targets_existing_unit_tests() -> None:
    module = _load_runner_module()

    assert module.DEFAULT_TESTS
    for _name, test_path in module.DEFAULT_TESTS:
        assert Path(test_path).is_file()
        assert Path(test_path).parent == REPO_ROOT / "tests" / "unit"


def test_v773_runner_invokes_pytest_instead_of_plain_python(monkeypatch) -> None:
    module = _load_runner_module()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def _fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    runner = module.TestRunner()

    assert runner.run_test("Basic Module Test", module.DEFAULT_TESTS[0][1])
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == [module.sys.executable, "-m", "pytest"]
    assert command[3] == str(module.DEFAULT_TESTS[0][1])
    assert command[4:] == ["-q", "--no-cov"]
    assert kwargs["cwd"] == module.REPO_ROOT
