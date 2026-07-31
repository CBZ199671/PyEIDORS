"""V777 registry-wide SPEC integrity regression gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = REPO_ROOT / "scripts" / "ci" / "spec_integrity_guard.py"


def _load_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_spec_integrity_guard", GUARD_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_registry(
    root: Path,
    *,
    root_text: str,
    invariant_text: str,
    task_text: str,
    bug_text: str,
    id_map_text: str,
) -> None:
    paths = {
        "SPEC.md": root_text,
        "docs/spec/invariants/core.md": invariant_text,
        "docs/spec/history/tasks-completed.md": task_text,
        "docs/spec/history/bugs.md": bug_text,
        "docs/spec/id-map.tsv": id_map_text,
    }
    for relative, text in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    registry = {
        "schema": "pyeidors-spec-registry-v1",
        "root_files": ["SPEC.md"],
        "invariant_files": ["docs/spec/invariants/core.md"],
        "active_task_files": ["SPEC.md"],
        "completed_task_files": ["docs/spec/history/tasks-completed.md"],
        "bug_files": ["SPEC.md", "docs/spec/history/bugs.md"],
        "id_map": "docs/spec/id-map.tsv",
    }
    registry_path = root / "docs/spec/registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _valid_fixture(root: Path) -> None:
    _write_registry(
        root,
        root_text=(
            "| V1 | root invariant cites V2/B1/T2 | V2,B1,T2 |\n"
            "| T1 | . | active task | V1 |\n"
        ),
        invariant_text=r"| V2 | domain invariant accepts `a\|b` | V1 |" + "\n",
        task_text="| T2 | x | completed task | V2 |\n",
        bug_text="| B1 | 2026-08-01 | fixed cause | V1,T2 |\n",
        id_map_text=(
            "id\tfile\n"
            "V1\tSPEC.md\n"
            "V2\tdocs/spec/invariants/core.md\n"
            "T1\tSPEC.md\n"
            "T2\tdocs/spec/history/tasks-completed.md\n"
            "B1\tdocs/spec/history/bugs.md\n"
        ),
    )


def test_v777_real_spec_registry_is_valid() -> None:
    guard = _load_guard()

    report = guard.validate_repository(REPO_ROOT)

    assert report.ok
    assert {record.identifier for record in report.records} >= {"V1", "T1", "B1"}


def test_v777_duplicate_ids_fail_across_registered_files(tmp_path: Path) -> None:
    guard = _load_guard()
    _valid_fixture(tmp_path)
    invariant = tmp_path / "docs/spec/invariants/core.md"
    invariant.write_text("| V1 | duplicate invariant | V1 |\n", encoding="utf-8")

    with pytest.raises(guard.SpecIntegrityError, match="duplicate-id"):
        guard.validate_repository(tmp_path)


def test_v777_missing_reference_fails_closed(tmp_path: Path) -> None:
    guard = _load_guard()
    _valid_fixture(tmp_path)
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace("V2,B1,T2", "V99,B1,T2"),
        encoding="utf-8",
    )

    with pytest.raises(guard.SpecIntegrityError, match="missing-reference.*V99"):
        guard.validate_repository(tmp_path)


def test_v777_invalid_task_status_fails_closed(tmp_path: Path) -> None:
    guard = _load_guard()
    _valid_fixture(tmp_path)
    spec_path = tmp_path / "SPEC.md"
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace("| T1 | . |", "| T1 | ? |"),
        encoding="utf-8",
    )

    with pytest.raises(guard.SpecIntegrityError, match="invalid-task-status"):
        guard.validate_repository(tmp_path)


def test_v777_id_map_drift_fails_closed(tmp_path: Path) -> None:
    guard = _load_guard()
    _valid_fixture(tmp_path)
    id_map = tmp_path / "docs/spec/id-map.tsv"
    id_map.write_text("id\tfile\nV1\tSPEC.md\n", encoding="utf-8")

    with pytest.raises(guard.SpecIntegrityError, match="id-map-drift"):
        guard.validate_repository(tmp_path)
