#!/usr/bin/env python3
"""Validate the split SPEC registry and all cross-document identifiers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


REGISTRY_SCHEMA = "pyeidors-spec-registry-v1"
ID_MAP_HEADER = "id\tfile"
ID_TOKEN_RE = re.compile(r"\b([VTB]\d+)\b")
V_ROW_RE = re.compile(r"^\| (V\d+) \|.*\|$")
T_ROW_RE = re.compile(r"^\| (T\d+) \| ([^|]+) \|.*\|$")
B_ROW_RE = re.compile(r"^\| (B\d+) \|.*\|$")
VALID_TASK_STATUSES = frozenset({"x", "~", "."})
ID_KIND_ORDER = {"V": 0, "T": 1, "B": 2}


@dataclass(frozen=True)
class SpecRecord:
    identifier: str
    file: str
    line: int
    status: str = ""


@dataclass(frozen=True)
class SpecIssue:
    code: str
    message: str


@dataclass(frozen=True)
class SpecValidationReport:
    records: tuple[SpecRecord, ...]
    references: tuple[str, ...]
    issues: tuple[SpecIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


class SpecIntegrityError(RuntimeError):
    """Raised when a registered SPEC document violates registry invariants."""

    def __init__(self, issues: Sequence[SpecIssue]):
        self.issues = tuple(issues)
        detail = "\n".join(f"[{issue.code}] {issue.message}" for issue in issues)
        super().__init__(detail)


def _id_sort_key(identifier: str) -> tuple[int, int]:
    return ID_KIND_ORDER[identifier[0]], int(identifier[1:])


def _normalized_relative_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"registry path must stay repo-relative: {path!r}")
    return candidate.as_posix()


def _registry_paths(payload: Mapping[str, object]) -> tuple[str, ...]:
    paths: list[str] = []
    for field in (
        "root_files",
        "invariant_files",
        "active_task_files",
        "completed_task_files",
        "bug_files",
    ):
        raw = payload.get(field, [])
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError(f"registry field {field!r} must be a list of paths")
        paths.extend(_normalized_relative_path(item) for item in raw)
    return tuple(dict.fromkeys(paths))


def _parse_records(
    documents: Mapping[str, str],
) -> tuple[list[SpecRecord], list[str], list[SpecIssue]]:
    records: list[SpecRecord] = []
    references: list[str] = []
    issues: list[SpecIssue] = []
    for file, text in documents.items():
        for line_number, line in enumerate(text.splitlines(), 1):
            references.extend(ID_TOKEN_RE.findall(line))
            if match := V_ROW_RE.match(line):
                records.append(SpecRecord(match.group(1), file, line_number))
                continue
            if match := T_ROW_RE.match(line):
                status = match.group(2).strip()
                records.append(
                    SpecRecord(match.group(1), file, line_number, status=status)
                )
                if status not in VALID_TASK_STATUSES:
                    issues.append(
                        SpecIssue(
                            "invalid-task-status",
                            f"{file}:{line_number} {match.group(1)} status={status!r}",
                        )
                    )
                continue
            if match := B_ROW_RE.match(line):
                records.append(SpecRecord(match.group(1), file, line_number))
    return records, references, issues


def _expected_id_map(records: Sequence[SpecRecord]) -> str:
    rows = [ID_MAP_HEADER]
    rows.extend(
        f"{record.identifier}\t{record.file}"
        for record in sorted(records, key=lambda item: _id_sort_key(item.identifier))
    )
    return "\n".join(rows) + "\n"


def validate_documents(
    documents: Mapping[str, str],
    *,
    active_task_files: Sequence[str],
    completed_task_files: Sequence[str],
    id_map_text: str,
) -> SpecValidationReport:
    """Validate already-loaded registry documents without filesystem effects."""

    records, references, issues = _parse_records(documents)
    by_id: dict[str, list[SpecRecord]] = {}
    for record in records:
        by_id.setdefault(record.identifier, []).append(record)
    for identifier, definitions in sorted(
        by_id.items(), key=lambda item: _id_sort_key(item[0])
    ):
        if len(definitions) > 1:
            locations = ", ".join(
                f"{record.file}:{record.line}" for record in definitions
            )
            issues.append(
                SpecIssue("duplicate-id", f"{identifier} defined at {locations}")
            )

    known_ids = set(by_id)
    for identifier in sorted(set(references) - known_ids, key=_id_sort_key):
        issues.append(
            SpecIssue("missing-reference", f"{identifier} has no registered row")
        )

    active_files = set(active_task_files)
    completed_files = set(completed_task_files)
    for record in records:
        if not record.identifier.startswith("T"):
            continue
        if record.file in active_files and record.status == "x":
            issues.append(
                SpecIssue(
                    "completed-task-in-active-file",
                    f"{record.identifier} completed but remains in {record.file}",
                )
            )
        if record.file in completed_files and record.status != "x":
            issues.append(
                SpecIssue(
                    "active-task-in-history-file",
                    f"{record.identifier} status={record.status!r} in {record.file}",
                )
            )

    expected_map = _expected_id_map(records)
    normalized_map = id_map_text.replace("\r\n", "\n")
    if normalized_map != expected_map:
        issues.append(
            SpecIssue(
                "id-map-drift",
                "docs/spec/id-map.tsv does not exactly match registered definitions",
            )
        )
    return SpecValidationReport(
        records=tuple(records),
        references=tuple(references),
        issues=tuple(issues),
    )


def validate_repository(repo_root: Path) -> SpecValidationReport:
    """Load and validate the repository SPEC registry."""

    root = Path(repo_root).resolve()
    registry_path = root / "docs" / "spec" / "registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("schema") != REGISTRY_SCHEMA:
        raise SpecIntegrityError(
            [
                SpecIssue(
                    "registry-schema",
                    f"{registry_path} schema must be {REGISTRY_SCHEMA!r}",
                )
            ]
        )
    try:
        registered_paths = _registry_paths(payload)
        id_map_path = _normalized_relative_path(str(payload["id_map"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SpecIntegrityError([SpecIssue("registry-shape", str(exc))]) from exc

    issues: list[SpecIssue] = []
    documents: dict[str, str] = {}
    for relative in registered_paths:
        path = root / relative
        if not path.is_file():
            issues.append(
                SpecIssue("missing-file", f"registered file missing: {relative}")
            )
            continue
        documents[relative] = path.read_text(encoding="utf-8")
    id_map_file = root / id_map_path
    if not id_map_file.is_file():
        issues.append(SpecIssue("missing-file", f"ID map missing: {id_map_path}"))
        id_map_text = ""
    else:
        id_map_text = id_map_file.read_text(encoding="utf-8")
    if issues:
        raise SpecIntegrityError(issues)

    report = validate_documents(
        documents,
        active_task_files=tuple(payload.get("active_task_files", [])),
        completed_task_files=tuple(payload.get("completed_task_files", [])),
        id_map_text=id_map_text,
    )
    if not report.ok:
        raise SpecIntegrityError(report.issues)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = validate_repository(args.repo_root)
    except (OSError, json.JSONDecodeError, SpecIntegrityError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        else:
            print(f"SPEC integrity failed:\n{exc}", file=sys.stderr)
        return 1
    counts = {
        kind: sum(record.identifier.startswith(kind) for record in report.records)
        for kind in "VTB"
    }
    payload = {"ok": True, "counts": counts, "references": len(report.references)}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "SPEC integrity OK: "
            + ", ".join(f"{kind}={count}" for kind, count in counts.items())
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
