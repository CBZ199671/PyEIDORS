#!/usr/bin/env python3
"""Hard-cut guard: block re-introducing legacy FEniCS naming/API tokens."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = [
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs",
]

ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".ini",
    ".txt",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
    ".mypy_cache",
    ".codex_logs",
}

EXCLUDED_PATHS = {
    REPO_ROOT / "docs" / "archive",
    REPO_ROOT / "docs" / "MIGRATION_PHASE2.md",
    REPO_ROOT / "scripts" / "ci" / "legacy_guard.py",
}

FORBIDDEN_PATTERNS = [
    re.compile(r"\bfenics_available\b"),
    re.compile(r"\bload_fenics_mesh\b"),
    re.compile(r"\battach_legacy_mesh_api\b"),
    re.compile(r"\bpatch_function_vector_api\b"),
    re.compile(r"\bStandardGaussNewtonReconstructor\b"),
    re.compile(r"\bdemo_fenics_"),
    re.compile(r"^\s*from\s+fenics\b", re.MULTILINE),
    re.compile(r"^\s*import\s+fenics\b", re.MULTILINE),
    re.compile(r"^\s*from\s+dolfin\b", re.MULTILINE),
    re.compile(r"^\s*import\s+dolfin\b", re.MULTILINE),
]


def _iter_files(path: Path):
    if not path.exists():
        return
    if path.is_file():
        if path.suffix.lower() in ALLOWED_SUFFIXES:
            yield path
        return

    for candidate in path.rglob("*"):
        if candidate.is_dir():
            continue
        if any(part in EXCLUDED_DIR_NAMES for part in candidate.parts):
            continue
        if any(str(candidate).startswith(str(excluded)) for excluded in EXCLUDED_PATHS):
            continue
        if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        yield candidate


def main() -> int:
    failures: list[tuple[Path, int, str]] = []

    for root in SCAN_ROOTS:
        for file_path in _iter_files(root):
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lines = text.splitlines()
            for pattern in FORBIDDEN_PATTERNS:
                for match in pattern.finditer(text):
                    line_no = text.count("\n", 0, match.start()) + 1
                    snippet = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else pattern.pattern
                    failures.append((file_path, line_no, snippet))

    if not failures:
        print("legacy guard passed: no forbidden legacy tokens found.")
        return 0

    print("legacy guard failed. Forbidden tokens were found:\n")
    for file_path, line_no, snippet in failures:
        rel = file_path.relative_to(REPO_ROOT)
        print(f"- {rel}:{line_no}: {snippet}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
