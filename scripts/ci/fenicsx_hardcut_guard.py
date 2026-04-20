#!/usr/bin/env python3
"""Hard-cut guard for the FEniCSx-only runtime."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = [
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "tests",
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs",
    REPO_ROOT / ".github",
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
    ".egg-info",
    "__pycache__",
    ".pytest_cache",
    "htmlcov",
    ".mypy_cache",
    ".codex_logs",
    ".pyeidors_cache",
    "build",
    "temp_abs_result",
}

EXCLUDED_PATHS = {
    REPO_ROOT / "docs" / "archive",
}


REMOVED_SCRIPT_PATTERNS = [
    re.compile(r"\brun_single_step_diff_realdata(?:_batch)?\.py\b"),
    re.compile(r"\brun_gn_absolute_eidors_style\.py\b"),
    re.compile(r"\brun_sparse_bayesian_reconstruction\.py\b"),
    re.compile(r"\brun_difference_single_step\.py\b"),
]

FORBIDDEN_PATTERNS = [
    re.compile(r"^\s*from\s+fenics\b", re.MULTILINE),
    re.compile(r"^\s*import\s+fenics\b", re.MULTILINE),
    re.compile(r"^\s*from\s+dolfin\b", re.MULTILINE),
    re.compile(r"^\s*import\s+dolfin\b", re.MULTILINE),
    re.compile(r"\bcuqi(?:py)?[-_]fenics\b", re.IGNORECASE),
    re.compile(r"\bfenics_available\b"),
    re.compile(r"\bload_fenics_mesh\b"),
    re.compile(r"\battach_" + "lega" + "cy_mesh_api\b"),
    re.compile(r"\bpatch_function_vector_api\b"),
    re.compile(r"\bStandardGaussNewtonReconstructor\b"),
    re.compile(r"\bdemo_fenics_"),
    *REMOVED_SCRIPT_PATTERNS,
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
        if any(
            part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info")
            for part in candidate.parts
        ):
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
                    snippet = (
                        lines[line_no - 1].strip()
                        if 0 < line_no <= len(lines)
                        else pattern.pattern
                    )
                    failures.append((file_path, line_no, snippet))

    if not failures:
        print("fenicsx hard-cut guard passed: no blocked tokens found.")
        return 0

    print("fenicsx hard-cut guard failed. Blocked tokens were found:\n")
    for file_path, line_no, snippet in failures:
        rel = file_path.relative_to(REPO_ROOT)
        print(f"- {rel}:{line_no}: {snippet}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
