"""Patch PyMFEM sources for the pinned external-prefix CUDA workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _replace_once(path: Path, old: str, new: str, needle: str) -> None:
    text = path.read_text(encoding="utf-8")
    if needle in text:
        return
    if old not in text:
        raise SystemExit(f"expected pattern not found in {path}: {needle}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def _ensure_trailing_assignment(path: Path, assignment: str) -> None:
    text = path.read_text(encoding="utf-8")
    if assignment in text:
        return
    suffix = "" if text.endswith("\n") else "\n"
    path.write_text(f"{text}{suffix}{assignment}\n", encoding="utf-8")


def patch_source_tree(source_root: Path) -> None:
    build_globals = source_root / "_build_system" / "build_globals.py"
    build_pymfem = source_root / "_build_system" / "build_pymfem.py"
    build_config = source_root / "_build_system" / "build_config.py"

    _ensure_trailing_assignment(build_globals, "build_py_done = False")

    _replace_once(
        build_globals,
        "metis_prefix = ''\nhypre_prefix = ''",
        "metis_prefix = '' if os.getenv('PYEIDORS_METIS_PREFIX') is None else os.getenv('PYEIDORS_METIS_PREFIX')\n"
        "hypre_prefix = '' if os.getenv('PYEIDORS_HYPRE_PREFIX') is None else os.getenv('PYEIDORS_HYPRE_PREFIX')",
        "PYEIDORS_HYPRE_PREFIX",
    )

    _replace_once(
        build_pymfem,
        "    command = [python, 'setup.py', 'build_ext', '--inplace', '--parallel',\n"
        "               str(max((cpu_count() - 1, 1)))]",
        "    parallel_jobs = int(os.getenv('PYEIDORS_PYMFEM_BUILD_JOBS', str(max((cpu_count() - 1, 1)))))\n"
        "    command = [python, 'setup.py', 'build_ext', '--inplace', '--parallel', str(parallel_jobs)]",
        "PYEIDORS_PYMFEM_BUILD_JOBS",
    )

    _replace_once(
        build_config,
        "    if self.hypre_prefix != '':\n"
        "        check = find_libpath_from_prefix('HYPRE', self.hypre_prefix)\n"
        "        assert check != '', \"libHYPRE.so is not found in the specified <path>/lib or lib64\"\n"
        "        hypre_prefix = os.path.expanduser(self.hypre_prefix)\n"
        "        build_hypre = False\n",
        "    if self.hypre_prefix != '':\n"
        "        check = find_libpath_from_prefix('HYPRE', self.hypre_prefix)\n"
        "        assert check != '', \"libHYPRE.so is not found in the specified <path>/lib or lib64\"\n"
        "        bglb.hypre_prefix = os.path.expanduser(self.hypre_prefix)\n"
        "        bglb.build_hypre = False\n",
        "bglb.hypre_prefix = os.path.expanduser(self.hypre_prefix)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path, help="Writable PyMFEM source tree")
    args = parser.parse_args()
    patch_source_tree(args.source_root.resolve())


if __name__ == "__main__":
    main()
