#!/usr/bin/env python3
"""Inventory and guard binary persistence formats.

The guard freezes known legacy NumPy archive writers while the project migrates
large binary artifacts to HDF5. New production ``.npz`` / ``.npy`` writers must
be rejected unless they are explicitly marked as legacy/test-only.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


DEFAULT_SCAN_ROOTS = ("src", "scripts", "tests")
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-cuda",
    "__pycache__",
    ".pyeidors_cache",
    "build",
    "dist",
    "htmlcov",
    "reports",
    "results",
}

TRACKED_SUFFIXES = (".npz", ".npy", ".msh", ".xdmf", ".h5", ".mat")
NUMPY_WRITER_APIS = {
    "np.save",
    "np.savez",
    "np.savez_compressed",
    "numpy.save",
    "numpy.savez",
    "numpy.savez_compressed",
}
NUMPY_READER_APIS = {"np.load", "numpy.load"}
MESH_WRITER_APIS = {"gmsh.write", "meshio.write"}
HDF5_APIS = {"h5py.File", "XDMFFile", "dolfinx.io.XDMFFile"}

NUMPY_WRITER_NAMES = {"save", "savez", "savez_compressed"}
NUMPY_READER_NAMES = {"load"}

# Frozen snapshot of production NumPy writers that existed before the HDF5
# migration. T51 forbids new production writers while T53..T56 migrate these.
LEGACY_NUMPY_WRITER_ALLOWLIST = {
    "scripts/mesh_tools/convert_matlab_mesh.py|main|np.savez|args.out_dir / 'mesh.npz'",
}


@dataclass(frozen=True)
class PersistenceFinding:
    path: str
    line: int
    qname: str
    kind: str
    api: str
    target: str
    suffixes: tuple[str, ...]
    classification: str
    legacy_id: str


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_test_path(path: str) -> bool:
    return path.startswith("tests/") or "/tests/" in path


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _safe_unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _joined_string_text(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            parts.append("{" + _safe_unparse(value.value) + "}")
    return "f'" + "".join(parts) + "'"


def _string_literals(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            values.append(child.value)
        elif isinstance(child, ast.JoinedStr):
            values.append(_joined_string_text(child))
    return values


def _suffixes_from_strings(values: Iterable[str]) -> tuple[str, ...]:
    suffixes: set[str] = set()
    for value in values:
        lower = value.lower()
        for suffix in TRACKED_SUFFIXES:
            if suffix in lower:
                suffixes.add(suffix)
    return tuple(sorted(suffixes))


def _finding_kind(api: str, suffixes: tuple[str, ...]) -> str:
    if api in NUMPY_WRITER_APIS:
        return "numpy_writer"
    if api in NUMPY_READER_APIS:
        return "numpy_reader"
    if api in MESH_WRITER_APIS:
        return "mesh_source_writer"
    if api in HDF5_APIS:
        return "hdf5_io"
    if suffixes:
        return "suffix_literal"
    return "other"


def _classify(path: str, kind: str, legacy_id: str) -> str:
    if _is_test_path(path):
        return "test-only"
    if kind == "numpy_writer" and legacy_id in LEGACY_NUMPY_WRITER_ALLOWLIST:
        return "legacy-production"
    return "production"


class PersistenceVisitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.stack: list[str] = []
        self.findings: list[PersistenceFinding] = []
        self.numpy_modules: set[str] = {"np", "numpy"}
        self.numpy_functions: dict[str, str] = {}
        self.gmsh_modules: set[str] = {"gmsh"}
        self.meshio_modules: set[str] = {"meshio"}
        self.h5py_modules: set[str] = {"h5py"}
        self.xdmf_file_names: set[str] = {"XDMFFile"}

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            local = alias.asname or alias.name
            if alias.name == "numpy":
                self.numpy_modules.add(local)
            elif alias.name == "gmsh":
                self.gmsh_modules.add(local)
            elif alias.name == "meshio":
                self.meshio_modules.add(local)
            elif alias.name == "h5py":
                self.h5py_modules.add(local)
            elif alias.name == "dolfinx.io.XDMFFile":
                self.xdmf_file_names.add(local)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if module == "numpy":
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name in NUMPY_WRITER_NAMES | NUMPY_READER_NAMES:
                    self.numpy_functions[local] = f"np.{alias.name}"
        elif module == "dolfinx.io":
            for alias in node.names:
                if alias.name == "XDMFFile":
                    self.xdmf_file_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        api = self._normalized_api(_call_name(node.func))
        target_node = node.args[0] if node.args else None
        target = _safe_unparse(target_node)
        suffixes = _suffixes_from_strings(_string_literals(target_node))
        if api in NUMPY_WRITER_APIS | NUMPY_READER_APIS | MESH_WRITER_APIS | HDF5_APIS:
            self._add(node, api=api, target=target, suffixes=suffixes)
        self.generic_visit(node)

    def _normalized_api(self, api: str) -> str:
        if api in self.numpy_functions:
            return self.numpy_functions[api]
        if api in self.xdmf_file_names:
            return "XDMFFile"
        if "." not in api:
            return api
        base, _, attr = api.rpartition(".")
        if base in self.numpy_modules and attr in NUMPY_WRITER_NAMES | NUMPY_READER_NAMES:
            return f"np.{attr}"
        if base in self.gmsh_modules and attr == "write":
            return "gmsh.write"
        if base in self.meshio_modules and attr == "write":
            return "meshio.write"
        if base in self.h5py_modules and attr == "File":
            return "h5py.File"
        return api

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str):
            suffixes = _suffixes_from_strings((node.value,))
            if suffixes:
                self._add(node, api="literal", target=repr(node.value), suffixes=suffixes)

    def _add(
        self,
        node: ast.AST,
        *,
        api: str,
        target: str,
        suffixes: tuple[str, ...],
    ) -> None:
        qname = ".".join(self.stack) if self.stack else "<module>"
        kind = _finding_kind(api, suffixes)
        legacy_id = f"{self.path}|{qname}|{api}|{target}"
        classification = _classify(self.path, kind, legacy_id)
        self.findings.append(
            PersistenceFinding(
                path=self.path,
                line=int(getattr(node, "lineno", 0) or 0),
                qname=qname,
                kind=kind,
                api=api,
                target=target,
                suffixes=suffixes,
                classification=classification,
                legacy_id=legacy_id,
            )
        )


def iter_python_files(root: Path, scan_roots: Iterable[str]) -> Iterable[Path]:
    for rel in scan_roots:
        base = root / rel
        if not base.exists():
            continue
        if base.is_file() and base.suffix == ".py":
            yield base
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            yield path


def scan_repo(root: Path, scan_roots: Iterable[str] = DEFAULT_SCAN_ROOTS) -> list[PersistenceFinding]:
    findings: list[PersistenceFinding] = []
    for path in sorted(iter_python_files(root, scan_roots)):
        rel = _relpath(path, root)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError:
            continue
        visitor = PersistenceVisitor(rel)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return sorted(findings, key=lambda item: (item.path, item.line, item.kind, item.api))


def guard_violations(findings: Iterable[PersistenceFinding]) -> list[PersistenceFinding]:
    violations: list[PersistenceFinding] = []
    for finding in findings:
        if finding.kind != "numpy_writer":
            continue
        if finding.classification != "production":
            continue
        violations.append(finding)
    return violations


def summary(findings: Iterable[PersistenceFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        key = f"{finding.classification}:{finding.kind}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scan-root",
        action="append",
        dest="scan_roots",
        help="Relative path to scan. Defaults to src, scripts, tests.",
    )
    parser.add_argument("--inventory-json", type=Path)
    parser.add_argument("--print-legacy-writer-ids", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args(argv)

    scan_roots = tuple(args.scan_roots or DEFAULT_SCAN_ROOTS)
    findings = scan_repo(args.root, scan_roots)
    violations = guard_violations(findings)

    if args.inventory_json:
        _write_json(
            args.inventory_json,
            {
                "schema": "pyeidors-persistence-format-inventory-v1",
                "summary": summary(findings),
                "findings": [asdict(finding) for finding in findings],
                "violations": [asdict(finding) for finding in violations],
            },
        )

    if args.print_legacy_writer_ids:
        for finding in findings:
            if finding.kind == "numpy_writer" and finding.classification != "test-only":
                print(finding.legacy_id)

    print(json.dumps({"summary": summary(findings), "violations": len(violations)}, sort_keys=True))
    if violations and not args.no_fail:
        for finding in violations:
            print(
                f"{finding.path}:{finding.line}: new production NumPy writer: "
                f"{finding.api}({finding.target})",
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
