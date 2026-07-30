#!/usr/bin/env python3
"""无额外 Jupyter 依赖地从头执行 Notebook。 / Execute a notebook top-to-bottom.

中文：这个有界执行器运行两个演示 Notebook 的 Python 单元，记录文本与
Matplotlib PNG 输出，并写回标准 nbformat-v4 文件。
English: This bounded executor runs the walkthrough Python cells, records text
and Matplotlib PNG outputs, and writes a normal nbformat-v4 notebook.
"""

from __future__ import annotations

import argparse
import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import traceback
from typing import Any


def _execute_code_cell(
    source: str,
    namespace: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    outputs: list[dict[str, Any]] = []
    stdout = StringIO()
    stderr = StringIO()
    result: Any = None
    failed = False
    try:
        syntax_tree = ast.parse(source, mode="exec")
        body = list(syntax_tree.body)
        expression = (
            body.pop().value if body and isinstance(body[-1], ast.Expr) else None
        )
        with redirect_stdout(stdout), redirect_stderr(stderr):
            if body:
                module = ast.Module(body=body, type_ignores=[])
                ast.fix_missing_locations(module)
                exec(compile(module, "<notebook-cell>", "exec"), namespace)
            if expression is not None:
                expression_tree = ast.Expression(body=expression)
                ast.fix_missing_locations(expression_tree)
                result = eval(
                    compile(expression_tree, "<notebook-cell>", "eval"),
                    namespace,
                )
    except Exception as exc:
        failed = True
        outputs.append(
            {
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "output_type": "error",
                "traceback": traceback.format_exc().splitlines(),
            }
        )

    if stdout.getvalue():
        outputs.insert(
            0,
            {
                "name": "stdout",
                "output_type": "stream",
                "text": stdout.getvalue().splitlines(keepends=True),
            },
        )
    if stderr.getvalue():
        outputs.insert(
            0,
            {
                "name": "stderr",
                "output_type": "stream",
                "text": stderr.getvalue().splitlines(keepends=True),
            },
        )
    if not failed and result is not None:
        outputs.append(
            {
                "data": {"text/plain": repr(result).splitlines(keepends=True)},
                "execution_count": None,
                "metadata": {},
                "output_type": "execute_result",
            }
        )

    pyplot = namespace.get("plt")
    if not failed and pyplot is not None:
        for figure_number in list(pyplot.get_fignums()):
            figure = pyplot.figure(figure_number)
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            outputs.append(
                {
                    "data": {
                        "image/png": base64.b64encode(buffer.getvalue()).decode("ascii")
                    },
                    "metadata": {},
                    "output_type": "display_data",
                }
            )
            pyplot.close(figure)
    return outputs, failed


def execute_notebook(path: Path, output_path: Path | None = None) -> Path:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__main__"}
    execution_count = 0
    old_cwd = Path.cwd()
    os.chdir(path.parent)
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            source = "".join(cell["source"])
            outputs, failed = _execute_code_cell(source, namespace)
            cell["execution_count"] = execution_count
            for output in outputs:
                if output.get("output_type") == "execute_result":
                    output["execution_count"] = execution_count
            cell["outputs"] = outputs
            if failed:
                raise RuntimeError(
                    f"notebook execution failed in code cell {execution_count}"
                )
    finally:
        os.chdir(old_cwd)

    destination = output_path or path
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = execute_notebook(
        args.notebook.resolve(),
        None if args.output is None else args.output.resolve(),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
