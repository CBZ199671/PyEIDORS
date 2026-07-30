#!/usr/bin/env python3
"""验证数据与 Notebook。 / Validate data and notebook execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from experiment_common import (
    FORMULATIONS,
    load_csv_records,
    summarize_accuracy_records,
    summarize_timing_records,
)


PACKAGE_DIR = Path(__file__).resolve().parent
EXPECTED_DIR = PACKAGE_DIR / "expected"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_notebook(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("nbformat") != 4:
        raise ValueError(f"{path.name} is not nbformat v4")
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    if not code_cells:
        raise ValueError(f"{path.name} has no code cells")
    unexecuted = [
        index
        for index, cell in enumerate(code_cells, start=1)
        if cell.get("execution_count") is None
    ]
    errors = [
        index
        for index, cell in enumerate(code_cells, start=1)
        if any(output.get("output_type") == "error" for output in cell["outputs"])
    ]
    if unexecuted or errors:
        raise ValueError(f"{path.name}: unexecuted={unexecuted}, error_cells={errors}")
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "markdown"
    )
    for section in (
        "## 目标 / Goal",
        "## 设置 / Setup",
        "## 步骤 / Steps",
        "## 检查 / Checks",
        "## 后续步骤 / Next Steps",
    ):
        if section not in markdown:
            raise ValueError(f"{path.name} is missing {section}")
    for unsupported in (r"\[", r"\]", r"\(", r"\)"):
        if unsupported in markdown:
            raise ValueError(
                f"{path.name} uses unsupported math delimiter {unsupported}"
            )
    if markdown.count("$$") < 6 or markdown.count("$$") % 2:
        raise ValueError(f"{path.name} has incomplete block-math delimiters")
    if "| 中文 | English |" not in markdown:
        raise ValueError(f"{path.name} is missing parallel bilingual explanations")
    if "## 符号与变量字典 / Symbol and variable dictionary" not in markdown:
        raise ValueError(f"{path.name} is missing the variable dictionary")
    full_source = "\n".join("".join(cell["source"]) for cell in payload["cells"])
    if "plot_forward_fixture" not in full_source:
        raise ValueError(f"{path.name} does not display the shared forward setup")
    if "plot_forward_solution" not in full_source:
        raise ValueError(f"{path.name} does not display the solved forward result")
    if path.name == "exact_rational_truth_walkthrough.ipynb":
        for required in (
            "convert_to(QQ)",
            "lu_solve",
            "classic_residual_is_exact_zero",
            "robin_residual_is_exact_zero",
            "classic_robin_exactly_identical",
            "truth_sha256",
            "Fraction.from_float",
        ):
            if required not in full_source:
                raise ValueError(f"{path.name} is missing exact-truth step {required}")
    chinese = re.compile(r"[\u4e00-\u9fff]")
    for index, cell in enumerate(
        (cell for cell in payload["cells"] if cell["cell_type"] == "markdown"),
        start=1,
    ):
        if not chinese.search("".join(cell["source"])):
            raise ValueError(f"{path.name} markdown cell {index} is not bilingual")
    return {
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": len(code_cells),
        "error_cell_count": 0,
    }


def validate_package(
    metrics_path: Path,
    timing_path: Path,
) -> dict[str, Any]:
    metrics = load_csv_records(metrics_path)
    timing = load_csv_records(timing_path)
    case_ids = [f"X{index:02d}" for index in range(1, 39)]
    solvers = {"PyEIDORS/DOLFINx", "NGSolve", "EIDORS"}

    accuracy_keys = [
        (row["case_id"], row["solver"], row["formulation"]) for row in metrics
    ]
    timing_keys = [
        (row["case_id"], row["solver"], row["formulation"]) for row in timing
    ]
    if len(metrics) != 228 or len(set(accuracy_keys)) != 228:
        raise ValueError("accuracy evidence must contain 228 unique records")
    if len(timing) != 228 or len(set(timing_keys)) != 228:
        raise ValueError("timing evidence must contain 228 unique records")
    expected_keys = {
        (case_id, solver, formulation)
        for case_id in case_ids
        for solver in solvers
        for formulation in FORMULATIONS
    }
    if set(accuracy_keys) != expected_keys or set(timing_keys) != expected_keys:
        raise ValueError("solver/formulation/case coverage is incomplete")

    accuracy_fields = (
        "truth_relative_l2",
        "truth_max_abs",
        "exact_reduced_scaled_backward_residual",
        "voltage_gauge_relative_residual",
        "reduced_condition_number_2_estimate",
    )
    for row in metrics:
        for field in accuracy_fields:
            value = float(row[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field} in {accuracy_keys[0]}")
    timing_fields = (
        "cold_median_seconds",
        "setup_median_seconds",
        "warm_reuse_median_seconds",
        "cold_over_warm_reuse_speedup",
    )
    for row in timing:
        for field in timing_fields:
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {field} in {timing_keys[0]}")
        if float(row["cold_median_seconds"]) <= float(row["warm_reuse_median_seconds"]):
            raise ValueError(f"cold must exceed warm retained solve: {row}")

    exact_reference = json.loads(
        (PACKAGE_DIR / "fixtures" / "X01" / "exact_reference.json").read_text(
            encoding="utf-8"
        )
    )
    certification = exact_reference["certification"]
    if not all(bool(value) for value in certification.values()):
        raise ValueError("portable exact reference certification failed")

    notebooks = {
        filename: _validate_notebook(PACKAGE_DIR / filename)
        for filename in (
            "pyeidors_walkthrough.ipynb",
            "ngsolve_walkthrough.ipynb",
            "exact_rational_truth_walkthrough.ipynb",
        )
    }
    return {
        "status": "ready_to_share",
        "metrics_sha256": _sha256(metrics_path),
        "timing_sha256": _sha256(timing_path),
        "accuracy": summarize_accuracy_records(metrics),
        "timing": summarize_timing_records(timing),
        "exact_reference": {
            "case_id": exact_reference["case_id"],
            "truth_sha256": exact_reference["truth_sha256"],
            "certification": certification,
        },
        "notebooks": notebooks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=EXPECTED_DIR / "cem_exact_extension_metrics.csv",
    )
    parser.add_argument(
        "--timing",
        type=Path,
        default=EXPECTED_DIR / "cem_exact_extension_timing.csv",
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_package(args.metrics.resolve(), args.timing.resolve())
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        "ready_to_share: "
        f"accuracy={report['accuracy']['record_count']} "
        f"notebooks={len(report['notebooks'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
