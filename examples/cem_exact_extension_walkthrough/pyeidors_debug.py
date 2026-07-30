#!/usr/bin/env python3
"""逐步调试 PyEIDORS/DOLFINx 经典 CEM 与 Robin CEM 实验。

Step through the PyEIDORS/DOLFINx Classic and Robin CEM experiment.
本文件使用 ``# %%`` 单元，既可在 VS Code 中逐段运行，也可从终端运行。
It uses ``# %%`` cells for both VS Code interactive execution and terminal use.
"""

# %% 1. 导入与路径 / Imports and paths
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
for source_path in (REPO_ROOT, REPO_ROOT / "src", PACKAGE_DIR):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from experiment_common import (  # noqa: E402
    build_classic_state,
    build_robin_state,
    exact_reference_metrics,
    formulation_diagnostics,
    load_assembled_blocks,
    load_csv_records,
    load_forward_fixture,
    load_portable_exact_reference,
    plot_forward_fixture,
    plot_forward_solution,
    solve_classic,
    solve_robin,
    summarize_accuracy_records,
)


DEFAULT_SUITE_OUTPUT = REPO_ROOT / "output" / "cem_exact_extension"
DEFAULT_REFERENCE = PACKAGE_DIR / "fixtures" / "X01" / "exact_reference.json"
DEFAULT_METRICS = PACKAGE_DIR / "expected" / "cem_exact_extension_metrics.csv"


# %% 2. 定位或生成一个共享案例 / Locate or create one shared case
def _case_from_id(case_id: str):
    from scripts.benchmarks.cem_exact_extension_suite import EXTENSION_CASES

    try:
        return next(case for case in EXTENSION_CASES if case.case_id == case_id)
    except StopIteration as exc:
        raise ValueError(f"unknown case id: {case_id}") from exc


def _fixture_from_manifest(suite_output: Path, case_id: str) -> dict[str, Any] | None:
    manifest_path = suite_output / "suite_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for raw in manifest["cases"]:
        if raw["case_id"] != case_id:
            continue
        fixture = dict(raw)
        fixture["case_dir"] = Path(raw["case_dir"])
        fixture["mat_path"] = Path(raw["mat_path"])
        fixture["msh_path"] = Path(raw["msh_path"])
        fixture["metadata_path"] = Path(raw["metadata_path"])
        return fixture
    return None


def ensure_pyeidors_case(
    case_id: str,
    suite_output: Path,
    *,
    regenerate: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回一个案例的夹具与当前报告。 / Return its fixture and current report."""

    from scripts.benchmarks.cem_exact_extension_suite import (
        prepare_extension_case_fixture,
        run_pyeidors_extension_case,
    )

    fixture = _fixture_from_manifest(suite_output, case_id)
    if fixture is None or regenerate:
        fixture = prepare_extension_case_fixture(
            suite_output,
            _case_from_id(case_id),
        )
    report_path = Path(fixture["case_dir"]) / "pyeidors_report.json"
    if regenerate or not report_path.exists():
        report = run_pyeidors_extension_case(fixture)
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return fixture, report


# %% 3. 执行可检查的分块代数 / Run the inspectable block algebra
def run_selected_case(
    case_id: str = "X01",
    suite_output: Path = DEFAULT_SUITE_OUTPUT,
    *,
    regenerate: bool = False,
    debug_break: bool = False,
) -> dict[str, Any]:
    """保留全部关键中间量。 / Retain every key intermediate as a named object."""

    fixture, report = ensure_pyeidors_case(
        case_id,
        suite_output,
        regenerate=regenerate,
    )
    block_path = Path(fixture["case_dir"]) / "pyeidors_assembled_blocks.mat"
    blocks = load_assembled_blocks(block_path)
    forward_fixture = load_forward_fixture(
        Path(fixture["mat_path"]),
        Path(fixture["metadata_path"]),
    )
    assert (
        forward_fixture.mesh_fingerprint == report["discretization"]["mesh_fingerprint"]
    )

    # 在此设置断点检查 A_R、C、D、I。 / Inspect A_R, C, D, and I here.
    if debug_break:
        breakpoint()

    classic_state = build_classic_state(blocks)
    classic_solution = solve_classic(classic_state, blocks.currents)

    # 在 robin_state 中检查 Q、A_R^{-1}CQ、Schur 作用和 T_r。
    # Inspect Q, A_R^{-1}CQ, the Schur action, and T_r in robin_state.
    robin_state = build_robin_state(blocks)
    robin_solution = solve_robin(robin_state, blocks.currents)
    solutions = {
        "classic": classic_solution,
        "robin_transconductance": robin_solution,
    }
    diagnostics = formulation_diagnostics(blocks, solutions)

    report_voltages = {
        formulation: np.asarray(values, dtype=np.float64)
        for formulation, values in report["raw_electrode_voltages"].items()
    }
    explicit_vs_report = {
        formulation: float(
            np.linalg.norm(
                solutions[formulation].electrode_voltage - report_voltages[formulation]
            )
        )
        for formulation in report_voltages
    }
    if any(value != 0.0 for value in explicit_vs_report.values()):
        raise RuntimeError(
            "walkthrough algebra did not exactly reproduce the stored report voltage"
        )

    exact_metrics: dict[str, dict[str, float]] = {}
    if case_id == "X01" and DEFAULT_REFERENCE.exists():
        exact_reference = load_portable_exact_reference(DEFAULT_REFERENCE)
        for formulation, solution in solutions.items():
            exact_metrics[formulation] = exact_reference_metrics(
                solution.electrode_voltage,
                exact_reference,
            )
    else:
        exact_reference = None

    expected_summary = summarize_accuracy_records(load_csv_records(DEFAULT_METRICS))
    return {
        "fixture": fixture,
        "report": report,
        "block_path": block_path,
        "blocks": blocks,
        "forward_fixture": forward_fixture,
        "classic_state": classic_state,
        "classic_solution": classic_solution,
        "robin_state": robin_state,
        "robin_solution": robin_solution,
        "diagnostics": diagnostics,
        "exact_reference": exact_reference,
        "exact_metrics": exact_metrics,
        "explicit_vs_report_l2_absolute": explicit_vs_report,
        "expected_38_case_summary": expected_summary,
    }


# %% 4. 终端入口 / Terminal entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="X01")
    parser.add_argument(
        "--suite-output",
        type=Path,
        default=DEFAULT_SUITE_OUTPUT,
    )
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--show-forward-setup",
        action="store_true",
        help="显示共享网格、电导率和边界电流 / show mesh, conductivity, and drive",
    )
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="显示 Classic/Robin 体电势和电极电压 / show solved fields and voltages",
    )
    parser.add_argument("--debug-break", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = run_selected_case(
        args.case,
        args.suite_output.resolve(),
        regenerate=args.regenerate,
        debug_break=args.debug_break,
    )
    print(f"case={args.case} blocks={workspace['block_path']}")
    if args.show_forward_setup:
        import matplotlib.pyplot as plt

        plot_forward_fixture(workspace["forward_fixture"], current_column=0)
        if args.show_results:
            plot_forward_solution(
                workspace["forward_fixture"],
                {
                    "classic": workspace["classic_solution"],
                    "robin_transconductance": workspace["robin_solution"],
                },
                current_column=0,
            )
        plt.show()
    elif args.show_results:
        import matplotlib.pyplot as plt

        plot_forward_solution(
            workspace["forward_fixture"],
            {
                "classic": workspace["classic_solution"],
                "robin_transconductance": workspace["robin_solution"],
            },
            current_column=0,
        )
        plt.show()
    for name, value in workspace["diagnostics"].items():
        print(f"{name}: {value:.12e}")
    for formulation, metrics in workspace["exact_metrics"].items():
        print(
            f"{formulation}: truth_relative_l2="
            f"{metrics['truth_relative_l2']:.12e}, "
            "scaled_backward_residual="
            f"{metrics['exact_reduced_scaled_backward_residual']:.12e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
