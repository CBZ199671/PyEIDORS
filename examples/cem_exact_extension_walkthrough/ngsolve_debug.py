#!/usr/bin/env python3
"""逐步调试 NGSolve 经典 CEM 与 Robin CEM 实验。

Step through the NGSolve Classic and Robin CEM experiment.
请在 README.md 所述的隔离 NGSolve 环境中运行；``# %%`` 可供 VS Code 逐段执行。
Run it in the isolated NGSolve environment; ``# %%`` enables VS Code cells.
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
for source_path in (REPO_ROOT, PACKAGE_DIR):
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


# %% 2. 定位共享 MSH/JSON 夹具 / Locate the common MSH/JSON fixture
def resolve_fixture(
    case_id: str,
    suite_output: Path,
) -> tuple[Path, Path, Path]:
    manifest_path = suite_output / "suite_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for fixture in manifest["cases"]:
            if fixture["case_id"] == case_id:
                case_dir = Path(fixture["case_dir"])
                return (
                    Path(fixture["msh_path"]),
                    Path(fixture["metadata_path"]),
                    case_dir,
                )
    if case_id == "X01":
        common_dir = PACKAGE_DIR / "fixtures" / "X01" / "common_mesh"
        output_dir = (
            REPO_ROOT / "output" / "cem_professor_walkthrough" / "ngsolve" / case_id
        )
        return (
            common_dir / "cem_exact_extension_p1.msh",
            common_dir / "cem_exact_extension_p1.json",
            output_dir,
        )
    raise FileNotFoundError(
        f"{case_id} fixture not found. Run the PyEIDORS prepare step first."
    )


# %% 3. 用 NGSolve 组装并暴露同一分块代数 / Assemble and expose the same blocks
def run_selected_case(
    case_id: str = "X01",
    suite_output: Path = DEFAULT_SUITE_OUTPUT,
    *,
    timing_repeats: int = 11,
    regenerate: bool = False,
    debug_break: bool = False,
) -> dict[str, Any]:
    from scripts.benchmarks.ngsolve_cem_exact_extension_case import run_case

    mesh_path, metadata_path, case_dir = resolve_fixture(case_id, suite_output)
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "ngsolve_report.json"
    block_path = case_dir / "ngsolve_assembled_blocks.mat"
    if regenerate or not report_path.exists() or not block_path.exists():
        report = run_case(
            mesh_path,
            metadata_path,
            report_path,
            timing_repeats=timing_repeats,
        )
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))

    blocks = load_assembled_blocks(block_path)
    forward_fixture = load_forward_fixture(
        metadata_path.with_suffix(".mat"),
        metadata_path,
    )
    assert (
        forward_fixture.mesh_fingerprint == report["discretization"]["mesh_fingerprint"]
    )
    # 在此设置断点检查 NGSolve 的 A_R、C、D、I。 / Inspect the blocks here.
    if debug_break:
        breakpoint()

    classic_state = build_classic_state(blocks)
    classic_solution = solve_classic(classic_state, blocks.currents)
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

    exact_reference = load_portable_exact_reference(DEFAULT_REFERENCE)
    exact_metrics = {
        formulation: exact_reference_metrics(
            solution.electrode_voltage,
            exact_reference,
        )
        for formulation, solution in solutions.items()
    }
    expected_summary = summarize_accuracy_records(load_csv_records(DEFAULT_METRICS))
    return {
        "mesh_path": mesh_path,
        "metadata_path": metadata_path,
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


# %% 4. 可选的完整 38 案例运行器 / Optional complete 38-case runner
def run_all_cases(suite_output: Path, *, timing_repeats: int = 11) -> int:
    from scripts.benchmarks.ngsolve_cem_exact_extension_suite import run_suite

    return run_suite(suite_output.resolve(), timing_repeats=timing_repeats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="X01")
    parser.add_argument(
        "--suite-output",
        type=Path,
        default=DEFAULT_SUITE_OUTPUT,
    )
    parser.add_argument("--timing-repeats", type=int, default=11)
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
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--debug-break", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.all:
        count = run_all_cases(
            args.suite_output,
            timing_repeats=args.timing_repeats,
        )
        print(f"NGSolve extension reports: {count} cases")
        return 0
    workspace = run_selected_case(
        args.case,
        args.suite_output.resolve(),
        timing_repeats=args.timing_repeats,
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
