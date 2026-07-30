from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from examples.cem_exact_extension_walkthrough.build_notebooks import (
    build_exact_truth_notebook,
    build_ngsolve_notebook,
    build_pyeidors_notebook,
)
from examples.cem_exact_extension_walkthrough.experiment_common import (
    CEMBlocks,
    formulation_diagnostics,
    load_forward_fixture,
    load_csv_records,
    plot_forward_solution,
    solve_both_formulations,
    summarize_accuracy_records,
)


PACKAGE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "cem_exact_extension_walkthrough"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_walkthrough_classic_and_robin_algebra_are_equivalent() -> None:
    robin_matrix = csc_matrix(
        np.asarray(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 3.5, -0.5],
                [0.0, -0.5, 2.5],
            ]
        )
    )
    coupling = csc_matrix(
        np.asarray(
            [
                [-0.30, -0.10, -0.20],
                [-0.15, -0.25, -0.10],
                [-0.05, -0.10, -0.25],
            ]
        )
    )
    electrode_matrix = csc_matrix(np.diag([1.5, 1.7, 1.6]))
    currents = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
        ]
    )
    blocks = CEMBlocks(
        robin_matrix=robin_matrix,
        coupling=coupling,
        electrode_matrix=electrode_matrix,
        currents=currents,
    )
    solutions = solve_both_formulations(blocks)
    diagnostics = formulation_diagnostics(blocks, solutions)

    assert diagnostics["electrode_voltage_relative_l2"] < 1e-14
    assert diagnostics["body_potential_relative_l2"] < 1e-14
    assert diagnostics["classic_scaled_backward_residual"] < 1e-14
    assert diagnostics["robin_scaled_backward_residual"] < 1e-14
    assert diagnostics["classic_voltage_gauge_max_abs"] < 1e-14
    assert diagnostics["robin_voltage_gauge_max_abs"] < 1e-14


def test_walkthrough_frozen_csv_reproduces_report_headlines() -> None:
    records = load_csv_records(
        PACKAGE_DIR / "expected" / "cem_exact_extension_metrics.csv"
    )
    summary = summarize_accuracy_records(records)

    assert summary["record_count"] == 228
    assert summary["case_count"] == 38
    assert summary["win_counts"]["classic"] == {
        "EIDORS": 11,
        "NGSolve": 0,
        "PyEIDORS/DOLFINx": 27,
    }
    assert summary["win_counts"]["robin_transconductance"] == {
        "EIDORS": 9,
        "NGSolve": 0,
        "PyEIDORS/DOLFINx": 29,
    }
    assert summary["geometric_means"]["classic"]["PyEIDORS/DOLFINx"] == (
        pytest.approx(1.109e-15, rel=5e-4)
    )
    assert summary["geometric_means"]["classic"]["EIDORS"] == pytest.approx(
        1.694e-15,
        rel=5e-4,
    )
    assert summary["geometric_means"]["classic"]["NGSolve"] == pytest.approx(
        1.120e-14,
        rel=5e-4,
    )
    for formulation in ("classic", "robin_transconductance"):
        q4 = summary["q4_summary"][formulation]
        assert q4["same_order_all_cases"] is True
        assert q4["ordering"] == [
            "PyEIDORS/DOLFINx",
            "EIDORS",
            "NGSolve",
        ]


def test_walkthrough_notebooks_have_teaching_sections() -> None:
    required = (
        "## 目标 / Goal",
        "## 设置 / Setup",
        "## 步骤 / Steps",
        "## 检查 / Checks",
        "## 后续步骤 / Next Steps",
    )
    for filename in ("pyeidors_walkthrough.ipynb", "ngsolve_walkthrough.ipynb"):
        payload = json.loads((PACKAGE_DIR / filename).read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell["source"])
            for cell in payload["cells"]
            if cell["cell_type"] == "markdown"
        )
        for heading in required:
            assert heading in markdown


def test_v766_real_float64_profile_provides_jupyter_kernel() -> None:
    flake = (REPOSITORY_ROOT / "flake.nix").read_text(encoding="utf-8")
    readme = (PACKAGE_DIR / "README.md").read_text(encoding="utf-8")
    launcher = PACKAGE_DIR / "launch_pyeidors_float64_kernel.sh"
    registration = PACKAGE_DIR / "register_vscode_kernel.py"

    default_shell = flake.split("default = pkgs.mkShell", maxsplit=1)[1].split(
        "complex = pkgs.mkShell",
        maxsplit=1,
    )[0]
    assert "py.ipykernel" in default_shell
    assert "py.jupyterlab" in default_shell
    assert "PyEIDORS real float64 (Nix)" in readme
    assert launcher.is_file()
    assert registration.is_file()


@pytest.mark.parametrize(
    "builder",
    (
        build_pyeidors_notebook,
        build_ngsolve_notebook,
        build_exact_truth_notebook,
    ),
)
def test_v767_walkthrough_math_and_explanations_are_bilingual(builder) -> None:
    notebook = builder()
    markdown_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    markdown = "\n".join(markdown_cells)

    assert r"\[" not in markdown
    assert r"\]" not in markdown
    assert r"\(" not in markdown
    assert r"\)" not in markdown
    assert markdown.count("$$") >= 6
    assert markdown.count("$$") % 2 == 0
    assert "| 中文 | English |" in markdown
    assert "## 目标 / Goal" in markdown
    assert "## 设置 / Setup" in markdown
    assert "## 步骤 / Steps" in markdown
    assert "## 检查 / Checks" in markdown
    assert "## 后续步骤 / Next Steps" in markdown

    chinese = re.compile(r"[\u4e00-\u9fff]")
    for cell in markdown_cells:
        assert chinese.search(cell), cell

    math_spans = [
        match.span() for match in re.finditer(r"\$\$.*?\$\$", markdown, flags=re.DOTALL)
    ]
    for match in re.finditer(r"\\begin\{", markdown):
        assert any(start <= match.start() < stop for start, stop in math_spans)


def test_v768_walkthroughs_define_variables_and_show_fair_forward_setup() -> None:
    required_shared_terms = (
        "## 符号与变量字典 / Symbol and variable dictionary",
        "N×N",
        "网格指纹",
        "Mesh fingerprint",
        "单元电导率",
        "Cell conductivity",
        "注入电流",
        "Injected current",
        "plot_forward_fixture",
    )
    for builder in (build_pyeidors_notebook, build_ngsolve_notebook):
        notebook = builder()
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        for term in required_shared_terms:
            assert term in source


def test_v768_portable_fixture_contains_the_fair_forward_problem() -> None:
    fixture_dir = PACKAGE_DIR / "fixtures" / "X01" / "common_mesh"
    fixture = load_forward_fixture(
        fixture_dir / "cem_exact_extension_p1.mat",
        fixture_dir / "cem_exact_extension_p1.json",
    )

    assert fixture.case_id == "X01"
    assert fixture.nodes.shape == (33, 2)
    assert fixture.cells.shape == (32, 3)
    assert fixture.tagged_edges.shape == (32, 3)
    assert fixture.cell_conductivity.shape == (32,)
    assert fixture.currents.shape == (16, 16)
    assert fixture.electrode_count == 16
    assert fixture.potential_order == 1
    assert fixture.scalar_dtype == "float64"
    assert fixture.conductivity_pattern == "uniform"
    assert np.all(fixture.cell_conductivity == 0.125)
    assert np.all(np.sum(fixture.currents, axis=0) == 0.0)
    assert (
        fixture.mesh_fingerprint
        == "7be7165ad3bdd3661ae06bea768622741ece609acde5720f3dd0c0cbde85c5bc"
    )


def test_v768_exact_truth_notebook_exposes_qq_solve_and_certification() -> None:
    notebook = build_exact_truth_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    for term in (
        "## 真值的适用范围 / Scope of the truth",
        "不是连续 PDE 的解析真值",
        "not the analytic truth of the continuum PDE",
        "## 符号与变量字典 / Symbol and variable dictionary",
        "Fraction",
        "DomainMatrix",
        "convert_to(QQ)",
        "lu_solve",
        "classic_residual_is_exact_zero",
        "robin_residual_is_exact_zero",
        "classic_robin_exactly_identical",
        "truth_sha256",
        "Fraction.from_float",
        "exact_reference_metrics",
    ):
        assert term in source


def test_v769_walkthroughs_visualize_classic_robin_forward_results() -> None:
    required_terms = (
        "求解结果可视化 / Forward-result visualization",
        "plot_forward_solution",
        "Classic 体电势",
        "Robin 体电势",
        "体电势差值",
        "电极电压",
    )
    for builder in (build_pyeidors_notebook, build_ngsolve_notebook):
        source = "\n".join("".join(cell["source"]) for cell in builder()["cells"])
        for term in required_terms:
            assert term in source

    matlab = (PACKAGE_DIR / "eidors_selected_case_walkthrough.m").read_text(
        encoding="utf-8"
    )
    assert "plot_forward_solution_matlab" in matlab
    assert "Classic/Robin 体电势与边界电压" in matlab


def test_v769_forward_result_figure_has_six_comparison_panels() -> None:
    fixture_dir = PACKAGE_DIR / "fixtures" / "X01" / "common_mesh"
    fixture = load_forward_fixture(
        fixture_dir / "cem_exact_extension_p1.mat",
        fixture_dir / "cem_exact_extension_p1.json",
    )
    size = fixture.nodes.shape[0]
    electrodes = fixture.electrode_count
    patterns = fixture.currents.shape[1]
    body = np.repeat(
        np.linspace(-1.0, 1.0, size, dtype=np.float64).reshape(-1, 1),
        patterns,
        axis=1,
    )
    voltage = np.repeat(
        np.linspace(-0.5, 0.5, electrodes, dtype=np.float64).reshape(-1, 1),
        patterns,
        axis=1,
    )
    solutions = {
        "classic": type(
            "VisibleSolution",
            (),
            {"body_potential": body, "electrode_voltage": voltage},
        )(),
        "robin_transconductance": type(
            "VisibleSolution",
            (),
            {
                "body_potential": body + 1e-12,
                "electrode_voltage": voltage - 1e-12,
            },
        )(),
    }

    figure, axes = plot_forward_solution(
        fixture,
        solutions,
        current_column=0,
    )

    assert axes.shape == (2, 3)
    assert "Classic" in axes[0, 0].get_title()
    assert "Robin" in axes[0, 1].get_title()
    assert "差值" in axes[0, 2].get_title()
    assert len(axes[1, 0].lines) == 2
    figure.clear()


def test_v767_all_walkthrough_scripts_have_chinese_guidance() -> None:
    chinese = re.compile(r"[\u4e00-\u9fff]")
    scripts = sorted(
        (
            *PACKAGE_DIR.glob("*.py"),
            *PACKAGE_DIR.glob("*.m"),
            *PACKAGE_DIR.glob("*.sh"),
        ),
        key=lambda path: path.name,
    )
    assert scripts
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert chinese.search(text), path.name
