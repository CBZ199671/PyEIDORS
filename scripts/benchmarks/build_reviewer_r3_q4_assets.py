#!/usr/bin/env python3
"""Build Reviewer 3 Question 4 GPU runtime benchmark assets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from benchmark_reviewer_case import GPU_SCOPE_NOTE, get_git_commit  # noqa: E402


TASK = "absolute_gn"
FRAMEWORK = "pyeidors"
SCENARIO = "low_z"
GN_PATH = "legacy_dense"
SUMMARY_MESH_LEVELS = ["medium", "fine"]
SWEEP_MESH_LEVEL = "medium"
SUMMARY_ITERATIONS = 5
SWEEP_ITERATIONS = [1, 3, 5]
WARMUPS = 1
REPEATS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "benchmarks" / "reviewer_r3_q4",
        help="Directory for Reviewer 3 Question 4 benchmark assets.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=REPO_ROOT / "eit_meshes",
        help="Directory containing cached meshes.",
    )
    parser.add_argument(
        "--container-name",
        default="pyeidors",
        help="Container name recorded in benchmark_environment.json.",
    )
    parser.add_argument(
        "--image-name",
        default="pyeidors:latest",
        help="Container/image label recorded in benchmark_environment.json.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    result = subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_absolute_gn_case(
    *,
    output_path: Path,
    mesh_dir: Path,
    mesh_level: str,
    device: str,
    iterations: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "benchmark_reviewer_case.py"),
        "--framework",
        FRAMEWORK,
        "--task",
        TASK,
        "--mesh-level",
        mesh_level,
        "--scenario",
        SCENARIO,
        "--device",
        device,
        "--warmups",
        str(WARMUPS),
        "--repeats",
        str(REPEATS),
        "--n-frames",
        "1",
        "--mesh-dir",
        str(mesh_dir),
        "--absolute-lambda",
        "1e-2",
        "--absolute-max-iter",
        str(iterations),
        "--gn-path",
        GN_PATH,
        "--output-json",
        str(output_path),
    ]
    run_command(command, REPO_ROOT)
    return json.loads(output_path.read_text(encoding="utf-8"))


def summarize_cpu_gpu_pair(cpu_row: dict[str, Any], gpu_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": cpu_row["task"],
        "mesh_level": cpu_row["mesh_level"],
        "mesh_name": cpu_row["mesh_name"],
        "scenario": cpu_row["scenario"],
        "iterations": cpu_row["iterations"],
        "nodes": cpu_row["n_nodes"],
        "elements": cpu_row["n_elements"],
        "cpu_median_time_sec": cpu_row["median"],
        "gpu_median_time_sec": gpu_row["median"],
        "speedup_gpu_vs_cpu": cpu_row["median"] / gpu_row["median"],
        "cpu_peak_rss_mb": cpu_row["peak_rss_mb"],
        "gpu_peak_rss_mb": gpu_row["peak_rss_mb"],
        "cpu_voltage_rmse": cpu_row["voltage_rmse"],
        "gpu_voltage_rmse": gpu_row["voltage_rmse"],
        "voltage_rmse_abs_diff": abs(cpu_row["voltage_rmse"] - gpu_row["voltage_rmse"]),
        "cpu_final_residual": cpu_row["final_residual"],
        "gpu_final_residual": gpu_row["final_residual"],
        "final_residual_abs_diff": abs(cpu_row["final_residual"] - gpu_row["final_residual"]),
        "cpu_linear_solver": cpu_row.get("linear_solver", ""),
        "gpu_linear_solver": gpu_row.get("linear_solver", ""),
        "cpu_regularization_structure": cpu_row.get("regularization_structure", ""),
        "gpu_regularization_structure": gpu_row.get("regularization_structure", ""),
        "gpu_scope_note": gpu_row.get("gpu_scope_note", GPU_SCOPE_NOTE),
    }


def build_summary_benchmark(output_dir: Path, mesh_dir: Path) -> dict[str, Any]:
    raw_dir = output_dir / "raw" / "summary"
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for mesh_level in SUMMARY_MESH_LEVELS:
        case_rows: dict[str, dict[str, Any]] = {}
        for device in ("cpu", "gpu"):
            output_path = raw_dir / (
                f"{FRAMEWORK}_{TASK}_{mesh_level}_{SCENARIO}_{device}_iter{SUMMARY_ITERATIONS}_{GN_PATH}.json"
            )
            row = run_absolute_gn_case(
                output_path=output_path,
                mesh_dir=mesh_dir,
                mesh_level=mesh_level,
                device=device,
                iterations=SUMMARY_ITERATIONS,
            )
            case_rows[device] = row
            raw_rows.append(row)
        summary_rows.append(summarize_cpu_gpu_pair(case_rows["cpu"], case_rows["gpu"]))

    raw_json_path = output_dir / "raw" / "summary_raw.json"
    summary_csv_path = output_dir / "table_r3_q4_gpu_runtime_summary.csv"
    raw_json_path.write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")
    write_csv(summary_csv_path, list(summary_rows[0].keys()), summary_rows)
    return {
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
        "raw_json": raw_json_path,
        "summary_csv": summary_csv_path,
    }


def build_iteration_sweep(output_dir: Path, mesh_dir: Path) -> dict[str, Any]:
    raw_dir = output_dir / "raw" / "iteration_sweep"
    raw_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []

    for iterations in SWEEP_ITERATIONS:
        case_rows: dict[str, dict[str, Any]] = {}
        for device in ("cpu", "gpu"):
            output_path = raw_dir / (
                f"{FRAMEWORK}_{TASK}_{SWEEP_MESH_LEVEL}_{SCENARIO}_{device}_iter{iterations}_{GN_PATH}.json"
            )
            row = run_absolute_gn_case(
                output_path=output_path,
                mesh_dir=mesh_dir,
                mesh_level=SWEEP_MESH_LEVEL,
                device=device,
                iterations=iterations,
            )
            case_rows[device] = row
            raw_rows.append(row)
        sweep_rows.append(summarize_cpu_gpu_pair(case_rows["cpu"], case_rows["gpu"]))

    raw_json_path = output_dir / "raw" / "iteration_sweep_raw.json"
    sweep_csv_path = output_dir / "table_r3_q4_gpu_runtime_iteration_sweep.csv"
    raw_json_path.write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")
    write_csv(sweep_csv_path, list(sweep_rows[0].keys()), sweep_rows)
    return {
        "raw_rows": raw_rows,
        "sweep_rows": sweep_rows,
        "raw_json": raw_json_path,
        "sweep_csv": sweep_csv_path,
    }


def render_runtime_figure(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    figure_png_path = output_dir / "figure_r3_q4_gpu_runtime.png"
    figure_svg_path = output_dir / "figure_r3_q4_gpu_runtime.svg"
    x = np.arange(len(summary_rows))
    width = 0.34

    with plt.rc_context(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

        cpu_times = [row["cpu_median_time_sec"] for row in summary_rows]
        gpu_times = [row["gpu_median_time_sec"] for row in summary_rows]
        mesh_labels = [f"{row['mesh_level']} ({row['elements']} elems)" for row in summary_rows]

        axes[0].bar(x - width / 2, cpu_times, width, label="CPU", color="#4C78A8")
        axes[0].bar(x + width / 2, gpu_times, width, label="GPU", color="#F58518")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(mesh_labels)
        axes[0].set_ylabel("Median runtime (s)")
        axes[0].legend(frameon=False)
        axes[0].grid(False)

        for idx, row in enumerate(summary_rows):
            ymax = max(cpu_times[idx], gpu_times[idx])
            axes[0].text(
                idx,
                ymax * 1.03,
                f"{row['speedup_gpu_vs_cpu']:.2f}x",
                ha="center",
                va="bottom",
            )

        sweep_iterations = [row["iterations"] for row in sweep_rows]
        sweep_speedups = [row["speedup_gpu_vs_cpu"] for row in sweep_rows]
        axes[1].plot(
            sweep_iterations,
            sweep_speedups,
            marker="o",
            linewidth=2.5,
            color="#2A9D8F",
        )
        axes[1].set_xlabel("GN iterations")
        axes[1].set_ylabel("GPU speedup vs CPU")
        axes[1].set_xticks(sweep_iterations)
        axes[1].grid(False)
        for xi, yi in zip(sweep_iterations, sweep_speedups):
            axes[1].annotate(f"{yi:.2f}x", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")

        fig.tight_layout()
        fig.savefig(figure_png_path, dpi=300)
        fig.savefig(figure_svg_path, format="svg")
        plt.close(fig)

    enforce_svg_font_family(figure_svg_path)
    return {
        "png": figure_png_path,
        "svg": figure_svg_path,
    }


def enforce_svg_font_family(svg_path: Path, font_family: str = "Times New Roman") -> None:
    text = svg_path.read_text(encoding="utf-8")
    text = re.sub(r"font-family:[^;\"']+", f"font-family:'{font_family}'", text)
    text = re.sub(
        r"-inkscape-font-specification:[^;\"']+",
        f"-inkscape-font-specification:'{font_family}'",
        text,
    )
    svg_path.write_text(text, encoding="utf-8")


def get_git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def write_benchmark_environment(
    output_dir: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "container_name": args.container_name,
        "image": args.image_name,
        "mount": "D:/workspace/PyEIDORS => /root/shared",
        "code_commit": get_git_commit(),
        "branch": get_git_branch(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "framework": FRAMEWORK,
        "task": TASK,
        "scenario": SCENARIO,
        "gn_path": GN_PATH,
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "summary_iterations": SUMMARY_ITERATIONS,
        "summary_mesh_levels": SUMMARY_MESH_LEVELS,
        "iteration_sweep_mesh_level": SWEEP_MESH_LEVEL,
        "iteration_sweep": SWEEP_ITERATIONS,
        "gpu_scope_note": GPU_SCOPE_NOTE,
        "summary_rows": summary_rows,
        "iteration_sweep_rows": sweep_rows,
    }
    (output_dir / "benchmark_environment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_reproducibility_commands(output_dir: Path) -> None:
    commands = [
        "# Run inside the Docker container",
        "docker exec pyeidors bash -lc \"cd /root/shared && python scripts/benchmarks/build_reviewer_r3_q4_assets.py\"",
        "",
        "# Main CPU/GPU summary cases",
        (
            "docker exec pyeidors bash -lc "
            "\"cd /root/shared && python scripts/benchmarks/benchmark_reviewer_case.py "
            "--framework pyeidors --task absolute_gn --mesh-level medium --scenario low_z "
            "--device cpu --warmups 1 --repeats 3 --n-frames 1 --mesh-dir eit_meshes "
            "--absolute-lambda 1e-2 --absolute-max-iter 5 --gn-path legacy_dense "
            "--output-json docs/benchmarks/reviewer_r3_q4/raw/summary/pyeidors_absolute_gn_medium_low_z_cpu_iter5_legacy_dense.json\""
        ),
        (
            "docker exec pyeidors bash -lc "
            "\"cd /root/shared && python scripts/benchmarks/benchmark_reviewer_case.py "
            "--framework pyeidors --task absolute_gn --mesh-level medium --scenario low_z "
            "--device gpu --warmups 1 --repeats 3 --n-frames 1 --mesh-dir eit_meshes "
            "--absolute-lambda 1e-2 --absolute-max-iter 5 --gn-path legacy_dense "
            "--output-json docs/benchmarks/reviewer_r3_q4/raw/summary/pyeidors_absolute_gn_medium_low_z_gpu_iter5_legacy_dense.json\""
        ),
        (
            "docker exec pyeidors bash -lc "
            "\"cd /root/shared && python scripts/benchmarks/benchmark_reviewer_case.py "
            "--framework pyeidors --task absolute_gn --mesh-level fine --scenario low_z "
            "--device cpu --warmups 1 --repeats 3 --n-frames 1 --mesh-dir eit_meshes "
            "--absolute-lambda 1e-2 --absolute-max-iter 5 --gn-path legacy_dense "
            "--output-json docs/benchmarks/reviewer_r3_q4/raw/summary/pyeidors_absolute_gn_fine_low_z_cpu_iter5_legacy_dense.json\""
        ),
        (
            "docker exec pyeidors bash -lc "
            "\"cd /root/shared && python scripts/benchmarks/benchmark_reviewer_case.py "
            "--framework pyeidors --task absolute_gn --mesh-level fine --scenario low_z "
            "--device gpu --warmups 1 --repeats 3 --n-frames 1 --mesh-dir eit_meshes "
            "--absolute-lambda 1e-2 --absolute-max-iter 5 --gn-path legacy_dense "
            "--output-json docs/benchmarks/reviewer_r3_q4/raw/summary/pyeidors_absolute_gn_fine_low_z_gpu_iter5_legacy_dense.json\""
        ),
    ]
    (output_dir / "reproducibility_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")


def write_highlights(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    medium_row = next(row for row in summary_rows if row["mesh_level"] == "medium")
    fine_row = next(row for row in summary_rows if row["mesh_level"] == "fine")
    lines = [
        "Reviewer 3 Q4 GPU runtime benchmark highlights",
        f"- Task: {TASK}",
        f"- Scenario: {SCENARIO}",
        f"- GN path: {GN_PATH}",
        f"- Warmups/Repetitions: {WARMUPS}/{REPEATS}",
        (
            f"- Medium mesh ({medium_row['elements']} elements): "
            f"CPU {medium_row['cpu_median_time_sec']:.3f}s vs GPU {medium_row['gpu_median_time_sec']:.3f}s "
            f"({medium_row['speedup_gpu_vs_cpu']:.2f}x)"
        ),
        (
            f"- Fine mesh ({fine_row['elements']} elements): "
            f"CPU {fine_row['cpu_median_time_sec']:.3f}s vs GPU {fine_row['gpu_median_time_sec']:.3f}s "
            f"({fine_row['speedup_gpu_vs_cpu']:.2f}x)"
        ),
        f"- Scope note: {GPU_SCOPE_NOTE}",
    ]
    (output_dir / "highlights.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_outputs = build_summary_benchmark(args.output_dir, args.mesh_dir)
    sweep_outputs = build_iteration_sweep(args.output_dir, args.mesh_dir)
    figure_paths = render_runtime_figure(
        args.output_dir,
        summary_outputs["summary_rows"],
        sweep_outputs["sweep_rows"],
    )
    write_benchmark_environment(
        args.output_dir,
        args,
        summary_outputs["summary_rows"],
        sweep_outputs["sweep_rows"],
    )
    write_reproducibility_commands(args.output_dir)
    write_highlights(args.output_dir, summary_outputs["summary_rows"])

    manifest = {
        "summary_csv": summary_outputs["summary_csv"].as_posix(),
        "summary_raw_json": summary_outputs["raw_json"].as_posix(),
        "iteration_sweep_csv": sweep_outputs["sweep_csv"].as_posix(),
        "iteration_sweep_raw_json": sweep_outputs["raw_json"].as_posix(),
        "figure_png": figure_paths["png"].as_posix(),
        "figure_svg": figure_paths["svg"].as_posix(),
        "benchmark_environment_json": (args.output_dir / "benchmark_environment.json").as_posix(),
        "reproducibility_commands_txt": (args.output_dir / "reproducibility_commands.txt").as_posix(),
        "highlights_txt": (args.output_dir / "highlights.txt").as_posix(),
    }
    (args.output_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
