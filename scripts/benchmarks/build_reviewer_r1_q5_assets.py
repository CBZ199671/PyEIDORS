#!/usr/bin/env python3
"""Build Reviewer 1 Comment 5 assets across MATLAB EIDORS and Docker PyEIDORS."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FAIRNESS_DIR = REPO_ROOT / "docs" / "benchmarks" / "reviewer_suite" / "fairness"
MATLAB_SCRIPT_DIR = REPO_ROOT / "compare_with_Eidors"

SOURCE_ZS = {
    "low_z": 1e-6,
    "high_z": 1e-2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=FAIRNESS_DIR / "r1c5_cem_vs_pyeit.json")
    parser.add_argument("--output-csv", type=Path, default=FAIRNESS_DIR / "r1c5_cem_vs_pyeit.csv")
    parser.add_argument("--source-dir", type=Path, default=FAIRNESS_DIR / "r1c5_sources")
    parser.add_argument("--config-dir", type=Path, default=FAIRNESS_DIR / "r1c5_source_configs")
    parser.add_argument("--matlab-exe", type=Path, default=Path(r"D:\Program Files\MATLAB\R2023b\bin\matlab.exe"))
    parser.add_argument("--docker-container", default="pyeidors")
    parser.add_argument("--mesh-level", choices=["coarse", "medium", "fine"], default="medium")
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--background", type=float, default=1.0)
    parser.add_argument("--phantom-conductivity", type=float, default=2.0)
    parser.add_argument("--phantom-center-x", type=float, default=0.30)
    parser.add_argument("--phantom-center-y", type=float, default=0.20)
    parser.add_argument("--phantom-radius", type=float, default=0.20)
    parser.add_argument("--difference-hyperparameter", type=float, default=None)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=1.0)
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def windows_path(path: Path) -> str:
    text = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", text):
        return text
    resolved = path.resolve()
    text = str(resolved)
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5].upper()
        tail = text[7:].replace("/", "\\")
        return f"{drive}:\\{tail}"
    return text


def posix_windows_string(path: Path) -> str:
    return windows_path(path).replace("\\", "/")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def build_eidors_source_configs(args: argparse.Namespace, commit: str) -> dict[str, Path]:
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    configs: dict[str, Path] = {}
    for source_z, z_value in SOURCE_ZS.items():
        csv_path = args.source_dir / f"eidors_source_{source_z}.csv"
        output_json = args.source_dir / f"eidors_source_{source_z}_meta.json"
        config_path = args.config_dir / f"eidors_source_{source_z}.json"
        payload = {
            "mesh_level": args.mesh_level,
            "n_elec": args.n_elec,
            "scenario": source_z,
            "source_framework": "eidors",
            "commit": commit,
            "contact_impedance": z_value,
            "background": args.background,
            "obj_sigma": args.phantom_conductivity,
            "obj_pos": [args.phantom_center_x, args.phantom_center_y],
            "obj_radius": args.phantom_radius,
            "forward_export_csv": windows_path(csv_path),
            "output_json": windows_path(output_json),
        }
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        configs[source_z] = config_path
    return configs


def run_matlab_export(args: argparse.Namespace, config_path: Path) -> None:
    matlab_script_dir = posix_windows_string(MATLAB_SCRIPT_DIR)
    config_win = posix_windows_string(config_path)
    batch_expr = f"cd('{matlab_script_dir}'); export_cem_forward_csv('{config_win}');"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        f"& '{windows_path(args.matlab_exe)}' -batch \"{batch_expr}\"",
    ]
    run_command(command)


def ensure_docker_container(name: str) -> None:
    run_command(["docker", "inspect", name])


def run_docker_core(args: argparse.Namespace) -> None:
    ensure_docker_container(args.docker_container)
    output_json_rel = repo_rel(args.output_json)
    output_csv_rel = repo_rel(args.output_csv)
    source_dir_rel = repo_rel(args.source_dir)
    low_csv_rel = repo_rel(args.source_dir / "eidors_source_low_z.csv")
    high_csv_rel = repo_rel(args.source_dir / "eidors_source_high_z.csv")
    command = (
        "cd /root/shared && "
        "source /opt/final_venv/bin/activate && "
        "python scripts/benchmarks/reviewer_r1_q5_cem_vs_pyeit_core.py "
        f"--output-json {output_json_rel} "
        f"--output-csv {output_csv_rel} "
        f"--source-dir {source_dir_rel} "
        f"--eidors-source-low-z {low_csv_rel} "
        f"--eidors-source-high-z {high_csv_rel} "
        f"--mesh-level {args.mesh_level} "
        f"--n-elec {args.n_elec} "
        f"--background {args.background} "
        f"--phantom-conductivity {args.phantom_conductivity} "
        f"--phantom-center-x {args.phantom_center_x} "
        f"--phantom-center-y {args.phantom_center_y} "
        f"--phantom-radius {args.phantom_radius} "
        f"--electrode-coverage {args.electrode_coverage} "
        f"--radius {args.radius}"
    )
    if args.difference_hyperparameter is not None:
        command += f" --difference-hyperparameter {args.difference_hyperparameter}"
    run_command(["docker", "exec", args.docker_container, "bash", "-lc", command])


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    configs = build_eidors_source_configs(args, commit)
    for source_z in ("low_z", "high_z"):
        run_matlab_export(args, configs[source_z])
    run_docker_core(args)
    print(f"Generated: {args.output_json}")
    print(f"Generated: {args.output_csv}")


if __name__ == "__main__":
    main()
