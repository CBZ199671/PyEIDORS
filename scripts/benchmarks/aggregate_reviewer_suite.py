#!/usr/bin/env python3
"""Aggregate SoftwareX reviewer benchmark outputs into publication assets."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_reviewer_case import get_git_commit  # noqa: E402


CANONICAL_COLUMNS = [
    "framework",
    "task",
    "mesh_level",
    "mesh_name",
    "n_nodes",
    "n_elements",
    "n_frames",
    "device",
    "repeats",
    "warmups",
    "mean",
    "std",
    "median",
    "iqr",
    "peak_rss_mb",
    "scenario",
    "commit",
]

REQUIRED_RESULT_FIELDS = [
    "framework",
    "task",
    "mesh_level",
    "mesh_name",
    "n_nodes",
    "n_elements",
    "n_frames",
    "device",
    "repeats",
    "warmups",
    "mean",
    "std",
    "median",
    "iqr",
    "peak_rss_mb",
    "scenario",
    "commit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "reviewer_suite" / "raw")
    parser.add_argument("--cross-dir", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "reviewer_suite" / "fairness" / "raw_cross")
    parser.add_argument("--mesh-control-json", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "reviewer_suite" / "fairness" / "mesh_matched_control.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "reviewer_suite" / "aggregated")
    parser.add_argument("--llm-dir", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "reviewer_r1_q3")
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "docs" / "benchmarks" / "reviewer_suite" / "state")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    return parser.parse_args()


def import_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def load_json_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith("_config.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            rows.extend(payload)
        elif isinstance(payload, dict):
            rows.append(payload)
    return rows


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_single_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def normalise_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for alias in ["mean_sec", "std_sec", "median_sec", "iqr_sec"]:
        if alias in df.columns:
            df = df.drop(columns=[alias])
    return df


def write_dataframe(df: pd.DataFrame, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    path_base.with_suffix(".json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")


def get_manifest_output_paths(manifest_case: dict[str, Any], latest: dict[str, Any]) -> list[Path]:
    rel_paths: list[str] = []
    output_json = str(latest.get("output_json", "") or "").strip()
    if output_json:
        rel_paths.append(output_json)
    for rel_path in manifest_case.get("outputs", []) or []:
        rel_text = str(rel_path or "").strip()
        if rel_text and rel_text not in rel_paths:
            rel_paths.append(rel_text)
    return [REPO_ROOT / Path(rel_path) for rel_path in rel_paths]


def validate_raw_result_for_case(
    payload: dict[str, Any],
    manifest_case: dict[str, Any],
    manifest_defaults: dict[str, Any],
) -> tuple[bool, str]:
    missing_fields = [field for field in REQUIRED_RESULT_FIELDS if field not in payload]
    if missing_fields:
        return False, f"missing_{missing_fields[0]}"

    expected_commit = str(manifest_defaults.get("commit", "") or "")
    expected_warmups = int(manifest_defaults.get("warmups", 0) or 0)
    expected_repeats = int(manifest_defaults.get("repeats", 0) or 0)
    expected_frames = int(manifest_case.get("n_frames", 0) or 0)

    comparisons = {
        "framework": str(manifest_case.get("framework", "") or ""),
        "task": str(manifest_case.get("task", "") or ""),
        "mesh_level": str(manifest_case.get("mesh_level", "") or ""),
        "scenario": str(manifest_case.get("scenario", "") or ""),
        "device": str(manifest_case.get("device", "") or ""),
        "commit": expected_commit,
    }
    for field, expected_value in comparisons.items():
        if str(payload.get(field, "") or "") != expected_value:
            return False, f"identity_mismatch_{field}"

    if int(payload.get("warmups", -1)) != expected_warmups:
        return False, "identity_mismatch_warmups"
    if int(payload.get("repeats", -1)) != expected_repeats:
        return False, "identity_mismatch_repeats"
    if int(payload.get("n_frames", -1)) != expected_frames:
        return False, "identity_mismatch_n_frames"
    return True, ""


def reconcile_benchmark_status(
    manifest_case: dict[str, Any],
    latest: dict[str, Any],
    manifest_defaults: dict[str, Any],
) -> tuple[str, str]:
    status = str(latest.get("status", "") or "")
    if status == "deferred":
        return status, str(latest.get("message", ""))
    if status not in {"", "failed", "running"}:
        return str(latest.get("status", "")), str(latest.get("message", ""))
    if str(manifest_case.get("kind", "")) != "benchmark":
        return str(latest.get("status", "")), str(latest.get("message", ""))

    for candidate_path in get_manifest_output_paths(manifest_case, latest):
        if candidate_path.suffix.lower() != ".json" or not candidate_path.exists():
            continue
        payload = load_single_json(candidate_path)
        if payload is None:
            continue
        valid, _reason = validate_raw_result_for_case(payload, manifest_case, manifest_defaults)
        if not valid:
            continue
        rel_path = candidate_path.relative_to(REPO_ROOT).as_posix()
        return "completed", f"reconciled_from_raw_json:{rel_path}"
    return str(latest.get("status", "")), str(latest.get("message", ""))


def build_run_status_summary(state_dir: Path) -> pd.DataFrame:
    manifest_path = state_dir / "run_manifest.json"
    status_path = state_dir / "run_status.jsonl"
    manifest_cases = []
    selected_phase = ""
    manifest_defaults: dict[str, Any] = {}
    if manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifest_cases = manifest_payload.get("cases", [])
        selected_phase = str(manifest_payload.get("phase", ""))
        manifest_defaults = {
            "commit": manifest_payload.get("commit", ""),
            "warmups": manifest_payload.get("warmups", 0),
            "repeats": manifest_payload.get("repeats", 0),
        }

    latest_events: dict[str, dict[str, Any]] = {}
    for event in load_jsonl_rows(status_path):
        case_id = str(event.get("case_id", "")).strip()
        if not case_id:
            continue
        latest_events[case_id] = event

    rows: list[dict[str, Any]] = []
    for manifest_case in manifest_cases:
        case_id = manifest_case.get("case_id", "")
        latest = latest_events.get(case_id, {})
        status, message = reconcile_benchmark_status(manifest_case, latest, manifest_defaults)
        row = {
            "case_id": case_id,
            "framework": manifest_case.get("framework", ""),
            "task": manifest_case.get("task", ""),
            "mesh_level": manifest_case.get("mesh_level", ""),
            "scenario": manifest_case.get("scenario", ""),
            "device": manifest_case.get("device", ""),
            "n_frames": manifest_case.get("n_frames", 0),
            "phase": manifest_case.get("phase", ""),
            "kind": manifest_case.get("kind", ""),
            "runner": manifest_case.get("runner", ""),
            "status": status or "deferred",
            "message": message or f"No run event recorded for phase {selected_phase or 'unknown'}",
            "updated_at": latest.get("updated_at", ""),
            "output_json": latest.get("output_json", "") or next(iter(manifest_case.get("outputs", []) or []), ""),
        }
        rows.append(row)

    for case_id, latest in latest_events.items():
        if any(row["case_id"] == case_id for row in rows):
            continue
        rows.append({
            "case_id": case_id,
            "framework": latest.get("framework", ""),
            "task": latest.get("task", ""),
            "mesh_level": latest.get("mesh_level", ""),
            "scenario": latest.get("scenario", ""),
            "device": latest.get("device", ""),
            "n_frames": latest.get("n_frames", 0),
            "phase": latest.get("phase", ""),
            "kind": latest.get("kind", ""),
            "runner": latest.get("runner", ""),
            "status": latest.get("status", ""),
            "message": latest.get("message", ""),
            "updated_at": latest.get("updated_at", ""),
            "output_json": latest.get("output_json", ""),
        })

    return pd.DataFrame(rows)


def build_main_summary(df: pd.DataFrame) -> pd.DataFrame:
    selectors = [
        ("Forward solve", {"framework": "pyeidors", "task": "forward", "mesh_level": "medium", "device": "cpu", "scenario": "low_z"}),
        ("Jacobian (CPU)", {"framework": "pyeidors", "task": "jacobian", "mesh_level": "medium", "device": "cpu", "scenario": "low_z"}),
        ("Jacobian (GPU)", {"framework": "pyeidors", "task": "jacobian", "mesh_level": "medium", "device": "gpu", "scenario": "low_z"}),
        ("Difference reconstruction", {"framework": "pyeidors", "task": "difference", "mesh_level": "medium", "device": "cpu", "scenario": "low_z"}),
        ("Absolute GN (CPU)", {"framework": "pyeidors", "task": "absolute_gn", "mesh_level": "medium", "device": "cpu", "scenario": "low_z"}),
        ("Absolute GN (GPU)", {"framework": "pyeidors", "task": "absolute_gn", "mesh_level": "medium", "device": "gpu", "scenario": "low_z"}),
        ("Multi-frame throughput (1000)", {"framework": "pyeidors", "task": "multi_frame_difference", "mesh_level": "medium", "device": "cpu", "scenario": "low_z", "n_frames": 1000}),
    ]
    rows: list[dict[str, Any]] = []
    for label, criteria in selectors:
        subset = df.copy()
        for key, value in criteria.items():
            subset = subset[subset[key] == value]
        if subset.empty:
            continue
        record = subset.iloc[0].to_dict()
        rows.append({
            "Summary item": label,
            "Framework": record["framework"],
            "Mesh": record["mesh_level"],
            "Device": record["device"],
            "Median (s)": record["median"],
            "IQR (s)": record["iqr"],
            "Peak RSS (MB)": record["peak_rss_mb"],
            "Voltage RMSE": record.get("voltage_rmse", record.get("avg_voltage_rmse")),
            "Conductivity relative error (%)": record.get("conductivity_relative_error_pct", record.get("avg_conductivity_relative_error_pct")),
        })
    return pd.DataFrame(rows)


def build_environment_rows(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    commit = get_git_commit()
    sample = df.iloc[0].to_dict() if not df.empty else {}
    env = {
        "version": "v1.1.0",
        "commit": commit,
        "mode": "reviewer_suite",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": import_version("numpy"),
        "pandas": import_version("pandas"),
        "torch": import_version("torch"),
        "pyeit": import_version("pyeit"),
        "gpu_scope_note": "GPU acceleration currently benefits inverse/tensor operations; forward PDE assembly remains on the FEniCS/CPU side.",
        "docker_workdir": "/root/shared",
        "dockerfile": "Dockerfile",
        "eidors_startup": r"run('D:\\Program Files\\MATLAB\\R2023b\\toolbox\\eidors-v3.12-ng\\eidors\\startup.m')",
        "mesh_levels": ["coarse", "medium", "fine"],
        "frame_counts": [1, 10, 100, 1000],
        "warmups": int(sample.get("warmups", 3)),
        "repeats": int(sample.get("repeats", 10)),
    }
    rows = pd.DataFrame([
        {"Item": "Code version", "Value": env["version"]},
        {"Item": "Commit", "Value": env["commit"]},
        {"Item": "Platform", "Value": env["platform"]},
        {"Item": "Python", "Value": env["python"]},
        {"Item": "NumPy", "Value": env["numpy"]},
        {"Item": "pandas", "Value": env["pandas"]},
        {"Item": "PyTorch", "Value": env["torch"]},
        {"Item": "pyEIT", "Value": env["pyeit"]},
        {"Item": "Warm-ups", "Value": env["warmups"]},
        {"Item": "Repeats", "Value": env["repeats"]},
        {"Item": "Meshes", "Value": ", ".join(env["mesh_levels"])},
        {"Item": "Frame counts", "Value": ", ".join(str(v) for v in env["frame_counts"])},
        {"Item": "GPU scope note", "Value": env["gpu_scope_note"]},
        {"Item": "EIDORS startup", "Value": env["eidors_startup"]},
    ])
    return env, rows


def build_release_manifest(env: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "v1.1.0",
        "commit": env["commit"],
        "zenodo_doi": "pending_release",
        "docker": {
            "workdir": "/root/shared",
            "dockerfile": "Dockerfile",
            "benchmark_dependency": "pyeit~=1.2.4",
        },
        "eidors": {
            "startup": env["eidors_startup"],
            "platform": "Windows MATLAB R2023b",
        },
        "commands": [
            "powershell -ExecutionPolicy Bypass -File scripts/benchmarks/run_reviewer_suite.ps1 -Phase all",
            "powershell -ExecutionPolicy Bypass -File scripts/benchmarks/run_reviewer_suite.ps1 -Phase heavy",
            "docker exec pyeidors bash -lc \"cd /root/shared && python scripts/reviewer_demos/run_llm_agent_case.py\"",
        ],
        "public_assets": [
            "benchmark scripts",
            "aggregated benchmark tables",
            "figure reproduction assets",
            "Dockerfile",
            "requirements/environment declarations",
            "exact commit hash",
            "shareable processed tank/maize data",
        ],
        "restricted_assets": [
            "raw experimental datasets subject to sharing restrictions will be released on request when permitted",
        ],
        "gpu_scope_note": env["gpu_scope_note"],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = normalise_dataframe(load_json_rows(args.raw_dir))
    if not raw_df.empty:
        raw_df = raw_df.sort_values(["task", "framework", "mesh_level", "device", "scenario", "n_frames"], ignore_index=True)
    write_dataframe(raw_df, args.output_dir / "aggregate_all_cases")

    if not raw_df.empty:
        metric_cols = [col for col in raw_df.columns if col not in CANONICAL_COLUMNS]
        s2_cols = CANONICAL_COLUMNS + [col for col in sorted(metric_cols) if col not in CANONICAL_COLUMNS]
        raw_df[s2_cols].to_csv(args.output_dir / "table_s2_system_benchmarks.csv", index=False)
        build_main_summary(raw_df).to_csv(args.output_dir / "table_main_benchmark_summary.csv", index=False)
        pyeit_df = raw_df[
            (raw_df["framework"] == "pyeit")
            & (raw_df["task"] == "difference")
            & (raw_df["scenario"].isin(["low_z", "high_z"]))
        ].copy()
        pyeit_df.to_csv(args.output_dir / "table_s3_pyeit_scenarios.csv", index=False)

    cross_df = normalise_dataframe(load_json_rows(args.cross_dir))
    if not cross_df.empty:
        cross_df = cross_df.sort_values(["source_framework", "framework"], ignore_index=True)
        cross_df.to_csv(args.output_dir / "table_s3_cross_generation.csv", index=False)

    if args.mesh_control_json.exists():
        mesh_payload = json.loads(args.mesh_control_json.read_text(encoding="utf-8"))
        mesh_df = pd.DataFrame(mesh_payload.get("rows", []))
        if not mesh_df.empty:
            mesh_df.to_csv(args.output_dir / "table_s3_mesh_matched_control.csv", index=False)

    env, env_rows = build_environment_rows(raw_df)
    (args.output_dir / "benchmark_environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    env_rows.to_csv(args.output_dir / "table_s1_benchmark_environment.csv", index=False)

    release_manifest = build_release_manifest(env)
    (args.output_dir / "release_manifest.json").write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")

    status_df = build_run_status_summary(args.state_dir)
    if not status_df.empty:
        write_dataframe(status_df, args.output_dir / "run_status_summary")

    if args.llm_dir.exists():
        llm_metrics = args.llm_dir / "metrics.json"
        llm_manifest = args.llm_dir / "asset_exports" / "manifest.json"
        summary = {
            "metrics": json.loads(llm_metrics.read_text(encoding="utf-8")) if llm_metrics.exists() else {},
            "assets": json.loads(llm_manifest.read_text(encoding="utf-8")) if llm_manifest.exists() else {},
        }
        (args.output_dir / "llm_poc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    commands = "\n".join([
        "powershell -ExecutionPolicy Bypass -File scripts/benchmarks/run_reviewer_suite.ps1 -Phase all",
        "powershell -ExecutionPolicy Bypass -File scripts/benchmarks/run_reviewer_suite.ps1 -Phase heavy",
        "Get-Content docs/benchmarks/reviewer_suite/state/current_case.json",
        "docker exec pyeidors bash -lc \"cd /root/shared && python scripts/reviewer_demos/run_llm_agent_case.py\"",
        "python scripts/benchmarks/aggregate_reviewer_suite.py",
    ])
    (args.output_dir / "reproducibility_commands.txt").write_text(commands + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
