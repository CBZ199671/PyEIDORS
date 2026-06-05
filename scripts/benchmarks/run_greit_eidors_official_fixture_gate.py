#!/usr/bin/env python3
"""Run the T49 GREIT gate only when a real MATLAB/EIDORS fixture is present."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.runtime_paths import pyeidors_output_path
from scripts.benchmarks.benchmark_greit_eidors_parity_48e import run_benchmark
from scripts.diagnostics.compare_greit_eidors_parity import compare_greit_eidors_parity
from scripts.diagnostics.eidors_greit_fixture import validate_fixture_hdf5


GATE_SCHEMA = "pyeidors-greit-eidors-official-fixture-gate-v1"
DEFAULT_CASE_ID = "reduced_48e_5936"
DEFAULT_FIXTURE_DIR = pyeidors_output_path("eidors_greit_fixtures")
DEFAULT_OUTPUT_DIR = pyeidors_output_path(
    "runtime_benchmarks", "greit_eidors_parity_48e_5936_t49_official_20260426"
)


def run_official_fixture_gate(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fixture: str | Path | None = None,
    fixture_dir: str | Path = DEFAULT_FIXTURE_DIR,
    case_id: str = DEFAULT_CASE_ID,
    n_frames: int = 512,
    voxel_shape: tuple[int, int, int] = (6, 6, 4),
    devices: Sequence[str] = ("cpu", "auto", "cuda"),
    dtype: str = "float64",
    strict: bool = False,
) -> dict[str, Any]:
    """Validate an official fixture and rerun T49, or write a blocked report."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = _resolve_fixture_path(fixture, fixture_dir, case_id)
    if not fixture_path.exists():
        payload = _blocked_payload(
            output_dir=out_dir,
            fixture_path=fixture_path,
            fixture_dir=Path(fixture_dir),
            case_id=case_id,
            reason="missing_official_fixture",
        )
        _write_gate_outputs(out_dir, payload)
        if strict:
            raise SystemExit(2)
        return payload

    try:
        fixture_validation = validate_fixture_hdf5(fixture_path)
    except Exception as exc:
        payload = _blocked_payload(
            output_dir=out_dir,
            fixture_path=fixture_path,
            fixture_dir=Path(fixture_dir),
            case_id=case_id,
            reason=f"invalid_official_fixture:{type(exc).__name__}: {exc}",
        )
        _write_gate_outputs(out_dir, payload)
        if strict:
            raise SystemExit(2)
        return payload

    n_measurements = _fixture_measurement_count(fixture_validation)
    computed_report_path = out_dir / "computed_from_fixture_parity_report.json"
    computed_parity = compare_greit_eidors_parity(
        fixture_path,
        report_out=computed_report_path,
        abs_tol=1.0e-8,
        rel_tol=1.0e-8,
    )
    benchmark_dir = out_dir / "t49_official_benchmark"
    benchmark = run_benchmark(
        output_dir=benchmark_dir,
        fixture=fixture_path,
        n_measurements=n_measurements,
        n_frames=n_frames,
        voxel_shape=voxel_shape,
        devices=devices,
        dtype=dtype,
        cases=(case_id,),
    )
    allowed = bool(
        benchmark["gate"]["official_equivalence_claim_allowed"]
        and computed_parity["all_passed"]
    )
    payload = {
        "schema": GATE_SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": str(benchmark["scope"]),
        "status": "passed" if allowed else "failed",
        "official_fixture_available": True,
        "official_eidors_fixture": True,
        "official_equivalence_claim_allowed": allowed,
        "fixture_path": str(fixture_path),
        "fixture_validation": fixture_validation,
        "t49_rerun_status": "completed",
        "t49_summary_path": str(benchmark_dir / "summary.json"),
        "t49_report_path": str(benchmark_dir / "README.md"),
        "computed_from_fixture_parity_report_path": str(computed_report_path),
        "computed_from_fixture_parity_passed": bool(computed_parity["all_passed"]),
        "computed_from_fixture_comparison_names": [
            item["name"] for item in computed_parity.get("comparisons", [])
        ],
        "benchmark_gate": benchmark["gate"],
        "benchmark_invariants": benchmark["invariants"],
    }
    _write_gate_outputs(out_dir, payload)
    if strict and not allowed:
        raise SystemExit(1)
    return _jsonable(payload)


def _resolve_fixture_path(
    fixture: str | Path | None, fixture_dir: str | Path, case_id: str
) -> Path:
    if fixture is not None:
        return Path(fixture)
    return Path(fixture_dir) / f"{case_id}_eidors_greit_fixture.mat"


def _fixture_measurement_count(fixture_validation: Mapping[str, Any]) -> int:
    shape = tuple(int(v) for v in fixture_validation["shapes"]["vh"])
    count = 1
    for value in shape:
        count *= int(value)
    return int(count)


def _blocked_payload(
    *,
    output_dir: Path,
    fixture_path: Path,
    fixture_dir: Path,
    case_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": GATE_SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "blocked_reason": reason,
        "official_fixture_available": False,
        "official_eidors_fixture": False,
        "official_equivalence_claim_allowed": False,
        "fixture_path": str(fixture_path),
        "t49_rerun_status": "not_run",
        "environment": {
            "matlab": shutil.which("matlab"),
            "octave": shutil.which("octave"),
        },
        "capture": {
            "case_id": case_id,
            "fixture_dir": str(fixture_dir),
            "matlab_batch_command": _matlab_capture_command(
                fixture_dir=fixture_dir,
                case_id=case_id,
            ),
            "rerun_command": _rerun_command(
                fixture_path=fixture_path,
                output_dir=output_dir,
            ),
        },
    }


def _matlab_capture_command(*, fixture_dir: Path, case_id: str) -> str:
    diagnostics_dir = PROJECT_ROOT / "scripts" / "diagnostics"
    return (
        'matlab -batch "'
        "addpath(genpath('<EIDORS_ROOT>')); "
        f"addpath('{diagnostics_dir.as_posix()}'); "
        "capture_eidors_greit_fixture("
        f"'out_dir','{fixture_dir.as_posix()}',"
        f"'case_id','{case_id}',"
        "'overwrite',true)\""
    )


def _rerun_command(*, fixture_path: Path, output_dir: Path) -> str:
    return (
        'nix develop --command bash -lc "'
        "uv run python scripts/benchmarks/run_greit_eidors_official_fixture_gate.py "
        f"--fixture {fixture_path.as_posix()} "
        f"--output-dir {output_dir.as_posix()} "
        '--strict"'
    )


def _write_gate_outputs(out_dir: Path, payload: Mapping[str, Any]) -> None:
    payload = _jsonable(dict(payload))
    (out_dir / "official_gate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(out_dir / "README.md", payload)


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# GREIT Official Fixture Gate",
        "",
        f"- schema: `{payload['schema']}`",
        f"- status: `{payload['status']}`",
        *(
            [f"- scope: `{payload['scope']}`"]
            if payload.get("scope") is not None
            else []
        ),
        f"- official fixture: `{payload['official_eidors_fixture']}`",
        f"- official-equivalence claim allowed: `{payload['official_equivalence_claim_allowed']}`",
        f"- fixture: `{payload['fixture_path']}`",
        "",
    ]
    if payload["status"] == "blocked":
        capture = payload["capture"]
        lines.extend(
            [
                "## Blocked",
                "",
                f"- reason: `{payload['blocked_reason']}`",
                f"- MATLAB on PATH: `{payload['environment']['matlab']}`",
                f"- Octave on PATH: `{payload['environment']['octave']}`",
                "",
                "## Capture",
                "",
                "```bash",
                str(capture["matlab_batch_command"]),
                "```",
                "",
                "## Rerun",
                "",
                "```bash",
                str(capture["rerun_command"]),
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## T49 Rerun",
                "",
                f"- status: `{payload['t49_rerun_status']}`",
                f"- summary: `{payload['t49_summary_path']}`",
                f"- report: `{payload['t49_report_path']}`",
                f"- computed fixture parity: `{payload['computed_from_fixture_parity_passed']}`",
                f"- computed parity report: `{payload['computed_from_fixture_parity_report_path']}`",
                f"- benchmark gate: `{payload['benchmark_gate']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_shape(raw: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("shape must be three positive integers")
    return parts


def _parse_devices(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--n-frames", type=int, default=512)
    parser.add_argument("--voxel-shape", default="6,6,4")
    parser.add_argument("--devices", default="cpu,auto,cuda")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the official fixture is missing or the gate fails.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_official_fixture_gate(
        output_dir=args.output_dir,
        fixture=args.fixture,
        fixture_dir=args.fixture_dir,
        case_id=args.case_id,
        n_frames=args.n_frames,
        voxel_shape=_parse_shape(args.voxel_shape),
        devices=_parse_devices(args.devices),
        dtype=args.dtype,
        strict=args.strict,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
