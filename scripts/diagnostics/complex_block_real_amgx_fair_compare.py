#!/usr/bin/env python3
"""Run fair complex64-CUDA vs block-real AmgX comparison cases."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = (
    PROJECT_ROOT / "scripts" / "diagnostics" / "complex_block_real_amgx_probe.py"
)


def _parse_cases(raw: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"case must be N_ELEC:REFINEMENT, got {token!r}")
        n_elec, refinement = token.split(":", 1)
        cases.append((int(n_elec), int(refinement)))
    if not cases:
        raise ValueError("at least one case is required")
    return cases


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )


def _nix_python(shell: str, *script_args: str) -> list[str]:
    return [
        "nix",
        "develop",
        f".#{shell}",
        "--command",
        "python",
        str(PROBE_SCRIPT),
        *script_args,
    ]


def _metric_float(block: dict[str, Any], key: str) -> float:
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _case_row(
    *,
    case_dir: Path,
    n_elec: int,
    refinement: int,
    max_solution_rel_l2: float,
    max_electrode_rel_l2: float,
) -> dict[str, Any]:
    export_metadata = json.loads(
        (case_dir / "metadata.json").read_text(encoding="utf-8")
    )
    solve_report = json.loads(
        (case_dir / "block_real_amgx.json").read_text(encoding="utf-8")
    )
    solution_error = solve_report.get("solution_error_vs_reference", {})
    electrode_error = solve_report.get("electrode_voltage_error_vs_reference", {})
    solution_rel_l2 = _metric_float(solution_error, "relative_l2")
    electrode_rel_l2 = _metric_float(electrode_error, "relative_l2")
    passed = bool(
        solution_rel_l2 <= float(max_solution_rel_l2)
        and electrode_rel_l2 <= float(max_electrode_rel_l2)
    )
    return {
        "case": f"{n_elec}e_ref{refinement}",
        "n_elec": int(n_elec),
        "refinement": int(refinement),
        "mesh": export_metadata.get("mesh", {}),
        "n_dofs": export_metadata.get("n_dofs"),
        "n_patterns": export_metadata.get("n_patterns"),
        "reference_kind": solve_report.get("reference_kind"),
        "runtime_vs_direct": export_metadata.get("runtime_vs_direct", {}),
        "runtime_electrode_voltage_vs_direct": export_metadata.get(
            "runtime_electrode_voltage_vs_direct", {}
        ),
        "solver": solve_report.get("solver", {}),
        "solution_error_vs_reference": solution_error,
        "electrode_voltage_error_vs_reference": electrode_error,
        "pass": passed,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Complex block-real AmgX fair comparison",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- complex_shell: `{payload['complex_shell']}`",
        f"- amgx_shell: `{payload['amgx_shell']}`",
        f"- max_solution_rel_l2: `{payload['max_solution_rel_l2']}`",
        f"- max_electrode_rel_l2: `{payload['max_electrode_rel_l2']}`",
        "",
        "| case | dofs | cells | solution relL2 | electrode relL2 | iterations | pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["cases"]:
        solver = row.get("solver", {})
        iterations = solver.get("iterations_per_rhs") or []
        iter_value = max(iterations) if iterations else ""
        mesh = row.get("mesh", {})
        lines.append(
            "| {case} | {dofs} | {cells} | {sol:.6g} | {elec:.6g} | {it} | {passed} |".format(
                case=row["case"],
                dofs=row.get("n_dofs", ""),
                cells=mesh.get("elements", ""),
                sol=_metric_float(row["solution_error_vs_reference"], "relative_l2"),
                elec=_metric_float(
                    row["electrode_voltage_error_vs_reference"], "relative_l2"
                ),
                it=iter_value,
                passed="yes" if row.get("pass") else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="8:1,16:1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/complex_block_real_amgx/fair_compare"),
    )
    parser.add_argument("--complex-shell", default="complex64-cuda")
    parser.add_argument("--amgx-shell", default="cuda-amgx")
    parser.add_argument("--radius", type=float, default=0.18)
    parser.add_argument("--height", type=float, default=0.16)
    parser.add_argument("--background", default="1+0.25j")
    parser.add_argument("--contact-impedance", default="1e-3+2e-4j")
    parser.add_argument("--reference-solver-preset", default="3d_gamg")
    parser.add_argument("--amgx-profile", default="real_jacobi_l1")
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--max-it", type=int, default=4000)
    parser.add_argument("--max-solution-rel-l2", type=float, default=1.0e-4)
    parser.add_argument("--max-electrode-rel-l2", type=float, default=1.0e-4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = _parse_cases(str(args.cases))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for n_elec, refinement in cases:
        case_dir = output_dir / f"{n_elec}e_ref{refinement}"
        case_dir.mkdir(parents=True, exist_ok=True)
        export_cmd = _nix_python(
            str(args.complex_shell),
            "export-system",
            "--output-dir",
            str(case_dir),
            "--n-elec",
            str(n_elec),
            "--refinement",
            str(refinement),
            "--radius",
            str(args.radius),
            "--height",
            str(args.height),
            "--background",
            str(args.background),
            "--contact-impedance",
            str(args.contact_impedance),
            "--reference-solver-preset",
            str(args.reference_solver_preset),
            "--petsc-device",
            "cuda",
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
            "--max-it",
            str(args.max_it),
        )
        solve_cmd = _nix_python(
            str(args.amgx_shell),
            "solve-block-real",
            "--input-dir",
            str(case_dir),
            "--output-json",
            str(case_dir / "block_real_amgx.json"),
            "--amgx-profile",
            str(args.amgx_profile),
            "--rtol",
            str(args.rtol),
            "--atol",
            str(args.atol),
            "--max-it",
            str(args.max_it),
        )
        _run(export_cmd, dry_run=bool(args.dry_run))
        _run(solve_cmd, dry_run=bool(args.dry_run))
        if not args.dry_run:
            rows.append(
                _case_row(
                    case_dir=case_dir,
                    n_elec=n_elec,
                    refinement=refinement,
                    max_solution_rel_l2=float(args.max_solution_rel_l2),
                    max_electrode_rel_l2=float(args.max_electrode_rel_l2),
                )
            )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complex_shell": str(args.complex_shell),
        "amgx_shell": str(args.amgx_shell),
        "reference_solver_preset": str(args.reference_solver_preset),
        "amgx_profile": str(args.amgx_profile),
        "max_solution_rel_l2": float(args.max_solution_rel_l2),
        "max_electrode_rel_l2": float(args.max_electrode_rel_l2),
        "cases": rows,
        "pass": all(bool(row.get("pass")) for row in rows) if rows else None,
    }
    if not args.dry_run:
        summary_json = output_dir / "summary.json"
        summary_md = output_dir / "summary.md"
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_markdown(summary_md, payload)
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
