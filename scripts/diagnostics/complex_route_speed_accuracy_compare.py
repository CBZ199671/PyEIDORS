#!/usr/bin/env python3
"""Compare complex CPU direct, native complex CUDA, and block-real AmgX solves."""

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
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, check=False)
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


def _metric_float(block: dict[str, Any] | None, key: str) -> float:
    if not isinstance(block, dict):
        return float("nan")
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _optional_metric_float(block: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _route_row(
    *,
    name: str,
    family: str,
    solve_seconds: float | None,
    solution_error: dict[str, Any] | None,
    electrode_error: dict[str, Any] | None,
    true_relative_residual_max: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "solve_seconds": solve_seconds,
        "solution_relative_l2": _metric_float(solution_error, "relative_l2"),
        "electrode_relative_l2": _metric_float(electrode_error, "relative_l2"),
        "true_relative_residual_max": true_relative_residual_max,
        "solution_error": solution_error or {},
        "electrode_voltage_error": electrode_error or {},
        **(extra or {}),
    }


def _case_row(
    *,
    case_dir: Path,
    n_elec: int,
    refinement: int,
    max_solution_rel_l2: float,
    max_electrode_rel_l2: float,
    max_true_residual: float = 1.0e-6,
) -> dict[str, Any]:
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    amgx_report = json.loads(
        (case_dir / "block_real_amgx.json").read_text(encoding="utf-8")
    )
    direct_reference = metadata.get("reference_solution", {})
    dense_reference = metadata.get("dense_reference_solution", {})
    runtime_reference = metadata.get("runtime_reference_solution", {})
    runtime_diagnostics = runtime_reference.get("backend_diagnostics", {})
    runtime_true_residual = metadata.get("runtime_true_residual", {})
    amgx_solver = amgx_report.get("solver", {})
    cpu_direct_skipped = bool(direct_reference.get("cpu_direct_skipped"))

    routes: list[dict[str, Any]] = []
    if not cpu_direct_skipped:
        routes.append(
            _route_row(
                name="cpu_sparse_lu_direct",
                family="cpu_direct_reference",
                solve_seconds=direct_reference.get("solve_seconds"),
                solution_error={"relative_l2": 0.0, "max_abs": 0.0, "l2": 0.0},
                electrode_error={"relative_l2": 0.0, "max_abs": 0.0, "l2": 0.0},
                true_relative_residual_max=0.0,
                extra={
                    "reference_kind": direct_reference.get("kind"),
                    "scalar_type": "complex128_reference",
                },
            )
        )
    if not cpu_direct_skipped and not dense_reference.get("skipped"):
        routes.append(
            _route_row(
                name="cpu_dense_direct",
                family="cpu_dense_reference_check",
                solve_seconds=dense_reference.get("solve_seconds"),
                solution_error=metadata.get("dense_direct_vs_sparse_direct"),
                electrode_error=metadata.get(
                    "dense_direct_electrode_voltage_vs_sparse_direct"
                ),
                true_relative_residual_max=0.0,
                extra={
                    "reference_kind": dense_reference.get("kind"),
                    "scalar_type": "complex128_reference",
                },
            )
        )
    native_gpu = _route_row(
        name=str(metadata.get("reference_route") or "complex64_cuda_runtime"),
        family=(
            "native_complex_cuda_reference"
            if cpu_direct_skipped
            else "native_complex_cuda"
        ),
        solve_seconds=runtime_reference.get("solve_seconds"),
        solution_error=(
            {"relative_l2": 0.0, "max_abs": 0.0, "l2": 0.0}
            if cpu_direct_skipped
            else metadata.get("runtime_vs_direct")
        ),
        electrode_error=(
            {"relative_l2": 0.0, "max_abs": 0.0, "l2": 0.0}
            if cpu_direct_skipped
            else metadata.get("runtime_electrode_voltage_vs_direct")
        ),
        true_relative_residual_max=_optional_metric_float(
            runtime_true_residual, "relative_max"
        ),
        extra={
            "petsc_scalar_type": metadata.get("petsc_scalar_type"),
            "petsc_device": metadata.get("petsc_device"),
            "backend_diagnostics": runtime_diagnostics,
        },
    )
    block_real_amgx = _route_row(
        name="block_real_cuda_amgx",
        family="block_real_amgx",
        solve_seconds=amgx_solver.get("solve_seconds"),
        solution_error=amgx_report.get("solution_error_vs_reference"),
        electrode_error=amgx_report.get("electrode_voltage_error_vs_reference"),
        true_relative_residual_max=_optional_metric_float(
            amgx_solver, "true_relative_residual_max"
        ),
        extra={
            "reference_kind": amgx_report.get("reference_kind"),
            "solver": amgx_solver,
        },
    )
    routes.extend([native_gpu, block_real_amgx])

    checked_routes = routes[1:]
    if cpu_direct_skipped:
        accuracy_passed = True
    else:
        accuracy_passed = all(
            float(route["solution_relative_l2"]) <= float(max_solution_rel_l2)
            and float(route["electrode_relative_l2"]) <= float(max_electrode_rel_l2)
            for route in checked_routes
        )
    residual_checked_routes = [
        route
        for route in routes
        if isinstance(route.get("true_relative_residual_max"), (int, float))
    ]
    residual_passed = all(
        float(route["true_relative_residual_max"]) <= float(max_true_residual)
        for route in residual_checked_routes
    )
    comparison_passed = all(
        float(route["solution_relative_l2"]) <= float(max_solution_rel_l2)
        and float(route["electrode_relative_l2"]) <= float(max_electrode_rel_l2)
        for route in checked_routes
    )
    passed = bool(accuracy_passed and residual_passed)
    return {
        "case": f"{n_elec}e_ref{refinement}",
        "n_elec": int(n_elec),
        "refinement": int(refinement),
        "mesh": metadata.get("mesh", {}),
        "n_dofs": metadata.get("n_dofs"),
        "n_patterns": metadata.get("n_patterns"),
        "matrix": metadata.get("matrix", {}),
        "gauge": metadata.get("gauge"),
        "cpu_direct_skipped": cpu_direct_skipped,
        "background": metadata.get("background"),
        "contact_impedance": metadata.get("contact_impedance"),
        "routes": routes,
        "accuracy_pass": accuracy_passed,
        "residual_pass": residual_passed,
        "comparison_pass": comparison_passed,
        "pass": passed,
    }


def _speedup(route: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    base_seconds = baseline.get("solve_seconds")
    route_seconds = route.get("solve_seconds")
    if not isinstance(base_seconds, (int, float)) or not isinstance(
        route_seconds, (int, float)
    ):
        return None
    if route_seconds <= 0:
        return None
    return float(base_seconds) / float(route_seconds)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Complex route speed and accuracy comparison",
        "",
        f"- generated_at: `{payload['generated_at']}`",
        f"- complex_shell: `{payload['complex_shell']}`",
        f"- amgx_shell: `{payload['amgx_shell']}`",
        f"- cpu_direct_mode: `{payload.get('cpu_direct_mode', 'sparse-and-dense')}`",
        f"- reference truth: `{payload.get('reference_truth', 'cpu sparse LU direct on the exported complex CEM matrix/RHS')}`",
        f"- max_solution_rel_l2: `{payload['max_solution_rel_l2']}`",
        f"- max_electrode_rel_l2: `{payload['max_electrode_rel_l2']}`",
        f"- max_true_residual: `{payload.get('max_true_residual', 1.0e-6)}`",
        "",
        "| case | route | family | solve s | speedup vs baseline | true residual max | solution relL2 | electrode relL2 | notes |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in payload["cases"]:
        routes = case.get("routes", [])
        baseline = routes[0] if routes else {}
        for route in routes:
            speedup = _speedup(route, baseline)
            solver = route.get("solver") or {}
            iterations = solver.get("iterations_per_rhs") or []
            notes = ""
            if iterations:
                notes = f"max_it={max(iterations)}"
            elif str(route.get("family", "")).startswith("native_complex_cuda"):
                diagnostics = route.get("backend_diagnostics") or {}
                fallback_reason = diagnostics.get("fallback_reason")
                if fallback_reason:
                    notes = f"fallback={fallback_reason}"
            lines.append(
                "| {case} | {route} | {family} | {seconds:.6g} | {speedup} | {resid} | {sol:.6g} | {elec:.6g} | {notes} |".format(
                    case=case["case"],
                    route=route["name"],
                    family=route["family"],
                    seconds=float(route["solve_seconds"])
                    if isinstance(route.get("solve_seconds"), (int, float))
                    else float("nan"),
                    speedup=f"{speedup:.3g}x" if speedup is not None else "",
                    resid="{:.6g}".format(route["true_relative_residual_max"])
                    if isinstance(route.get("true_relative_residual_max"), (int, float))
                    else "",
                    sol=float(route["solution_relative_l2"]),
                    elec=float(route["electrode_relative_l2"]),
                    notes=notes,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="8:1,16:1,16:2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/complex_route_speed_accuracy_compare"),
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
    parser.add_argument(
        "--cpu-direct-mode",
        choices=["sparse-and-dense", "sparse", "none"],
        default="sparse-and-dense",
    )
    parser.add_argument("--dense-direct-max-dofs", type=int, default=1500)
    parser.add_argument("--max-true-residual", type=float, default=1.0e-6)
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
            "--cpu-direct-mode",
            str(args.cpu_direct_mode),
            "--dense-direct-max-dofs",
            str(args.dense_direct_max_dofs),
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
                    max_true_residual=float(args.max_true_residual),
                )
            )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complex_shell": str(args.complex_shell),
        "amgx_shell": str(args.amgx_shell),
        "reference_solver_preset": str(args.reference_solver_preset),
        "amgx_profile": str(args.amgx_profile),
        "cpu_direct_mode": str(args.cpu_direct_mode),
        "reference_truth": (
            "native complex CUDA runtime for route-vs-route deltas; true residual gates correctness"
            if str(args.cpu_direct_mode) == "none"
            else "cpu sparse LU direct on the exported complex CEM matrix/RHS"
        ),
        "max_true_residual": float(args.max_true_residual),
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
