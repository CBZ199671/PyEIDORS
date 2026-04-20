#!/usr/bin/env python3
"""Run recoverable pytest shards for the unit test suite.

The project-level pytest config enables coverage by default. This runner is for
focused validation gates, so every generated pytest command includes --no-cov.
Invoke the runner itself through the Nix shell, for example:

    nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --list
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
UNIT_TEST_ROOT = REPO_ROOT / "tests" / "unit"
DEFAULT_REPORT_ROOT = REPO_ROOT / "test_results" / "sharded_unit"

PYTEST_BASE_ARGS = ("pytest", "--no-cov")

SMOKE_SHARDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "fp-refactor-smoke": (
        "Focused FEniCSx/PETSc refactor smoke tests.",
        (
            "tests/unit/test_forward_solver_presets.py",
            "tests/unit/test_jacobian_linearization.py",
            "tests/unit/test_forward_petsc_helper_branches.py",
            "tests/unit/test_forward_solver_branch_suite.py",
            "tests/unit/test_forward_mat_solve_policy.py",
            "tests/unit/test_forward_vectorized_runtime.py",
            "tests/unit/test_gn_fast_linear_solver.py",
        ),
    ),
}


@dataclass(frozen=True)
class Shard:
    name: str
    description: str
    files: tuple[Path, ...]
    virtual: bool = False
    default_in_all: bool = True

    @property
    def relative_files(self) -> tuple[str, ...]:
        return tuple(str(path.relative_to(REPO_ROOT)) for path in self.files)


@dataclass(frozen=True)
class ShardRule:
    name: str
    description: str
    prefixes: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    exact: tuple[str, ...] = ()
    default_in_all: bool = True

    def matches(self, filename: str) -> bool:
        return (
            filename in self.exact
            or any(filename.startswith(prefix) for prefix in self.prefixes)
            or any(token in filename for token in self.contains)
        )


CATEGORY_RULES: tuple[ShardRule, ...] = (
    ShardRule(
        name="forward",
        description="Forward model, PETSc helper, CEM, and stimulation tests.",
        prefixes=("test_forward_",),
        exact=(
            "test_complete_eit_system.py",
            "test_current_drive_units.py",
            "test_eit_pde_extended.py",
            "test_electrode_position_y_axis.py",
            "test_pattern_manager_helper_branches.py",
            "test_simplified_eit_system.py",
        ),
    ),
    ShardRule(
        name="inverse-gn",
        description="Gauss-Newton, Jacobian, regularization, and reduced inverse tests.",
        prefixes=(
            "test_absolute_",
            "test_adjoint_",
            "test_difference_",
            "test_direct_jacobian_",
            "test_gauss_newton_",
            "test_gn_",
            "test_jacobian_",
            "test_lowrank_",
            "test_reduced_",
            "test_regularization_",
        ),
        exact=("test_inexact_controller.py",),
    ),
    ShardRule(
        name="cache",
        description="Cache stores, semantic signatures, and cache lifecycle tests.",
        prefixes=("test_cache_",),
        contains=("cache",),
    ),
    ShardRule(
        name="sparse",
        description="Sparse Bayesian, sparse MAP, and sparse optimizer tests.",
        prefixes=("test_sparse_",),
    ),
    ShardRule(
        name="mesh-femx",
        description="Mesh generation, FEniCSx helpers, and mesh pipeline tests.",
        prefixes=(
            "test_femx_",
            "test_mesh",
            "test_optimized_mesh_",
            "test_real_mesh_",
        ),
    ),
    ShardRule(
        name="perf-cuda",
        description="Performance policy, CUDA backend, and acceleration profile tests.",
        prefixes=(
            "test_benchmark_",
            "test_compare_cuda_",
            "test_cuda_",
            "test_perf_",
            "test_script_acceleration_",
            "test_script_entrypoint_",
        ),
    ),
    ShardRule(
        name="gui",
        description="Qt GUI, workstation, interop dialog, and runtime widget tests.",
        exact=(
            "test_acquisition_controller.py",
            "test_conductivity_3d_widget_runtime.py",
            "test_database_backfill_shutdown.py",
            "test_eit_app_gui_smoke.py",
            "test_eit_app_interop_environment.py",
            "test_eit_app_interop_hub.py",
            "test_eit_app_measurement_layout.py",
            "test_runtime_threads.py",
        ),
    ),
    ShardRule(
        name="hardware",
        description="Hardware abstraction, serial, relay, protocol, and device discovery tests.",
        default_in_all=False,
        exact=(
            "test_eit_app_connection_preflight.py",
            "test_eit_app_protocol_legacy.py",
            "test_eit_app_relay_transport.py",
            "test_eit_app_serial_device.py",
            "test_eit_app_serial_port_discovery.py",
            "test_eit_app_simulator.py",
            "test_eit_app_windows_serial_transport.py",
            "test_frame_io_legacy_compat.py",
        ),
    ),
    ShardRule(
        name="env-cli",
        description="Environment, CLI, scripts, workflow, and entrypoint tests.",
        prefixes=(
            "test_env_",
            "test_misc_entrypoint_",
            "test_patch_pymfem_",
            "test_recon_",
            "test_run_real_",
            "test_workflow_",
        ),
        exact=("test_unit_consistency_checks.py",),
    ),
    ShardRule(
        name="coverage-gap",
        description="Coverage harvest and branch-gap tests.",
        prefixes=("test_coverage_",),
        exact=(
            "test_gap_harvest_easy_branches.py",
            "test_helper_branch_mix.py",
            "test_object_signature_branch_coverage.py",
            "test_runtime_helpers_coverage.py",
            "test_small_gap_modules.py",
        ),
    ),
    ShardRule(
        name="data-visualization",
        description="Measurement data, plotting, visualization, and interop tests.",
        prefixes=(
            "test_interop_",
            "test_measurement_",
            "test_plot_",
            "test_visualization_",
        ),
    ),
)


def discover_unit_tests(root: Path = UNIT_TEST_ROOT) -> tuple[Path, ...]:
    """Discover top-level unit test files in deterministic order."""
    return tuple(sorted(root.glob("test_*.py")))


def _assign_category(filename: str) -> str:
    for rule in CATEGORY_RULES:
        if rule.matches(filename):
            return rule.name
    return "core-misc"


def build_category_shards(
    files: Sequence[Path] | None = None,
    *,
    include_optional: bool = True,
) -> tuple[Shard, ...]:
    """Build exclusive category shards that cover all unit test files once."""
    unit_files = tuple(files) if files is not None else discover_unit_tests()
    grouped: dict[str, list[Path]] = {rule.name: [] for rule in CATEGORY_RULES}
    grouped["core-misc"] = []

    for path in unit_files:
        grouped[_assign_category(path.name)].append(path)

    descriptions = {rule.name: rule.description for rule in CATEGORY_RULES}
    descriptions["core-misc"] = "Core, numeric, and miscellaneous tests."
    default_flags = {rule.name: rule.default_in_all for rule in CATEGORY_RULES}
    default_flags["core-misc"] = True

    shards: list[Shard] = []
    for name, paths in grouped.items():
        if not include_optional and not default_flags[name]:
            continue
        if paths:
            shards.append(
                Shard(
                    name=name,
                    description=descriptions[name],
                    files=tuple(sorted(paths)),
                    default_in_all=default_flags[name],
                )
            )
    return tuple(shards)


def build_smoke_shards() -> tuple[Shard, ...]:
    """Build virtual focused shards that may overlap category shards."""
    shards: list[Shard] = []
    for name, (description, rel_files) in SMOKE_SHARDS.items():
        files = tuple(REPO_ROOT / rel for rel in rel_files)
        shards.append(Shard(name=name, description=description, files=files, virtual=True))
    return tuple(shards)


def build_all_shards(
    include_smoke: bool = True,
    *,
    include_optional: bool = True,
) -> tuple[Shard, ...]:
    shards = list(build_category_shards(include_optional=include_optional))
    if include_smoke:
        shards.extend(build_smoke_shards())
    return tuple(shards)


def select_shards(
    names: Sequence[str],
    *,
    include_smoke: bool = True,
    category_only: bool = False,
) -> tuple[Shard, ...]:
    available = (
        build_category_shards()
        if category_only
        else build_all_shards(include_smoke, include_optional=True)
    )
    by_name = {shard.name: shard for shard in available}
    missing = [name for name in names if name not in by_name]
    if missing:
        known = ", ".join(sorted(by_name))
        raise ValueError(f"unknown shard(s): {', '.join(missing)}; known shards: {known}")
    return tuple(by_name[name] for name in names)


def emitted_shell_command(shard: Shard, extra_pytest_args: Sequence[str] = ()) -> list[str]:
    return [
        "nix",
        "develop",
        "-c",
        "uv",
        "run",
        *PYTEST_BASE_ARGS,
        *shard.relative_files,
        *extra_pytest_args,
        "-q",
    ]


def format_shell_command(command: Sequence[str]) -> str:
    """Render a subprocess argv list as a shell-safe command line."""
    return shlex.join(command)


def current_python_command(shard: Shard, extra_pytest_args: Sequence[str] = ()) -> list[str]:
    return [
        sys.executable,
        "-m",
        *PYTEST_BASE_ARGS,
        *shard.relative_files,
        *extra_pytest_args,
        "-q",
    ]


def _command_for_runner(
    shard: Shard,
    runner: str,
    extra_pytest_args: Sequence[str],
) -> list[str]:
    if runner == "current-python":
        return current_python_command(shard, extra_pytest_args)
    if runner == "nix":
        return emitted_shell_command(shard, extra_pytest_args)
    if runner == "uv":
        return ["uv", "run", *PYTEST_BASE_ARGS, *shard.relative_files, *extra_pytest_args, "-q"]
    raise ValueError(f"unsupported runner: {runner}")


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _normalize_report_dir(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_to_repo(path: Path) -> str:
    absolute = path if path.is_absolute() else REPO_ROOT / path
    return str(absolute.relative_to(REPO_ROOT))


def run_shard(
    shard: Shard,
    *,
    runner: str,
    timeout: int,
    report_dir: Path,
    extra_pytest_args: Sequence[str] = (),
) -> dict[str, object]:
    """Run one shard and persist stdout/stderr logs."""
    command = _command_for_runner(shard, runner, extra_pytest_args)
    report_dir = _normalize_report_dir(report_dir)
    started = time.perf_counter()
    stdout_log = report_dir / f"{shard.name}.stdout.log"
    stderr_log = report_dir / f"{shard.name}.stderr.log"

    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        _write_text(stdout_log, completed.stdout)
        _write_text(stderr_log, completed.stderr)
        status = "passed" if completed.returncode == 0 else "failed"
        return {
            "name": shard.name,
            "status": status,
            "returncode": completed.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "file_count": len(shard.files),
            "virtual": shard.virtual,
            "command": command,
            "stdout_log": _relative_to_repo(stdout_log),
            "stderr_log": _relative_to_repo(stderr_log),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        _write_text(stdout_log, stdout)
        _write_text(stderr_log, stderr)
        return {
            "name": shard.name,
            "status": "timeout",
            "returncode": None,
            "elapsed_seconds": round(elapsed, 3),
            "file_count": len(shard.files),
            "virtual": shard.virtual,
            "command": command,
            "stdout_log": _relative_to_repo(stdout_log),
            "stderr_log": _relative_to_repo(stderr_log),
            "timeout_seconds": timeout,
        }


def render_shard_table(shards: Iterable[Shard]) -> str:
    rows = [
        "Shard | Files | Virtual | Default `--all` | Description",
        "--- | ---: | --- | --- | ---",
    ]
    for shard in shards:
        rows.append(
            f"{shard.name} | {len(shard.files)} | {str(shard.virtual).lower()} | "
            f"{str(shard.default_in_all).lower()} | "
            f"{shard.description}"
        )
    return "\n".join(rows)


def _json_shards(shards: Sequence[Shard]) -> str:
    payload = [
        {
            "name": shard.name,
            "description": shard.description,
            "file_count": len(shard.files),
            "virtual": shard.virtual,
            "default_in_all": shard.default_in_all,
            "files": shard.relative_files,
            "command": emitted_shell_command(shard),
        }
        for shard in shards
    ]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available shards.")
    parser.add_argument("--json", action="store_true", help="Use JSON for --list output.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument("--run", action="store_true", help="Run selected shards.")
    parser.add_argument("--all", action="store_true", help="Select all category shards.")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include virtual smoke shards when --all is used.",
    )
    parser.add_argument(
        "--include-hardware",
        action="store_true",
        help="Include the opt-in hardware shard in broad --run/--dry-run/--all selections.",
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        help="Shard name to select. May be passed multiple times.",
    )
    parser.add_argument(
        "--runner",
        choices=("current-python", "uv", "nix"),
        default="current-python",
        help="Subprocess runner used with --run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-shard timeout in seconds.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for shard logs and summary JSON.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to pytest. May be passed multiple times.",
    )
    return parser.parse_args(argv)


def _selected_shards(args: argparse.Namespace) -> tuple[Shard, ...]:
    if args.all:
        shards = list(build_category_shards(include_optional=args.include_hardware))
        if args.include_smoke:
            shards.extend(build_smoke_shards())
        return tuple(shards)
    if args.shard:
        return select_shards(args.shard, include_smoke=True)
    if args.list or (not args.run and not args.dry_run):
        return build_all_shards(include_smoke=True, include_optional=True)
    return build_all_shards(include_smoke=True, include_optional=args.include_hardware)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    shards = _selected_shards(args)

    if args.list or (not args.run and not args.dry_run):
        print(_json_shards(shards) if args.json else render_shard_table(shards))
        return 0

    if args.dry_run:
        for shard in shards:
            print(format_shell_command(emitted_shell_command(shard, args.pytest_arg)))
        return 0

    report_dir = _normalize_report_dir(args.report_dir or (DEFAULT_REPORT_ROOT / _iso_timestamp()))
    report_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_shard(
            shard,
            runner=args.runner,
            timeout=args.timeout,
            report_dir=report_dir,
            extra_pytest_args=args.pytest_arg,
        )
        for shard in shards
    ]
    summary = {
        "generated_at": _iso_timestamp(),
        "repo_root": str(REPO_ROOT),
        "runner": args.runner,
        "timeout_seconds": args.timeout,
        "results": results,
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for result in results:
        print(
            f"{result['name']}: {result['status']} "
            f"({result['elapsed_seconds']}s, {result['file_count']} files)"
        )
    print(f"summary: {_relative_to_repo(summary_path)}")

    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
