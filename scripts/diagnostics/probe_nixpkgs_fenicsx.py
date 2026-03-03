#!/usr/bin/env python3
"""Probe nearby nixpkgs revisions for the known Darwin FEniCSx upstream failure.

The script tests one upstream DOLFINx pytest target across a small commit window
around the currently locked nixpkgs revision and records reproducible logs.

Typical usage from repo root:
  python scripts/diagnostics/probe_nixpkgs_fenicsx.py --window 2 --update-lock
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


NETWORK_ERROR_SNIPPETS = (
    "could not resolve host",
    "failed to connect",
    "connection timed out",
    "operation timed out",
    "temporary failure in name resolution",
    "unable to access",
    "http 429",
    "http 5",
    "unable to download",
    "download failed",
    "tls",
)


@dataclass
class ProbeResult:
    order: int
    revision: str
    returncode: int
    status: str
    log_path: str
    proxy_retry_used: bool


def _cmd_str(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _network_like_failure(output: str) -> bool:
    low = output.lower()
    return any(snippet in low for snippet in NETWORK_ERROR_SNIPPETS)


def _proxy_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env["HTTP_PROXY"] = "http://127.0.0.1:7897"
    env["HTTPS_PROXY"] = "http://127.0.0.1:7897"
    env["ALL_PROXY"] = "socks5://127.0.0.1:7897"
    no_proxy = env.get("NO_PROXY", "")
    extra = "localhost,127.0.0.1,::1"
    env["NO_PROXY"] = f"{no_proxy},{extra}" if no_proxy else extra
    return env


def run_command(
    cmd: List[str],
    *,
    cwd: Path,
    allow_proxy_retry: bool,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], bool, subprocess.CompletedProcess[str] | None]:
    """Run command and retry once with proxy env if it failed due to network issues."""
    first = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if first.returncode == 0 or not allow_proxy_retry:
        return first, False, None

    combined = f"{first.stdout}\n{first.stderr}"
    if not _network_like_failure(combined):
        return first, False, None

    second = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_proxy_env(env),
        text=True,
        capture_output=True,
        check=False,
    )
    return second, True, first


def load_locked_nixpkgs_rev(flake_lock: Path) -> str:
    payload = json.loads(flake_lock.read_text())
    return payload["nodes"]["nixpkgs"]["locked"]["rev"]


def ensure_nixpkgs_clone(
    *,
    clone_dir: Path,
    branch: str,
    depth: int,
    repo_url: str,
    cwd: Path,
) -> None:
    if not clone_dir.exists():
        clone_cmd = [
            "git",
            "clone",
            "--filter=blob:none",
            "--single-branch",
            "--branch",
            branch,
            "--depth",
            str(depth),
            repo_url,
            str(clone_dir),
        ]
        result, used_proxy, first = run_command(
            clone_cmd,
            cwd=cwd,
            allow_proxy_retry=True,
        )
        if result.returncode != 0:
            if used_proxy and first is not None:
                sys.stderr.write(
                    "[probe] git clone failed (direct then proxy retry).\n"
                    f"[probe] direct rc={first.returncode}\n{first.stderr}\n"
                    f"[probe] proxy rc={result.returncode}\n{result.stderr}\n"
                )
            else:
                sys.stderr.write(
                    f"[probe] git clone failed rc={result.returncode}\n{result.stderr}\n"
                )
            raise SystemExit(2)
        return

    fetch_cmd = [
        "git",
        "-C",
        str(clone_dir),
        "fetch",
        "--depth",
        str(depth),
        "origin",
        branch,
    ]
    result, used_proxy, first = run_command(fetch_cmd, cwd=cwd, allow_proxy_retry=True)
    if result.returncode != 0:
        if used_proxy and first is not None:
            sys.stderr.write(
                "[probe] git fetch failed (direct then proxy retry).\n"
                f"[probe] direct rc={first.returncode}\n{first.stderr}\n"
                f"[probe] proxy rc={result.returncode}\n{result.stderr}\n"
            )
        else:
            sys.stderr.write(f"[probe] git fetch failed rc={result.returncode}\n{result.stderr}\n")
        raise SystemExit(2)


def ensure_commit_available(clone_dir: Path, branch: str, revision: str, cwd: Path) -> None:
    present_cmd = ["git", "-C", str(clone_dir), "cat-file", "-e", f"{revision}^{{commit}}"]
    probe = subprocess.run(present_cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    deepen_step = 200
    total_steps = 0

    while probe.returncode != 0:
        total_steps += 1
        if total_steps > 20:
            raise SystemExit(
                f"[probe] Could not find revision {revision} in {branch} after repeated deepen fetches."
            )

        deepen_cmd = [
            "git",
            "-C",
            str(clone_dir),
            "fetch",
            "--deepen",
            str(deepen_step),
            "origin",
            branch,
        ]
        result, used_proxy, first = run_command(deepen_cmd, cwd=cwd, allow_proxy_retry=True)
        if result.returncode != 0:
            if used_proxy and first is not None:
                raise SystemExit(
                    "[probe] git fetch --deepen failed (direct then proxy retry).\n"
                    f"[probe] direct rc={first.returncode}\n{first.stderr}\n"
                    f"[probe] proxy rc={result.returncode}\n{result.stderr}\n"
                )
            raise SystemExit(
                f"[probe] git fetch --deepen failed rc={result.returncode}\n{result.stderr}"
            )

        probe = subprocess.run(present_cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def git_lines(cmd: List[str], cwd: Path) -> list[str]:
    out = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def build_candidate_indices(current_idx: int, total: int, window: int) -> list[int]:
    """Return indices in order: current, older1, newer1, older2, newer2, ..."""
    indices: list[int] = [current_idx]
    for offset in range(1, window + 1):
        older = current_idx + offset
        newer = current_idx - offset
        if older < total:
            indices.append(older)
        if newer >= 0:
            indices.append(newer)
    # preserve order and uniqueness
    seen: set[int] = set()
    uniq: list[int] = []
    for i in indices:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def write_summary(results: list[ProbeResult], out_json: Path, out_tsv: Path) -> None:
    out_json.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")

    lines = ["order\trevision\treturncode\tstatus\tproxy_retry_used\tlog_path"]
    for r in results:
        lines.append(
            f"{r.order}\t{r.revision}\t{r.returncode}\t{r.status}\t{int(r.proxy_retry_used)}\t{r.log_path}"
        )
    out_tsv.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=2, help="Number of commits to probe on each side")
    parser.add_argument("--depth", type=int, default=400, help="Initial shallow clone depth")
    parser.add_argument(
        "--branch",
        default="nixos-unstable",
        help="Nixpkgs branch to walk (default: nixos-unstable)",
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/NixOS/nixpkgs.git",
        help="Nixpkgs git URL",
    )
    parser.add_argument(
        "--upstream-dir",
        default=".codex_logs/upstream_python_20260303-010524",
        help="Path to upstream dolfinx python source checkout for pytest",
    )
    parser.add_argument(
        "--test-id",
        default=(
            "test/unit/fem/test_assembler.py::"
            "TestPETScAssemblers::test_symmetry_interior_facet_assembly[mesh1]"
        ),
        help="Pytest node id to run",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum number of candidate revisions to test",
    )
    parser.add_argument(
        "--revisions",
        default="",
        help="Comma-separated explicit revision list (overrides --window walk)",
    )
    parser.add_argument(
        "--revisions-file",
        default="",
        help="Path to file with one revision per line (overrides --window walk)",
    )
    parser.add_argument(
        "--stop-on-pass",
        action="store_true",
        help="Stop probing immediately after the first passing revision",
    )
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="Update flake.lock to first passing revision",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    flake_lock = repo_root / "flake.lock"
    upstream_dir = (repo_root / args.upstream_dir).resolve()

    if not flake_lock.exists():
        raise SystemExit(f"[probe] Missing {flake_lock}")
    if not upstream_dir.exists():
        raise SystemExit(f"[probe] Upstream dir does not exist: {upstream_dir}")

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = repo_root / ".codex_logs" / "nixpkgs_probe" / now
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[probe] repo root: {repo_root}")
    print(f"[probe] run dir:   {run_dir}")

    locked_rev = load_locked_nixpkgs_rev(flake_lock)
    print(f"[probe] locked nixpkgs revision: {locked_rev}")

    clone_dir = repo_root / ".codex_logs" / "nixpkgs_probe" / f"nixpkgs-{args.branch}"
    ensure_nixpkgs_clone(
        clone_dir=clone_dir,
        branch=args.branch,
        depth=args.depth,
        repo_url=args.repo_url,
        cwd=repo_root,
    )
    ensure_commit_available(clone_dir, args.branch, locked_rev, repo_root)

    explicit_revs: list[str] = []
    if args.revisions:
        explicit_revs.extend([r.strip() for r in args.revisions.split(",") if r.strip()])
    if args.revisions_file:
        revisions_file = (repo_root / args.revisions_file).resolve()
        if not revisions_file.exists():
            raise SystemExit(f"[probe] revisions file does not exist: {revisions_file}")
        explicit_revs.extend(
            [
                line.strip()
                for line in revisions_file.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        )

    # Preserve order while deduplicating.
    if explicit_revs:
        seen: set[str] = set()
        deduped: list[str] = []
        for rev in explicit_revs:
            if rev not in seen:
                deduped.append(rev)
                seen.add(rev)
        candidate_revs = deduped[: args.max_candidates]
        for rev in candidate_revs:
            ensure_commit_available(clone_dir, args.branch, rev, repo_root)
    else:
        rev_list = git_lines(
            ["git", "-C", str(clone_dir), "rev-list", "--first-parent", f"origin/{args.branch}"],
            repo_root,
        )

        try:
            current_idx = rev_list.index(locked_rev)
        except ValueError as exc:
            raise SystemExit(
                "[probe] Locked revision not found on first-parent history of branch."
            ) from exc

        candidate_indices = build_candidate_indices(current_idx, len(rev_list), args.window)
        candidate_indices = candidate_indices[: args.max_candidates]
        candidate_revs = [rev_list[i] for i in candidate_indices]

    print("[probe] candidate revisions:")
    for i, rev in enumerate(candidate_revs, start=1):
        marker = "(locked)" if rev == locked_rev else ""
        print(f"  {i}. {rev} {marker}".rstrip())

    results: list[ProbeResult] = []
    first_passing: str | None = None

    for order, rev in enumerate(candidate_revs, start=1):
        short = rev[:12]
        log_path = run_dir / f"{order:02d}-{short}.log"

        cmd = [
            "nix",
            "--extra-experimental-features",
            "nix-command flakes",
            "develop",
            "--override-input",
            "nixpkgs",
            f"github:NixOS/nixpkgs/{rev}",
            "-c",
            "bash",
            "-lc",
            (
                f"cd {shlex.quote(str(upstream_dir))} && "
                f"python -P -m pytest -c pyproject.toml -s -vv {shlex.quote(args.test_id)}"
            ),
        ]

        print(f"[probe] ({order}/{len(candidate_revs)}) testing {rev}")
        print(f"[probe] command: {_cmd_str(cmd)}")

        proc, used_proxy, first_attempt = run_command(
            cmd,
            cwd=repo_root,
            allow_proxy_retry=True,
        )

        output = f"$ {_cmd_str(cmd)}\n\n{proc.stdout}\n{proc.stderr}"
        if used_proxy and first_attempt is not None:
            output = (
                f"$ {_cmd_str(cmd)}\n\n"
                f"# direct-attempt-rc: {first_attempt.returncode}\n"
                f"# direct-attempt-output\n{first_attempt.stdout}\n{first_attempt.stderr}\n"
                f"# proxy-retry-rc: {proc.returncode}\n"
                f"# proxy-retry-output\n{proc.stdout}\n{proc.stderr}"
            )

        log_path.write_text(output)

        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append(
            ProbeResult(
                order=order,
                revision=rev,
                returncode=proc.returncode,
                status=status,
                log_path=str(log_path),
                proxy_retry_used=used_proxy,
            )
        )

        print(f"[probe] result: {status} (rc={proc.returncode}) -> {log_path}")
        if status == "PASS" and first_passing is None:
            first_passing = rev
            if args.stop_on_pass:
                print("[probe] stop-on-pass enabled, ending probe early.")
                break

    summary_json = run_dir / "summary.json"
    summary_tsv = run_dir / "summary.tsv"
    write_summary(results, summary_json, summary_tsv)

    print("\n[probe] summary")
    for r in results:
        print(
            f"  {r.order}. {r.revision[:12]} {r.status:<4} "
            f"rc={r.returncode:<3} proxy_retry={int(r.proxy_retry_used)}"
        )
    print(f"[probe] summary json: {summary_json}")
    print(f"[probe] summary tsv:  {summary_tsv}")

    if first_passing is None:
        print("[probe] No passing revision found in the tested window.")
        return 1

    print(f"[probe] First passing revision: {first_passing}")

    if args.update_lock:
        lock_cmd = [
            "nix",
            "--extra-experimental-features",
            "nix-command flakes",
            "flake",
            "lock",
            "--override-input",
            "nixpkgs",
            f"github:NixOS/nixpkgs/{first_passing}",
        ]
        print(f"[probe] Updating flake.lock: {_cmd_str(lock_cmd)}")
        lock_proc, lock_proxy_retry, lock_first = run_command(
            lock_cmd,
            cwd=repo_root,
            allow_proxy_retry=True,
        )

        lock_log = run_dir / "lock-update.log"
        if lock_proxy_retry and lock_first is not None:
            lock_output = (
                f"$ {_cmd_str(lock_cmd)}\n\n"
                f"# direct-attempt-rc: {lock_first.returncode}\n"
                f"# direct-attempt-output\n{lock_first.stdout}\n{lock_first.stderr}\n"
                f"# proxy-retry-rc: {lock_proc.returncode}\n"
                f"# proxy-retry-output\n{lock_proc.stdout}\n{lock_proc.stderr}"
            )
        else:
            lock_output = f"$ {_cmd_str(lock_cmd)}\n\n{lock_proc.stdout}\n{lock_proc.stderr}"
        lock_log.write_text(lock_output)

        if lock_proc.returncode != 0:
            raise SystemExit(
                "[probe] Failed to update flake.lock. "
                f"See {lock_log} for details."
            )

        print(f"[probe] flake.lock updated to passing revision. Log: {lock_log}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
