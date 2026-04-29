#!/usr/bin/env python3
"""T4 — forward KSP session reuse benchmark (G1 evidence).

Drives a sequence of forward solves with sigma updates under multiple
``forward_pc_refresh_policy`` regimes and records, per call, the PC setup
seconds + KSP iteration counts + session-reuse status. Aggregates to
cumulative setup seconds + iter histogram + reuse/refresh tally and emits
a HDF5 + JSON + Markdown artifact bundle proving G1 (persistent KSP across
GN-style sigma trajectory) saves cumulative PC setup time.

Cites V13 (forward KSP session reused across calls), V14 (refresh policy
``auto``/``never``/``always``/``lag``), V52 (artifact records ``env_path``),
V67 (HDF5 default for binary cache/save artifacts).

Example
-------

    python scripts/benchmarks/benchmark_forward_ksp_session_reuse.py \\
        --mesh-dim 3 --n-elec 16 --n-iter 8 \\
        --solver-preset 3d_gamg --petsc-device cpu \\
        --out-dir reports/runtime_benchmarks/forward_ksp_session_reuse_t4_$(date +%Y%m%d)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import build_eit_mesh
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.forward.process_setup_cache import clear_process_forward_setup_cache
from pyeidors.io import write_hdf5_artifact

try:
    from pyeidors.geometry.mesh3d_generator import (
        GMSH_AVAILABLE,
        create_cylinder_3d_eit_mesh,
    )
except Exception:
    GMSH_AVAILABLE = False
    create_cylinder_3d_eit_mesh = None  # type: ignore[assignment]


ALLOWED_REGIMES = ("auto", "never", "always", "lag")
DEFAULT_REGIMES = "auto,never"
SCHEMA_VERSION = 1
V_CITES = ("V13", "V14", "V52", "V67", "V80")


def _make_tagged_unit_square(*, n_elec: int = 4, refinement: int = 4):
    """Tagged 2D unit-square mesh suitable for CEM forward smoke runs."""
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, refinement, refinement)
    fdim = mesh.topology.dim - 1
    boundary_facets = dmesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    ).astype(np.int32)
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x[:, :2]

    centroids = np.zeros((boundary_facets.size, 2), dtype=np.float64)
    for idx, facet in enumerate(boundary_facets):
        centroids[idx, :] = coords[f2v.links(int(facet))].mean(axis=0)

    x = centroids[:, 0]
    y = centroids[:, 1]
    eps = 1e-10
    t = np.zeros_like(x)
    left = np.isclose(x, 0.0, atol=eps)
    top = (~left) & np.isclose(y, 1.0, atol=eps)
    right = (~left) & (~top) & np.isclose(x, 1.0, atol=eps)
    bottom = (~left) & (~top) & (~right) & np.isclose(y, 0.0, atol=eps)
    t[left] = y[left]
    t[top] = 1.0 + x[top]
    t[right] = 2.0 + (1.0 - y[right])
    t[bottom] = 3.0 + (1.0 - x[bottom])

    tags = (
        np.floor(np.clip(t, 0.0, 4.0 - eps) / (4.0 / float(n_elec))).astype(np.int32)
        + 2
    ).astype(np.int32)
    order = np.argsort(boundary_facets)
    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets[order], tags[order])
    association = {f"electrode_{idx + 1}": idx + 2 for idx in range(n_elec)}
    return build_eit_mesh(
        mesh,
        facet_tags=facet_tags,
        association_table=association,
        radius=1.0,
    )


def _build_mesh(args: argparse.Namespace):
    if int(args.mesh_dim) == 2:
        return _make_tagged_unit_square(
            n_elec=int(args.n_elec),
            refinement=int(args.mesh_refinement),
        )
    if not GMSH_AVAILABLE or create_cylinder_3d_eit_mesh is None:
        raise RuntimeError(
            "3D mesh requires the gmsh python bindings. "
            "Run `nix develop` to enter the supported FEniCSx/gmsh environment."
        )
    out_dir = Path(args.mesh_cache_dir) if args.mesh_cache_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    return create_cylinder_3d_eit_mesh(
        n_elec=int(args.n_elec),
        radius=float(args.mesh_radius),
        height=float(args.mesh_height),
        refinement=int(args.mesh_refinement),
        electrode_coverage=0.5,
        output_dir=str(out_dir) if out_dir is not None else None,
        mesh_name=str(args.mesh_name) if args.mesh_name else None,
    )


def _build_forward_model(eit_mesh, args: argparse.Namespace, *, regime: str):
    pattern = PatternConfig(
        n_elec=int(args.n_elec),
        stim_pattern=str(args.stim_pattern),
        meas_pattern=str(args.meas_pattern),
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    backend_config: dict[str, Any] = {
        "solver_preset": str(args.solver_preset),
        "ksp_type": str(args.ksp_type),
        "pc_type": str(args.pc_type),
        "petsc_device": str(args.petsc_device),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_it": int(args.max_it),
        "forward_pc_refresh_policy": str(regime),
        "forward_pc_refresh_iter_threshold": int(args.refresh_iter_threshold),
        "forward_pc_refresh_lag": int(args.refresh_lag),
        "reuse_preconditioner": True,
    }
    return EITForwardModel(
        n_elec=int(args.n_elec),
        pattern_config=pattern,
        z=np.full(int(args.n_elec), float(args.contact_impedance), dtype=np.float64),
        mesh=eit_mesh,
        linear_backend="petsc",
        backend_config=backend_config,
    )


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _sigma_sequence_hash(sigma_sequence: np.ndarray) -> str:
    return _array_sha256(np.ascontiguousarray(sigma_sequence, dtype=np.float64))


def _generate_sigma_sequence(
    *,
    n_iter: int,
    n_dof: int,
    base_conductivity: float,
    sigma_noise_scale: float,
    sigma_floor: float,
    rng_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    base_sigma = np.full(n_dof, float(base_conductivity), dtype=np.float64)
    perturbation = rng.standard_normal((n_iter, n_dof))
    sequence = base_sigma[np.newaxis, :] * (
        1.0 + float(sigma_noise_scale) * perturbation
    )
    sequence = np.maximum(sequence, float(sigma_floor))
    return np.ascontiguousarray(sequence, dtype=np.float64)


def _mesh_artifact_provenance(
    eit_mesh,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mesh = eit_mesh.mesh
    topology = mesh.topology
    tdim = int(topology.dim)
    fdim = max(0, tdim - 1)
    num_facets = 0
    try:
        topology.create_connectivity(fdim, 0)
        facet_map = topology.index_map(fdim)
        num_facets = int(facet_map.size_local if facet_map is not None else 0)
    except Exception:
        num_facets = 0
    try:
        coords_hash = _array_sha256(eit_mesh.coordinates())
    except Exception:
        coords_hash = ""
    try:
        cells_hash = _array_sha256(eit_mesh.cells())
    except Exception:
        cells_hash = ""
    content_hash_payload = json.dumps(
        {
            "coords_sha256": coords_hash,
            "cells_sha256": cells_hash,
            "num_vertices": int(eit_mesh.num_vertices()),
            "num_cells": int(eit_mesh.num_cells()),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "requested_dim": int(args.mesh_dim),
        "topology_dim": tdim,
        "geometry_dim": int(mesh.geometry.dim),
        "n_elec": int(args.n_elec),
        "mesh_refinement": int(args.mesh_refinement),
        "mesh_radius": float(args.mesh_radius),
        "mesh_height": float(args.mesh_height),
        "mesh_cache_dir": str(args.mesh_cache_dir or ""),
        "mesh_name": str(args.mesh_name or ""),
        "mesh_file": str(getattr(eit_mesh, "mesh_file", "") or ""),
        "mesh_family": str(getattr(eit_mesh, "mesh_family", "") or ""),
        "geometry_version": str(getattr(eit_mesh, "geometry_version", "") or ""),
        "generator_revision": str(getattr(eit_mesh, "generator_revision", "") or ""),
        "num_vertices": int(eit_mesh.num_vertices()),
        "num_cells": int(eit_mesh.num_cells()),
        "num_facets": num_facets,
        "coordinates_sha256": coords_hash,
        "cells_sha256": cells_hash,
        "mesh_content_hash": hashlib.sha256(content_hash_payload).hexdigest(),
    }


def _safe_env_path() -> str:
    usr_env = Path("/usr/bin/env")
    if usr_env.is_file():
        return str(usr_env)
    return shutil.which("env") or ""


def _run_regime(
    eit_mesh,
    args: argparse.Namespace,
    *,
    regime: str,
    rng_seed: int,
    sigma_sequence: np.ndarray | None = None,
) -> dict[str, Any]:
    fwd = _build_forward_model(eit_mesh, args, regime=regime)
    n_dof = fem.Function(fwd.V_sigma).x.array.size
    n_iter = int(args.n_iter)
    if sigma_sequence is None:
        sigma_sequence = _generate_sigma_sequence(
            n_iter=n_iter,
            n_dof=n_dof,
            base_conductivity=float(args.base_conductivity),
            sigma_noise_scale=float(args.sigma_noise_scale),
            sigma_floor=float(args.sigma_floor),
            rng_seed=int(rng_seed),
        )
    else:
        sigma_sequence = np.ascontiguousarray(sigma_sequence, dtype=np.float64)
        expected_shape = (n_iter, n_dof)
        if sigma_sequence.shape != expected_shape:
            raise ValueError(
                f"sigma_sequence shape {sigma_sequence.shape} does not match "
                f"{expected_shape} for regime {regime!r}."
            )
        if not np.isfinite(sigma_sequence).all():
            raise FloatingPointError("sigma_sequence contains non-finite values.")
        if np.any(sigma_sequence < float(args.sigma_floor)):
            raise ValueError("sigma_sequence violates sigma_floor.")
    sigma_hash = _sigma_sequence_hash(sigma_sequence)

    iter_max = np.zeros(n_iter, dtype=np.int64)
    iter_total = np.zeros(n_iter, dtype=np.int64)
    iter_mean = np.zeros(n_iter, dtype=np.float64)
    setup_seconds = np.zeros(n_iter, dtype=np.float64)
    solve_seconds = np.zeros(n_iter, dtype=np.float64)
    wall_seconds = np.zeros(n_iter, dtype=np.float64)
    session_reused = np.zeros(n_iter, dtype=bool)
    refresh_triggered = np.zeros(n_iter, dtype=bool)
    pc_session_total_setups = np.zeros(n_iter, dtype=np.int64)
    pc_session_solves = np.zeros(n_iter, dtype=np.int64)
    refresh_reasons: list[str] = []
    measurement_norm = np.zeros(n_iter, dtype=np.float64)
    refresh_policy_observed = ""
    n_rhs = 0

    for k in range(n_iter):
        sigma_k = sigma_sequence[k]
        wall_t0 = time.perf_counter()
        data, _U = fwd.fwd_solve(EITImage(elem_data=sigma_k, fwd_model=fwd))
        wall_seconds[k] = float(time.perf_counter() - wall_t0)
        diag = dict(fwd.get_backend_diagnostics())
        iter_arr_raw = diag.get("forward_ksp_iterations_per_rhs") or []
        iter_arr = np.asarray(iter_arr_raw, dtype=np.int64).reshape(-1)
        if iter_arr.size:
            iter_max[k] = int(iter_arr.max())
            iter_total[k] = int(iter_arr.sum())
            iter_mean[k] = float(iter_arr.mean())
            n_rhs = max(n_rhs, int(iter_arr.size))
        setup_seconds[k] = float(diag.get("forward_setup_seconds") or 0.0)
        solve_seconds[k] = float(diag.get("forward_solve_seconds") or 0.0)
        session_reused[k] = bool(diag.get("forward_pc_session_reused") or False)
        refresh_triggered[k] = bool(diag.get("forward_pc_refresh_triggered") or False)
        pc_session_total_setups[k] = int(
            diag.get("forward_pc_session_total_setups") or 0
        )
        pc_session_solves[k] = int(diag.get("forward_pc_session_solves") or 0)
        raw_reason = diag.get("forward_pc_refresh_reason")
        refresh_reasons.append(str(raw_reason) if raw_reason else "reused")
        measurement_norm[k] = float(np.linalg.norm(np.asarray(data.meas)))
        refresh_policy_observed = str(diag.get("forward_pc_refresh_policy") or regime)

    reasons_hist: dict[str, int] = {}
    for reason in refresh_reasons:
        reasons_hist[reason] = reasons_hist.get(reason, 0) + 1

    cumulative_setup = float(setup_seconds.sum())
    first_setup = float(setup_seconds[0]) if setup_seconds.size else 0.0
    subsequent = setup_seconds[1:] if setup_seconds.size > 1 else np.empty(0)
    summary = {
        "regime": str(regime),
        "regime_observed": refresh_policy_observed,
        "n_calls": n_iter,
        "n_reused": int(session_reused.sum()),
        "n_refresh": int(refresh_triggered.sum()),
        "sigma_sequence_hash": sigma_hash,
        "sigma_sequence_shape": [int(v) for v in sigma_sequence.shape],
        "cumulative_setup_seconds": cumulative_setup,
        "first_call_setup_seconds": first_setup,
        "subsequent_setup_seconds_total": float(subsequent.sum()),
        "subsequent_setup_seconds_mean": (
            float(subsequent.mean()) if subsequent.size else 0.0
        ),
        "wall_seconds_total": float(wall_seconds.sum()),
        "solve_seconds_total": float(solve_seconds.sum()),
        "iter_max_mean": float(iter_max.mean()) if iter_max.size else 0.0,
        "iter_max_p50": _percentile(iter_max, 50.0),
        "iter_max_p95": _percentile(iter_max, 95.0),
        "iter_total_sum": int(iter_total.sum()),
        "n_rhs_per_call": int(n_rhs),
        "refresh_reasons": reasons_hist,
        "final_total_setups": int(pc_session_total_setups[-1])
        if pc_session_total_setups.size
        else 0,
    }
    arrays = {
        f"regime_{regime}_iter_max_per_call": iter_max,
        f"regime_{regime}_iter_total_per_call": iter_total,
        f"regime_{regime}_iter_mean_per_call": iter_mean,
        f"regime_{regime}_setup_seconds": setup_seconds,
        f"regime_{regime}_solve_seconds": solve_seconds,
        f"regime_{regime}_wall_seconds": wall_seconds,
        f"regime_{regime}_session_reused": session_reused,
        f"regime_{regime}_refresh_triggered": refresh_triggered,
        f"regime_{regime}_pc_session_total_setups": pc_session_total_setups,
        f"regime_{regime}_pc_session_solves": pc_session_solves,
        f"regime_{regime}_measurement_norm": measurement_norm,
    }
    return {
        "summary": summary,
        "arrays": arrays,
        "refresh_reasons_list": refresh_reasons,
        "sigma_sequence": sigma_sequence,
    }


def _format_md(per_regime: dict[str, dict[str, Any]], info: dict[str, Any]) -> str:
    header = (
        "| regime | calls | reused | refresh | cum_setup_s | first_setup_s | "
        "subseq_mean_s | iter_max_mean | iter_max_p95 | total_setups |"
    )
    sep = (
        "|--------|------:|-------:|--------:|------------:|--------------:|"
        "--------------:|--------------:|-------------:|-------------:|"
    )
    rows = []
    for regime in info["regimes"]:
        s = per_regime[regime]["summary"]
        rows.append(
            f"| {regime} | {s['n_calls']} | {s['n_reused']} | {s['n_refresh']} | "
            f"{s['cumulative_setup_seconds']:.6f} | "
            f"{s['first_call_setup_seconds']:.6f} | "
            f"{s['subsequent_setup_seconds_mean']:.6f} | "
            f"{s['iter_max_mean']:.2f} | {s['iter_max_p95']:.1f} | "
            f"{s['final_total_setups']} |"
        )
    lines = [
        "# T4 — Forward KSP session reuse benchmark (G1 evidence)",
        "",
        f"- generated: `{info['generated_at']}`",
        f"- env_path: `{info['env_path']}`",
        f"- mesh_dim: {info['mesh_dim']}, n_elec: {info['n_elec']}, "
        f"n_iter: {info['n_iter']}, sigma_noise_scale: {info['sigma_noise_scale']}",
        f"- solver_preset: `{info['solver_preset']}`, "
        f"ksp_type: `{info['ksp_type']}`, pc_type: `{info['pc_type']}`, "
        f"petsc_device: `{info['petsc_device']}`",
        "",
        header,
        sep,
        *rows,
    ]
    if "auto" in per_regime and "never" in per_regime:
        auto_setup = per_regime["auto"]["summary"]["cumulative_setup_seconds"]
        never_setup = per_regime["never"]["summary"]["cumulative_setup_seconds"]
        saved = never_setup - auto_setup
        ratio = (auto_setup / never_setup) if never_setup > 0 else float("nan")
        lines.extend(
            [
                "",
                f"**G1 cumulative setup saved (never − auto)**: `{saved:.6f}s`",
                f"**warm/cold setup ratio (auto / never)**: `{ratio:.4f}`",
            ]
        )
    refresh_block = []
    for regime in info["regimes"]:
        s = per_regime[regime]["summary"]
        if s["refresh_reasons"]:
            refresh_block.append(f"- `{regime}`: {json.dumps(s['refresh_reasons'])}")
    if refresh_block:
        lines.extend(["", "## refresh_reasons", *refresh_block])
    lines.extend(["", f"V cites: {', '.join(info['v_cites'])}", ""])
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T4 forward KSP session reuse benchmark (G1)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="output directory for HDF5 + JSON + Markdown artifact",
    )
    parser.add_argument("--mesh-dim", type=int, default=3, choices=(2, 3))
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--mesh-refinement", type=int, default=2)
    parser.add_argument("--mesh-radius", type=float, default=0.1)
    parser.add_argument("--mesh-height", type=float, default=0.08)
    parser.add_argument("--mesh-cache-dir", type=str, default="")
    parser.add_argument("--mesh-name", type=str, default="")
    parser.add_argument("--regimes", type=str, default=DEFAULT_REGIMES)
    parser.add_argument("--solver-preset", type=str, default="auto")
    parser.add_argument("--ksp-type", type=str, default="auto")
    parser.add_argument("--pc-type", type=str, default="auto")
    parser.add_argument("--petsc-device", type=str, default="cpu")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--max-it", type=int, default=1000)
    parser.add_argument("--refresh-iter-threshold", type=int, default=0)
    parser.add_argument("--refresh-lag", type=int, default=0)
    parser.add_argument("--sigma-noise-scale", type=float, default=0.05)
    parser.add_argument("--sigma-floor", type=float, default=1e-3)
    parser.add_argument("--base-conductivity", type=float, default=1.0)
    parser.add_argument("--contact-impedance", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stim-pattern", type=str, default="{ad}")
    parser.add_argument("--meas-pattern", type=str, default="{ad}")
    args = parser.parse_args(argv)
    return args


def _normalize_regimes(spec: str) -> list[str]:
    regimes = [token.strip() for token in str(spec).split(",") if token.strip()]
    if not regimes:
        raise ValueError("--regimes must specify at least one regime")
    invalid = [r for r in regimes if r not in ALLOWED_REGIMES]
    if invalid:
        raise ValueError(f"unknown regimes {invalid!r}; allowed: {ALLOWED_REGIMES}")
    seen: set[str] = set()
    deduped: list[str] = []
    for r in regimes:
        if r in seen:
            continue
        seen.add(r)
        deduped.append(r)
    return deduped


def main(argv: list[str] | None = None) -> int:
    command_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes = _normalize_regimes(args.regimes)

    eit_mesh = _build_mesh(args)

    per_regime: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    sigma_sequence: np.ndarray | None = None
    sigma_hash = ""
    sigma_shape: list[int] = []
    for k_regime, regime in enumerate(regimes):
        clear_process_forward_setup_cache()
        result = _run_regime(
            eit_mesh,
            args,
            regime=regime,
            rng_seed=int(args.seed),
            sigma_sequence=sigma_sequence,
        )
        if sigma_sequence is None:
            sigma_sequence = np.ascontiguousarray(result["sigma_sequence"])
            sigma_hash = str(result["summary"]["sigma_sequence_hash"])
            sigma_shape = [int(v) for v in sigma_sequence.shape]
        elif str(result["summary"]["sigma_sequence_hash"]) != sigma_hash:
            raise RuntimeError("regimes did not share an identical sigma sequence.")
        per_regime[regime] = result
        arrays.update(result["arrays"])

    mesh_provenance = _mesh_artifact_provenance(eit_mesh, args)
    info: dict[str, Any] = {
        "task": "T4",
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env_path": _safe_env_path(),
        "command_argv": command_argv,
        "mesh": mesh_provenance,
        "mesh_dim": int(args.mesh_dim),
        "n_elec": int(args.n_elec),
        "n_iter": int(args.n_iter),
        "sigma_sequence_hash": sigma_hash,
        "sigma_sequence_shape": sigma_shape,
        "sigma_noise_scale": float(args.sigma_noise_scale),
        "sigma_floor": float(args.sigma_floor),
        "base_conductivity": float(args.base_conductivity),
        "contact_impedance": float(args.contact_impedance),
        "solver_preset": str(args.solver_preset),
        "ksp_type": str(args.ksp_type),
        "pc_type": str(args.pc_type),
        "petsc_device": str(args.petsc_device),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_it": int(args.max_it),
        "refresh_iter_threshold": int(args.refresh_iter_threshold),
        "refresh_lag": int(args.refresh_lag),
        "regimes": regimes,
        "seed": int(args.seed),
        "stim_pattern": str(args.stim_pattern),
        "meas_pattern": str(args.meas_pattern),
        "v_cites": list(V_CITES),
    }
    summary_payload: dict[str, Any] = {
        **info,
        "per_regime": {r: per_regime[r]["summary"] for r in regimes},
    }
    if "auto" in per_regime and "never" in per_regime:
        auto_setup = per_regime["auto"]["summary"]["cumulative_setup_seconds"]
        never_setup = per_regime["never"]["summary"]["cumulative_setup_seconds"]
        summary_payload["g1_cumulative_setup_saved_seconds"] = float(
            never_setup - auto_setup
        )
        summary_payload["g1_warm_cold_setup_ratio"] = (
            float(auto_setup / never_setup) if never_setup > 0 else None
        )

    json_path = out_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    h5_path = out_dir / "ksp_session_reuse_runs.h5"
    write_hdf5_artifact(h5_path, arrays, summary_payload)

    md_path = out_dir / "summary.md"
    md_path.write_text(_format_md(per_regime, info), encoding="utf-8")

    print(f"[T4] artifact written: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
