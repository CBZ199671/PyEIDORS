#!/usr/bin/env python3
"""Benchmark Gmsh ``.msh`` import versus DOLFINx XDMF/HDF5 mesh cache load."""

from __future__ import annotations

import argparse
from configparser import ConfigParser
import hashlib
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.geometry._helpers import validate_mesh_data_tags
from pyeidors.geometry.dolfinx_mesh_cache import (
    dolfinx_cache_metadata_path_for_mesh,
    load_dolfinx_mesh_cache,
    write_dolfinx_mesh_cache,
    xdmf_cache_path_for_mesh,
    xdmf_h5_path_for_mesh,
)


SCHEMA = "pyeidors-mesh-io-format-benchmark-v1"


@dataclass(frozen=True)
class MeshCasePaths:
    input_path: Path
    source_msh: Path
    cache_probe: Path
    xdmf_file: Path
    hdf5_file: Path
    metadata_file: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        type=Path,
        action="append",
        required=True,
        help="Mesh path to benchmark. Repeat for representative 2D/3D cases.",
    )
    parser.add_argument(
        "--gdim",
        type=int,
        choices=(2, 3),
        default=None,
        help="Geometry dimension. If omitted, read cache metadata or infer from name.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--max-hdf5-to-msh-ratio",
        type=float,
        default=1.0,
        help="Gate threshold: HDF5 median / .msh median must be <= this value unless explained.",
    )
    parser.add_argument(
        "--explain-slower-cache",
        default="",
        help="Explanation recorded when a cache-load speed regression is expected.",
    )
    parser.add_argument(
        "--fail-on-speed-regression",
        action="store_true",
        help="Exit nonzero if HDF5 is slower than threshold and no explanation is supplied.",
    )
    return parser.parse_args(argv)


def run_benchmarks(
    mesh_paths: Sequence[Path],
    *,
    repeats: int,
    warmups: int,
    gdim: int | None,
    max_hdf5_to_msh_ratio: float,
    explain_slower_cache: str = "",
) -> dict[str, Any]:
    cases = [
        benchmark_mesh_case(
            path,
            repeats=repeats,
            warmups=warmups,
            gdim=gdim,
            max_hdf5_to_msh_ratio=max_hdf5_to_msh_ratio,
            explain_slower_cache=explain_slower_cache,
        )
        for path in mesh_paths
    ]
    return {
        "schema": SCHEMA,
        "repeats": int(repeats),
        "warmups": int(warmups),
        "case_count": len(cases),
        "cases": cases,
        "summary": {
            "all_equality_checks_passed": all(
                bool(case["checks"]["all_equal"]) for case in cases
            ),
            "all_speed_gates_passed_or_explained": all(
                bool(case["speed_gate"]["passed_or_explained"]) for case in cases
            ),
            "max_hdf5_to_msh_median_ratio": max(
                (float(case["timings"]["hdf5_to_msh_median_ratio"]) for case in cases),
                default=0.0,
            ),
        },
    }


def benchmark_mesh_case(
    path: Path,
    *,
    repeats: int,
    warmups: int,
    gdim: int | None,
    max_hdf5_to_msh_ratio: float,
    explain_slower_cache: str = "",
) -> dict[str, Any]:
    repeats = max(int(repeats), 1)
    warmups = max(int(warmups), 0)
    paths = resolve_mesh_case_paths(Path(path))
    effective_gdim = int(gdim) if gdim is not None else infer_gdim(paths.input_path)

    ensure_cache_ready(paths, gdim=effective_gdim)

    msh_timings, msh_mesh_data = time_repeated(
        lambda: load_msh_mesh_data(paths.source_msh, gdim=effective_gdim),
        repeats=repeats,
        warmups=warmups,
    )
    hdf5_timings, hdf5_mesh_data = time_repeated(
        lambda: load_hdf5_cache_data(paths.cache_probe, gdim=effective_gdim),
        repeats=repeats,
        warmups=warmups,
    )

    msh_assoc = association_for_mesh_data(
        msh_mesh_data, paths.source_msh, effective_gdim
    )
    hdf5_assoc = dict(hdf5_mesh_data.association_table)
    checks = compare_mesh_data(
        msh_mesh_data,
        hdf5_mesh_data,
        msh_association=msh_assoc,
        hdf5_association=hdf5_assoc,
    )
    timing_payload = build_timing_payload(msh_timings, hdf5_timings)
    ratio = float(timing_payload["hdf5_to_msh_median_ratio"])
    speed_passed = ratio <= float(max_hdf5_to_msh_ratio)
    explained = bool(str(explain_slower_cache).strip())
    return {
        "input_path": str(paths.input_path),
        "source_msh": str(paths.source_msh),
        "xdmf_file": str(paths.xdmf_file),
        "hdf5_file": str(paths.hdf5_file),
        "metadata_file": str(paths.metadata_file),
        "gdim": effective_gdim,
        "timings": timing_payload,
        "checks": checks,
        "speed_gate": {
            "max_hdf5_to_msh_ratio": float(max_hdf5_to_msh_ratio),
            "passed": bool(speed_passed),
            "explanation": str(explain_slower_cache),
            "passed_or_explained": bool(speed_passed or explained),
        },
    }


def resolve_mesh_case_paths(path: Path) -> MeshCasePaths:
    input_path = path.resolve()
    suffix = input_path.suffix.lower()
    metadata = read_cache_metadata(input_path)
    source_payload = metadata.get("source_msh_file") if metadata else None
    source_msh = (
        Path(source_payload) if source_payload else input_path.with_suffix(".msh")
    )
    if suffix == ".msh":
        source_msh = input_path
        cache_probe = input_path
    elif suffix in {".xdmf", ".h5", ".hdf5"}:
        cache_probe = input_path
    else:
        raise ValueError(f"--mesh must point to .msh, .xdmf, .h5, or .hdf5: {path}")

    if not source_msh.exists():
        raise FileNotFoundError(f"Source .msh is required for comparison: {source_msh}")

    return MeshCasePaths(
        input_path=input_path,
        source_msh=source_msh,
        cache_probe=cache_probe,
        xdmf_file=xdmf_cache_path_for_mesh(source_msh),
        hdf5_file=xdmf_h5_path_for_mesh(source_msh),
        metadata_file=dolfinx_cache_metadata_path_for_mesh(source_msh),
    )


def read_cache_metadata(path: Path) -> dict[str, Any]:
    metadata_file = dolfinx_cache_metadata_path_for_mesh(path)
    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def infer_gdim(path: Path) -> int:
    metadata = read_cache_metadata(path)
    if "gdim" in metadata:
        return int(metadata["gdim"])
    lowered = path.stem.lower()
    if "mesh3d" in lowered or "_3d" in lowered or "cyl3d" in lowered:
        return 3
    return 2


def ensure_cache_ready(paths: MeshCasePaths, *, gdim: int) -> None:
    if load_dolfinx_mesh_cache(paths.cache_probe, gdim=gdim) is not None:
        return
    mesh_data = load_msh_mesh_data(paths.source_msh, gdim=gdim)
    association = association_for_mesh_data(mesh_data, paths.source_msh, gdim)
    if not write_dolfinx_mesh_cache(
        mesh_data,
        source_msh_file=paths.source_msh,
        association_table=association,
        gdim=gdim,
    ):
        raise RuntimeError(f"Unable to write XDMF/HDF5 cache for {paths.source_msh}")


def load_msh_mesh_data(path: Path, *, gdim: int):
    return gmshio.read_from_msh(str(path), MPI.COMM_WORLD, rank=0, gdim=int(gdim))


def load_hdf5_cache_data(path: Path, *, gdim: int):
    cache_data = load_dolfinx_mesh_cache(path, gdim=int(gdim))
    if cache_data is None:
        raise RuntimeError(f"Unable to load DOLFINx XDMF/HDF5 cache for {path}")
    return cache_data


def association_for_mesh_data(
    mesh_data: Any, source_msh: Path, gdim: int
) -> dict[str, int]:
    association = validate_mesh_data_tags(mesh_data, gdim=int(gdim))
    if association:
        return {str(key): int(value) for key, value in association.items()}
    return read_association_table(
        source_msh.with_name(f"{source_msh.stem}_association_table.ini")
    )


def read_association_table(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    config = ConfigParser()
    config.read(path)
    section = None
    if "ASSOCIATION TABLE" in config:
        section = config["ASSOCIATION TABLE"]
    elif "boundary_ids" in config:
        section = config["boundary_ids"]
    if section is None:
        return {}
    values: dict[str, int] = {}
    for key, value in section.items():
        try:
            values[str(key).strip()] = int(value)
        except ValueError:
            continue
    return values


def time_repeated(
    fn: Callable[[], Any],
    *,
    repeats: int,
    warmups: int,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[list[float], Any]:
    result: Any = None
    for _ in range(max(int(warmups), 0)):
        result = fn()
    samples: list[float] = []
    for _ in range(max(int(repeats), 1)):
        start = timer()
        result = fn()
        samples.append(float(timer() - start))
    return samples, result


def build_timing_payload(
    msh_timings: Sequence[float], hdf5_timings: Sequence[float]
) -> dict[str, Any]:
    msh_stats = timing_stats(msh_timings)
    hdf5_stats = timing_stats(hdf5_timings)
    msh_median = max(float(msh_stats["median_sec"]), 1.0e-15)
    hdf5_median = float(hdf5_stats["median_sec"])
    ratio = hdf5_median / msh_median
    return {
        "msh_import_sec": msh_stats,
        "hdf5_load_sec": hdf5_stats,
        "hdf5_to_msh_median_ratio": float(ratio),
        "msh_to_hdf5_median_speedup": float(1.0 / ratio) if ratio > 0 else float("inf"),
    }


def timing_stats(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("timing samples cannot be empty")
    return {
        "samples": values,
        "min_sec": float(min(values)),
        "median_sec": float(statistics.median(values)),
        "mean_sec": float(statistics.fmean(values)),
        "max_sec": float(max(values)),
    }


def compare_mesh_data(
    msh_mesh_data: Any,
    hdf5_mesh_data: Any,
    *,
    msh_association: dict[str, int],
    hdf5_association: dict[str, int],
) -> dict[str, Any]:
    msh_mesh = msh_mesh_data.mesh
    hdf5_mesh = hdf5_mesh_data.mesh
    msh_summary = mesh_summary(msh_mesh)
    hdf5_summary = mesh_summary(hdf5_mesh)
    msh_facet_tags = tag_signature(getattr(msh_mesh_data, "facet_tags", None))
    hdf5_facet_tags = tag_signature(getattr(hdf5_mesh_data, "facet_tags", None))
    msh_cell_tags = tag_signature(getattr(msh_mesh_data, "cell_tags", None))
    hdf5_cell_tags = tag_signature(getattr(hdf5_mesh_data, "cell_tags", None))
    facet_equal = tag_signatures_equal(msh_facet_tags, hdf5_facet_tags)
    cell_equal = tag_signatures_equal(msh_cell_tags, hdf5_cell_tags)
    association_equal = normalize_association(msh_association) == normalize_association(
        hdf5_association
    )
    checks = {
        "vertices_equal": msh_summary["num_vertices"] == hdf5_summary["num_vertices"],
        "cells_equal": msh_summary["num_cells"] == hdf5_summary["num_cells"],
        "topology_dim_equal": msh_summary["topology_dim"]
        == hdf5_summary["topology_dim"],
        "geometry_dim_equal": msh_summary["geometry_dim"]
        == hdf5_summary["geometry_dim"],
        "facet_tags_equal": bool(facet_equal),
        "cell_tags_equal": bool(cell_equal),
        "association_table_equal": bool(association_equal),
    }
    diagnostics = {
        "facet_tag_entity_pairs_equal": tag_entity_pairs_equal(
            msh_facet_tags, hdf5_facet_tags
        ),
        "cell_tag_entity_pairs_equal": tag_entity_pairs_equal(
            msh_cell_tags, hdf5_cell_tags
        ),
    }
    return {
        **checks,
        **diagnostics,
        "all_equal": all(bool(value) for value in checks.values()),
        "msh": {
            **msh_summary,
            "facet_tags": msh_facet_tags,
            "cell_tags": msh_cell_tags,
            "association_table": normalize_association(msh_association),
        },
        "hdf5": {
            **hdf5_summary,
            "facet_tags": hdf5_facet_tags,
            "cell_tags": hdf5_cell_tags,
            "association_table": normalize_association(hdf5_association),
        },
    }


def mesh_summary(mesh: Any) -> dict[str, int]:
    tdim = int(mesh.topology.dim)
    geometry_dim = int(mesh.geometry.dim)
    vertex_map = mesh.topology.index_map(0)
    cell_map = mesh.topology.index_map(tdim)
    return {
        "num_vertices": int(vertex_map.size_local if vertex_map is not None else 0),
        "num_cells": int(cell_map.size_local if cell_map is not None else 0),
        "topology_dim": tdim,
        "geometry_dim": geometry_dim,
    }


def tag_signature(tags: Any | None) -> dict[str, Any]:
    if tags is None:
        return {
            "present": False,
            "dim": None,
            "count": 0,
            "value_counts": {},
            "pairs_sha256": pairs_sha256([], []),
            "pairs_sample": [],
        }
    indices = np.asarray(getattr(tags, "indices", []), dtype=np.int64).ravel()
    values = np.asarray(getattr(tags, "values", []), dtype=np.int64).ravel()
    if indices.size != values.size:
        raise ValueError("MeshTags indices and values sizes differ")
    if indices.size:
        order = np.lexsort((values, indices))
        indices = indices[order]
        values = values[order]
    unique, counts = np.unique(values, return_counts=True) if values.size else ([], [])
    return {
        "present": True,
        "dim": None if getattr(tags, "dim", None) is None else int(tags.dim),
        "count": int(indices.size),
        "value_counts": {
            str(int(value)): int(count) for value, count in zip(unique, counts)
        },
        "pairs_sha256": pairs_sha256(indices, values),
        "pairs_sample": [
            [int(index), int(value)] for index, value in zip(indices[:16], values[:16])
        ],
    }


def tag_signatures_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        bool(left["present"]) == bool(right["present"])
        and left["dim"] == right["dim"]
        and left["count"] == right["count"]
        and left["value_counts"] == right["value_counts"]
    )


def tag_entity_pairs_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        bool(left["present"]) == bool(right["present"])
        and left["dim"] == right["dim"]
        and left["count"] == right["count"]
        and left["pairs_sha256"] == right["pairs_sha256"]
    )


def pairs_sha256(indices: Iterable[int], values: Iterable[int]) -> str:
    index_array = np.asarray(list(indices), dtype=np.int64).ravel()
    value_array = np.asarray(list(values), dtype=np.int64).ravel()
    if index_array.size != value_array.size:
        raise ValueError("tag pair indices and values sizes differ")
    if index_array.size:
        pairs = np.column_stack((index_array, value_array))
    else:
        pairs = np.empty((0, 2), dtype=np.int64)
    return hashlib.sha256(
        np.ascontiguousarray(pairs, dtype=np.int64).tobytes()
    ).hexdigest()


def normalize_association(value: dict[str, int]) -> dict[str, int]:
    return {str(key): int(val) for key, val in sorted(value.items())}


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_benchmarks(
        args.mesh,
        repeats=args.repeats,
        warmups=args.warmups,
        gdim=args.gdim,
        max_hdf5_to_msh_ratio=args.max_hdf5_to_msh_ratio,
        explain_slower_cache=args.explain_slower_cache,
    )
    if MPI.COMM_WORLD.rank == 0:
        write_payload(args.output_json, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
    if (
        args.fail_on_speed_regression
        and not payload["summary"]["all_speed_gates_passed_or_explained"]
    ):
        return 2
    if not payload["summary"]["all_equality_checks_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
