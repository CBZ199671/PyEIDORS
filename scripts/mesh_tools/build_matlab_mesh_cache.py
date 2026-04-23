#!/usr/bin/env python3
"""Convert MATLAB mesh.h5 + electrodes.json into .msh cache usable by PyEIDORS."""

from __future__ import annotations

import argparse
import json
from configparser import ConfigParser
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import meshio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from matlab_mesh_hdf5 import load_matlab_mesh_arrays


def _sorted_edge(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _boundary_edges(elements: np.ndarray) -> np.ndarray:
    edge_counts: Dict[Tuple[int, int], int] = {}
    for tri in elements:
        a, b, c = (int(tri[0]), int(tri[1]), int(tri[2]))
        for edge in (_sorted_edge(a, b), _sorted_edge(b, c), _sorted_edge(c, a)):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = [edge for edge, count in edge_counts.items() if count == 1]
    return np.asarray(boundary, dtype=np.int32)


def _mean_angle(coords: np.ndarray, center: np.ndarray) -> float:
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _wrap_diff(a: float, b: float) -> float:
    diff = a - b
    while diff <= -np.pi:
        diff += 2 * np.pi
    while diff > np.pi:
        diff -= 2 * np.pi
    return abs(diff)


def _nearest_electrode(
    angle: float, electrode_angles: np.ndarray, order: np.ndarray
) -> int:
    diffs = [_wrap_diff(angle, a) for a in electrode_angles]
    return int(order[int(np.argmin(diffs))])


def build_boundary_tags(
    nodes: np.ndarray,
    boundary_edges: np.ndarray,
    electrodes: Sequence[Dict[str, object]],
) -> np.ndarray:
    center = nodes.mean(axis=0)
    electrode_angles: List[float] = []
    for idx, electrode in enumerate(electrodes):
        node_indices = [int(i) for i in electrode["node_indices"] if int(i) >= 0]
        if node_indices:
            angle = _mean_angle(nodes[node_indices], center)
        else:
            angle = -np.pi + idx * 1e-3
        electrode_angles.append(angle)

    order = np.argsort(np.asarray(electrode_angles))
    sorted_angles = np.asarray(electrode_angles)[order]

    tags = np.zeros(boundary_edges.shape[0], dtype=np.int32)
    for i, edge in enumerate(boundary_edges):
        midpoint = nodes[np.asarray(edge, dtype=np.int32)].mean(axis=0, keepdims=True)
        angle = _mean_angle(midpoint, center)
        elec_idx = _nearest_electrode(angle, sorted_angles, order)
        tags[i] = elec_idx + 2  # domain=1, electrodes start at 2
    return tags


def _build_meshio_points(nodes_xy: np.ndarray) -> np.ndarray:
    if nodes_xy.shape[1] == 3:
        return nodes_xy.astype(float)
    points = np.zeros((nodes_xy.shape[0], 3), dtype=float)
    points[:, : nodes_xy.shape[1]] = nodes_xy
    return points


def write_cache(
    nodes: np.ndarray,
    elements: np.ndarray,
    boundary_edges: np.ndarray,
    boundary_tags: np.ndarray,
    out_dir: Path,
    name: str,
    n_electrodes: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    msh_file = out_dir / f"{name}.msh"
    assoc_file = out_dir / f"{name}_association_table.ini"

    points = _build_meshio_points(nodes)
    tri_tags = np.full(elements.shape[0], 1, dtype=np.int32)

    field_data = {"domain": np.array([1, 2], dtype=np.int32)}
    for idx in range(n_electrodes):
        field_data[f"electrode_{idx + 1}"] = np.array([idx + 2, 1], dtype=np.int32)

    mesh = meshio.Mesh(
        points=points,
        cells=[
            ("triangle", elements.astype(np.int32)),
            ("line", boundary_edges.astype(np.int32)),
        ],
        cell_data={
            "gmsh:physical": [tri_tags, boundary_tags.astype(np.int32)],
            "gmsh:geometrical": [tri_tags, boundary_tags.astype(np.int32)],
        },
        field_data=field_data,
    )
    meshio.write(msh_file, mesh, file_format="gmsh22")

    cfg = ConfigParser()
    cfg["ASSOCIATION TABLE"] = {"domain": "1"}
    for idx in range(n_electrodes):
        cfg["ASSOCIATION TABLE"][f"electrode_{idx + 1}"] = str(idx + 2)
    with assoc_file.open("w", encoding="utf-8") as fh:
        cfg.write(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-h5", type=Path, help="HDF5 MATLAB mesh bridge arrays.")
    parser.add_argument(
        "--mesh-npz",
        type=Path,
        help="Legacy read-only NumPy MATLAB mesh bridge arrays.",
    )
    parser.add_argument("--electrodes-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mesh-name", type=str, default="matlab_import")
    args = parser.parse_args()

    if args.mesh_h5 is None and args.mesh_npz is None:
        parser.error("--mesh-h5 is required; --mesh-npz is legacy read-only fallback.")
    mesh_arrays_path = args.mesh_h5 if args.mesh_h5 is not None else args.mesh_npz
    nodes, raw_elements = load_matlab_mesh_arrays(mesh_arrays_path)
    elements = np.asarray(raw_elements, dtype=np.int32) - 1  # MATLAB -> 0-based

    electrodes = json.loads(args.electrodes_json.read_text(encoding="utf-8"))
    for electrode in electrodes:
        electrode["node_indices"] = [int(i) - 1 for i in electrode["node_indices"]]

    boundary_edges = _boundary_edges(elements)
    boundary_tags = build_boundary_tags(nodes, boundary_edges, electrodes)
    write_cache(
        nodes,
        elements,
        boundary_edges,
        boundary_tags,
        args.out_dir,
        args.mesh_name,
        len(electrodes),
    )


if __name__ == "__main__":
    main()
