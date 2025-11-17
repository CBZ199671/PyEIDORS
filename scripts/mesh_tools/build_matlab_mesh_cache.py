#!/usr/bin/env python3
"""Convert MATLAB mesh.npz + electrodes.json into XDMF cache usable by PyEidors."""

from __future__ import annotations

import argparse
import json
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from dolfin import Mesh, MeshEditor, MeshFunction, facets, XDMFFile, Point


def build_mesh(nodes: np.ndarray, elements: np.ndarray) -> Mesh:
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(len(nodes))
    editor.init_cells(len(elements))
    for idx, (x, y) in enumerate(nodes):
        editor.add_vertex(idx, Point(float(x), float(y), 0.0))
    for idx, conn in enumerate(elements):
        editor.add_cell(idx, conn.astype(int))
    editor.close()
    mesh.init(1, 2)
    return mesh


def build_boundary_markers(mesh: Mesh, electrodes: Sequence[Dict[str, object]], nodes: np.ndarray) -> MeshFunction:
    marker = MeshFunction("size_t", mesh, 1, 0)
    marker.set_all(1)
    marker_array = marker.array()
    center = nodes.mean(axis=0)

    def avg_angle(coords: np.ndarray) -> float:
        ang = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        return float(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))))

    electrode_angles = []
    for idx, elec in enumerate(electrodes):
        node_indices = [int(i) for i in elec["node_indices"] if i >= 0]
        if node_indices:
            coords = nodes[node_indices]
            electrode_angles.append(avg_angle(coords))
        else:
            electrode_angles.append(-np.pi + idx * 1e-3)

    order = np.argsort(electrode_angles)
    ordered_angles = np.array(electrode_angles)[order]

    def angle_diff(a: float, b: float) -> float:
        diff = a - b
        while diff <= -np.pi:
            diff += 2 * np.pi
        while diff > np.pi:
            diff -= 2 * np.pi
        return abs(diff)

    def nearest_electrode(angle: float) -> int:
        diffs = [angle_diff(angle, a) for a in ordered_angles]
        idx = int(np.argmin(diffs))
        return int(order[idx])

    for facet in facets(mesh):
        if not facet.exterior():
            continue
        verts = facet.entities(0)
        coords = nodes[verts]
        angle = avg_angle(coords)
        elec_idx = nearest_electrode(angle)
        marker_array[facet.index()] = elec_idx + 2

    return marker


def write_cache(mesh: Mesh, boundary_markers: MeshFunction, out_dir: Path, name: str, n_electrodes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_file = out_dir / f"{name}_domain.xdmf"
    boundaries_file = out_dir / f"{name}_boundaries.xdmf"
    assoc_file = out_dir / f"{name}_association_table.ini"

    with XDMFFile(str(domain_file)) as fh:
        fh.write(mesh)
    with XDMFFile(str(boundaries_file)) as fh:
        fh.write(boundary_markers)

    cfg = ConfigParser()
    cfg["ASSOCIATION TABLE"] = {"domain": "1"}
    for idx in range(n_electrodes):
        cfg["ASSOCIATION TABLE"][f"electrode_{idx+1}"] = str(idx + 2)
    with assoc_file.open("w", encoding="utf-8") as fh:
        cfg.write(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--electrodes-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mesh-name", type=str, default="matlab_import")
    args = parser.parse_args()

    data = np.load(args.mesh_npz)
    nodes = data["nodes"]
    elements = data["elements"].astype(int) - 1  # Convert to 0-based

    electrodes = json.loads(args.electrodes_json.read_text(encoding="utf-8"))
    for elec in electrodes:
        elec["node_indices"] = [int(i) - 1 for i in elec["node_indices"]]

    mesh = build_mesh(nodes, elements)
    boundaries = build_boundary_markers(mesh, electrodes, nodes)
    write_cache(mesh, boundaries, args.out_dir, args.mesh_name, len(electrodes))


if __name__ == "__main__":
    main()
