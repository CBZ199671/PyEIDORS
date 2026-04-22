#!/usr/bin/env python3
"""Convert MATLAB fmdl mesh to PyEIDORS mesh JSON and HDF5 formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import scipy.io as sio

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from matlab_mesh_hdf5 import write_matlab_mesh_hdf5


def load_matlab_mesh(path: Path) -> Dict[str, Any]:
    data = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    nodes = np.asarray(data["nodes"], dtype=float)
    elems = np.asarray(data["elems"], dtype=int)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    if elems.ndim == 1:
        elems = elems.reshape(1, -1)
    elec_raw = data["electrodes"]
    electrodes = []
    for elec in np.atleast_1d(elec_raw):
        nodes_idx = np.asarray(
            getattr(elec, "nodes", getattr(elec, "node", [])),
            dtype=int,
        ).tolist()
        electrodes.append(
            {
                "node_indices": nodes_idx,
                "z_contact": float(getattr(elec, "z_contact", 0.0)),
            }
        )
    return {"nodes": nodes, "elements": elems, "electrodes": electrodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--mesh-h5-name",
        default="mesh.h5",
        help="HDF5 bridge-array filename written inside out_dir.",
    )
    args = parser.parse_args()

    mesh = load_matlab_mesh(args.mat_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_matlab_mesh_hdf5(
        args.out_dir / args.mesh_h5_name,
        nodes=mesh["nodes"],
        elements=mesh["elements"],
        metadata={"source_mat_file": str(args.mat_file)},
    )
    with (args.out_dir / "electrodes.json").open("w", encoding="utf-8") as fh:
        json.dump(mesh["electrodes"], fh, indent=2)


if __name__ == "__main__":
    main()
