#!/usr/bin/env python3
"""Convert MATLAB fmdl mesh to PyEIDORS mesh JSON and NPZ formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.io as sio


def load_matlab_mesh(path: Path) -> Dict[str, Any]:
    data = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    nodes = np.asarray(data['nodes'], dtype=float)
    elems = np.asarray(data['elems'], dtype=int)
    elec_raw = data['electrodes']
    electrodes = []
    for elec in np.atleast_1d(elec_raw):
        nodes_idx = np.asarray(getattr(elec, 'nodes', getattr(elec, 'node', [])), dtype=int).tolist()
        electrodes.append({
            "node_indices": nodes_idx,
            "z_contact": float(getattr(elec, 'z_contact', 0.0)),
        })
    return {"nodes": nodes, "elements": elems, "electrodes": electrodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_file", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    mesh = load_matlab_mesh(args.mat_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "mesh.npz", nodes=mesh["nodes"], elements=mesh["elements"])
    with (args.out_dir / "electrodes.json").open("w", encoding="utf-8") as fh:
        json.dump(mesh["electrodes"], fh, indent=2)

if __name__ == "__main__":
    main()
