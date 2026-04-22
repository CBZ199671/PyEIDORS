from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio


REPO_ROOT = Path(__file__).resolve().parents[2]
MESH_TOOLS = REPO_ROOT / "scripts" / "mesh_tools"
if str(MESH_TOOLS) not in sys.path:
    sys.path.insert(0, str(MESH_TOOLS))

import build_matlab_mesh_cache  # noqa: E402
import convert_matlab_mesh  # noqa: E402
from matlab_mesh_hdf5 import (  # noqa: E402
    MATLAB_MESH_HDF5_SCHEMA,
    load_matlab_mesh_arrays,
    matlab_mesh_hdf5_path,
    write_matlab_mesh_hdf5,
)


def test_matlab_mesh_hdf5_roundtrip_and_matlab_friendly_layout(tmp_path: Path) -> None:
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elements = np.array([[1, 2, 3]], dtype=np.int64)

    path = write_matlab_mesh_hdf5(
        tmp_path / "mesh.h5",
        nodes=nodes,
        elements=elements,
        metadata={"source": "unit-test"},
    )

    assert path == tmp_path / "mesh.h5"
    assert not (tmp_path / "mesh.npz").exists()
    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema"] == MATLAB_MESH_HDF5_SCHEMA
        assert set(handle.keys()) == {"elements", "nodes"}
        assert json.loads(handle.attrs["metadata_json"])["index_base"] == 1
    loaded_nodes, loaded_elements = load_matlab_mesh_arrays(path)
    np.testing.assert_allclose(loaded_nodes, nodes)
    np.testing.assert_array_equal(loaded_elements, elements)


def test_matlab_mesh_bridge_retains_legacy_npz_reader(tmp_path: Path) -> None:
    legacy = tmp_path / "mesh.npz"
    np.savez(
        legacy,
        nodes=np.array([[0.0, 0.0], [1.0, 0.0]]),
        elements=np.array([[1, 2, 1]], dtype=np.int64),
    )

    nodes, elements = build_matlab_mesh_cache.load_matlab_mesh_arrays(legacy)

    np.testing.assert_allclose(nodes, [[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(elements, [[1, 2, 1]])
    assert legacy.exists()
    assert not (tmp_path / "mesh.h5").exists()


def test_matlab_mesh_hdf5_path_rejects_new_npz_output(tmp_path: Path) -> None:
    assert matlab_mesh_hdf5_path(tmp_path / "mesh").suffix == ".h5"
    assert matlab_mesh_hdf5_path(tmp_path / "mesh.hdf5").suffix == ".hdf5"
    try:
        matlab_mesh_hdf5_path(tmp_path / "mesh.npz")
    except ValueError as exc:
        assert "must use .h5 or .hdf5" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")


def test_convert_matlab_mesh_writes_h5_and_electrodes_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mat_path = tmp_path / "source.mat"
    out_dir = tmp_path / "bridge"
    electrodes = np.empty((2,), dtype=[("nodes", "O"), ("z_contact", "O")])
    electrodes[0] = (np.array([1, 2]), np.array([[0.01]]))
    electrodes[1] = (np.array([2, 3]), np.array([[0.02]]))
    sio.savemat(
        mat_path,
        {
            "nodes": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "elems": np.array([[1, 2, 3]]),
            "electrodes": electrodes,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["convert_matlab_mesh.py", str(mat_path), str(out_dir)],
    )
    convert_matlab_mesh.main()

    assert (out_dir / "mesh.h5").exists()
    assert not (out_dir / "mesh.npz").exists()
    loaded_nodes, loaded_elements = load_matlab_mesh_arrays(out_dir / "mesh.h5")
    assert loaded_nodes.shape == (3, 2)
    np.testing.assert_array_equal(loaded_elements, [[1, 2, 3]])
    electrodes_payload = json.loads((out_dir / "electrodes.json").read_text())
    assert electrodes_payload[0]["node_indices"] == [1, 2]


def test_build_matlab_mesh_cache_accepts_hdf5_bridge_arrays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mesh_h5 = write_matlab_mesh_hdf5(
        tmp_path / "mesh.h5",
        nodes=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        elements=np.array([[1, 2, 3]], dtype=np.int64),
    )
    electrodes_json = tmp_path / "electrodes.json"
    electrodes_json.write_text(
        json.dumps(
            [
                {"node_indices": [1, 2], "z_contact": 0.01},
                {"node_indices": [2, 3], "z_contact": 0.01},
                {"node_indices": [1, 3], "z_contact": 0.01},
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "cache"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_matlab_mesh_cache.py",
            "--mesh-h5",
            str(mesh_h5),
            "--electrodes-json",
            str(electrodes_json),
            "--out-dir",
            str(out_dir),
            "--mesh-name",
            "matlab_import",
        ],
    )
    build_matlab_mesh_cache.main()

    assert (out_dir / "matlab_import.msh").exists()
    assert (out_dir / "matlab_import_association_table.ini").exists()
