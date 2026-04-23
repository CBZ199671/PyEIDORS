from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.geometry.dolfinx_mesh_cache import write_dolfinx_mesh_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_mesh_io_formats.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_mesh_io_formats", SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _unit_square_mesh_data():
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_tags = dmesh.meshtags(
        mesh,
        tdim,
        cells,
        np.full(cells.shape, 1, dtype=np.int32),
    )
    facets = dmesh.locate_entities_boundary(
        mesh,
        fdim,
        lambda x: np.full(x.shape[1], True, dtype=bool),
    ).astype(np.int32)
    order = np.argsort(facets)
    facet_tags = dmesh.meshtags(
        mesh,
        fdim,
        facets[order],
        np.full(facets.shape, 2, dtype=np.int32)[order],
    )

    class _Group:
        def __init__(self, dim: int, tag: int):
            self.dim = dim
            self.tag = tag

    return SimpleNamespace(
        mesh=mesh,
        facet_tags=facet_tags,
        cell_tags=cell_tags,
        physical_groups={
            "domain": _Group(2, 1),
            "electrode_1": _Group(1, 2),
        },
    )


def test_mesh_io_benchmark_writes_json_with_tag_equality(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    source_msh = tmp_path / "case.msh"
    source_msh.write_text("placeholder source", encoding="utf-8")
    mesh_data = _unit_square_mesh_data()
    association = {"domain": 1, "electrode_1": 2}
    assert write_dolfinx_mesh_cache(
        mesh_data,
        source_msh_file=source_msh,
        association_table=association,
        gdim=2,
    )
    monkeypatch.setattr(
        module,
        "load_msh_mesh_data",
        lambda _path, *, gdim: mesh_data,
    )

    payload = module.run_benchmarks(
        [source_msh],
        repeats=2,
        warmups=0,
        gdim=2,
        max_hdf5_to_msh_ratio=999.0,
    )
    output_json = tmp_path / "mesh_io.json"
    module.write_payload(output_json, payload)

    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["schema"] == module.SCHEMA
    assert saved["summary"]["all_equality_checks_passed"] is True
    case = saved["cases"][0]
    assert case["source_msh"].endswith("case.msh")
    assert case["hdf5_file"].endswith("case.h5")
    assert case["checks"]["vertices_equal"] is True
    assert case["checks"]["cells_equal"] is True
    assert case["checks"]["facet_tags_equal"] is True
    assert case["checks"]["cell_tags_equal"] is True
    assert case["checks"]["association_table_equal"] is True
    assert len(case["timings"]["msh_import_sec"]["samples"]) == 2
    assert len(case["timings"]["hdf5_load_sec"]["samples"]) == 2
    assert case["timings"]["hdf5_to_msh_median_ratio"] >= 0.0


def test_tag_signature_detects_tag_value_drift() -> None:
    module = _load_module()
    left = SimpleNamespace(
        indices=np.array([2, 1], dtype=np.int32),
        values=np.array([5, 4], dtype=np.int32),
        dim=1,
    )
    right = SimpleNamespace(
        indices=np.array([1, 2], dtype=np.int32),
        values=np.array([4, 6], dtype=np.int32),
        dim=1,
    )
    same_value_counts_with_remapped_entities = SimpleNamespace(
        indices=np.array([20, 10], dtype=np.int32),
        values=np.array([5, 4], dtype=np.int32),
        dim=1,
    )

    left_signature = module.tag_signature(left)
    remapped_signature = module.tag_signature(same_value_counts_with_remapped_entities)

    assert left_signature["pairs_sample"] == [[1, 4], [2, 5]]
    assert "pairs" not in left_signature
    assert module.tag_signatures_equal(left_signature, remapped_signature)
    assert not module.tag_entity_pairs_equal(left_signature, remapped_signature)
    assert not module.tag_signatures_equal(
        left_signature,
        module.tag_signature(right),
    )
