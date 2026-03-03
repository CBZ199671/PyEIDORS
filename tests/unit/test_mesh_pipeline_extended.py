"""Extended tests for mesh generators/loaders using patched gmsh backends."""

from __future__ import annotations

import configparser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dolfinx import mesh as dmesh
from mpi4py import MPI

from pyeidors.data.structures import ElectrodePosition, MeshConfig
from pyeidors.geometry import mesh_generator as mesh_gen_module
from pyeidors.geometry import mesh_loader as mesh_loader_module
from pyeidors.geometry import optimized_mesh_generator as opt_mesh_module


def _make_fake_mesh_data():
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    fdim = mesh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    facets = np.asarray(facets, dtype=np.int32)
    values = np.full(facets.shape, 2, dtype=np.int32)
    order = np.argsort(facets)
    facet_tags = dmesh.meshtags(mesh, fdim, facets[order], values[order])

    class _Group:
        def __init__(self, tag: int):
            self.tag = int(tag)

    physical_groups = {"domain": _Group(1), "electrode_1": _Group(2), "gaps": _Group(3)}
    return SimpleNamespace(mesh=mesh, facet_tags=facet_tags, cell_tags=None, physical_groups=physical_groups)


class _FakeOcc:
    def __init__(self):
        self._counter = 0

    def _next(self):
        self._counter += 1
        return self._counter

    def addPoint(self, x, y, z, meshSize=None):
        _ = (x, y, z, meshSize)
        return self._next()

    def addLine(self, p1, p2):
        _ = (p1, p2)
        return self._next()

    def addCurveLoop(self, lines):
        _ = lines
        return self._next()

    def addPlaneSurface(self, loops):
        _ = loops
        return self._next()

    def synchronize(self):
        return None


class _FakeModelMesh:
    def embed(self, dim, entities, target_dim, target_entity):
        _ = (dim, entities, target_dim, target_entity)

    def setSize(self, entities, size):
        _ = (entities, size)

    def generate(self, dim):
        _ = dim


class _FakeModel:
    def __init__(self):
        self.occ = _FakeOcc()
        self.mesh = _FakeModelMesh()
        self._name = None
        self.physical_groups = []

    def add(self, name: str):
        self._name = name

    def addPhysicalGroup(self, dim, entities, tag, name=None):
        self.physical_groups.append((dim, tuple(entities), int(tag), name))

    def getEntities(self, dim):
        return [(dim, 1)]


class _FakeGmsh:
    def __init__(self):
        self.model = _FakeModel()
        self._initialized = False
        self.written_files: list[str] = []

    def initialize(self):
        self._initialized = True

    def finalize(self):
        self._initialized = False

    def isInitialized(self):
        return self._initialized

    def clear(self):
        self.model = _FakeModel()

    def write(self, path: str):
        self.written_files.append(path)
        Path(path).write_text("fake-msh", encoding="utf-8")


def test_mesh_generator_generate_with_fake_gmsh(tmp_path, monkeypatch):
    fake_gmsh = _FakeGmsh()
    fake_mesh_data = _make_fake_mesh_data()
    monkeypatch.setattr(mesh_gen_module, "gmsh", fake_gmsh)
    monkeypatch.setattr(
        mesh_gen_module.gmshio,
        "model_to_mesh",
        lambda model, comm, rank, gdim: fake_mesh_data,
    )

    config = MeshConfig(radius=1.0, refinement=6, electrode_vertices=4, gap_vertices=1, mesh_size=0.15)
    electrode_positions = ElectrodePosition.create_circular(n_elec=8)
    generator = mesh_gen_module.MeshGenerator(config=config, electrodes=electrode_positions)

    metadata = generator.generate(output_dir=tmp_path, return_metadata=True, save_msh=True, mesh_name="patched")
    mesh = metadata["mesh"]
    assert mesh.num_cells() > 0
    assert "domain" in metadata["association_table"]
    assert metadata["mesh_file"] is not None
    assert fake_gmsh.written_files


def test_optimized_generator_and_cache_functions(tmp_path, monkeypatch):
    fake_gmsh = _FakeGmsh()
    fake_mesh_data = _make_fake_mesh_data()
    monkeypatch.setattr(opt_mesh_module, "gmsh", fake_gmsh)
    monkeypatch.setattr(opt_mesh_module, "GMSH_AVAILABLE", True)
    monkeypatch.setattr(
        opt_mesh_module.gmshio,
        "model_to_mesh",
        lambda model, comm, rank, gdim: fake_mesh_data,
    )
    monkeypatch.setattr(
        opt_mesh_module.gmshio,
        "read_from_msh",
        lambda file, comm, rank, gdim: fake_mesh_data,
    )

    config = opt_mesh_module.OptimizedMeshConfig(radius=1.0, refinement=5, electrode_vertices=3, gap_vertices=1)
    electrode_cfg = opt_mesh_module.ElectrodePosition(L=8, coverage=0.5)
    generator = opt_mesh_module.OptimizedMeshGenerator(config=config, electrodes=electrode_cfg)

    mesh = generator.generate(output_dir=tmp_path, mesh_name="opt_patch")
    assert mesh.num_vertices() > 0
    assert mesh.association_table["domain"] == 1

    converter = opt_mesh_module.OptimizedMeshConverter(
        mesh_file=str(tmp_path / "opt_patch.msh"),
        output_dir=str(tmp_path),
    )
    converted_mesh, facet_tags, association = converter.convert()
    assert converted_mesh.num_cells() > 0
    assert facet_tags is not None
    assert "domain" in association

    loaded = opt_mesh_module._load_cached_mesh(tmp_path, "opt_patch")
    assert loaded is not None
    assert loaded.num_cells() > 0

    created = opt_mesh_module.load_or_create_mesh(
        mesh_dir=str(tmp_path),
        mesh_name="opt_patch",
        n_elec=8,
        radius=1.0,
        refinement=4,
        electrode_coverage=0.5,
    )
    assert created.num_vertices() > 0


def test_mesh_loader_functions_with_fake_read(tmp_path, monkeypatch):
    fake_mesh_data = _make_fake_mesh_data()
    monkeypatch.setattr(
        mesh_loader_module.gmshio,
        "read_from_msh",
        lambda file, comm, rank, gdim: fake_mesh_data,
    )

    mesh_name = "cached_mesh"
    (tmp_path / f"{mesh_name}.msh").write_text("msh", encoding="utf-8")
    assoc = configparser.ConfigParser()
    assoc["ASSOCIATION TABLE"] = {"domain": "1", "electrode_1": "2", "gaps": "3"}
    with (tmp_path / f"{mesh_name}_association_table.ini").open("w", encoding="utf-8") as f:
        assoc.write(f)
    np.save(tmp_path / "sample.npy", np.array([1, 2, 3]))
    (tmp_path / "sample.xdmf").write_text("<xdmf/>", encoding="utf-8")

    loader = mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path))
    loaded_mesh = loader.load_mesh(mesh_name)
    assert loaded_mesh.num_cells() > 0
    assert loaded_mesh.association_table["domain"] == 1

    listed = loader.list_available_meshes()
    assert mesh_name in listed["msh"]
    assert "sample" in listed["numpy"]
    assert "sample" in listed["xdmf"]

    arr = loader.load_numpy_mesh("sample.npy")
    assert arr.shape == (3,)

    default_mesh = loader.get_default_mesh()
    assert default_mesh.num_vertices() > 0


def test_mesh_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path / "missing"))

    existing = tmp_path / "meshes"
    existing.mkdir()
    loader = mesh_loader_module.MeshLoader(mesh_dir=str(existing))

    with pytest.raises(FileNotFoundError):
        loader.load_mesh("not_found")

    with pytest.raises(FileNotFoundError):
        loader.load_numpy_mesh("none.npy")

    with pytest.raises(FileNotFoundError):
        loader.get_default_mesh()
