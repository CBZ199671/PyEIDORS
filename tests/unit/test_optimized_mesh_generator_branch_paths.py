"""Additional branch coverage for optimized mesh generator helpers."""

from __future__ import annotations

import configparser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.geometry import optimized_mesh_generator as opt_mesh_module


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

    def add(self, name):
        _ = name

    def addPhysicalGroup(self, dim, entities, tag, name=None):
        _ = (dim, entities, tag, name)

    def getEntities(self, dim):
        return [(dim, 1)]


class _FakeGmsh:
    def __init__(self, initialized: bool):
        self.model = _FakeModel()
        self._initialized = initialized
        self.finalized = 0
        self.cleared = 0

    def initialize(self):
        self._initialized = True

    def finalize(self):
        self.finalized += 1
        self._initialized = False

    def isInitialized(self):
        return self._initialized

    def clear(self):
        self.cleared += 1
        self.model = _FakeModel()

    def write(self, path: str):
        Path(path).write_text("fake-msh", encoding="utf-8")


def _fake_mesh_data():
    class _Group:
        def __init__(self, tag):
            self.tag = int(tag)

    return SimpleNamespace(
        mesh="mesh",
        facet_tags="facet-tags",
        cell_tags="cell-tags",
        physical_groups={
            "domain": _Group(1),
            "gaps": _Group(18),
            **{f"electrode_{idx}": _Group(idx + 1) for idx in range(1, 17)},
        },
    )


def test_generate_importerror_tempdir_and_existing_gmsh_paths(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(opt_mesh_module, "GMSH_AVAILABLE", False)
    generator = opt_mesh_module.OptimizedMeshGenerator(
        config=opt_mesh_module.OptimizedMeshConfig(
            radius=1.0, refinement=2, electrode_vertices=3, gap_vertices=1
        ),
        electrodes=opt_mesh_module.ElectrodePosition(L=4, coverage=0.5),
    )
    with pytest.raises(ImportError, match="gmsh Python bindings"):
        generator.generate()

    fake_gmsh = _FakeGmsh(initialized=True)
    monkeypatch.setattr(opt_mesh_module, "GMSH_AVAILABLE", True)
    monkeypatch.setattr(opt_mesh_module, "gmsh", fake_gmsh)
    monkeypatch.setattr(
        opt_mesh_module.tempfile,
        "mkdtemp",
        lambda: str(
            (tmp_path / "auto").mkdir(parents=True, exist_ok=True)
            or (tmp_path / "auto")
        ),
    )
    monkeypatch.setattr(opt_mesh_module.time, "time", lambda: 123.456789)
    monkeypatch.setattr(
        opt_mesh_module.gmshio,
        "model_to_mesh",
        lambda model, comm, rank, gdim: _fake_mesh_data(),
    )
    monkeypatch.setattr(
        opt_mesh_module,
        "build_eit_mesh",
        lambda *args, **kwargs: SimpleNamespace(
            num_vertices=lambda: 4,
            association_table=kwargs["association_table"],
            mesh_file=kwargs["mesh_file"],
        ),
    )

    mesh = generator.generate(output_dir=None, mesh_name=None)
    assert mesh.num_vertices() == 4
    assert mesh.mesh_file.endswith(".msh")
    assert fake_gmsh.cleared >= 1
    assert fake_gmsh.finalized == 0


def test_create_load_and_cached_mesh_branch_paths(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        opt_mesh_module,
        "OptimizedMeshGenerator",
        lambda config, electrodes: SimpleNamespace(
            generate=lambda output_dir=None, mesh_name=None: {
                "config": config,
                "electrodes": electrodes,
                "output_dir": output_dir,
                "mesh_name": mesh_name,
            }
        ),
    )
    created = opt_mesh_module.create_eit_mesh(
        n_elec=8,
        radius=1.2,
        refinement=3,
        electrode_coverage=0.25,
        output_dir=str(tmp_path),
        mesh_name="demo",
    )
    assert created["config"].radius == 1.2
    assert created["mesh_name"] == "demo"

    monkeypatch.setattr(
        opt_mesh_module.gmshio,
        "read_from_msh",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad gdim")),
    )
    assert opt_mesh_module._load_cached_mesh(tmp_path, "missing", gdim=2) is None
    (tmp_path / "broken.msh").write_text("msh", encoding="utf-8")
    assert opt_mesh_module._load_cached_mesh(tmp_path, "broken", gdim=2) is None

    good_mesh = tmp_path / "assoc_only.msh"
    good_mesh.write_text("msh", encoding="utf-8")
    assoc = configparser.ConfigParser()
    assoc["other"] = {"domain": "1"}
    with (tmp_path / "assoc_only_association_table.ini").open(
        "w", encoding="utf-8"
    ) as fh:
        assoc.write(fh)

    monkeypatch.setattr(
        opt_mesh_module.gmshio,
        "read_from_msh",
        lambda *args, **kwargs: _fake_mesh_data(),
    )
    monkeypatch.setattr(
        opt_mesh_module,
        "association_from_mesh_data",
        lambda _mesh_data: {"domain": 1, "gaps": 3},
    )
    monkeypatch.setattr(opt_mesh_module, "estimate_radius", lambda _mesh: 0.4)
    monkeypatch.setattr(
        opt_mesh_module,
        "build_eit_mesh",
        lambda *args, **kwargs: SimpleNamespace(
            mesh="mesh",
            topology=SimpleNamespace(dim=2),
            facet_tags=kwargs["facet_tags"],
            association_table=kwargs["association_table"],
            mesh_family=None,
        ),
    )
    monkeypatch.setattr(
        opt_mesh_module, "infer_mesh_family_from_mesh", lambda _mesh: "tetra"
    )
    loaded = opt_mesh_module._load_cached_mesh(tmp_path, "assoc_only", gdim=2)
    assert loaded is not None
    assert loaded.association_table["domain"] == 1
    assert loaded.association_table["electrode_1"] == 2
    assert loaded.association_table["gaps"] == 18


def test_cached_3d_validator_and_load_or_create_branches(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    mesh_missing_domain = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"gaps": 2},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision=opt_mesh_module.DEFAULT_3D_GENERATOR_REVISION,
        mesh_file=None,
    )
    assert (
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_missing_domain, n_elec=2)
        is False
    )

    mesh_no_facet = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags=None,
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision=opt_mesh_module.DEFAULT_3D_GENERATOR_REVISION,
        mesh_file="mesh.msh",
    )
    assert (
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_no_facet, n_elec=2)
        is False
    )

    mesh_exc = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision=opt_mesh_module.DEFAULT_3D_GENERATOR_REVISION,
        mesh_file="mesh.msh",
    )
    monkeypatch.setattr(
        opt_mesh_module.ufl,
        "Measure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad ds")),
    )
    assert opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_exc, n_elec=2) is False

    monkeypatch.setattr(opt_mesh_module.ufl, "Measure", lambda *args, **kwargs: "ds")
    monkeypatch.setattr(
        opt_mesh_module.fem, "Constant", lambda mesh, value: ("const", mesh, value)
    )
    monkeypatch.setattr(opt_mesh_module.fem, "form", lambda expr: expr)
    monkeypatch.setattr(opt_mesh_module.fem, "assemble_scalar", lambda expr: 0.0)
    mesh_zero = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
        mesh_family="tetra",
        geometry_version="legacy",
        generator_revision="g3d0",
        mesh_file="mesh.msh",
    )
    assert opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_zero, n_elec=2) is False

    mesh_sidecar = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: 1.0),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision=opt_mesh_module.DEFAULT_3D_GENERATOR_REVISION,
        mesh_file=str(tmp_path / "hex_mesh.msh"),
    )
    Path(mesh_sidecar.mesh_file).write_text("msh", encoding="utf-8")
    sidecar_path = opt_mesh_module.structured_sidecar_path_for_mesh(
        mesh_sidecar.mesh_file
    )
    sidecar_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        opt_mesh_module,
        "load_structured_sidecar",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad sidecar")),
    )
    assert (
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_sidecar, n_elec=2) is False
    )

    with pytest.raises(ValueError, match="dimension must be 2 or 3"):
        opt_mesh_module.load_or_create_mesh(mesh_dir=str(tmp_path), dimension=4)

    monkeypatch.setattr(
        opt_mesh_module, "_load_cached_mesh", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        opt_mesh_module,
        "create_eit_mesh",
        lambda **kwargs: SimpleNamespace(
            mesh_file=str(tmp_path / "mesh2d.msh"), kind="2d", kwargs=kwargs
        ),
    )
    monkeypatch.setattr(
        opt_mesh_module, "put_process_cached_mesh", lambda *args, **kwargs: None
    )
    mesh2d = opt_mesh_module.load_or_create_mesh(
        mesh_dir=str(tmp_path),
        mesh_name=None,
        n_elec=8,
        dimension=2,
        radius=1.0,
        refinement=3,
        electrode_coverage=0.4,
        geometry_dtype=np.float32,
        extra_flag="unused",
    )
    assert mesh2d.kind == "2d"
    assert mesh2d.kwargs["geometry_dtype"] == np.dtype(np.float32)
    assert mesh2d.kwargs["mesh_name"] == "mesh_8e_r1_ref3_cov0p4_f32"


def test_load_or_create_3d_ring_order_uses_distinct_cache_and_generator_kwargs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    created_calls: list[dict[str, object]] = []

    def _fake_create_cylinder_3d_eit_mesh(**kwargs):
        created_calls.append(dict(kwargs))
        return SimpleNamespace(
            mesh_file=str(tmp_path / f"{kwargs['mesh_name']}.msh"),
            kind="3d",
            kwargs=kwargs,
        )

    monkeypatch.setattr(
        opt_mesh_module, "_load_cached_mesh", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        opt_mesh_module,
        "create_cylinder_3d_eit_mesh",
        _fake_create_cylinder_3d_eit_mesh,
    )
    monkeypatch.setattr(
        opt_mesh_module, "put_process_cached_mesh", lambda *args, **kwargs: None
    )

    mesh = opt_mesh_module.load_or_create_mesh(
        mesh_dir=str(tmp_path),
        mesh_name=None,
        n_elec=16,
        dimension=3,
        radius=0.18,
        height=0.16,
        refinement=2,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        electrode_level_fractions=(0.25, 0.75),
        z_center=0.0,
        mesh_family="hex",
        geometry_version="geomv2",
        electrode_layout="ring_major",
    )

    assert mesh.kind == "3d"
    assert len(created_calls) == 1
    call = created_calls[0]
    assert call["electrode_layout"] == "ring_major"
    assert "_elring_major_" in str(call["mesh_name"])


def test_cached_3d_validator_covers_nonfinite_measure_and_sidecar_exception_paths(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        opt_mesh_module.ufl, "Measure", lambda *args, **kwargs: lambda tag: float(tag)
    )
    monkeypatch.setattr(
        opt_mesh_module.fem, "Constant", lambda _mesh, value: float(value)
    )
    monkeypatch.setattr(opt_mesh_module.fem, "form", lambda expr: expr)

    mesh_nonfinite = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
        mesh_family="tetra",
        geometry_version="legacy",
        generator_revision="g3d0",
        mesh_file="mesh.msh",
    )
    monkeypatch.setattr(
        opt_mesh_module.fem,
        "assemble_scalar",
        lambda expr: float("nan") if expr == 3.0 else 1.0,
    )
    assert (
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_nonfinite, n_elec=2)
        is False
    )

    mesh_file = tmp_path / "validator_sidecar_fail.msh"
    mesh_file.write_text("msh", encoding="utf-8")
    sidecar_path = opt_mesh_module.structured_sidecar_path_for_mesh(mesh_file)
    sidecar_path.write_text("{}", encoding="utf-8")
    mesh_sidecar = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: 1.0),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision=opt_mesh_module.DEFAULT_3D_GENERATOR_REVISION,
        mesh_file=str(mesh_file),
    )
    monkeypatch.setattr(opt_mesh_module.fem, "assemble_scalar", lambda expr: 1.0)
    monkeypatch.setattr(
        opt_mesh_module,
        "load_structured_sidecar",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad sidecar")),
    )
    assert (
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_sidecar, n_elec=2) is False
    )
