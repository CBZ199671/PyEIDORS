"""Branch-focused tests for 3D mesh helper logic and generator dispatch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pyeidors.geometry.mesh3d_generator as mesh3d_module


class _FakeGroup:
    def __init__(self, tag: int):
        self.tag = int(tag)


class _FakeOcc:
    def __init__(self):
        self.counter = 0
        self.physical_groups = []
        self.center_of_mass = {}

    def _next(self) -> int:
        self.counter += 1
        return self.counter

    def addPoint(self, _x, _y, _z):
        return self._next()

    def addLine(self, _p0, _p1):
        return self._next()

    def addCurveLoop(self, _lines):
        return self._next()

    def addPlaneSurface(self, _loops):
        return self._next()

    def extrude(self, _entities, _dx, _dy, _dz):
        return [(2, 20), (2, 21), (2, 22), (2, 23), (3, 99)]

    def synchronize(self):
        return None

    def getCenterOfMass(self, _dim, tag):
        return self.center_of_mass[int(tag)]


class _FakeMeshModel:
    def __init__(self):
        self.size_calls = []
        self.generate_calls = []

    def setSize(self, entities, size):
        self.size_calls.append((tuple(entities), float(size)))

    def generate(self, dim):
        self.generate_calls.append(int(dim))


class _FakeModel:
    def __init__(self):
        self.occ = _FakeOcc()
        self.mesh = _FakeMeshModel()
        self._bbox = {}
        self._boundary = {}
        self.physical_groups = []
        self._entities = {0: [(0, 1)], 2: []}
        self.name = None

    def add(self, name):
        self.name = str(name)

    def getEntities(self, dim):
        return list(self._entities.get(int(dim), []))

    def getBoundingBox(self, _dim, tag):
        return self._bbox[int(tag)]

    def getBoundary(self, entities, oriented=False, recursive=False):
        _ = (oriented, recursive)
        return list(self._boundary.get(int(entities[0][1]), []))

    def addPhysicalGroup(self, dim, entities, tag, name=None):
        self.physical_groups.append(
            (int(dim), tuple(int(v) for v in entities), int(tag), name)
        )


class _FakeGmsh:
    def __init__(self):
        self.model = _FakeModel()
        self.initialized = False
        self.writes = []

    def initialize(self):
        self.initialized = True

    def finalize(self):
        self.initialized = False

    def isInitialized(self):
        return self.initialized

    def clear(self):
        self.model = _FakeModel()

    def write(self, path):
        self.writes.append(str(path))
        Path(path).write_text("msh", encoding="utf-8")


def _valid_sidecar_payload() -> dict:
    return {
        "version": mesh3d_module.STRUCTURED_SIDECAR_VERSION,
        "mesh_family": "hex",
        "geometry_version": "geomv2",
        "generator_revision": "g3d3",
        "block_topology": ["core"],
        "blocks": [{"id": 0}],
        "structured_node_to_mesh_node": [0, 1],
        "structured_cell_to_block": [0],
        "structured_cell_local_ijk": [[0, 0, 0]],
        "boundary_faces": [],
        "field_tags": {"domain": 1},
    }


def test_level_fraction_and_config_validation_branches():
    assert (
        mesh3d_module.normalize_electrode_level_fractions(None)
        == mesh3d_module.DEFAULT_ZIGZAG_LEVEL_FRACTIONS
    )
    assert mesh3d_module.normalize_electrode_level_fractions(0.5) == (0.5,)
    assert mesh3d_module.normalize_electrode_level_fractions([0.2, 0.8]) == (0.2, 0.8)
    with pytest.raises(ValueError, match="at least one entry"):
        mesh3d_module.normalize_electrode_level_fractions(())
    with pytest.raises(ValueError, match="entries must be in"):
        mesh3d_module.normalize_electrode_level_fractions([0.0, 0.8])

    with pytest.raises(ValueError, match="radius must be positive"):
        mesh3d_module.Cylinder3DMeshConfig(radius=0.0)
    with pytest.raises(ValueError, match="height must be positive"):
        mesh3d_module.Cylinder3DMeshConfig(height=0.0)
    with pytest.raises(ValueError, match="refinement must be positive"):
        mesh3d_module.Cylinder3DMeshConfig(refinement=0)
    with pytest.raises(ValueError, match="electrode_vertices must be >= 2"):
        mesh3d_module.Cylinder3DMeshConfig(electrode_vertices=1)
    with pytest.raises(ValueError, match="gap_vertices must be >= 0"):
        mesh3d_module.Cylinder3DMeshConfig(gap_vertices=-1)
    with pytest.raises(ValueError, match="electrode_height_ratio must be in"):
        mesh3d_module.Cylinder3DMeshConfig(electrode_height_ratio=1.5)
    with pytest.raises(ValueError, match="require at least two"):
        mesh3d_module.Cylinder3DMeshConfig(electrode_level_fractions=(0.5,))
    with pytest.raises(ValueError, match="electrode windows overlap"):
        mesh3d_module.Cylinder3DMeshConfig(
            electrode_height_ratio=0.9,
            electrode_level_fractions=(0.4, 0.45),
        )

    cfg = mesh3d_module.Cylinder3DMeshConfig(
        radius=2.0, height=4.0, z_center=1.0, refinement=4
    )
    assert cfg.mesh_size == 0.25
    assert cfg.z_min == -1.0
    assert cfg.z_max == 3.0


def test_electrode_arc_angle_classification_and_window_helpers():
    with pytest.raises(ValueError, match="positive integer"):
        mesh3d_module.ElectrodeArcConfig(n_elec=0)
    with pytest.raises(ValueError, match="coverage must be in"):
        mesh3d_module.ElectrodeArcConfig(n_elec=8, coverage=1.5)

    arc = mesh3d_module.ElectrodeArcConfig(
        n_elec=4, coverage=0.5, rotation=0.1, anticlockwise=False
    )
    positions = arc.positions
    assert len(positions) == 4
    assert positions[1:] == positions[1:][::-1][::-1]

    assert mesh3d_module._normalize_angle(-0.5) > 0.0
    assert mesh3d_module._angle_in_arc(0.0, -0.1, 0.1)
    assert mesh3d_module._angle_in_arc(0.0, 2 * np.pi - 0.1, 0.1)
    assert mesh3d_module._classify_theta(0.0, [(0.0, 0.2), (1.0, 1.2)]) == 1
    assert mesh3d_module._classify_theta(0.5, [(0.0, 0.2)]) is None

    cfg = mesh3d_module.Cylinder3DMeshConfig(
        height=1.0, electrode_height_ratio=0.2, electrode_level_fractions=(0.25, 0.75)
    )
    windows = mesh3d_module._electrode_vertical_windows(cfg)
    assert len(windows) == 2
    mid0 = 0.5 * (windows[0][0] + windows[0][1])
    assert mesh3d_module._window_contains(mid0, windows[0])
    assert mesh3d_module._find_electrode_window_index(mid0, cfg) == 0
    assert mesh3d_module._find_electrode_window_index(0.5, cfg) is None
    assert len(mesh3d_module._build_z_stage_breakpoints(cfg)) >= 4
    intervals = mesh3d_module._z_stage_intervals(cfg)
    assert all(stop > start for start, stop in intervals)

    with pytest.raises(ValueError, match="Resolved electrode window collapsed"):
        mesh3d_module.Cylinder3DMeshConfig(
            height=1e-12,
            refinement=1,
            electrode_height_ratio=1.0,
            electrode_level_fractions=(0.25, 0.75),
        )

    electrode_positions = mesh3d_module.ElectrodeArcConfig(
        n_elec=2, coverage=0.5
    ).positions
    assert mesh3d_module._classify_sidewall_patch(
        theta=0.0, z_center=0.5, positions=electrode_positions, config=cfg
    ) == ("blank_side", None)
    assert mesh3d_module._classify_sidewall_patch(
        theta=0.0, z_center=mid0, positions=[(1.0, 1.2)], config=cfg
    ) == ("gaps", None)
    assert mesh3d_module._classify_sidewall_patch(
        theta=0.0,
        z_center=0.5,
        positions=electrode_positions,
        config=mesh3d_module.Cylinder3DMeshConfig(electrode_level_fractions=(0.1, 0.9)),
    )[0] in {"blank_side", "gaps", "electrode"}

    ring_arc = mesh3d_module.ElectrodeArcConfig(n_elec=8, coverage=0.5)
    assert (
        mesh3d_module._total_3d_electrode_count(
            config=cfg,
            electrodes=ring_arc,
        )
        == 16
    )
    ring_positions = ring_arc.positions
    theta0 = 0.5 * (ring_positions[0][0] + ring_positions[0][1])
    mid1 = 0.5 * (windows[1][0] + windows[1][1])
    assert mesh3d_module._classify_sidewall_patch(
        theta=theta0,
        z_center=mid0,
        positions=ring_positions,
        config=cfg,
    ) == ("electrode", 1)
    assert mesh3d_module._classify_sidewall_patch(
        theta=theta0,
        z_center=mid1,
        positions=ring_positions,
        config=cfg,
    ) == ("electrode", 9)


def test_sidecar_and_output_path_helpers_cover_validation_and_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _valid_sidecar_payload()
    assert mesh3d_module.validate_structured_sidecar_payload(dict(payload)) == payload

    bad_cases = [
        ({}, "missing required keys"),
        ({**payload, "version": "bad"}, "version mismatch"),
        ({**payload, "mesh_family": "tetra"}, "mesh_family='hex' only"),
        ({**payload, "geometry_version": "legacy"}, "geometry_version='geomv2' only"),
        ({**payload, "blocks": []}, "at least one block"),
        (
            {**payload, "structured_node_to_mesh_node": []},
            "structured_node_to_mesh_node",
        ),
        ({**payload, "structured_cell_to_block": []}, "structured_cell_to_block"),
        ({**payload, "structured_cell_local_ijk": "bad"}, "structured_cell_local_ijk"),
        ({**payload, "structured_cell_local_ijk": []}, "cell metadata length mismatch"),
        ({**payload, "boundary_faces": "bad"}, "boundary_faces as a list"),
        ({**payload, "field_tags": {}}, "non-empty field_tags"),
    ]
    for case, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            mesh3d_module.validate_structured_sidecar_payload(case)

    assoc_path = tmp_path / "assoc.ini"
    mesh3d_module.write_association_table(assoc_path, {"domain": 1, "electrode_1": 2})
    assert "electrode_1 = 2" in assoc_path.read_text(encoding="utf-8")

    sidecar_path = mesh3d_module.structured_sidecar_path_for_mesh(tmp_path / "demo.msh")
    assert sidecar_path.name == "demo_structured.json"

    out_sidecar = tmp_path / "demo_structured.json"
    mesh3d_module._write_structured_sidecar(out_sidecar, payload)
    loaded = mesh3d_module.load_structured_sidecar(out_sidecar)
    assert loaded["field_tags"]["domain"] == 1

    monkeypatch.setattr(
        mesh3d_module.tempfile, "mkdtemp", lambda: str(tmp_path / "auto")
    )
    monkeypatch.setattr(mesh3d_module.time, "time", lambda: 123.456789)
    out_dir, mesh_name, msh_path, assoc = mesh3d_module._prepare_output_paths(
        output_dir=None, mesh_name=None, prefix="mesh3d"
    )
    assert out_dir == tmp_path / "auto"
    assert mesh_name.startswith("mesh3d_")
    assert msh_path.name.endswith(".msh")
    assert assoc.name.endswith("_association_table.ini")


def test_surface_selection_square_to_disk_and_hex_geometry_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_gmsh = _FakeGmsh()
    fake_gmsh.model._bbox = {
        10: (0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
        11: (0.0, 0.0, 2.0, 1.0, 1.0, 2.0),
        12: (0.0, 0.0, 1.8, 1.0, 1.0, 2.2),
        20: (0.0, 0.0, 0.0, 1.0, 1.0, 2.0),
        21: (0.0, 0.0, 0.0, 1.0, 1.0, 1e-9),
    }
    monkeypatch.setattr(mesh3d_module, "gmsh", fake_gmsh)

    with pytest.raises(RuntimeError, match="Expected a top surface"):
        mesh3d_module._top_surface_from_extrusion([(1, 3)])
    assert mesh3d_module._top_surface_from_extrusion([(2, 10), (2, 11), (2, 12)]) == 11
    assert mesh3d_module._lateral_surfaces_from_extrusion(
        [(2, 20), (2, 11), (2, 21)], top_surface=11
    ) == [20]

    assert mesh3d_module._square_to_disk(0.0, 0.0) == (0.0, 0.0)
    x1, y1 = mesh3d_module._square_to_disk(0.8, 0.2)
    x2, y2 = mesh3d_module._square_to_disk(0.2, 0.8)
    assert np.isfinite([x1, y1, x2, y2]).all()

    cfg = mesh3d_module.Cylinder3DMeshConfig(
        refinement=1, electrode_level_fractions=(0.25, 0.75)
    )
    electrodes = mesh3d_module.ElectrodeArcConfig(n_elec=4, coverage=0.5)
    gen_square = mesh3d_module._GeomV2HexCylinder3DMeshGenerator(
        cfg,
        electrodes,
        generator_revision=mesh3d_module.SQUARE_TO_DISK_3D_GENERATOR_REVISION,
    )
    z_levels = gen_square._z_levels()
    assert z_levels[0] == pytest.approx(cfg.z_min)
    assert z_levels[-1] == pytest.approx(cfg.z_max)
    points, hexes, meta = gen_square._structured_geometry_square_to_disk()
    assert points.shape[1] == 3
    assert hexes.shape[1] == 8
    assert meta["block_topology"] == ["square_to_disk"]
    assert gen_square._structured_geometry()[2]["block_topology"] == ["square_to_disk"]

    gen_o = mesh3d_module._GeomV2HexCylinder3DMeshGenerator(
        cfg, electrodes, generator_revision="g3d3"
    )
    points_o, hexes_o, meta_o = gen_o._structured_geometry_o_grid()
    assert points_o.shape[1] == 3
    assert hexes_o.shape[1] == 8
    assert meta_o["block_topology"] == ["core", "east", "north", "west", "south"]

    faces = gen_o._cell_faces(np.arange(8, dtype=np.int32))
    assert len(faces) == 6

    points_small = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    hexes_small = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    quads, tags, field_data, boundary_faces = gen_o._boundary_quads(
        points_small, hexes_small
    )
    assert quads.shape[1] == 4
    assert tags.shape[0] == quads.shape[0]
    assert "top" in field_data and "bottom" in field_data
    assert len(boundary_faces) == quads.shape[0]


def test_legacy_and_geomv2_tetra_generator_helpers_and_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cfg = mesh3d_module.Cylinder3DMeshConfig(
        refinement=1,
        electrode_vertices=3,
        gap_vertices=1,
        electrode_level_fractions=(0.25, 0.75),
        electrode_layout="zigzag",
    )
    electrodes = mesh3d_module.ElectrodeArcConfig(n_elec=4, coverage=0.5)

    fake_gmsh = _FakeGmsh()
    fake_gmsh.model._entities[2] = [(2, 30), (2, 31), (2, 32)]
    fake_gmsh.model._bbox = {
        30: (0.0, 0.0, 0.0, 1.0, 1.0, cfg.height),
        31: (0.0, 0.0, 0.0, 1.0, 1.0, 1e-10),
        32: (0.0, 0.0, 0.0, 1.0, 1.0, cfg.height),
        20: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        21: (0.0, 0.0, 0.2, 1.0, 1.0, 0.8),
        22: (0.0, 0.0, 0.2, 1.0, 1.0, 0.8),
        23: (0.0, 0.0, 0.2, 1.0, 1.0, 0.8),
    }
    fake_gmsh.model._boundary = {
        30: [(1, 1), (1, 2)],
        31: [(1, 3)],
        32: [(1, 4)],
    }
    fake_gmsh.model.occ.center_of_mass = {
        21: (1.0, 0.0, 0.25),
        22: (0.0, 1.0, 0.75),
        23: (-1.0, 0.0, 0.25),
    }
    monkeypatch.setattr(mesh3d_module, "gmsh", fake_gmsh)

    legacy = mesh3d_module._LegacyTetraCylinder3DMeshGenerator(cfg, electrodes)
    geom = legacy._create_geometry()
    assert geom["volume_tag"] == 99
    assert geom["lines"]
    side_surfaces = legacy._resolve_side_surfaces([1, 2, 4], cfg.height)
    assert side_surfaces == [30, 32]
    mapped = legacy._map_side_surfaces_to_lines([30, 32], [1, 2, 4])
    assert mapped[1] == 30 and mapped[4] == 32

    with pytest.raises(RuntimeError, match="electrode_3"):
        legacy._set_physical_groups(
            {
                "volume_tag": 99,
                "lines": [1, 2, 3, 4],
                "electrode_ranges": [(0, 1), (1, 2), (2, 3), (3, 4)],
                "side_surfaces": [30, 32],
                "side_by_line": {1: 30, 2: 32},
            }
        )

    legacy._set_physical_groups(
        {
            "volume_tag": 99,
            "lines": [1, 2, 3, 4],
            "electrode_ranges": [(0, 1), (1, 2), (2, 3), (3, 4)],
            "side_surfaces": [30, 32, 40, 41],
            "side_by_line": {1: 30, 2: 32, 3: 40, 4: 41},
        }
    )
    assert any(group[3] == "gaps" for group in fake_gmsh.model.physical_groups) is False

    geomv2 = mesh3d_module._GeomV2TetraCylinder3DMeshGenerator(
        cfg,
        mesh3d_module.ElectrodeArcConfig(n_elec=1, coverage=0.5),
        generator_revision="g3d3",
    )
    base_surface, lines = geomv2._create_base_surface()
    assert isinstance(base_surface, int)
    assert len(lines) > 0

    monkeypatch.setattr(mesh3d_module, "GMSH_AVAILABLE", False)
    with pytest.raises(ImportError, match="gmsh Python bindings"):
        geomv2.generate(output_dir=tmp_path, mesh_name="geomv2_fail")
    with pytest.raises(ImportError, match="gmsh Python bindings"):
        legacy.generate(output_dir=tmp_path, mesh_name="legacy_fail")
    monkeypatch.setattr(mesh3d_module, "GMSH_AVAILABLE", True)

    fake_mesh_data = SimpleNamespace(
        mesh="mesh",
        facet_tags="facet",
        cell_tags="cell",
        physical_groups={
            "domain": _FakeGroup(1),
            "gaps": _FakeGroup(6),
            **{f"electrode_{idx}": _FakeGroup(idx + 1) for idx in range(1, 5)},
        },
    )
    monkeypatch.setattr(
        mesh3d_module.gmshio, "model_to_mesh", lambda *_args, **_kwargs: fake_mesh_data
    )
    monkeypatch.setattr(mesh3d_module, "estimate_radius", lambda _mesh: 1.0)
    monkeypatch.setattr(
        mesh3d_module,
        "build_eit_mesh",
        lambda *_args, **kwargs: SimpleNamespace(
            mesh_family=kwargs["mesh_family"],
            geometry_version=kwargs["geometry_version"],
            generator_revision=kwargs["generator_revision"],
            association_table=kwargs["association_table"],
            mesh_file=kwargs["mesh_file"],
        ),
    )

    monkeypatch.setattr(
        legacy,
        "_create_geometry",
        lambda: {
            "volume_tag": 99,
            "lines": [1],
            "electrode_ranges": [(0, 0)] * 4,
            "side_surfaces": [30],
            "side_by_line": {1: 30},
        },
    )
    monkeypatch.setattr(legacy, "_set_physical_groups", lambda _geometry: None)
    out_legacy = legacy.generate(output_dir=tmp_path, mesh_name="legacy_ok")
    assert out_legacy.geometry_version == "legacy"

    monkeypatch.setattr(
        mesh3d_module, "_top_surface_from_extrusion", lambda _extruded: 20
    )
    monkeypatch.setattr(
        mesh3d_module,
        "_lateral_surfaces_from_extrusion",
        lambda _extruded, _top: [21, 22, 23],
    )
    monkeypatch.setattr(
        mesh3d_module, "_find_electrode_window_index", lambda _z, _cfg: 0
    )
    fake_gmsh.clear = lambda: None
    fake_gmsh.model.occ.center_of_mass = {
        21: (1.0, 0.0, 0.25),
        22: (0.0, 1.0, 0.75),
        23: (-1.0, 0.0, 0.25),
    }
    geomv2_small = mesh3d_module._GeomV2TetraCylinder3DMeshGenerator(
        cfg,
        mesh3d_module.ElectrodeArcConfig(n_elec=2, coverage=0.5),
        generator_revision="g3d3",
    )
    sequence = iter([("electrode", 1), ("gaps", None), ("electrode", 2)])
    monkeypatch.setattr(
        mesh3d_module, "_classify_sidewall_patch", lambda **_kwargs: next(sequence)
    )
    out_geomv2 = geomv2_small.generate(output_dir=tmp_path, mesh_name="geomv2_ok")
    assert out_geomv2.geometry_version == "geomv2"

    class _FakeGenerator:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def generate(self, output_dir=None, mesh_name=None):
            return SimpleNamespace(output_dir=output_dir, mesh_name=mesh_name)

    monkeypatch.setattr(
        mesh3d_module, "_GeomV2HexCylinder3DMeshGenerator", _FakeGenerator
    )
    monkeypatch.setattr(
        mesh3d_module, "_LegacyTetraCylinder3DMeshGenerator", _FakeGenerator
    )
    monkeypatch.setattr(
        mesh3d_module, "_GeomV2TetraCylinder3DMeshGenerator", _FakeGenerator
    )

    with pytest.raises(ValueError, match="mesh_family='hex' currently supports"):
        mesh3d_module.create_cylinder_3d_eit_mesh(
            mesh_family="hex", geometry_version="legacy"
        )

    out_hex = mesh3d_module.create_cylinder_3d_eit_mesh(
        output_dir=str(tmp_path),
        mesh_name="hex_ok",
        mesh_family="hex",
        geometry_version="geomv2",
    )
    assert out_hex.mesh_name == "hex_ok"

    out_legacy_dispatch = mesh3d_module.create_cylinder_3d_eit_mesh(
        output_dir=str(tmp_path),
        mesh_name="legacy_dispatch",
        mesh_family="tetra",
        geometry_version="legacy",
    )
    assert out_legacy_dispatch.mesh_name == "legacy_dispatch"

    out_geom_dispatch = mesh3d_module.create_cylinder_3d_eit_mesh(
        output_dir=str(tmp_path),
        mesh_name="geom_dispatch",
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d7",
    )
    assert out_geom_dispatch.mesh_name == "geom_dispatch"


def test_mesh3d_generator_remaining_edge_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cfg = mesh3d_module.Cylinder3DMeshConfig(
        refinement=1,
        electrode_vertices=3,
        gap_vertices=1,
        electrode_level_fractions=(0.25, 0.75),
        electrode_layout="zigzag",
    )
    electrodes = mesh3d_module.ElectrodeArcConfig(n_elec=4, coverage=0.5)
    fake_gmsh = _FakeGmsh()
    monkeypatch.setattr(mesh3d_module, "gmsh", fake_gmsh)

    legacy = mesh3d_module._LegacyTetraCylinder3DMeshGenerator(cfg, electrodes)
    fake_gmsh.model.occ.extrude = lambda *_args, **_kwargs: [(2, 20)]
    with pytest.raises(RuntimeError, match="Failed to create 3D volume"):
        legacy._create_geometry()

    fake_gmsh.model._boundary = {30: [(2, 99), (1, 8), (1, 1)]}
    mapped = legacy._map_side_surfaces_to_lines([30], [1, 2])
    assert mapped == {1: 30}

    fake_gmsh.model.physical_groups.clear()
    legacy._set_physical_groups(
        {
            "volume_tag": 99,
            "lines": [1, 2, 3, 4],
            "electrode_ranges": [(0, 1), (1, 2), (2, 3), (3, 4)],
            "side_surfaces": [30, 31, 32, 33, 40],
            "side_by_line": {1: 30, 2: 31, 3: 32, 4: 33},
        }
    )
    assert any(group[3] == "gaps" for group in fake_gmsh.model.physical_groups)

    fake_mesh_data = SimpleNamespace(
        mesh="mesh",
        facet_tags="facet",
        cell_tags="cell",
        physical_groups={
            "domain": _FakeGroup(1),
            "gaps": _FakeGroup(6),
            **{f"electrode_{idx}": _FakeGroup(idx + 1) for idx in range(1, 5)},
        },
    )
    monkeypatch.setattr(mesh3d_module, "GMSH_AVAILABLE", True)
    monkeypatch.setattr(
        mesh3d_module.gmshio, "model_to_mesh", lambda *_args, **_kwargs: fake_mesh_data
    )
    monkeypatch.setattr(mesh3d_module, "estimate_radius", lambda _mesh: 1.0)
    monkeypatch.setattr(
        mesh3d_module,
        "build_eit_mesh",
        lambda *_args, **kwargs: SimpleNamespace(
            mesh_family=kwargs["mesh_family"],
            geometry_version=kwargs["geometry_version"],
            generator_revision=kwargs["generator_revision"],
            association_table=kwargs["association_table"],
            mesh_file=kwargs["mesh_file"],
        ),
    )

    clear_calls = {"legacy": 0, "geomv2": 0}
    fake_gmsh.initialized = True

    def _track_clear_legacy():
        clear_calls["legacy"] += 1
        fake_gmsh.model = _FakeModel()

    fake_gmsh.clear = _track_clear_legacy
    monkeypatch.setattr(
        legacy,
        "_create_geometry",
        lambda: {
            "volume_tag": 99,
            "lines": [1],
            "electrode_ranges": [(0, 0)] * 4,
            "side_surfaces": [30],
            "side_by_line": {1: 30},
        },
    )
    monkeypatch.setattr(legacy, "_set_physical_groups", lambda _geometry: None)
    out_legacy = legacy.generate(output_dir=tmp_path, mesh_name="legacy_initialized")
    assert out_legacy.geometry_version == "legacy"
    assert clear_calls["legacy"] >= 1

    geomv2 = mesh3d_module._GeomV2TetraCylinder3DMeshGenerator(
        cfg,
        mesh3d_module.ElectrodeArcConfig(n_elec=1, coverage=0.5),
        generator_revision="g3d3",
    )

    def _track_clear_geomv2():
        clear_calls["geomv2"] += 1
        fake_gmsh.model = _FakeModel()
        fake_gmsh.model.occ.center_of_mass = {21: (1.0, 0.0, 0.25)}

    fake_gmsh.clear = _track_clear_geomv2
    monkeypatch.setattr(geomv2, "_create_base_surface", lambda: (1, [1]))
    fake_gmsh.model.occ.extrude = lambda *_args, **_kwargs: [(2, 20), (2, 21), (3, 99)]
    monkeypatch.setattr(
        mesh3d_module, "_top_surface_from_extrusion", lambda _extruded: 20
    )
    monkeypatch.setattr(
        mesh3d_module, "_lateral_surfaces_from_extrusion", lambda _extruded, _top: [21]
    )
    monkeypatch.setattr(
        mesh3d_module, "_find_electrode_window_index", lambda _z, _cfg: 0
    )
    monkeypatch.setattr(
        mesh3d_module, "_classify_sidewall_patch", lambda **_kwargs: ("electrode", 1)
    )
    out_geomv2 = geomv2.generate(output_dir=tmp_path, mesh_name="geomv2_initialized")
    assert out_geomv2.geometry_version == "geomv2"
    assert clear_calls["geomv2"] >= 1

    geomv2_missing = mesh3d_module._GeomV2TetraCylinder3DMeshGenerator(
        cfg,
        mesh3d_module.ElectrodeArcConfig(n_elec=2, coverage=0.5),
        generator_revision="g3d3",
    )
    monkeypatch.setattr(mesh3d_module, "_z_stage_intervals", lambda _cfg: [(0.0, 0.0)])
    with pytest.raises(RuntimeError, match="Failed to create 3D geomv2 tetra volume"):
        geomv2_missing.generate(output_dir=tmp_path, mesh_name="geomv2_no_volume")

    monkeypatch.setattr(
        mesh3d_module, "_z_stage_intervals", lambda _cfg: [(0.0, 0.0), (0.0, 1.0)]
    )
    fake_gmsh.model.occ.extrude = lambda *_args, **_kwargs: [(2, 20), (2, 21), (3, 99)]
    monkeypatch.setattr(
        mesh3d_module, "_lateral_surfaces_from_extrusion", lambda _extruded, _top: [21]
    )
    monkeypatch.setattr(
        mesh3d_module, "_classify_sidewall_patch", lambda **_kwargs: ("gaps", None)
    )
    with pytest.raises(RuntimeError, match="electrode_1"):
        geomv2_missing.generate(output_dir=tmp_path, mesh_name="geomv2_missing_surface")

    monkeypatch.setattr(mesh3d_module, "MESHIO_AVAILABLE", False)
    with pytest.raises(ImportError, match="meshio is required"):
        mesh3d_module._GeomV2HexCylinder3DMeshGenerator(cfg, electrodes).generate(
            output_dir=tmp_path
        )
    monkeypatch.setattr(mesh3d_module, "MESHIO_AVAILABLE", True)

    gen_hex = mesh3d_module._GeomV2HexCylinder3DMeshGenerator(cfg, electrodes)
    monkeypatch.setattr(
        mesh3d_module,
        "_z_stage_intervals",
        lambda _cfg: [(0.0, 0.24), (0.24, 0.48), (0.48, 0.72), (0.72, 1.0)],
    )
    z_levels = gen_hex._z_levels()
    assert z_levels[0] == pytest.approx(0.0)
    assert z_levels[-1] == pytest.approx(1.0)

    monkeypatch.setattr(gen_hex, "_signed_quad_area", lambda _coords: -1.0)
    _, _, meta = gen_hex._structured_geometry_o_grid()
    assert meta["block_topology"][0] == "core"
