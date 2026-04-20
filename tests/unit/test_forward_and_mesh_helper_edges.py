"""Extra helper-branch tests for forward-model and mesh-cache utilities."""

from __future__ import annotations

import configparser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.data.structures import EITMesh, PatternConfig
import pyeidors.forward.eit_forward_model as forward_module
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.geometry import optimized_mesh_generator as opt_mesh_module


class _IntDict(dict):
    def __int__(self) -> int:
        return 0


class _FakeMat:
    def __init__(
        self,
        mat_type: str = "seqaij",
        *,
        raise_on_get_type: bool = False,
        raise_on_set_type: bool = False,
    ):
        self.mat_type = str(mat_type)
        self.raise_on_get_type = raise_on_get_type
        self.raise_on_set_type = raise_on_set_type
        self.convert_result = None
        self.destroy_calls = 0
        self.set_type_calls: list[str] = []

    def getType(self):
        if self.raise_on_get_type:
            raise RuntimeError("getType failed")
        return self.mat_type

    def convert(self, _mat_type):
        return self.convert_result

    def destroy(self):
        self.destroy_calls += 1
        raise RuntimeError("destroy failed")

    def setType(self, mat_type):
        if self.raise_on_set_type:
            raise RuntimeError("setType failed")
        self.set_type_calls.append(str(mat_type))
        self.mat_type = str(mat_type)


class _FakeVec:
    def __init__(
        self,
        vec_type: str = "seq",
        *,
        raise_on_get_type: bool = False,
        raise_on_set_type: bool = False,
    ):
        self.vec_type = str(vec_type)
        self.raise_on_get_type = raise_on_get_type
        self.raise_on_set_type = raise_on_set_type
        self.set_type_calls: list[str] = []

    def getType(self):
        if self.raise_on_get_type:
            raise RuntimeError("getType failed")
        return self.vec_type

    def setType(self, vec_type):
        if self.raise_on_set_type:
            raise RuntimeError("setType failed")
        self.set_type_calls.append(str(vec_type))
        self.vec_type = str(vec_type)


def test_resolve_electrode_tags_supports_nested_mapping_and_integer_fallback():
    model = EITForwardModel.__new__(EITForwardModel)
    model.n_elec = 2
    model.association_table = {"electrodes": _IntDict({"1": 10, "2": 11})}
    assert model._resolve_electrode_tags() == [10, 11]

    model.association_table = {2: 20, 5: 50}
    assert model._resolve_electrode_tags() == [20, 50]


def test_compute_electrode_boundary_measures_warns_on_zero_measure(
    monkeypatch: pytest.MonkeyPatch,
):
    model = EITForwardModel.__new__(EITForwardModel)
    model.electrode_tags = [7, 9]
    model.mesh = SimpleNamespace(
        comm=SimpleNamespace(allreduce=lambda value, op=None: value)
    )
    model.ds_electrodes = lambda tag: float(tag)

    monkeypatch.setattr(
        forward_module.fem, "Constant", lambda _mesh, value: float(value)
    )
    monkeypatch.setattr(forward_module.fem, "form", lambda expr: expr)
    monkeypatch.setattr(
        forward_module.fem,
        "assemble_scalar",
        lambda expr: 0.0 if np.isclose(expr, 7.0) else 2.5,
    )

    with pytest.warns(RuntimeWarning, match="zero measure"):
        measures = model._compute_electrode_boundary_measures()

    assert measures[7] == 0.0
    assert measures[9] == 2.5


def test_vec_to_numpy_falls_back_to_getarray():
    class _Vec:
        @staticmethod
        def getArray(*, readonly=True):
            _ = readonly
            return np.array([1.0, 2.0, 3.0], dtype=float)

    out = EITForwardModel._vec_to_numpy(_Vec())
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0], dtype=float))


def test_resolve_pattern_matrix_and_type_helpers(monkeypatch: pytest.MonkeyPatch):
    model = EITForwardModel.__new__(EITForwardModel)
    model.n_elec = 4
    model.pattern_manager = SimpleNamespace(stim_matrix=np.eye(4, dtype=float))

    with pytest.raises(ValueError, match="2D array"):
        model._resolve_pattern_matrix(np.ones(4, dtype=float))

    with pytest.raises(ValueError, match="shape mismatch"):
        model._resolve_pattern_matrix(np.ones((2, 3), dtype=float))

    model.pattern_manager = SimpleNamespace(stim_matrix=np.ones((2, 3), dtype=float))
    with pytest.raises(ValueError, match="Pattern width mismatch"):
        model._resolve_pattern_matrix()

    vec_with_array = SimpleNamespace(array=np.array([4.0, 5.0], dtype=float))
    np.testing.assert_allclose(
        EITForwardModel._vec_to_numpy(vec_with_array), np.array([4.0, 5.0], dtype=float)
    )

    monkeypatch.setattr(forward_module, "PETSc", object())
    converted = _FakeMat("densecuda")
    mat = _FakeMat("seqaij")
    mat.convert_result = converted
    assert EITForwardModel._ensure_mat_type(mat, "densecuda") is converted
    assert mat.destroy_calls == 1

    mat_get_fail = _FakeMat("seqaij", raise_on_get_type=True, raise_on_set_type=True)
    assert EITForwardModel._ensure_mat_type(mat_get_fail, "aij") is mat_get_fail

    vec_get_fail = _FakeVec("seq", raise_on_get_type=True, raise_on_set_type=True)
    assert EITForwardModel._ensure_vec_type(vec_get_fail, "cuda") is vec_get_fail


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


def _minimal_eit_mesh(*, comm_size: int = 1, facet_tags="facet-tags") -> EITMesh:
    return EITMesh(
        mesh=SimpleNamespace(comm=SimpleNamespace(size=comm_size)),
        facet_tags=facet_tags,
    )


def test_forward_model_constructor_and_electrode_tag_validation_cover_error_paths():
    pattern = PatternConfig(n_elec=2)

    with pytest.raises(TypeError, match="expects an EITMesh instance"):
        EITForwardModel(
            n_elec=2,
            pattern_config=pattern,
            z=np.ones(2, dtype=float),
            mesh=object(),
        )

    with pytest.raises(RuntimeError, match="MPI size=1 only"):
        EITForwardModel(
            n_elec=2,
            pattern_config=pattern,
            z=np.ones(2, dtype=float),
            mesh=_minimal_eit_mesh(comm_size=2),
        )

    with pytest.raises(ValueError, match="Contact impedance length"):
        EITForwardModel(
            n_elec=2,
            pattern_config=pattern,
            z=np.ones(1, dtype=float),
            mesh=_minimal_eit_mesh(),
        )

    with pytest.raises(ValueError, match="lacks electrode facet tags"):
        EITForwardModel(
            n_elec=2,
            pattern_config=pattern,
            z=np.ones(2, dtype=float),
            mesh=_minimal_eit_mesh(facet_tags=None),
        )

    model = EITForwardModel.__new__(EITForwardModel)
    model.n_elec = 2
    model.association_table = {
        "electrodes": _IntDict({"1": "bad"}),
        "electrode_2": "bad",
        2: "bad",
    }
    with pytest.raises(ValueError, match="missing electrode tags"):
        model._resolve_electrode_tags()


def test_load_cached_mesh_reads_association_section_and_sidecar_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mesh_name = "with_sidecar"
    (tmp_path / f"{mesh_name}.msh").write_text("msh", encoding="utf-8")
    assoc = configparser.ConfigParser()
    assoc["ASSOCIATION TABLE"] = {"domain": "1", "electrode_1": "2", "gaps": "3"}
    with (tmp_path / f"{mesh_name}_association_table.ini").open(
        "w", encoding="utf-8"
    ) as fh:
        assoc.write(fh)

    sidecar_path = opt_mesh_module.structured_sidecar_path_for_mesh(
        tmp_path / f"{mesh_name}.msh"
    )
    sidecar_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        opt_mesh_module.gmshio,
        "read_from_msh",
        lambda *args, **kwargs: _fake_mesh_data(),
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
            geometry_version=kwargs["geometry_version"],
            generator_revision=kwargs["generator_revision"],
            mesh_family=None,
        ),
    )
    monkeypatch.setattr(
        opt_mesh_module, "infer_mesh_family_from_mesh", lambda _mesh: "hex"
    )
    monkeypatch.setattr(
        opt_mesh_module,
        "load_structured_sidecar",
        lambda _path: {"geometry_version": "geomv9", "generator_revision": "g9"},
    )

    loaded = opt_mesh_module._load_cached_mesh(tmp_path, mesh_name, gdim=2)
    assert loaded is not None
    assert loaded.association_table["domain"] == 1
    assert loaded.association_table["electrode_1"] == 2
    assert loaded.association_table["gaps"] == 18
    assert loaded.geometry_version == "geomv9"
    assert loaded.generator_revision == "g9"


def test_cached_3d_mesh_validator_shortcuts_and_sidecar_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mesh_2d = SimpleNamespace(topology=SimpleNamespace(dim=2))
    assert opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_2d, n_elec=2) is True

    mesh_hex_missing_file = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision=opt_mesh_module.DEFAULT_3D_GENERATOR_REVISION,
        mesh_file=None,
    )
    monkeypatch.setattr(
        opt_mesh_module.ufl, "Measure", lambda *args, **kwargs: (lambda tag: float(tag))
    )
    monkeypatch.setattr(
        opt_mesh_module.fem, "Constant", lambda _mesh, value: float(value)
    )
    monkeypatch.setattr(opt_mesh_module.fem, "form", lambda expr: expr)
    monkeypatch.setattr(opt_mesh_module.fem, "assemble_scalar", lambda expr: expr)
    assert (
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_hex_missing_file, n_elec=2)
        is False
    )

    mesh_file = tmp_path / "hex_ok.msh"
    mesh_file.write_text("msh", encoding="utf-8")
    sidecar_path = opt_mesh_module.structured_sidecar_path_for_mesh(mesh_file)
    sidecar_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        opt_mesh_module, "load_structured_sidecar", lambda _path: {"ok": True}
    )

    mesh_ok = SimpleNamespace(
        topology=SimpleNamespace(dim=3),
        association_table={"domain": 1, "gaps": 2, "electrode_1": 3, "electrode_2": 4},
        facet_tags="facet",
        mesh="mesh",
        comm=SimpleNamespace(allreduce=lambda value, op=None: value),
        mesh_family="hex",
        geometry_version="geomv2",
        generator_revision=opt_mesh_module.DEFAULT_3D_GENERATOR_REVISION,
        mesh_file=str(mesh_file),
    )
    assert opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_ok, n_elec=2) is True


def test_cached_3d_validator_handles_nonfinite_measures_and_sidecar_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        opt_mesh_module.ufl, "Measure", lambda *args, **kwargs: (lambda tag: float(tag))
    )
    monkeypatch.setattr(
        opt_mesh_module.fem, "Constant", lambda _mesh, value: float(value)
    )
    monkeypatch.setattr(opt_mesh_module.fem, "form", lambda expr: expr)

    mesh_bad_measure = SimpleNamespace(
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
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_bad_measure, n_elec=2)
        is False
    )

    mesh_file = tmp_path / "hex_bad_sidecar.msh"
    mesh_file.write_text("msh", encoding="utf-8")
    sidecar_path = opt_mesh_module.structured_sidecar_path_for_mesh(mesh_file)
    sidecar_path.write_text("{}", encoding="utf-8")
    mesh_bad_sidecar = SimpleNamespace(
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
        lambda _path: (_ for _ in ()).throw(RuntimeError("broken sidecar")),
    )
    assert (
        opt_mesh_module._cached_3d_cem_mesh_is_complete(mesh_bad_sidecar, n_elec=2)
        is False
    )


def test_clockwise_electrode_positions_and_cached_mesh_sidecar_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    clockwise = opt_mesh_module.ElectrodePosition(
        L=4, coverage=0.5, anticlockwise=False
    ).positions
    anticlockwise = opt_mesh_module.ElectrodePosition(
        L=4, coverage=0.5, anticlockwise=True
    ).positions
    assert clockwise[0] == anticlockwise[0]
    assert clockwise[1:] == anticlockwise[1:][::-1]

    mesh_name = "cached_3d_sidecar_fail"
    msh_file = tmp_path / f"{mesh_name}.msh"
    msh_file.write_text("msh", encoding="utf-8")
    sidecar_path = opt_mesh_module.structured_sidecar_path_for_mesh(msh_file)
    sidecar_path.write_text("{}", encoding="utf-8")

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
    monkeypatch.setattr(opt_mesh_module, "estimate_radius", lambda _mesh: 0.5)
    monkeypatch.setattr(
        opt_mesh_module,
        "load_structured_sidecar",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad sidecar")),
    )
    monkeypatch.setattr(
        opt_mesh_module,
        "build_eit_mesh",
        lambda *args, **kwargs: SimpleNamespace(
            mesh="mesh",
            topology=SimpleNamespace(dim=3),
            facet_tags=kwargs["facet_tags"],
            association_table=kwargs["association_table"],
            geometry_version=kwargs["geometry_version"],
            generator_revision=kwargs["generator_revision"],
            mesh_family=None,
        ),
    )
    monkeypatch.setattr(
        opt_mesh_module, "infer_mesh_family_from_mesh", lambda _mesh: "tetra"
    )
    monkeypatch.setattr(
        opt_mesh_module, "_cached_3d_cem_mesh_is_complete", lambda _mesh, n_elec: False
    )

    assert (
        opt_mesh_module._load_cached_mesh(tmp_path, mesh_name, gdim=3, n_elec=16)
        is None
    )
