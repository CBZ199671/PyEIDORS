"""Branch-focused tests for loader, core helpers, weighting, and device utilities."""

from __future__ import annotations

import configparser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pyeidors.core_system_helpers as core_helpers
import pyeidors.geometry.mesh_loader as mesh_loader_module
from pyeidors.data.structures import EITData, EITImage
from pyeidors.inverse.solvers import gauss_newton_device as device_module
from pyeidors.inverse.solvers import gauss_newton_weights as weight_module


class _FakeFemFunction:
    def __init__(self, space):
        size = int(getattr(space, "size", 3))
        self.x = SimpleNamespace(array=np.zeros(size, dtype=float))


class _FakeBuiltMesh:
    def __init__(self, association_table):
        self.association_table = association_table
        self.topology = SimpleNamespace(dim=3)
        self.mesh_family = None

    def cells(self):
        return np.zeros((1, 8), dtype=np.int32)

    def num_vertices(self):
        return 8

    def num_cells(self):
        return 1


def test_mesh_loader_helpers_cover_family_version_revision_and_table_parsing(tmp_path: Path):
    mesh2d = SimpleNamespace(topology=SimpleNamespace(dim=2), cells=lambda: np.zeros((1, 3), dtype=np.int32))
    assert mesh_loader_module.infer_mesh_family_from_mesh(mesh2d) is None
    assert mesh_loader_module.infer_mesh_family_from_mesh(
        SimpleNamespace(topology=SimpleNamespace(dim=3), cells=lambda: np.zeros((0, 8), dtype=np.int32))
    ) is None
    assert mesh_loader_module.infer_mesh_family_from_mesh(
        SimpleNamespace(topology=SimpleNamespace(dim=3), cells=lambda: np.zeros((1, 8), dtype=np.int32))
    ) == "hex"
    assert mesh_loader_module.infer_mesh_family_from_mesh(
        SimpleNamespace(topology=SimpleNamespace(dim=3), cells=lambda: np.zeros((1, 4), dtype=np.int32))
    ) == "tetra"
    assert mesh_loader_module.infer_mesh_family_from_mesh(
        SimpleNamespace(topology=SimpleNamespace(dim=3), cells=lambda: np.zeros((1, 6), dtype=np.int32))
    ) is None

    assert mesh_loader_module.infer_geometry_version("demo_geomv2_mesh") == "geomv2"
    assert mesh_loader_module.infer_geometry_version("legacy_mesh") == "legacy"
    assert mesh_loader_module.infer_generator_revision("demo_g3d7_mesh") == "g3d7"
    assert mesh_loader_module.infer_generator_revision("legacy_mesh") == mesh_loader_module.LEGACY_3D_GENERATOR_REVISION

    with pytest.raises(ValueError, match="gdim must be 2 or 3"):
        mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path), gdim=4)
    with pytest.raises(FileNotFoundError, match="Mesh directory does not exist"):
        mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path / "missing"), gdim=2)

    loader = mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path), gdim=2)
    assert loader._load_association_table(tmp_path / "missing.ini") == {}

    cfg = configparser.ConfigParser()
    cfg["boundary_ids"] = {"a": "1", "bad": "not-int"}
    with (tmp_path / "assoc.ini").open("w", encoding="utf-8") as fh:
        cfg.write(fh)
    assert loader._load_association_table(tmp_path / "assoc.ini") == {"a": 1}

    cfg2 = configparser.ConfigParser()
    cfg2["other"] = {"a": "1"}
    with (tmp_path / "assoc2.ini").open("w", encoding="utf-8") as fh:
        cfg2.write(fh)
    assert loader._load_association_table(tmp_path / "assoc2.ini") == {}


def test_mesh_loader_load_mesh_numpy_default_and_factory_cover_remaining_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    loader = mesh_loader_module.MeshLoader(mesh_dir=str(tmp_path), gdim=3)
    with pytest.raises(FileNotFoundError, match="Mesh file does not exist"):
        loader.load_mesh("missing")
    with pytest.raises(FileNotFoundError, match="File does not exist"):
        loader.load_numpy_mesh("missing.npy")
    with pytest.raises(FileNotFoundError, match="No .msh caches found"):
        loader.get_default_mesh()

    mesh_name = "fallback_mesh"
    msh_path = tmp_path / f"{mesh_name}.msh"
    msh_path.write_text("msh", encoding="utf-8")
    sidecar = tmp_path / f"{mesh_name}_structured.json"
    sidecar.write_text("{}", encoding="utf-8")

    class _Group:
        def __init__(self, tag):
            self.tag = tag

    fake_mesh_data = SimpleNamespace(
        mesh="raw-mesh",
        facet_tags="facet-tags",
        cell_tags="cell-tags",
        physical_groups={"domain": _Group(1), "electrode_1": _Group(2)},
    )

    monkeypatch.setattr(mesh_loader_module.gmshio, "read_from_msh", lambda *_args, **_kwargs: fake_mesh_data)
    monkeypatch.setattr(mesh_loader_module, "estimate_radius", lambda _mesh: 0.25)
    monkeypatch.setattr(mesh_loader_module, "structured_sidecar_path_for_mesh", lambda _path: sidecar)
    monkeypatch.setattr(
        mesh_loader_module,
        "load_structured_sidecar",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad sidecar")),
    )
    monkeypatch.setattr(
        mesh_loader_module,
        "build_eit_mesh",
        lambda *_args, **kwargs: _FakeBuiltMesh(kwargs["association_table"]),
    )

    loaded = loader.load_mesh(mesh_name)
    assert loaded.association_table == {"domain": 1, "electrode_1": 2}
    assert loaded.mesh_family == "hex"

    np.save(tmp_path / "array.npy", np.array([1, 2, 3], dtype=int))
    assert loader.load_numpy_mesh("array.npy").tolist() == [1, 2, 3]
    assert mesh_loader_module.create_simple_mesh_loader(str(tmp_path), gdim=3).gdim == 3


def test_core_system_helpers_cover_conductivity_difference_images_and_info(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(core_helpers.fem, "Function", _FakeFemFunction)

    fwd_model = SimpleNamespace(V_sigma=SimpleNamespace(size=4), V=SimpleNamespace(size=6))
    image = EITImage(elem_data=np.array([1.0, 2.0], dtype=float), fwd_model=fwd_model)
    assert core_helpers.conductivity_to_image(fwd_model, image) is image

    fake_fun = _FakeFemFunction(SimpleNamespace(size=3))
    fake_fun.x.array[:] = np.array([0.1, 0.2, 0.3], dtype=float)
    img_from_fun = core_helpers.conductivity_to_image(fwd_model, fake_fun)
    np.testing.assert_allclose(img_from_fun.elem_data, np.array([0.1, 0.2, 0.3], dtype=float))

    arr_img = core_helpers.conductivity_to_image(fwd_model, np.array([3.0, 4.0], dtype=float))
    np.testing.assert_allclose(arr_img.elem_data, np.array([3.0, 4.0], dtype=float))
    with pytest.raises(ValueError, match="Unsupported conductivity input type"):
        core_helpers.conductivity_to_image(fwd_model, 3.14)

    data = EITData(meas=np.array([2.0, 4.0], dtype=float), stim_pattern=None, n_elec=8, n_stim=1, n_meas=2)
    ref = EITData(meas=np.array([1.0, 2.0], dtype=float), stim_pattern=None, n_elec=8, n_stim=1, n_meas=2)
    assert core_helpers.difference_measurement(data, None) is data
    diff = core_helpers.difference_measurement(data, ref, mode="normalized", orientation="reference_minus_target")
    assert diff.type == "difference"
    np.testing.assert_allclose(diff.reference_meas, ref.meas)
    np.testing.assert_allclose(diff.target_meas, data.meas)
    assert diff.difference_mode == "normalized"
    assert diff.difference_orientation == "reference_minus_target"

    eit_system = SimpleNamespace(base_conductivity=0.8, fwd_model=fwd_model)
    homog = core_helpers.create_homogeneous_image(eit_system)
    np.testing.assert_allclose(homog.elem_data, np.full(4, 0.8, dtype=float))

    eit_system.fwd_model.V_sigma = SimpleNamespace(
        size=4,
        tabulate_dof_coordinates=lambda: np.array(
            [[0.0, 0.0], [0.5, 0.0], [2.0, 0.0], [0.0, 2.0]],
            dtype=float,
        ),
    )
    phantom = core_helpers.add_circular_phantom(
        eit_system,
        base_conductivity=1.0,
        phantom_conductivity=2.0,
        phantom_center=(0.0, 0.0),
        phantom_radius=0.75,
    )
    np.testing.assert_allclose(phantom.elem_data, np.array([2.0, 2.0, 1.0, 1.0], dtype=float))

    system = SimpleNamespace(
        n_elec=16,
        pattern_config="pattern",
        mesh_config="mesh",
        difference_mode="raw",
        difference_orientation="target_minus_reference",
        difference_preset="eidors_one_step_noser",
        absolute_preset="eidors_abs_gn",
        hyperparameter=1e-3,
        jacobian_background_conductivity=1.0,
        performance_mode="aggressive",
        linear_backend="petsc",
        cache_scope="process",
        get_cache_stats=lambda: {"hits": 3},
        _is_initialized=False,
        fwd_model=fwd_model,
    )
    info = core_helpers.collect_system_info(system)
    assert info["initialized"] is False
    assert "n_elements" not in info

    system._is_initialized = True
    system.fwd_model.pattern_manager = SimpleNamespace(n_meas_total=32, n_stim=8)
    info2 = core_helpers.collect_system_info(system)
    assert info2["n_elements"] == 4
    assert info2["n_nodes"] == 6
    assert info2["n_measurements"] == 32
    assert info2["n_stimulation_patterns"] == 8


def test_weight_helpers_cover_scaling_difference_and_strategy_selection():
    baseline = np.array([1.0, -1.0], dtype=float)
    measured = np.array([1e-15, -1e-15], dtype=float)
    np.testing.assert_allclose(weight_module.scale_baseline_to_measured(baseline, None), baseline)
    np.testing.assert_allclose(weight_module.scale_baseline_to_measured(np.zeros(2, dtype=float), measured), np.zeros(2))
    np.testing.assert_allclose(weight_module.scale_baseline_to_measured(baseline, measured), baseline)

    diff = weight_module.difference_with_baseline(baseline, np.array([1.1, -0.9], dtype=float), 0.5)
    np.testing.assert_allclose(diff, np.array([0.5, 0.5], dtype=float))
    np.testing.assert_allclose(weight_module.difference_with_baseline(baseline, None, 0.5), baseline)

    np.testing.assert_allclose(
        weight_module.build_weight_reference("scaled_baseline", baseline, measured, 0.5),
        baseline,
    )
    np.testing.assert_allclose(
        weight_module.build_weight_reference("difference", baseline, np.array([1.1, -0.9], dtype=float), 0.5),
        np.array([0.5, 0.5], dtype=float),
    )
    np.testing.assert_allclose(weight_module.build_weight_reference("other", baseline, measured, 0.5), baseline)


def test_gauss_newton_device_helpers_cover_normalization_disable_tf32_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    assert device_module.normalize_runtime_device(None) == "auto"
    assert device_module.normalize_runtime_device("") == "auto"
    assert device_module.normalize_runtime_device("cuda:1") == "cuda:1"
    assert device_module.normalize_runtime_device("mps:0") == "mps:0"
    assert device_module.normalize_runtime_device("weird", default="cpu") == "cpu"
    assert device_module.normalize_runtime_device_label("cuda:1") == "cuda"
    assert device_module.normalize_runtime_device_label("mps:0") == "mps"

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda _value: (_ for _ in ()).throw(RuntimeError("bad")))
    device_module._disable_tf32()
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    auto_cuda = device_module.resolve_torch_device("auto", verbose=True, petsc_device_effective="cuda")
    assert auto_cuda.effective == "cpu"
    assert auto_cuda.fallback_reason == "torch_cuda_unavailable"
    assert "Using CPU for computation" in capsys.readouterr().out

    auto_cpu = device_module.resolve_torch_device("auto", verbose=False, petsc_device_effective="cpu")
    assert auto_cpu.fallback_reason == "auto_cpu_policy"

    with pytest.raises(RuntimeError, match="device='cuda' requires"):
        device_module.resolve_torch_device("cuda", verbose=False, petsc_device_effective="cpu")

    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="device='mps' requested"):
            device_module.resolve_torch_device("mps", verbose=False, petsc_device_effective="cpu")
