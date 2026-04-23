"""Coverage-oriented tests for runtime helper modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pyeidors.geometry import mesh_converter as mesh_converter_module
from pyeidors.inverse.solvers.gauss_newton_device import resolve_torch_device
from pyeidors.inverse.solvers.sparse_optimizers import solve_fista, solve_irls


def test_mesh_converter_convert_writes_association(monkeypatch, tmp_path: Path):
    class _Group:
        def __init__(self, tag: int):
            self.tag = tag

    fake_mesh_data = SimpleNamespace(
        mesh="mesh_obj",
        facet_tags="facet_tags",
        cell_tags="cell_tags",
        physical_groups={"electrode_1": _Group(2), "electrode_2": _Group(3)},
    )

    monkeypatch.setattr(
        mesh_converter_module.gmshio,
        "read_from_msh",
        lambda *_args, **_kwargs: fake_mesh_data,
    )
    monkeypatch.setattr(
        mesh_converter_module,
        "build_eit_mesh",
        lambda *args, **kwargs: {
            "mesh": args[0],
            "association_table": kwargs["association_table"],
        },
    )

    converter = mesh_converter_module.MeshConverter(
        mesh_file=str(tmp_path / "sample.msh"),
        output_dir=str(tmp_path),
    )
    mesh, facet_tags, association_table = converter.convert()

    assert facet_tags == "facet_tags"
    assert association_table == {"electrode_1": 2, "electrode_2": 3}
    assert mesh["association_table"] == association_table
    assert (tmp_path / "sample_association_table.ini").exists()


def test_resolve_torch_device_cpu_cuda_mps_paths(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cpu_dev = resolve_torch_device("cpu", verbose=False)
    assert cpu_dev.type == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *_args, **_kwargs: "fake-cuda"
    )
    cuda_dev = resolve_torch_device("cuda:0", verbose=False)
    assert cuda_dev.type == "cuda"

    if getattr(torch.backends, "mps", None) is not None:
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        mps_dev = resolve_torch_device("mps", verbose=False)
        assert mps_dev.type == "mps"


def test_sparse_optimizers_gpu_flag_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = SimpleNamespace(
        use_gpu=True,
        gpu_dtype="float32",
        linear_max_iterations=8,
        linear_tolerance=1e-10,
        smoothing_beta=1e-6,
    )

    A = np.eye(6, dtype=float)
    b = np.linspace(-0.2, 0.2, 6)
    warm = np.zeros(6, dtype=float)

    x_fista = solve_fista(
        A, b, noise_sigma=0.1, prior_scale=0.4, warm_start=warm, config=cfg
    )
    x_irls = solve_irls(
        A, b, noise_sigma=0.1, prior_scale=0.4, warm_start=warm, config=cfg
    )

    assert x_fista.shape == (6,)
    assert x_irls.shape == (6,)
    assert np.isfinite(x_fista).all()
    assert np.isfinite(x_irls).all()
