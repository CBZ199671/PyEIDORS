"""Small coverage tests for simple mesh wrapper and package main."""

from __future__ import annotations

from types import SimpleNamespace

from pyeidors.geometry import simple_mesh_generator as simple_mesh_module
from pyeidors.main import main


def test_simple_mesh_generator_uses_create_eit_mesh(monkeypatch):
    captured = {}

    def _fake_create_eit_mesh(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(simple_mesh_module, "create_eit_mesh", _fake_create_eit_mesh)

    gen = simple_mesh_module.SimpleEITMeshGenerator(
        n_elec=12,
        radius=1.0,
        mesh_size=0.2,
        electrode_width=0.1,
    )
    out = gen.generate_circular_mesh(output_dir=".", save_files=False)
    assert out.ok is True
    assert captured["n_elec"] == 12
    assert captured["radius"] == 1.0
    assert captured["refinement"] >= 2
    assert captured["electrode_coverage"] > 0

    out2 = simple_mesh_module.create_simple_eit_mesh(
        n_elec=8,
        radius=1.2,
        mesh_size=0.15,
        electrode_coverage=0.25,
        output_dir=".",
    )
    assert out2.ok is True
    assert captured["n_elec"] == 8
    assert captured["radius"] == 1.2
    assert captured["electrode_coverage"] == 0.25


def test_main_entrypoint_prints(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from pyeidors!" in captured.out
