"""Optional ADIOS4DOLFINx checkpoint helper tests."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from pyeidors.geometry import adios4dolfinx_checkpoint as checkpoint


def test_adios4dolfinx_checkpoint_missing_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(checkpoint, "adios4dolfinx_available", lambda: False)
    mesh_data = SimpleNamespace(mesh=object(), facet_tags=None, cell_tags=None)
    assert (
        checkpoint.write_adios4dolfinx_checkpoint(
            mesh_data,
            source_msh_file=tmp_path / "mesh.msh",
        )
        is None
    )


def test_adios4dolfinx_checkpoint_writer_delegates_mesh_and_tags(
    tmp_path: Path, monkeypatch
):
    calls: list[tuple[str, object]] = []

    class _FakeAdios4Dolfinx:
        @staticmethod
        def write_mesh(filename, mesh, **kwargs):
            calls.append(("mesh", filename, mesh, kwargs))
            Path(filename).mkdir(parents=True, exist_ok=True)

        @staticmethod
        def write_meshtags(filename, mesh, tags, **kwargs):
            calls.append(("meshtags", filename, tags, kwargs))

    monkeypatch.setattr(checkpoint, "adios4dolfinx_available", lambda: True)
    monkeypatch.setitem(sys.modules, "adios4dolfinx", _FakeAdios4Dolfinx)

    mesh_data = SimpleNamespace(
        mesh=object(),
        facet_tags=SimpleNamespace(name=""),
        cell_tags=SimpleNamespace(name=""),
    )
    out = checkpoint.write_adios4dolfinx_checkpoint(
        mesh_data,
        source_msh_file=tmp_path / "mesh.msh",
        engine="BP5",
    )

    assert out == str(tmp_path / "mesh_adios4dolfinx.bp")
    assert [call[0] for call in calls] == ["mesh", "meshtags", "meshtags"]
    assert calls[0][3]["engine"] == "BP5"
    assert calls[1][3]["meshtag_name"] == "facet_tags"
    assert calls[2][3]["meshtag_name"] == "cell_tags"
