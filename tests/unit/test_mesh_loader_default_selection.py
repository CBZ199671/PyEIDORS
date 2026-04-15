from __future__ import annotations

import os
import time
from pathlib import Path

from pyeidors.geometry.mesh_loader import MeshLoader


def test_get_default_mesh_prefers_latest_compatible_dimension(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old_2d = tmp_path / "mesh_old_2d.msh"
    new_2d = tmp_path / "mesh_new_2d.msh"
    mesh3d = tmp_path / "mesh3d_latest.msh"

    old_2d.write_text("old-2d", encoding="utf-8")
    new_2d.write_text("new-2d", encoding="utf-8")
    mesh3d.write_text("newest-3d", encoding="utf-8")

    now = time.time()
    os.utime(old_2d, (now - 30, now - 30))
    os.utime(new_2d, (now - 10, now - 10))
    os.utime(mesh3d, (now, now))

    loader = MeshLoader(mesh_dir=str(tmp_path), gdim=2)
    seen: list[str] = []

    def _fake_load_mesh(mesh_name: str):
        seen.append(mesh_name)
        return mesh_name

    monkeypatch.setattr(loader, "load_mesh", _fake_load_mesh)

    assert loader.get_default_mesh() == "mesh_new_2d"
    assert seen == ["mesh_new_2d"]
