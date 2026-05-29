from __future__ import annotations

import inspect

import numpy as np

import eit_app.ui.electrode_overlay as overlay_module
from eit_app.ui.electrode_overlay import (
    ElectrodeArcSpec,
    ElectrodePatchSpec,
    default_arc_segments,
    default_patch_quads,
)


def test_v303_electrode_overlay_geometry_direct_fills_segments(monkeypatch) -> None:
    def _fail_stack(*_args, **_kwargs):
        raise AssertionError("electrode overlay geometry must direct-fill")

    monkeypatch.setattr(overlay_module.np, "column_stack", _fail_stack)
    monkeypatch.setattr(overlay_module.np, "vstack", _fail_stack)
    assert "np.column_stack" not in inspect.getsource(default_arc_segments)
    patch_source = inspect.getsource(default_patch_quads)
    assert "np.column_stack" not in patch_source
    assert "np.vstack" not in patch_source
    assert "tris.append" not in patch_source
    assert "triangles = np.empty" in patch_source

    segments = default_arc_segments(
        [ElectrodeArcSpec(theta_start=0.0, theta_end=np.pi / 2.0)],
        radius=2.0,
        n_samples=3,
    )
    points, triangles = default_patch_quads(
        [ElectrodePatchSpec(0.0, np.pi / 2.0, -0.5, 0.5)],
        radius=2.0,
        n_theta=3,
    )

    assert len(segments) == 1
    assert segments[0].shape == (3, 2)
    assert points.shape == (6, 3)
    assert points.dtype == np.float32
    assert triangles.shape == (4, 3)
