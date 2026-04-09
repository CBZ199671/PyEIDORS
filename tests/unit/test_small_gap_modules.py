"""Small high-yield gap tests across remaining utility modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib
import numpy as np
import pytest

import pyeidors.inverse.solvers.sparse_projection as sparse_projection
import pyeidors.utils.plot_font_i18n as font_i18n
from pyeidors.data.structures import EITImage
from pyeidors.inverse.reduced.inexact_controller import InexactController
from pyeidors.inverse.contracts import SolverOutput
from pyeidors.inverse.workflows.base import resolve_reconstruction_output
from pyeidors.visualization.eit_plot_renderers import render_reconstruction_comparison

matplotlib.use("Agg")


def test_register_optional_fonts_warns_when_addfont_fails(monkeypatch: pytest.MonkeyPatch):
    first = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    font_i18n._WARNED_KEYS.discard(f"font-register-{first}")
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self == first:
            return True
        return original_exists(self)

    def boom(_path: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(font_i18n.font_manager.fontManager, "addfont", boom)
    font_i18n._register_optional_fonts()
    assert f"font-register-{first}" in font_i18n._WARNED_KEYS


def test_render_reconstruction_comparison_saves_output(eit_system, tmp_path: Path):
    viz = SimpleNamespace(
        _text=lambda key, **kwargs: key if not kwargs else f"{key}:{kwargs['value']:.4f}",
    )
    sigma = np.ones(eit_system.mesh.num_cells(), dtype=float)
    save_path = tmp_path / "compare.png"
    fig = render_reconstruction_comparison(
        viz,
        eit_system.mesh,
        sigma,
        sigma * 1.1,
        title="demo",
        save_path=str(save_path),
    )
    assert fig is not None
    assert save_path.exists()


def test_resolve_reconstruction_output_rejects_invalid_payload_types():
    with pytest.raises(TypeError, match="Expected SolverOutput"):
        resolve_reconstruction_output(reconstruction="bad", fwd_model=object())

    fake_output = SolverOutput(
        conductivity="bad",
        residual_history=None,
        sigma_change_history=None,
    )
    with pytest.raises(TypeError, match="must be a DOLFINx Function or numpy array"):
        resolve_reconstruction_output(reconstruction=fake_output, fwd_model=object())


def test_sparse_projection_for_zero_width_and_nonfinite_lipschitz(monkeypatch: pytest.MonkeyPatch):
    rng = mock.MagicMock()
    rng.standard_normal.return_value = np.zeros(0, dtype=float)
    vec = sparse_projection._init_power_vector(np.empty((2, 0), dtype=float), rng)
    assert vec.size == 0

    calls = {"n": 0}

    def fake_safe_dot(_A, _b, _label):
        calls["n"] += 1
        if calls["n"] <= 3:
            return np.ones(2, dtype=float)
        return np.array([np.nan, np.nan], dtype=float)

    monkeypatch.setattr(sparse_projection, "safe_dot", fake_safe_dot)
    result = sparse_projection.estimate_lipschitz_constant(np.eye(2, dtype=float), iters=1)
    assert result == pytest.approx(2e-12)


def test_inexact_controller_invalid_mode_falls_back_to_eisenstat_walker():
    ctl = InexactController(mode="invalid", eta0=0.2, eta_min=1e-3, eta_max=0.5)
    assert ctl.mode == "eisenstat-walker"
