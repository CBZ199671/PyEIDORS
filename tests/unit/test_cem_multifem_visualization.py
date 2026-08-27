"""V812 common-mesh CEM forward-visualization regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.benchmarks.cem_block_audit import (
    assemble_analytic_blocks,
    build_nonuniform_fixture,
)
from scripts.benchmarks.cem_multifem_common import solve_robin_from_blocks
from scripts.benchmarks.cem_multifem_visualization import (
    METHOD_ORDER,
    TargetConfig,
    assemble_p1_cem_blocks,
    build_forward_fields,
    render_visualizations,
)


def _synthetic_sources() -> tuple[dict, TargetConfig]:
    fixture = build_nonuniform_fixture()
    conductivity = np.full(
        fixture.cells.shape[0], fixture.conductivity, dtype=np.float64
    )
    blocks = assemble_p1_cem_blocks(
        fixture.nodes,
        fixture.cells,
        fixture.tagged_edges,
        conductivity,
        fixture.contact_impedance,
    )
    solution = solve_robin_from_blocks(
        K=blocks["K"],
        B=blocks["B"],
        C_plus=blocks["C_plus"],
        D=blocks["D"],
        currents=fixture.currents,
    )
    centroids = np.mean(fixture.nodes[fixture.cells], axis=1)
    center = centroids[0]
    nonzero_distances = np.linalg.norm(centroids - center, axis=1)
    radius = 0.45 * float(np.min(nonzero_distances[nonzero_distances > 0.0]))
    sources = {
        "nodes": fixture.nodes,
        "cells": fixture.cells,
        "tagged_edges": fixture.tagged_edges,
        "contact_impedance": fixture.contact_impedance,
        "currents": fixture.currents,
        "mfem_body_potential": solution.body_potential,
        "voltages": {
            method: solution.electrode_voltage.copy() for method in METHOD_ORDER
        },
    }
    config = TargetConfig(
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius=radius,
        background_conductivity=fixture.conductivity,
        target_conductivity=1.0,
    )
    return sources, config


def test_v812_neutral_p1_assembly_matches_analytic_fixture() -> None:
    fixture = build_nonuniform_fixture()
    expected, _ = assemble_analytic_blocks(fixture)
    actual = assemble_p1_cem_blocks(
        fixture.nodes,
        fixture.cells,
        fixture.tagged_edges,
        np.full(fixture.cells.shape[0], fixture.conductivity),
        fixture.contact_impedance,
    )
    for key in ("K", "B", "C_plus", "D", "A_R"):
        assert np.allclose(actual[key], expected[key], rtol=5.0e-14, atol=5.0e-14)


def test_v812_target_and_forward_field_share_one_common_mesh() -> None:
    sources, config = _synthetic_sources()
    fields = build_forward_fields(sources, config)
    assert fields.conductivity.shape == (sources["cells"].shape[0],)
    assert fields.target_body.shape == (sources["nodes"].shape[0],)
    assert fields.target_mask.any()
    assert (~fields.target_mask).any()
    assert np.all(fields.conductivity[fields.target_mask] == config.target_conductivity)
    assert np.all(
        fields.conductivity[~fields.target_mask] == config.background_conductivity
    )
    assert max(fields.residuals.values()) < 5.0e-11
    assert np.linalg.norm(fields.body_perturbation) > 0.0
    assert np.linalg.norm(fields.voltage_perturbation) > 0.0


def test_v812_static_png_svg_render(tmp_path: Path) -> None:
    if not Path("/mnt/c/Windows/Fonts/times.ttf").is_file():
        pytest.skip("Times New Roman host font is unavailable")
    sources, config = _synthetic_sources()
    fields = build_forward_fields(sources, config)
    outputs = render_visualizations(sources, fields, config, tmp_path)
    assert len(outputs) == 8
    assert all(path.is_file() and path.stat().st_size > 1000 for path in outputs)
    for path in outputs:
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert "<svg" in path.read_text(encoding="utf-8")[:1000]
