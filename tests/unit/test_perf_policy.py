"""Tests for shared performance policy helpers."""

from __future__ import annotations

import pytest

from pyeidors.perf.policy import (
    parse_block_size_candidates,
    resolve_experimental_mode,
    resolve_forward_mat_solve,
    resolve_line_search_mode,
    resolve_solver_mode,
)


def test_resolve_solver_mode_defaults_by_mesh_dimension():
    assert resolve_solver_mode("auto", mesh_dim=3) == "fast"
    assert resolve_solver_mode("auto", mesh_dim=2) == "strict"
    assert resolve_solver_mode("strict", mesh_dim=3) == "strict"


def test_resolve_line_search_mode_defaults_by_mesh_dimension():
    assert resolve_line_search_mode("auto", mesh_dim=3) == "fast"
    assert resolve_line_search_mode("auto", mesh_dim=2) == "full"
    assert resolve_line_search_mode("fast", mesh_dim=2) == "fast"


def test_resolve_experimental_mode_keeps_features_opt_in():
    assert resolve_experimental_mode("auto") == "off"
    assert resolve_experimental_mode("on") == "on"
    assert resolve_experimental_mode("unexpected") == "off"


def test_resolve_forward_mat_solve_only_keeps_auto_for_3d_fast():
    assert resolve_forward_mat_solve("auto", mesh_dim=3, solver_mode="fast") == "auto"
    assert resolve_forward_mat_solve("auto", mesh_dim=3, solver_mode="strict") == "off"
    assert resolve_forward_mat_solve("auto", mesh_dim=2, solver_mode="fast") == "off"
    assert resolve_forward_mat_solve("on", mesh_dim=2, solver_mode="strict") == "on"


def test_parse_block_size_candidates_normalizes_strings_and_iterables():
    assert parse_block_size_candidates("256, 64, 256, 128") == [64, 128, 256]
    assert parse_block_size_candidates([512, "128", 512, 64]) == [64, 128, 512]


@pytest.mark.parametrize("bad_value", ["", "0,0", "abc", [0, 0]])
def test_parse_block_size_candidates_rejects_invalid_input(bad_value):
    with pytest.raises(ValueError):
        parse_block_size_candidates(bad_value)
