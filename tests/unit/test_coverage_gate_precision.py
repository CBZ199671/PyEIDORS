from __future__ import annotations

import importlib
from pathlib import Path
import tomllib

from coverage.results import should_fail_under
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v637_pytest_coverage_gate_uses_two_decimal_precision() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
    precision = config["tool"]["coverage"]["report"]["precision"]

    assert "--cov-fail-under=87" in addopts
    assert precision == 2
    assert should_fail_under(86.72, 87, precision) is True
    assert should_fail_under(87.0, 87, precision) is False


@pytest.mark.parametrize(
    ("package_name", "export_name", "submodule_name"),
    [
        ("pyeidors.data", "PatternConfig", "structures"),
        ("pyeidors.electrodes", "StimMeasPatternManager", "layout"),
        ("pyeidors.femx", "function_size", "helpers"),
        ("pyeidors.geometry", "MeshConverter", None),
        ("pyeidors.interop", "STANDARD_INTEROP_FORMAT", "geometry_exchange"),
        ("pyeidors.inverse.matrix_free", "DualMeshJacobianOperator", "dual_mesh"),
        ("pyeidors.inverse.prior", "graph_laplacian", "laplace"),
        ("pyeidors.inverse.reduced", "InexactController", "inexact_controller"),
        (
            "pyeidors.inverse.regularization",
            "BaseRegularization",
            "base_regularization",
        ),
        ("pyeidors.inverse.workflows", "perform_absolute_reconstruction", "absolute"),
        ("pyeidors.io", "HDF5Artifact", "hdf5_artifacts"),
        ("pyeidors.perf", "DEFAULT_ACCELERATION_PROFILE", "policy"),
        ("pyeidors.physics", "UnitCheckLevel", "unit_consistency"),
    ],
)
def test_package_lazy_getattr_and_dir_branches(
    package_name: str, export_name: str, submodule_name: str | None
) -> None:
    package = importlib.import_module(package_name)
    package.__dict__.pop(export_name, None)

    value = getattr(package, export_name)
    assert value is not None
    assert export_name in dir(package)

    if submodule_name is not None:
        package.__dict__.pop(submodule_name, None)
        submodule = getattr(package, submodule_name)
        assert submodule.__name__ == f"{package_name}.{submodule_name}"
        assert submodule_name in dir(package)

    with pytest.raises(AttributeError):
        getattr(package, "__missing_v637__")


def test_realtime_cell_to_node_averages_adjacent_cells_and_orphans() -> None:
    from pyeidors.realtime.mesh_utils import cell_to_node

    class Mesh:
        def cells(self) -> list[list[int]]:
            return [[0, 1, 2], [2, 3, 0]]

        def num_vertices(self) -> int:
            return 5

    values = cell_to_node(Mesh(), np.array([2.0, 4.0]))

    np.testing.assert_allclose(values, [3.0, 2.0, 3.0, 4.0, 0.0])
