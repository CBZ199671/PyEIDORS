"""T76 entrance gate: sparse Bayesian module consolidation contract.

Path C consolidation collapsed the sparse solver tier from 7 modules
to 5: the legacy ``sparse_bayesian.py`` import alias and the
``sparse_bayesian_backends.py`` mixin file are gone, the mixin
methods now live directly on
:class:`SparseBayesianReconstructor`. These tests freeze that shape so
a future contributor cannot silently re-introduce the indirection
layer or resurrect the alias module.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def test_legacy_alias_module_removed() -> None:
    """``pyeidors.inverse.solvers.sparse_bayesian`` no longer exists."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pyeidors.inverse.solvers.sparse_bayesian")


def test_mixin_module_removed() -> None:
    """``pyeidors.inverse.solvers.sparse_bayesian_backends`` no longer exists."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pyeidors.inverse.solvers.sparse_bayesian_backends")


def test_solvers_package_exports_reconstructor_directly_from_engine() -> None:
    """``solvers/__init__.py`` re-exports the canonical engine class."""
    from pyeidors.inverse.solvers import (
        SparseBayesianConfig,
        SparseBayesianReconstructor,
    )
    from pyeidors.inverse.solvers import sparse_bayesian_engine

    assert (
        SparseBayesianReconstructor
        is sparse_bayesian_engine.SparseBayesianReconstructor
    )
    assert SparseBayesianConfig is sparse_bayesian_engine.SparseBayesianConfig


def test_reconstructor_has_no_mixin_in_mro() -> None:
    """``SparseBayesianReconstructor`` is a plain class, not a mixin subclass."""
    from pyeidors.inverse.solvers.sparse_bayesian_engine import (
        SparseBayesianReconstructor,
    )

    mro_names = [cls.__name__ for cls in SparseBayesianReconstructor.__mro__]
    assert mro_names == ["SparseBayesianReconstructor", "object"]


def test_reconstructor_owns_all_folded_wrapper_methods() -> None:
    """All 14 historical mixin methods are now defined directly on the class."""
    from pyeidors.inverse.solvers.sparse_bayesian_engine import (
        SparseBayesianReconstructor,
    )

    expected_methods = {
        "_solve_sparse_map",
        "_coarse_initialization",
        "_get_coarse_matrix",
        "_compute_projection",
        "_estimate_lipschitz_constant",
        "_solve_with_cuqi_map",
        "_solve_fista",
        "_solve_irls",
        "_multilevel_correction",
        "_block_refinement",
        "_linear_model",
        "_sparse_prior",
        "_gaussian_likelihood",
        "_bayesian_problem",
    }
    own_attrs = set(vars(SparseBayesianReconstructor))
    missing = expected_methods - own_attrs
    assert not missing, f"reconstructor missing folded mixin methods: {missing!r}"


def test_engine_module_imports_kernel_helpers_directly() -> None:
    """Engine module-level globals expose the kernel functions used by tests."""
    from pyeidors.inverse.solvers import sparse_bayesian_engine

    for name in (
        "solve_sparse_map",
        "coarse_initialization",
        "multilevel_correction",
        "block_refinement",
        "solve_fista",
        "solve_irls",
        "build_coarse_hierarchy",
        "compute_projection",
        "estimate_lipschitz_constant",
        "get_coarse_matrix",
    ):
        assert hasattr(sparse_bayesian_engine, name), (
            f"engine module must re-export kernel helper {name!r} for monkeypatch consumers"
        )
        assert callable(getattr(sparse_bayesian_engine, name))


def test_solve_sparse_map_signature_unchanged() -> None:
    """Public reconstruction entrypoint signature stays stable across the refactor."""
    from pyeidors.inverse.solvers.sparse_bayesian_engine import (
        SparseBayesianReconstructor,
    )

    sig = inspect.signature(SparseBayesianReconstructor._solve_sparse_map)
    assert list(sig.parameters) == [
        "self",
        "jacobian",
        "data_vector",
        "noise_sigma",
        "prior_scale",
    ]
