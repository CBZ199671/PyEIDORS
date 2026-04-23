"""Tests for semantic cache object signatures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import textwrap
import time

import numpy as np

from pyeidors.cache.object_signature import (
    backend_signature_from_forward_model,
    signature_of_cache_obj,
    stable_signature_hash,
)


def _load_function(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.demo


def test_signature_is_stable_for_semantically_equal_payloads():
    payload_a = {
        "alpha": 1,
        "beta": np.arange(8, dtype=np.float64).reshape(2, 4),
        "gamma": {"x": 2, "y": 3},
    }
    payload_b = {
        "gamma": {"y": 3, "x": 2},
        "beta": np.arange(8, dtype=np.float64).reshape(2, 4),
        "alpha": 1,
    }
    assert signature_of_cache_obj(payload_a) == signature_of_cache_obj(payload_b)
    assert stable_signature_hash(payload_a) == stable_signature_hash(payload_b)


def test_callable_signature_changes_when_source_mtime_changes(tmp_path: Path):
    module_path = tmp_path / "callable_source.py"
    module_path.write_text(
        textwrap.dedent(
            """
            def demo(x):
                return x + 1
            """
        ),
        encoding="utf-8",
    )
    func = _load_function(module_path, "demo_mod_1")
    hash_before = stable_signature_hash(func)

    time.sleep(0.01)
    module_path.write_text(
        textwrap.dedent(
            """
            def demo(x):
                return x + 2
            """
        ),
        encoding="utf-8",
    )
    func_after = _load_function(module_path, "demo_mod_2")
    hash_after = stable_signature_hash(func_after)

    assert hash_before != hash_after


def test_backend_signature_is_stable_across_cuda_petsc_alias_resolution():
    backend_config = SimpleNamespace(
        ksp_type="preonly",
        pc_type="lu",
        rtol=1e-10,
        atol=1e-12,
        max_it=2000,
        reuse_preconditioner=True,
        mat_solve_mode="auto",
        petsc_device="cuda",
    )
    mesh = SimpleNamespace(comm=SimpleNamespace(Get_size=lambda: 1))
    eit_mesh = SimpleNamespace(structured_sidecar_version=None)

    pre_solve = SimpleNamespace(
        backend_config=backend_config,
        linear_backend="petsc",
        forward_backend="dolfinx",
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d3",
        performance_mode="aggressive",
        mesh=mesh,
        eit_mesh=eit_mesh,
        _petsc_backend_info={
            "petsc_device_effective": "cuda",
            "petsc_mat_type": "aijcusparse",
            "petsc_vec_type": "cuda",
            "petsc_dense_mat_type": "densecuda",
            "gpu_constraint_strategy": None,
            "forward_backend_effective": "dolfinx",
        },
    )
    post_solve = SimpleNamespace(
        backend_config=backend_config,
        linear_backend="petsc",
        forward_backend="dolfinx",
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d3",
        performance_mode="aggressive",
        mesh=mesh,
        eit_mesh=eit_mesh,
        _petsc_backend_info={
            "petsc_device_effective": "cuda",
            "petsc_mat_type": "seqaijcusparse",
            "petsc_vec_type": "cuda",
            "petsc_dense_mat_type": "seqdensecuda",
            "gpu_constraint_strategy": "electrode-zero",
            "forward_backend_effective": "dolfinx",
        },
    )

    assert backend_signature_from_forward_model(
        pre_solve
    ) == backend_signature_from_forward_model(post_solve)
