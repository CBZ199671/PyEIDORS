"""Tests for semantic cache object signatures."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import textwrap
import time
import warnings

import numpy as np

from pyeidors.cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
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
            "gpu_constraint_strategy": "reference-electrode-row",
            "forward_backend_effective": "dolfinx",
        },
    )

    assert backend_signature_from_forward_model(
        pre_solve
    ) == backend_signature_from_forward_model(post_solve)


def test_pattern_and_backend_signatures_track_in_place_mutations():
    pattern_manager = SimpleNamespace(
        stim_matrix=np.array([[1.0, -1.0]], dtype=float),
        meas_matrices=[np.eye(2, dtype=float)],
        n_stim=1,
        n_meas_total=2,
        n_meas_per_stim=[2],
    )
    fwd_model = SimpleNamespace(pattern_manager=pattern_manager)

    pattern_before = pattern_signature_from_forward_model(fwd_model)
    pattern_manager.stim_matrix[0, 0] = 2.0
    assert pattern_signature_from_forward_model(fwd_model) != pattern_before

    backend_config = SimpleNamespace(
        ksp_type="preonly",
        pc_type="lu",
        rtol=1e-10,
        atol=1e-12,
        max_it=2000,
        reuse_preconditioner=True,
        mat_solve_mode="auto",
        petsc_device="auto",
    )
    backend_model = SimpleNamespace(
        backend_config=backend_config,
        linear_backend="petsc",
        forward_backend="dolfinx",
        mesh_family="tetra",
        geometry_version="geomv2",
        generator_revision="g3d3",
        performance_mode="aggressive",
        mesh=SimpleNamespace(comm=SimpleNamespace(Get_size=lambda: 1)),
        eit_mesh=SimpleNamespace(structured_sidecar_version="v1"),
        _petsc_backend_info={"petsc_device_effective": "cpu"},
        _stable_cpu_petsc_types=lambda: ("seqaij", "seq"),
    )

    backend_before = backend_signature_from_forward_model(backend_model)
    backend_config.ksp_type = "cg"
    assert backend_signature_from_forward_model(backend_model) != backend_before


def test_v76_model_signature_preserves_complex_contact_impedance() -> None:
    real_like = SimpleNamespace(
        n_elec=2,
        potential_order=1,
        z=np.array([1e-6 + 0j, 2e-6 + 0j], dtype=np.complex64),
        geometry_scale_to_m=1.0,
        eit_mesh=None,
    )
    complex_z = SimpleNamespace(
        n_elec=2,
        potential_order=1,
        z=np.array([1e-6 + 1e-7j, 2e-6 + 0j], dtype=np.complex64),
        geometry_scale_to_m=1.0,
        eit_mesh=None,
    )
    changed_complex_z = SimpleNamespace(
        n_elec=2,
        potential_order=1,
        z=np.array([1e-6 + 2e-7j, 2e-6 + 0j], dtype=np.complex64),
        geometry_scale_to_m=1.0,
        eit_mesh=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        real_like_signature = model_signature_from_forward_model(real_like)
        complex_signature = model_signature_from_forward_model(complex_z)
        changed_complex_signature = model_signature_from_forward_model(
            changed_complex_z
        )

    assert complex_signature != real_like_signature
    assert changed_complex_signature != complex_signature
