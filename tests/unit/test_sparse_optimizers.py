"""Unit tests for sparse optimizer kernels."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import pyeidors.inverse.solvers.sparse_optimizers as sparse_opt_module
from pyeidors.inverse.solvers.sparse_optimizers import solve_fista, solve_irls


def _config(**overrides):
    base = {
        "use_gpu": False,
        "gpu_dtype": "float32",
        "linear_max_iterations": 80,
        "linear_tolerance": 1e-8,
        "smoothing_beta": 1e-6,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_solve_fista_cpu_with_and_without_warm_start():
    A = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=float)
    b = np.array([1.0, 2.0, 0.0], dtype=float)
    cfg = _config()

    x_cold = solve_fista(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=0.5,
        prior_scale=0.3,
        warm_start=None,
        config=cfg,
    )
    x_warm = solve_fista(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=0.5,
        prior_scale=0.3,
        warm_start=np.array([0.1, 0.2], dtype=float),
        config=cfg,
    )

    assert x_cold.shape == (2,)
    assert x_warm.shape == (2,)
    assert np.isfinite(x_cold).all()
    assert np.isfinite(x_warm).all()
    assert np.linalg.norm(A @ x_cold - b) < np.linalg.norm(b)


def test_solve_fista_gpu_requested_without_cuda_falls_back_to_cpu():
    A = np.eye(3, dtype=float)
    b = np.array([1.0, -1.0, 0.5], dtype=float)
    cfg = _config(use_gpu=True)

    result = solve_fista(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=1.0,
        prior_scale=1.0,
        warm_start=np.zeros(3, dtype=float),
        config=cfg,
    )

    assert result.shape == (3,)
    assert np.isfinite(result).all()


def test_solve_irls_cpu_and_lstsq_fallback(monkeypatch):
    A = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    b = np.array([1.0, 1.0, 0.0], dtype=float)
    cfg = _config()

    regular = solve_irls(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=0.8,
        prior_scale=0.25,
        warm_start=np.zeros(2, dtype=float),
        config=cfg,
    )
    assert regular.shape == (2,)
    assert np.isfinite(regular).all()

    calls = {"solve": 0}

    def _raise_singular(*args, **kwargs):  # noqa: ANN001
        calls["solve"] += 1
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(np.linalg, "solve", _raise_singular)
    fallback = solve_irls(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=0.8,
        prior_scale=0.25,
        warm_start=np.zeros(2, dtype=float),
        config=cfg,
    )

    assert calls["solve"] > 0
    assert fallback.shape == (2,)
    assert np.isfinite(fallback).all()


def test_solve_irls_gpu_requested_without_cuda_falls_back_to_cpu():
    A = np.eye(4, dtype=float)
    b = np.array([1.0, 0.0, -1.0, 2.0], dtype=float)
    cfg = _config(use_gpu=True, gpu_dtype="float16")

    result = solve_irls(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=1.0,
        prior_scale=1.0,
        warm_start=np.zeros(4, dtype=float),
        config=cfg,
    )

    assert result.shape == (4,)
    assert np.isfinite(result).all()


def test_sparse_optimizer_forced_gpu_tensor_paths(monkeypatch):
    cfg = _config(
        use_gpu=True, gpu_dtype="float64", linear_max_iterations=2, linear_tolerance=0.0
    )
    monkeypatch.setattr(
        sparse_opt_module,
        "_resolve_gpu_context",
        lambda _config: (torch, torch.device("cpu"), torch.float64),
    )

    A = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=float)
    b = np.array([1.0, -0.5], dtype=float)

    x_fista = solve_fista(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=1.0,
        prior_scale=0.8,
        warm_start=np.zeros(2, dtype=float),
        config=cfg,
    )
    assert x_fista.shape == (2,)
    assert np.isfinite(x_fista).all()

    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced")),
    )
    monkeypatch.setattr(
        torch.linalg,
        "lstsq",
        lambda *_args, **_kwargs: SimpleNamespace(
            solution=torch.zeros(2, dtype=torch.float64)
        ),
    )
    x_irls = solve_irls(
        linear_matrix=A,
        data_vector=b,
        noise_sigma=1.0,
        prior_scale=0.8,
        warm_start=np.zeros(2, dtype=float),
        config=cfg,
    )
    assert x_irls.shape == (2,)
    assert np.isfinite(x_irls).all()
