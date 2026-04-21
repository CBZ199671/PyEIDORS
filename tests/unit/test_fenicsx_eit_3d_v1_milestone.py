"""FEniCSx-EIT-3D-v1 milestone smoke test."""

from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from pyeidors.data.channels import bad_channel_mask
from pyeidors.inverse import (
    GREIT_METRIC_KEYS,
    CellMesh,
    DualMesh,
    DualMeshJacobianOperator,
    VoxelGrid,
    build_3d_greit_rm,
    build_one_step_rm,
    graph_laplacian,
    greit_metrics,
    reconstruct_difference_batch,
    rm_signature,
    write_forward_rm_benchmark_artifact,
    write_greit_metrics_artifact,
)


def _cell_mesh_from_centers(centers: np.ndarray, *, name: str) -> CellMesh:
    centers = np.asarray(centers, dtype=float)
    offsets = np.array(
        [
            [-1e-3, -1e-3, -1e-3],
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-3],
        ],
        dtype=float,
    )
    coordinates: list[np.ndarray] = []
    cells: list[list[int]] = []
    for center in centers:
        start = len(coordinates)
        coordinates.extend(center + offsets)
        cells.append(list(range(start, start + offsets.shape[0])))
    return CellMesh(np.asarray(coordinates), np.asarray(cells), name=name)


def test_fenicsx_eit_3d_v1_milestone_offline_rm_online_matmul(tmp_path) -> None:
    coarse = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
        name="coarse-inverse-voxels",
    )
    fine_centers = np.vstack(
        [
            center + offset
            for center in coarse.cell_centers()
            for offset in ([-0.18, 0.0, 0.0], [0.18, 0.0, 0.0])
        ]
    )
    fine = _cell_mesh_from_centers(fine_centers, name="fine-cem-surrogate")
    dual = DualMesh(fine, coarse)

    n_coarse = coarse.num_cells()
    n_fine = dual.n_fine_cells
    base_coarse_j = np.vstack(
        [
            np.eye(n_coarse),
            np.array([[4.0, -3.0, 2.0, -1.0, 1.5, -0.5, 3.5, -2.5]]),
        ]
    )
    counts = np.asarray(dual.coarse2fine.sum(axis=0)).reshape(-1)
    fine_j = base_coarse_j @ np.diag(1.0 / counts) @ dual.coarse2fine.T.toarray()
    operator = DualMeshJacobianOperator(dual, fine_j)
    coarse_j = operator.to_dense()
    np.testing.assert_allclose(coarse_j, base_coarse_j)

    target = np.zeros(n_coarse, dtype=float)
    target[[0, 3]] = 1.0
    fine_target = dual.project_to_fine(target)
    np.testing.assert_allclose(operator.Jv(target), fine_j @ fine_target)
    np.testing.assert_allclose(
        operator.JTr(np.ones(base_coarse_j.shape[0])),
        coarse_j.T @ np.ones(base_coarse_j.shape[0]),
    )

    mask = bad_channel_mask(base_coarse_j.shape[0], bad_channels=[8])
    weights = np.array([2.0, 0.75, 1.5, 1.0, 3.0, 0.5, 1.25, 2.5, 1e6])
    laplace = graph_laplacian(coarse)
    rm_tikhonov = build_one_step_rm(
        coarse_j,
        lambda_=1e-8,
        mode="tikhonov",
        form="measurement",
        channel_mask=mask,
        measurement_weights=weights,
    )
    rm_noser = build_one_step_rm(
        coarse_j,
        lambda_=1e-8,
        mode="noser",
        form="measurement",
        channel_mask=mask,
        measurement_weights=weights,
    )
    rm_laplace = build_one_step_rm(
        coarse_j,
        regularization=laplace + sparse.eye(n_coarse, format="csr") * 1e-9,
        lambda_=1e-8,
        mode="laplace",
        form="measurement",
        channel_mask=mask,
        measurement_weights=weights,
    )
    assert rm_tikhonov.shape == rm_noser.shape == rm_laplace.shape == (n_coarse, 9)

    reference = np.linspace(2.0, 4.0, coarse_j.shape[0])
    normalized = operator.Jv(target)
    normalized[8] = 123.0
    frame = reference * (1.0 + normalized)
    online = reconstruct_difference_batch(
        rm_noser,
        np.vstack([frame, frame]),
        normalize=True,
        v_ref=reference,
        channel_mask=mask,
        measurement_weights=weights,
        device="cpu",
        return_metadata=True,
    )
    np.testing.assert_allclose(online.values[0], target, atol=3e-8)
    assert online.metadata["online_hot_path"] == "rm_matmul"
    assert online.metadata["forward_solve_count"] == 0
    assert online.metadata["ksp_solve_count"] == 0

    greit = build_3d_greit_rm(
        jacobian=coarse_j,
        inverse_mesh=coarse,
        target_radius=0.2,
        noise_figure=1e-8,
        channel_mask=mask,
        measurement_weights=weights,
    )
    greit_image = greit.reconstruct(
        frame,
        normalize=True,
        v_ref=reference,
        device="cpu",
    )
    metrics = greit_metrics(
        greit_image, target.astype(bool), centers=coarse.cell_centers()
    )
    assert set(metrics) == set(GREIT_METRIC_KEYS)
    metrics_path = write_greit_metrics_artifact(
        metrics, tmp_path / "greit_metrics.json"
    )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["metric_keys"] == list(GREIT_METRIC_KEYS)

    signature = rm_signature(
        forward_mesh_hash="fine-cem-surrogate-hash",
        inverse_mesh_hash="coarse-voxel-hash",
        coarse2fine=dual.coarse2fine,
        electrode_geometry={"count": 48, "rings": [16, 16, 16]},
        stim_meas_protocol={"stim": "adjacent", "meas": "skip-current"},
        background={"sigma0": 1.0, "z0": 0.01},
        difference_mode="normalized",
        bad_channel_mask=mask,
        noise_covariance=weights,
        regularization_type="noser",
        hyperparameters={"lambda": 1e-8, "noise_figure": 1e-8},
        device="cuda",
    )
    assert len(signature) == 64
    bench_path = write_forward_rm_benchmark_artifact(
        tmp_path / "forward_rm_benchmark.json",
        offline_rm_build_seconds=1.25,
        online_rm_apply_seconds=0.002,
        metadata={"rm_signature": signature},
    )
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    assert bench["offline_rm_build_seconds"] > bench["online_rm_apply_seconds"]

    forward_diag = {
        "mesh_family": fine.name,
        "coarse_unknowns": n_coarse,
        "fine_cells": n_fine,
        "forward_reuse_preconditioner_requested": True,
        "forward_pc_session_reused": True,
    }
    checklist = {
        "fine_cem": forward_diag["fine_cells"] > forward_diag["coarse_unknowns"],
        "coarse_voxel": coarse.num_cells() == n_coarse,
        "coarse2fine": dual.coarse2fine.shape == (n_fine, n_coarse),
        "reusable_ksp_pc": forward_diag["forward_pc_session_reused"],
        "adjoint_j_on_coarse": operator.JTr(np.ones(coarse_j.shape[0])).shape
        == (n_coarse,),
        "one_step_rm": rm_noser.shape == (n_coarse, coarse_j.shape[0]),
        "normalized_dv": np.isfinite(normalized).all(),
        "gpu_online_kernel": online.metadata["online_hot_path"] == "rm_matmul",
        "greit_metrics": set(metrics) == set(GREIT_METRIC_KEYS),
        "bad_channel_w": mask[8] and online.metadata["n_frames"] == 2,
    }
    assert all(checklist.values()), checklist
