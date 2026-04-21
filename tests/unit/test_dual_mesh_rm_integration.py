"""Integration smoke for dual-mesh one-step difference reconstruction."""

from __future__ import annotations

import numpy as np

from pyeidors.data.channels import bad_channel_mask
from pyeidors.inverse.dual_mesh import CellMesh, DualMesh, VoxelGrid
from pyeidors.inverse.reconstruction_matrix import (
    build_one_step_rm,
    reconstruct_difference_batch,
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
    return CellMesh(
        coordinates=np.asarray(coordinates, dtype=float),
        cells=np.asarray(cells, dtype=int),
        name=name,
    )


def _fine_centers_from_coarse_grid(coarse: VoxelGrid) -> np.ndarray:
    offsets = np.array([[-0.18, 0.0, 0.0], [0.18, 0.0, 0.0]], dtype=float)
    return np.vstack([center + offsets for center in coarse.cell_centers()])


def _synthetic_sphere_target(
    coarse: VoxelGrid,
) -> tuple[np.ndarray, np.ndarray]:
    centers = coarse.cell_centers()
    sphere_center = np.array([0.6, 0.6, 1.0], dtype=float)
    target_mask = np.linalg.norm(centers - sphere_center, axis=1) <= 0.75
    target = target_mask.astype(float)
    if not np.any(target_mask):
        raise AssertionError("synthetic target must cover at least one coarse cell")
    return target, target_mask


def _synthetic_fine_sensitivity(
    base_coarse_jacobian: np.ndarray,
    dual: DualMesh,
) -> np.ndarray:
    projection = dual.coarse2fine
    counts = np.asarray(projection.sum(axis=0)).reshape(-1)
    return base_coarse_jacobian @ np.diag(1.0 / counts) @ projection.T.toarray()


def _eidors_style_blob_metrics(
    reconstruction: np.ndarray,
    target_mask: np.ndarray,
    centers: np.ndarray,
) -> dict[str, float | bool]:
    target = target_mask.astype(float)
    correlation = float(np.corrcoef(reconstruction, target)[0, 1])
    positive = np.maximum(reconstruction, 0.0)
    recon_center = (positive[:, None] * centers).sum(axis=0) / positive.sum()
    target_center = centers[target_mask].mean(axis=0)
    position_error = float(np.linalg.norm(recon_center - target_center))
    amplitude_ratio = float(reconstruction[target_mask].sum() / target.sum())
    return {
        "correlation": correlation,
        "position_error": position_error,
        "amplitude_ratio": amplitude_ratio,
        "peak_in_target": bool(target_mask[int(np.argmax(reconstruction))]),
    }


def test_dual_mesh_one_step_rm_smoke_on_synthetic_3d_sphere() -> None:
    coarse = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
        name="coarse-3d-voxels",
    )
    fine = _cell_mesh_from_centers(
        _fine_centers_from_coarse_grid(coarse),
        name="fine-cem-surrogate",
    )
    dual = DualMesh(fine_mesh=fine, coarse_mesh=coarse)

    target, target_mask = _synthetic_sphere_target(coarse)
    fine_target = dual.project_to_fine(target)
    np.testing.assert_allclose(dual.restrict_to_coarse(fine_target), target)

    n_coarse = coarse.num_cells()
    base_coarse_jacobian = np.vstack(
        [
            np.eye(n_coarse, dtype=float),
            np.array([[6.0, -4.0, 3.0, 1.5, -2.0, 2.5, 0.25, -1.0]]),
        ]
    )
    fine_jacobian = _synthetic_fine_sensitivity(base_coarse_jacobian, dual)
    coarse_jacobian = fine_jacobian @ dual.coarse2fine.toarray()
    np.testing.assert_allclose(coarse_jacobian, base_coarse_jacobian)

    mask = bad_channel_mask(coarse_jacobian.shape[0], bad_channels=[8])
    weights = np.array([2.0, 0.75, 1.5, 1.0, 3.0, 0.5, 1.25, 2.5, 1e6])
    rm_result = build_one_step_rm(
        coarse_jacobian,
        lambda_=1e-8,
        mode="noser",
        form="measurement",
        channel_mask=mask,
        measurement_weights=weights,
        return_metadata=True,
    )

    reference = np.linspace(2.0, 3.6, coarse_jacobian.shape[0])
    normalized_dv = fine_jacobian @ fine_target
    normalized_dv[8] = 123.0
    second_target = 0.5 * target
    second_fine_target = dual.project_to_fine(second_target)
    second_normalized_dv = fine_jacobian @ second_fine_target
    second_normalized_dv[8] = -77.0
    frames = np.vstack(
        [
            reference * (1.0 + normalized_dv),
            reference * (1.0 + second_normalized_dv),
        ]
    )

    reconstruction = reconstruct_difference_batch(
        rm_result.rm,
        frames,
        normalize=True,
        v_ref=reference,
        channel_mask=mask,
        measurement_weights=weights,
        device="cpu",
        return_metadata=True,
    )

    np.testing.assert_allclose(reconstruction.values[0], target, atol=2e-8)
    np.testing.assert_allclose(reconstruction.values[1], second_target, atol=2e-8)
    assert dual.summary() == {
        "n_fine_cells": 16,
        "n_coarse_cells": 8,
        "projection_nnz": 16,
        "method": "piecewise_constant",
        "outside": "nearest",
    }
    assert rm_result.metadata["mode"] == "noser"
    assert rm_result.metadata["form"] == "measurement"
    assert rm_result.metadata["bad_channel_count"] == 1
    assert rm_result.metadata["measurement_weight_kind"] == "diagonal"
    assert reconstruction.metadata["batched"] is True
    assert reconstruction.metadata["device_effective"] == "cpu"

    metrics = _eidors_style_blob_metrics(
        reconstruction.values[0],
        target_mask,
        coarse.cell_centers(),
    )
    assert metrics["peak_in_target"] is True
    assert metrics["correlation"] > 0.999999
    assert metrics["position_error"] < 1e-8
    assert abs(float(metrics["amplitude_ratio"]) - 1.0) < 2e-8
