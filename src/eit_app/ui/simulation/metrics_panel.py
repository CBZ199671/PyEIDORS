"""Read-only display of reconstruction quality metrics."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.array_geometry_cache import cached_cell_centers
from eit_app.ui.theme import set_subtle_value


_METRIC_SCAN_CHUNK_ITEMS = 1_048_576


def _cell_centroids(
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
) -> np.ndarray | None:
    if node_coords is None or cell_connectivity is None:
        return None
    coords = np.asarray(node_coords)
    cells = np.asarray(cell_connectivity)
    if coords.ndim != 2 or cells.ndim != 2 or coords.size == 0 or cells.size == 0:
        return None
    try:
        if cells.min(initial=0) < 0 or cells.max(initial=-1) >= len(coords):
            return None
    except (TypeError, ValueError):
        return None
    return cached_cell_centers(coords, cells)


def _metric_float_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = np.asarray(arr, dtype=np.float64)
    return np.asarray(arr).reshape(-1)


def _metric_samples(
    values: np.ndarray,
    *,
    node_coords: np.ndarray | None = None,
    cell_connectivity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    samples = _metric_float_array(values)
    if samples.size == 0:
        return None
    centroids = _cell_centroids(node_coords, cell_connectivity)
    if centroids is not None and len(samples) == len(centroids):
        return centroids, samples
    if node_coords is not None:
        coords = np.asarray(node_coords)
        if coords.ndim == 2 and len(samples) == len(coords):
            return coords, samples
    return None


def _positions_match(
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    if left is right:
        return True
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        return False
    return bool(np.array_equal(left_arr, right_arr))


def _finite_row_mask_or_none(
    positions: np.ndarray,
    values: np.ndarray | None = None,
) -> np.ndarray | None:
    pos = np.asarray(positions)
    n_rows = pos.shape[0]
    chunk_items = max(1, int(_METRIC_SCAN_CHUNK_ITEMS))
    work_size = min(chunk_items, max(n_rows, 1))
    row_work = np.empty(work_size, dtype=bool)
    axis_work = np.empty(work_size, dtype=bool)
    for start in range(0, n_rows, chunk_items):
        stop = min(start + chunk_items, n_rows)
        finite = _finite_row_chunk_mask(
            pos,
            start,
            stop,
            out=row_work[: stop - start],
            work=axis_work[: stop - start],
            values=values,
        )
        if bool(finite.all()):
            continue

        mask = np.empty(n_rows, dtype=bool)
        if start > 0:
            mask[:start] = True
        mask[start:stop] = finite
        for tail_start in range(stop, n_rows, chunk_items):
            tail_stop = min(tail_start + chunk_items, n_rows)
            tail_finite = _finite_row_chunk_mask(
                pos,
                tail_start,
                tail_stop,
                out=row_work[: tail_stop - tail_start],
                work=axis_work[: tail_stop - tail_start],
                values=values,
            )
            mask[tail_start:tail_stop] = tail_finite
        return mask
    return None


def _finite_row_chunk_mask(
    positions: np.ndarray,
    start: int,
    stop: int,
    *,
    out: np.ndarray,
    work: np.ndarray,
    values: np.ndarray | None = None,
) -> np.ndarray:
    pos = np.asarray(positions)
    if pos.shape[1] == 0:
        out.fill(False)
        return out
    np.isfinite(pos[start:stop, 0], out=out)
    for axis in range(1, pos.shape[1]):
        np.isfinite(pos[start:stop, axis], out=work)
        np.logical_and(out, work, out=out)
    if values is not None:
        np.isfinite(values[start:stop], out=work)
        np.logical_and(out, work, out=out)
    return out


def _nearest_resample(
    source_positions: np.ndarray,
    source_values: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray | None:
    source_pos = np.asarray(source_positions)
    target_pos = np.asarray(target_positions)
    source_values = _metric_float_array(source_values)
    if source_pos.ndim != 2 or target_pos.ndim != 2:
        return None
    if len(source_pos) != len(source_values) or len(source_pos) == 0:
        return None
    dim = min(source_pos.shape[1], target_pos.shape[1])
    if dim <= 0:
        return None
    source_pos = source_pos[:, :dim]
    target_pos = target_pos[:, :dim]
    mapped_dtype = np.result_type(source_values.dtype, np.float32)
    mapped = np.full(len(target_pos), np.nan, dtype=mapped_dtype)
    source_mask = _finite_row_mask_or_none(source_pos, source_values)
    target_mask = _finite_row_mask_or_none(target_pos)
    if source_mask is None:
        valid_source_pos = source_pos
        valid_source_values = source_values.astype(mapped_dtype, copy=False)
    else:
        if not bool(source_mask.any()):
            return None
        valid_source_pos, valid_source_values = _compact_source_samples(
            source_pos,
            source_values,
            source_mask,
            value_dtype=mapped_dtype,
        )
    if target_mask is None:
        query_pos = target_pos
    else:
        if not bool(target_mask.any()):
            return None
        query_pos = _compact_rows_by_mask(target_pos, target_mask)
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(valid_source_pos).query(query_pos, k=1)
    except Exception:
        idx = _nearest_indices_bruteforce(
            valid_source_pos,
            query_pos,
        )
    idx_arr = np.asarray(idx, dtype=np.intp)
    if target_mask is None:
        np.take(valid_source_values, idx_arr, out=mapped)
    else:
        _fill_masked_resample_values(
            valid_source_values,
            idx_arr,
            target_mask,
            mapped,
        )
    return mapped


def _compact_rows_by_mask(rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows_arr = np.asarray(rows)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    out = np.empty(
        (int(np.count_nonzero(mask_arr)), rows_arr.shape[1]), dtype=rows_arr.dtype
    )
    out_idx = 0
    for row_idx, is_valid in enumerate(mask_arr):
        if not bool(is_valid):
            continue
        out[out_idx, :] = rows_arr[row_idx, :]
        out_idx += 1
    return out


def _compact_source_samples(
    positions: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    value_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    positions_arr = np.asarray(positions)
    values_arr = np.asarray(values)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    valid_count = int(np.count_nonzero(mask_arr))
    out_positions = np.empty(
        (valid_count, positions_arr.shape[1]),
        dtype=positions_arr.dtype,
    )
    out_values = np.empty(valid_count, dtype=value_dtype)
    out_idx = 0
    for row_idx, is_valid in enumerate(mask_arr):
        if not bool(is_valid):
            continue
        out_positions[out_idx, :] = positions_arr[row_idx, :]
        out_values[out_idx] = values_arr[row_idx]
        out_idx += 1
    return out_positions, out_values


def _fill_masked_resample_values(
    source_values: np.ndarray,
    source_indices: np.ndarray,
    target_mask: np.ndarray,
    out: np.ndarray,
) -> None:
    query_pos = 0
    for target_idx, is_valid in enumerate(np.asarray(target_mask, dtype=bool)):
        if not bool(is_valid):
            continue
        out[target_idx] = source_values[int(source_indices[query_pos])]
        query_pos += 1


def _nearest_indices_bruteforce(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    source_arr = np.asarray(source_positions)
    target_arr = np.asarray(target_positions)
    if source_arr.ndim != 2 or target_arr.ndim != 2:
        raise ValueError("source_positions and target_positions must be 2D arrays.")
    if source_arr.shape[0] == 0:
        raise ValueError("source_positions must be non-empty.")
    dim = min(source_arr.shape[1], target_arr.shape[1])
    source_dtype = (
        source_arr.dtype
        if np.issubdtype(source_arr.dtype, np.floating)
        else np.dtype(np.float32)
    )
    target_dtype = (
        target_arr.dtype
        if np.issubdtype(target_arr.dtype, np.floating)
        else np.dtype(np.float32)
    )
    work_dtype = np.result_type(source_dtype, target_dtype, np.float32)
    source = np.asarray(source_arr[:, :dim], dtype=work_dtype)
    target = np.asarray(target_arr[:, :dim], dtype=work_dtype)
    indices = np.empty(target_arr.shape[0], dtype=np.intp)
    distances = np.empty(source_arr.shape[0], dtype=work_dtype)
    work = np.empty(source_arr.shape[0], dtype=work_dtype)
    for row, point in enumerate(target):
        np.subtract(source[:, 0], point[0], out=distances)
        np.square(distances, out=distances)
        for axis in range(1, dim):
            np.subtract(source[:, axis], point[axis], out=work)
            np.square(work, out=work)
            distances += work
        indices[row] = int(np.argmin(distances))
    return indices


def _finite_pair_stats(
    ground_truth: np.ndarray,
    reconstructed: np.ndarray,
) -> tuple[int, float, float, float, float, float, float]:
    gt = np.asarray(ground_truth).reshape(-1)
    rc = np.asarray(reconstructed).reshape(-1)
    n = min(gt.size, rc.size)
    chunk_items = max(1, int(_METRIC_SCAN_CHUNK_ITEMS))
    count = 0
    diff_sq_sum = 0.0
    gt_sq_sum = 0.0
    gt_sum = 0.0
    rc_sum = 0.0
    rc_sq_sum = 0.0
    cross_sum = 0.0
    work_size = min(n, chunk_items)
    work = np.empty(work_size, dtype=np.result_type(gt.dtype, rc.dtype))
    finite = np.empty(work_size, dtype=bool)
    finite_work = np.empty(work_size, dtype=bool)
    for start in range(0, n, chunk_items):
        stop = min(start + chunk_items, n)
        gt_chunk = gt[start:stop]
        rc_chunk = rc[start:stop]
        chunk_size = stop - start
        finite_chunk = finite[:chunk_size]
        finite_work_chunk = finite_work[:chunk_size]
        np.isfinite(gt_chunk, out=finite_chunk)
        np.isfinite(rc_chunk, out=finite_work_chunk)
        np.logical_and(finite_chunk, finite_work_chunk, out=finite_chunk)
        if not bool(finite_chunk.any()):
            continue
        count += int(np.count_nonzero(finite_chunk))
        work_chunk = work[:chunk_size]

        np.subtract(gt_chunk, rc_chunk, out=work_chunk)
        np.square(work_chunk, out=work_chunk)
        diff_sq_sum += float(np.sum(work_chunk, where=finite_chunk))

        np.multiply(gt_chunk, gt_chunk, out=work_chunk)
        gt_sq_sum += float(np.sum(work_chunk, where=finite_chunk))

        gt_sum += float(np.sum(gt_chunk, where=finite_chunk))
        rc_sum += float(np.sum(rc_chunk, where=finite_chunk))

        np.multiply(rc_chunk, rc_chunk, out=work_chunk)
        rc_sq_sum += float(np.sum(work_chunk, where=finite_chunk))

        np.multiply(gt_chunk, rc_chunk, out=work_chunk)
        cross_sum += float(np.sum(work_chunk, where=finite_chunk))

    return count, diff_sq_sum, gt_sq_sum, gt_sum, rc_sum, rc_sq_sum, cross_sum


def _mesh_counts(
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
) -> tuple[int, int] | None:
    if node_coords is None or cell_connectivity is None:
        return None
    try:
        coords = np.asarray(node_coords)
        cells = np.asarray(cell_connectivity)
    except Exception:
        return None
    if coords.ndim != 2 or cells.ndim != 2:
        return None
    if len(coords) <= 0 or len(cells) <= 0:
        return None
    return int(len(coords)), int(len(cells))


def _format_mesh_count_value(counts: tuple[int, int] | None) -> str:
    if counts is None:
        return "\u2014"
    n_nodes, n_elements = counts
    return t(
        "sim.metrics.mesh_value",
        nodes=f"{n_nodes:,}",
        elements=f"{n_elements:,}",
    )


class MetricsPanel(QGroupBox):
    """Displays reconstruction quality metrics (error, correlation, etc.)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._truth_mesh_counts: tuple[int, int] | None = None
        self._recon_mesh_counts: tuple[int, int] | None = None
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(6)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._truth_mesh_label = QLabel("\u2014")
        set_subtle_value(self._truth_mesh_label)
        self._lbl_truth_mesh = QLabel("")
        layout.addRow(self._lbl_truth_mesh, self._truth_mesh_label)

        self._recon_mesh_label = QLabel("\u2014")
        set_subtle_value(self._recon_mesh_label)
        self._lbl_recon_mesh = QLabel("")
        layout.addRow(self._lbl_recon_mesh, self._recon_mesh_label)

        self._l2_label = QLabel("\u2014")
        set_subtle_value(self._l2_label)
        self._lbl_l2 = QLabel("")
        layout.addRow(self._lbl_l2, self._l2_label)

        self._corr_label = QLabel("\u2014")
        set_subtle_value(self._corr_label)
        self._lbl_corr = QLabel("")
        layout.addRow(self._lbl_corr, self._corr_label)

        self._rmse_label = QLabel("\u2014")
        set_subtle_value(self._rmse_label)
        self._lbl_rmse = QLabel("")
        layout.addRow(self._lbl_rmse, self._rmse_label)

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setTitle(t("sim.metrics.title"))
        self._lbl_truth_mesh.setText(t("sim.metrics.truth_mesh_label"))
        self._lbl_recon_mesh.setText(t("sim.metrics.recon_mesh_label"))
        self._lbl_l2.setText(t("sim.metrics.l2_label"))
        self._lbl_corr.setText(t("sim.metrics.correlation_label"))
        self._lbl_rmse.setText(t("sim.metrics.rmse_label"))
        self._refresh_mesh_count_labels()

    def _refresh_mesh_count_labels(self) -> None:
        self._truth_mesh_label.setText(
            _format_mesh_count_value(self._truth_mesh_counts)
        )
        self._recon_mesh_label.setText(
            _format_mesh_count_value(self._recon_mesh_counts)
        )

    def update_mesh_stats(
        self,
        *,
        ground_truth_node_coords: np.ndarray | None = None,
        ground_truth_cell_connectivity: np.ndarray | None = None,
        reconstructed_node_coords: np.ndarray | None = None,
        reconstructed_cell_connectivity: np.ndarray | None = None,
    ) -> None:
        self._truth_mesh_counts = _mesh_counts(
            ground_truth_node_coords,
            ground_truth_cell_connectivity,
        )
        self._recon_mesh_counts = _mesh_counts(
            reconstructed_node_coords,
            reconstructed_cell_connectivity,
        )
        self._refresh_mesh_count_labels()

    def update_metrics(
        self,
        ground_truth: np.ndarray,
        reconstructed: np.ndarray,
        *,
        ground_truth_node_coords: np.ndarray | None = None,
        ground_truth_cell_connectivity: np.ndarray | None = None,
        reconstructed_node_coords: np.ndarray | None = None,
        reconstructed_cell_connectivity: np.ndarray | None = None,
    ) -> None:
        """Compute and display metrics comparing ground truth to reconstruction."""
        self.update_mesh_stats(
            ground_truth_node_coords=ground_truth_node_coords,
            ground_truth_cell_connectivity=ground_truth_cell_connectivity,
            reconstructed_node_coords=reconstructed_node_coords,
            reconstructed_cell_connectivity=reconstructed_cell_connectivity,
        )
        if len(ground_truth) == 0 or len(reconstructed) == 0:
            self._clear_metric_values()
            return

        gt_samples = _metric_samples(
            ground_truth,
            node_coords=ground_truth_node_coords,
            cell_connectivity=ground_truth_cell_connectivity,
        )
        rc_samples = _metric_samples(
            reconstructed,
            node_coords=reconstructed_node_coords,
            cell_connectivity=reconstructed_cell_connectivity,
        )
        if gt_samples is not None and rc_samples is not None:
            gt_pos, gt = gt_samples
            rc_pos, rc_values = rc_samples
            if _positions_match(gt_pos, rc_pos):
                rc = np.asarray(rc_values).reshape(-1)
            else:
                rc = _nearest_resample(rc_pos, rc_values, gt_pos)
                if rc is None:
                    self._clear_metric_values()
                    return
        else:
            n = min(len(ground_truth), len(reconstructed))
            gt = _metric_float_array(ground_truth)[:n]
            rc = _metric_float_array(reconstructed)[:n]

        (
            finite_count,
            diff_sq_sum,
            gt_sq_sum,
            gt_sum,
            rc_sum,
            rc_sq_sum,
            cross_sum,
        ) = _finite_pair_stats(gt, rc)
        if finite_count <= 0:
            self._clear_metric_values()
            return

        # Relative L2 error
        diff_norm = float(np.sqrt(diff_sq_sum))
        gt_norm = float(np.sqrt(gt_sq_sum))
        l2_err = diff_norm / gt_norm if gt_norm > 0 else float("inf")
        self._l2_label.setText(f"{l2_err:.4f}")

        # Correlation coefficient
        gt_var_sum = gt_sq_sum - (gt_sum * gt_sum) / finite_count
        rc_var_sum = rc_sq_sum - (rc_sum * rc_sum) / finite_count
        if gt_var_sum > 0.0 and rc_var_sum > 0.0:
            cov_sum = cross_sum - (gt_sum * rc_sum) / finite_count
            corr = float(cov_sum / np.sqrt(gt_var_sum * rc_var_sum))
        else:
            corr = 0.0
        self._corr_label.setText(f"{corr:.4f}")

        # RMSE
        rmse = float(np.sqrt(diff_sq_sum / finite_count))
        self._rmse_label.setText(f"{rmse:.6f}")

    def _clear_metric_values(self) -> None:
        self._l2_label.setText("\u2014")
        self._corr_label.setText("\u2014")
        self._rmse_label.setText("\u2014")

    def clear(self) -> None:
        self._truth_mesh_counts = None
        self._recon_mesh_counts = None
        self._refresh_mesh_count_labels()
        self._clear_metric_values()
