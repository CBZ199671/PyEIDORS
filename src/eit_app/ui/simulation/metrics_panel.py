"""Read-only display of reconstruction quality metrics."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.theme import set_subtle_value


def _cell_centroids(
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
) -> np.ndarray | None:
    if node_coords is None or cell_connectivity is None:
        return None
    coords = np.asarray(node_coords, dtype=float)
    cells = np.asarray(cell_connectivity, dtype=int)
    if coords.ndim != 2 or cells.ndim != 2 or coords.size == 0 or cells.size == 0:
        return None
    if cells.min(initial=0) < 0 or cells.max(initial=-1) >= len(coords):
        return None
    return coords[cells].mean(axis=1)


def _metric_samples(
    values: np.ndarray,
    *,
    node_coords: np.ndarray | None = None,
    cell_connectivity: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    samples = np.asarray(values, dtype=float).reshape(-1)
    if samples.size == 0:
        return None
    centroids = _cell_centroids(node_coords, cell_connectivity)
    if centroids is not None and len(samples) == len(centroids):
        return centroids, samples
    if node_coords is not None:
        coords = np.asarray(node_coords, dtype=float)
        if coords.ndim == 2 and len(samples) == len(coords):
            return coords, samples
    return None


def _nearest_resample(
    source_positions: np.ndarray,
    source_values: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray | None:
    source_pos = np.asarray(source_positions, dtype=float)
    target_pos = np.asarray(target_positions, dtype=float)
    source_values = np.asarray(source_values, dtype=float).reshape(-1)
    if source_pos.ndim != 2 or target_pos.ndim != 2:
        return None
    if len(source_pos) != len(source_values) or len(source_pos) == 0:
        return None
    dim = min(source_pos.shape[1], target_pos.shape[1])
    if dim <= 0:
        return None
    source_pos = source_pos[:, :dim]
    target_pos = target_pos[:, :dim]
    source_finite = np.isfinite(source_pos).all(axis=1) & np.isfinite(source_values)
    target_finite = np.isfinite(target_pos).all(axis=1)
    if not np.any(source_finite) or not np.any(target_finite):
        return None
    mapped = np.full(len(target_pos), np.nan, dtype=float)
    valid_source_pos = source_pos[source_finite]
    valid_source_values = source_values[source_finite]
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(valid_source_pos).query(target_pos[target_finite], k=1)
    except Exception:
        delta = target_pos[target_finite, None, :] - valid_source_pos[None, :, :]
        idx = np.argmin(np.sum(delta * delta, axis=2), axis=1)
    mapped[target_finite] = valid_source_values[np.asarray(idx, dtype=int)]
    return mapped


class MetricsPanel(QGroupBox):
    """Displays reconstruction quality metrics (error, correlation, etc.)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        # Title assigned by _retranslate() so it follows the UI language.
        super().__init__("", parent)
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(6)

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
        self._lbl_l2.setText(t("sim.metrics.l2_label"))
        self._lbl_corr.setText(t("sim.metrics.correlation_label"))
        self._lbl_rmse.setText(t("sim.metrics.rmse_label"))

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
        if len(ground_truth) == 0 or len(reconstructed) == 0:
            self.clear()
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
            rc = _nearest_resample(rc_pos, rc_values, gt_pos)
            if rc is None:
                self.clear()
                return
        else:
            n = min(len(ground_truth), len(reconstructed))
            gt = np.asarray(ground_truth, dtype=float).reshape(-1)[:n]
            rc = np.asarray(reconstructed, dtype=float).reshape(-1)[:n]

        finite = np.isfinite(gt) & np.isfinite(rc)
        if not np.any(finite):
            self.clear()
            return
        gt = gt[finite]
        rc = rc[finite]

        # Relative L2 error
        diff_norm = np.linalg.norm(gt - rc)
        gt_norm = np.linalg.norm(gt)
        l2_err = diff_norm / gt_norm if gt_norm > 0 else float("inf")
        self._l2_label.setText(f"{l2_err:.4f}")

        # Correlation coefficient
        if np.std(gt) > 0 and np.std(rc) > 0:
            corr = float(np.corrcoef(gt, rc)[0, 1])
        else:
            corr = 0.0
        self._corr_label.setText(f"{corr:.4f}")

        # RMSE
        rmse = np.sqrt(np.mean((gt - rc) ** 2))
        self._rmse_label.setText(f"{rmse:.6f}")

    def clear(self) -> None:
        self._l2_label.setText("\u2014")
        self._corr_label.setText("\u2014")
        self._rmse_label.setText("\u2014")
