"""Read-only display of reconstruction quality metrics."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel, QWidget

from eit_app.ui.theme import set_subtle_value


class MetricsPanel(QGroupBox):
    """Displays reconstruction quality metrics (error, correlation, etc.)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Metrics", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(6)

        self._l2_label = QLabel("\u2014")
        set_subtle_value(self._l2_label)
        layout.addRow("Relative L2 error:", self._l2_label)

        self._corr_label = QLabel("\u2014")
        set_subtle_value(self._corr_label)
        layout.addRow("Correlation:", self._corr_label)

        self._rmse_label = QLabel("\u2014")
        set_subtle_value(self._rmse_label)
        layout.addRow("RMSE:", self._rmse_label)

    def update_metrics(
        self,
        ground_truth: np.ndarray,
        reconstructed: np.ndarray,
    ) -> None:
        """Compute and display metrics comparing ground truth to reconstruction."""
        if len(ground_truth) == 0 or len(reconstructed) == 0:
            self.clear()
            return

        # Ensure same length
        n = min(len(ground_truth), len(reconstructed))
        gt = ground_truth[:n]
        rc = reconstructed[:n]

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
