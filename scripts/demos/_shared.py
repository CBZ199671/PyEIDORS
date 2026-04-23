"""Shared helpers for lightweight DOLFINx EIT demo scripts."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_random_anomaly(rng: np.random.Generator) -> dict[str, object]:
    """Sample a circular anomaly with moderate random position and contrast."""
    radius = float(rng.uniform(0.08, 0.18))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    dist = float(rng.uniform(0.0, 0.5))
    center = (dist * math.cos(angle), dist * math.sin(angle))
    contrast = float(rng.uniform(1.5, 3.0))
    if rng.random() < 0.35:
        contrast = float(rng.uniform(0.2, 0.8))
    return {"center": center, "radius": radius, "conductivity": contrast}


def save_voltage_comparison_figure(
    *,
    output_path: Path,
    measured: np.ndarray,
    predicted: np.ndarray,
    suptitle: str,
    scatter_title: str = "Boundary Voltage Scatter",
    sequence_title: str = "Boundary Voltage Sequence",
) -> None:
    """Render the shared two-panel voltage comparison figure used by demos."""
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(measured, predicted, s=14, alpha=0.7, label="Predicted vs Ground Truth")
    vmin = min(float(np.min(measured)), float(np.min(predicted)))
    vmax = max(float(np.max(measured)), float(np.max(predicted)))
    ax1.plot([vmin, vmax], [vmin, vmax], "r--", label="y = x")
    ax1.set_title(scatter_title)
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Predicted")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    idx = np.arange(len(measured))
    ax2.plot(idx, measured, "b-", lw=1.2, label="Ground Truth")
    ax2.plot(idx, predicted, "r--", lw=1.2, label="Predicted")
    ax2.set_title(sequence_title)
    ax2.set_xlabel("Measurement Index")
    ax2.set_ylabel("Voltage")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
