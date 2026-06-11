"""3D conductivity display with opacity / clipping controls.

Used by ``SimulationResultsWidget`` for 3D tetra / hex volume phantoms — far
smoother through PyVista/VTK when the runtime can safely embed a native
OpenGL window, and still a real PyVista/VTK offscreen 3D view when WSLg
must keep the main Qt window on Wayland for crisp HiDPI text.

PyVista (built on VTK) is the visualisation library that the FEniCSx
project itself ships with — see
https://docs.fenicsproject.org/dolfinx/latest/python/demos/demo_pyvista.html
— so the render path is both first-class for our forward solver's
mesh format and hardware-accelerated.

Two non-obvious design points worth keeping in mind for future
maintenance:

1.  ``QtInteractor`` remains the preferred display path for real 3D
    simulation output on native desktop runtimes.  When embedded VTK is
    unsafe (for example WSLg/Wayland), the widget uses PyVista offscreen.
    If PyVista/VTK cannot render, the GUI shows an explicit unavailable
    caption instead of drawing a Matplotlib 3D substitute.

2.  When enabled, ``QtInteractor`` is constructed only after the host
    widget is shown and owns a native child window.  Initialising VTK
    while the host is still parentless / hidden can materialise orphan
    top-level windows or stale X handles.

3.  ``auto_update=False``.  The pyvistaqt default is a 5 Hz background
    render timer, which means VTK redraws the scene 5×/second forever
    even if the user is just hovering over the form.  For our
    use-case nothing changes between user gestures so we drive renders
    explicitly from ``update_image`` / slider callbacks instead.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.array_geometry_cache import _compute_cell_centers, cached_cell_centers
from eit_app.ui.electrode_overlay import (
    ElectrodeGeometry,
    default_patch_quads,
)
from eit_app.ui.theme import (
    plot_palette,
    set_button_role,
    set_hint_text,
    set_section_header,
    subscribe_theme_mode,
)


log = logging.getLogger(__name__)


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_OFFSCREEN_DRAG_FPS_ENV = "EIT_APP_3D_DRAG_FPS"
_OFFSCREEN_DRAG_RENDER_SCALE_ENV = "EIT_APP_3D_DRAG_RENDER_SCALE"
_OFFSCREEN_DRAG_IDLE_MS_ENV = "EIT_APP_3D_DRAG_IDLE_MS"
_AUTO_POINTS_CELLS_ENV = "EIT_APP_3D_AUTO_POINTS_CELLS"
_POINT_CLOUD_MAX_POINTS_ENV = "EIT_APP_3D_POINT_CLOUD_MAX_POINTS"
_PROGRESSIVE_VOLUME_UPGRADE_ENV = "EIT_APP_3D_PROGRESSIVE_VOLUME_UPGRADE"
_PROGRESSIVE_VOLUME_DELAY_MS_ENV = "EIT_APP_3D_PROGRESSIVE_VOLUME_DELAY_MS"
_PYVISTA_OFFSCREEN_NEGATIVE_CACHE_ENV = "EIT_APP_3D_PYVISTA_OFFSCREEN_NEGATIVE_CACHE"
_DEFAULT_AUTO_POINTS_CELLS = 12000
SUPPORTED_3D_CELL_VERTEX_COUNTS = frozenset({4, 8})
DISPLAY_MODE_VOLUME = "volume"
DISPLAY_MODE_POINTS = "points"
DISPLAY_MODES = frozenset({DISPLAY_MODE_VOLUME, DISPLAY_MODE_POINTS})
ANOMALY_MODE_POSITIVE = "positive"
ANOMALY_MODE_NEGATIVE = "negative"
ANOMALY_MODE_ABSOLUTE = "absolute"
ANOMALY_MODES = frozenset(
    {ANOMALY_MODE_POSITIVE, ANOMALY_MODE_NEGATIVE, ANOMALY_MODE_ABSOLUTE}
)
_INHOMOGENEITY_RELATIVE_FLOOR = 0.02
_ANOMALY_PEAK_FRACTION = 0.35
_ANOMALY_MAD_SIGMA = 3.0
_ANOMALY_MAX_VISIBLE_FRACTION = 0.08
_ANOMALY_CROWDED_PERCENTILE = 95.0
_ANOMALY_CROWDED_PEAK_CAP = 0.75
_ANOMALY_SPATIAL_MIN_CANDIDATES = 8
_ANOMALY_SPATIAL_RADIUS_FACTOR = 2.75
_ANOMALY_COMPONENT_KEEP_FRACTION = 0.22
_ANOMALY_CENTRAL_REGION_EDGE_WEIGHT = 0.08
_ANOMALY_CENTRAL_REGION_PEAK_FRACTION = 0.30
_FINITE_SCAN_CHUNK_ITEMS = 1_048_576
_POINT_CLOUD_SAMPLE_CHUNK_ITEMS = 1_048_576
_CELL_FACE_OFFSETS = {
    4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    8: (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}
_PYVISTA_OFFSCREEN_FAILURE_REASON: str | None = None


def _display_float_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        arr = np.real(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = np.asarray(arr, dtype=np.float32)
    return np.asarray(arr).reshape(-1)


def _finite_mask_or_none(values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(values).reshape(-1)
    chunk_items = max(1, int(_FINITE_SCAN_CHUNK_ITEMS))
    work = np.empty(min(chunk_items, max(arr.size, 1)), dtype=bool)
    for start in range(0, arr.size, chunk_items):
        chunk = arr[start : start + chunk_items]
        chunk_mask = work[: chunk.size]
        np.isfinite(chunk, out=chunk_mask)
        if bool(chunk_mask.all()):
            continue

        finite_mask = np.empty(arr.shape, dtype=bool)
        if start > 0:
            finite_mask[:start] = True
        finite_mask[start : start + chunk.size] = chunk_mask
        for tail_start in range(start + chunk.size, arr.size, chunk_items):
            tail = arr[tail_start : tail_start + chunk_items]
            tail_mask = work[: tail.size]
            np.isfinite(tail, out=tail_mask)
            finite_mask[tail_start : tail_start + tail.size] = tail_mask
        return finite_mask
    return None


def _all_finite_values(values: np.ndarray) -> bool:
    arr = np.asarray(values).reshape(-1)
    chunk_items = max(1, int(_FINITE_SCAN_CHUNK_ITEMS))
    work = np.empty(min(chunk_items, max(arr.size, 1)), dtype=bool)
    for start in range(0, arr.size, chunk_items):
        chunk = arr[start : start + chunk_items]
        chunk_mask = work[: chunk.size]
        np.isfinite(chunk, out=chunk_mask)
        if not bool(chunk_mask.all()):
            return False
    return True


def _any_finite_values(values: np.ndarray) -> bool:
    arr = np.asarray(values).reshape(-1)
    chunk_items = max(1, int(_FINITE_SCAN_CHUNK_ITEMS))
    work = np.empty(min(chunk_items, max(arr.size, 1)), dtype=bool)
    for start in range(0, arr.size, chunk_items):
        chunk = arr[start : start + chunk_items]
        chunk_mask = work[: chunk.size]
        np.isfinite(chunk, out=chunk_mask)
        if bool(chunk_mask.any()):
            return True
    return False


def _nan_invalid_nearest_distances(nearest: np.ndarray) -> bool:
    arr = np.asarray(nearest).reshape(-1)
    chunk_items = max(1, int(_FINITE_SCAN_CHUNK_ITEMS))
    work = np.empty(min(chunk_items, max(arr.size, 1)), dtype=bool)
    has_valid = False
    for start in range(0, arr.size, chunk_items):
        chunk = arr[start : start + chunk_items]
        mask = work[: chunk.size]
        np.isfinite(chunk, out=mask)
        np.logical_not(mask, out=mask)
        if bool(mask.any()):
            np.copyto(chunk, np.nan, where=mask)
        np.greater(chunk, 1.0e-12, out=mask)
        if bool(mask.any()):
            has_valid = True
        np.logical_not(mask, out=mask)
        if bool(mask.any()):
            np.copyto(chunk, np.nan, where=mask)
    return has_valid


def _score_count_peak_above_floor(
    score: np.ndarray,
    floor: float,
    *,
    all_finite: bool,
    finite_mask: np.ndarray | None = None,
    return_mask: bool = False,
) -> tuple[int, float] | tuple[int, float, np.ndarray]:
    candidate_mask = np.greater(score, floor)
    if not all_finite:
        if finite_mask is None:
            finite_mask = _finite_mask_or_none(score)
        if finite_mask is not None:
            np.logical_and(candidate_mask, finite_mask, out=candidate_mask)
    candidate_count = int(np.count_nonzero(candidate_mask))
    if candidate_count == 0:
        if return_mask:
            return 0, float("nan"), candidate_mask
        return 0, float("nan")
    peak = float(np.max(score, where=candidate_mask, initial=-np.inf))
    if return_mask:
        return candidate_count, peak, candidate_mask
    return candidate_count, peak


def _nanpercentile_with_finite_mask(
    score: np.ndarray,
    percentile: float,
    *,
    finite_mask: np.ndarray | None,
    invalid_mask: np.ndarray,
) -> float:
    if finite_mask is not None:
        np.logical_not(finite_mask, out=invalid_mask)
        np.copyto(score, np.nan, where=invalid_mask)
    return float(np.nanpercentile(score, percentile))


def _nanmedian_with_finite_mask(values: np.ndarray, finite_mask: np.ndarray) -> float:
    work = np.array(values, copy=True)
    np.logical_not(finite_mask, out=finite_mask)
    np.copyto(work, np.nan, where=finite_mask)
    return float(np.nanmedian(work))


def _cell_anomaly_mask(
    cell_sigma: np.ndarray,
    mode: str = ANOMALY_MODE_ABSOLUTE,
    *,
    cell_centers: np.ndarray | None = None,
    prefer_central_region: bool = False,
) -> np.ndarray:
    values = _display_float_values(cell_sigma)
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool)
    finite_values = _finite_mask_or_none(values)
    all_finite = finite_values is None
    if not all_finite and not finite_values.any():
        return np.zeros(values.shape, dtype=bool)
    if mode not in ANOMALY_MODES:
        raise ValueError(f"unknown anomaly mode: {mode!r}")
    if all_finite:
        median = float(np.median(values))
        spread = float(np.std(values))
    else:
        median = float(np.nanmedian(values))
        spread = float(np.nanstd(values))
    floor = max(abs(median) * _INHOMOGENEITY_RELATIVE_FLOOR, 1.0e-6)
    if not np.isfinite(spread) or spread <= floor:
        return np.zeros(values.shape, dtype=bool)

    if prefer_central_region and cell_centers is not None:
        region_score = values - median
        np.abs(region_score, out=region_score)
        region_mask = _central_region_anomaly_mask(
            region_score,
            floor,
            cell_centers,
            all_finite=all_finite,
            finite_mask=finite_values,
        )
        if np.any(region_mask):
            return region_mask

    score = values - median
    if mode == ANOMALY_MODE_ABSOLUTE:
        np.abs(score, out=score)
        residual_mad = float(np.nanmedian(score))
    else:
        np.abs(score, out=score)
        residual_mad = float(np.nanmedian(score))
        np.subtract(values, median, out=score)
        if mode == ANOMALY_MODE_NEGATIVE:
            np.negative(score, out=score)
    candidate_count, peak, mask = _score_count_peak_above_floor(
        score,
        floor,
        all_finite=all_finite,
        finite_mask=finite_values,
        return_mask=True,
    )
    if candidate_count == 0:
        return mask
    robust_floor = min(
        _ANOMALY_MAD_SIGMA * 1.4826 * residual_mad,
        peak * _ANOMALY_CROWDED_PEAK_CAP,
    )
    threshold = max(floor, peak * _ANOMALY_PEAK_FRACTION, robust_floor)
    np.greater_equal(score, threshold, out=mask)
    visible_fraction = float(np.count_nonzero(mask)) / float(max(values.size, 1))
    if visible_fraction > _ANOMALY_MAX_VISIBLE_FRACTION and candidate_count > 1:
        crowded_threshold = _nanpercentile_with_finite_mask(
            score,
            _ANOMALY_CROWDED_PERCENTILE,
            finite_mask=None if all_finite else finite_values,
            invalid_mask=mask,
        )
        crowded_threshold = min(
            crowded_threshold,
            peak * _ANOMALY_CROWDED_PEAK_CAP,
        )
        threshold = max(threshold, crowded_threshold)
        threshold = float(np.nextafter(threshold, np.inf))
    tolerance = max(1.0e-12, abs(threshold) * 1.0e-12)
    np.greater_equal(score, threshold - tolerance, out=mask)
    return _spatially_coherent_anomaly_mask(mask, score, cell_centers)


def _central_region_anomaly_mask(
    score: np.ndarray,
    floor: float,
    cell_centers: np.ndarray,
    *,
    all_finite: bool,
    finite_mask: np.ndarray | None,
) -> np.ndarray:
    centers = np.asarray(cell_centers)
    mask = np.zeros(score.shape, dtype=bool)
    if centers.ndim != 2 or centers.shape[0] != score.size or centers.shape[1] < 3:
        return mask
    if score.size < _ANOMALY_SPATIAL_MIN_CANDIDATES:
        return mask

    xyz = centers[:, :3]
    if not _all_finite_values(xyz):
        return mask

    domain_center = np.mean(xyz, axis=0)
    distances = np.linalg.norm(xyz - domain_center, axis=1)
    domain_radius = float(np.max(distances))
    if not np.isfinite(domain_radius) or domain_radius <= 0.0:
        return mask

    centrality = np.clip(1.0 - distances / domain_radius, 0.0, 1.0)
    weighted_score = score * (
        _ANOMALY_CENTRAL_REGION_EDGE_WEIGHT
        + (1.0 - _ANOMALY_CENTRAL_REGION_EDGE_WEIGHT) * centrality * centrality
    )
    candidate_count, peak, candidate_mask = _score_count_peak_above_floor(
        weighted_score,
        floor,
        all_finite=all_finite,
        finite_mask=finite_mask,
        return_mask=True,
    )
    if candidate_count < _ANOMALY_SPATIAL_MIN_CANDIDATES:
        return mask

    threshold = max(floor, peak * _ANOMALY_CENTRAL_REGION_PEAK_FRACTION)
    np.greater_equal(weighted_score, threshold, out=candidate_mask)
    if not all_finite and finite_mask is not None:
        np.logical_and(candidate_mask, finite_mask, out=candidate_mask)

    return _center_preferred_component_mask(
        candidate_mask,
        weighted_score,
        centers,
    )


def _center_preferred_component_mask(
    mask: np.ndarray,
    score: np.ndarray,
    cell_centers: np.ndarray,
) -> np.ndarray:
    candidate_count = int(np.count_nonzero(mask))
    coherent = np.zeros_like(mask, dtype=bool)
    if candidate_count < _ANOMALY_SPATIAL_MIN_CANDIDATES:
        return coherent

    centers = np.asarray(cell_centers)
    candidate_idx, candidate_centers = _candidate_indices_and_centers(
        mask, centers, candidate_count
    )
    if not _all_finite_values(candidate_centers):
        return coherent

    try:
        from scipy.spatial import cKDTree
    except Exception:  # pragma: no cover - optional visualization refinement
        return coherent

    try:
        tree = cKDTree(candidate_centers)
        distances, _ = tree.query(candidate_centers, k=2)
    except Exception:  # pragma: no cover - scipy edge case fallback
        return coherent

    nearest = np.asarray(distances[:, 1])
    if not _nan_invalid_nearest_distances(nearest):
        return coherent
    radius = float(np.nanmedian(nearest) * _ANOMALY_SPATIAL_RADIUS_FACTOR)
    if not np.isfinite(radius) or radius <= 0.0:
        return coherent

    neighbours = tree.query_ball_point(candidate_centers, radius)
    seen = np.zeros(candidate_idx.size, dtype=bool)
    components: list[np.ndarray] = []
    for start in range(candidate_idx.size):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for nxt in neighbours[current]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))
        components.append(np.asarray(component, dtype=np.int64))

    if not components:
        return coherent

    score_values = _display_float_values(score)
    domain_center = np.mean(centers[:, :3], axis=0)
    domain_distances = np.linalg.norm(centers[:, :3] - domain_center, axis=1)
    domain_radius = float(np.max(domain_distances))
    if not np.isfinite(domain_radius) or domain_radius <= 0.0:
        return coherent

    best_idx = -1
    best_priority = -np.inf
    for component_idx, component in enumerate(components):
        if component.size < 2:
            continue
        mass = 0.0
        centroid = np.zeros(3, dtype=np.float64)
        for local_idx in component:
            global_idx = int(candidate_idx[int(local_idx)])
            value = float(score_values[global_idx])
            if not np.isnan(value):
                mass += value
            centroid += centers[global_idx, :3]
        centroid /= float(component.size)
        central_distance = float(np.linalg.norm(centroid - domain_center))
        central_weight = max(0.0, 1.0 - central_distance / domain_radius)
        priority = mass * max(central_weight * central_weight, 0.02)
        if priority > best_priority:
            best_priority = priority
            best_idx = component_idx

    if best_idx < 0 or not np.isfinite(best_priority) or best_priority <= 0.0:
        return coherent

    keep = np.zeros(candidate_idx.size, dtype=bool)
    keep[components[best_idx]] = True
    _apply_candidate_keep_mask(coherent, candidate_idx, keep)
    return coherent


def _spatially_coherent_anomaly_mask(
    mask: np.ndarray,
    score: np.ndarray,
    cell_centers: np.ndarray | None,
) -> np.ndarray:
    """Keep coherent anomaly blobs while dropping isolated high-score speckles."""

    if cell_centers is None:
        return mask
    candidate_count = int(np.count_nonzero(mask))
    if candidate_count < _ANOMALY_SPATIAL_MIN_CANDIDATES:
        return mask

    centers = np.asarray(cell_centers)
    if centers.ndim != 2 or centers.shape[0] != mask.size or centers.shape[1] < 3:
        return mask
    candidate_idx, candidate_centers = _candidate_indices_and_centers(
        mask, centers, candidate_count
    )
    if not _all_finite_values(candidate_centers):
        return mask

    try:
        from scipy.spatial import cKDTree
    except Exception:  # pragma: no cover - optional visualization refinement
        return mask

    try:
        tree = cKDTree(candidate_centers)
        distances, _ = tree.query(candidate_centers, k=2)
    except Exception:  # pragma: no cover - scipy edge case fallback
        return mask

    nearest = np.asarray(distances[:, 1])
    if not _nan_invalid_nearest_distances(nearest):
        return mask
    radius = float(np.nanmedian(nearest) * _ANOMALY_SPATIAL_RADIUS_FACTOR)
    if not np.isfinite(radius) or radius <= 0.0:
        return mask

    neighbours = tree.query_ball_point(candidate_centers, radius)
    seen = np.zeros(candidate_idx.size, dtype=bool)
    components: list[np.ndarray] = []
    for start in range(candidate_idx.size):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for nxt in neighbours[current]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))
        components.append(np.asarray(component, dtype=np.int64))

    if len(components) <= 1:
        return mask

    score_values = _display_float_values(score)
    masses = _component_score_masses(score_values, candidate_idx, components)
    if masses.size == 0 or not _any_finite_values(masses):
        return mask

    best_idx = int(np.nanargmax(masses))
    best_mass = float(masses[best_idx])
    if not np.isfinite(best_mass) or best_mass <= 0.0:
        return mask

    min_component_size = (
        2
        if candidate_idx.size < 16
        else max(3, int(np.ceil(candidate_idx.size * 0.015)))
    )
    keep = np.zeros(candidate_idx.size, dtype=bool)
    for component_idx, component in enumerate(components):
        if component_idx != best_idx and component.size < min_component_size:
            continue
        if component_idx == best_idx or masses[component_idx] >= (
            best_mass * _ANOMALY_COMPONENT_KEEP_FRACTION
        ):
            keep[component] = True

    if not np.any(keep):
        return mask
    coherent = np.zeros_like(mask, dtype=bool)
    _apply_candidate_keep_mask(coherent, candidate_idx, keep)
    return coherent


def _apply_candidate_keep_mask(
    coherent: np.ndarray, candidate_idx: np.ndarray, keep: np.ndarray
) -> None:
    for local_idx, is_kept in enumerate(np.asarray(keep, dtype=bool).reshape(-1)):
        if not bool(is_kept):
            continue
        global_idx = int(candidate_idx[int(local_idx)])
        if 0 <= global_idx < coherent.size:
            coherent[global_idx] = True


def _candidate_indices_and_centers(
    mask: np.ndarray, centers: np.ndarray, candidate_count: int
) -> tuple[np.ndarray, np.ndarray]:
    centers_arr = np.asarray(centers)
    center_dtype = (
        centers_arr.dtype
        if np.issubdtype(centers_arr.dtype, np.floating)
        else np.dtype(np.float32)
    )
    candidate_idx = np.empty(int(candidate_count), dtype=np.int64)
    candidate_centers = np.empty((int(candidate_count), 3), dtype=center_dtype)
    out_idx = 0
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    for center_idx, is_active in enumerate(mask_arr):
        if not bool(is_active):
            continue
        candidate_idx[out_idx] = center_idx
        candidate_centers[out_idx] = centers_arr[center_idx, :3]
        out_idx += 1
    return candidate_idx, candidate_centers


def _component_score_masses(
    score_values: np.ndarray,
    candidate_idx: np.ndarray,
    components: list[np.ndarray],
) -> np.ndarray:
    masses = np.empty(len(components), dtype=np.float64)
    for component_pos, component in enumerate(components):
        total = 0.0
        for local_idx in component:
            value = float(score_values[int(candidate_idx[int(local_idx)])])
            if np.isnan(value):
                continue
            total += value
        masses[component_pos] = total
    return masses


def _face_cell_values(cell_sigma: np.ndarray, source_indices: np.ndarray) -> np.ndarray:
    values = np.asarray(cell_sigma)
    out = np.empty(len(source_indices), dtype=values.dtype)
    np.take(values, source_indices, out=out)
    return out


def _cell_inhomogeneity_mask(cell_sigma: np.ndarray) -> np.ndarray:
    return _cell_anomaly_mask(cell_sigma, ANOMALY_MODE_ABSOLUTE)


def _conductivity_color_limits(cell_sigma: np.ndarray) -> tuple[float, float]:
    values = _display_float_values(cell_sigma)
    if values.size == 0:
        return 0.0, 1.0
    finite_mask = _finite_mask_or_none(values)
    if finite_mask is not None and not finite_mask.any():
        return 0.0, 1.0

    if finite_mask is None:
        sigma_min = float(np.min(values))
        sigma_max = float(np.max(values))
        median = float(np.median(values))
    else:
        sigma_min = float(np.min(values, where=finite_mask, initial=np.inf))
        sigma_max = float(np.max(values, where=finite_mask, initial=-np.inf))
        median = _nanmedian_with_finite_mask(values, finite_mask)
    if not all(np.isfinite(value) for value in (sigma_min, sigma_max, median)):
        return 0.0, 1.0

    floor = max(abs(median) * _INHOMOGENEITY_RELATIVE_FLOOR, 1.0e-6)
    if sigma_max - sigma_min < 2.0 * floor:
        return median - floor, median + floor
    return sigma_min, sigma_max


def _sanitize_display_value_limits(
    value_limits: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if value_limits is None:
        return None
    low, high = (float(value_limits[0]), float(value_limits[1]))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high > low:
        return low, high
    center = 0.5 * (low + high)
    span = max(abs(center) * 0.05, 1.0e-6)
    return center - span, center + span


def _display_color_limits(
    values: np.ndarray,
    value_limits: tuple[float, float] | None,
) -> tuple[float, float]:
    sanitized = _sanitize_display_value_limits(value_limits)
    if sanitized is not None:
        return sanitized
    return _conductivity_color_limits(values)


def _cell_center_sigma(
    sigma: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return cell-centered conductivity values for 3D display modes."""
    raw_values = np.asarray(sigma)
    if np.issubdtype(raw_values.dtype, np.floating):
        values = raw_values.reshape(-1)
    else:
        values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
    if values.shape[0] == cells.shape[0]:
        return values, "cell"
    return _cell_mean_values(values, cells), "point"


def _display_coords_array(node_coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(node_coords)
    if coords.dtype == np.dtype(np.float32):
        return coords
    return np.asarray(coords, dtype=np.float32)


def _display_cells_array(cell_connectivity: np.ndarray) -> np.ndarray:
    cells = np.asarray(cell_connectivity)
    if np.issubdtype(cells.dtype, np.integer):
        return cells
    return np.asarray(cells, dtype=np.intp)


def _display_sigma_array(conductivity: np.ndarray) -> np.ndarray:
    sigma = np.asarray(conductivity)
    if np.iscomplexobj(sigma):
        sigma = np.real(sigma)
    if np.asarray(sigma).dtype == np.dtype(np.float32):
        return np.asarray(sigma)
    return np.asarray(sigma, dtype=np.float32)


def _cell_centers(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    centers = cached_cell_centers(coords, cells, coordinate_dims=3)
    if centers is None:
        return _compute_cell_centers(
            _display_coords_array(coords)[:, :3],
            np.asarray(cells),
        )
    return centers


def _pyvista_point_size(n_cells: int) -> float:
    return float(np.clip(1100.0 / max(np.sqrt(max(n_cells, 1)), 1.0), 4.0, 14.0))


def _pyvista_surface(dataset):
    try:
        return dataset.extract_surface(algorithm="dataset_surface")
    except TypeError as exc:
        if "algorithm" not in str(exc):
            raise
        return dataset.extract_surface()


def _pyvista_feature_outline(dataset, *, feature_angle: float):
    return _pyvista_surface(dataset).extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        feature_angle=feature_angle,
        non_manifold_edges=False,
        manifold_edges=False,
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _env_float(name: str, default: float, *, lower: float, upper: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return min(max(value, lower), upper)


def _env_int(name: str, default: int, *, lower: int, upper: int) -> int:
    value = _env_float(name, float(default), lower=float(lower), upper=float(upper))
    return int(round(value))


def _auto_points_cell_threshold() -> int:
    return _env_int(
        _AUTO_POINTS_CELLS_ENV,
        _DEFAULT_AUTO_POINTS_CELLS,
        lower=0,
        upper=2_000_000,
    )


def _should_auto_points(n_cells: int) -> bool:
    threshold = _auto_points_cell_threshold()
    return threshold > 0 and int(n_cells) >= threshold


def _point_cloud_max_points() -> int:
    raw = os.environ.get(_POINT_CLOUD_MAX_POINTS_ENV, "").strip().lower()
    if raw in {"0", "false", "no", "off", "none", "disabled"}:
        return 0
    return _env_int(_POINT_CLOUD_MAX_POINTS_ENV, 60000, lower=0, upper=2_000_000)


def _progressive_volume_upgrade_enabled() -> bool:
    return _env_flag(_PROGRESSIVE_VOLUME_UPGRADE_ENV)


def _progressive_volume_delay_ms() -> int:
    return _env_int(
        _PROGRESSIVE_VOLUME_DELAY_MS_ENV,
        750,
        lower=0,
        upper=60_000,
    )


def _should_skip_pyvista_offscreen(n_cells: int, display_mode: str) -> bool:
    del n_cells, display_mode
    return False


def _should_skip_pyvista_offscreen_for_reason(
    n_cells: int,
    display_mode: str,
    reason: str,
) -> bool:
    del n_cells, display_mode, reason
    return False


def _pyvista_offscreen_negative_cache_enabled() -> bool:
    raw = os.environ.get(_PYVISTA_OFFSCREEN_NEGATIVE_CACHE_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off", "none", "disabled"}


def _pyvista_offscreen_failure_reason() -> str | None:
    if not _pyvista_offscreen_negative_cache_enabled():
        return None
    return _PYVISTA_OFFSCREEN_FAILURE_REASON


def _mark_pyvista_offscreen_failure(reason: object) -> None:
    if not _pyvista_offscreen_negative_cache_enabled():
        return
    global _PYVISTA_OFFSCREEN_FAILURE_REASON
    text = str(reason).strip()
    _PYVISTA_OFFSCREEN_FAILURE_REASON = text or "unknown PyVista offscreen failure"


def _clear_pyvista_offscreen_failure_cache() -> None:
    global _PYVISTA_OFFSCREEN_FAILURE_REASON
    _PYVISTA_OFFSCREEN_FAILURE_REASON = None


def _evenly_spaced_range(size: int, max_count: int) -> np.ndarray:
    size = max(int(size), 0)
    max_count = int(max_count)
    if max_count <= 0 or size <= 0:
        return np.empty((0,), dtype=np.int64)
    if size <= max_count:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, num=max_count, dtype=np.int64)


def _sample_background_indices(
    anomaly_mask: np.ndarray,
    anomaly_idx: np.ndarray,
    max_count: int,
) -> np.ndarray:
    if max_count <= 0:
        return np.empty((0,), dtype=np.int64)
    mask_arr = np.asarray(anomaly_mask, dtype=bool).reshape(-1)
    n_points = int(mask_arr.size)
    background_count = n_points - int(anomaly_idx.size)
    if background_count <= 0:
        return np.empty((0,), dtype=np.int64)
    actual_background_count = background_count
    if actual_background_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if actual_background_count <= max_count:
        return _background_indices_from_mask(mask_arr, actual_background_count)
    ranks = _evenly_spaced_range(background_count, max_count)
    if anomaly_idx.size == 0:
        return ranks
    candidates = ranks
    while True:
        shifted = ranks + np.searchsorted(anomaly_idx, candidates, side="right")
        if np.array_equal(shifted, candidates):
            return candidates.astype(np.int64, copy=False)
        candidates = shifted


def _background_indices_from_mask(
    mask: np.ndarray, background_count: int
) -> np.ndarray:
    out = np.empty(int(background_count), dtype=np.int64)
    out_pos = 0
    for point_idx, is_anomaly in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        if bool(is_anomaly):
            continue
        if out_pos >= out.size:
            break
        out[out_pos] = point_idx
        out_pos += 1
    return out[:out_pos]


def _true_indices_from_mask(mask: np.ndarray, true_count: int) -> np.ndarray:
    out = np.empty(int(true_count), dtype=np.int64)
    out_pos = 0
    for point_idx, is_true in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        if not bool(is_true):
            continue
        if out_pos >= out.size:
            break
        out[out_pos] = point_idx
        out_pos += 1
    return out[:out_pos]


def _sample_true_indices(
    mask: np.ndarray,
    true_count: int,
    max_count: int,
) -> np.ndarray:
    if max_count <= 0 or true_count <= 0:
        return np.empty((0,), dtype=np.int64)
    if true_count <= max_count:
        return _true_indices_from_mask(mask, true_count)

    ranks = _evenly_spaced_range(true_count, max_count)
    out = np.empty(ranks.size, dtype=np.int64)
    rank_pos = 0
    true_seen = 0
    mask_flat = np.asarray(mask, dtype=bool).reshape(-1)
    chunk_items = max(1, int(_POINT_CLOUD_SAMPLE_CHUNK_ITEMS))
    for start in range(0, mask_flat.size, chunk_items):
        chunk = mask_flat[start : start + chunk_items]
        for local_idx, is_true in enumerate(chunk):
            if not bool(is_true):
                continue
            if true_seen == int(ranks[rank_pos]):
                out[rank_pos] = start + local_idx
                rank_pos += 1
                if rank_pos >= out.size:
                    return out
            true_seen += 1
    return out[:rank_pos]


def _point_cloud_sample_indices(
    cell_sigma: np.ndarray,
    centers: np.ndarray,
    *,
    anomaly_mode: str,
    max_points: int | None = None,
    prefer_central_region: bool = False,
) -> np.ndarray:
    values = _display_float_values(cell_sigma)
    n_points = int(values.size)
    if n_points == 0:
        return np.array([], dtype=np.int64)
    limit = _point_cloud_max_points() if max_points is None else int(max_points)
    if limit <= 0 or n_points <= limit:
        return np.arange(n_points, dtype=np.int64)

    center_values = np.asarray(centers)
    if center_values.ndim != 2 or center_values.shape[0] != n_points:
        return _evenly_spaced_range(n_points, limit)

    # Keep the full-data pass O(n); spatial coherence is applied later on the
    # sampled display set for highlight rendering.
    anomaly_mask = _cell_anomaly_mask(
        values,
        anomaly_mode,
        cell_centers=center_values if prefer_central_region else None,
        prefer_central_region=prefer_central_region,
    )
    anomaly_count = int(np.count_nonzero(anomaly_mask))
    if anomaly_count >= limit:
        return _sample_true_indices(anomaly_mask, anomaly_count, limit)
    anomaly_idx = (
        _true_indices_from_mask(anomaly_mask, anomaly_count)
        if anomaly_count > 0
        else np.empty((0,), dtype=np.int64)
    )

    background_budget = limit - int(anomaly_idx.size)
    sampled_background = _sample_background_indices(
        anomaly_mask,
        anomaly_idx,
        background_budget,
    )
    sampled = np.empty(anomaly_idx.size + sampled_background.size, dtype=np.int64)
    sampled[: anomaly_idx.size] = anomaly_idx
    sampled[anomaly_idx.size :] = sampled_background
    sampled.sort()
    return sampled


def _point_cloud_display_arrays(
    centers: np.ndarray,
    cell_sigma: np.ndarray,
    sample_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample = np.asarray(sample_idx, dtype=np.int64).reshape(-1)
    center_values = np.asarray(centers)
    if np.issubdtype(center_values.dtype, np.floating):
        display_centers = np.empty(
            (sample.size, center_values.shape[1]), dtype=center_values.dtype
        )
        np.take(center_values, sample, axis=0, out=display_centers)
    else:
        display_centers = np.empty(
            (sample.size, center_values.shape[1]), dtype=np.float32
        )
        for row_idx, source_idx in enumerate(sample):
            display_centers[row_idx] = center_values[int(source_idx)]

    sigma_values = _display_float_values(cell_sigma)
    display_sigma = np.empty(sample.size, dtype=sigma_values.dtype)
    np.take(sigma_values, sample, out=display_sigma)
    return display_centers, display_sigma


def _point_cloud_highlight_arrays(
    display_centers: np.ndarray,
    display_sigma: np.ndarray,
    inhom_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers_arr = np.asarray(display_centers)
    sigma_arr = np.asarray(display_sigma)
    mask_arr = np.asarray(inhom_mask, dtype=bool).reshape(-1)
    n_points = min(mask_arr.size, centers_arr.shape[0], sigma_arr.shape[0])
    if n_points <= 0:
        return (
            np.empty((0, *centers_arr.shape[1:]), dtype=centers_arr.dtype),
            np.empty((0,), dtype=sigma_arr.dtype),
        )
    active_mask = mask_arr[:n_points]
    active_count = int(np.count_nonzero(active_mask))
    if active_count <= 0:
        return (
            np.empty((0, *centers_arr.shape[1:]), dtype=centers_arr.dtype),
            np.empty((0,), dtype=sigma_arr.dtype),
        )

    highlight_centers = np.empty(
        (active_count, *centers_arr.shape[1:]), dtype=centers_arr.dtype
    )
    highlight_sigma = np.empty(active_count, dtype=sigma_arr.dtype)
    out_idx = 0
    for point_idx, is_active in enumerate(active_mask):
        if not bool(is_active):
            continue
        highlight_centers[out_idx] = centers_arr[point_idx]
        highlight_sigma[out_idx] = sigma_arr[point_idx]
        out_idx += 1
    return highlight_centers, highlight_sigma


def _extract_cells_from_mask(grid, inhom_mask: np.ndarray):
    mask_arr = np.asarray(inhom_mask, dtype=bool).reshape(-1)
    if mask_arr.size == 0:
        return None
    try:
        return grid.extract_cells(mask_arr)
    except Exception:
        inhom_indices = np.flatnonzero(mask_arr)
        if inhom_indices.size == 0:
            return None
        return grid.extract_cells(inhom_indices)


def _running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except OSError:
        return False


def embedded_vtk_status() -> tuple[bool, str]:
    """Decide whether it is safe to embed pyvistaqt's VTK widget.

    The FEniCSx-recommended PyVista path is available on native desktop
    runtimes, but WSLg/X11 has repeatedly crashed the whole process from
    inside VTK's native window setup (``BadWindow / X_ConfigureWindow``).
    That class of failure cannot be caught by Python, so WSL keeps the
    embedded interactor disabled unless XCB is forced.  The caller may
    still use PyVista offscreen for a real VTK-rendered 3D image.
    """
    if _env_flag("EIT_APP_DISABLE_EMBEDDED_VTK"):
        return False, "disabled by EIT_APP_DISABLE_EMBEDDED_VTK"
    if _env_flag("EIT_APP_ENABLE_EMBEDDED_VTK"):
        return True, "forced by EIT_APP_ENABLE_EMBEDDED_VTK"

    qpa = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if qpa in {"offscreen", "minimal"}:
        return False, f"Qt platform is {qpa!r}"

    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False, "no DISPLAY or WAYLAND_DISPLAY is available"

    if _running_under_wsl():
        if qpa == "xcb":
            return True, "WSLg is using Qt XCB, compatible with vtkXOpenGLRenderWindow"
        return False, "WSLg embedded VTK requires QT_QPA_PLATFORM=xcb"

    return True, "runtime looks compatible"


def embedded_vtk_enabled() -> bool:
    enabled, _reason = embedded_vtk_status()
    return enabled


class _InteractorHost(QFrame):
    """QFrame whose ``realized`` signal fires once after the first
    real ``showEvent`` — i.e. when Qt has actually placed the widget
    in the visible hierarchy.

    Used to defer VTK / pyvistaqt construction until Qt has placed the
    host inside the visible hierarchy.  We deliberately do *not* force
    a native window on this frame: ``QVTKRenderWindowInteractor``
    creates and owns the native child window it passes to VTK, and an
    extra native host layer has proven fragile on WSLg/X11.
    """

    realized = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fired = False

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        if self._fired:
            return
        self._fired = True
        # Defer to the next event-loop tick so pending layout and Qt
        # surface setup finishes before constructing QVTK.
        QTimer.singleShot(0, self.realized.emit)


class _OffscreenRenderLabel(QLabel):
    """Pixmap canvas for PyVista offscreen frames with basic camera controls."""

    dragged = Signal(float, float)
    zoomed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_pos: QPoint | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._last_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            delta = pos - self._last_pos
            self._last_pos = pos
            self.dragged.emit(float(delta.x()), float(delta.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._last_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.zoomed.emit(float(event.angleDelta().y()))
        event.accept()


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    """Parse a CSS-style ``#rrggbb`` colour into 0–1 floats for VTK."""
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return (1.0, 1.0, 1.0)
    return (
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
    )


def _boundary_faces(cells: np.ndarray) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Return boundary faces plus the source volume-cell index for each face."""
    faces: dict[tuple[int, ...], tuple[tuple[int, ...], int] | None] = {}
    offsets_for_cell = _CELL_FACE_OFFSETS.get(int(cells.shape[1]))
    if offsets_for_cell is None:
        return [], np.empty((0,), dtype=np.int64)

    for cell_idx, cell in enumerate(cells):
        for offsets in offsets_for_cell:
            face = tuple(int(cell[offset]) for offset in offsets)
            key = tuple(sorted(face))
            faces[key] = None if key in faces else (face, cell_idx)

    kept_count = sum(1 for payload in faces.values() if payload is not None)
    if kept_count <= 0:
        return [], np.empty((0,), dtype=np.int64)

    boundary_faces: list[tuple[int, ...]] = [()] * kept_count
    source_cells = np.empty(kept_count, dtype=np.int64)
    kept_idx = 0
    for payload in faces.values():
        if payload is None:
            continue
        face, cell_idx = payload
        boundary_faces[kept_idx] = face
        source_cells[kept_idx] = int(cell_idx)
        kept_idx += 1
    return boundary_faces, source_cells


def _valid_boundary_faces_and_sources(
    faces: list[tuple[int, ...]], source_cells: np.ndarray, n_coords: int
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    if not faces:
        return [], np.empty((0,), dtype=np.intp)

    all_valid = True
    for face in faces:
        for idx in face:
            if idx < 0 or idx >= n_coords:
                all_valid = False
                break
        if not all_valid:
            break
    if all_valid:
        return faces, np.asarray(source_cells, dtype=np.intp)

    valid_count = 0
    for face in faces:
        if all(0 <= idx < n_coords for idx in face):
            valid_count += 1
    if valid_count <= 0:
        return [], np.empty((0,), dtype=np.intp)

    valid_faces: list[tuple[int, ...]] = [()] * valid_count
    source_indices = np.empty(valid_count, dtype=np.intp)
    out_idx = 0
    for face, source_cell in zip(faces, source_cells, strict=False):
        if not all(0 <= idx < n_coords for idx in face):
            continue
        valid_faces[out_idx] = face
        source_indices[out_idx] = int(source_cell)
        out_idx += 1
    return valid_faces, source_indices


def _cell_mean_values(point_values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Average point values per cell without materializing ``values[cells]``."""

    values = np.asarray(point_values)
    if np.iscomplexobj(values):
        values = np.real(values)
    if np.issubdtype(values.dtype, np.floating):
        dtype = np.result_type(values.dtype, np.float32)
        source = np.asarray(values, dtype=dtype).reshape(-1)
    else:
        source = np.asarray(values, dtype=np.float32).reshape(-1)
        dtype = source.dtype
    cells_i = np.asarray(cells)
    if not np.issubdtype(cells_i.dtype, np.integer):
        cells_i = np.asarray(cells_i, dtype=np.intp)
    if cells_i.ndim != 2 or cells_i.shape[0] == 0:
        return np.empty((0,), dtype=dtype)
    out = np.zeros(cells_i.shape[0], dtype=dtype)
    work = np.empty(cells_i.shape[0], dtype=dtype)
    for local_idx in range(cells_i.shape[1]):
        np.take(source, cells_i[:, local_idx], out=work)
        out += work
    out /= float(cells_i.shape[1])
    return out


def _face_nanmean_value(point_values: np.ndarray, face: tuple[int, ...]) -> float:
    total = 0.0
    count = 0
    for idx in face:
        value = float(point_values[int(idx)])
        if np.isnan(value):
            continue
        total += value
        count += 1
    if count == 0:
        return float("nan")
    return total / float(count)


def _face_vertices(coords: np.ndarray, face: tuple[int, ...]) -> np.ndarray:
    coords_arr = np.asarray(coords)
    vertices = np.empty((len(face), 3), dtype=coords_arr.dtype)
    for row, idx in enumerate(face):
        vertices[row] = coords_arr[int(idx), :3]
    return vertices


def _face_vertices_array(
    coords: np.ndarray, faces: list[tuple[int, ...]]
) -> np.ndarray:
    coords_arr = np.asarray(coords)
    if not faces:
        return np.empty((0, 0, 3), dtype=coords_arr.dtype)
    vertices_per_face = len(faces[0])
    vertices = np.empty((len(faces), vertices_per_face, 3), dtype=coords_arr.dtype)
    for face_idx, face in enumerate(faces):
        for vertex_idx, idx in enumerate(face):
            vertices[face_idx, vertex_idx] = coords_arr[int(idx), :3]
    return vertices


def _highlight_face_vertices_and_values(
    coords: np.ndarray,
    cells: np.ndarray,
    cell_sigma: np.ndarray,
    inhom_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords_arr = np.asarray(coords)
    cells_arr = np.asarray(cells)
    sigma_arr = np.asarray(cell_sigma)
    mask_arr = np.asarray(inhom_mask, dtype=bool).reshape(-1)
    offsets_for_cell = _CELL_FACE_OFFSETS.get(int(cells_arr.shape[1]))
    if offsets_for_cell is None or mask_arr.size == 0:
        return (
            np.empty((0, 0, 3), dtype=coords_arr.dtype),
            np.empty((0,), dtype=sigma_arr.dtype),
        )

    active_mask = mask_arr[: cells_arr.shape[0]]
    active_count = int(np.count_nonzero(active_mask))
    if active_count <= 0:
        vertices_per_face = len(offsets_for_cell[0])
        return (
            np.empty((0, vertices_per_face, 3), dtype=coords_arr.dtype),
            np.empty((0,), dtype=sigma_arr.dtype),
        )

    vertices_per_face = len(offsets_for_cell[0])
    max_faces = active_count * len(offsets_for_cell)
    vertices = np.empty((max_faces, vertices_per_face, 3), dtype=coords_arr.dtype)
    values = np.empty(max_faces, dtype=sigma_arr.dtype)
    n_coords = len(coords_arr)
    out_idx = 0
    for cell_idx, is_active in enumerate(active_mask):
        if not bool(is_active):
            continue
        cell = cells_arr[cell_idx]
        for offsets in offsets_for_cell:
            valid = True
            for vertex_idx, offset in enumerate(offsets):
                node_idx = int(cell[offset])
                if node_idx < 0 or node_idx >= n_coords:
                    valid = False
                    break
                vertices[out_idx, vertex_idx] = coords_arr[node_idx, :3]
            if not valid:
                continue
            values[out_idx] = sigma_arr[cell_idx]
            out_idx += 1

    if out_idx <= 0:
        return (
            np.empty((0, vertices_per_face, 3), dtype=coords_arr.dtype),
            np.empty((0,), dtype=sigma_arr.dtype),
        )
    return vertices[:out_idx], values[:out_idx]


def _configure_vtk_logging() -> None:
    """Keep harmless VTK warnings (e.g. missing Xcursor) out of GUI logs."""
    try:
        from vtkmodules.vtkCommonCore import vtkLogger

        vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)
    except Exception:  # pragma: no cover — best-effort log hygiene
        pass


class Conductivity3DWidget(QWidget):
    """Hardware-accelerated 3D conductivity viewer with transparency controls.

    The widget mirrors ``ConductivityImageWidget``'s public surface
    (``update_image`` / ``clear`` / ``set_loading`` / ``setTitle``) so
    it can be swapped in by the dispatcher in
    ``SimulationResultsWidget``.
    """

    def __init__(
        self, title: str = "Conductivity", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._default_title = title

        # Last data + cached actors.  Actors are created once per
        # update_image and toggled via SetVisibility for fast
        # checkbox / slider response without re-extracting cells or
        # recomputing edges.
        self._last_image: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, str | None]
        ] = None
        self._mesh_actor = None
        self._highlight_actor = None
        self._wire_actor = None
        self._electrode_actor = None
        self._offscreen_electrode_actor = None
        self._electrode_geometry: ElectrodeGeometry | None = None

        self._render_backend = "caption"
        self._display_mode = DISPLAY_MODE_VOLUME
        self._anomaly_mode = ANOMALY_MODE_POSITIVE
        self._prefer_central_anomaly_region = False
        self._point_cloud_original_count = 0
        self._point_cloud_display_count = 0
        self._last_vtk_disabled_reason: str | None = None
        self._colorbar_label = "S/m"
        self._colormap = "viridis"
        self._value_limits: tuple[float, float] | None = None
        self._mpl3d_host = None
        self._mpl3d_canvas = None
        self._offscreen_plotter = None
        self._offscreen_mesh_actor = None
        self._offscreen_highlight_actor = None
        self._offscreen_wire_actor = None
        self._offscreen_window_size: tuple[int, int] | None = None

        self._plotter = None
        self._plotter_ready = False
        # Holds the most recent payload while the plotter is still
        # being built (host hasn't fired its first showEvent yet).  As
        # soon as _init_plotter completes, _drain_pending_render kicks
        # in and renders this.
        self._pending_render: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, str | None]
        ] = None
        self._progressive_volume_pending_signature: tuple[int, int, int] | None = None
        self._suppress_auto_points_once = False

        # Drag throttle: coalesce rapid mouseMove events for the
        # offscreen backend.  Default to a high-performance 60 fps,
        # full-resolution interaction path; users on constrained
        # machines can lower EIT_APP_3D_DRAG_FPS or
        # EIT_APP_3D_DRAG_RENDER_SCALE.
        self._offscreen_drag_fps = _env_float(
            _OFFSCREEN_DRAG_FPS_ENV,
            60.0,
            lower=1.0,
            upper=120.0,
        )
        self._offscreen_drag_interval_ms = max(
            1, int(round(1000.0 / self._offscreen_drag_fps))
        )
        self._offscreen_drag_render_scale = _env_float(
            _OFFSCREEN_DRAG_RENDER_SCALE_ENV,
            1.0,
            lower=0.25,
            upper=1.0,
        )
        self._offscreen_render_timer = QTimer(self)
        self._offscreen_render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._offscreen_render_timer.setSingleShot(True)
        self._offscreen_render_timer.setInterval(self._offscreen_drag_interval_ms)
        self._offscreen_render_timer.timeout.connect(self._refresh_offscreen_pixmap)
        # Track drag state so optional render-scale downsampling applies
        # only while the user is actively rotating / zooming.
        self._is_dragging_offscreen = False
        self._drag_release_timer = QTimer(self)
        self._drag_release_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._drag_release_timer.setSingleShot(True)
        self._drag_release_timer.setInterval(
            _env_int(_OFFSCREEN_DRAG_IDLE_MS_ENV, 80, lower=16, upper=1000)
        )
        self._drag_release_timer.timeout.connect(self._on_drag_idle)
        self._progressive_volume_timer = QTimer(self)
        self._progressive_volume_timer.setSingleShot(True)
        self._progressive_volume_timer.timeout.connect(
            self._run_progressive_volume_upgrade
        )

        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        subscribe_theme_mode(self._on_theme_mode_changed)

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._title_label = QLabel(self._default_title)
        set_section_header(self._title_label)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("padding: 4px 0;")
        self._title_label.setMinimumSize(0, 0)
        outer.addWidget(self._title_label)

        self._stack_host = QFrame()
        self._stack = QStackedLayout(self._stack_host)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._caption_label = QLabel("")
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption_label.setWordWrap(True)
        # No forced min-height — the slot's outer layout gives it all
        # remaining space via the stretch factor on _stack_host; a
        # fixed 180 px floor dragged the whole main window up when the
        # 3D slot was even present.
        self._caption_label.setMinimumSize(0, 0)
        self._stack.addWidget(self._caption_label)

        self._offscreen_host = QWidget()
        self._offscreen_layout = QVBoxLayout(self._offscreen_host)
        self._offscreen_layout.setContentsMargins(0, 0, 0, 0)
        self._offscreen_label = _OffscreenRenderLabel()
        self._offscreen_label.dragged.connect(self._on_offscreen_dragged)
        self._offscreen_label.zoomed.connect(self._on_offscreen_zoomed)
        self._offscreen_layout.addWidget(self._offscreen_label)
        self._stack.addWidget(self._offscreen_host)

        # _InteractorHost defers its ``realized`` signal until the
        # widget is actually shown for the first time AND Qt has had
        # one event-loop tick to finish realisation — that's the
        # earliest moment we can safely hand the underlying native
        # window to VTK.
        self._interactor_host = _InteractorHost()
        self._interactor_host.realized.connect(self._init_plotter)
        self._interactor_layout = QVBoxLayout(self._interactor_host)
        self._interactor_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._interactor_host)
        self._stack.setCurrentWidget(self._caption_label)

        outer.addWidget(self._stack_host, 1)

        # Control panel: display mode + opacity slider + visibility toggles + reset.
        #
        # The tools are deliberately split across two compact rows.
        # A single long row looks fine in maximized windows, but each
        # top-pane slot can be only ~500 px wide at the default launch
        # size, and a one-line bar makes labels/buttons clip or spill
        # across the splitter.
        self._controls = QFrame()
        self._controls.setObjectName("conductivity3DControls")
        self._controls.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        grid = QGridLayout(self._controls)
        grid.setContentsMargins(6, 2, 6, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        self._controls_layout = grid

        self._display_mode_label = QLabel("")
        set_hint_text(self._display_mode_label)
        self._display_mode_label.setMinimumWidth(0)
        grid.addWidget(self._display_mode_label, 0, 0)

        self._display_mode_group = QButtonGroup(self)
        self._display_mode_group.setExclusive(True)
        self._volume_mode_btn = QPushButton("")
        self._volume_mode_btn.setCheckable(True)
        self._volume_mode_btn.setChecked(True)
        set_button_role(self._volume_mode_btn, "tertiary")
        self._volume_mode_btn.setMinimumWidth(34)
        self._volume_mode_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._volume_mode_btn.clicked.connect(
            lambda checked: (
                self.set_display_mode(DISPLAY_MODE_VOLUME) if checked else None
            )
        )
        self._display_mode_group.addButton(self._volume_mode_btn)
        grid.addWidget(self._volume_mode_btn, 0, 1)

        self._points_mode_btn = QPushButton("")
        self._points_mode_btn.setCheckable(True)
        set_button_role(self._points_mode_btn, "tertiary")
        self._points_mode_btn.setMinimumWidth(48)
        self._points_mode_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._points_mode_btn.clicked.connect(
            lambda checked: (
                self.set_display_mode(DISPLAY_MODE_POINTS) if checked else None
            )
        )
        self._display_mode_group.addButton(self._points_mode_btn)
        grid.addWidget(self._points_mode_btn, 0, 2)

        self._anomaly_mode_label = QLabel("")
        set_hint_text(self._anomaly_mode_label)
        self._anomaly_mode_label.setMinimumWidth(0)
        grid.addWidget(self._anomaly_mode_label, 0, 3)

        self._anomaly_mode_group = QButtonGroup(self)
        self._anomaly_mode_group.setExclusive(True)
        self._positive_anomaly_btn = QPushButton("")
        self._positive_anomaly_btn.setCheckable(True)
        self._positive_anomaly_btn.setChecked(True)
        set_button_role(self._positive_anomaly_btn, "tertiary")
        self._positive_anomaly_btn.setMinimumWidth(34)
        self._positive_anomaly_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._positive_anomaly_btn.clicked.connect(
            lambda checked: (
                self.set_anomaly_mode(ANOMALY_MODE_POSITIVE) if checked else None
            )
        )
        self._anomaly_mode_group.addButton(self._positive_anomaly_btn)
        grid.addWidget(self._positive_anomaly_btn, 0, 4)

        self._negative_anomaly_btn = QPushButton("")
        self._negative_anomaly_btn.setCheckable(True)
        set_button_role(self._negative_anomaly_btn, "tertiary")
        self._negative_anomaly_btn.setMinimumWidth(34)
        self._negative_anomaly_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._negative_anomaly_btn.clicked.connect(
            lambda checked: (
                self.set_anomaly_mode(ANOMALY_MODE_NEGATIVE) if checked else None
            )
        )
        self._anomaly_mode_group.addButton(self._negative_anomaly_btn)
        grid.addWidget(self._negative_anomaly_btn, 0, 5)

        self._absolute_anomaly_btn = QPushButton("")
        self._absolute_anomaly_btn.setCheckable(True)
        set_button_role(self._absolute_anomaly_btn, "tertiary")
        self._absolute_anomaly_btn.setMinimumWidth(40)
        self._absolute_anomaly_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._absolute_anomaly_btn.clicked.connect(
            lambda checked: (
                self.set_anomaly_mode(ANOMALY_MODE_ABSOLUTE) if checked else None
            )
        )
        self._anomaly_mode_group.addButton(self._absolute_anomaly_btn)
        grid.addWidget(self._absolute_anomaly_btn, 0, 6)

        self._opacity_label = QLabel("")
        set_hint_text(self._opacity_label)
        self._opacity_label.setMinimumWidth(0)
        grid.addWidget(self._opacity_label, 1, 0)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(5, 100)
        self._opacity_slider.setValue(45)
        self._opacity_slider.setMinimumWidth(48)
        self._opacity_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        grid.addWidget(self._opacity_slider, 1, 1, 1, 3)

        self._opacity_value = QLabel("0.45")
        set_hint_text(self._opacity_value)
        self._opacity_value.setMinimumWidth(0)
        grid.addWidget(self._opacity_value, 1, 4)

        self._highlight_check = QCheckBox("")
        self._highlight_check.setChecked(True)
        self._highlight_check.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._highlight_check.toggled.connect(self._on_highlight_toggled)
        grid.addWidget(self._highlight_check, 1, 5)

        self._wire_check = QCheckBox("")
        self._wire_check.setChecked(True)
        self._wire_check.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._wire_check.toggled.connect(self._on_wire_toggled)
        grid.addWidget(self._wire_check, 1, 6)

        # Electrode overlay toggle.  Disabled until a forward result
        # provides actual electrode geometry — checking it without a
        # cached patch list would render nothing and confuse the user.
        self._electrode_check = QCheckBox("")
        self._electrode_check.setChecked(False)
        self._electrode_check.setEnabled(False)
        self._electrode_check.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._electrode_check.toggled.connect(self._on_electrode_toggled)
        grid.addWidget(self._electrode_check, 1, 7)

        self._reset_btn = QPushButton("")
        set_button_role(self._reset_btn, "tertiary")
        self._reset_btn.setMinimumWidth(64)
        self._reset_btn.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self._reset_btn.clicked.connect(self._reset_camera)
        grid.addWidget(self._reset_btn, 0, 7)
        grid.setColumnStretch(3, 1)

        outer.addWidget(self._controls)
        # Hidden by default — only shown while the VTK interactor is
        # the active page.  This keeps the bar from contributing to
        # the 2D page's footprint inside the stacked dispatcher.
        self._controls.hide()

    # ------------------------------------------------------------------
    # Eager plotter init
    # ------------------------------------------------------------------

    def _init_plotter(self) -> None:
        """Construct the QtInteractor with auto_update disabled.

        Called exactly once, via ``_InteractorHost.realized`` — i.e.
        after Qt has finished placing the host inside the visible
        widget tree and given it a real native window.  Initialising
        VTK any earlier (in ``__init__`` or before the first show)
        crashes the renderer with ``BadWindow / X_ConfigureWindow``.

        ``auto_update=False`` disables pyvistaqt's 5 Hz background
        render timer; we drive renders explicitly from update_image,
        the slider, and the toggle checkboxes.
        """
        if self._plotter_ready:
            return

        vtk_enabled, reason = embedded_vtk_status()
        if not vtk_enabled:
            log.info("embedded PyVista/VTK viewer disabled: %s", reason)
            if self._pending_render is not None:
                sigma, coords, cells, title = self._pending_render
                self._pending_render = None
                if title is not None:
                    self.setTitle(title)
                self._last_image = (sigma, coords, cells, title)
                self._render_without_embedded_vtk(
                    sigma,
                    coords,
                    cells,
                    reason=reason,
                )
            else:
                self._show_caption(
                    t("sim.results.viewer3d_embedded_disabled"), kind="placeholder"
                )
            return

        try:
            import pyvista  # noqa: F401  (side-effect: VTK init)
            from pyvistaqt import QtInteractor

            _configure_vtk_logging()
        except Exception as exc:  # pragma: no cover — env without VTK
            log.warning("pyvistaqt unavailable; using safe 3D renderer: %s", exc)
            if self._pending_render is not None:
                sigma, coords, cells, title = self._pending_render
                self._pending_render = None
                if title is not None:
                    self.setTitle(title)
                self._last_image = (sigma, coords, cells, title)
                self._render_without_embedded_vtk(
                    sigma,
                    coords,
                    cells,
                    reason=str(exc),
                )
            else:
                self._show_caption(t("sim.results.viewer3d_unavailable"), kind="error")
            return

        palette = plot_palette()
        self._plotter = QtInteractor(
            self._interactor_host,
            off_screen=False,
            auto_update=False,
            multi_samples=4,
        )
        self._plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#ffffff")))
        self._plotter.add_axes()
        self._interactor_layout.addWidget(self._plotter)
        self._plotter_ready = True

        if self._pending_render is not None:
            sigma, coords, cells, title = self._pending_render
            self._pending_render = None
            if title is not None:
                self.setTitle(title)
            self._last_image = (sigma, coords, cells, title)
            self._build_scene(sigma, coords, cells)
            self._render_backend = "vtk"

    # ------------------------------------------------------------------
    # Public API (mirrors ConductivityImageWidget)
    # ------------------------------------------------------------------

    def setTitle(self, title: str) -> None:
        self._default_title = title
        self._title_label.setText(title)

    def display_mode(self) -> str:
        return self._display_mode

    def set_display_mode(self, mode: str) -> None:
        """Switch between transparent volume and cell-center point-cloud views."""
        if mode not in DISPLAY_MODES:
            raise ValueError(f"unknown 3D display mode: {mode!r}")
        if mode == self._display_mode:
            self._sync_display_mode_buttons()
            return

        self._display_mode = mode
        if mode == DISPLAY_MODE_VOLUME:
            self._suppress_auto_points_once = True
        self._sync_display_mode_buttons()
        if self._last_image is None:
            return
        sigma, coords, cells, title = self._last_image
        self.update_image(
            sigma,
            coords,
            cells,
            title=title,
            colorbar_label=self._colorbar_label,
            colormap=self._colormap,
            value_limits=self._value_limits,
        )

    def _sync_display_mode_buttons(self) -> None:
        self._volume_mode_btn.setChecked(self._display_mode == DISPLAY_MODE_VOLUME)
        self._points_mode_btn.setChecked(self._display_mode == DISPLAY_MODE_POINTS)

    def anomaly_mode(self) -> str:
        return self._anomaly_mode

    def set_anomaly_mode(self, mode: str) -> None:
        """Choose which conductivity deviation sign is highlighted."""
        if mode not in ANOMALY_MODES:
            raise ValueError(f"unknown anomaly mode: {mode!r}")
        if mode == self._anomaly_mode:
            self._sync_anomaly_mode_buttons()
            return

        self._anomaly_mode = mode
        self._sync_anomaly_mode_buttons()
        if self._last_image is None:
            return
        sigma, coords, cells, title = self._last_image
        self.update_image(
            sigma,
            coords,
            cells,
            title=title,
            colorbar_label=self._colorbar_label,
            colormap=self._colormap,
            value_limits=self._value_limits,
        )

    def set_prefer_central_anomaly_region(self, enabled: bool) -> None:
        """Prefer a large central coherent region over boundary extrema."""
        prefer = bool(enabled)
        if prefer == self._prefer_central_anomaly_region:
            return
        self._prefer_central_anomaly_region = prefer
        if self._last_image is None:
            return
        sigma, coords, cells, title = self._last_image
        self.update_image(
            sigma,
            coords,
            cells,
            title=title,
            colorbar_label=self._colorbar_label,
            colormap=self._colormap,
            value_limits=self._value_limits,
        )

    def _sync_anomaly_mode_buttons(self) -> None:
        self._positive_anomaly_btn.setChecked(
            self._anomaly_mode == ANOMALY_MODE_POSITIVE
        )
        self._negative_anomaly_btn.setChecked(
            self._anomaly_mode == ANOMALY_MODE_NEGATIVE
        )
        self._absolute_anomaly_btn.setChecked(
            self._anomaly_mode == ANOMALY_MODE_ABSOLUTE
        )

    def _render_without_embedded_vtk(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
        *,
        reason: str,
    ) -> None:
        offscreen_failure = _pyvista_offscreen_failure_reason()
        log_reason = (
            f"{reason}; PyVista offscreen disabled after previous failure: "
            f"{offscreen_failure}"
            if offscreen_failure is not None
            else reason
        )
        if log_reason != self._last_vtk_disabled_reason:
            action = (
                "showing 3D unavailable caption"
                if offscreen_failure is not None
                else "trying PyVista offscreen"
            )
            log.info(
                "embedded PyVistaQt viewer unavailable: %s; %s",
                log_reason,
                action,
            )
            self._last_vtk_disabled_reason = log_reason
        self._pending_render = None
        self._discard_actors()
        if offscreen_failure is not None:
            self._show_pyvista_offscreen_unavailable(offscreen_failure)
            return
        try:
            rendered_offscreen = self._render_pyvista_offscreen_scene(
                sigma,
                coords,
                cells,
            )
        except Exception as exc:  # pragma: no cover - graphics runtime edge case
            log.warning(
                "PyVista offscreen renderer failed; 3D view unavailable: %s",
                exc,
            )
            _mark_pyvista_offscreen_failure(exc)
            self._discard_offscreen_plotter()
            self._show_pyvista_offscreen_unavailable(str(exc))
            return
        if not rendered_offscreen:
            self._show_pyvista_offscreen_unavailable(
                _pyvista_offscreen_failure_reason() or reason
            )

    def _show_pyvista_offscreen_unavailable(self, reason: str | None) -> None:
        text = t("sim.results.viewer3d_unavailable")
        if reason:
            text = f"{text}\n{reason}"
        self._show_caption(text, kind="error")

    def _render_matplotlib_scene(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        del sigma, coords, cells
        self._show_pyvista_offscreen_unavailable(
            "Matplotlib 3D rendering is disabled for conductivity volume views."
        )

    def update_image(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None = None,
        *,
        colorbar_label: str = "S/m",
        colormap: str = "viridis",
        value_limits: tuple[float, float] | None = None,
    ) -> None:
        """Render a 3D volume conductivity field."""
        cells = _display_cells_array(cell_connectivity)
        coords = _display_coords_array(node_coords)
        if (
            coords.ndim != 2
            or cells.ndim != 2
            or coords.shape[1] < 3
            or cells.shape[1] not in SUPPORTED_3D_CELL_VERTEX_COUNTS
        ):
            self._show_caption(t("sim.results.viewer3d_bad_mesh"), kind="error")
            return

        sigma = _display_sigma_array(conductivity)
        if sigma.shape[0] not in (cells.shape[0], coords.shape[0]):
            self._show_caption(t("sim.results.viewer3d_size_mismatch"), kind="error")
            return

        if title is not None:
            self.setTitle(title)
        self._colorbar_label = colorbar_label
        self._colormap = colormap
        self._value_limits = _sanitize_display_value_limits(value_limits)
        self._last_image = (sigma, coords, cells, title)
        auto_points = (
            self._display_mode == DISPLAY_MODE_VOLUME
            and not self._suppress_auto_points_once
            and _should_auto_points(cells.shape[0])
        )
        self._suppress_auto_points_once = False
        if auto_points:
            self._display_mode = DISPLAY_MODE_POINTS
            self._sync_display_mode_buttons()
            self._schedule_progressive_volume_upgrade(sigma, coords, cells)
        else:
            self._progressive_volume_pending_signature = None
            self._progressive_volume_timer.stop()

        vtk_enabled, reason = embedded_vtk_status()
        if not vtk_enabled:
            self._render_without_embedded_vtk(
                sigma,
                coords,
                cells,
                reason=reason,
            )
            return

        # Switching the stacked layout to the interactor host first
        # gets the host into the visible tree.  On the very first
        # switch this triggers _InteractorHost.showEvent, which
        # eventually fires .realized → _init_plotter → builds the
        # scene from _pending_render.  On subsequent calls the
        # plotter is already ready and we render straight away.
        self._render_backend = "vtk"
        self._stack.setCurrentWidget(self._interactor_host)
        self._controls.show()

        if not self._plotter_ready:
            self._pending_render = (sigma, coords, cells, title)
            return

        self._build_scene(sigma, coords, cells)

    def _schedule_progressive_volume_upgrade(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        if not _progressive_volume_upgrade_enabled():
            return
        self._progressive_volume_pending_signature = (id(sigma), id(coords), id(cells))
        self._progressive_volume_timer.start(_progressive_volume_delay_ms())

    def _run_progressive_volume_upgrade(self) -> None:
        signature = self._progressive_volume_pending_signature
        self._progressive_volume_pending_signature = None
        if signature is None or self._last_image is None:
            return
        sigma, coords, cells, _title = self._last_image
        if signature != (id(sigma), id(coords), id(cells)):
            return
        if self._display_mode != DISPLAY_MODE_POINTS:
            return
        self._suppress_auto_points_once = True
        self.set_display_mode(DISPLAY_MODE_VOLUME)

    def clear(self) -> None:
        """Drop any rendered data and show the placeholder caption."""
        self._discard_actors()
        self._last_image = None
        self._pending_render = None
        self._progressive_volume_pending_signature = None
        self._progressive_volume_timer.stop()
        self._show_caption(t("sim.results.viewer3d_no_data"), kind="placeholder")

    def set_loading(self, message: str | None = None) -> None:
        """Show a centered loading caption while a forward / inverse solve runs."""
        text = message or t("sim.results.viewer3d_loading")
        self._show_caption(text, kind="loading")

    def set_electrode_geometry(self, geometry: ElectrodeGeometry | None) -> None:
        """Cache electrode geometry and (re)build the cached actor if shown.

        Building is gated on ``geometry.is_3d()`` because the 3D widget
        is only meaningful for cylinder data — a 2D-only ElectrodeGeometry
        passed in (e.g. when the underlying forward solve was 2D and the
        3D widget happens to be alive but not visible) yields no patches
        and leaves the toggle disabled.
        """
        self._electrode_geometry = geometry
        has_patches = bool(geometry and geometry.is_3d() and geometry.patches)
        self._electrode_check.setEnabled(has_patches)
        if not has_patches:
            self._electrode_check.setChecked(False)
            self._remove_all_electrode_actors()
            self._render_active_backend_if_visible()
            return
        # Build (or rebuild) the cached actor on whichever backend is
        # currently active, then honour the user's checkbox state.
        self._refresh_electrode_actors()
        self._render_active_backend_if_visible()

    def _remove_all_electrode_actors(self) -> None:
        if self._plotter is not None and self._electrode_actor is not None:
            try:
                self._plotter.remove_actor(self._electrode_actor, render=False)
            except Exception:  # pragma: no cover — VTK quirk
                pass
        self._electrode_actor = None
        if (
            self._offscreen_plotter is not None
            and self._offscreen_electrode_actor is not None
        ):
            try:
                self._offscreen_plotter.remove_actor(
                    self._offscreen_electrode_actor, render=False
                )
            except Exception:  # pragma: no cover — VTK quirk
                pass
        self._offscreen_electrode_actor = None

    def _refresh_electrode_actors(self) -> None:
        """Build cached electrode geometry on the currently active backend.

        Other backends defer construction to the next time their scene
        rebuilds via update_image() — there is no point materialising a
        VTK actor on a backend the user is not looking at.
        """
        geometry = self._electrode_geometry
        if geometry is None or not geometry.is_3d() or not geometry.patches:
            return
        if self._render_backend == "vtk":
            self._build_vtk_electrode_actor(geometry)
        elif self._render_backend == "pyvista_offscreen":
            self._build_offscreen_electrode_actor(geometry)

    def _render_active_backend_if_visible(self) -> None:
        if self._render_backend == "vtk" and self._plotter is not None:
            self._plotter.render()
        elif self._render_backend == "pyvista_offscreen":
            self._refresh_offscreen_pixmap()

    def _build_vtk_electrode_actor(self, geometry: ElectrodeGeometry) -> None:
        if self._plotter is None:
            return
        try:
            import pyvista  # noqa: F401  (side-effect: VTK init)
        except Exception:  # pragma: no cover — env without VTK
            return
        if self._electrode_actor is not None:
            try:
                self._plotter.remove_actor(self._electrode_actor, render=False)
            except Exception:  # pragma: no cover — VTK quirk
                pass
            self._electrode_actor = None
        polydata = self._build_electrode_polydata(geometry)
        if polydata is None:
            return
        palette = plot_palette()
        color = palette.get("highlight", "#f39c12")
        self._electrode_actor = self._plotter.add_mesh(
            polydata,
            color=color,
            opacity=0.9,
            show_edges=False,
            show_scalar_bar=False,
        )
        self._electrode_actor.SetVisibility(bool(self._electrode_check.isChecked()))

    def _build_offscreen_electrode_actor(self, geometry: ElectrodeGeometry) -> None:
        if self._offscreen_plotter is None:
            return
        try:
            import pyvista  # noqa: F401  (side-effect: VTK init)
        except Exception:  # pragma: no cover — env without VTK
            return
        if self._offscreen_electrode_actor is not None:
            try:
                self._offscreen_plotter.remove_actor(
                    self._offscreen_electrode_actor, render=False
                )
            except Exception:  # pragma: no cover — VTK quirk
                pass
            self._offscreen_electrode_actor = None
        polydata = self._build_electrode_polydata(geometry)
        if polydata is None:
            return
        palette = plot_palette()
        color = palette.get("highlight", "#f39c12")
        self._offscreen_electrode_actor = self._offscreen_plotter.add_mesh(
            polydata,
            color=color,
            opacity=0.9,
            show_edges=False,
            show_scalar_bar=False,
        )
        self._offscreen_electrode_actor.SetVisibility(
            bool(self._electrode_check.isChecked())
        )

    def _build_electrode_polydata(self, geometry: ElectrodeGeometry):
        try:
            import pyvista as pv
        except Exception:  # pragma: no cover — env without VTK
            return None
        points, triangles = default_patch_quads(geometry.patches, geometry.radius)
        if points.shape[0] == 0 or triangles.shape[0] == 0:
            return None
        # PyVista expects [3, i0, i1, i2, 3, i0, i1, i2, ...] face buffer.
        face_buffer = np.empty((triangles.shape[0], 4), dtype=np.int64)
        face_buffer[:, 0] = 3
        face_buffer[:, 1:] = triangles
        return pv.PolyData(points, face_buffer.ravel())

    def _on_electrode_toggled(self, checked: bool) -> None:
        # Cache hit on every toggle — actors already built by
        # set_electrode_geometry / build_scene; just flip visibility.
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_electrode_actor is None and checked:
                self._build_offscreen_electrode_actor(self._electrode_geometry)
            if self._offscreen_electrode_actor is not None:
                self._offscreen_electrode_actor.SetVisibility(bool(checked))
            self._refresh_offscreen_pixmap()
            return
        if self._render_backend == "vtk":
            if (
                self._electrode_actor is None
                and checked
                and self._electrode_geometry is not None
            ):
                self._build_vtk_electrode_actor(self._electrode_geometry)
            if self._electrode_actor is not None:
                self._electrode_actor.SetVisibility(bool(checked))
            if self._plotter is not None:
                self._plotter.render()

    # ------------------------------------------------------------------
    # Scene construction (heavy work — runs once per update_image only)
    # ------------------------------------------------------------------

    def _discard_actors(self) -> None:
        if self._plotter is None:
            return
        for actor_attr in (
            "_mesh_actor",
            "_highlight_actor",
            "_wire_actor",
            "_electrode_actor",
        ):
            actor = getattr(self, actor_attr, None)
            if actor is not None:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:  # pragma: no cover — VTK quirk
                    pass
                setattr(self, actor_attr, None)

    def _discard_offscreen_plotter(self) -> None:
        if self._offscreen_plotter is not None:
            try:
                self._offscreen_plotter.close()
            except Exception:  # pragma: no cover — best-effort VTK cleanup
                pass
        self._offscreen_plotter = None
        self._offscreen_mesh_actor = None
        self._offscreen_highlight_actor = None
        self._offscreen_wire_actor = None
        self._offscreen_electrode_actor = None
        self._offscreen_window_size = None

    def _offscreen_render_size(self) -> tuple[int, int]:
        dpr = max(float(self.devicePixelRatioF()), 1.0)
        # Full-resolution drag is the default for high-performance
        # workstations.  Optional downsampling is still available via
        # EIT_APP_3D_DRAG_RENDER_SCALE for constrained machines.
        scale = (
            self._offscreen_drag_render_scale if self._is_dragging_offscreen else 1.0
        )
        width = max(
            320,
            int(round(max(self._offscreen_label.width(), 1) * dpr * scale)),
        )
        height = max(
            240,
            int(round(max(self._offscreen_label.height(), 1) * dpr * scale)),
        )
        return min(width, 2400), min(height, 1800)

    def _offscreen_pixmap_target(self) -> tuple[int, int, float]:
        """Return physical pixmap size + DPR for stable QLabel display.

        If drag downsampling is enabled, Qt would otherwise treat the
        smaller framebuffer as a smaller logical image and the scene would
        visibly shrink until the final full-resolution frame arrives.
        """
        dpr = max(float(self.devicePixelRatioF()), 1.0)
        label_width = max(float(self._offscreen_label.width()), 1.0)
        label_height = max(float(self._offscreen_label.height()), 1.0)
        width = min(max(1, int(round(label_width * dpr))), 2400)
        height = min(max(1, int(round(label_height * dpr))), 1800)
        return width, height, dpr

    def _refresh_offscreen_pixmap(self) -> bool:
        plotter = self._offscreen_plotter
        if plotter is None:
            return False
        width, height = self._offscreen_render_size()
        try:
            window_size = (width, height)
            if self._offscreen_window_size != window_size:
                plotter.window_size = window_size
                self._offscreen_window_size = window_size
            plotter.render()
            image = np.ascontiguousarray(plotter.screenshot(return_img=True))
        except Exception as exc:  # pragma: no cover — VTK runtime edge case
            log.warning("PyVista offscreen render failed: %s", exc)
            return False
        if image.ndim != 3 or image.shape[2] < 3:
            return False
        image = image[:, :, :3]
        qimage = QImage(
            image.data,
            int(image.shape[1]),
            int(image.shape[0]),
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        target_width, target_height, target_dpr = self._offscreen_pixmap_target()
        if pixmap.width() != target_width or pixmap.height() != target_height:
            pixmap = pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        pixmap.setDevicePixelRatio(target_dpr)
        self._offscreen_label.setPixmap(pixmap)
        return True

    def _add_pyvista_point_cloud_actors(
        self,
        *,
        pv,
        plotter,
        centers: np.ndarray,
        cell_sigma: np.ndarray,
        sigma_min: float,
        sigma_max: float,
        opacity: float,
        colorbar_label: str,
        colormap: str,
        text_color: tuple[float, float, float],
        offscreen: bool,
    ) -> None:
        sample_idx = _point_cloud_sample_indices(
            cell_sigma,
            centers,
            anomaly_mode=self._anomaly_mode,
            prefer_central_region=self._prefer_central_anomaly_region,
        )
        display_centers, display_sigma = _point_cloud_display_arrays(
            centers,
            cell_sigma,
            sample_idx,
        )
        self._point_cloud_original_count = int(np.asarray(cell_sigma).size)
        self._point_cloud_display_count = int(display_sigma.size)

        cloud = pv.PolyData(display_centers)
        cloud["sigma"] = display_sigma
        scalar_bar_y = 0.12 if offscreen else 0.05
        scalar_bar_height = 0.55 if offscreen else 0.6
        mesh_actor = plotter.add_mesh(
            cloud,
            scalars="sigma",
            cmap=colormap,
            clim=[sigma_min, sigma_max],
            opacity=opacity,
            render_points_as_spheres=True,
            point_size=_pyvista_point_size(display_centers.shape[0]),
            show_scalar_bar=True,
            scalar_bar_args={
                "title": colorbar_label,
                "color": text_color,
                "vertical": True,
                "position_x": 0.88,
                "position_y": scalar_bar_y,
                "width": 0.06 if offscreen else 0.07,
                "height": scalar_bar_height,
                "title_font_size": 14,
                "label_font_size": 10 if offscreen else 11,
            },
        )
        if offscreen:
            self._offscreen_mesh_actor = mesh_actor
        else:
            self._mesh_actor = mesh_actor

        inhom_mask = _cell_anomaly_mask(
            display_sigma,
            self._anomaly_mode,
            cell_centers=display_centers,
            prefer_central_region=self._prefer_central_anomaly_region,
        )
        highlight_centers, highlight_sigma = _point_cloud_highlight_arrays(
            display_centers,
            display_sigma,
            inhom_mask,
        )
        if highlight_sigma.size == 0:
            return
        highlight_cloud = pv.PolyData(highlight_centers)
        highlight_cloud["sigma"] = highlight_sigma
        highlight_actor = plotter.add_mesh(
            highlight_cloud,
            scalars="sigma",
            cmap=colormap,
            clim=[sigma_min, sigma_max],
            opacity=1.0,
            render_points_as_spheres=True,
            point_size=max(
                _pyvista_point_size(display_centers.shape[0]) * 1.55,
                6.0,
            ),
            show_scalar_bar=False,
        )
        highlight_actor.SetVisibility(bool(self._highlight_check.isChecked()))
        if offscreen:
            self._offscreen_highlight_actor = highlight_actor
        else:
            self._highlight_actor = highlight_actor

    def _render_pyvista_offscreen_scene(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> bool:
        try:
            import pyvista as pv

            _configure_vtk_logging()
        except Exception as exc:
            log.info("PyVista offscreen renderer unavailable: %s", exc)
            _mark_pyvista_offscreen_failure(exc)
            return False

        self._discard_offscreen_plotter()

        n_cells = cells.shape[0]
        cell_sigma, scalar_mode = _cell_center_sigma(sigma, cells)
        if scalar_mode == "cell":
            scalar_kw = {"scalars": "sigma", "preference": "cell"}
        else:
            scalar_kw = {"scalars": "sigma", "preference": "point"}

        sigma_min, sigma_max = _display_color_limits(cell_sigma, self._value_limits)

        palette = plot_palette()
        width, height = self._offscreen_render_size()
        try:
            plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        except Exception as exc:  # pragma: no cover — VTK runtime edge case
            log.info("PyVista offscreen plotter init failed: %s", exc)
            _mark_pyvista_offscreen_failure(exc)
            return False
        plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#ffffff")))
        plotter.add_axes()
        self._offscreen_plotter = plotter
        self._offscreen_window_size = (width, height)

        opacity = self._opacity_slider.value() / 100.0
        text_color = _hex_to_rgb(palette.get("text", "#222"))
        if self._display_mode == DISPLAY_MODE_POINTS:
            self._add_pyvista_point_cloud_actors(
                pv=pv,
                plotter=plotter,
                centers=_cell_centers(coords, cells),
                cell_sigma=cell_sigma,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                opacity=opacity,
                colorbar_label=self._colorbar_label,
                colormap=self._colormap,
                text_color=text_color,
                offscreen=True,
            )
            plotter.reset_camera()
            self._stack.setCurrentWidget(self._offscreen_host)
            self._controls.show()
            self._render_backend = "pyvista_offscreen"
            if self._electrode_geometry is not None:
                self._build_offscreen_electrode_actor(self._electrode_geometry)
            if not self._refresh_offscreen_pixmap():
                _mark_pyvista_offscreen_failure("initial offscreen render failed")
                self._discard_offscreen_plotter()
                return False
            return True

        verts_per_cell = cells.shape[1]
        if verts_per_cell == 4:
            cell_type = pv.CellType.TETRA
        elif verts_per_cell == 8:
            cell_type = pv.CellType.HEXAHEDRON
        else:
            return False

        cell_array = np.empty((n_cells, verts_per_cell + 1), dtype=np.int64)
        cell_array[:, 0] = verts_per_cell
        cell_array[:, 1:] = cells
        cell_types = np.full(n_cells, cell_type, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array.ravel(), cell_types, coords)
        if scalar_mode == "cell":
            grid.cell_data["sigma"] = cell_sigma
        else:
            grid.point_data["sigma"] = sigma

        self._offscreen_mesh_actor = plotter.add_mesh(
            grid,
            cmap=self._colormap,
            opacity=opacity,
            clim=[sigma_min, sigma_max],
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": self._colorbar_label,
                "color": text_color,
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.12,
                "width": 0.06,
                "height": 0.55,
                "title_font_size": 14,
                "label_font_size": 10,
            },
            **scalar_kw,
        )

        if scalar_mode == "cell" and self._display_mode == DISPLAY_MODE_VOLUME:
            inhom_mask = _cell_anomaly_mask(
                cell_sigma,
                self._anomaly_mode,
                cell_centers=_cell_centers(coords, cells),
                prefer_central_region=self._prefer_central_anomaly_region,
            )
            inhom_grid = _extract_cells_from_mask(grid, inhom_mask)
            if inhom_grid is not None and inhom_grid.n_cells > 0:
                self._offscreen_highlight_actor = plotter.add_mesh(
                    inhom_grid,
                    scalars="sigma",
                    preference="cell",
                    cmap=self._colormap,
                    clim=[sigma_min, sigma_max],
                    opacity=1.0,
                    show_edges=False,
                    show_scalar_bar=False,
                )
                self._offscreen_highlight_actor.SetVisibility(
                    bool(self._highlight_check.isChecked())
                )

        outline = _pyvista_feature_outline(grid, feature_angle=30.0)
        if outline.n_points > 0:
            self._offscreen_wire_actor = plotter.add_mesh(
                outline,
                color=palette.get("border", "#888"),
                line_width=1.0,
                opacity=0.45,
                show_scalar_bar=False,
            )
            self._offscreen_wire_actor.SetVisibility(bool(self._wire_check.isChecked()))

        plotter.reset_camera()
        self._stack.setCurrentWidget(self._offscreen_host)
        self._controls.show()
        self._render_backend = "pyvista_offscreen"
        # Re-attach cached electrode patches if a forward result already
        # supplied geometry — discard_offscreen_plotter above wiped the
        # actor pointer.
        if self._electrode_geometry is not None:
            self._build_offscreen_electrode_actor(self._electrode_geometry)
        if not self._refresh_offscreen_pixmap():
            _mark_pyvista_offscreen_failure("initial offscreen render failed")
            self._discard_offscreen_plotter()
            return False
        return True

    def _build_scene(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        """Build the bulk + highlight + wireframe actors *once* per
        update_image and cache them.  Slider / checkbox interactions
        then mutate actor properties (opacity, visibility) without
        having to rebuild the grid or recompute feature edges — that
        is what keeps interactive response below 16 ms per gesture.
        """
        import pyvista as pv

        plotter = self._plotter
        self._discard_actors()

        n_cells = cells.shape[0]
        cell_sigma, scalar_mode = _cell_center_sigma(sigma, cells)
        if scalar_mode == "cell":
            scalar_kw = {"scalars": "sigma", "preference": "cell"}
        else:
            # Node-centered sigma — VTK takes that natively.
            scalar_kw = {"scalars": "sigma", "preference": "point"}

        sigma_min, sigma_max = _display_color_limits(cell_sigma, self._value_limits)
        opacity = self._opacity_slider.value() / 100.0
        palette = plot_palette()
        text_color = _hex_to_rgb(palette.get("text", "#222"))

        if self._display_mode == DISPLAY_MODE_POINTS:
            self._add_pyvista_point_cloud_actors(
                pv=pv,
                plotter=plotter,
                centers=_cell_centers(coords, cells),
                cell_sigma=cell_sigma,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                opacity=opacity,
                colorbar_label=self._colorbar_label,
                colormap=self._colormap,
                text_color=text_color,
                offscreen=False,
            )
            plotter.reset_camera()
            plotter.render()
            return

        # Build the unstructured volume grid.  VTK expects
        # [n_pts, p0, p1, ...] rows; support both tetra meshes from the
        # CPU path and hex meshes from the CUDA-structured path.
        verts_per_cell = cells.shape[1]
        if verts_per_cell == 4:
            cell_type = pv.CellType.TETRA
        elif verts_per_cell == 8:
            cell_type = pv.CellType.HEXAHEDRON
        else:  # update_image() guards this; keep a defensive fallback.
            self._show_caption(t("sim.results.viewer3d_bad_mesh"), kind="error")
            return

        cell_array = np.empty((n_cells, verts_per_cell + 1), dtype=np.int64)
        cell_array[:, 0] = verts_per_cell
        cell_array[:, 1:] = cells
        cell_types = np.full(n_cells, cell_type, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array.ravel(), cell_types, coords)
        if scalar_mode == "cell":
            grid.cell_data["sigma"] = cell_sigma
        else:
            grid.point_data["sigma"] = sigma

        # Bulk volume: alpha-blended so we can see through to interior
        # cells whose conductivity differs from background.
        self._mesh_actor = plotter.add_mesh(
            grid,
            cmap=self._colormap,
            opacity=opacity,
            clim=[sigma_min, sigma_max],
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": self._colorbar_label,
                "color": text_color,
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.05,
                "width": 0.07,
                "height": 0.6,
                "title_font_size": 14,
                "label_font_size": 11,
            },
            **scalar_kw,
        )

        # Highlight overlay: cells whose conductivity is far from the
        # median (i.e. the "inhomogeneity") rendered opaque so a small
        # central inclusion still reads even when the bulk opacity is
        # high.  Built always; visibility toggles with the checkbox.
        if scalar_mode == "cell" and self._display_mode == DISPLAY_MODE_VOLUME:
            inhom_mask = _cell_anomaly_mask(
                cell_sigma,
                self._anomaly_mode,
                cell_centers=_cell_centers(coords, cells),
                prefer_central_region=self._prefer_central_anomaly_region,
            )
            inhom_grid = _extract_cells_from_mask(grid, inhom_mask)
            if inhom_grid is not None and inhom_grid.n_cells > 0:
                self._highlight_actor = plotter.add_mesh(
                    inhom_grid,
                    scalars="sigma",
                    preference="cell",
                    cmap=self._colormap,
                    clim=[sigma_min, sigma_max],
                    opacity=1.0,
                    show_edges=False,
                    show_scalar_bar=False,
                )
                self._highlight_actor.SetVisibility(
                    bool(self._highlight_check.isChecked())
                )

        # Wireframe overlay: feature edges of the boundary surface,
        # gives the bulk shape a clean silhouette at low opacity.
        outline = _pyvista_feature_outline(grid, feature_angle=30.0)
        if outline.n_points > 0:
            self._wire_actor = plotter.add_mesh(
                outline,
                color=palette.get("border", "#888"),
                line_width=1.0,
                opacity=0.4,
                show_scalar_bar=False,
            )
            self._wire_actor.SetVisibility(bool(self._wire_check.isChecked()))

        # Re-attach electrode patch actor if we already have geometry —
        # _discard_actors() above wiped the cached pointer.
        if self._electrode_geometry is not None:
            self._build_vtk_electrode_actor(self._electrode_geometry)

        plotter.reset_camera()
        plotter.render()

    # ------------------------------------------------------------------
    # Interactive controls — actor mutation only, never a full rebuild
    # ------------------------------------------------------------------

    def _on_opacity_changed(self, value: int) -> None:
        opacity = value / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_mesh_actor is not None:
                self._offscreen_mesh_actor.GetProperty().SetOpacity(opacity)
            self._refresh_offscreen_pixmap()
            return
        if self._mesh_actor is None or self._plotter is None:
            return
        self._mesh_actor.GetProperty().SetOpacity(opacity)
        self._plotter.render()

    def _on_highlight_toggled(self, checked: bool) -> None:
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_highlight_actor is not None:
                self._offscreen_highlight_actor.SetVisibility(bool(checked))
            self._refresh_offscreen_pixmap()
            return
        if self._highlight_actor is None or self._plotter is None:
            return
        self._highlight_actor.SetVisibility(bool(checked))
        self._plotter.render()

    def _on_wire_toggled(self, checked: bool) -> None:
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_wire_actor is not None:
                self._offscreen_wire_actor.SetVisibility(bool(checked))
            self._refresh_offscreen_pixmap()
            return
        if self._wire_actor is None or self._plotter is None:
            return
        self._wire_actor.SetVisibility(bool(checked))
        self._plotter.render()

    def _reset_camera(self) -> None:
        """Reset both data extent AND viewing angle across all backends.

        Each backend stores its camera state differently.  PyVista's
        reset_camera fits the actors but does not restore an isometric
        orientation by itself, so the button restores both the data
        range and a known canonical view direction.
        """
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_plotter is not None:
                self._offscreen_plotter.reset_camera()
                # Bring the orientation back to isometric so a click
                # always lands on the same canonical view, no matter
                # how the user has dragged the scene.
                try:
                    self._offscreen_plotter.view_isometric()
                except Exception:  # pragma: no cover — VTK quirk
                    pass
                self._refresh_offscreen_pixmap()
            return
        if self._plotter is not None:
            self._plotter.reset_camera()
            try:
                self._plotter.view_isometric()
            except Exception:  # pragma: no cover — VTK quirk
                pass
            self._plotter.render()

    def _on_offscreen_dragged(self, dx: float, dy: float) -> None:
        if (
            self._render_backend != "pyvista_offscreen"
            or self._offscreen_plotter is None
        ):
            return
        camera = self._offscreen_plotter.camera
        camera.Azimuth(-dx * 0.45)
        camera.Elevation(dy * 0.45)
        camera.OrthogonalizeViewUp()
        # Mark drag-active so the optional render-scale profile applies;
        # reset shortly after the user pauses so any downsampled profile
        # gets one final full-resolution frame.
        self._is_dragging_offscreen = True
        self._drag_release_timer.start()
        self._schedule_offscreen_refresh()

    def _on_offscreen_zoomed(self, delta_y: float) -> None:
        if (
            self._render_backend != "pyvista_offscreen"
            or self._offscreen_plotter is None
        ):
            return
        self._offscreen_plotter.camera.Zoom(1.12 if delta_y > 0 else 0.89)
        self._is_dragging_offscreen = True
        self._drag_release_timer.start()
        self._schedule_offscreen_refresh()

    def _schedule_offscreen_refresh(self) -> None:
        """Coalesce rapid drag events into the configured refresh rate."""
        if not self._offscreen_render_timer.isActive():
            self._offscreen_render_timer.start()

    def _on_drag_idle(self) -> None:
        """User stopped dragging — render one final crisp frame."""
        if self._render_backend != "pyvista_offscreen":
            self._is_dragging_offscreen = False
            return
        self._is_dragging_offscreen = False
        self._refresh_offscreen_pixmap()

    # ------------------------------------------------------------------
    # Caption / theme handling
    # ------------------------------------------------------------------

    def _show_caption(self, text: str, *, kind: str) -> None:
        self._render_backend = "caption"
        self._discard_offscreen_plotter()
        palette = plot_palette()
        color = {
            "placeholder": palette.get("caption", "#888"),
            "loading": palette.get("caption_loading", "#1f5d8b"),
            "error": palette.get("caption_error", "#c0392b"),
        }.get(kind, palette.get("caption", "#888"))
        self._caption_label.setText(text)
        self._caption_label.setStyleSheet(
            f"color: {color}; font-size: 13px; padding: 36px;"
        )
        self._stack.setCurrentWidget(self._caption_label)
        # Hide the controls bar whenever the interactor isn't the
        # active page.  The controls only make sense against a live
        # VTK scene, and hiding them removes their contribution to
        # the widget's minimum-size floor.
        self._controls.hide()

    def _on_theme_mode_changed(self, _mode: str) -> None:
        if self._render_backend == "pyvista_offscreen" and self._last_image is not None:
            self._render_pyvista_offscreen_scene(*self._last_image[:3])
            return
        if self._plotter is not None:
            palette = plot_palette()
            self._plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#fff")))
            if self._last_image is not None:
                # Cheap rebuild: same data, picks up new border / text
                # colours on the wire actor + scalar bar.
                self._build_scene(*self._last_image[:3])
        if self._last_image is None and self._caption_label.text():
            self._show_caption(self._caption_label.text(), kind="placeholder")

    # ------------------------------------------------------------------
    # i18n
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self._display_mode_label.setText(t("sim.results.viewer3d_display"))
        self._volume_mode_btn.setText(t("sim.results.viewer3d_display_volume_short"))
        self._volume_mode_btn.setToolTip(t("sim.results.viewer3d_display_volume"))
        self._points_mode_btn.setText(t("sim.results.viewer3d_display_points_short"))
        self._points_mode_btn.setToolTip(t("sim.results.viewer3d_display_points"))
        self._anomaly_mode_label.setText(t("sim.results.viewer3d_anomaly_mode"))
        self._positive_anomaly_btn.setText(
            t("sim.results.viewer3d_anomaly_positive_short")
        )
        self._positive_anomaly_btn.setToolTip(
            t("sim.results.viewer3d_anomaly_positive")
        )
        self._negative_anomaly_btn.setText(
            t("sim.results.viewer3d_anomaly_negative_short")
        )
        self._negative_anomaly_btn.setToolTip(
            t("sim.results.viewer3d_anomaly_negative")
        )
        self._absolute_anomaly_btn.setText(
            t("sim.results.viewer3d_anomaly_absolute_short")
        )
        self._absolute_anomaly_btn.setToolTip(
            t("sim.results.viewer3d_anomaly_absolute")
        )
        self._opacity_label.setText(t("sim.results.viewer3d_opacity_short"))
        self._opacity_label.setToolTip(t("sim.results.viewer3d_opacity"))
        self._highlight_check.setText(t("sim.results.viewer3d_highlight_short"))
        self._highlight_check.setToolTip(t("sim.results.viewer3d_highlight"))
        self._wire_check.setText(t("sim.results.viewer3d_wireframe_short"))
        self._wire_check.setToolTip(t("sim.results.viewer3d_wireframe"))
        self._electrode_check.setText(t("sim.results.electrodes_toggle_short"))
        self._electrode_check.setToolTip(t("sim.results.electrodes_toggle"))
        self._reset_btn.setText(t("sim.results.viewer3d_reset_short"))
        self._reset_btn.setToolTip(t("sim.results.viewer3d_reset"))
        if self._last_image is None and not self._caption_label.text():
            self._show_caption(t("sim.results.viewer3d_no_data"), kind="placeholder")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Tear down the VTK render thread cleanly.

        Without this, the Python interpreter exits while VTK's render
        timer / signal queue is still alive and Qt prints
        ``QThreadStorage: entry destroyed before end of thread``.
        """
        self._discard_actors()
        self._discard_offscreen_plotter()
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:  # pragma: no cover — best-effort shutdown
                pass
            self._plotter = None
            self._plotter_ready = False
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        if self._render_backend == "pyvista_offscreen":
            QTimer.singleShot(0, self._refresh_offscreen_pixmap)
