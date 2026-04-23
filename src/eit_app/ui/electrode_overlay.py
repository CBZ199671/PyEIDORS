"""Electrode geometry overlays for simulation conductivity widgets.

Computes the on-mesh electrode footprints from a ``ForwardModelConfig``-shaped
mapping so the 2D matplotlib widget can draw boundary arcs and the 3D PyVista
widget can draw cylindrical side patches.

Both downstream widgets cache the geometry between updates and toggle visibility
of cached actors / collections rather than rebuilding — toggling the electrode
overlay must never rebuild the conductivity scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any, Sequence

import numpy as np


__all__ = [
    "ElectrodeArcSpec",
    "ElectrodePatchSpec",
    "ElectrodeGeometry",
    "electrode_geometry_from_config",
    "default_arc_segments",
    "default_patch_quads",
]


@dataclass(frozen=True)
class ElectrodeArcSpec:
    """A 2D boundary electrode arc (used by the matplotlib widget)."""

    theta_start: float  # radians
    theta_end: float  # radians


@dataclass(frozen=True)
class ElectrodePatchSpec:
    """A 3D side-wall electrode patch (used by the PyVista widget)."""

    theta_start: float
    theta_end: float
    z_lower: float
    z_upper: float


@dataclass
class ElectrodeGeometry:
    """Resolved electrode footprints for the most recent forward configuration.

    ``mode`` is either ``"2d"`` (arcs only) or ``"3d_cylinder"`` (patches only).
    Empty / invalid configurations return None from
    :func:`electrode_geometry_from_config` rather than an empty geometry,
    so callers can use a simple truthy check before drawing.
    """

    mode: str
    radius: float
    arcs: tuple[ElectrodeArcSpec, ...] = field(default_factory=tuple)
    patches: tuple[ElectrodePatchSpec, ...] = field(default_factory=tuple)
    z_min: float = 0.0
    z_max: float = 0.0

    def is_3d(self) -> bool:
        return self.mode == "3d_cylinder"


def _arc_positions(
    n_elec: int,
    coverage: float,
    rotation: float = pi / 2.0,
    anticlockwise: bool = True,
) -> list[tuple[float, float]]:
    """Mirror of ``ElectrodeArcConfig.positions`` from pyeidors mesh3d_generator.

    Kept local to avoid pulling the heavy mesh-generator module into the
    GUI render path.  The math is intentionally identical so what the user
    sees on the canvas matches the gmsh-built electrode patches exactly.
    """
    if n_elec <= 0 or coverage <= 0.0:
        return []
    electrode_size = 2.0 * pi / n_elec * float(coverage)
    gap_size = 2.0 * pi / n_elec * (1.0 - float(coverage))
    first_center = float(rotation)
    first_start = first_center - electrode_size / 2.0
    positions: list[tuple[float, float]] = []
    for i in range(int(n_elec)):
        step = i * (electrode_size + gap_size)
        start = first_start + step
        end = start + electrode_size
        positions.append((start, end))
    if not anticlockwise and len(positions) > 1:
        positions[1:] = positions[1:][::-1]
    return positions


def _normalise_level_fractions(
    fractions: Any,
    n_rings: int,
) -> tuple[float, ...]:
    """Coerce a frontend-supplied level fractions value into a sane tuple."""
    if fractions in (None, "", ()):
        # Match electrode_level_fractions_for_rings defaults.
        from eit_app.models.forward_model_config import (
            electrode_level_fractions_for_rings,
        )

        return electrode_level_fractions_for_rings(n_rings)
    try:
        values = tuple(float(v) for v in fractions)
    except (TypeError, ValueError):
        return tuple()
    cleaned = tuple(v for v in values if 0.0 <= v <= 1.0)
    return cleaned


def electrode_geometry_from_config(config: dict | None) -> ElectrodeGeometry | None:
    """Build the overlay geometry for a single forward-solver result.

    Returns None if the config is missing or its electrode layout cannot be
    derived (e.g. zero electrodes, malformed level fractions).
    """
    if not config:
        return None

    try:
        mesh_dim = int(config.get("mesh_dimension", 2))
        n_elec_per_ring = int(config.get("n_elec", 16))
        n_rings = max(int(config.get("n_rings", 1)), 1)
        radius = float(config.get("radius", 1.0))
    except (TypeError, ValueError):
        return None

    if n_elec_per_ring <= 0 or radius <= 0.0:
        return None

    coverage = float(config.get("electrode_coverage", 0.5) or 0.5)
    coverage = float(min(max(coverage, 0.0), 1.0))

    arcs = tuple(
        ElectrodeArcSpec(theta_start=start, theta_end=end)
        for start, end in _arc_positions(n_elec_per_ring, coverage)
    )

    if mesh_dim != 3:
        return ElectrodeGeometry(
            mode="2d",
            radius=radius,
            arcs=arcs,
        )

    height = float(config.get("height", 1.0) or 0.0)
    if height <= 0.0:
        return ElectrodeGeometry(mode="2d", radius=radius, arcs=arcs)

    z_center = float(config.get("z_center", 0.0) or 0.0)
    z_min = z_center - 0.5 * height
    z_max = z_center + 0.5 * height
    height_ratio = float(config.get("electrode_height_ratio", 0.2) or 0.2)
    half_h = 0.5 * height * float(min(max(height_ratio, 0.0), 1.0))

    fractions = _normalise_level_fractions(
        config.get("electrode_level_fractions"), n_rings
    )
    if not fractions:
        return ElectrodeGeometry(mode="2d", radius=radius, arcs=arcs)

    patches: list[ElectrodePatchSpec] = []
    for frac in fractions:
        center_z = z_min + float(frac) * height
        z_lower = max(z_min, center_z - half_h)
        z_upper = min(z_max, center_z + half_h)
        if z_upper - z_lower <= 1.0e-9:
            continue
        for arc in arcs:
            patches.append(
                ElectrodePatchSpec(
                    theta_start=arc.theta_start,
                    theta_end=arc.theta_end,
                    z_lower=z_lower,
                    z_upper=z_upper,
                )
            )

    if not patches:
        return ElectrodeGeometry(mode="2d", radius=radius, arcs=arcs)

    return ElectrodeGeometry(
        mode="3d_cylinder",
        radius=radius,
        arcs=arcs,
        patches=tuple(patches),
        z_min=z_min,
        z_max=z_max,
    )


def default_arc_segments(
    arcs: Sequence[ElectrodeArcSpec],
    radius: float,
    *,
    n_samples: int = 8,
) -> list[np.ndarray]:
    """Sample each electrode arc into XY polylines for matplotlib LineCollection."""
    segments: list[np.ndarray] = []
    samples = max(int(n_samples), 2)
    for arc in arcs:
        thetas = np.linspace(arc.theta_start, arc.theta_end, samples)
        xs = radius * np.cos(thetas)
        ys = radius * np.sin(thetas)
        segments.append(np.column_stack((xs, ys)))
    return segments


def default_patch_quads(
    patches: Sequence[ElectrodePatchSpec],
    radius: float,
    *,
    n_theta: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate each cylindrical electrode patch into a triangle mesh.

    Returns (points, triangles) suitable for ``pv.PolyData`` construction:
    points are XYZ row vectors, triangles are zero-based vertex indices.
    Each patch contributes ``2 * (n_theta - 1)`` triangles.
    """
    samples = max(int(n_theta), 2)
    if not patches:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
        )

    pts: list[np.ndarray] = []
    tris: list[tuple[int, int, int]] = []
    base = 0
    for patch in patches:
        thetas = np.linspace(patch.theta_start, patch.theta_end, samples)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)
        # Lay out vertices as 2 rows × n_theta cols (lower z first, then upper z).
        lower = np.column_stack(
            (radius * cos_t, radius * sin_t, np.full_like(cos_t, patch.z_lower))
        )
        upper = np.column_stack(
            (radius * cos_t, radius * sin_t, np.full_like(cos_t, patch.z_upper))
        )
        pts.append(lower)
        pts.append(upper)
        for i in range(samples - 1):
            l0 = base + i
            l1 = base + i + 1
            u0 = base + samples + i
            u1 = base + samples + i + 1
            tris.append((l0, l1, u1))
            tris.append((l0, u1, u0))
        base += 2 * samples

    points = np.vstack(pts).astype(np.float32, copy=False)
    triangles = np.asarray(tris, dtype=np.int64)
    return points, triangles
