"""Circle-bucket domain audit helpers for dense EIT experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from .eit_digit_metrics import ADJACENT_PATTERN, adjacent_measurement_count


BUCKET_DOMAIN_AUDIT_FIELDS = [
    "domain",
    "bucket_radius",
    "n_elec",
    "mesh_h",
    "n_cells",
    "n_nodes",
    "n_dofs",
    "electrode_index",
    "electrode_center_angle",
    "electrode_arc_length",
    "electrode_width",
    "stim_pattern",
    "meas_pattern",
    "n_measurements",
]


@dataclass(frozen=True)
class BucketElectrode:
    """One equally spaced boundary electrode arc on the circular bucket."""

    electrode_index: int
    center_angle_deg: float
    start_angle_rad: float
    end_angle_rad: float
    arc_length: float
    chord_width: float


@dataclass(frozen=True)
class BucketDomainAuditRow:
    """CSV-ready domain/electrode audit row."""

    domain: str
    bucket_radius: float
    n_elec: int
    mesh_h: float
    n_cells: int
    n_nodes: int
    n_dofs: int
    electrode_index: int
    electrode_center_angle: float
    electrode_arc_length: float
    electrode_width: float
    stim_pattern: str
    meas_pattern: str
    n_measurements: int

    def as_csv_row(self) -> dict[str, float | int | str]:
        return {
            "domain": self.domain,
            "bucket_radius": self.bucket_radius,
            "n_elec": self.n_elec,
            "mesh_h": self.mesh_h,
            "n_cells": self.n_cells,
            "n_nodes": self.n_nodes,
            "n_dofs": self.n_dofs,
            "electrode_index": self.electrode_index,
            "electrode_center_angle": self.electrode_center_angle,
            "electrode_arc_length": self.electrode_arc_length,
            "electrode_width": self.electrode_width,
            "stim_pattern": self.stim_pattern,
            "meas_pattern": self.meas_pattern,
            "n_measurements": self.n_measurements,
        }


@dataclass(frozen=True)
class CircleBucketDomain:
    """Dense circular bucket domain, electrode audit, and truth field."""

    domain: str
    bucket_radius: float
    n_elec: int
    mesh_h: float
    nodes: np.ndarray
    cells: np.ndarray
    cell_centers: np.ndarray
    cell_areas: np.ndarray
    sigma_true: np.ndarray
    electrodes: tuple[BucketElectrode, ...]
    anomaly_center: tuple[float, float]
    anomaly_radius: float
    background_conductivity: float
    anomaly_conductivity: float
    stim_pattern: str = ADJACENT_PATTERN
    meas_pattern: str = ADJACENT_PATTERN

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dofs(self) -> int:
        return self.n_cells

    @property
    def n_measurements(self) -> int:
        return adjacent_measurement_count(self.n_elec)

    @property
    def dense_threshold_passed(self) -> bool:
        return is_dense_bucket_mesh(
            mesh_h=self.mesh_h,
            bucket_radius=self.bucket_radius,
            n_cells=self.n_cells,
        )


def _as_positive_float(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _normalize_domain(domain: str) -> str:
    name = str(domain).strip().lower().replace("-", "_")
    if name != "circle_bucket":
        raise ValueError("domain must be 'circle_bucket'")
    return name


def _validate_n_elec(n_elec: int) -> int:
    count = int(n_elec)
    if count < 4:
        raise ValueError("n_elec must be >= 4")
    if count % 2 != 0:
        raise ValueError("n_elec must be even for adjacent/far-side audits")
    return count


def _circle_nodes(*, radius: float, mesh_h: float, n_elec: int) -> np.ndarray:
    boundary_count = max(
        int(n_elec) * 8,
        int(math.ceil((2.0 * math.pi * radius) / (0.75 * mesh_h))),
    )
    angles = np.linspace(0.0, 2.0 * math.pi, boundary_count, endpoint=False)
    boundary = radius * np.column_stack([np.cos(angles), np.sin(angles)])

    interior: list[tuple[float, float]] = [(0.0, 0.0)]
    dy = mesh_h * math.sqrt(3.0) / 2.0
    y_values = np.arange(-radius + dy, radius, dy, dtype=float)
    for row_idx, y_value in enumerate(y_values):
        x_offset = 0.5 * mesh_h if row_idx % 2 else 0.0
        x_values = np.arange(-radius + mesh_h + x_offset, radius, mesh_h)
        for x_value in x_values:
            if x_value * x_value + y_value * y_value <= (radius - 0.25 * mesh_h) ** 2:
                interior.append((float(x_value), float(y_value)))

    raw_points = np.vstack([boundary, np.asarray(interior, dtype=float)])
    rounded = np.round(raw_points, decimals=12)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    points = raw_points[np.sort(unique_indices)]
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii > radius + 1e-10):
        raise RuntimeError("circle bucket nodes leaked outside the disk")
    return points


def _triangle_areas(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p0 = nodes[cells[:, 0], :]
    p1 = nodes[cells[:, 1], :]
    p2 = nodes[cells[:, 2], :]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0]
    )
    return 0.5 * np.abs(cross)


def _circle_cells(
    *,
    nodes: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangulation = Delaunay(nodes)
    raw_cells = np.asarray(triangulation.simplices, dtype=np.int32)
    centers = nodes[raw_cells].mean(axis=1)
    inside = np.linalg.norm(centers, axis=1) <= radius + 1e-10
    cells = raw_cells[inside]
    centers = centers[inside]
    areas = _triangle_areas(nodes, cells)
    non_degenerate = areas > 1e-14
    cells = cells[non_degenerate]
    centers = centers[non_degenerate]
    areas = areas[non_degenerate]
    if cells.size == 0:
        raise RuntimeError("circle bucket mesh has no valid cells")
    if np.any(np.linalg.norm(centers, axis=1) > radius + 1e-10):
        raise RuntimeError("circle bucket cell centers leaked outside the disk")
    return cells, centers, areas


def build_bucket_electrodes(
    *,
    n_elec: int = 16,
    bucket_radius: float = 1.0,
    electrode_coverage: float = 0.5,
) -> tuple[BucketElectrode, ...]:
    """Build equal-angle electrode arcs on the circular bucket boundary."""

    count = _validate_n_elec(n_elec)
    radius = _as_positive_float(bucket_radius, name="bucket_radius")
    coverage = float(electrode_coverage)
    if not math.isfinite(coverage) or not 0.0 < coverage <= 1.0:
        raise ValueError("electrode_coverage must be in (0, 1]")

    step = 2.0 * math.pi / float(count)
    arc_angle = step * coverage
    arc_length = radius * arc_angle
    chord_width = 2.0 * radius * math.sin(arc_angle / 2.0)
    electrodes: list[BucketElectrode] = []
    for idx in range(count):
        center = idx * step
        electrodes.append(
            BucketElectrode(
                electrode_index=idx,
                center_angle_deg=math.degrees(center),
                start_angle_rad=center - arc_angle / 2.0,
                end_angle_rad=center + arc_angle / 2.0,
                arc_length=arc_length,
                chord_width=chord_width,
            )
        )
    return tuple(electrodes)


def _sigma_truth_for_cells(
    *,
    centers: np.ndarray,
    bucket_radius: float,
    anomaly_center: tuple[float, float],
    anomaly_radius: float,
    background_conductivity: float,
    anomaly_conductivity: float,
) -> np.ndarray:
    center = np.asarray(anomaly_center, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("anomaly_center must be two finite coordinates")
    radius = _as_positive_float(anomaly_radius, name="anomaly_radius")
    if float(np.linalg.norm(center)) + radius > bucket_radius + 1e-12:
        raise ValueError("anomaly disk must fit inside the bucket")
    sigma = np.full(centers.shape[0], float(background_conductivity), dtype=float)
    mask = np.sum((centers - center) ** 2, axis=1) <= radius**2
    if not np.any(mask):
        raise RuntimeError("anomaly did not cover any mesh cells")
    sigma[mask] = float(anomaly_conductivity)
    if not np.all(np.isfinite(sigma)):
        raise RuntimeError("sigma_true contains non-finite values")
    return sigma


def is_dense_bucket_mesh(
    *,
    mesh_h: float,
    bucket_radius: float,
    n_cells: int,
) -> bool:
    """Return the T22 density gate: enough cells or explicit small mesh size."""

    return int(n_cells) >= 800 or float(mesh_h) <= float(bucket_radius) / 20.0


def assert_dense_bucket_mesh(
    domain: CircleBucketDomain,
    *,
    allow_coarse_smoke: bool = False,
) -> None:
    """Assert the dense-bucket threshold unless explicitly running a smoke."""

    if domain.dense_threshold_passed or allow_coarse_smoke:
        return
    raise RuntimeError(
        "circle_bucket mesh is coarse: need n_cells>=800 or "
        "mesh_h<=bucket_radius/20; pass allow_coarse_smoke only for smoke audits"
    )


def build_circle_bucket_domain(
    *,
    domain: str = "circle_bucket",
    bucket_radius: float = 1.0,
    n_elec: int = 16,
    mesh_h: float = 0.05,
    electrode_coverage: float = 0.5,
    anomaly_center: tuple[float, float] = (0.35, 0.2),
    anomaly_radius: float = 0.22,
    background_conductivity: float = 1.0,
    anomaly_conductivity: float = 1.15,
    allow_coarse_smoke: bool = False,
) -> CircleBucketDomain:
    """Build a dense circular bucket mesh with electrode and truth audits."""

    domain_name = _normalize_domain(domain)
    radius = _as_positive_float(bucket_radius, name="bucket_radius")
    h_value = _as_positive_float(mesh_h, name="mesh_h")
    count = _validate_n_elec(n_elec)
    if h_value >= radius:
        raise ValueError("mesh_h must be smaller than bucket_radius")

    nodes = _circle_nodes(radius=radius, mesh_h=h_value, n_elec=count)
    cells, centers, areas = _circle_cells(nodes=nodes, radius=radius)
    electrodes = build_bucket_electrodes(
        n_elec=count,
        bucket_radius=radius,
        electrode_coverage=electrode_coverage,
    )
    sigma_true = _sigma_truth_for_cells(
        centers=centers,
        bucket_radius=radius,
        anomaly_center=anomaly_center,
        anomaly_radius=anomaly_radius,
        background_conductivity=background_conductivity,
        anomaly_conductivity=anomaly_conductivity,
    )
    bucket = CircleBucketDomain(
        domain=domain_name,
        bucket_radius=radius,
        n_elec=count,
        mesh_h=h_value,
        nodes=nodes,
        cells=cells,
        cell_centers=centers,
        cell_areas=areas,
        sigma_true=sigma_true,
        electrodes=electrodes,
        anomaly_center=(float(anomaly_center[0]), float(anomaly_center[1])),
        anomaly_radius=float(anomaly_radius),
        background_conductivity=float(background_conductivity),
        anomaly_conductivity=float(anomaly_conductivity),
    )
    assert_dense_bucket_mesh(bucket, allow_coarse_smoke=allow_coarse_smoke)
    return bucket


def build_bucket_domain_audit_rows(
    bucket: CircleBucketDomain,
) -> list[BucketDomainAuditRow]:
    """Return one CSV audit row per electrode arc."""

    rows: list[BucketDomainAuditRow] = []
    for electrode in bucket.electrodes:
        rows.append(
            BucketDomainAuditRow(
                domain=bucket.domain,
                bucket_radius=bucket.bucket_radius,
                n_elec=bucket.n_elec,
                mesh_h=bucket.mesh_h,
                n_cells=bucket.n_cells,
                n_nodes=bucket.n_nodes,
                n_dofs=bucket.n_dofs,
                electrode_index=electrode.electrode_index,
                electrode_center_angle=electrode.center_angle_deg,
                electrode_arc_length=electrode.arc_length,
                electrode_width=electrode.chord_width,
                stim_pattern=bucket.stim_pattern,
                meas_pattern=bucket.meas_pattern,
                n_measurements=bucket.n_measurements,
            )
        )
    return rows


def _draw_circle(ax, radius: float, *, color: str = "#111111", linewidth: float = 1.0):
    import matplotlib.patches as patches

    ax.add_patch(
        patches.Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            zorder=5,
        )
    )


def _draw_anomaly_outline(ax, bucket: CircleBucketDomain) -> None:
    import matplotlib.patches as patches

    ax.add_patch(
        patches.Circle(
            bucket.anomaly_center,
            bucket.anomaly_radius,
            fill=False,
            edgecolor="#d62728",
            linewidth=1.4,
            linestyle="--",
            zorder=6,
        )
    )


def _draw_electrode_arcs(ax, bucket: CircleBucketDomain) -> None:
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    radius = bucket.bucket_radius
    for electrode in bucket.electrodes:
        arc_angles = np.linspace(
            electrode.start_angle_rad,
            electrode.end_angle_rad,
            18,
            dtype=float,
        )
        color = colors[electrode.electrode_index % len(colors)]
        ax.plot(
            radius * np.cos(arc_angles),
            radius * np.sin(arc_angles),
            color=color,
            linewidth=3.2,
            solid_capstyle="round",
            zorder=7,
        )
        center_rad = math.radians(electrode.center_angle_deg)
        ax.text(
            1.09 * radius * math.cos(center_rad),
            1.09 * radius * math.sin(center_rad),
            str(electrode.electrode_index),
            ha="center",
            va="center",
            fontsize=6.5,
        )


def _set_bucket_axes(ax, bucket: CircleBucketDomain, title: str) -> None:
    radius = bucket.bucket_radius
    pad = 0.16 * radius
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-radius - pad, radius + pad)
    ax.set_ylim(-radius - pad, radius + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(False)


def plot_bucket_domain_audit(
    bucket: CircleBucketDomain,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Render mesh, electrode arcs, density, and sigma truth audit."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    triangulation = mtri.Triangulation(
        bucket.nodes[:, 0],
        bucket.nodes[:, 1],
        bucket.cells,
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.8, 9.2),
        constrained_layout=True,
    )
    fig.suptitle("T22 circle bucket domain audit", fontsize=14)

    mesh_ax = axes[0, 0]
    mesh_ax.triplot(
        triangulation,
        color="#4d4d4d",
        linewidth=0.18,
        alpha=0.55,
    )
    _draw_circle(mesh_ax, bucket.bucket_radius, linewidth=1.2)
    _draw_electrode_arcs(mesh_ax, bucket)
    _set_bucket_axes(mesh_ax, bucket, "Mesh and electrodes")

    truth_ax = axes[0, 1]
    truth_image = truth_ax.tripcolor(
        triangulation,
        facecolors=bucket.sigma_true,
        shading="flat",
        cmap="viridis",
        edgecolors="#ffffff",
        linewidth=0.08,
    )
    _draw_circle(truth_ax, bucket.bucket_radius, linewidth=1.0)
    _draw_anomaly_outline(truth_ax, bucket)
    _set_bucket_axes(truth_ax, bucket, "Sigma truth")
    fig.colorbar(truth_image, ax=truth_ax, fraction=0.046, pad=0.03)

    density_ax = axes[1, 0]
    density_image = density_ax.tripcolor(
        triangulation,
        facecolors=bucket.cell_areas,
        shading="flat",
        cmap="magma",
        edgecolors="#ffffff",
        linewidth=0.05,
    )
    _draw_circle(density_ax, bucket.bucket_radius, linewidth=1.0)
    _set_bucket_axes(density_ax, bucket, "Cell area density")
    fig.colorbar(density_image, ax=density_ax, fraction=0.046, pad=0.03)

    info_ax = axes[1, 1]
    info_ax.axis("off")
    first = bucket.electrodes[0]
    status = "dense PASS" if bucket.dense_threshold_passed else "coarse smoke"
    sigma_cells = int(
        np.count_nonzero(np.isclose(bucket.sigma_true, bucket.anomaly_conductivity))
    )
    area_min = float(np.min(bucket.cell_areas))
    area_med = float(np.median(bucket.cell_areas))
    area_max = float(np.max(bucket.cell_areas))
    lines = [
        f"domain: {bucket.domain}",
        f"radius: {bucket.bucket_radius:.6g}",
        f"mesh_h: {bucket.mesh_h:.6g}",
        f"n_nodes: {bucket.n_nodes}",
        f"n_cells/n_dofs: {bucket.n_cells}",
        f"dense gate: {status}",
        f"n_elec: {bucket.n_elec}",
        f"patterns: {bucket.stim_pattern}/{bucket.meas_pattern}",
        f"measurements: {bucket.n_measurements}",
        f"electrode arc: {first.arc_length:.6g}",
        f"electrode width: {first.chord_width:.6g}",
        "anomaly center: "
        f"({bucket.anomaly_center[0]:.4g}, {bucket.anomaly_center[1]:.4g})",
        f"anomaly radius: {bucket.anomaly_radius:.6g}",
        f"anomaly cells: {sigma_cells}",
        f"cell area min/med/max: {area_min:.3g}/{area_med:.3g}/{area_max:.3g}",
    ]
    info_ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        transform=info_ax.transAxes,
    )

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def format_bucket_domain_report(bucket: CircleBucketDomain) -> str:
    """Format a short Chinese audit report for the circular bucket domain."""

    first = bucket.electrodes[0]
    status = "PASS" if bucket.dense_threshold_passed else "COARSE_SMOKE"
    sigma_cells = int(
        np.count_nonzero(np.isclose(bucket.sigma_true, bucket.anomaly_conductivity))
    )
    return "\n".join(
        [
            "# T22 圆形小水桶域审计",
            "",
            f"- domain: `{bucket.domain}`，圆盘半径 `{bucket.bucket_radius}`。",
            f"- mesh: `mesh_h={bucket.mesh_h}`，"
            f"`n_nodes={bucket.n_nodes}`，`n_cells={bucket.n_cells}`，"
            f"`n_dofs={bucket.n_dofs}`，dense gate `{status}`。",
            f"- electrodes: `{bucket.n_elec}` 个等角边界弧段；"
            f"首电极中心角 `{first.center_angle_deg:.6g}` deg，"
            f"弧长 `{first.arc_length:.6g}`，弦宽 `{first.chord_width:.6g}`。",
            f"- patterns: `{bucket.stim_pattern}/{bucket.meas_pattern}`，"
            f"`n_measurements={bucket.n_measurements}`。",
            "- anomaly truth: center "
            f"`({bucket.anomaly_center[0]:.6g},{bucket.anomaly_center[1]:.6g})`，"
            f"radius `{bucket.anomaly_radius:.6g}`，"
            f"sigma `{bucket.background_conductivity}`→"
            f"`{bucket.anomaly_conductivity}`，覆盖 `{sigma_cells}` cells。",
            "- 可信度：该任务只审计几何/网格/电极/真值；"
            "正逆问题结论等 T23 接入后再给。",
            "",
        ]
    )


__all__ = [
    "BUCKET_DOMAIN_AUDIT_FIELDS",
    "BucketDomainAuditRow",
    "BucketElectrode",
    "CircleBucketDomain",
    "assert_dense_bucket_mesh",
    "build_bucket_domain_audit_rows",
    "build_bucket_electrodes",
    "build_circle_bucket_domain",
    "format_bucket_domain_report",
    "is_dense_bucket_mesh",
    "plot_bucket_domain_audit",
]
