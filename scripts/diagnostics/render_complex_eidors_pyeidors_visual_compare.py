#!/usr/bin/env python3
"""Render visual parity figures for the complex EIDORS/PyEIDORS harness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.diagnostics.complex_eidors_pyeidors_step_compare import (  # noqa: E402
    ComplexCase,
    OUT_ROOT,
)

for font_path in (
    Path("/mnt/c/Windows/Fonts/times.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
    Path("/mnt/c/Windows/Fonts/timesi.ttf"),
    Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
):
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
    }
)


@dataclass(frozen=True)
class Channel:
    key: str
    label: str
    unit: str
    cmap: str
    fixed_limits: tuple[float, float] | None = None


CHANNELS = (
    Channel("real", "Real Re", "S/m", "viridis"),
    Channel("imag", "Imag Im", "S/m", "viridis"),
    Channel("abs", "Magnitude |.|", "S/m", "viridis"),
    Channel("phase", "Phase angle", "rad", "twilight_shifted", (-np.pi, np.pi)),
)


def _load_h5(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        return {key: np.asarray(h5[key]) for key in h5.keys()}


def _load_case(
    case_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    payload_path = case_dir / "payload.mat"
    eidors_path = case_dir / "eidors_result.mat"
    py_path = case_dir / "pyeidors_result.h5"
    for path in (payload_path, eidors_path, py_path):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = loadmat(payload_path)
    eidors = loadmat(eidors_path)
    pyeidors = _load_h5(py_path)
    return payload, eidors, pyeidors


def _cell_centers(payload: dict[str, np.ndarray]) -> np.ndarray:
    nodes = np.asarray(payload["nodes"], dtype=float)
    elems = np.asarray(payload["elems"], dtype=np.int64) - 1
    if elems.ndim != 2 or elems.shape[1] < 4:
        raise ValueError("payload elems must be tetrahedral cells")
    return np.mean(nodes[elems[:, :4]], axis=1)


def _as_complex_vector(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128).reshape(-1)


def _channel(values: np.ndarray, channel: Channel) -> np.ndarray:
    arr = _as_complex_vector(values)
    if channel.key == "real":
        return arr.real
    if channel.key == "imag":
        return arr.imag
    if channel.key == "abs":
        return np.abs(arr)
    if channel.key == "phase":
        return np.angle(arr)
    raise ValueError(f"unknown channel {channel.key!r}")


def _phase_diff(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.angle(
        np.exp(
            1j * (_channel(candidate, CHANNELS[-1]) - _channel(reference, CHANNELS[-1]))
        )
    )


def _difference(
    candidate: np.ndarray, reference: np.ndarray, channel: Channel
) -> np.ndarray:
    if channel.key == "phase":
        return _phase_diff(candidate, reference)
    return _channel(candidate, channel) - _channel(reference, channel)


def _limits_from_truth(truth: np.ndarray, channel: Channel) -> tuple[float, float]:
    if channel.fixed_limits is not None:
        return channel.fixed_limits
    vals = _channel(truth, channel)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if np.isclose(lo, hi):
        pad = max(abs(lo) * 0.05, 1.0e-12)
        return lo - pad, hi + pad
    return lo, hi


def _robust_symmetric_limit(values: list[np.ndarray]) -> float:
    merged = np.concatenate(
        [np.asarray(value, dtype=float).reshape(-1) for value in values]
    )
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), 99.0))
    return max(limit, float(np.nanmax(np.abs(finite))) * 1.0e-6, 1.0e-12)


def _central_slice_mask(centers: np.ndarray, half_width: float | None) -> np.ndarray:
    y_abs = np.abs(np.asarray(centers[:, 1], dtype=float))
    if half_width is None:
        half_width = max(float(np.nanpercentile(y_abs, 20.0)), 1.0e-6)
    mask = y_abs <= float(half_width)
    if np.count_nonzero(mask) < max(100, centers.shape[0] // 20):
        order = np.argsort(y_abs)
        keep = max(100, centers.shape[0] // 5)
        mask = np.zeros(centers.shape[0], dtype=bool)
        mask[order[:keep]] = True
    return mask


def _panel_stats(values: np.ndarray) -> str:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "no finite data"
    return "min={mn:.4g}\nmax={mx:.4g}".format(
        mn=float(np.min(finite)),
        mx=float(np.max(finite)),
    )


def _scatter_panel(
    ax: plt.Axes,
    x: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Any:
    artist = ax.scatter(
        x,
        z,
        c=values,
        s=14,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    ax.text(
        0.02,
        0.98,
        _panel_stats(values),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
    )
    return artist


def render_channel_grid(
    case_dir: Path,
    centers: np.ndarray,
    slice_mask: np.ndarray,
    truth: np.ndarray,
    eidors_sigma: np.ndarray,
    pyeidors_sigma: np.ndarray,
) -> Path:
    x = centers[slice_mask, 0]
    z = centers[slice_mask, 2]
    datasets = (
        ("Truth", truth),
        ("EIDORS reconstruction", eidors_sigma),
        ("PyEIDORS reconstruction", pyeidors_sigma),
    )
    fig, axes = plt.subplots(
        len(CHANNELS),
        len(datasets),
        figsize=(14, 14),
        constrained_layout=True,
    )
    for row, channel in enumerate(CHANNELS):
        vmin, vmax = _limits_from_truth(truth, channel)
        row_artists = []
        for col, (name, values) in enumerate(datasets):
            vals = _channel(values, channel)[slice_mask]
            artist = _scatter_panel(
                axes[row, col],
                x,
                z,
                vals,
                title=f"{name} - {channel.label}",
                cmap=channel.cmap,
                vmin=vmin,
                vmax=vmax,
            )
            row_artists.append(artist)
        cbar = fig.colorbar(row_artists[-1], ax=axes[row, :].tolist(), shrink=0.82)
        cbar.set_label(channel.unit)
    fig.suptitle("Complex admittance visual parity, central x-z slice", fontsize=14)
    out = case_dir / "visual_compare_channels_xz.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_difference_grid(
    case_dir: Path,
    centers: np.ndarray,
    slice_mask: np.ndarray,
    truth: np.ndarray,
    eidors_sigma: np.ndarray,
    pyeidors_sigma: np.ndarray,
) -> Path:
    x = centers[slice_mask, 0]
    z = centers[slice_mask, 2]
    columns = (
        ("EIDORS - Truth", eidors_sigma, truth),
        ("PyEIDORS - Truth", pyeidors_sigma, truth),
        ("PyEIDORS - EIDORS", pyeidors_sigma, eidors_sigma),
    )
    fig, axes = plt.subplots(
        len(CHANNELS),
        len(columns),
        figsize=(14, 14),
        constrained_layout=True,
    )
    for row, channel in enumerate(CHANNELS):
        diffs = [
            _difference(candidate, reference, channel)[slice_mask]
            for _, candidate, reference in columns
        ]
        limit = _robust_symmetric_limit(diffs)
        row_artists = []
        for col, ((name, _, _), diff) in enumerate(zip(columns, diffs, strict=True)):
            artist = _scatter_panel(
                axes[row, col],
                x,
                z,
                diff,
                title=f"{name} - {channel.label}",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            row_artists.append(artist)
        cbar = fig.colorbar(row_artists[-1], ax=axes[row, :].tolist(), shrink=0.82)
        cbar.set_label(channel.unit)
    fig.suptitle(
        "Complex admittance reconstruction differences, central x-z slice", fontsize=14
    )
    out = case_dir / "visual_compare_differences_xz.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_phase_zoom(
    case_dir: Path,
    centers: np.ndarray,
    slice_mask: np.ndarray,
    truth: np.ndarray,
    eidors_sigma: np.ndarray,
    pyeidors_sigma: np.ndarray,
) -> Path:
    x = centers[slice_mask, 0]
    z = centers[slice_mask, 2]
    truth_phase = np.angle(_as_complex_vector(truth))
    local_limits = (float(np.min(truth_phase)), float(np.max(truth_phase)))
    datasets = (
        ("Truth phase, fixed [-pi, pi]", truth, (-np.pi, np.pi)),
        ("Truth phase, local contrast", truth, local_limits),
        ("EIDORS phase, fixed [-pi, pi]", eidors_sigma, (-np.pi, np.pi)),
        ("PyEIDORS phase, fixed [-pi, pi]", pyeidors_sigma, (-np.pi, np.pi)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes_flat = axes.reshape(-1)
    artists = []
    for ax, (title, values, limits) in zip(axes_flat, datasets, strict=True):
        vals = np.angle(_as_complex_vector(values))[slice_mask]
        artists.append(
            _scatter_panel(
                ax,
                x,
                z,
                vals,
                title=title,
                cmap="twilight_shifted",
                vmin=float(limits[0]),
                vmax=float(limits[1]),
            )
        )
    for ax, artist in zip(axes_flat, artists, strict=True):
        cbar = fig.colorbar(artist, ax=ax, shrink=0.78)
        cbar.set_label("rad")
    fig.suptitle("Phase visibility: physical scale vs local contrast", fontsize=14)
    out = case_dir / "visual_compare_phase_zoom_xz.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_histograms(
    case_dir: Path,
    truth: np.ndarray,
    eidors_sigma: np.ndarray,
    pyeidors_sigma: np.ndarray,
) -> Path:
    datasets = (
        ("Truth", truth, "#222222"),
        ("EIDORS", eidors_sigma, "#1f77b4"),
        ("PyEIDORS", pyeidors_sigma, "#d62728"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, channel in zip(axes.reshape(-1), CHANNELS, strict=True):
        for name, values, color in datasets:
            vals = _channel(values, channel)
            ax.hist(
                vals[np.isfinite(vals)],
                bins=80,
                histtype="step",
                linewidth=1.4,
                label=name,
                color=color,
            )
        ax.set_title(channel.label)
        ax.set_xlabel(channel.unit)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Value distributions across all elements", fontsize=14)
    out = case_dir / "visual_compare_histograms.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def render_visual_compare(
    *,
    out_root: Path,
    case: ComplexCase,
    slice_half_width: float | None,
) -> list[Path]:
    case_dir = out_root / case.name
    payload, eidors, pyeidors = _load_case(case_dir)
    centers = _cell_centers(payload)
    slice_mask = _central_slice_mask(centers, slice_half_width)
    truth = _as_complex_vector(payload["truth_elem_data"])
    base_sigma = complex(np.asarray(payload["base_sigma"]).reshape(-1)[0])
    eidors_sigma = base_sigma + _as_complex_vector(eidors["rec_delta"])
    pyeidors_sigma = _as_complex_vector(pyeidors["rec_sigma"])
    if not (truth.size == eidors_sigma.size == pyeidors_sigma.size == centers.shape[0]):
        raise ValueError("truth/reconstruction/mesh element counts do not match")
    return [
        render_channel_grid(
            case_dir, centers, slice_mask, truth, eidors_sigma, pyeidors_sigma
        ),
        render_difference_grid(
            case_dir, centers, slice_mask, truth, eidors_sigma, pyeidors_sigma
        ),
        render_phase_zoom(
            case_dir, centers, slice_mask, truth, eidors_sigma, pyeidors_sigma
        ),
        render_histograms(case_dir, truth, eidors_sigma, pyeidors_sigma),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_ROOT,
        help="Output root used by complex_eidors_pyeidors_step_compare.py.",
    )
    parser.add_argument(
        "--slice-half-width",
        type=float,
        default=None,
        help="Central x-z slice half-width in meters. Defaults to a robust mesh-based width.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = render_visual_compare(
        out_root=Path(args.out_root).resolve(),
        case=ComplexCase(),
        slice_half_width=args.slice_half_width,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
