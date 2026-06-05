#!/usr/bin/env python3
"""Compare two-layer EIT drive/measurement protocols with a 3D linear model.

The model is a fast protocol-screening simulation. It uses homogeneous 3D
point-electrode lead fields and the reciprocity sensitivity
dot(E_drive, E_meas), then reconstructs the same phantom for every protocol
with a shared Tikhonov + graph-smoothness inverse.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from pyeidors.runtime_paths import pyeidors_output_path


EPS = 1.0e-12


@dataclass(frozen=True)
class Pair:
    positive: int
    negative: int
    family: str


@dataclass(frozen=True)
class Protocol:
    key: str
    label: str
    stim_pairs: tuple[Pair, ...]
    meas_pairs: tuple[Pair, ...]


@dataclass(frozen=True)
class Grid:
    coords: np.ndarray
    volume: float
    shape: tuple[int, int, int]
    ijk: np.ndarray
    x_values: np.ndarray
    y_values: np.ndarray
    z_values: np.ndarray


@dataclass(frozen=True)
class ReconstructionResult:
    protocol: Protocol
    channel_count: int
    best_alpha: float
    best_lambda: float
    scaled_reconstruction: np.ndarray
    raw_reconstruction: np.ndarray
    scale: float
    corr: float
    nrmse: float
    dice: float
    cg_info: int


def wrap(idx: int, n: int = 16) -> int:
    return int(idx) % int(n)


def top(idx: int) -> int:
    return wrap(idx)


def bottom(idx: int) -> int:
    return 16 + wrap(idx)


def same_layer_pairs(skip: int, *, unique_opposite: bool = False) -> tuple[Pair, ...]:
    pairs: list[Pair] = []
    limit = 8 if unique_opposite else 16
    for offset, name in ((top, "top"), (bottom, "bottom")):
        for k in range(limit):
            pairs.append(
                Pair(
                    offset(k),
                    offset(k + skip),
                    f"{name}_skip{skip}",
                )
            )
    return tuple(pairs)


def vertical_pairs() -> tuple[Pair, ...]:
    return tuple(Pair(top(k), bottom(k), "vertical") for k in range(16))


def diagonal_pairs(shift: int, *, family_prefix: str = "diag") -> tuple[Pair, ...]:
    sign = "p" if shift >= 0 else "m"
    return tuple(
        Pair(top(k), bottom(k + shift), f"{family_prefix}_{sign}{abs(shift)}")
        for k in range(16)
    )


def same_adjacent_measurements() -> tuple[Pair, ...]:
    return same_layer_pairs(1)


def hybrid_measurements() -> tuple[Pair, ...]:
    return (
        same_adjacent_measurements()
        + vertical_pairs()
        + diagonal_pairs(1, family_prefix="meas_diag")
        + diagonal_pairs(-1, family_prefix="meas_diag")
    )


def build_protocols() -> list[Protocol]:
    layer_adj = Protocol(
        key="A_layer_adjacent",
        label="A layer-adjacent",
        stim_pairs=same_layer_pairs(1),
        meas_pairs=same_adjacent_measurements(),
    )
    layer_skip4 = Protocol(
        key="B_layer_skip4",
        label="B layer skip-4",
        stim_pairs=same_layer_pairs(4),
        meas_pairs=same_adjacent_measurements(),
    )
    layer_opposite = Protocol(
        key="C_layer_opposite",
        label="C layer opposite",
        stim_pairs=same_layer_pairs(8, unique_opposite=True),
        meas_pairs=same_adjacent_measurements(),
    )
    adjacent_3d = Protocol(
        key="D_adjacent_3d",
        label="D adjacent 3D",
        stim_pairs=same_layer_pairs(1) + vertical_pairs(),
        meas_pairs=hybrid_measurements(),
    )
    mixed_highres = Protocol(
        key="E_mixed_highres",
        label="E mixed high-res",
        stim_pairs=(
            same_layer_pairs(4)
            + same_layer_pairs(8, unique_opposite=True)
            + vertical_pairs()
            + diagonal_pairs(4, family_prefix="stim_diag")
            + diagonal_pairs(-4, family_prefix="stim_diag")
        ),
        meas_pairs=hybrid_measurements(),
    )
    return [layer_adj, layer_skip4, layer_opposite, adjacent_3d, mixed_highres]


def electrode_positions(
    *, radius: float = 1.0, z_top: float = 0.35, z_bottom: float = -0.35
) -> np.ndarray:
    out = np.empty((32, 3), dtype=np.float64)
    for layer, z in ((0, z_top), (1, z_bottom)):
        start = 16 * layer
        for k in range(16):
            theta = 2.0 * np.pi * k / 16.0
            out[start + k, :] = (radius * np.cos(theta), radius * np.sin(theta), z)
    return out


def build_grid(
    *,
    xy_count: int = 25,
    z_count: int = 13,
    radius: float = 0.9,
    z_extent: float = 0.45,
) -> Grid:
    x_values = np.linspace(-radius, radius, int(xy_count), dtype=np.float64)
    y_values = np.linspace(-radius, radius, int(xy_count), dtype=np.float64)
    z_values = np.linspace(-z_extent, z_extent, int(z_count), dtype=np.float64)
    dx = float(x_values[1] - x_values[0]) if x_values.size > 1 else 1.0
    dy = float(y_values[1] - y_values[0]) if y_values.size > 1 else 1.0
    dz = float(z_values[1] - z_values[0]) if z_values.size > 1 else 1.0
    coords: list[tuple[float, float, float]] = []
    ijk: list[tuple[int, int, int]] = []
    for iz, z in enumerate(z_values):
        for iy, y in enumerate(y_values):
            for ix, x in enumerate(x_values):
                if x * x + y * y <= radius * radius + 1.0e-12:
                    coords.append((float(x), float(y), float(z)))
                    ijk.append((ix, iy, iz))
    return Grid(
        coords=np.asarray(coords, dtype=np.float64),
        volume=dx * dy * dz,
        shape=(int(xy_count), int(xy_count), int(z_count)),
        ijk=np.asarray(ijk, dtype=np.int32),
        x_values=x_values,
        y_values=y_values,
        z_values=z_values,
    )


def make_truth(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    coords = grid.coords
    phantom = np.zeros(coords.shape[0], dtype=np.float64)
    centers = np.asarray(
        [
            (0.34, 0.20, 0.22, 0.18, 0.75),
            (-0.30, -0.28, -0.22, 0.17, 0.60),
        ],
        dtype=np.float64,
    )
    support = np.zeros(coords.shape[0], dtype=bool)
    for cx, cy, cz, radius, amplitude in centers:
        dist2 = (
            (coords[:, 0] - cx) ** 2
            + (coords[:, 1] - cy) ** 2
            + (coords[:, 2] - cz) ** 2
        )
        inside = dist2 <= radius * radius
        phantom[inside] += amplitude
        support |= inside
    return phantom, support


def field_for_pair(
    pair: Pair,
    coords: np.ndarray,
    electrodes: np.ndarray,
) -> np.ndarray:
    pos = electrodes[pair.positive]
    neg = electrodes[pair.negative]
    r_pos = coords - pos.reshape(1, 3)
    r_neg = coords - neg.reshape(1, 3)
    d_pos = np.maximum(np.linalg.norm(r_pos, axis=1), 1.0e-6)
    d_neg = np.maximum(np.linalg.norm(r_neg, axis=1), 1.0e-6)
    return (r_pos / d_pos[:, None] ** 3 - r_neg / d_neg[:, None] ** 3) / (4.0 * np.pi)


def potential_at(point: np.ndarray, pair: Pair, electrodes: np.ndarray) -> float:
    pos = electrodes[pair.positive]
    neg = electrodes[pair.negative]
    d_pos = max(float(np.linalg.norm(point - pos)), 1.0e-6)
    d_neg = max(float(np.linalg.norm(point - neg)), 1.0e-6)
    return (1.0 / d_pos - 1.0 / d_neg) / (4.0 * np.pi)


def baseline_voltage(stim: Pair, meas: Pair, electrodes: np.ndarray) -> float:
    p = electrodes[meas.positive]
    n = electrodes[meas.negative]
    return potential_at(p, stim, electrodes) - potential_at(n, stim, electrodes)


def shares_electrode(left: Pair, right: Pair) -> bool:
    return bool(
        {
            int(left.positive),
            int(left.negative),
        }
        & {
            int(right.positive),
            int(right.negative),
        }
    )


def enumerate_channels(
    protocol: Protocol, electrodes: np.ndarray
) -> list[tuple[int, int, float]]:
    channels: list[tuple[int, int, float]] = []
    baselines: list[float] = []
    for stim_idx, stim in enumerate(protocol.stim_pairs):
        for meas_idx, meas in enumerate(protocol.meas_pairs):
            if shares_electrode(stim, meas):
                continue
            v0 = baseline_voltage(stim, meas, electrodes)
            baselines.append(abs(v0))
            channels.append((stim_idx, meas_idx, v0))
    if not channels:
        raise RuntimeError(f"Protocol {protocol.key} has no valid channels.")
    baseline_floor = 0.05 * float(np.median(np.asarray(baselines, dtype=np.float64)))
    baseline_floor = max(baseline_floor, 1.0e-6)
    normalized_channels: list[tuple[int, int, float]] = []
    for stim_idx, meas_idx, v0 in channels:
        sign = 1.0 if float(v0) >= 0.0 else -1.0
        normalized_channels.append(
            (stim_idx, meas_idx, sign * max(abs(v0), baseline_floor))
        )
    return normalized_channels


def build_jacobian(
    protocol: Protocol,
    grid: Grid,
    electrodes: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    channels = enumerate_channels(protocol, electrodes)
    fields: dict[tuple[int, int], np.ndarray] = {}

    def get_field(pair: Pair) -> np.ndarray:
        key = (int(pair.positive), int(pair.negative))
        if key not in fields:
            fields[key] = field_for_pair(pair, grid.coords, electrodes)
        return fields[key]

    jac = np.empty((len(channels), grid.coords.shape[0]), dtype=np.float32)
    for row_idx, (stim_idx, meas_idx, normalized_v0) in enumerate(channels):
        stim = protocol.stim_pairs[stim_idx]
        meas = protocol.meas_pairs[meas_idx]
        drive_field = get_field(stim)
        meas_field = get_field(meas)
        sensitivity = -np.einsum("ij,ij->i", drive_field, meas_field)
        jac[row_idx, :] = (sensitivity * grid.volume / float(normalized_v0)).astype(
            np.float32
        )
    return jac, channels


def build_graph_difference(grid: Grid) -> sparse.csr_matrix:
    index_by_ijk = {tuple(map(int, ijk)): idx for idx, ijk in enumerate(grid.ijk)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row = 0
    for idx, (ix, iy, iz) in enumerate(grid.ijk):
        for delta in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            neighbor = (
                int(ix + delta[0]),
                int(iy + delta[1]),
                int(iz + delta[2]),
            )
            other = index_by_ijk.get(neighbor)
            if other is None:
                continue
            rows.extend((row, row))
            cols.extend((idx, int(other)))
            data.extend((-1.0, 1.0))
            row += 1
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(row, grid.coords.shape[0]),
        dtype=np.float64,
    )


def solve_with_alpha(
    jac: np.ndarray,
    y: np.ndarray,
    diff: sparse.csr_matrix,
    *,
    alpha: float,
    smooth_weight: float,
    maxiter: int,
) -> tuple[np.ndarray, float, int]:
    n_param = jac.shape[1]
    diag_scale = float(np.mean(np.sum(jac * jac, axis=0, dtype=np.float64)))
    lambda_value = float(alpha) * float(np.sqrt(max(diag_scale, EPS)))
    rhs = np.asarray(jac.T @ y, dtype=np.float64)
    lam2 = lambda_value * lambda_value

    def matvec(values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        data_term = np.asarray(jac.T @ (jac @ x), dtype=np.float64)
        smooth = diff.T @ (diff @ x)
        return data_term + lam2 * (x + float(smooth_weight) * smooth)

    operator = spla.LinearOperator(
        shape=(n_param, n_param),
        matvec=matvec,
        dtype=np.float64,
    )
    reconstruction, info = spla.cg(
        operator,
        rhs,
        rtol=1.0e-4,
        atol=0.0,
        maxiter=int(maxiter),
    )
    return np.asarray(reconstruction, dtype=np.float64), lambda_value, int(info)


def reconstruction_metrics(
    reconstruction: np.ndarray,
    truth: np.ndarray,
    truth_support: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    denom = float(np.dot(reconstruction, reconstruction))
    scale = float(np.dot(truth, reconstruction) / max(denom, EPS))
    scaled = scale * reconstruction
    if np.std(scaled) <= EPS or np.std(truth) <= EPS:
        corr = 0.0
    else:
        corr = float(np.corrcoef(scaled, truth)[0, 1])
    nrmse = float(np.linalg.norm(scaled - truth) / max(np.linalg.norm(truth), EPS))
    positive = np.maximum(scaled, 0.0)
    count = int(np.count_nonzero(truth_support))
    if count <= 0 or not np.any(positive > 0.0):
        dice = 0.0
    else:
        threshold = np.partition(positive, -count)[-count]
        recon_support = positive >= threshold
        overlap = int(np.count_nonzero(recon_support & truth_support))
        dice = float(2.0 * overlap / (np.count_nonzero(recon_support) + count))
    return scaled, scale, corr, nrmse, dice


def reconstruct_protocol(
    protocol: Protocol,
    grid: Grid,
    electrodes: np.ndarray,
    truth: np.ndarray,
    truth_support: np.ndarray,
    diff: sparse.csr_matrix,
    rng: np.random.Generator,
    *,
    noise_std: float,
    alphas: tuple[float, ...],
    smooth_weight: float,
    maxiter: int,
) -> ReconstructionResult:
    jac, channels = build_jacobian(protocol, grid, electrodes)
    clean = np.asarray(jac @ truth, dtype=np.float64)
    noise = rng.normal(0.0, float(noise_std), size=clean.shape)
    measured = clean + noise
    best: ReconstructionResult | None = None
    for alpha in alphas:
        raw, lambda_value, cg_info = solve_with_alpha(
            jac,
            measured,
            diff,
            alpha=float(alpha),
            smooth_weight=float(smooth_weight),
            maxiter=int(maxiter),
        )
        scaled, scale, corr, nrmse, dice = reconstruction_metrics(
            raw,
            truth,
            truth_support,
        )
        result = ReconstructionResult(
            protocol=protocol,
            channel_count=len(channels),
            best_alpha=float(alpha),
            best_lambda=float(lambda_value),
            scaled_reconstruction=scaled,
            raw_reconstruction=raw,
            scale=float(scale),
            corr=float(corr),
            nrmse=float(nrmse),
            dice=float(dice),
            cg_info=int(cg_info),
        )
        if best is None or (
            result.nrmse,
            -result.corr,
            -result.dice,
        ) < (best.nrmse, -best.corr, -best.dice):
            best = result
    if best is None:
        raise RuntimeError(f"Failed to reconstruct protocol {protocol.key}.")
    return best


def to_volume(values: np.ndarray, grid: Grid) -> np.ndarray:
    volume = np.full(grid.shape, np.nan, dtype=np.float64)
    for value, (ix, iy, iz) in zip(values, grid.ijk):
        volume[int(ix), int(iy), int(iz)] = float(value)
    return volume


def closest_z_index(grid: Grid, target: float) -> int:
    return int(np.argmin(np.abs(grid.z_values - float(target))))


def save_comparison_figure(
    output_path: Path,
    grid: Grid,
    truth: np.ndarray,
    results: list[ReconstructionResult],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
        }
    )
    truth_vol = to_volume(truth, grid)
    result_volumes = [to_volume(item.scaled_reconstruction, grid) for item in results]
    vmax = float(np.nanmax(np.abs(truth_vol)))
    for volume in result_volumes:
        vmax = max(vmax, float(np.nanpercentile(np.abs(volume), 99.5)))
    vmax = max(vmax, 1.0e-6)
    slice_targets = (0.225, 0.0, -0.225)
    slice_titles = ("Upper slice", "Middle slice", "Lower slice")
    n_cols = 1 + len(results)
    fig, axes = plt.subplots(
        len(slice_targets),
        n_cols,
        figsize=(2.8 * n_cols, 8.0),
        constrained_layout=True,
    )
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    images = []
    for row_idx, (target_z, slice_title) in enumerate(zip(slice_targets, slice_titles)):
        z_idx = closest_z_index(grid, target_z)
        panels = [truth_vol[:, :, z_idx].T] + [
            volume[:, :, z_idx].T for volume in result_volumes
        ]
        titles = ["Truth"] + [
            (
                f"{item.protocol.label}\n"
                f"r={item.corr:.2f}, nRMSE={item.nrmse:.2f}, "
                f"Dice={item.dice:.2f}"
            )
            for item in results
        ]
        for col_idx, (panel, title) in enumerate(zip(panels, titles)):
            ax = axes[row_idx, col_idx]
            image = ax.imshow(
                panel,
                origin="lower",
                extent=(
                    float(grid.x_values[0]),
                    float(grid.x_values[-1]),
                    float(grid.y_values[0]),
                    float(grid.y_values[-1]),
                ),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            images.append(image)
            if row_idx == 0:
                ax.set_title(title, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{slice_title}\nz={grid.z_values[z_idx]:.3f}")
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    fig.colorbar(images[-1], ax=axes.ravel().tolist(), shrink=0.76, label="Delta sigma")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_score_table(output_path: Path, results: list[ReconstructionResult]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "protocol_key",
                "protocol_label",
                "n_stim",
                "n_base_measurements",
                "n_valid_channels",
                "best_alpha",
                "best_lambda",
                "corr",
                "nrmse",
                "dice",
                "cg_info",
            ],
        )
        writer.writeheader()
        ranked = sorted(results, key=lambda item: (item.nrmse, -item.corr, -item.dice))
        for rank, item in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "protocol_key": item.protocol.key,
                    "protocol_label": item.protocol.label,
                    "n_stim": len(item.protocol.stim_pairs),
                    "n_base_measurements": len(item.protocol.meas_pairs),
                    "n_valid_channels": item.channel_count,
                    "best_alpha": f"{item.best_alpha:.8g}",
                    "best_lambda": f"{item.best_lambda:.8g}",
                    "corr": f"{item.corr:.8g}",
                    "nrmse": f"{item.nrmse:.8g}",
                    "dice": f"{item.dice:.8g}",
                    "cg_info": item.cg_info,
                }
            )


def save_protocol_numbering(
    output_dir: Path,
    protocol: Protocol,
    electrodes: np.ndarray,
) -> None:
    stim_path = output_dir / "best_protocol_stimulation.csv"
    meas_path = output_dir / "best_protocol_measurement_base.csv"
    channel_path = output_dir / "best_protocol_valid_channels.csv"
    with stim_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stim_id",
                "positive_electrode",
                "negative_electrode",
                "family",
            ],
        )
        writer.writeheader()
        for idx, pair in enumerate(protocol.stim_pairs, start=1):
            writer.writerow(
                {
                    "stim_id": f"E{idx:03d}",
                    "positive_electrode": pair.positive + 1,
                    "negative_electrode": pair.negative + 1,
                    "family": pair.family,
                }
            )
    with meas_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "meas_id",
                "positive_electrode",
                "negative_electrode",
                "family",
            ],
        )
        writer.writeheader()
        for idx, pair in enumerate(protocol.meas_pairs, start=1):
            writer.writerow(
                {
                    "meas_id": f"M{idx:03d}",
                    "positive_electrode": pair.positive + 1,
                    "negative_electrode": pair.negative + 1,
                    "family": pair.family,
                }
            )
    channels = enumerate_channels(protocol, electrodes)
    with channel_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "channel_id",
                "stim_id",
                "meas_id",
                "stim_positive",
                "stim_negative",
                "meas_positive",
                "meas_negative",
            ],
        )
        writer.writeheader()
        for idx, (stim_idx, meas_idx, _v0) in enumerate(channels, start=1):
            stim = protocol.stim_pairs[stim_idx]
            meas = protocol.meas_pairs[meas_idx]
            writer.writerow(
                {
                    "channel_id": f"C{idx:05d}",
                    "stim_id": f"E{stim_idx + 1:03d}",
                    "meas_id": f"M{meas_idx + 1:03d}",
                    "stim_positive": stim.positive + 1,
                    "stim_negative": stim.negative + 1,
                    "meas_positive": meas.positive + 1,
                    "meas_negative": meas.negative + 1,
                }
            )


def save_summary(
    output_path: Path,
    args: argparse.Namespace,
    grid: Grid,
    results: list[ReconstructionResult],
) -> None:
    ranked = sorted(results, key=lambda item: (item.nrmse, -item.corr, -item.dice))
    payload = {
        "model": {
            "type": "homogeneous_3d_point_electrode_linearized_eit",
            "xy_count": int(args.xy_count),
            "z_count": int(args.z_count),
            "n_voxels": int(grid.coords.shape[0]),
            "noise_std_relative_voltage": float(args.noise_std),
            "smooth_weight": float(args.smooth_weight),
            "alphas": [float(v) for v in parse_alphas(str(args.alphas))],
        },
        "ranking": [
            {
                "rank": idx,
                "protocol_key": item.protocol.key,
                "protocol_label": item.protocol.label,
                "n_stim": len(item.protocol.stim_pairs),
                "n_base_measurements": len(item.protocol.meas_pairs),
                "n_valid_channels": item.channel_count,
                "corr": item.corr,
                "nrmse": item.nrmse,
                "dice": item.dice,
                "best_alpha": item.best_alpha,
                "best_lambda": item.best_lambda,
            }
            for idx, item in enumerate(ranked, start=1)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_alphas(raw: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in str(raw).split(",") if item.strip())
    if not values:
        raise ValueError("At least one alpha value is required.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 2-layer 16-electrode EIT protocols by reconstruction."
    )
    parser.add_argument(
        "--output-dir",
        default=str(pyeidors_output_path("two_layer_protocol_comparison")),
        help="Directory for figures, tables, and numbering CSV files.",
    )
    parser.add_argument("--xy-count", type=int, default=23)
    parser.add_argument("--z-count", type=int, default=11)
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument(
        "--alphas",
        default="0.04,0.08,0.16,0.32",
        help="Comma-separated dimensionless regularization multipliers.",
    )
    parser.add_argument("--smooth-weight", type=float, default=0.25)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    electrodes = electrode_positions()
    grid = build_grid(xy_count=args.xy_count, z_count=args.z_count)
    truth, truth_support = make_truth(grid)
    diff = build_graph_difference(grid)
    protocols = build_protocols()
    rng = np.random.default_rng(int(args.seed))
    alphas = parse_alphas(str(args.alphas))
    results: list[ReconstructionResult] = []
    for protocol in protocols:
        print(
            f"[simulate] {protocol.key}: "
            f"{len(protocol.stim_pairs)} stim x {len(protocol.meas_pairs)} base meas",
            flush=True,
        )
        result = reconstruct_protocol(
            protocol,
            grid,
            electrodes,
            truth,
            truth_support,
            diff,
            rng,
            noise_std=float(args.noise_std),
            alphas=alphas,
            smooth_weight=float(args.smooth_weight),
            maxiter=int(args.maxiter),
        )
        print(
            f"[result] {protocol.key}: channels={result.channel_count}, "
            f"corr={result.corr:.3f}, nRMSE={result.nrmse:.3f}, "
            f"Dice={result.dice:.3f}, alpha={result.best_alpha:g}",
            flush=True,
        )
        results.append(result)

    ranked = sorted(results, key=lambda item: (item.nrmse, -item.corr, -item.dice))
    best = ranked[0]
    save_comparison_figure(
        output_dir / "protocol_reconstruction_comparison.png",
        grid,
        truth,
        results,
    )
    save_score_table(output_dir / "protocol_scores.csv", results)
    save_protocol_numbering(output_dir, best.protocol, electrodes)
    save_summary(output_dir / "summary.json", args, grid, results)
    print(f"[best] {best.protocol.key}: {best.protocol.label}", flush=True)
    print(f"[output] {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
