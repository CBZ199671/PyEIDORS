"""Point audit helpers for adjacent-measurement holdout experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from typing import Iterable

from ._sweep_core import SweepRow


POINT_AUDIT_FIELDS = [
    "stim_index",
    "stim_e1",
    "stim_e2",
    "meas_index_full",
    "meas_e1",
    "meas_e2",
    "global_index_256",
    "frame_index_16",
    "frame_index_13",
    "frame_index_10",
    "point_status",
    "voltage_reference",
    "voltage_anomaly",
    "voltage_diff",
    "fit_voltage_reference",
    "fit_voltage_anomaly",
    "fit_voltage_diff",
    "fit_residual",
]


@dataclass(frozen=True)
class HoldoutPointAuditRow(SweepRow):
    """One candidate adjacent voltage measurement point in a stimulation frame."""

    csv_fieldnames: ClassVar[tuple[str, ...]] = tuple(POINT_AUDIT_FIELDS)

    stim_index: int
    stim_e1: int
    stim_e2: int
    meas_index_full: int
    meas_e1: int
    meas_e2: int
    global_index_256: int
    frame_index_16: int
    frame_index_13: int | None
    frame_index_10: int | None
    point_status: str
    voltage_reference: float | None = None
    voltage_anomaly: float | None = None
    voltage_diff: float | None = None
    fit_voltage_reference: float | None = None
    fit_voltage_anomaly: float | None = None
    fit_voltage_diff: float | None = None
    fit_residual: float | None = None


@dataclass(frozen=True)
class HoldoutPointAuditSummary:
    """Counts for the 256 -> 208 -> 160 point audit."""

    n_elec: int
    frame_count: int
    full_candidate_count: int
    drive_removed_count: int
    kept_208_count: int
    holdout_far3_count: int
    fit_train_160_count: int
    points_per_full_frame: int
    points_per_kept_frame: int
    points_per_train_frame: int


def _validate_adjacent_holdout_args(n_elec: int, holdout: str) -> int:
    count = int(n_elec)
    if count < 8:
        raise ValueError("n_elec must be >= 8 for far3 adjacent holdout")
    if count % 2 != 0:
        raise ValueError("n_elec must be even so the far-side pair is defined")
    if str(holdout).strip().lower() != "far3":
        raise ValueError("holdout must be 'far3'")
    return count


def adjacent_pair(start_index: int, n_elec: int) -> tuple[int, int]:
    """Return adjacent electrode pair for a zero-based start electrode."""

    count = int(n_elec)
    start = int(start_index) % count
    return start, (start + 1) % count


def drive_removed_frame_indices(n_elec: int) -> set[int]:
    """Frame-local full indices removed because they touch the drive pair."""

    count = int(n_elec)
    return {0, 1, count - 1}


def far3_frame_indices(n_elec: int) -> set[int]:
    """Frame-local full indices for far-side pair plus its two neighbors."""

    count = int(n_elec)
    far = count // 2
    return {far - 1, far, far + 1}


def build_holdout_point_audit(
    *,
    n_elec: int = 16,
    holdout: str = "far3",
) -> tuple[list[HoldoutPointAuditRow], HoldoutPointAuditSummary]:
    """Build the 256 -> 208 -> 160 adjacent-measurement point audit."""

    count = _validate_adjacent_holdout_args(n_elec, holdout)
    drive_removed = drive_removed_frame_indices(count)
    far_removed = far3_frame_indices(count)
    overlap = drive_removed & far_removed
    if overlap:
        raise RuntimeError(f"drive-removed and far3 indices overlap: {sorted(overlap)}")

    kept_indices = [
        frame_idx for frame_idx in range(count) if frame_idx not in drive_removed
    ]
    train_indices = [
        frame_idx for frame_idx in kept_indices if frame_idx not in far_removed
    ]
    frame13_map = {frame_idx: idx for idx, frame_idx in enumerate(kept_indices)}
    frame10_map = {frame_idx: idx for idx, frame_idx in enumerate(train_indices)}

    rows: list[HoldoutPointAuditRow] = []
    for stim_index in range(count):
        stim_e1, stim_e2 = adjacent_pair(stim_index, count)
        for frame_index_16 in range(count):
            meas_index_full = (stim_index + frame_index_16) % count
            meas_e1, meas_e2 = adjacent_pair(meas_index_full, count)
            if frame_index_16 in drive_removed:
                point_status = "drive_removed"
            elif frame_index_16 in far_removed:
                point_status = "holdout_far3"
            else:
                point_status = "fit_train_160"

            rows.append(
                HoldoutPointAuditRow(
                    stim_index=stim_index,
                    stim_e1=stim_e1,
                    stim_e2=stim_e2,
                    meas_index_full=meas_index_full,
                    meas_e1=meas_e1,
                    meas_e2=meas_e2,
                    global_index_256=stim_index * count + frame_index_16,
                    frame_index_16=frame_index_16,
                    frame_index_13=frame13_map.get(frame_index_16),
                    frame_index_10=frame10_map.get(frame_index_16),
                    point_status=point_status,
                )
            )

    summary = summarize_holdout_point_audit(rows, n_elec=count)
    return rows, summary


def summarize_holdout_point_audit(
    rows: Iterable[HoldoutPointAuditRow],
    *,
    n_elec: int,
) -> HoldoutPointAuditSummary:
    """Summarize and assert the expected adjacent point counts."""

    row_list = list(rows)
    count = int(n_elec)
    status_counts: dict[str, int] = {}
    for row in row_list:
        status_counts[row.point_status] = status_counts.get(row.point_status, 0) + 1

    full_candidate_count = len(row_list)
    drive_removed_count = status_counts.get("drive_removed", 0)
    holdout_far3_count = status_counts.get("holdout_far3", 0)
    fit_train_160_count = status_counts.get("fit_train_160", 0)
    kept_208_count = full_candidate_count - drive_removed_count

    expected_full = count * count
    expected_drive_removed = count * 3
    expected_kept = count * (count - 3)
    expected_holdout = count * 3
    expected_train = count * (count - 6)
    if full_candidate_count != expected_full:
        raise RuntimeError(
            f"full candidate count mismatch: {full_candidate_count} != {expected_full}"
        )
    if drive_removed_count != expected_drive_removed:
        raise RuntimeError(
            "drive-removed count mismatch: "
            f"{drive_removed_count} != {expected_drive_removed}"
        )
    if kept_208_count != expected_kept:
        raise RuntimeError(f"kept count mismatch: {kept_208_count} != {expected_kept}")
    if holdout_far3_count != expected_holdout:
        raise RuntimeError(
            f"far3 holdout count mismatch: {holdout_far3_count} != {expected_holdout}"
        )
    if fit_train_160_count != expected_train:
        raise RuntimeError(
            f"fit-train count mismatch: {fit_train_160_count} != {expected_train}"
        )

    return HoldoutPointAuditSummary(
        n_elec=count,
        frame_count=count,
        full_candidate_count=full_candidate_count,
        drive_removed_count=drive_removed_count,
        kept_208_count=kept_208_count,
        holdout_far3_count=holdout_far3_count,
        fit_train_160_count=fit_train_160_count,
        points_per_full_frame=count,
        points_per_kept_frame=count - 3,
        points_per_train_frame=count - 6,
    )


def plot_holdout_point_audit(
    rows: Iterable[HoldoutPointAuditRow],
    output_path: Path,
    *,
    n_elec: int = 16,
    dpi: int = 200,
) -> Path:
    """Plot the 256 -> 208 -> 160 point-status audit."""

    row_list = list(rows)
    if not row_list:
        raise ValueError("rows must not be empty")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    def stage_status(row: HoldoutPointAuditRow, stage: str) -> str:
        if stage == "full":
            return "full_candidate"
        if stage == "kept":
            return (
                "drive_removed" if row.point_status == "drive_removed" else "kept_208"
            )
        if stage == "train":
            return row.point_status
        if row.point_status == "holdout_far3":
            return "fit_predicted"
        return row.point_status

    palette = {
        "full_candidate": "#8a8a8a",
        "drive_removed": "#d62728",
        "kept_208": "#1f77b4",
        "holdout_far3": "#ff7f0e",
        "fit_train_160": "#2ca02c",
        "fit_predicted": "#9467bd",
    }
    labels = {
        "full_candidate": "candidate 256",
        "drive_removed": "drive-related removed",
        "kept_208": "kept 208",
        "holdout_far3": "far3 holdout",
        "fit_train_160": "fit train 160",
        "fit_predicted": "fit predicted",
    }
    stages = [
        ("full", "A. Full adjacent candidates: 16 x 16 = 256"),
        ("kept", "B. Remove drive-related points: 208 kept"),
        ("train", "C. Remove far3 holdout: 160 fit-train points"),
        ("fitted", "D. Fitted 208 target: 160 train + 48 predicted"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 9.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle(
        "16-electrode adjacent point audit (256 -> 208 -> 160)",
        fontsize=15,
    )

    marker_map = {
        "full_candidate": "o",
        "drive_removed": "x",
        "kept_208": "o",
        "holdout_far3": "X",
        "fit_train_160": "o",
        "fit_predicted": "D",
    }
    for ax, (stage, title) in zip(axes.ravel(), stages, strict=True):
        statuses = sorted({stage_status(row, stage) for row in row_list})
        for status in statuses:
            status_rows = [
                row for row in row_list if stage_status(row, stage) == status
            ]
            ax.scatter(
                [row.stim_index for row in status_rows],
                [row.meas_index_full for row in status_rows],
                s=34,
                c=palette[status],
                marker=marker_map[status],
                linewidths=1.1,
                label=labels[status],
                alpha=0.9,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(int(n_elec)))
        ax.set_yticks(range(int(n_elec)))
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.legend(loc="upper right", fontsize=7.5, frameon=True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Stim start electrode s; drive pair = (s, s+1)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Meas start electrode m; pair = (m, m+1)")

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output
