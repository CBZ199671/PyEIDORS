from __future__ import annotations

import numpy as np

from pyeidors.data.structures import PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager
from pyeidors.data.holdout_point_audit import (
    build_holdout_point_audit,
    drive_removed_frame_indices,
    far3_frame_indices,
)


def test_holdout_point_audit_counts_16e_adjacent_256_to_208_to_160() -> None:
    rows, summary = build_holdout_point_audit(n_elec=16, holdout="far3")

    assert len(rows) == 256
    assert summary.full_candidate_count == 256
    assert summary.drive_removed_count == 48
    assert summary.kept_208_count == 208
    assert summary.holdout_far3_count == 48
    assert summary.fit_train_160_count == 160
    assert summary.points_per_full_frame == 16
    assert summary.points_per_kept_frame == 13
    assert summary.points_per_train_frame == 10


def test_holdout_point_audit_first_frame_removed_pairs_are_expected() -> None:
    rows, _ = build_holdout_point_audit(n_elec=16, holdout="far3")
    frame0 = [row for row in rows if row.stim_index == 0]

    drive_pairs = {
        (row.meas_e1, row.meas_e2)
        for row in frame0
        if row.point_status == "drive_removed"
    }
    far3_pairs = {
        (row.meas_e1, row.meas_e2)
        for row in frame0
        if row.point_status == "holdout_far3"
    }
    train_pairs = {
        (row.meas_e1, row.meas_e2)
        for row in frame0
        if row.point_status == "fit_train_160"
    }

    assert drive_pairs == {(15, 0), (0, 1), (1, 2)}
    assert far3_pairs == {(7, 8), (8, 9), (9, 10)}
    assert len(train_pairs) == 10
    assert not drive_pairs & far3_pairs
    assert not drive_pairs & train_pairs
    assert not far3_pairs & train_pairs


def test_holdout_point_audit_frame_indices_match_far3_rule() -> None:
    rows, _ = build_holdout_point_audit(n_elec=16, holdout="far3")
    frame0 = [row for row in rows if row.stim_index == 0]

    assert drive_removed_frame_indices(16) == {15, 0, 1}
    assert far3_frame_indices(16) == {7, 8, 9}
    assert {
        row.frame_index_13 for row in frame0 if row.point_status == "holdout_far3"
    } == {5, 6, 7}
    assert {
        row.frame_index_10 for row in frame0 if row.point_status == "fit_train_160"
    } == set(range(10))


def test_v36_holdout_frames_match_rotated_pattern_manager_order() -> None:
    rows, _ = build_holdout_point_audit(n_elec=16, holdout="far3")
    manager = StimMeasPatternManager(
        PatternConfig(
            n_elec=16,
            stim_pattern="{ad}",
            meas_pattern="{ad}",
            drive_mode="total_current",
            drive_value=1.0,
        )
    )

    for stim_index in range(16):
        pattern_pairs = []
        for meas_row in manager.meas_matrices[stim_index]:
            pos = int(np.flatnonzero(meas_row > 0)[0])
            neg = int(np.flatnonzero(meas_row < 0)[0])
            pattern_pairs.append((pos, neg))

        audit_rows = [
            row
            for row in rows
            if row.stim_index == stim_index and row.point_status != "drive_removed"
        ]
        audit_pairs = [(row.meas_e1, row.meas_e2) for row in audit_rows]

        assert audit_pairs == pattern_pairs
        assert audit_pairs[0] == ((stim_index + 2) % 16, (stim_index + 3) % 16)
        assert audit_pairs[-1] == ((stim_index + 14) % 16, (stim_index + 15) % 16)
        assert [row.frame_index_13 for row in audit_rows] == list(range(13))
