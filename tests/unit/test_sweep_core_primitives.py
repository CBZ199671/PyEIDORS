"""T81 phase 2: shared sweep/report primitive contracts."""

from __future__ import annotations

from dataclasses import dataclass

from pyeidors.data._sweep_core import (
    dataclass_csv_row,
    format_aligned_table,
    write_csv_rows,
)
from pyeidors.data.bucket_dense_experiments import (
    BUCKET_DENSE_FIELD_FIELDS,
    BUCKET_DENSE_SUMMARY_FIELDS,
    BucketDenseFieldRow,
    BucketDenseSummaryRow,
)
from pyeidors.data.factor_sweep import CSV_FIELDS, FactorSweepRow
from pyeidors.data.voltage_digit_sweep import (
    VoltageDigitFieldRow,
    VoltageDigitSweepSummary,
)


@dataclass(frozen=True)
class _TinyRow:
    name: str
    maybe: float | None
    enabled: bool

    def as_csv_row(self) -> dict[str, str | float]:
        return dataclass_csv_row(self)


def test_dataclass_csv_row_preserves_order_and_csv_coercions() -> None:
    row = _TinyRow("case-a", None, True)

    assert dataclass_csv_row(row) == {
        "name": "case-a",
        "maybe": "",
        "enabled": "true",
    }
    assert list(dataclass_csv_row(row)) == ["name", "maybe", "enabled"]


def test_consolidated_sweep_rows_keep_historical_csv_field_order() -> None:
    voltage_summary = VoltageDigitSweepSummary(
        target_voltage_digits=5,
        achieved_voltage_effective_digits=4.5,
        voltage_rmse=0.1,
        sigma_rmse=0.2,
        sigma_relative_rmse=0.3,
        sigma_mae=0.4,
        sigma_max_abs_error=0.5,
        sigma_effective_digits=6.0,
    )
    voltage_field = VoltageDigitFieldRow(
        target_voltage_digits=5,
        cell_index=2,
        sigma_true=1.0,
        sigma_recon=1.2,
        sigma_error=0.2,
        abs_sigma_error=0.2,
    )
    factor = FactorSweepRow(
        sweep="baseline",
        changed_factor="baseline",
        level="baseline",
        n_elec=16,
        fem_grid=4,
        ridge=0.01,
        target_voltage_digits=6,
        enob="nominal",
        noise_relative=0.0,
        noser_exponent=0.5,
        n_measurements=10,
        voltage_rmse=0.01,
        achieved_voltage_effective_digits=7.0,
        sigma_rmse=0.02,
        sigma_relative_rmse=0.03,
        sigma_mae=0.04,
        sigma_max_abs_error=0.05,
        sigma_effective_digits=8.0,
    )

    assert list(voltage_summary.as_csv_row()) == [
        "target_voltage_digits",
        "achieved_voltage_effective_digits",
        "voltage_rmse",
        "sigma_rmse",
        "sigma_relative_rmse",
        "sigma_mae",
        "sigma_max_abs_error",
        "sigma_effective_digits",
    ]
    assert list(voltage_field.as_csv_row()) == [
        "target_voltage_digits",
        "cell_index",
        "sigma_true",
        "sigma_recon",
        "sigma_error",
        "abs_sigma_error",
    ]
    assert list(factor.as_csv_row()) == CSV_FIELDS


def test_bucket_rows_keep_none_and_bool_csv_contracts() -> None:
    summary = BucketDenseSummaryRow(
        experiment="voltage_digit_sweep",
        domain="circle",
        mesh_h=0.16,
        n_cells=12,
        n_dofs=34,
        n_elec=16,
        n_measurements=208,
        ridge=0.01,
        recon_method="digits_5",
        target_voltage_digits=None,
        holdout_voltage_rmse=None,
        diff_voltage_rmse=None,
        sigma_rmse=0.1,
        sigma_relative_rmse=0.2,
        sigma_mae=0.3,
        sigma_max_abs_error=0.4,
        sigma_effective_digits=5.0,
        centroid_error=0.6,
        eccentricity=0.7,
        artifact_area=0.8,
        artifact_energy=0.9,
        artifact_peak=1.0,
    )
    field = BucketDenseFieldRow(
        experiment="holdout_far3",
        recon_method="raw_160",
        cell_index=3,
        cell_x=0.1,
        cell_y=-0.2,
        sigma_true=1.0,
        sigma_recon=0.9,
        sigma_error=-0.1,
        inside_bucket=True,
    )

    assert list(summary.as_csv_row()) == BUCKET_DENSE_SUMMARY_FIELDS
    assert summary.as_csv_row()["target_voltage_digits"] == ""
    assert summary.as_csv_row()["holdout_voltage_rmse"] == ""
    assert list(field.as_csv_row()) == BUCKET_DENSE_FIELD_FIELDS
    assert field.as_csv_row()["inside_bucket"] == "true"


def test_write_csv_rows_and_aligned_table_are_stable(tmp_path) -> None:
    rows = [
        _TinyRow("left", 1.25, True),
        _TinyRow("right", None, False),
    ]

    output = write_csv_rows(tmp_path / "rows.csv", rows, ["name", "maybe", "enabled"])

    assert output.read_text(encoding="utf-8").splitlines() == [
        "name,maybe,enabled",
        "left,1.25,true",
        "right,,false",
    ]
    assert (
        format_aligned_table(["a", "bb"], [["1", "xx"], ["100", "y"]], limit=1)
        == "a | bb\n--+---\n1 | xx\n... 1 more rows"
    )
