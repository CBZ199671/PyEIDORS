# T81 phase 1 — `src/pyeidors/data/` sweep & audit module index

Status: **phase 2d landed**. Phase 1 was index-only; phase 2a added
small shared sweep/report primitives; phase 2b added byte-stable CSV golden
fixtures; phase 2c adds a tiny HDF5 sheet-name fixture gate for those migrated
serialization edges; phase 2d starts the heavier schema-base consolidation
while keeping T81 open. Remaining row-schema work is still deliberately scoped
— see the rationale at the end of this document.

This audit is the entrance gate the SPEC row for T81 mandates: index
each module's dataclass schema, map overlapping fields across modules,
and propose a per-file disposition before any source code is moved into
a shared `_sweep_core` module. The schemas are paper / report driven
and have drifted on a per-experiment basis; mechanically lifting them
into a common base class without first writing this map can create a
maintenance burden bigger than the duplication it is supposed to remove
(see also the cautionary note in `SPEC.md` §T.T81).

The 10 modules in scope (≈ 7 040 lines):

| module | LOC | dataclasses |
|---|---:|---|
| `bucket_dense_experiments.py` | 1792 | `BucketDenseSummaryRow`, `BucketDenseFieldRow`, `BucketFull256CompareSummaryRow`, `BucketDenseExperimentCase`, `BucketFull256CompareCase`, `_StructureMetrics` |
| `holdout_fit_diff.py` | 1249 | `HoldoutFitDiffSummary`, `HoldoutFitDiffFieldRow`, `HoldoutStructureMetricRow`, `HoldoutFitFrameCurve`, `HoldoutFitDiffCase` |
| `factor_sweep.py` | 845 | `FactorSweepRow` |
| `eit_digit_metrics.py` | 759 | `EITDigitSummary`, `EITLinearizedModel` |
| `bucket_domain_audit.py` | 625 | `BucketElectrode`, `BucketDomainAuditRow`, `CircleBucketDomain` |
| `visual_audit.py` | 618 | `VisualAuditArtifact`, `VisualAuditExperiment`, `VisualAuditRow`, `VisualAuditRun` |
| `voltage_digit_sweep.py` | 410 | `VoltageDigitSweepSummary`, `VoltageDigitFieldRow` |
| `dynamic_sequence.py` | 410 | `DynamicMeasurementSequence` |
| `temporal_filtering.py` | 398 | `MeasurementTemporalFilterResult` |
| `holdout_point_audit.py` | 357 | `HoldoutPointAuditRow`, `HoldoutPointAuditSummary` |

## 1. Common pattern: `*Row` / `*Summary` ⇒ `as_csv_row()`

Eight of the ten modules expose at least one `@dataclass(frozen=True)`
that ends with an `as_csv_row(self) -> dict[str, float | int | str]`
method. Each method:

- emits a string-keyed dict;
- uses an inline `optional(value) -> "" if None else value` helper for
  nullable numeric columns;
- is consumed by an `csv.DictWriter`-style writer in the same file.

This is the most obvious shared base candidate — a `SweepRow` ABC with
a default `as_csv_row` driven by `dataclasses.asdict`. The subtlety is
twofold:

1. Several rows do small per-field formatting (booleans → `"true"`/
   `"false"` lowercase, percentages, CSV-friendly enum coercion); a
   naive `asdict` loses these.
2. The CSV column order is paper-stable: when the same workbook is
   appended across runs, reordering columns breaks the consuming
   spreadsheet templates. A shared base would have to serialize in
   field-declaration order (`__dataclass_fields__`).

Net: a shared base is **viable**, but the migration must preserve
each existing row's column order + per-cell coercion. Unit fixtures
that compare a CSV bytewise should be added before any source change.

## 2. Cross-module field overlap map

Field name (or close synonym) → set of dataclasses that carry it:

| field | carriers |
|---|---|
| `recon_method` | `BucketDenseSummaryRow`, `BucketDenseFieldRow`, `BucketFull256CompareSummaryRow`, `HoldoutFitDiffSummary`, `HoldoutFitDiffFieldRow`, `VoltageDigitSweepSummary` |
| `n_inverse_points` | `BucketFull256CompareSummaryRow`, `HoldoutFitDiffSummary` |
| `n_elec` / `n_measurements` | `BucketDenseSummaryRow`, `BucketFull256CompareSummaryRow` |
| `holdout_voltage_rmse` / `diff_voltage_rmse` | `BucketDenseSummaryRow`, `HoldoutFitDiffSummary` |
| `sigma_rmse` / `sigma_relative_rmse` / `sigma_mae` / `sigma_max_abs_error` / `sigma_effective_digits` | `BucketDenseSummaryRow`, `HoldoutFitDiffSummary` (with `full_*` / `recon_*` / `delta_*` prefix variants) |
| `centroid_error` / `eccentricity` / `artifact_area` / `artifact_energy` / `artifact_peak` | `BucketDenseSummaryRow`, `HoldoutStructureMetricRow`, `_StructureMetrics` |
| `cell_index` / `sigma_true` / `sigma_recon` / `sigma_error` | `BucketDenseFieldRow`, `HoldoutFitDiffFieldRow` (`sigma_recon_full` / `sigma_recon_candidate`) |
| `experiment` / `domain` | `BucketDenseSummaryRow`, `BucketDenseFieldRow`, `BucketFull256CompareSummaryRow`, `BucketDomainAuditRow` |
| `mesh_h` / `n_cells` / `n_dofs` | `BucketDenseSummaryRow`, `BucketFull256CompareSummaryRow` |
| `ridge` | `BucketDenseSummaryRow`, `BucketFull256CompareSummaryRow` |

The recurring core that *would* graduate cleanly into a base class is

```python
@dataclass(frozen=True)
class ReconMetricRow:           # candidate base
    recon_method: str
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_effective_digits: float
```

with `BucketDenseSummaryRow` / `HoldoutFitDiffSummary` /
`VoltageDigitSweepSummary` becoming subclasses that add their own
columns (artifact metrics, holdout/diff variants, voltage-digit gauges
respectively).

## 3. Per-file disposition

| module | disposition (proposed phase 2) | rationale |
|---|---|---|
| `bucket_dense_experiments.py` | **soft-merge**: extract `ReconMetricRow` base + `_StructureMetrics` shared with `holdout_fit_diff`. Keep `BucketFull256CompareSummaryRow` / `BucketDenseExperimentCase` / `BucketFull256CompareCase` per-file (paper-specific column choices). | Largest file but bulk is sweep orchestration, not row schemas. |
| `holdout_fit_diff.py` | **soft-merge**: rebase `HoldoutFitDiffSummary` on `ReconMetricRow`, share `_StructureMetrics` ⇄ `HoldoutStructureMetricRow`. Keep `HoldoutFitFrameCurve` (frame-time series, no field overlap). | High cross-overlap with `bucket_dense_experiments`; low risk. |
| `voltage_digit_sweep.py` | **soft-merge**: rebase `VoltageDigitSweepSummary` on `ReconMetricRow`. `VoltageDigitFieldRow` keeps its small bespoke schema. | Single sweep, narrow surface. |
| `factor_sweep.py` | **leave-alone (phase 2)**: `FactorSweepRow` schema is paper-driven (column titles encode tunable knobs); no cross-file overlap big enough to justify a base. | One module, one row type; refactor cost > savings. |
| `eit_digit_metrics.py` | **leave-alone**: `EITDigitSummary` / `EITLinearizedModel` are not sweep-row carriers; this is metric-engine code that *feeds* sweeps. | Different abstraction layer. |
| `bucket_domain_audit.py` | **leave-alone**: domain-geometry config + `BucketDomainAuditRow`. `experiment` / `domain` field overlap is too thin. | Geometry / phantom config, not sweep output. |
| `visual_audit.py` | **leave-alone**: artifact-level audit; rows are figure-pointer style, not metric-row style. | Different schema family. |
| `dynamic_sequence.py` | **leave-alone**: `DynamicMeasurementSequence` is a measurement container, not a sweep row. V64 dynamic-data contract pinned. | Different abstraction layer. |
| `temporal_filtering.py` | **leave-alone**: `MeasurementTemporalFilterResult` is filter metadata. | Different abstraction layer. |
| `holdout_point_audit.py` | **soft-merge candidate**: `HoldoutPointAuditRow` / `HoldoutPointAuditSummary` partially overlap holdout_fit_diff fields. Defer until the holdout summary base lands first. | Smaller surface; do after `holdout_fit_diff`. |

In one line: of the 10 modules, **3** carry rows that would benefit
from a shared `ReconMetricRow` + `_StructureMetrics` base
(`bucket_dense_experiments`, `holdout_fit_diff`,
`voltage_digit_sweep`), one is a follow-up candidate
(`holdout_point_audit`), and the remaining 6 should stay as-is. The
two largest files (≥ 1 200 LOC each) get *partial* row-schema
consolidation, not a wholesale rewrite, because the bulk of their
content is orchestration / experiment-specific glue.

## 4. Phase 2a boundary

Phase 2a intentionally stays below the risky `ReconMetricRow` base-class move.
It only introduces `pyeidors.data._sweep_core` helpers for:

- dataclass field-order CSV rows,
- stable `None` / boolean CSV cell coercion,
- repeated CSV writer boilerplate,
- terminal aligned summary tables used by sweep CLIs.

Migrated callers are limited to low-risk serialization edges:
`voltage_digit_sweep`, `factor_sweep`, `bucket_dense_experiments`,
`holdout_fit_diff`, and the voltage/factor sweep CLI writers/tables. Numerical
paths, report text content, row dataclass fields, and historical CSV headers are
unchanged. `tests/unit/test_sweep_core_primitives.py` locks the shared primitive
contract and representative migrated row order/coercion.

T81 remains `~`: the heavier `ReconMetricRow` / `_StructureMetrics` base
consolidation still needs the gates below before T81 can close.

## 4.1. Phase 2b boundary

Phase 2b adds the first CSV byte-stability gate without migrating more source
code. The new fixtures under `tests/fixtures/sweep_csv_columns/` cover the
already-migrated phase 2a rows from `voltage_digit_sweep`, `factor_sweep`,
`bucket_dense_experiments`, and `holdout_fit_diff`. The corresponding unit test
renders each row through its public `as_csv_row()` method plus `csv.DictWriter`
and asserts exact bytes, including column order, empty optional cells, and
lowercase boolean cells.

This is deliberately not the heavier `ReconMetricRow` / `_StructureMetrics`
schema-base move. If those rows are consolidated later, this fixture set should
be extended before the migration rather than relaxed afterward.

## 4.2. Phase 2c boundary

Phase 2c adds the HDF5 sheet-name gate before the heavier row-schema move. The
new `write_hdf5_row_tables` helper writes CSV-compatible sweep rows as HDF5
artifact datasets under the shared `arrays/` group and stores stable
`table_names` / `table_columns` metadata. No existing sweep caller is migrated
in this phase.

The committed fixture
`tests/fixtures/sweep_hdf5_tables/phase2a_table_names.h5` records the current
phase 2a migrated table names. `tests/unit/test_sweep_hdf5_sheet_golden_fixture.py`
generates the same HDF5 layout from the helper and compares group / dataset
paths plus table metadata against the fixture. Future `ReconMetricRow` /
`_StructureMetrics` consolidation should extend this fixture before moving
additional tables.

## 4.3. Phase 2d boundary

Phase 2d lands the first heavier schema-base consolidation without changing
dataclass field order. `pyeidors.data._sweep_core` now owns zero-field mixins
`SweepRow`, `ReconMetricRow`, and `StructureMetricRow`, plus the reusable
`StructureMetrics` value object. The zero-field design is intentional:
dataclass field inheritance would move base fields ahead of child fields and
therefore break historical CSV columns.

Migrated row classes:

- `VoltageDigitSweepSummary` → `ReconMetricRow`; `VoltageDigitFieldRow` →
  `SweepRow`.
- `BucketDenseSummaryRow` / `BucketFull256CompareSummaryRow` →
  `StructureMetricRow`; `BucketDenseFieldRow` → `SweepRow`.
- local `bucket_dense_experiments._StructureMetrics` duplicate removed via
  alias to shared `StructureMetrics`.
- `HoldoutFitDiffSummary` → `ReconMetricRow` with explicit
  `recon_*` metric field view; `HoldoutFitDiffFieldRow` → `SweepRow`;
  `HoldoutStructureMetricRow` → `StructureMetricRow`.

`tests/unit/test_sweep_schema_base_consolidation.py` locks subclassing,
dataclass field order, shared metric views, and the `_StructureMetrics` alias.
CSV and HDF5 golden gates remain the byte-level guard for output stability.

## 5. Phase 2 completion conditions (gate)

Before heavier row-base consolidation lands, the following must be in
place (otherwise paper reproducibility breaks; SPEC §T.T81 boundary):

1. **CSV byte-stable fixture per consolidated row**: phase 2b has
   landed tiny golden CSV files for the phase 2a migrated rows in
   `tests/fixtures/sweep_csv_columns/`. If `ReconMetricRow` /
   `_StructureMetrics` migration proceeds later, add the next row
   fixtures before moving those schemas. This locks column order +
   boolean / numeric coercion.
2. **HDF5 sheet-name fixture**: an HDF5 archive with the historical
   group / dataset names is committed. Phase 2c has landed
   `tests/fixtures/sweep_hdf5_tables/phase2a_table_names.h5` for the
   phase 2a migrated tables; future consolidated row writers must
   match or deliberately extend this layout under the HDF5 fixture
   gate.
3. **`__dataclass_fields__` order test**: each consolidated row has
   `tuple(rec_cls.__dataclass_fields__) == EXPECTED_ORDER` so a
   future contributor cannot reorder columns.
4. **V70 / V71 numerical regression**: re-run the
   `仿真各情况加噪声梯度测试` bucket sweep + `add_noise` parity
   harness; sigma RMSE values must match the pre-merge run to
   `1e-12`.
5. **V72 perf budget**: `pytest tests/unit -q --no-cov` stays under
   the 10-minute local gate after the new base classes import +
   metaclass machinery.
6. **No deletion** of any existing `*Row` / `*Summary` field
   (only **migration into base** + addition of new helpers).

When all six gates pass, T81 status flips from `~` to `x` and the
phase 2 commit can land.

## 6. Phase 1 commit boundary

Phase 1 (this commit) introduces:

- This audit doc (`docs/code-fusion/T81_data_sweep_audit.md`).
- A presence test
  (`tests/unit/test_data_sweep_audit_presence.py`) that locks the
  doc + the 10-module coverage list — so a future contributor cannot
  silently delete the audit before phase 2.

No `src/pyeidors/data/*.py` source file is modified in phase 1; the
sweep / row schemas remain bytewise identical to commit `4a18629`
(the immediately preceding T79 commit).
