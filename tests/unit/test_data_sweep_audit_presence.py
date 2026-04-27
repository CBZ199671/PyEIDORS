"""T81 phase 1 entrance gate: data-sweep audit doc + 10-module coverage.

T81 spec downgraded the data sweep / audit consolidation to an
"audit + index first, only then merge" task because the row schemas
are paper / report driven and a premature shared base class can
become a maintenance burden bigger than the duplication it removes.
Phase 1 (this commit) ships ``docs/code-fusion/T81_data_sweep_audit.md``
and freezes its presence + 10-file coverage so a later contributor
cannot skip the audit on the way to phase 2.

When phase 2 lands (shared ``ReconMetricRow`` + ``_StructureMetrics``
base, CSV / HDF5 byte-stable fixtures), this test stays green; the
phase 2 commit only adds *new* tests for the byte-stable fixture
contract.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DOC = REPO_ROOT / "docs" / "code-fusion" / "T81_data_sweep_audit.md"

# Modules the audit must catalogue. Sorted so a missing entry produces
# a deterministic diff in the assertion below.
AUDITED_MODULES = sorted(
    [
        "bucket_dense_experiments.py",
        "bucket_domain_audit.py",
        "dynamic_sequence.py",
        "eit_digit_metrics.py",
        "factor_sweep.py",
        "holdout_fit_diff.py",
        "holdout_point_audit.py",
        "temporal_filtering.py",
        "visual_audit.py",
        "voltage_digit_sweep.py",
    ]
)


def test_audit_document_exists() -> None:
    """Phase 1 deliverable: the audit document is committed at a stable path."""
    assert AUDIT_DOC.is_file(), (
        f"T81 phase 1 audit doc missing at {AUDIT_DOC.relative_to(REPO_ROOT)}; "
        "see SPEC.md §T.T81 — phase 1 must ship the index before phase 2 source changes."
    )


def test_audit_document_indexes_all_ten_modules() -> None:
    """Each of the 10 ``src/pyeidors/data/`` sweep modules appears in the audit."""
    text = AUDIT_DOC.read_text(encoding="utf-8")
    missing = [name for name in AUDITED_MODULES if name not in text]
    assert not missing, (
        f"T81 audit doc must index every sweep module; missing: {missing!r}"
    )


def test_audit_document_records_phase_2_entry_gate() -> None:
    """The audit explicitly records the byte-stable + V70/V71/V72 entry gates."""
    text = AUDIT_DOC.read_text(encoding="utf-8")
    for required in (
        "CSV byte-stable fixture",
        "HDF5",
        "V70",
        "V71",
        "V72",
    ):
        assert required in text, (
            f"T81 audit doc must reference phase 2 entry gate token {required!r}"
        )


def test_audit_modules_match_actual_data_directory() -> None:
    """Audit list mirrors what actually lives in ``src/pyeidors/data/``.

    Catches drift if a sweep file is added / renamed without updating
    the audit. Background data-structure modules (``structures.py``,
    ``measurement_dataset.py``, ``adc_quantization.py``,
    ``channels.py`` etc.) are intentionally NOT in the audit list —
    they are different abstraction layers (containers / contracts /
    quantisation), not sweep-row carriers — but the test only enforces
    that every audited module *exists*.
    """
    data_dir = REPO_ROOT / "src" / "pyeidors" / "data"
    for name in AUDITED_MODULES:
        assert (data_dir / name).is_file(), (
            f"audited sweep module {name!r} missing from src/pyeidors/data/"
        )
