"""T82 phase 4 gate: disk artifact manifest schema audit stays current."""

from __future__ import annotations

from pathlib import Path

from pyeidors.cache.disk_artifacts import (
    FUTURE_DISK_ARTIFACT_KINDS,
    INTEGRATED_DISK_ARTIFACT_KINDS,
    READ_ONLY_DISK_ARTIFACT_KINDS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_DOC = (
    REPO_ROOT / "docs" / "code-fusion" / "T82_disk_artifact_manifest_schema_audit.md"
)


def test_t82_manifest_schema_audit_document_exists() -> None:
    assert AUDIT_DOC.is_file(), (
        "T82 phase 4 audit doc missing; it must record integrated artifact kinds "
        "and future scope before disk-cache unification can close."
    )


def test_t82_manifest_schema_audit_records_canonical_fields() -> None:
    text = AUDIT_DOC.read_text(encoding="utf-8")
    for token in (
        "`artifact_kind`",
        "`artifact_key`",
        "`artifact_manifest`",
        "`subkeys`",
        "`mesh_provenance`",
        "`key_payload`",
        "`files`",
        "`metadata`",
    ):
        assert token in text, f"T82 audit doc missing schema token {token!r}"


def test_t82_manifest_schema_audit_separates_integrated_and_future_scope() -> None:
    text = AUDIT_DOC.read_text(encoding="utf-8")
    for integrated in INTEGRATED_DISK_ARTIFACT_KINDS:
        assert f"`{integrated}`" in text, (
            f"T82 audit doc must list integrated artifact kind {integrated!r}"
        )
    for future in FUTURE_DISK_ARTIFACT_KINDS:
        assert f"`{future}`" in text, f"T82 audit doc must mark future scope {future!r}"
    for read_only in READ_ONLY_DISK_ARTIFACT_KINDS:
        assert f"`{read_only}`" in text, (
            f"T82 audit doc must mark read-only scope {read_only!r}"
        )


def test_t82_manifest_schema_audit_records_governance_close_boundary() -> None:
    text = AUDIT_DOC.read_text(encoding="utf-8")
    assert "T82 governance is complete" in text
    assert "T82 can close here" in text
    assert "as separate future tasks" in text
