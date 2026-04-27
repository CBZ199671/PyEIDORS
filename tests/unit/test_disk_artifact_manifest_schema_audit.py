"""T82 phase 4 gate: disk artifact manifest schema audit stays current."""

from __future__ import annotations

from pathlib import Path

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
    for integrated in ("`hdf5-artifact`", "`dolfinx-mesh-cache`"):
        assert integrated in text, (
            f"T82 audit doc must list integrated artifact kind {integrated}"
        )
    for future in (
        "`adios4dolfinx-checkpoint`",
        "`adios2-vtx-side-artifact`",
        "`cache-manager-disk-object`",
        "legacy `.npz` artifacts",
        "`MeshCacheLayer` protocol",
    ):
        assert future in text, f"T82 audit doc must mark future scope {future}"


def test_t82_manifest_schema_audit_keeps_t82_open_boundary() -> None:
    text = AUDIT_DOC.read_text(encoding="utf-8")
    assert "T82 is not complete yet" in text
    assert "Keep T82 status `~`" in text
