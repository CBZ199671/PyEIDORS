"""T90 — hash audit doc presence + per-file sha256 inventory freeze."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PYEIDORS = REPO_ROOT / "src" / "pyeidors"
AUDIT_DOC = REPO_ROOT / "docs" / "code-fusion" / "T90_hash_helper_audit.md"

EXPECTED_SHA256_PER_FILE: dict[str, int] = {
    "cache/disk_artifacts.py": 2,
    "cache/keys.py": 6,
    "cache/object_signature.py": 2,
    "cache/process_lru.py": 1,
    "forward/cuda_structured_backend.py": 0,
    "forward/eit_forward_model.py": 3,
    "inverse/greit.py": 2,
    "inverse/greit_registry.py": 1,
    "inverse/jacobian/direct_jacobian.py": 0,
    "inverse/jacobian/linearized.py": 0,
    "inverse/prior/rtr.py": 3,
    "inverse/prior/tv_irls.py": 1,
    "inverse/reconstruction_matrix.py": 2,
    "inverse/reduced/snapshot_bank.py": 0,
    "inverse/solvers/gauss_newton_linear_system.py": 2,
    "inverse/solvers/gauss_newton_startup_cache.py": 0,
    "inverse/solvers/sparse_bayesian_engine.py": 0,
    "io/hdf5_artifacts.py": 2,
    "perf/capabilities.py": 1,
}

EXPECTED_AUDIT_SECTIONS = (
    "## §1 Scope",
    "## §2 Inventory baseline",
    "## §3 Classification",
    "## §4 V76 semantic-cache check",
    "## §5 Recommendations",
    "## §6 Gate",
)


def _sha256_count_in(path: Path) -> int:
    pattern = re.compile(r"hashlib\.sha256\(")
    return sum(1 for _ in pattern.finditer(path.read_text(encoding="utf-8")))


def test_audit_doc_exists_with_required_sections() -> None:
    assert AUDIT_DOC.exists(), f"audit doc missing: {AUDIT_DOC}"
    text = AUDIT_DOC.read_text(encoding="utf-8")
    for section in EXPECTED_AUDIT_SECTIONS:
        assert section in text, f"audit doc missing section: {section}"


def test_per_file_sha256_inventory_matches_audit_baseline() -> None:
    actual: dict[str, int] = {}
    for relpath in EXPECTED_SHA256_PER_FILE:
        actual[relpath] = _sha256_count_in(SRC_PYEIDORS / relpath)
    assert actual == EXPECTED_SHA256_PER_FILE, (
        "sha256 inventory drift; update docs/code-fusion/T90_hash_helper_audit.md "
        "and the EXPECTED_SHA256_PER_FILE baseline together. "
        f"actual={actual!r}"
    )


def test_no_undocumented_files_added_sha256_calls() -> None:
    documented = {
        str((SRC_PYEIDORS / rel).resolve()) for rel in EXPECTED_SHA256_PER_FILE
    }
    pattern = re.compile(r"hashlib\.sha256\(")
    undocumented: list[str] = []
    for py in SRC_PYEIDORS.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if str(py.resolve()) in documented:
            continue
        if pattern.search(py.read_text(encoding="utf-8")):
            undocumented.append(str(py.relative_to(REPO_ROOT)))
    assert not undocumented, (
        "new files added hashlib.sha256(); update T90 audit + baseline. "
        f"undocumented={undocumented!r}"
    )


def test_sigma_hash_sites_use_streaming_payload_helper() -> None:
    for relpath in (
        "inverse/jacobian/direct_jacobian.py",
        "inverse/jacobian/linearized.py",
        "inverse/solvers/gauss_newton_startup_cache.py",
    ):
        text = (SRC_PYEIDORS / relpath).read_text(encoding="utf-8")
        assert "hash_array_payload" in text
        assert ".tobytes(" not in text
        assert "hashlib.sha256(" not in text


def test_gn_linear_system_cache_hashes_stream_payloads() -> None:
    text = (
        SRC_PYEIDORS / "inverse" / "solvers" / "gauss_newton_linear_system.py"
    ).read_text(encoding="utf-8")
    assert "hash_array_payload" in text
    assert ".tobytes(" not in text


def test_remaining_array_digest_sites_use_streaming_payload_helper() -> None:
    for relpath in (
        "forward/cuda_structured_backend.py",
        "inverse/greit_registry.py",
        "inverse/greit.py",
        "inverse/prior/rtr.py",
        "inverse/prior/tv_irls.py",
        "inverse/reconstruction_matrix.py",
        "inverse/reduced/snapshot_bank.py",
        "inverse/solvers/sparse_bayesian_engine.py",
    ):
        text = (SRC_PYEIDORS / relpath).read_text(encoding="utf-8")
        assert (
            "hash_array_payload" in text or "update_digest_with_array_payload" in text
        )
        assert ".tobytes(" not in text
