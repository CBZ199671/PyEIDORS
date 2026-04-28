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
    "forward/cuda_structured_backend.py": 1,
    "forward/eit_forward_model.py": 5,
    "inverse/greit.py": 2,
    "inverse/jacobian/direct_jacobian.py": 1,
    "inverse/jacobian/linearized.py": 1,
    "inverse/prior/rtr.py": 3,
    "inverse/prior/tv_irls.py": 1,
    "inverse/reconstruction_matrix.py": 2,
    "inverse/reduced/snapshot_bank.py": 3,
    "inverse/solvers/gauss_newton_linear_system.py": 10,
    "inverse/solvers/gauss_newton_startup_cache.py": 1,
    "inverse/solvers/sparse_bayesian_engine.py": 1,
    "io/hdf5_artifacts.py": 1,
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
        str((SRC_PYEIDORS / rel).resolve())
        for rel in EXPECTED_SHA256_PER_FILE
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
