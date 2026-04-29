from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "cache"
    / "migrate_artifacts_to_hdf5.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("migrate_artifacts_to_hdf5", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_migration_cli_dry_run_emits_manifest_without_writing_targets(
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_module()
    np.savez_compressed(tmp_path / "legacy_rm.npz", rm=np.eye(2))
    np.save(tmp_path / "legacy_vector.npy", np.arange(3, dtype=np.float64))
    manifest_path = tmp_path / "manifest.json"

    code = module.run(
        [
            "--root",
            str(tmp_path),
            "--dry-run",
            "--manifest",
            str(manifest_path),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == module.MANIFEST_SCHEMA
    assert payload["mode"] == "dry-run"
    assert payload["counts"] == {"error": 0, "migrated": 0, "planned": 2, "skipped": 0}
    assert manifest_path.exists()
    assert not (tmp_path / "legacy_rm.h5").exists()
    assert not (tmp_path / "legacy_vector.h5").exists()


def test_migration_cli_apply_writes_hdf5_and_keeps_sources(
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_module()
    np.savez_compressed(tmp_path / "legacy_rm.npz", rm=np.eye(2))
    np.save(tmp_path / "legacy_vector.npy", np.arange(3, dtype=np.float64))

    code = module.run(["--root", str(tmp_path), "--apply"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "apply"
    assert payload["counts"]["migrated"] == 2
    assert (tmp_path / "legacy_rm.npz").exists()
    assert (tmp_path / "legacy_vector.npy").exists()

    rm_artifact = read_hdf5_artifact(tmp_path / "legacy_rm.h5")
    vector_artifact = read_hdf5_artifact(tmp_path / "legacy_vector.h5")

    assert rm_artifact.metadata["legacy_source_read_only"] is True
    assert rm_artifact.metadata["legacy_format"] == "npz"
    np.testing.assert_allclose(rm_artifact.arrays["rm"], np.eye(2))
    assert vector_artifact.metadata["legacy_format"] == "npy"
    np.testing.assert_allclose(vector_artifact.arrays["array"], [0.0, 1.0, 2.0])
