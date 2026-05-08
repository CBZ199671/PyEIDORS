"""Tests for common 3D GREIT warmup artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from pyeidors.io.hdf5_artifacts import read_hdf5_artifact
from pyeidors.inverse import (
    GREIT_COMMON_CONFIG_WARMUP_SCHEMA,
    GREIT_EIDORS_HDF5_SCHEMA,
    GREIT_RM_HDF5_SCHEMA,
    GREITRM,
    common_config_runtime_metadata,
    greit_common_config,
    greit_common_config_artifact_path,
    greit_common_config_ids,
    load_greit_common_config,
    load_rm_artifact,
    precompute_greit_common_config,
    register_greit_common_config_artifact,
    resolve_greit_common_config_artifact_path,
)
from scripts.diagnostics import precompute_greit_common_configs as warmup_cli


def test_common_config_precompute_writes_hdf5_and_reuses_existing(
    tmp_path: Path,
) -> None:
    result = precompute_greit_common_config("16", artifact_dir=tmp_path)

    assert result.built is True
    assert result.loaded is True
    assert result.artifact_path.suffix == ".h5"
    assert result.config.config_id == "16e"
    assert result.greit.shape == (108, 208)
    assert result.greit.voxel_shape == (6, 6, 3)

    artifact = read_hdf5_artifact(result.artifact_path)
    assert artifact.schema == GREIT_RM_HDF5_SCHEMA
    assert "rm" in artifact.arrays
    assert "RM" not in artifact.arrays
    assert (
        artifact.metadata["common_config_schema"] == GREIT_COMMON_CONFIG_WARMUP_SCHEMA
    )
    assert artifact.metadata["common_config_id"] == "16e"
    assert artifact.metadata["artifact_format"] == "hdf5"
    assert artifact.metadata["online_hot_path"] == "rm_matmul"
    assert artifact.metadata["fixture_only"] is True
    assert artifact.metadata["official_fixture_scope"] == "48e official fixture passed"
    assert artifact.metadata["protocol_5936_official_status"] == "pending_T97"
    assert artifact.metadata["official_equivalence_claim_allowed"] is False

    warm = precompute_greit_common_config("16e", artifact_dir=tmp_path)
    assert warm.built is False
    assert warm.loaded is True
    np.testing.assert_allclose(warm.greit.rm, result.greit.rm)


def test_common_config_load_prepare_online_and_runtime_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="precompute/register"):
        resolve_greit_common_config_artifact_path("16e", artifact_dir=tmp_path)

    expected_path = greit_common_config_artifact_path("16", artifact_dir=tmp_path)
    assert (
        resolve_greit_common_config_artifact_path(
            "16e",
            artifact_dir=tmp_path,
            must_exist=False,
        )
        == expected_path
    )

    precompute_greit_common_config("16e", artifact_dir=tmp_path)
    loaded = load_greit_common_config(
        "16e",
        artifact_dir=tmp_path,
        prepare_online=True,
        device="cpu",
    )
    assert loaded.prepared_online is True
    assert loaded.greit.rm_handle is not None
    dv = np.ones(greit_common_config("16e").n_measurements, dtype=float)
    recon = loaded.greit.reconstruct(dv, normalize=False, return_metadata=True)
    assert recon.values.shape == greit_common_config("16e").voxel_shape
    assert recon.metadata["online_hot_path"] == "rm_matmul"

    metadata = common_config_runtime_metadata("16e", artifact_dir=tmp_path)
    assert metadata["greit_common_config"] == "16e"
    assert metadata["greit_common_config_artifact_path"] == str(expected_path)
    assert metadata["online_hot_path"] == "rm_matmul"


def test_register_external_common_config_artifact_keeps_hdf5_rm_loadable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_eidors_greit.h5"
    rm = np.eye(108, 208, dtype=np.float64)
    greit = GREITRM(
        rm=rm,
        metadata=MappingProxyType(
            {
                "algorithm": "greit-3d",
                "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
                "artifact_format": "hdf5",
                "eidors_parity": True,
                "keep_model_components": True,
                "online_hot_path": "rm_matmul",
            }
        ),
        voxel_shape=(6, 6, 3),
    )
    greit.save(source)
    source_payload = read_hdf5_artifact(source)
    assert "RM" in source_payload.arrays

    registered = register_greit_common_config_artifact(
        "16e",
        source,
        artifact_dir=tmp_path / "common",
    )

    assert registered.artifact_path.suffix == ".h5"
    assert registered.metadata["common_config_id"] == "16e"
    assert registered.metadata["eidors_parity"] is True
    generic = load_rm_artifact(registered.artifact_path)
    assert generic.rm.shape == (108, 208)
    np.testing.assert_allclose(generic.rm, rm)


def test_precompute_common_config_cli_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    exit_code = warmup_cli.main(
        [
            "--config",
            "16e",
            "--artifact-dir",
            str(tmp_path),
            "--manifest-out",
            str(manifest),
        ]
    )

    assert exit_code == 0
    payload = read_hdf5_artifact(tmp_path / "greit3d_common_16e.h5")
    assert payload.metadata["common_config_id"] == "16e"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["schema"] == GREIT_COMMON_CONFIG_WARMUP_SCHEMA
    assert manifest_payload["config_count"] == 1
    assert manifest_payload["configs"][0]["common_config_id"] == "16e"
    assert tuple(greit_common_config_ids()) == ("16e", "32e", "48e")
