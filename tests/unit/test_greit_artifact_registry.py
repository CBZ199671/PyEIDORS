from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.inverse import (
    GREIT_EIDORS_HDF5_SCHEMA,
    greit_artifact_signature,
    greit_artifact_signature_payload,
    load_greit_registry_manifest,
    resolve_greit_artifact,
    resolve_or_build_greit_artifact,
)


class _FakeForwardModel:
    def __init__(self) -> None:
        self._centers = np.array(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=float,
        )
        self._sensitivity = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, -1.0],
            ],
            dtype=float,
        )

    def cell_centers(self) -> np.ndarray:
        return self._centers

    def point_in_volume(self, centers: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(centers).shape[0], dtype=bool)

    def fwd_solve(self, image) -> SimpleNamespace:
        sigma = np.asarray(image.elem_data, dtype=float).reshape(-1)
        return SimpleNamespace(meas=self._sensitivity @ sigma)


def _base_config() -> dict[str, object]:
    return {
        "mesh_dimension": 3,
        "mesh_hash": "mesh-a",
        "fwd_model_hash": "fwd-a",
        "n_elec": 2,
        "n_rings": 1,
        "n_layers": 1,
        "radius": 1.0,
        "height": 1.0,
        "electrode_length_m_override": 0.25,
        "electrode_area_m2_override": 0.05,
        "electrode_height_ratio": 0.2,
        "electrode_level_fractions": [0.25, 0.75],
        "electrode_layout": "ring_major",
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "custom_stim_matrix": np.eye(2),
        "custom_meas_matrices": np.eye(2)[None, :, :],
        "measurement_protocol": "eidors_full_3d",
        "measurement_count": 4,
        "channel_order": [0, 1, 2, 3],
        "bad_channel_mask": [False, False, False, False],
        "background_conductivity": 1.0,
        "contact_impedance": 0.0,
        "normalize_measurements": True,
        "imgsz": (2, 1, 1),
        "xvec": [-1.0, 0.0, 1.0],
        "yvec": [-0.5, 0.5],
        "zvec": [-0.5, 0.5],
        "downsample": None,
        "target_distribution": None,
        "target_size": None,
        "target_radius": 0.2,
        "target_contrast": 0.5,
        "desired_solution_fn": "sigmoid",
        "desired_solution_params": {"desired_img_threshold": 0.0},
        "noise_covar": 1.0,
        "weight": 0.5,
        "noise_figure": None,
        "image_SNR": None,
        "training_mode": "forward",
        "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
        "builder_backend": "native",
        "builder_semantic_version": "native-greit-finite-target-v2",
    }


def test_v92_signature_axes_are_hard_fields() -> None:
    base = _base_config()
    base_sig = greit_artifact_signature(base)
    mutations = {
        "mesh_dimension": 2,
        "mesh_hash": "mesh-b",
        "n_elec": 3,
        "n_rings": 2,
        "n_layers": 2,
        "radius": 1.1,
        "height": 1.2,
        "electrode_length_m_override": 0.3,
        "electrode_area_m2_override": 0.06,
        "electrode_height_ratio": 0.25,
        "electrode_level_fractions": [0.2, 0.8],
        "electrode_layout": "level_major",
        "stim_pattern": "{op}",
        "meas_pattern": "{mono}",
        "measurement_count": 5,
        "channel_order": [3, 2, 1, 0],
        "bad_channel_mask": [False, True, False, False],
        "background_conductivity": 1.2,
        "normalize_measurements": False,
        "imgsz": (3, 1, 1),
        "xvec": [-1.0, -0.2, 0.4, 1.0],
        "yvec": [-0.6, 0.6],
        "zvec": [-0.4, 0.4],
        "downsample": [2, 1, 1],
        "target_distribution": [[0.0, 0.0, 0.0]],
        "target_size": 0.3,
        "target_radius": 0.3,
        "target_size_semantics": "absolute",
        "target_contrast": 0.75,
        "desired_solution_fn": "custom",
        "desired_solution_params": {"desired_img_threshold": 0.01},
        "noise_covar": 2.0,
        "weight": 0.7,
        "noise_figure": 0.8,
        "image_SNR": 20.0,
        "training_mode": "linearized",
        "artifact_schema": "other-schema",
        "builder_backend": "matlab-eidors",
        "builder_semantic_version": "official-v1",
    }

    for key, value in mutations.items():
        changed = dict(base)
        changed[key] = value
        assert greit_artifact_signature(changed) != base_sig, key


def test_resolve_or_build_native_greit_artifact_registers_warm_hit(tmp_path) -> None:
    config = _base_config()
    first = resolve_or_build_greit_artifact(
        config,
        registry_dir=tmp_path,
        fwd_model=_FakeForwardModel(),
    )

    assert first.built is True
    assert first.artifact_path.exists()
    assert first.greit.y is not None
    assert first.greit.d is not None
    assert first.greit.rec_model is not None
    assert first.greit.metadata["fixture_only"] is False
    assert first.greit.metadata["eidors_parity"] is True
    assert first.greit.metadata["greit_registry_signature"] == first.signature
    manifest = load_greit_registry_manifest(tmp_path)
    assert first.signature in manifest["entries"]

    second = resolve_or_build_greit_artifact(
        config,
        registry_dir=tmp_path,
        auto_build=False,
    )
    assert second.built is False
    assert second.cache_status == "disk_hit"
    assert second.artifact_path == first.artifact_path

    changed = dict(config)
    changed["n_rings"] = 2
    assert resolve_greit_artifact(changed, registry_dir=tmp_path) is None
    with pytest.raises(FileNotFoundError):
        resolve_or_build_greit_artifact(
            changed,
            registry_dir=tmp_path,
            auto_build=False,
        )


def test_native_greit_builder_scales_target_size_by_tank_radius(tmp_path) -> None:
    config = _base_config()
    config["target_radius"] = None
    config["target_size"] = 0.25
    config["radius"] = 2.0

    lookup = resolve_or_build_greit_artifact(
        config,
        registry_dir=tmp_path,
        fwd_model=_FakeForwardModel(),
    )

    assert lookup.greit.metadata["target_size_semantics"] == "fraction_of_tank_radius"
    assert lookup.greit.metadata["target_radius_effective"] == pytest.approx(0.5)
    assert lookup.signature_payload["target_radius_effective"] == pytest.approx(0.5)


def test_native_greit_builder_applies_channel_order_from_signature_config(
    tmp_path,
) -> None:
    config = _base_config()
    config["channel_order"] = [2, 0, 3, 1]
    model = _FakeForwardModel()

    lookup = resolve_or_build_greit_artifact(
        config,
        registry_dir=tmp_path,
        fwd_model=model,
    )

    expected_vh = model._sensitivity @ np.ones(2, dtype=float)
    np.testing.assert_allclose(lookup.greit.vh, expected_vh[[2, 0, 3, 1]])
    assert lookup.greit.metadata["fairness_contract"]["measurement_order_hash"]
    assert lookup.greit.metadata["eidors_parity"] is True


def test_native_greit_builder_masks_cylindrical_rec_model(tmp_path) -> None:
    config = _base_config()
    config.update(
        {
            "radius": 1.0,
            "height": 1.0,
            "imgsz": (4, 4, 1),
            "xvec": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "yvec": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "zvec": [-0.5, 0.5],
            "rec_mask": "cylindrical_fem_volume_v1",
        }
    )

    lookup = resolve_or_build_greit_artifact(
        config,
        registry_dir=tmp_path,
        fwd_model=_FakeForwardModel(),
    )

    rec_model = np.asarray(lookup.greit.rec_model, dtype=float)
    assert rec_model.shape[0] < 16
    assert float(np.max(np.hypot(rec_model[:, 0], rec_model[:, 1]))) <= 1.0 + 1e-12
    assert lookup.greit.metadata["rec_mask"] == "cylindrical_fem_volume_v1"


def test_signature_payload_stores_hashes_for_large_arrays() -> None:
    payload = greit_artifact_signature_payload(_base_config())

    assert payload["stim_pattern_hash"]
    assert payload["meas_pattern_hash"]
    assert payload["channel_order_hash"]
    assert payload["greit_rec_grid"]["imgsz"] == [2, 1, 1]
    assert payload["greit_rec_grid"]["mask"] == "cylindrical_fem_volume_v1"
    assert payload["target_size_semantics"] == "fraction_of_tank_radius"


def test_matlab_eidors_backend_script_contains_official_calls(tmp_path) -> None:
    from scripts.diagnostics.build_greit_artifact_with_matlab_eidors import (
        build_matlab_script,
    )

    script = build_matlab_script(
        _base_config(),
        tmp_path / "official.mat",
        "D:/eidors",
    )

    assert "GREIT3D_distribution" in script
    assert "mk_GREIT_model" in script
    assert "opt.keep_model_components = true" in script
    assert "save(out_file" in script
