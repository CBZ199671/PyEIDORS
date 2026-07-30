"""Bridge v3 realtime protocol proof and actual-current gates."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.interop import (
    ActualCurrentResolution,
    ProtocolChannelMapping,
    prove_protocol_mapping,
    resolve_actual_stimulation,
)


def _protocols():
    model_stim = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    model_meas = (
        np.asarray(
            [
                [0.0, 0.0, 1.0, -1.0],
                [1.0, 0.0, -1.0, 0.0],
            ]
        ),
        np.asarray(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -1.0],
            ]
        ),
    )
    hardware_stim = np.asarray(
        [
            2.0 * model_stim[1],
            0.5 * model_stim[0],
        ]
    )
    hardware_meas = (
        np.asarray(
            [
                -model_meas[1][1],
                model_meas[1][0],
            ]
        ),
        np.asarray(
            [
                -model_meas[0][1],
                model_meas[0][0],
            ]
        ),
    )
    return model_stim, model_meas, hardware_stim, hardware_meas


def test_v759_unique_permutation_sign_and_current_scale_are_proven() -> None:
    model_stim, model_meas, hardware_stim, hardware_meas = _protocols()

    mapping = prove_protocol_mapping(
        model_stim_matrix=model_stim,
        model_meas_matrices=model_meas,
        hardware_stim_matrix=hardware_stim,
        hardware_meas_matrices=hardware_meas,
    )

    assert mapping.stimulation_permutation == (1, 0)
    np.testing.assert_allclose(mapping.stimulation_scales, [0.5, 2.0])
    np.testing.assert_allclose(
        mapping.runtime_stim_matrix,
        [[0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 2.0, -2.0]],
    )
    # Hardware flat order: h0=[-m1.1, m1.0], h1=[-m0.1, m0.0].
    hardware_values = np.asarray([10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(
        mapping.apply(hardware_values),
        [40.0, -30.0, 20.0, -10.0],
    )
    assert len(mapping.runtime_fingerprint) == 64
    assert mapping.to_mapping()["proof"].startswith("unique_exact")
    restored = ProtocolChannelMapping.from_mapping(mapping.to_mapping())
    restored.validate_for_model(
        model_stim_matrix=model_stim,
        measurement_count=4,
    )
    np.testing.assert_allclose(
        restored.apply(hardware_values),
        [40.0, -30.0, 20.0, -10.0],
    )


def test_v759_persisted_mapping_tamper_is_rejected() -> None:
    model_stim, model_meas, hardware_stim, hardware_meas = _protocols()
    mapping = prove_protocol_mapping(
        model_stim_matrix=model_stim,
        model_meas_matrices=model_meas,
        hardware_stim_matrix=hardware_stim,
        hardware_meas_matrices=hardware_meas,
    ).to_mapping()
    mapping["channel_signs"][0] *= -1

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ProtocolChannelMapping.from_mapping(mapping)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda model_stim, _model_meas, hardware_stim, _hardware_meas: (
                hardware_stim.__setitem__(1, hardware_stim[0])
            ),
            "Missing hardware stimulation match",
        ),
        (
            lambda _model_stim, _model_meas, _hardware_stim, hardware_meas: (
                hardware_meas[1].__setitem__(1, hardware_meas[1][0])
            ),
            "Missing hardware measurement match",
        ),
    ],
)
def test_v759_missing_protocol_rows_block(
    mutate,
    message: str,
) -> None:
    model_stim, model_meas, hardware_stim, hardware_meas = _protocols()
    mutate(model_stim, model_meas, hardware_stim, hardware_meas)

    with pytest.raises(ValueError, match=message):
        prove_protocol_mapping(
            model_stim_matrix=model_stim,
            model_meas_matrices=model_meas,
            hardware_stim_matrix=hardware_stim,
            hardware_meas_matrices=hardware_meas,
        )


def test_v759_duplicate_measurement_rows_are_ambiguous() -> None:
    model_stim, model_meas, hardware_stim, hardware_meas = _protocols()
    duplicate_model_meas = (
        np.asarray([model_meas[0][0], model_meas[0][0]]),
        model_meas[1],
    )
    hardware_meas = (
        hardware_meas[0],
        np.asarray([model_meas[0][0], model_meas[0][0]]),
    )

    with pytest.raises(ValueError, match="Ambiguous hardware measurement"):
        prove_protocol_mapping(
            model_stim_matrix=model_stim,
            model_meas_matrices=duplicate_model_meas,
            hardware_stim_matrix=hardware_stim,
            hardware_meas_matrices=hardware_meas,
        )


def test_v759_actual_current_priority_and_units() -> None:
    model_stim, _model_meas, _hardware_stim, _hardware_meas = _protocols()

    resolved = resolve_actual_stimulation(
        model_stim,
        frame_metadata={"stim_amp_uA": [200.0, 300.0]},
        session_metadata={"stim_amp_uA": 400.0},
        device_config={"stim_amp_uA": 500.0},
    )

    assert resolved.source == "frame"
    np.testing.assert_allclose(resolved.row_scales, [2.0e-4, 3.0e-4])
    np.testing.assert_allclose(
        resolved.stim_matrix,
        model_stim * np.asarray([[2.0e-4], [3.0e-4]]),
    )
    assert len(resolved.runtime_physics_hash) == 64
    restored = ActualCurrentResolution.from_mapping(resolved.to_mapping())
    assert restored.source == "frame"
    np.testing.assert_allclose(restored.stim_matrix, resolved.stim_matrix)


def test_v759_actual_current_tamper_is_rejected() -> None:
    model_stim, _model_meas, _hardware_stim, _hardware_meas = _protocols()
    payload = resolve_actual_stimulation(
        model_stim,
        frame_metadata={"stim_amp_uA": 200.0},
    ).to_mapping()
    payload["row_scales"][0] *= 2.0

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ActualCurrentResolution.from_mapping(payload)


@pytest.mark.parametrize(
    "metadata",
    [
        {"actual_current_a": 0.0},
        {"actual_current_a": np.nan},
        {
            "actual_stim_matrix": [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        },
    ],
)
def test_v759_invalid_actual_current_blocks(metadata) -> None:
    model_stim, _model_meas, _hardware_stim, _hardware_meas = _protocols()

    with pytest.raises(ValueError):
        resolve_actual_stimulation(
            model_stim,
            frame_metadata=metadata,
        )
