"""Realtime GUI Bridge v3 protocol/current binding without a Qt event loop."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from eit_app.models.forward_model_config import ForwardModelConfig
from eit_app.models.frame_model import FrameData
from eit_app.ui.main_window import EITWorkstation
from pyeidors.interop import ProtocolChannelMapping


def test_v759_realtime_request_proves_mapping_and_frame_current(
    monkeypatch,
) -> None:
    import pyeidors.interop as interop

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
            2.0e-4 * model_stim[1],
            3.0e-4 * model_stim[0],
        ]
    )
    hardware_meas = (
        np.asarray([-model_meas[1][1], model_meas[1][0]]),
        np.asarray([-model_meas[0][1], model_meas[0][0]]),
    )
    registered = SimpleNamespace(
        model_id="a" * 64,
        forward_fingerprint="b" * 64,
        protocol_layout_hash="c" * 64,
        protocol_physics_hash="d" * 64,
    )
    context = SimpleNamespace(
        registered=registered,
        electrode_specs=tuple(SimpleNamespace(kind="pem") for _ in range(4)),
        protocol=SimpleNamespace(stim_matrix=model_stim),
        effective_meas_matrices=model_meas,
    )

    class FakeRegistry:
        def bound_model(self, flow: str):
            assert flow == "realtime"
            return registered

    class FakeFactory:
        def __init__(self, registry) -> None:
            assert isinstance(registry, FakeRegistry)

        def for_flow(self, flow: str):
            assert flow == "realtime"
            return context

    monkeypatch.setattr(interop, "ModelContextFactory", FakeFactory)
    config = ForwardModelConfig(
        mesh_dimension=2,
        n_elec=4,
        n_rings=1,
        measurement_protocol="custom",
        custom_stim_matrix=hardware_stim / np.asarray([[2.0e-4], [3.0e-4]]),
        custom_meas_matrices=list(hardware_meas),
        drive_mode="normalized",
        drive_value=1.0,
    )
    fake_window = SimpleNamespace(
        _bridge_model_registry=FakeRegistry(),
        _rec_ctrl=SimpleNamespace(_session_metadata={"stim_amp_uA": 900.0}),
        _device_config={"stim_amp_uA": 800.0},
        _current_hardware_forward_model_config=lambda: config,
    )
    frame = FrameData(
        real=np.asarray([10.0, 20.0, 30.0, 40.0]),
        imag=np.zeros(4),
        timestamp=0.0,
        frame_index=1,
        metadata={"actual_stim_matrix": hardware_stim},
    )

    metadata = EITWorkstation._bound_realtime_bridge_metadata(fake_window, frame)

    assert metadata["model_id"] == registered.model_id
    assert metadata["actual_current_resolution"]["source"] == "frame"
    mapping = ProtocolChannelMapping.from_mapping(metadata["channel_mapping"])
    assert mapping.stimulation_permutation == (1, 0)
    np.testing.assert_allclose(
        mapping.apply(frame.real),
        [40.0, -30.0, 20.0, -10.0],
    )
