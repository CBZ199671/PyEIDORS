from __future__ import annotations

from eit_app.hardware.factory import create_device_from_config
from eit_app.measurement_layout import measurement_layout_from_config


def test_simulator_respects_future_measurement_layout_configuration() -> None:
    layout = measurement_layout_from_config({"n_elec": 32})
    device = create_device_from_config("simulator", {"n_elec": 32})

    device.connect()
    device.start_measurement()
    frame = device.read_frame()
    device.stop_measurement()
    impedance = device.measure_contact_impedance()
    device.disconnect()

    assert frame.real.shape == (layout["points_per_frame"],)
    assert frame.imag.shape == (layout["points_per_frame"],)
    assert frame.metadata["points_per_frame"] == layout["points_per_frame"]
    assert frame.metadata["n_elec"] == 32
    assert impedance.shape == (32,)
