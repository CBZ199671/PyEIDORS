from __future__ import annotations

import io
import json
from types import SimpleNamespace

import h5py
import numpy as np


def _request_payload(tmp_path, *, scenario_id: str = "ecd-cwr-sim-test"):
    return {
        "schema_version": "ecd-cwr-cem-backend-request-v1",
        "command": "ecd_cwr_simulate_cem",
        "scenario_id": scenario_id,
        "model": {
            "electrode_model": "cem",
            "contact_impedance_mode": "per_electrode_zc",
            "electrode_count": 16,
            "stim_pattern": "{ad}",
            "measurement_pattern": "{ad}",
            "rotate_measurements": True,
            "use_measurement_current": True,
            "output_layout": "row_major_16x16_256_complex",
        },
        "scenario": {
            "scenario_id": scenario_id,
            "model_kind": "cem_per_electrode_zc",
            "target_count": 2,
            "target_placement": "center",
            "conductivity_pattern": "mixed",
            "contact_impedance": {"label": "zc_x20", "multiplier": 20.0},
            "noise": {"label": "noise_inf", "snr_db": "Infinity"},
            "fault_mode": "adjacent_dual",
            "fault_electrodes": [0, 1],
        },
        "output": {
            "hdf5_path": str(tmp_path / f"{scenario_id}.h5"),
            "label_json_path": str(tmp_path / f"{scenario_id}.label.json"),
            "layout": "row_major_16x16_256_complex",
            "full_observation_count": 256,
            "retained_observation_count": 208,
        },
    }


def test_ecd_cwr_request_builds_full_256_cem_forward_and_writes_outputs(tmp_path):
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult
    from eit_app.ecd_cwr_simulation import RESULT_SCHEMA, run_ecd_cwr_simulation_request

    captured = {"requests": []}

    def fake_execute(request, progress_cb=None):
        captured["requests"].append(request)
        call_index = len(captured["requests"])
        values = np.arange(256, dtype=np.float32) + 1j * np.arange(
            256,
            dtype=np.float32,
        )
        if call_index == 2:
            values = np.ones(256, dtype=np.complex64)
        return ForwardSolverResult(
            boundary_voltages=values,
            homogeneous_voltages=np.ones(256, dtype=np.complex64),
            ground_truth_conductivity=np.ones(4, dtype=np.float32),
            node_coords=np.zeros((4, 2), dtype=np.float32),
            cell_connectivity=np.zeros((1, 4), dtype=np.int32),
            n_elements=1,
            n_measurements=256,
            forward_model_config={"mock": True},
        )

    payload = _request_payload(tmp_path)

    metadata = run_ecd_cwr_simulation_request(payload, execute_forward=fake_execute)

    assert len(captured["requests"]) == 2
    request = captured["requests"][0]
    reference_request = captured["requests"][1]
    cfg = request.forward_model_config
    assert cfg["stim_pattern"] == "{ad}"
    assert cfg["meas_pattern"] == "{ad}"
    assert cfg["rotate_meas"] is True
    assert cfg["use_meas_current"] is True
    assert cfg["contact_impedance"][0] == 0.2
    assert cfg["contact_impedance"][1] == 0.2
    assert cfg["contact_impedance"][2] == 0.01
    assert len(request.inhomogeneities) == 2
    assert request.inhomogeneities[0].conductivity == 2.0
    assert request.inhomogeneities[1].conductivity == 0.5
    reference_cfg = reference_request.forward_model_config
    assert reference_cfg["contact_impedance"] == [0.01] * 16
    assert reference_cfg["simulation_reference_frame"] is True
    assert len(reference_request.inhomogeneities) == 0
    assert metadata["schema"] == RESULT_SCHEMA
    assert metadata["full_observation_count"] == 256
    assert metadata["retained_observation_count"] == 208

    with h5py.File(tmp_path / "ecd-cwr-sim-test.h5", "r") as handle:
        assert handle.attrs["schema"] == RESULT_SCHEMA
        assert handle["raw_complex_256"].shape == (16, 16)
        assert handle["reference_complex_256"].shape == (16, 16)
        assert handle["retained_complex_208"].shape == (208,)
        assert handle["retained_indices_208"].shape == (208,)
        assert handle["ground_truth_conductivity"].shape == (4,)
        assert handle["node_coords"].shape == (4, 2)
        assert handle["cell_connectivity"].shape == (1, 4)
        np.testing.assert_allclose(handle["ground_truth_conductivity"][:], np.ones(4))
        np.testing.assert_allclose(
            handle["reference_complex_256"][:], np.ones((16, 16))
        )
        retained = handle["retained_indices_208"][:]
        assert int(retained[0]) == 2
        assert int(retained[-1]) == 254

    label = json.loads((tmp_path / "ecd-cwr-sim-test.label.json").read_text("utf-8"))
    assert label["fault_mode"] == "adjacent_dual"
    assert label["fault_electrodes"] == [0, 1]
    assert label["contact_impedance"][0] == 0.2


def test_backend_worker_serve_accepts_ecd_cwr_simulation_command(
    monkeypatch,
    capsys,
    tmp_path,
):
    import eit_app.backend_worker as worker
    import eit_app.ecd_cwr_simulation as ecd_cwr

    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_payload(tmp_path)), "utf-8")

    def fake_run(input_path, *, progress_cb=None):
        assert str(input_path) == str(request_path)
        if progress_cb is not None:
            progress_cb("mock progress")
        return {
            "schema": ecd_cwr.RESULT_SCHEMA,
            "scenario_id": "ecd-cwr-sim-test",
            "hdf5_path": str(tmp_path / "result.h5"),
            "label_json_path": str(tmp_path / "result.label.json"),
        }

    monkeypatch.setattr(ecd_cwr, "run_ecd_cwr_simulation_request_file", fake_run)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "id": "ecd-1",
                    "command": "ecd_cwr_simulate_cem",
                    "input": str(request_path),
                },
                sort_keys=True,
            )
            + "\n"
        ),
    )

    assert worker._serve(SimpleNamespace()) == 0

    messages = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert messages == [
        {"id": "ecd-1", "message": "mock progress", "type": "progress"},
        {
            "id": "ecd-1",
            "metadata": {
                "schema": ecd_cwr.RESULT_SCHEMA,
                "scenario_id": "ecd-cwr-sim-test",
                "hdf5_path": str(tmp_path / "result.h5"),
                "label_json_path": str(tmp_path / "result.label.json"),
            },
            "status": "ok",
            "type": "done",
        },
    ]


def test_ecd_cwr_request_can_emit_contact_jacobian(tmp_path):
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult
    from eit_app.ecd_cwr_simulation import run_ecd_cwr_simulation_request

    captured = {"requests": []}

    def fake_execute(request, progress_cb=None):
        captured["requests"].append(request)
        cfg = request.forward_model_config
        contact_impedance = np.asarray(cfg["contact_impedance"], dtype=np.float64)
        values = np.ones(256, dtype=np.complex64)
        for electrode, zc in enumerate(contact_impedance):
            values += np.complex64((zc - 0.01) * (electrode + 1))
        return ForwardSolverResult(
            boundary_voltages=values,
            homogeneous_voltages=np.ones(256, dtype=np.complex64),
            ground_truth_conductivity=np.ones(4, dtype=np.float32),
            node_coords=np.zeros((4, 2), dtype=np.float32),
            cell_connectivity=np.zeros((1, 4), dtype=np.int32),
            n_elements=1,
            n_measurements=256,
            forward_model_config={"mock": True},
        )

    payload = _request_payload(tmp_path, scenario_id="ecd-cwr-sim-jz")
    payload["model"]["emit_contact_jacobian"] = True
    payload["model"]["contact_jacobian_step"] = 1.0e-4

    run_ecd_cwr_simulation_request(payload, execute_forward=fake_execute)

    assert len(captured["requests"]) == 18
    with h5py.File(tmp_path / "ecd-cwr-sim-jz.h5", "r") as handle:
        assert handle["contact_jacobian_208x16"].shape == (208, 16)
        assert handle.attrs["contact_jacobian_step"] == 1.0e-4
        expected = np.tile(np.arange(1, 17, dtype=np.float32), (208, 1))
        np.testing.assert_allclose(
            handle["contact_jacobian_208x16"][:].real,
            expected,
            rtol=1.0e-3,
            atol=1.0e-3,
        )


def test_ecd_cwr_request_can_emit_multi_frequency_observations(tmp_path):
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult
    from eit_app.ecd_cwr_simulation import run_ecd_cwr_simulation_request

    captured = {"requests": []}

    def fake_execute(request, progress_cb=None):
        captured["requests"].append(request)
        cfg = request.forward_model_config
        contact_impedance = np.asarray(cfg["contact_impedance"], dtype=np.float64)
        frequency = float(cfg.get("simulation_frequency_hz", 0.0))
        reference_offset = 100.0 if cfg.get("simulation_reference_frame") else 0.0
        value = (
            reference_offset
            + frequency / 1000.0
            + float(np.sum(contact_impedance))
            + 1j * float(contact_impedance[0])
        )
        values = np.full(256, np.complex64(value), dtype=np.complex64)
        return ForwardSolverResult(
            boundary_voltages=values,
            homogeneous_voltages=np.ones(256, dtype=np.complex64),
            ground_truth_conductivity=np.ones(4, dtype=np.float32),
            node_coords=np.zeros((4, 2), dtype=np.float32),
            cell_connectivity=np.zeros((1, 4), dtype=np.int32),
            n_elements=1,
            n_measurements=256,
            forward_model_config={"mock": True},
        )

    payload = _request_payload(tmp_path, scenario_id="ecd-cwr-sim-mf")
    payload["model"]["frequencies_hz"] = [10_000.0, 50_000.0]
    payload["model"]["frequency_contact_impedance_multipliers"] = [1.0, 0.5]

    metadata = run_ecd_cwr_simulation_request(payload, execute_forward=fake_execute)

    assert metadata["frequency_count"] == 2
    assert len(captured["requests"]) == 6
    assert "simulation_frequency_hz" not in captured["requests"][0].forward_model_config
    assert "simulation_frequency_hz" not in captured["requests"][1].forward_model_config
    assert (
        captured["requests"][2].forward_model_config["simulation_frequency_hz"]
        == 10_000.0
    )
    assert (
        captured["requests"][3].forward_model_config["simulation_frequency_hz"]
        == 10_000.0
    )
    assert (
        captured["requests"][4].forward_model_config["simulation_frequency_hz"]
        == 50_000.0
    )
    assert (
        captured["requests"][5].forward_model_config["simulation_frequency_hz"]
        == 50_000.0
    )

    with h5py.File(tmp_path / "ecd-cwr-sim-mf.h5", "r") as handle:
        np.testing.assert_allclose(handle["frequency_hz"][:], [10_000.0, 50_000.0])
        np.testing.assert_allclose(
            handle["frequency_contact_impedance_multipliers"][:],
            [1.0, 0.5],
        )
        assert handle.attrs["frequency_count"] == 2
        assert handle["frequency_raw_complex_256"].shape == (2, 16, 16)
        assert handle["frequency_reference_complex_256"].shape == (2, 16, 16)
        assert handle["frequency_retained_complex_208"].shape == (2, 208)
        assert handle["frequency_reference_retained_complex_208"].shape == (2, 208)
        assert handle["frequency_contact_impedance_16"].shape == (2, 16)
        assert handle["frequency_reference_contact_impedance_16"].shape == (2, 16)
        np.testing.assert_allclose(
            handle["frequency_contact_impedance_16"][1, :3],
            [0.1, 0.1, 0.005],
        )
        np.testing.assert_allclose(
            handle["frequency_reference_contact_impedance_16"][1, :3],
            [0.005, 0.005, 0.005],
        )
        retained = handle["retained_indices_208"][:]
        np.testing.assert_allclose(
            handle["frequency_retained_complex_208"][0],
            handle["frequency_raw_complex_256"][0].reshape(-1)[retained],
        )

    label = json.loads((tmp_path / "ecd-cwr-sim-mf.label.json").read_text("utf-8"))
    assert label["frequency_hz"] == [10_000.0, 50_000.0]
    assert label["frequency_contact_impedance_multipliers"] == [1.0, 0.5]


def test_ecd_cwr_global_open_uses_finite_stable_contact_impedance(tmp_path):
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult
    from eit_app.ecd_cwr_simulation import run_ecd_cwr_simulation_request

    captured = {"requests": []}

    def fake_execute(request, progress_cb=None):
        captured["requests"].append(request)
        return ForwardSolverResult(
            boundary_voltages=np.ones(256, dtype=np.complex64),
            homogeneous_voltages=np.ones(256, dtype=np.complex64),
            ground_truth_conductivity=np.ones(4, dtype=np.float32),
            node_coords=np.zeros((4, 2), dtype=np.float32),
            cell_connectivity=np.zeros((1, 4), dtype=np.int32),
            n_elements=1,
            n_measurements=256,
            forward_model_config={"mock": True},
        )

    payload = _request_payload(tmp_path, scenario_id="ecd-cwr-sim-global-open")
    payload["scenario"]["contact_impedance"] = {
        "label": "zc_open",
        "multiplier": "Infinity",
    }
    payload["scenario"]["fault_mode"] = "global"
    payload["scenario"]["fault_electrodes"] = list(range(16))

    run_ecd_cwr_simulation_request(payload, execute_forward=fake_execute)

    target_cfg = captured["requests"][0].forward_model_config
    reference_cfg = captured["requests"][1].forward_model_config
    assert target_cfg["contact_impedance"] == [1000.0] * 16
    assert reference_cfg["contact_impedance"] == [0.01] * 16
    label = json.loads(
        (tmp_path / "ecd-cwr-sim-global-open.label.json").read_text("utf-8")
    )
    assert label["contact_impedance"] == [1000.0] * 16
