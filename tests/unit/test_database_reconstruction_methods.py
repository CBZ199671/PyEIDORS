from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from eit_app.controllers import reconstruction_controller as rc
from eit_app.controllers.batch_reconstruction_controller import (
    BatchReconstructionRequest,
    _build_request,
    _display_float_array,
    _display_int_array,
    _save_outputs,
)
from eit_app.models.frame_model import FrameData
from eit_app.models.reconstruction_methods import (
    CANONICAL_SINGLE_STEP_LAMBDA_EFF,
    DATABASE_RECONSTRUCTION_METHODS,
    PSEUDO3D_NOSER_RM_METHOD,
    database_method_uses_iterations,
    prepare_database_reconstruction_method,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _select_method(combo, method: str) -> None:
    for index in range(combo.count()):
        if combo.itemData(index) == method:
            combo.setCurrentIndex(index)
            return
    raise AssertionError(f"method not found: {method}")


def test_v118_database_method_catalog_exposes_rm_and_absolute_routes() -> None:
    methods = [option.method for option in DATABASE_RECONSTRUCTION_METHODS]
    assert methods[:3] == ["noser_rm", "laplace_rm", "curvature_rm"]
    assert PSEUDO3D_NOSER_RM_METHOD in methods
    assert "debug_fine_mesh_noser" not in methods
    assert "gn-absolute" in methods
    assert not database_method_uses_iterations("noser_rm")
    assert not database_method_uses_iterations("laplace_rm")
    assert not database_method_uses_iterations("curvature_rm")
    assert database_method_uses_iterations("gn-absolute")


def test_database_pseudo3d_route_prepares_2d_inverse_with_3d_display_metadata() -> None:
    prepared = prepare_database_reconstruction_method(
        PSEUDO3D_NOSER_RM_METHOD,
        regularization_alpha=0.25,
        max_iterations=20,
        custom_lambda_eff_enabled=True,
        metadata={
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "height": 0.16,
            "radius": 0.18,
            "drive_mode": "total_current",
            "drive_value": 100.0e-6,
            "petsc_device": "cuda",
            "device": "cuda",
            "forward_backend": "cuda_structured",
            "acceleration_profile": "gpu3d",
        },
    )

    assert prepared.method == "gn-difference"
    assert prepared.max_iterations == 1
    assert prepared.regularization_alpha == pytest.approx(0.25)
    assert prepared.metadata["simulation_inverse_route"] == PSEUDO3D_NOSER_RM_METHOD
    assert prepared.metadata["rm_regularization"] == "noser"
    assert prepared.metadata["rm_form"] == "measurement"
    assert prepared.metadata["mesh_dimension"] == 2
    assert prepared.metadata["n_elec"] == 8
    assert prepared.metadata["n_rings"] == 1
    assert prepared.metadata["pseudo3d_output"] is True
    assert prepared.metadata["pseudo3d_layered_output"] is True
    assert prepared.metadata["pseudo3d_source_mesh_dimension"] == 3
    assert prepared.metadata["pseudo3d_source_n_elec"] == 8
    assert prepared.metadata["pseudo3d_source_n_rings"] == 2
    assert prepared.metadata["pseudo3d_layer_count"] == 2
    assert prepared.metadata["pseudo3d_layer_n_elec"] == 8
    assert prepared.metadata["pseudo3d_display_mesh_dimension"] == 3
    assert prepared.metadata["pseudo3d_display_height"] == pytest.approx(0.16)
    assert prepared.metadata["petsc_device"] == "cpu"
    assert prepared.metadata["device"] == "cpu"
    assert prepared.metadata["rm_device"] == "cpu"
    assert prepared.metadata["forward_backend"] == "dolfinx"
    assert prepared.metadata["acceleration_profile"] == "default"
    assert prepared.metadata["pseudo3d_source_geometry"]["petsc_device"] == "cuda"


def test_v118_database_rm_route_prepares_normalized_single_step_request() -> None:
    prepared = prepare_database_reconstruction_method(
        "laplace_rm",
        regularization_alpha=7.0,
        max_iterations=99,
        metadata={"compute_precision": "float32"},
    )

    assert prepared.method == "gn-difference"
    assert prepared.max_iterations == 1
    assert prepared.regularization_alpha == CANONICAL_SINGLE_STEP_LAMBDA_EFF
    assert prepared.metadata["difference_mode"] == "normalized"
    assert prepared.metadata["reconstruction_runtime"] == "single_step_cached"
    assert prepared.metadata["simulation_inverse_route"] == "laplace_rm"
    assert prepared.metadata["simulation_inverse_route_kind"] == "rm"
    assert prepared.metadata["rm_regularization"] == "laplace"
    assert prepared.metadata["rm_form"] == "param"
    assert prepared.metadata["lambda_eff_custom_enabled"] is False
    assert prepared.metadata["compute_precision"] == "float32"


def test_v118_database_custom_lambda_marks_cold_rm_rebuild() -> None:
    prepared = prepare_database_reconstruction_method(
        "noser_rm",
        regularization_alpha=0.123,
        max_iterations=10,
        custom_lambda_eff_enabled=True,
        metadata={},
    )

    assert prepared.method == "gn-difference"
    assert prepared.regularization_alpha == 0.123
    assert prepared.metadata["difference_lambda"] == 0.123
    assert prepared.metadata["lambda_eff_custom_enabled"] is True
    assert prepared.metadata["hyperparameter_effective_source"] == "custom_rm_rebuild"
    assert prepared.metadata["rm_rebuild_required_by_custom_lambda"] is True


def test_v118_batch_build_request_resolves_rm_route_to_controller_metadata(
    tmp_path: Path,
) -> None:
    batch = BatchReconstructionRequest(
        input_folder=tmp_path,
        output_folder=tmp_path,
        method="curvature_rm",
        method_label="Curvature RM",
        reference_csv=tmp_path / "ref.csv",
        use_part="real",
        regularization_alpha=CANONICAL_SINGLE_STEP_LAMBDA_EFF,
        max_iterations=44,
        save_recon_image=False,
        save_voltage_fit=False,
        mesh_dimension=3,
        mesh_refinement=0.2,
        metadata={"n_elec": 8, "compute_precision": "float32"},
    )
    frame = FrameData(
        real=np.array([1.0, 2.0, 3.0]),
        imag=np.zeros(3),
        timestamp=0.0,
        frame_index=0,
    )

    request = _build_request(batch, reference=frame, target=frame)

    assert request.method == "gn-difference"
    assert request.max_iterations == 1
    assert request.mesh_dimension == 3
    assert request.mesh_refinement == 0.2
    assert request.metadata["n_elec"] == 8
    assert request.metadata["mesh_dimension"] == 3
    assert request.metadata["mesh_size"] == 0.2
    assert request.metadata["difference_mode"] == "normalized"
    assert request.metadata["simulation_inverse_route"] == "curvature_rm"
    assert request.metadata["rm_regularization"] == "curvature"
    assert request.metadata["compute_precision"] == "float32"


def test_batch_pseudo3d_request_uses_2d_inverse_dimension(tmp_path: Path) -> None:
    batch = BatchReconstructionRequest(
        input_folder=tmp_path,
        output_folder=tmp_path,
        method=PSEUDO3D_NOSER_RM_METHOD,
        method_label="Pseudo-3D",
        reference_csv=tmp_path / "ref.csv",
        use_part="real",
        regularization_alpha=CANONICAL_SINGLE_STEP_LAMBDA_EFF,
        max_iterations=44,
        save_recon_image=False,
        save_voltage_fit=False,
        mesh_dimension=3,
        mesh_refinement=0.2,
        metadata={"n_elec": 8, "n_rings": 2, "height": 0.16},
    )
    frame = FrameData(
        real=np.array([1.0, 2.0, 3.0]),
        imag=np.zeros(3),
        timestamp=0.0,
        frame_index=0,
    )

    request = _build_request(batch, reference=frame, target=frame)

    assert request.method == "gn-difference"
    assert request.mesh_dimension == 2
    assert request.metadata["mesh_dimension"] == 2
    assert request.metadata["n_elec"] == 8
    assert request.metadata["n_rings"] == 1
    assert request.metadata["pseudo3d_output"] is True
    assert request.metadata["pseudo3d_layered_output"] is True
    assert request.metadata["pseudo3d_source_mesh_dimension"] == 3
    assert request.metadata["pseudo3d_source_n_rings"] == 2
    assert request.metadata["pseudo3d_layer_count"] == 2
    assert request.metadata["pseudo3d_layer_n_elec"] == 8


def test_pseudo3d_result_extrudes_2d_triangle_into_tetra_layers() -> None:
    result = rc.ReconstructionResult(
        conductivity=np.array([2.0], dtype=np.float32),
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        metadata={
            "pseudo3d_output": True,
            "pseudo3d_display_layers": 3,
            "pseudo3d_display_height": 0.2,
        },
    )

    extruded = rc._maybe_apply_pseudo3d_result(result)

    assert extruded.metadata["pseudo3d_extruded"] is True
    assert extruded.metadata["mesh_dimension"] == 3
    assert extruded.node_coords.shape == (9, 3)
    assert extruded.node_coords.dtype == np.dtype(np.float32)
    assert extruded.cell_connectivity.shape == (6, 4)
    assert extruded.conductivity.shape == (6,)
    np.testing.assert_allclose(extruded.conductivity, np.full(6, 2.0))
    assert np.ptp(extruded.node_coords[:, 2]) == pytest.approx(0.2)


def test_v629_pseudo3d_layered_result_interpolates_two_2d_reconstructions() -> None:
    result = rc.ReconstructionResult(
        conductivity=np.array([[1.0], [3.0]], dtype=np.float32),
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        metadata={
            "pseudo3d_output": True,
            "pseudo3d_layered_output": True,
            "pseudo3d_display_layers": 3,
            "pseudo3d_display_height": 0.2,
            "pseudo3d_source_n_rings": 2,
            "electrode_level_fractions": (0.0, 1.0),
        },
    )

    extruded = rc._maybe_apply_pseudo3d_result(result)

    assert extruded.metadata["pseudo3d_layered_extruded"] is True
    assert extruded.metadata["pseudo3d_interpolation"] == "linear_z_between_2d_layers"
    assert extruded.metadata["mesh_dimension"] == 3
    assert extruded.cell_connectivity.shape == (6, 4)
    assert extruded.conductivity.shape == (6,)
    np.testing.assert_allclose(extruded.conductivity, [1.5, 1.5, 1.5, 2.5, 2.5, 2.5])


def test_v629_pseudo3d_layered_request_runs_one_2d_inverse_per_source_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare_database_reconstruction_method(
        PSEUDO3D_NOSER_RM_METHOD,
        regularization_alpha=CANONICAL_SINGLE_STEP_LAMBDA_EFF,
        max_iterations=10,
        metadata={
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "height": 0.2,
            "radius": 0.18,
        },
    )
    n_meas = 208
    reference = FrameData(
        real=np.arange(n_meas, dtype=np.float64),
        imag=np.zeros(n_meas, dtype=np.float64),
        timestamp=0.0,
        frame_index=1,
    )
    target = FrameData(
        real=np.arange(n_meas, dtype=np.float64) + 1.0,
        imag=np.zeros(n_meas, dtype=np.float64),
        timestamp=0.0,
        frame_index=2,
    )
    request = rc.ReconstructionRequest(
        reference_frame=reference,
        target_frame=target,
        use_part="real",
        method=prepared.method,
        regularization_alpha=prepared.regularization_alpha,
        max_iterations=prepared.max_iterations,
        mesh_dimension=prepared.metadata["mesh_dimension"],
        mesh_refinement=0.2,
        metadata=prepared.metadata,
    )
    captured: list[rc.ReconstructionRequest] = []

    def fake_single_step(
        layer_request: rc.ReconstructionRequest,
        *,
        progress_cb=None,
    ) -> rc.ReconstructionResult:
        del progress_cb
        captured.append(layer_request)
        layer = int(layer_request.metadata["pseudo3d_layer_index"])
        return rc.ReconstructionResult(
            conductivity=np.array([1.0 + layer], dtype=np.float32),
            node_coords=np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
            cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
            measured=np.ones(layer_request.reference_frame.n_meas),
            simulated=np.ones(layer_request.reference_frame.n_meas),
            metadata=dict(layer_request.metadata),
        )

    monkeypatch.setattr(rc, "_run_single_step_cached_request", fake_single_step)

    result = rc.run_reconstruction_request(request)

    assert result.error_msg is None
    assert len(captured) == 2
    assert [call.reference_frame.n_meas for call in captured] == [40, 40]
    assert [call.target_frame.n_meas for call in captured] == [40, 40]
    assert [call.metadata["n_elec"] for call in captured] == [8, 8]
    assert [call.metadata["n_rings"] for call in captured] == [1, 1]
    assert all(call.mesh_dimension == 2 for call in captured)
    assert all(call.metadata["pseudo3d_output"] is False for call in captured)
    assert result.metadata["pseudo3d_layered_executed"] is True
    assert result.metadata["pseudo3d_layer_measurement_counts"] == [40, 40]
    assert result.metadata["mesh_dimension"] == 3
    assert result.measured is not None
    assert result.measured.shape == (208,)


def test_v360_batch_output_save_preserves_display_payload_dtype() -> None:
    source = inspect.getsource(_save_outputs)

    assert "np.asarray(result.conductivity, dtype=float)" not in source
    assert "np.asarray(result.node_coords, dtype=float)" not in source
    assert "np.asarray(result.measured, dtype=float)" not in source
    assert "np.asarray(result.simulated, dtype=float)" not in source
    assert "_display_float_array(result.conductivity)" in source
    assert "_display_int_array(result.cell_connectivity)" in source

    values = np.array([1.0, 2.0], dtype=np.float32)
    display_values = _display_float_array(values)

    assert display_values.dtype == np.dtype(np.float32)
    assert np.shares_memory(display_values, values)

    complex_values = np.array([1.0 + 1.5j, 2.0 + 2.5j], dtype=np.complex64)
    complex_display = _display_float_array(complex_values)

    assert complex_display.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(
        complex_display,
        np.array([1.0, 2.0], dtype=np.float32),
    )

    cells = np.array([[0, 1, 2]], dtype=np.int32)
    display_cells = _display_int_array(cells)

    assert display_cells.dtype == np.dtype(np.int32)
    assert np.shares_memory(display_cells, cells)


def test_v118_single_frame_dialog_hides_iterations_for_difference_methods() -> None:
    from eit_app.ui.dialogs.reconstruction_dialog import ReconstructionDialog

    app = _app()
    dialog = ReconstructionDialog(
        reference_entry={"frame_index": 0, "csv_path": "/tmp/ref.csv"},
        target_entry={"frame_index": 1, "csv_path": "/tmp/tgt.csv"},
    )
    dialog.show()
    app.processEvents()

    try:
        _select_method(dialog._algo_combo, "noser_rm")
        app.processEvents()
        assert dialog._iter_spin.isHidden()
        assert not dialog._alpha_spin.isEnabled()
        assert dialog._custom_lambda_check.isVisible()

        _select_method(dialog._algo_combo, "gn-absolute")
        app.processEvents()
        assert dialog._iter_spin.isVisible()
        assert dialog._iter_spin.isEnabled()
        assert dialog._alpha_spin.isEnabled()
        assert dialog._custom_lambda_check.isHidden()
    finally:
        dialog.close()


class _NoopSignal:
    def connect(self, _slot) -> None:
        return None


def test_v127_single_frame_dialog_keeps_settings_out_of_reconstruct_dialog() -> None:
    from eit_app.ui.dialogs.reconstruction_dialog import ReconstructionDialog

    app = _app()
    emitted: list[dict] = []
    dialog = ReconstructionDialog(
        reference_entry={"frame_index": 0, "csv_path": "/tmp/ref.csv"},
        target_entry={"frame_index": 1, "csv_path": "/tmp/tgt.csv"},
    )
    dialog.run_requested.connect(emitted.append)
    dialog.show()
    app.processEvents()

    try:
        assert not hasattr(dialog, "_settings_panel")
        dialog._on_run()
    finally:
        dialog.close()

    assert emitted
    config = emitted[0]
    assert "reconstruction_settings" not in config
    assert "mesh_dimension" not in config
    assert "mesh_refinement" not in config


def test_v127_database_tab_injects_standalone_reconstruction_settings() -> None:
    from eit_app.ui.database.database_tab import DatabaseTab

    app = _app()
    emitted: list[dict] = []
    tab = DatabaseTab(
        SimpleNamespace(
            query_sessions=lambda **_filters: [],
            is_shutting_down=False,
            session_added=_NoopSignal(),
            frame_added=_NoopSignal(),
            backfill_progress=_NoopSignal(),
            backfill_done=_NoopSignal(),
        )
    )
    tab.reconstruct_requested.connect(emitted.append)
    tab._reconstruction_settings_override = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.05,
        "mesh_size": 0.05,
        "n_elec": 32,
        "n_rings": 4,
        "stim_pattern": "{op}",
        "drive_value": 2.5e-5,
    }
    tab.show()
    app.processEvents()

    try:
        tab._emit_reconstruct_requested(
            {
                "method": "gn-absolute",
                "target_entry": {"frame_index": 1, "csv_path": "/tmp/tgt.csv"},
            }
        )
    finally:
        tab.close()

    assert emitted
    config = emitted[0]
    assert config["mesh_dimension"] == 3
    assert config["mesh_refinement"] == 0.05
    assert config["reconstruction_settings"]["n_elec"] == 32
    assert config["reconstruction_settings"]["n_rings"] == 4
    assert config["reconstruction_settings"]["stim_pattern"] == "{op}"
    assert config["reconstruction_settings"]["drive_value"] == 2.5e-5


def test_v127_reconstruction_settings_dialog_prefills_frame_metadata() -> None:
    from eit_app.ui.dialogs.reconstruction_settings_panel import (
        ReconstructionSettingsDialog,
        metadata_from_frame_entries,
    )

    app = _app()
    frame_meta = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.08,
        "n_elec": 24,
        "n_rings": 2,
        "stim_pattern": "{op}",
        "meas_pattern": "{ad}",
        "radius": 0.18,
        "height": 0.16,
        "drive_value": 2.5e-5,
    }
    metadata = metadata_from_frame_entries(
        {
            "frame_index": 0,
            "csv_path": "/tmp/ref.csv",
            "frame_metadata_json": json.dumps(frame_meta),
        },
        {
            "frame_index": 1,
            "csv_path": "/tmp/tgt.csv",
            "frame_metadata_json": json.dumps(frame_meta | {"n_elec": 32}),
        },
    )
    dialog = ReconstructionSettingsDialog(initial_metadata=metadata)
    dialog.show()
    app.processEvents()

    try:
        assert dialog.mesh_dimension() == 3
        assert dialog.metadata()["n_elec"] == 32
        dialog._panel._n_rings.setValue(4)
        dialog._panel._mesh_refinement.setValue(0.05)
        settings = dialog.metadata()
    finally:
        dialog.close()

    assert settings["mesh_dimension"] == 3
    assert settings["mesh_refinement"] == 0.05
    assert settings["n_elec"] == 32
    assert settings["n_rings"] == 4
    assert settings["stim_pattern"] == "{op}"
    assert settings["drive_value"] == 2.5e-5


def test_v127_batch_dialog_emits_external_advanced_settings(
    tmp_path: Path,
) -> None:
    from eit_app.ui.dialogs.batch_reconstruction_dialog import BatchReconstructionDialog

    app = _app()
    emitted: list[dict] = []
    (tmp_path / "session_metadata.yaml").write_text(
        "\n".join(
            [
                "mesh_dimension: 3",
                "mesh_refinement: 0.07",
                "n_elec: 16",
                "n_rings: 3",
                "electrode_layout: ring_major",
                "stim_pattern: '{ad}'",
                "meas_pattern: '{op}'",
                "radius: 0.18",
                "height: 0.16",
                "contact_impedance: 0.02",
            ]
        ),
        encoding="utf-8",
    )
    dialog = BatchReconstructionDialog(
        default_input=tmp_path,
        default_output=tmp_path,
        reconstruction_settings={
            "mesh_dimension": 3,
            "mesh_refinement": 0.07,
            "mesh_size": 0.07,
            "n_elec": 48,
            "n_rings": 3,
            "meas_pattern": "{op}",
            "contact_impedance": 0.02,
        },
    )
    dialog.start_requested.connect(emitted.append)
    dialog.show()
    app.processEvents()

    try:
        assert not hasattr(dialog, "_settings_panel")
        dialog._ref_edit.setText(str(tmp_path / "ref.csv"))
        dialog._on_run()
    finally:
        dialog.close()

    assert emitted
    config = emitted[0]
    assert config["mesh_dimension"] == 3
    assert config["mesh_refinement"] == 0.07
    assert config["reconstruction_settings"]["n_elec"] == 48
    assert config["reconstruction_settings"]["n_rings"] == 3
    assert config["reconstruction_settings"]["meas_pattern"] == "{op}"
    assert config["reconstruction_settings"]["contact_impedance"] == 0.02


def test_v118_batch_dialog_hides_iterations_for_difference_methods(
    tmp_path: Path,
) -> None:
    from eit_app.ui.dialogs.batch_reconstruction_dialog import (
        BatchReconstructionDialog,
    )

    app = _app()
    dialog = BatchReconstructionDialog(
        default_input=tmp_path / "input",
        default_output=tmp_path / "output",
    )
    dialog._ref_edit.setText(str(tmp_path / "ref.csv"))
    dialog.show()
    app.processEvents()

    try:
        _select_method(dialog._algo_combo, "laplace_rm")
        app.processEvents()
        assert dialog._iter_spin.isHidden()
        assert not dialog._alpha_spin.isEnabled()
        assert dialog._custom_lambda_check.isVisible()

        _select_method(dialog._algo_combo, "gn-absolute")
        app.processEvents()
        assert dialog._iter_spin.isVisible()
        assert dialog._iter_spin.isEnabled()
        assert dialog._ref_row_w.isEnabled() is False
    finally:
        dialog.close()
