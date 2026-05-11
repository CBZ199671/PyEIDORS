from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from eit_app.controllers.batch_reconstruction_controller import (
    BatchReconstructionRequest,
    _build_request,
)
from eit_app.models.frame_model import FrameData
from eit_app.models.reconstruction_methods import (
    CANONICAL_SINGLE_STEP_LAMBDA_EFF,
    DATABASE_RECONSTRUCTION_METHODS,
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
    assert "debug_fine_mesh_noser" in methods
    assert "gn-absolute" in methods
    assert not database_method_uses_iterations("noser_rm")
    assert not database_method_uses_iterations("laplace_rm")
    assert not database_method_uses_iterations("curvature_rm")
    assert database_method_uses_iterations("gn-absolute")


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
    assert request.metadata["n_elec"] == 8
    assert request.metadata["difference_mode"] == "normalized"
    assert request.metadata["simulation_inverse_route"] == "curvature_rm"
    assert request.metadata["rm_regularization"] == "curvature"
    assert request.metadata["compute_precision"] == "float32"


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
