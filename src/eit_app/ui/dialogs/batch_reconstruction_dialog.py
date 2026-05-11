"""Batch reconstruction dialog.

Collects input folder, output folder, algorithm, optional reference
frame, and save options, then delegates to BatchReconstructionController.
Shows live progress via a QProgressBar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.models.reconstruction_methods import (
    CANONICAL_SINGLE_STEP_LAMBDA_EFF,
    DATABASE_RECONSTRUCTION_METHODS,
    database_method_requires_reference,
    database_method_supports_custom_lambda_eff,
    database_method_uses_iterations,
    database_method_uses_locked_lambda_eff,
    normalize_database_reconstruction_method,
)
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role


class BatchReconstructionDialog(QDialog):
    """Dialog that builds a batch reconstruction job and delegates execution.

    Signal:
        start_requested(dict) — emitted when user clicks Run with the full
        configuration. Parent wires this to BatchReconstructionController.
    """

    start_requested = Signal(dict)
    cancel_requested = Signal()

    def __init__(
        self,
        *,
        default_input: Path | None = None,
        default_output: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(760)
        self.resize(820, 640)
        self._default_input = default_input
        self._default_output = default_output
        self._is_running = False
        self._editable_alpha_value = 1.0
        self._custom_lambda_eff_value = CANONICAL_SINGLE_STEP_LAMBDA_EFF
        self._last_output_folder: str | None = None
        # Cached "finished" state so language switch re-renders the summary
        # line in the active locale.
        self._finished_state: tuple[str, int, int] | None = (
            None  # (tone, succeeded, failed)
        )
        self._progress_cache: tuple[int, int, str] | None = (
            None  # (current, total, raw_msg)
        )
        # ETA tracking — stamped the moment the first frame arrives
        # after _set_running(True), and used to estimate remaining
        # time off the rate so far.  `_progress_baseline` handles the
        # case where the controller starts emitting current>0 before
        # the dialog knows (e.g. fast folder scans).
        self._run_started_at: float | None = None
        self._progress_baseline: int = 0
        self._build_ui()
        self._connect_signals()
        self._update_algorithm_state()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(14)

        # Header card
        header = QWidget()
        header.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #1f5d8b, stop:1 #2a6fa0);"
            " border-radius: 10px;"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 14, 18, 14)
        header_layout.setSpacing(4)

        self._title_label = QLabel("")
        self._title_label.setStyleSheet(
            "background: transparent; color: #ffffff;"
            " font-size: 17px; font-weight: 700; border: none;"
        )
        header_layout.addWidget(self._title_label)

        self._subtitle_label = QLabel("")
        self._subtitle_label.setWordWrap(True)
        self._subtitle_label.setStyleSheet(
            "background: transparent; color: #dbe8f4; font-size: 12px; border: none;"
        )
        header_layout.addWidget(self._subtitle_label)
        root.addWidget(header)

        root.addWidget(self._build_folders_section())
        root.addWidget(self._build_algorithm_section())
        root.addWidget(self._build_output_section())
        root.addWidget(self._build_progress_section())

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self._close_btn = QPushButton("")
        set_button_role(self._close_btn, "subtle")
        btn_row.addWidget(self._close_btn)

        self._open_output_btn = QPushButton("")
        set_button_role(self._open_output_btn, "success")
        self._open_output_btn.setVisible(False)
        self._open_output_btn.setMinimumWidth(170)
        btn_row.addWidget(self._open_output_btn)

        self._cancel_btn = QPushButton("")
        set_button_role(self._cancel_btn, "danger")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setMinimumWidth(130)
        btn_row.addWidget(self._cancel_btn)

        self._run_btn = QPushButton("")
        set_button_role(self._run_btn, "primary")
        self._run_btn.setMinimumWidth(130)
        btn_row.addWidget(self._run_btn)

        root.addLayout(btn_row)

    def _build_folders_section(self) -> QWidget:
        self._folders_box = QGroupBox("")  # retranslated
        layout = QFormLayout(self._folders_box)
        layout.setSpacing(8)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._input_edit = QLineEdit()
        if self._default_input:
            self._input_edit.setText(str(self._default_input))
        self._input_browse_btn = QPushButton("")
        set_button_role(self._input_browse_btn, "subtle")
        self._input_browse_btn.setMinimumWidth(90)
        in_row = QHBoxLayout()
        in_row.setContentsMargins(0, 0, 0, 0)
        in_row.setSpacing(6)
        in_row.addWidget(self._input_edit, 1)
        in_row.addWidget(self._input_browse_btn)
        in_w = QWidget()
        in_w.setLayout(in_row)
        self._lbl_input = QLabel("")
        layout.addRow(self._lbl_input, in_w)

        self._output_edit = QLineEdit()
        if self._default_output:
            self._output_edit.setText(str(self._default_output))
        else:
            # Fall back to <app>/results for convenience
            default_root = Path.cwd() / "results"
            try:
                default_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            self._output_edit.setText(str(default_root))
        self._output_browse_btn = QPushButton("")
        set_button_role(self._output_browse_btn, "subtle")
        self._output_browse_btn.setMinimumWidth(90)
        out_row = QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0)
        out_row.setSpacing(6)
        out_row.addWidget(self._output_edit, 1)
        out_row.addWidget(self._output_browse_btn)
        out_w = QWidget()
        out_w.setLayout(out_row)
        self._lbl_output = QLabel("")
        layout.addRow(self._lbl_output, out_w)

        return self._folders_box

    def _build_algorithm_section(self) -> QWidget:
        self._algo_box = QGroupBox("")  # retranslated
        layout = QFormLayout(self._algo_box)
        layout.setSpacing(8)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._algo_combo = AutoCloseComboBox()
        for option in DATABASE_RECONSTRUCTION_METHODS:
            self._algo_combo.addItem(option.label, option.method)
        self._lbl_method = QLabel("")
        layout.addRow(self._lbl_method, self._algo_combo)

        self._use_part_combo = AutoCloseComboBox()
        self._use_part_combo.addItems(["real", "imag", "mag"])
        self._lbl_part = QLabel("")
        layout.addRow(self._lbl_part, self._use_part_combo)

        self._alpha_spin = QDoubleSpinBox()
        self._alpha_spin.setRange(0.0001, 1000.0)
        self._alpha_spin.setValue(1.0)
        self._alpha_spin.setDecimals(4)
        self._alpha_spin.setSingleStep(0.1)
        self._alpha_spin.valueChanged.connect(self._remember_alpha_value)
        self._lbl_alpha = QLabel("")
        layout.addRow(self._lbl_alpha, self._alpha_spin)

        self._custom_lambda_check = QCheckBox("")
        self._custom_lambda_check.toggled.connect(self._on_custom_lambda_toggled)
        layout.addRow(self._custom_lambda_check)

        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(1, 200)
        self._iter_spin.setValue(10)
        self._lbl_iter = QLabel("")
        layout.addRow(self._lbl_iter, self._iter_spin)

        # Reference frame (only for difference methods)
        self._ref_edit = QLineEdit()
        self._ref_browse_btn = QPushButton("")
        set_button_role(self._ref_browse_btn, "subtle")
        self._ref_browse_btn.setMinimumWidth(90)
        ref_row = QHBoxLayout()
        ref_row.setContentsMargins(0, 0, 0, 0)
        ref_row.setSpacing(6)
        ref_row.addWidget(self._ref_edit, 1)
        ref_row.addWidget(self._ref_browse_btn)
        self._ref_row_w = QWidget()
        self._ref_row_w.setLayout(ref_row)
        self._ref_row_label = QLabel("")
        layout.addRow(self._ref_row_label, self._ref_row_w)

        return self._algo_box

    def _build_output_section(self) -> QWidget:
        self._outputs_box = QGroupBox("")  # retranslated
        layout = QVBoxLayout(self._outputs_box)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(6)

        self._save_recon_check = QCheckBox("")
        self._save_recon_check.setChecked(True)
        layout.addWidget(self._save_recon_check)

        self._save_voltage_check = QCheckBox("")
        self._save_voltage_check.setChecked(True)
        layout.addWidget(self._save_voltage_check)

        return self._outputs_box

    def _build_progress_section(self) -> QWidget:
        self._progress_box = QGroupBox("")  # retranslated
        layout = QVBoxLayout(self._progress_box)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(6)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setMinimumHeight(22)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            "color: #5b6573; font-size: 12px;"
            " background: transparent; padding: 4px 2px;"
        )
        self._progress_label.setWordWrap(True)
        layout.addWidget(self._progress_label)

        return self._progress_box

    # ---- Signals & handlers ----

    def _connect_signals(self) -> None:
        self._input_browse_btn.clicked.connect(self._on_browse_input)
        self._output_browse_btn.clicked.connect(self._on_browse_output)
        self._ref_browse_btn.clicked.connect(self._on_browse_ref)
        self._algo_combo.currentIndexChanged.connect(self._update_algorithm_state)
        self._input_edit.textChanged.connect(self._update_run_enabled)
        self._output_edit.textChanged.connect(self._update_run_enabled)
        self._ref_edit.textChanged.connect(self._update_run_enabled)

        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._close_btn.clicked.connect(self.reject)
        self._open_output_btn.clicked.connect(self._on_open_output_folder)
        self._update_run_enabled()

    def _update_reference_requirement(self, *args) -> None:
        self._update_algorithm_state()

    def _update_algorithm_state(self, *args) -> None:
        method = self._current_method()
        needs_ref = database_method_requires_reference(method)
        self._ref_row_label.setEnabled(needs_ref)
        self._ref_row_w.setEnabled(needs_ref)
        self._update_hyperparameter_control(method)
        self._update_iteration_control(method)
        self._update_run_enabled()

    def _update_run_enabled(self, *args) -> None:
        if self._is_running:
            return
        has_input = bool(self._input_edit.text().strip())
        has_output = bool(self._output_edit.text().strip())
        needs_ref = database_method_requires_reference(self._current_method())
        has_ref = bool(self._ref_edit.text().strip())
        ok = has_input and has_output and (not needs_ref or has_ref)
        self._run_btn.setEnabled(ok)

    def _current_method(self) -> str:
        data = self._algo_combo.currentData()
        if data is None:
            data = self._algo_combo.currentText()
        return normalize_database_reconstruction_method(str(data))

    def _remember_alpha_value(self, value: float) -> None:
        method = self._current_method()
        if (
            database_method_supports_custom_lambda_eff(method)
            and self._custom_lambda_check.isChecked()
        ):
            self._custom_lambda_eff_value = float(value)
        elif not database_method_uses_locked_lambda_eff(method):
            self._editable_alpha_value = float(value)

    def _on_custom_lambda_toggled(self, checked: bool) -> None:
        method = self._current_method()
        if not checked and database_method_supports_custom_lambda_eff(method):
            self._custom_lambda_eff_value = float(self._alpha_spin.value())
        self._update_hyperparameter_control(method)

    def _update_hyperparameter_control(self, method: str) -> None:
        locked_lambda = database_method_uses_locked_lambda_eff(method)
        custom_available = database_method_supports_custom_lambda_eff(method)
        custom_enabled = custom_available and self._custom_lambda_check.isChecked()
        blocked = self._alpha_spin.blockSignals(True)
        try:
            self._custom_lambda_check.setVisible(custom_available)
            self._custom_lambda_check.setEnabled(
                custom_available and not self._is_running
            )
            if custom_available:
                self._custom_lambda_check.setToolTip(t("dlg.batch.custom_lambda_tip"))
            if locked_lambda and not custom_enabled:
                self._alpha_spin.setValue(CANONICAL_SINGLE_STEP_LAMBDA_EFF)
                self._alpha_spin.setEnabled(False)
                self._alpha_spin.setToolTip(t("dlg.batch.lambda_locked_tip"))
                self._lbl_alpha.setText(t("dlg.batch.lambda_eff_label"))
                self._lbl_alpha.setToolTip(t("dlg.batch.lambda_locked_tip"))
            else:
                if custom_enabled:
                    self._alpha_spin.setValue(self._custom_lambda_eff_value)
                    self._alpha_spin.setToolTip(t("dlg.batch.custom_lambda_tip"))
                    self._lbl_alpha.setText(t("dlg.batch.lambda_eff_label"))
                    self._lbl_alpha.setToolTip(t("dlg.batch.custom_lambda_tip"))
                else:
                    self._alpha_spin.setValue(self._editable_alpha_value)
                    self._alpha_spin.setToolTip("")
                    self._lbl_alpha.setText(t("dlg.batch.alpha_label"))
                    self._lbl_alpha.setToolTip("")
                self._alpha_spin.setEnabled(not self._is_running)
        finally:
            self._alpha_spin.blockSignals(blocked)

    def _update_iteration_control(self, method: str) -> None:
        show_iterations = database_method_uses_iterations(method)
        self._lbl_iter.setVisible(show_iterations)
        self._iter_spin.setVisible(show_iterations)
        self._lbl_iter.setEnabled(show_iterations and not self._is_running)
        self._iter_spin.setEnabled(show_iterations and not self._is_running)

    def _on_browse_input(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            t("dlg.batch.file_dialog.input"),
            self._input_edit.text() or str(Path.home()),
        )
        if path:
            self._input_edit.setText(path)

    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            t("dlg.batch.file_dialog.output"),
            self._output_edit.text() or str(Path.home()),
        )
        if path:
            self._output_edit.setText(path)

    def _on_browse_ref(self) -> None:
        start_dir = self._input_edit.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("dlg.batch.file_dialog.ref"),
            start_dir,
            t("dlg.batch.file_dialog.csv_filter"),
        )
        if path:
            self._ref_edit.setText(path)

    def _on_run(self) -> None:
        method = self._current_method()
        label = self._algo_combo.currentText()
        needs_ref = database_method_requires_reference(method)
        ref_csv = self._ref_edit.text().strip() or None
        self._last_output_folder = self._output_edit.text().strip() or None
        self._open_output_btn.setVisible(False)
        custom_lambda_enabled = (
            database_method_supports_custom_lambda_eff(method)
            and self._custom_lambda_check.isChecked()
        )
        config: dict[str, Any] = {
            "input_folder": self._input_edit.text().strip(),
            "output_folder": self._output_edit.text().strip(),
            "method": method,
            "method_label": label,
            "reference_csv": ref_csv if needs_ref else None,
            "use_part": self._use_part_combo.currentText(),
            "regularization_alpha": self._alpha_spin.value(),
            "lambda_eff_custom_enabled": custom_lambda_enabled,
            "custom_lambda_eff": self._alpha_spin.value()
            if custom_lambda_enabled
            else CANONICAL_SINGLE_STEP_LAMBDA_EFF,
            "max_iterations": self._iter_spin.value(),
            "save_recon_image": self._save_recon_check.isChecked(),
            "save_voltage_fit": self._save_voltage_check.isChecked(),
        }
        self._set_running(True)
        self.start_requested.emit(config)

    def _on_cancel(self) -> None:
        self._progress_label.setText(t("dlg.batch.cancelling"))
        self.cancel_requested.emit()

    def _set_running(self, running: bool) -> None:
        self._is_running = running
        if running:
            # Reset ETA tracking for a fresh run.  Stamp the moment
            # set_running(True) fires — the first set_progress call
            # may carry a non-zero current if the controller processed
            # a few items before emitting.
            import time as _time

            self._run_started_at = _time.monotonic()
            self._progress_baseline = 0
        else:
            self._run_started_at = None
        self._run_btn.setVisible(not running)
        self._cancel_btn.setVisible(running)
        for w in (
            self._input_edit,
            self._input_browse_btn,
            self._output_edit,
            self._output_browse_btn,
            self._ref_edit,
            self._ref_browse_btn,
            self._algo_combo,
            self._use_part_combo,
            self._alpha_spin,
            self._custom_lambda_check,
            self._iter_spin,
            self._save_recon_check,
            self._save_voltage_check,
        ):
            w.setEnabled(not running)
        if not running:
            self._update_algorithm_state()

    # ---- Progress API (called by parent wiring controller signals) ----

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
        self._progress_cache = (current, total, message)
        if message:
            # Caller supplied a human-readable string (error, cancelling,
            # final summary) — show it verbatim without decorating ETA.
            self._progress_label.setText(message)
            return
        eta_text = self._format_eta(current, total)
        if eta_text is None:
            self._progress_label.setText(
                t("dlg.batch.progress_default", current=current, total=total)
            )
        else:
            self._progress_label.setText(
                t(
                    "dlg.batch.progress_with_eta",
                    current=current,
                    total=total,
                    eta=eta_text,
                )
            )

    def _format_eta(self, current: int, total: int) -> str | None:
        """Return a localized 'X remaining' string, or None when
        ETA is not yet estimable.

        Uses a rolling rate = (frames done since baseline) / elapsed
        so the estimate stabilises as more items complete.  The first
        few seconds show no ETA rather than wildly fluctuating
        numbers — we need at least 2 completions and 1s of elapsed
        time to produce a meaningful figure.
        """
        import time as _time

        if self._run_started_at is None or total <= 0 or current >= total:
            return None
        done_since_start = max(0, current - self._progress_baseline)
        elapsed = _time.monotonic() - self._run_started_at
        if done_since_start < 2 or elapsed < 1.0:
            return None
        rate = done_since_start / elapsed  # items per second
        if rate <= 0:
            return None
        remaining_items = total - current
        remaining_sec = int(round(remaining_items / rate))
        if remaining_sec <= 0:
            return None

        if remaining_sec < 60:
            return t("dlg.batch.eta_seconds", seconds=remaining_sec)
        if remaining_sec < 3600:
            minutes, seconds = divmod(remaining_sec, 60)
            return t("dlg.batch.eta_minutes", minutes=minutes, seconds=seconds)
        hours, minutes = divmod(remaining_sec // 60, 60)
        return t("dlg.batch.eta_hours", hours=hours, minutes=minutes)

    def on_finished(self, succeeded: int, failed: int) -> None:
        self._set_running(False)
        if succeeded > 0 and failed == 0:
            tone = "color: #1b7947;"
            self._finished_state = ("ok", succeeded, failed)
        elif failed > 0 and succeeded > 0:
            tone = "color: #a06a10;"
            self._finished_state = ("mixed", succeeded, failed)
        else:
            tone = "color: #a04040;"
            self._finished_state = ("fail", succeeded, failed)
        self._progress_label.setStyleSheet(
            f"{tone} font-size: 12px; font-weight: 600;"
            " background: transparent; padding: 4px 2px;"
        )
        self._apply_finished_text()
        self._run_btn.setEnabled(True)
        if self._last_output_folder and Path(self._last_output_folder).exists():
            self._open_output_btn.setVisible(True)

    def on_error(self, message: str) -> None:
        self._set_running(False)
        self._finished_state = ("error", 0, 0)
        self._error_message = message
        self._progress_label.setStyleSheet(
            "color: #a04040; font-size: 12px; font-weight: 600;"
            " background: transparent; padding: 4px 2px;"
        )
        self._progress_label.setText(t("dlg.batch.error", message=message))

    def _apply_finished_text(self) -> None:
        if self._finished_state is None:
            return
        kind, succeeded, failed = self._finished_state
        key_map = {
            "ok": "dlg.batch.finished_ok",
            "mixed": "dlg.batch.finished_mixed",
            "fail": "dlg.batch.finished_fail",
        }
        key = key_map.get(kind)
        if key:
            self._progress_label.setText(t(key, succeeded=succeeded, failed=failed))

    def _on_open_output_folder(self) -> None:
        if not self._last_output_folder:
            return
        from eit_app.ui.main_window import _open_folder_in_file_manager

        _open_folder_in_file_manager(self._last_output_folder)

    # ── i18n ──

    def _retranslate(self) -> None:
        self.setWindowTitle(t("dlg.batch.title"))
        self._title_label.setText(t("dlg.batch.heading"))
        self._subtitle_label.setText(t("dlg.batch.subtitle"))
        self._close_btn.setText(t("dlg.batch.close_button"))
        self._open_output_btn.setText(t("dlg.batch.open_output_button"))
        self._cancel_btn.setText(t("dlg.batch.cancel_button"))
        self._run_btn.setText(t("dlg.batch.run_button"))

        self._folders_box.setTitle(t("dlg.batch.folders_group"))
        self._input_edit.setPlaceholderText(t("dlg.batch.input_placeholder"))
        self._input_browse_btn.setText(t("dlg.batch.browse_button"))
        self._lbl_input.setText(t("dlg.batch.input_label"))
        self._output_edit.setPlaceholderText(t("dlg.batch.output_placeholder"))
        self._output_browse_btn.setText(t("dlg.batch.browse_button"))
        self._lbl_output.setText(t("dlg.batch.output_label"))

        self._algo_box.setTitle(t("dlg.batch.algo_params_group"))
        self._lbl_method.setText(t("dlg.batch.method_label"))
        self._lbl_part.setText(t("dlg.batch.part_label"))
        self._lbl_iter.setText(t("dlg.batch.iter_label"))
        self._custom_lambda_check.setText(t("dlg.batch.custom_lambda_check"))
        self._ref_edit.setPlaceholderText(t("dlg.batch.ref_placeholder"))
        self._ref_browse_btn.setText(t("dlg.batch.ref_browse_button"))
        self._ref_row_label.setText(t("dlg.batch.ref_label"))

        self._outputs_box.setTitle(t("dlg.batch.outputs_group"))
        self._save_recon_check.setText(t("dlg.batch.save_image_check"))
        self._save_voltage_check.setText(t("dlg.batch.save_voltage_check"))

        self._progress_box.setTitle(t("dlg.batch.progress_group"))

        # Re-render the dynamic progress line / terminal state in the new
        # locale so mid-run language switches don't strand stale copy.
        if self._finished_state is not None:
            if self._finished_state[0] == "error":
                self._progress_label.setText(
                    t("dlg.batch.error", message=getattr(self, "_error_message", ""))
                )
            else:
                self._apply_finished_text()
        elif self._progress_cache is not None:
            current, total, message = self._progress_cache
            self._progress_label.setText(
                message or t("dlg.batch.progress_default", current=current, total=total)
            )
        else:
            self._progress_label.setText(t("dlg.batch.ready"))
        self._update_algorithm_state()
