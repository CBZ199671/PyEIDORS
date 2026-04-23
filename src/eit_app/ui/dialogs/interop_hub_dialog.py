"""Interop Hub dialog for EIDORS <-> PyEIDORS migration workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.interop import (
    EidorsEnvironment,
    EidorsExportJob,
    EidorsImportPreview,
    EidorsScriptCaptureService,
    InteropBundleExporter,
    InteropBundleImporter,
    LoadedBridgePackage,
)
from eit_app.interop.environment import to_posix_path
from eit_app.ui.path_explorer import pick_visual_path
from eit_app.ui.theme import (
    set_button_role,
    set_hint_text,
    set_subtle_value,
)


ExportSnapshotProvider = Callable[[], dict[str, dict[str, Any]]]
ImportApplyCallback = Callable[[str, LoadedBridgePackage], str]
SmokeValidateCallback = Callable[[LoadedBridgePackage], str]


# Stable identifiers for the status-panel rows.  The visible label follows
# the active UI language, but these ids stay constant so callers can keep
# addressing the same row across language switches.
_STATUS_ROW_IDS = ("matlab", "startup", "source", "bridge_package")


class InteropHubDialog(QDialog):
    """Standalone workbench for importing from and exporting to EIDORS."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        capture_service: EidorsScriptCaptureService | None = None,
        importer: InteropBundleImporter | None = None,
        exporter: InteropBundleExporter | None = None,
        export_snapshot_provider: ExportSnapshotProvider | None = None,
        apply_import_callback: ImportApplyCallback | None = None,
        smoke_validate_callback: SmokeValidateCallback | None = None,
    ) -> None:
        super().__init__(parent)
        # Title assigned in _retranslate() so it follows the active language.
        self.resize(1180, 820)

        self._capture_service = capture_service or EidorsScriptCaptureService()
        self._importer = importer or InteropBundleImporter()
        self._exporter = exporter or InteropBundleExporter()
        self._export_snapshot_provider = export_snapshot_provider
        self._apply_import_callback = apply_import_callback
        self._smoke_validate_callback = smoke_validate_callback

        self._environments: list[EidorsEnvironment] = []
        self._loaded_bundle: LoadedBridgePackage | None = None
        self._preview: EidorsImportPreview | None = None
        # Track which rows are in the "pending" sentinel state so
        # retranslation can keep them in that state rather than snapping
        # them to "未指定 / Not set" whenever the language flips.
        self._status_row_states: dict[str, str] = {
            row: "unspecified" for row in _STATUS_ROW_IDS
        }

        self._build_ui()
        self._load_profiles_into_list()
        self._refresh_manual_environment_state()
        self._refresh_source_status()

        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _make_path_row(
        self,
        line_edit: QLineEdit,
        *,
        title_key: str,
        mode: str,
        filter_key: str = "",
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        pick_btn = QPushButton("")
        set_button_role(pick_btn, "subtle")
        pick_btn.setMinimumWidth(110)
        pick_btn.clicked.connect(
            lambda: self._browse_into(
                line_edit, title_key=title_key, mode=mode, filter_key=filter_key
            )
        )

        layout.addWidget(line_edit, 1)
        layout.addWidget(pick_btn)
        # Remember the button + its translation key so _retranslate() can
        # refresh the label without rebuilding the row.
        self._pick_buttons.append(pick_btn)
        return row

    def _browse_into(
        self,
        line_edit: QLineEdit,
        *,
        title_key: str,
        mode: str,
        filter_key: str = "",
    ) -> None:
        filter_spec = t(filter_key) if filter_key else "All files (*)"
        path = pick_visual_path(
            self,
            title=t(title_key),
            mode=mode,
            filter_spec=filter_spec,
            initial_path=line_edit.text().strip(),
        )
        if path:
            line_edit.setText(path)

    def _build_ui(self) -> None:
        # List populated by _make_path_row; refreshed by _retranslate().
        self._pick_buttons: list[QPushButton] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self._intro_label = QLabel("")
        self._intro_label.setWordWrap(True)
        set_hint_text(self._intro_label)
        root.addWidget(self._intro_label)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        # Remember tab widgets so _retranslate() can refresh their titles
        # without rebuilding the tab bar.
        self._import_page = self._build_import_tab()
        self._export_page = self._build_export_tab()
        self._profiles_page = self._build_profiles_tab()
        self._tabs.addTab(self._import_page, "")
        self._tabs.addTab(self._export_page, "")
        self._tabs.addTab(self._profiles_page, "")

    def _build_import_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._status_box = QGroupBox("")
        status_layout = QGridLayout(self._status_box)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setHorizontalSpacing(12)
        status_layout.setVerticalSpacing(6)
        self._status_title_labels: dict[str, QLabel] = {}
        self._status_value_labels: dict[str, QLabel] = {}
        for index, row_id in enumerate(_STATUS_ROW_IDS):
            title = QLabel("")
            # Use the uiSectionHeader role so the color tracks the theme
            # (was hardcoded #39506b which is unreadable on dark canvas).
            title.setProperty("uiSectionHeader", True)
            title.setStyleSheet("font-weight: 700;")
            value = QLabel("")
            set_subtle_value(value)
            self._status_title_labels[row_id] = title
            self._status_value_labels[row_id] = value
            status_layout.addWidget(title, index, 0)
            status_layout.addWidget(value, index, 1)
        layout.addWidget(self._status_box)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self._env_box = QGroupBox("")
        env_layout = QFormLayout(self._env_box)
        env_layout.setContentsMargins(12, 12, 12, 12)
        env_layout.setSpacing(8)

        self._env_combo = QComboBox()
        self._env_combo.currentIndexChanged.connect(self._sync_environment_fields)

        self._env_hint = QLabel("")
        self._env_hint.setWordWrap(True)
        set_hint_text(self._env_hint)
        env_layout.addRow(self._env_hint)

        self._matlab_edit = QLineEdit()
        self._matlab_edit.textChanged.connect(self._refresh_manual_environment_state)
        self._lbl_matlab = QLabel("")
        env_layout.addRow(
            self._lbl_matlab,
            self._make_path_row(
                self._matlab_edit,
                title_key="dlg.interop.env.pick_matlab_title",
                mode="file",
                filter_key="dlg.interop.env.matlab_filter",
            ),
        )

        self._startup_edit = QLineEdit()
        self._startup_edit.textChanged.connect(self._refresh_manual_environment_state)
        self._lbl_startup = QLabel("")
        env_layout.addRow(
            self._lbl_startup,
            self._make_path_row(
                self._startup_edit,
                title_key="dlg.interop.env.pick_startup_title",
                mode="file",
                filter_key="dlg.interop.env.startup_filter",
            ),
        )
        left_layout.addWidget(self._env_box)

        self._source_box = QGroupBox("")
        source_layout = QFormLayout(self._source_box)
        source_layout.setContentsMargins(12, 12, 12, 12)
        source_layout.setSpacing(8)
        self._source_edit = QLineEdit()
        self._source_edit.textChanged.connect(self._refresh_source_status)
        self._lbl_source = QLabel("")
        source_layout.addRow(
            self._lbl_source,
            self._make_path_row(
                self._source_edit,
                title_key="dlg.interop.source.pick_title",
                mode="file_or_directory",
                filter_key="dlg.interop.source.pick_filter",
            ),
        )

        self._capture_output_edit = QLineEdit()
        self._capture_output_edit.setText(
            str((Path.cwd() / "data" / "interop").resolve())
        )
        self._lbl_capture = QLabel("")
        source_layout.addRow(
            self._lbl_capture,
            self._make_path_row(
                self._capture_output_edit,
                title_key="dlg.interop.source.pick_capture_title",
                mode="directory",
            ),
        )

        self._source_hint = QLabel("")
        self._source_hint.setWordWrap(True)
        set_hint_text(self._source_hint)
        source_layout.addRow(self._source_hint)
        left_layout.addWidget(self._source_box)

        self._actions_box = QGroupBox("")
        actions_layout = QVBoxLayout(self._actions_box)
        actions_layout.setContentsMargins(12, 12, 12, 12)
        actions_layout.setSpacing(8)
        action_row = QWidget()
        action_row_layout = QHBoxLayout(action_row)
        action_row_layout.setContentsMargins(0, 0, 0, 0)
        action_row_layout.setSpacing(6)
        self._preview_btn = QPushButton("")
        set_button_role(self._preview_btn, "primary")
        self._preview_btn.clicked.connect(self._generate_preview)
        self._reload_btn = QPushButton("")
        set_button_role(self._reload_btn, "subtle")
        self._reload_btn.clicked.connect(self._reload_current_bundle)
        action_row_layout.addWidget(self._preview_btn)
        action_row_layout.addWidget(self._reload_btn)
        actions_layout.addWidget(action_row)
        self._import_status = QLabel("")
        self._import_status.setWordWrap(True)
        set_subtle_value(self._import_status)
        actions_layout.addWidget(self._import_status)
        left_layout.addWidget(self._actions_box)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._preview_box = QGroupBox("")
        preview_layout = QVBoxLayout(self._preview_box)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)

        self._preview_overview = QLabel("")
        self._preview_overview.setWordWrap(True)
        # Use uiSectionHeader role so the color follows the theme
        # (was hardcoded #284a6e which becomes unreadable on dark canvas).
        self._preview_overview.setProperty("uiSectionHeader", True)
        self._preview_overview.setStyleSheet("font-weight: 700;")
        preview_layout.addWidget(self._preview_overview)

        self._preview_counts = QLabel("")
        self._preview_counts.setWordWrap(True)
        set_hint_text(self._preview_counts)
        preview_layout.addWidget(self._preview_counts)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._source_table = QTableWidget(0, 2)
        self._source_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._source_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._source_table.horizontalHeader().setMinimumSectionSize(120)
        self._source_table.verticalHeader().setVisible(False)
        self._source_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._source_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        self._mapping_table = QTableWidget(0, 2)
        self._mapping_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._mapping_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._mapping_table.horizontalHeader().setMinimumSectionSize(120)
        self._mapping_table.verticalHeader().setVisible(False)
        self._mapping_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._mapping_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        preview_splitter.addWidget(self._source_table)
        preview_splitter.addWidget(self._mapping_table)
        preview_splitter.setChildrenCollapsible(False)
        preview_splitter.setSizes([360, 360])
        preview_layout.addWidget(preview_splitter, 1)

        self._warnings_box = QPlainTextEdit()
        self._warnings_box.setReadOnly(True)
        preview_layout.addWidget(self._warnings_box)

        import_row = QWidget()
        import_row_layout = QHBoxLayout(import_row)
        import_row_layout.setContentsMargins(0, 0, 0, 0)
        import_row_layout.setSpacing(6)
        self._import_target_combo = QComboBox()
        for target_key in (
            "dlg.interop.import_target.hardware",
            "dlg.interop.import_target.simulation",
            "dlg.interop.import_target.dataset",
            "dlg.interop.import_target.measurements",
            "dlg.interop.import_target.geometry",
        ):
            # Use the key suffix as the userData so the logic below doesn't
            # depend on the (translatable) display text.
            data = target_key.rsplit(".", 1)[-1]
            self._import_target_combo.addItem("", data)
        self._auto_smoke_check = QCheckBox("")
        self._auto_smoke_check.setChecked(True)
        self._apply_import_btn = QPushButton("")
        set_button_role(self._apply_import_btn, "primary")
        self._apply_import_btn.setEnabled(False)
        self._apply_import_btn.clicked.connect(self._apply_import)
        import_row_layout.addWidget(self._import_target_combo, 1)
        import_row_layout.addWidget(self._auto_smoke_check)
        import_row_layout.addWidget(self._apply_import_btn)
        preview_layout.addWidget(import_row)

        smoke_row = QWidget()
        smoke_row_layout = QHBoxLayout(smoke_row)
        smoke_row_layout.setContentsMargins(0, 0, 0, 0)
        smoke_row_layout.setSpacing(6)
        self._run_smoke_btn = QPushButton("")
        set_button_role(self._run_smoke_btn, "subtle")
        self._run_smoke_btn.setEnabled(False)
        self._run_smoke_btn.clicked.connect(self._run_smoke_validation)
        smoke_row_layout.addWidget(self._run_smoke_btn)
        smoke_row_layout.addStretch(1)
        preview_layout.addWidget(smoke_row)

        self._validation_log = QPlainTextEdit()
        self._validation_log.setReadOnly(True)
        preview_layout.addWidget(self._validation_log)
        right_layout.addWidget(self._preview_box, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([430, 690])
        return page

    def _build_export_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._export_form_box = QGroupBox("")
        form = QFormLayout(self._export_form_box)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        self._export_source_combo = QComboBox()
        for key in (
            "dlg.interop.export.source.simulation",
            "dlg.interop.export.source.hardware",
            "dlg.interop.export.source.recording",
        ):
            data = key.rsplit(".", 1)[-1]
            self._export_source_combo.addItem("", data)
        self._lbl_export_source = QLabel("")
        form.addRow(self._lbl_export_source, self._export_source_combo)

        self._export_dir_edit = QLineEdit()
        self._export_dir_edit.setText(
            str((Path.cwd() / "data" / "interop_export").resolve())
        )
        self._lbl_export_output = QLabel("")
        form.addRow(
            self._lbl_export_output,
            self._make_path_row(
                self._export_dir_edit,
                title_key="dlg.interop.export.pick_output_title",
                mode="directory",
            ),
        )

        self._export_hint = QLabel("")
        self._export_hint.setWordWrap(True)
        set_hint_text(self._export_hint)
        form.addRow(self._export_hint)

        checks_row = QWidget()
        checks_layout = QHBoxLayout(checks_row)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        checks_layout.setSpacing(8)
        self._export_geometry_check = QCheckBox("")
        self._export_geometry_check.setChecked(True)
        self._export_data_check = QCheckBox("")
        self._export_data_check.setChecked(True)
        self._export_scripts_check = QCheckBox("")
        self._export_scripts_check.setChecked(True)
        for widget in (
            self._export_geometry_check,
            self._export_data_check,
            self._export_scripts_check,
        ):
            checks_layout.addWidget(widget)
        checks_layout.addStretch(1)
        self._lbl_export_include = QLabel("")
        form.addRow(self._lbl_export_include, checks_row)

        self._export_btn = QPushButton("")
        set_button_role(self._export_btn, "primary")
        self._export_btn.clicked.connect(self._generate_export)
        form.addRow(self._export_btn)
        layout.addWidget(self._export_form_box)

        self._export_log = QPlainTextEdit()
        self._export_log.setReadOnly(True)
        layout.addWidget(self._export_log, 1)
        return page

    def _build_profiles_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._profiles_list = QListWidget()
        self._profiles_list.currentRowChanged.connect(self._on_profile_selected)
        layout.addWidget(self._profiles_list, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        self._profile_form_box = QGroupBox("")
        profile_form = QFormLayout(self._profile_form_box)
        profile_form.setContentsMargins(12, 12, 12, 12)
        profile_form.setSpacing(8)
        self._profile_name_edit = QLineEdit()
        self._profile_matlab_edit = QLineEdit()
        self._profile_startup_edit = QLineEdit()
        self._profile_script_edit = QLineEdit()
        self._profile_output_edit = QLineEdit()
        self._lbl_profile_name = QLabel("")
        self._lbl_profile_matlab = QLabel("")
        self._lbl_profile_startup = QLabel("")
        self._lbl_profile_script = QLabel("")
        self._lbl_profile_output = QLabel("")
        profile_form.addRow(self._lbl_profile_name, self._profile_name_edit)
        profile_form.addRow(self._lbl_profile_matlab, self._profile_matlab_edit)
        profile_form.addRow(self._lbl_profile_startup, self._profile_startup_edit)
        profile_form.addRow(self._lbl_profile_script, self._profile_script_edit)
        profile_form.addRow(self._lbl_profile_output, self._profile_output_edit)
        right_layout.addWidget(self._profile_form_box)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        self._save_profile_btn = QPushButton("")
        set_button_role(self._save_profile_btn, "primary")
        self._save_profile_btn.clicked.connect(self._save_current_profile)
        self._remove_profile_btn = QPushButton("")
        set_button_role(self._remove_profile_btn, "danger")
        self._remove_profile_btn.clicked.connect(self._remove_selected_profile)
        button_layout.addWidget(self._save_profile_btn)
        button_layout.addWidget(self._remove_profile_btn)
        right_layout.addWidget(button_row)

        self._profile_note = QLabel("")
        self._profile_note.setWordWrap(True)
        set_hint_text(self._profile_note)
        right_layout.addWidget(self._profile_note)
        right_layout.addStretch(1)
        layout.addWidget(right, 1)
        return page

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, row_id: str, state: str) -> None:
        """Mark row `row_id` as being in state `state` and refresh its text.

        `state` is one of the dlg.interop.status.* suffixes; "ready_fmt"
        is handled via a separate _set_status_ready() helper so the suffix
        parameter can be injected.
        """
        self._status_row_states[row_id] = state
        label = self._status_value_labels.get(row_id)
        if label is not None:
            label.setText(t(f"dlg.interop.status.{state}"))

    def _set_status_ready(self, row_id: str, suffix: str) -> None:
        self._status_row_states[row_id] = f"ready_fmt:{suffix}"
        label = self._status_value_labels.get(row_id)
        if label is not None:
            label.setText(t("dlg.interop.status.ready_fmt", suffix=suffix))

    def _refresh_status_labels(self) -> None:
        """Re-resolve every status value label against the active language."""
        for row_id, state in self._status_row_states.items():
            label = self._status_value_labels.get(row_id)
            if label is None:
                continue
            if state.startswith("ready_fmt:"):
                suffix = state.split(":", 1)[1]
                label.setText(t("dlg.interop.status.ready_fmt", suffix=suffix))
            else:
                label.setText(t(f"dlg.interop.status.{state}"))

    def _refresh_manual_environment_state(self) -> None:
        matlab = self._matlab_edit.text().strip()
        startup = self._startup_edit.text().strip()

        self._set_status("matlab", "specified" if matlab else "unspecified")
        self._set_status("startup", "specified" if startup else "unspecified")
        # Bridge-package row: leave "pending" after the first manual edit
        # unless it already moved into a more specific state.
        bridge_state = self._status_row_states.get("bridge_package", "unspecified")
        if bridge_state == "unspecified":
            self._set_status("bridge_package", "pending")

    def _rebuild_profile_combo(self, profiles: list[EidorsEnvironment]) -> None:
        blocked = self._env_combo.blockSignals(True)
        self._env_combo.clear()
        self._env_combo.addItem(t("dlg.interop.env.manual_entry"), None)
        for environment in profiles:
            self._env_combo.addItem(
                environment.name or t("dlg.interop.env.saved_default_name"),
                environment,
            )
        self._env_combo.blockSignals(blocked)

    def _selected_environment(self, combo: QComboBox) -> EidorsEnvironment | None:
        data = combo.currentData()
        return data if isinstance(data, EidorsEnvironment) else None

    def _manual_environment_from_fields(self) -> EidorsEnvironment | None:
        matlab = self._matlab_edit.text().strip()
        startup = self._startup_edit.text().strip()
        if not matlab and not startup:
            return None
        return EidorsEnvironment(
            name=t("dlg.interop.profiles.manual_name"),
            matlab_command=matlab,
            matlab_root="",
            eidors_startup=startup,
            source="manual",
            last_script_path=self._source_edit.text().strip(),
            last_output_dir=self._capture_output_edit.text().strip(),
        )

    def _sync_environment_fields(self) -> None:
        environment = self._selected_environment(self._env_combo)
        if environment is None:
            return
        self._matlab_edit.setText(environment.matlab_command)
        self._startup_edit.setText(environment.eidors_startup)
        if environment.last_script_path:
            self._source_edit.setText(environment.last_script_path)
        if environment.last_output_dir:
            self._capture_output_edit.setText(environment.last_output_dir)
        self._refresh_manual_environment_state()

    def _refresh_source_status(self) -> None:
        source_text = self._source_edit.text().strip()
        if not source_text:
            self._set_status("source", "not_selected")
            return
        source = Path(to_posix_path(source_text))
        if not source.exists():
            self._set_status("source", "not_found")
            return
        suffix = source.suffix.lower() if source.is_file() else "<dir>"
        self._set_status_ready("source", suffix)

    # ------------------------------------------------------------------
    # Preview & import
    # ------------------------------------------------------------------

    def _reload_current_bundle(self) -> None:
        if self._loaded_bundle is None:
            self._import_status.setText(t("dlg.interop.msg.bundle_no_preview"))
            return
        self._update_preview(
            self._loaded_bundle,
            self._importer.preview_loaded_package(self._loaded_bundle),
        )

    def _generate_preview(self) -> None:
        source_text = self._source_edit.text().strip()
        if not source_text:
            QMessageBox.warning(
                self, t("dlg.interop.title"), t("dlg.interop.msg.no_source")
            )
            return

        source = Path(source_text)
        if source.suffix.lower() == ".m":
            missing_parts: list[str] = []
            if not self._matlab_edit.text().strip():
                missing_parts.append("MATLAB")
            if not self._startup_edit.text().strip():
                missing_parts.append("startup.m")
            if missing_parts:
                joiner = t("dlg.interop.msg.missing_joiner")
                QMessageBox.warning(
                    self,
                    t("dlg.interop.title"),
                    t(
                        "dlg.interop.msg.missing_before_script",
                        parts=joiner.join(missing_parts),
                    ),
                )
                return

        environment = self._manual_environment_from_fields()
        if environment is not None:
            environment = EidorsEnvironment(
                name=environment.name,
                matlab_command=self._matlab_edit.text().strip(),
                matlab_root=environment.matlab_root,
                eidors_startup=self._startup_edit.text().strip(),
                source=environment.source,
                last_script_path=source_text,
                last_output_dir=self._capture_output_edit.text().strip(),
                matlab_host_os=environment.matlab_host_os,
                startup_host_os=environment.startup_host_os,
                runtime_kind=environment.runtime_kind,
            )
            self._capture_service.save_last_environment(environment)

        try:
            loaded = self._capture_service.capture_or_load(
                source,
                environment=environment,
                output_dir=self._capture_output_edit.text().strip(),
            )
            preview = self._importer.preview_loaded_package(loaded)
        except Exception as exc:
            QMessageBox.critical(
                self,
                t("dlg.interop.title"),
                t("dlg.interop.msg.preview_failed", error=str(exc)),
            )
            self._import_status.setText(
                t("dlg.interop.msg.preview_failed", error=str(exc))
            )
            self._set_status("bridge_package", "failed")
            return

        self._update_preview(loaded, preview)

    def _update_preview(
        self, loaded: LoadedBridgePackage, preview: EidorsImportPreview
    ) -> None:
        self._loaded_bundle = loaded
        self._preview = preview
        self._apply_import_btn.setEnabled(True)
        self._run_smoke_btn.setEnabled(True)
        self._set_status("bridge_package", "ready")

        source_rows: list[tuple[str, str]] = []
        mapping_rows: list[tuple[str, str]] = []
        for key, value in preview.geometry_summary.items():
            source_rows.append((f"Geometry \u00b7 {key}", str(value)))
        for key, value in preview.measurement_summary.items():
            source_rows.append((f"Measurements \u00b7 {key}", str(value)))
        for key, value in preview.recognized_fields.items():
            mapping_rows.append((f"Recognized \u00b7 {key}", str(value)))
        for key, value in preview.inferred_fields.items():
            mapping_rows.append((f"Inferred \u00b7 {key}", str(value)))
        for key in preview.missing_fields:
            mapping_rows.append(
                (f"Missing \u00b7 {key}", t("dlg.interop.preview.missing_fallback"))
            )

        for table, rows in (
            (self._source_table, source_rows),
            (self._mapping_table, mapping_rows),
        ):
            table.setRowCount(len(rows))
            for row_index, (left, right) in enumerate(rows):
                table.setItem(row_index, 0, QTableWidgetItem(left))
                table.setItem(row_index, 1, QTableWidgetItem(right))
            table.resizeColumnsToContents()

        self._preview_overview.setText(
            t(
                "dlg.interop.preview.overview",
                dim=preview.forward_model_config.display_dimension(),
                n_elec=preview.forward_model_config.n_elec,
                pts=preview.forward_model_config.point_count(),
            )
        )
        self._preview_counts.setText(
            t(
                "dlg.interop.preview.counts",
                recognized=len(preview.recognized_fields),
                inferred=len(preview.inferred_fields),
                missing=len(preview.missing_fields),
            )
        )

        warnings = list(preview.warnings) or [t("dlg.interop.preview.no_warnings")]
        self._warnings_box.setPlainText("\n".join(f"- {item}" for item in warnings))
        self._validation_log.clear()

        self._import_status.setText(
            t(
                "dlg.interop.preview.done",
                dim=preview.forward_model_config.display_dimension(),
                n_elec=preview.forward_model_config.n_elec,
                pts=preview.forward_model_config.point_count(),
            )
        )

    def _apply_import(self) -> None:
        if self._loaded_bundle is None:
            return
        if self._apply_import_callback is None:
            QMessageBox.information(
                self, t("dlg.interop.title"), t("dlg.interop.msg.no_callback_import")
            )
            return

        target = str(self._import_target_combo.currentData())
        try:
            message = self._apply_import_callback(target, self._loaded_bundle)
        except Exception as exc:
            QMessageBox.critical(
                self,
                t("dlg.interop.title"),
                t("dlg.interop.msg.import_failed", error=str(exc)),
            )
            self._import_status.setText(
                t("dlg.interop.msg.import_failed", error=str(exc))
            )
            return

        smoke_message = ""
        if (
            self._auto_smoke_check.isChecked()
            and self._smoke_validate_callback is not None
        ):
            smoke_message = self._run_smoke_validation(show_dialog=False)

        self._import_status.setText(message)
        final_message = (
            message if not smoke_message else f"{message}\n\n{smoke_message}"
        )
        QMessageBox.information(self, t("dlg.interop.title"), final_message)

    def _run_smoke_validation(self, *, show_dialog: bool = True) -> str:
        if self._loaded_bundle is None:
            return t("dlg.interop.msg.smoke_no_bundle")
        if self._smoke_validate_callback is None:
            return t("dlg.interop.msg.no_callback_smoke")
        try:
            message = self._smoke_validate_callback(self._loaded_bundle)
        except Exception as exc:
            message = t("dlg.interop.msg.smoke_failed", error=str(exc))
            self._validation_log.appendPlainText(f"[FAIL] {message}")
            if show_dialog:
                QMessageBox.warning(self, t("dlg.interop.title"), message)
            return message
        self._validation_log.appendPlainText(f"[OK] {message}")
        if show_dialog:
            QMessageBox.information(self, t("dlg.interop.title"), message)
        return message

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _generate_export(self) -> None:
        if self._export_snapshot_provider is None:
            QMessageBox.information(
                self, t("dlg.interop.title"), t("dlg.interop.msg.no_callback_export")
            )
            return
        snapshots = self._export_snapshot_provider()
        source_kind = str(self._export_source_combo.currentData())
        snapshot = snapshots.get(source_kind)
        if not snapshot:
            QMessageBox.warning(
                self, t("dlg.interop.title"), t("dlg.interop.msg.no_snapshot")
            )
            return

        environment = self._manual_environment_from_fields()
        job = EidorsExportJob(
            source_kind=source_kind,
            output_dir=self._export_dir_edit.text().strip(),
            include_geometry=self._export_geometry_check.isChecked(),
            include_measurements=self._export_data_check.isChecked(),
            include_scripts=self._export_scripts_check.isChecked(),
            source_name=str(snapshot.get("name", source_kind)),
        )
        try:
            root = self._exporter.export_bundle(
                job,
                forward_model_config=snapshot["forward_model_config"],
                environment=environment,
                geometry_payload=snapshot.get("geometry_payload"),
                measurements=snapshot.get("measurements"),
                reconstruction_preset=snapshot.get("reconstruction_preset"),
                notes=list(snapshot.get("notes", [])),
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                t("dlg.interop.title"),
                t("dlg.interop.msg.export_failed", error=str(exc)),
            )
            self._export_log.appendPlainText(f"[ERROR] {exc}")
            return

        self._export_log.appendPlainText(
            t("dlg.interop.export.success", root=str(root))
        )
        self._export_log.appendPlainText(
            t("dlg.interop.export.source_tag", source_kind=source_kind)
        )
        self._export_log.appendPlainText("")

    # ------------------------------------------------------------------
    # Profile list
    # ------------------------------------------------------------------

    def _load_profiles_into_list(self) -> None:
        profiles = self._capture_service.load_profiles()
        # Cache profile list so _retranslate() can rebuild the combo
        # without re-hitting the capture service (and so combo entries
        # in the environment selector stay in the right order).
        self._environments = list(profiles)
        self._profiles_list.clear()
        for profile in profiles:
            item = QListWidgetItem(profile.name or t("dlg.interop.profiles.unnamed"))
            item.setData(Qt.ItemDataRole.UserRole, profile)
            self._profiles_list.addItem(item)
        self._rebuild_profile_combo(profiles)

    def _on_profile_selected(self, row: int) -> None:
        item = self._profiles_list.item(row)
        profile = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if not isinstance(profile, EidorsEnvironment):
            for widget in (
                self._profile_name_edit,
                self._profile_matlab_edit,
                self._profile_startup_edit,
                self._profile_script_edit,
                self._profile_output_edit,
            ):
                widget.clear()
            return
        self._profile_name_edit.setText(profile.name)
        self._profile_matlab_edit.setText(profile.matlab_command)
        self._profile_startup_edit.setText(profile.eidors_startup)
        self._profile_script_edit.setText(profile.last_script_path)
        self._profile_output_edit.setText(profile.last_output_dir)
        self._matlab_edit.setText(profile.matlab_command)
        self._startup_edit.setText(profile.eidors_startup)
        if profile.last_script_path:
            self._source_edit.setText(profile.last_script_path)
        if profile.last_output_dir:
            self._capture_output_edit.setText(profile.last_output_dir)
        self._refresh_manual_environment_state()

    def _save_current_profile(self) -> None:
        profile = EidorsEnvironment(
            name=self._profile_name_edit.text().strip()
            or t("dlg.interop.profiles.custom_default"),
            matlab_command=self._matlab_edit.text().strip(),
            matlab_root="",
            eidors_startup=self._startup_edit.text().strip(),
            source="manual",
            last_script_path=self._source_edit.text().strip(),
            last_output_dir=self._capture_output_edit.text().strip(),
        )
        profiles = self._capture_service.load_profiles()
        profiles = [item for item in profiles if item.name != profile.name]
        profiles.append(profile)
        self._capture_service.save_profiles(profiles)
        self._load_profiles_into_list()
        self._append_diag(t("dlg.interop.msg.profile_saved", name=profile.name))

    def _remove_selected_profile(self) -> None:
        row = self._profiles_list.currentRow()
        if row < 0:
            return
        item = self._profiles_list.item(row)
        profile = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(profile, EidorsEnvironment):
            return
        profiles = [
            item
            for item in self._capture_service.load_profiles()
            if item.name != profile.name
        ]
        self._capture_service.save_profiles(profiles)
        self._load_profiles_into_list()
        self._append_diag(t("dlg.interop.msg.profile_removed", name=profile.name))

    def _append_diag(self, message: str) -> None:
        """Log a diagnostic message to the validation log.

        The import status area in the Import tab is the most user-visible
        place to surface transient profile events (saved / removed), so
        we pipe non-fatal diagnostics through there.  The validation log
        below also mirrors it for a persistent record.
        """
        self._import_status.setText(message)
        self._validation_log.appendPlainText(f"[INFO] {message}")

    # ------------------------------------------------------------------
    # Translation refresh
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        """Refresh every user-visible string to the active language."""
        self.setWindowTitle(t("dlg.interop.title"))
        self._intro_label.setText(t("dlg.interop.intro"))

        # Tab titles
        self._tabs.setTabText(0, t("dlg.interop.tabs.import"))
        self._tabs.setTabText(1, t("dlg.interop.tabs.export"))
        self._tabs.setTabText(2, t("dlg.interop.tabs.profiles"))

        # Import tab — status box
        self._status_box.setTitle(t("dlg.interop.status.title"))
        self._status_title_labels["matlab"].setText("MATLAB")
        self._status_title_labels["startup"].setText("EIDORS startup")
        self._status_title_labels["source"].setText(t("dlg.interop.source.label"))
        self._status_title_labels["bridge_package"].setText("Bridge package")
        self._refresh_status_labels()

        # Import tab — step 1 environment
        self._env_box.setTitle(t("dlg.interop.env.title"))
        self._env_hint.setText(t("dlg.interop.env.hint"))
        self._lbl_matlab.setText(t("dlg.interop.env.matlab_label"))
        self._matlab_edit.setPlaceholderText(t("dlg.interop.env.matlab_placeholder"))
        self._lbl_startup.setText(t("dlg.interop.env.startup_label"))
        self._startup_edit.setPlaceholderText(t("dlg.interop.env.startup_placeholder"))

        # Import tab — step 2 source
        self._source_box.setTitle(t("dlg.interop.source.title"))
        self._lbl_source.setText(t("dlg.interop.source.label"))
        self._source_edit.setPlaceholderText(t("dlg.interop.source.placeholder"))
        self._lbl_capture.setText(t("dlg.interop.source.capture_label"))
        self._source_hint.setText(t("dlg.interop.source.hint"))

        # Import tab — step 3 actions
        self._actions_box.setTitle(t("dlg.interop.actions.title"))
        self._preview_btn.setText(t("dlg.interop.actions.preview_button"))
        self._reload_btn.setText(t("dlg.interop.actions.reload_button"))
        # Only stamp the sentinel when the status area has not yet been
        # customised by an actual operation.
        if not self._import_status.text() or self._import_status.text() in {
            t("dlg.interop.actions.no_preview_yet"),
            "",
        }:
            self._import_status.setText(t("dlg.interop.actions.no_preview_yet"))

        # Import tab — step 4 preview
        self._preview_box.setTitle(t("dlg.interop.preview.title"))
        if not self._preview_overview.text():
            self._preview_overview.setText(t("dlg.interop.preview.waiting"))
        self._source_table.setHorizontalHeaderLabels(
            [
                t("dlg.interop.preview.source_col_header"),
                t("dlg.interop.preview.value_col_header"),
            ]
        )
        self._mapping_table.setHorizontalHeaderLabels(
            [
                t("dlg.interop.preview.mapping_col_header"),
                t("dlg.interop.preview.value_col_header"),
            ]
        )
        self._warnings_box.setPlaceholderText(
            t("dlg.interop.preview.warnings_placeholder")
        )
        self._validation_log.setPlaceholderText(
            t("dlg.interop.preview.smoke_placeholder")
        )

        # Import target combo labels
        target_keys = [
            "dlg.interop.import_target.hardware",
            "dlg.interop.import_target.simulation",
            "dlg.interop.import_target.dataset",
            "dlg.interop.import_target.measurements",
            "dlg.interop.import_target.geometry",
        ]
        for index, key in enumerate(target_keys):
            self._import_target_combo.setItemText(index, t(key))

        self._auto_smoke_check.setText(t("dlg.interop.auto_smoke_check"))
        self._apply_import_btn.setText(t("dlg.interop.import_button"))
        self._run_smoke_btn.setText(t("dlg.interop.smoke_button"))

        # Export tab
        self._export_form_box.setTitle(t("dlg.interop.export.title"))
        export_source_keys = [
            "dlg.interop.export.source.simulation",
            "dlg.interop.export.source.hardware",
            "dlg.interop.export.source.recording",
        ]
        for index, key in enumerate(export_source_keys):
            self._export_source_combo.setItemText(index, t(key))
        self._lbl_export_source.setText(t("dlg.interop.export.source_label"))
        self._lbl_export_output.setText(t("dlg.interop.export.output_label"))
        self._export_hint.setText(t("dlg.interop.export.hint"))
        self._lbl_export_include.setText(t("dlg.interop.export.include_label"))
        self._export_geometry_check.setText(t("dlg.interop.export.include_geometry"))
        self._export_data_check.setText(t("dlg.interop.export.include_data"))
        self._export_scripts_check.setText(t("dlg.interop.export.include_scripts"))
        self._export_btn.setText(t("dlg.interop.export.generate_button"))
        self._export_log.setPlaceholderText(t("dlg.interop.export.log_placeholder"))

        # Profiles tab
        self._profile_form_box.setTitle(t("dlg.interop.profiles.group_title"))
        self._lbl_profile_name.setText(t("dlg.interop.profiles.name_label"))
        self._lbl_profile_matlab.setText(t("dlg.interop.profiles.matlab_label"))
        self._lbl_profile_startup.setText(t("dlg.interop.profiles.startup_label"))
        self._lbl_profile_script.setText(t("dlg.interop.profiles.script_label"))
        self._lbl_profile_output.setText(t("dlg.interop.profiles.output_label"))
        self._save_profile_btn.setText(t("dlg.interop.profiles.save_button"))
        self._remove_profile_btn.setText(t("dlg.interop.profiles.remove_button"))
        self._profile_note.setText(t("dlg.interop.profiles.note"))

        # Path-picker buttons (all share the same label)
        pick_label = t("dlg.interop.path_pick_button")
        for btn in self._pick_buttons:
            btn.setText(pick_label)

        # Keep the environment combo's saved-profile entries in sync with
        # the translated placeholder strings.
        current_data = self._env_combo.currentData()
        self._rebuild_profile_combo(self._environments)
        # Restore selection if the same environment is still present.
        if isinstance(current_data, EidorsEnvironment):
            for idx in range(self._env_combo.count()):
                if self._env_combo.itemData(idx) is current_data:
                    self._env_combo.setCurrentIndex(idx)
                    break
