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
        self.setWindowTitle("Interop Hub")
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

        self._build_ui()
        self._load_profiles_into_list()
        self._refresh_manual_environment_state()
        self._refresh_source_status()

    def _make_path_row(
        self,
        line_edit: QLineEdit,
        *,
        title: str,
        mode: str,
        filter_spec: str = "All files (*)",
    ) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        pick_btn = QPushButton("选择...")
        set_button_role(pick_btn, "subtle")
        pick_btn.setMinimumWidth(110)
        pick_btn.clicked.connect(
            lambda: self._browse_into(line_edit, title=title, mode=mode, filter_spec=filter_spec)
        )

        layout.addWidget(line_edit, 1)
        layout.addWidget(pick_btn)
        return row

    def _browse_into(
        self,
        line_edit: QLineEdit,
        *,
        title: str,
        mode: str,
        filter_spec: str = "All files (*)",
    ) -> None:
        path = pick_visual_path(
            self,
            title=title,
            mode=mode,
            filter_spec=filter_spec,
            initial_path=line_edit.text().strip(),
        )
        if path:
            line_edit.setText(path)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        intro = QLabel(
            "在这里我们把 EIDORS 与 PyEIDORS 之间的迁移做成一条可视化、可确认、可回滚的工作流。"
        )
        intro.setWordWrap(True)
        set_hint_text(intro)
        root.addWidget(intro)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_import_tab(), "Import from EIDORS")
        self._tabs.addTab(self._build_export_tab(), "Export to EIDORS")
        self._tabs.addTab(self._build_profiles_tab(), "Profiles & Paths")

    def _build_import_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        status_box = QGroupBox("当前手动指定状态")
        status_layout = QGridLayout(status_box)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setHorizontalSpacing(12)
        status_layout.setVerticalSpacing(6)
        self._status_labels: dict[str, QLabel] = {}
        for index, key in enumerate(("MATLAB", "EIDORS startup", "Source", "Bridge package")):
            title = QLabel(key)
            title.setStyleSheet("font-weight: 700; color: #39506b;")
            value = QLabel("未指定")
            set_subtle_value(value)
            self._status_labels[key] = value
            status_layout.addWidget(title, index, 0)
            status_layout.addWidget(value, index, 1)
        layout.addWidget(status_box)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        env_box = QGroupBox("Step 1 · Specify Environment")
        env_layout = QFormLayout(env_box)
        env_layout.setContentsMargins(12, 12, 12, 12)
        env_layout.setSpacing(8)

        self._env_combo = QComboBox()
        self._env_combo.currentIndexChanged.connect(self._sync_environment_fields)

        env_hint = QLabel("请点击“选择...”手动指定 MATLAB 与 startup.m 路径。统一文件浏览器会按当前环境显示可访问的 Linux / WSL / Windows 位置；环境画像可在 Profiles & Paths 页管理。")
        env_hint.setWordWrap(True)
        set_hint_text(env_hint)
        env_layout.addRow(env_hint)

        self._matlab_edit = QLineEdit()
        self._matlab_edit.setPlaceholderText("matlab.exe path")
        self._matlab_edit.textChanged.connect(self._refresh_manual_environment_state)
        env_layout.addRow(
            "MATLAB:",
            self._make_path_row(
                self._matlab_edit,
                title="选择 MATLAB 可执行文件",
                mode="file",
                filter_spec="Executable (*.exe *.bin *.sh);;All files (*)",
            ),
        )

        self._startup_edit = QLineEdit()
        self._startup_edit.setPlaceholderText("startup.m path")
        self._startup_edit.textChanged.connect(self._refresh_manual_environment_state)
        env_layout.addRow(
            "EIDORS startup:",
            self._make_path_row(
                self._startup_edit,
                title="选择 EIDORS startup.m",
                mode="file",
                filter_spec="MATLAB script (*.m);;All files (*)",
            ),
        )
        left_layout.addWidget(env_box)

        source_box = QGroupBox("Step 2 · Select Source")
        source_layout = QFormLayout(source_box)
        source_layout.setContentsMargins(12, 12, 12, 12)
        source_layout.setSpacing(8)
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("选择 EIDORS .m 脚本、bridge 目录、legacy .mat 或 bridge JSON")
        self._source_edit.textChanged.connect(self._refresh_source_status)
        source_layout.addRow(
            "Source:",
            self._make_path_row(
                self._source_edit,
                title="选择 EIDORS 脚本、bridge 文件或 bridge 目录",
                mode="file_or_directory",
                filter_spec="Supported (*.m *.mat *.json);;MATLAB script (*.m);;MAT file (*.mat);;JSON (*.json);;All files (*)",
            ),
        )

        self._capture_output_edit = QLineEdit()
        self._capture_output_edit.setText(str((Path.cwd() / "data" / "interop").resolve()))
        source_layout.addRow(
            "Capture output:",
            self._make_path_row(
                self._capture_output_edit,
                title="选择桥接采集输出目录",
                mode="directory",
            ),
        )

        self._source_hint = QLabel("支持三种来源：用户脚本、已有 bridge 工程、legacy 几何 .mat。")
        self._source_hint.setWordWrap(True)
        set_hint_text(self._source_hint)
        source_layout.addRow(self._source_hint)
        left_layout.addWidget(source_box)

        actions_box = QGroupBox("Step 3 · 采集与预览")
        actions_layout = QVBoxLayout(actions_box)
        actions_layout.setContentsMargins(12, 12, 12, 12)
        actions_layout.setSpacing(8)
        action_row = QWidget()
        action_row_layout = QHBoxLayout(action_row)
        action_row_layout.setContentsMargins(0, 0, 0, 0)
        action_row_layout.setSpacing(6)
        self._preview_btn = QPushButton("生成预览")
        set_button_role(self._preview_btn, "primary")
        self._preview_btn.clicked.connect(self._generate_preview)
        self._reload_btn = QPushButton("重载上次结果")
        set_button_role(self._reload_btn, "subtle")
        self._reload_btn.clicked.connect(self._reload_current_bundle)
        action_row_layout.addWidget(self._preview_btn)
        action_row_layout.addWidget(self._reload_btn)
        actions_layout.addWidget(action_row)
        self._import_status = QLabel("尚未生成迁移预览。")
        self._import_status.setWordWrap(True)
        set_subtle_value(self._import_status)
        actions_layout.addWidget(self._import_status)
        left_layout.addWidget(actions_box)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        preview_box = QGroupBox("Step 4 · 预览与导入")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)

        self._preview_overview = QLabel("等待 bridge 包预览。")
        self._preview_overview.setWordWrap(True)
        self._preview_overview.setStyleSheet("font-weight: 700; color: #284a6e;")
        preview_layout.addWidget(self._preview_overview)

        self._preview_counts = QLabel("")
        self._preview_counts.setWordWrap(True)
        set_hint_text(self._preview_counts)
        preview_layout.addWidget(self._preview_counts)

        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._source_table = QTableWidget(0, 2)
        self._source_table.setHorizontalHeaderLabels(["EIDORS 来源", "值"])
        self._source_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._source_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._source_table.horizontalHeader().setMinimumSectionSize(120)
        self._source_table.verticalHeader().setVisible(False)
        self._source_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._source_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        self._mapping_table = QTableWidget(0, 2)
        self._mapping_table.setHorizontalHeaderLabels(["PyEIDORS 映射", "值"])
        self._mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
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
        self._warnings_box.setPlaceholderText("Warnings and unresolved fields will appear here.")
        preview_layout.addWidget(self._warnings_box)

        import_row = QWidget()
        import_row_layout = QHBoxLayout(import_row)
        import_row_layout.setContentsMargins(0, 0, 0, 0)
        import_row_layout.setSpacing(6)
        self._import_target_combo = QComboBox()
        self._import_target_combo.addItem("硬件配置模板", "hardware")
        self._import_target_combo.addItem("仿真配置", "simulation")
        self._import_target_combo.addItem("数据集配置", "dataset")
        self._import_target_combo.addItem("仅边界电压数据", "measurements")
        self._import_target_combo.addItem("仅几何资产", "geometry")
        self._auto_smoke_check = QCheckBox("导入后自动做一次逆问题冒烟验证")
        self._auto_smoke_check.setChecked(True)
        self._apply_import_btn = QPushButton("导入到 PyEIDORS")
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
        self._run_smoke_btn = QPushButton("运行冒烟验证")
        set_button_role(self._run_smoke_btn, "subtle")
        self._run_smoke_btn.setEnabled(False)
        self._run_smoke_btn.clicked.connect(self._run_smoke_validation)
        smoke_row_layout.addWidget(self._run_smoke_btn)
        smoke_row_layout.addStretch(1)
        preview_layout.addWidget(smoke_row)

        self._validation_log = QPlainTextEdit()
        self._validation_log.setReadOnly(True)
        self._validation_log.setPlaceholderText("导入后的逆问题烟测结果会显示在这里。")
        preview_layout.addWidget(self._validation_log)
        right_layout.addWidget(preview_box, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([430, 690])
        return page

    def _build_export_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        form_box = QGroupBox("导出到 EIDORS")
        form = QFormLayout(form_box)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        self._export_source_combo = QComboBox()
        self._export_source_combo.addItem("当前仿真配置", "simulation")
        self._export_source_combo.addItem("当前硬件布局配置", "hardware")
        self._export_source_combo.addItem("当前录制/重构结果", "recording")
        form.addRow("Source:", self._export_source_combo)

        self._export_dir_edit = QLineEdit()
        self._export_dir_edit.setText(str((Path.cwd() / "data" / "interop_export").resolve()))
        form.addRow(
            "Output dir:",
            self._make_path_row(
                self._export_dir_edit,
                title="选择导出 Bridge 工程目录",
                mode="directory",
            ),
        )

        export_hint = QLabel("导出 bridge 工程时，会优先写入当前手动指定的 MATLAB / startup.m 路径；若未指定，也仍可只导出数据与配置。")
        export_hint.setWordWrap(True)
        set_hint_text(export_hint)
        form.addRow(export_hint)

        checks_row = QWidget()
        checks_layout = QHBoxLayout(checks_row)
        checks_layout.setContentsMargins(0, 0, 0, 0)
        checks_layout.setSpacing(8)
        self._export_geometry_check = QCheckBox("Geometry")
        self._export_geometry_check.setChecked(True)
        self._export_data_check = QCheckBox("Boundary voltages")
        self._export_data_check.setChecked(True)
        self._export_scripts_check = QCheckBox("Runnable EIDORS script")
        self._export_scripts_check.setChecked(True)
        for widget in (
            self._export_geometry_check,
            self._export_data_check,
            self._export_scripts_check,
        ):
            checks_layout.addWidget(widget)
        checks_layout.addStretch(1)
        form.addRow("Include:", checks_row)

        self._export_btn = QPushButton("Generate Bridge Project")
        set_button_role(self._export_btn, "primary")
        self._export_btn.clicked.connect(self._generate_export)
        form.addRow(self._export_btn)
        layout.addWidget(form_box)

        self._export_log = QPlainTextEdit()
        self._export_log.setReadOnly(True)
        self._export_log.setPlaceholderText("导出说明、生成路径和任何降级行为都会写在这里。")
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

        profile_form_box = QGroupBox("Saved Environments")
        profile_form = QFormLayout(profile_form_box)
        profile_form.setContentsMargins(12, 12, 12, 12)
        profile_form.setSpacing(8)
        self._profile_name_edit = QLineEdit()
        self._profile_matlab_edit = QLineEdit()
        self._profile_startup_edit = QLineEdit()
        self._profile_script_edit = QLineEdit()
        self._profile_output_edit = QLineEdit()
        profile_form.addRow("Name:", self._profile_name_edit)
        profile_form.addRow("MATLAB:", self._profile_matlab_edit)
        profile_form.addRow("startup.m:", self._profile_startup_edit)
        profile_form.addRow("Last script:", self._profile_script_edit)
        profile_form.addRow("Last output:", self._profile_output_edit)
        right_layout.addWidget(profile_form_box)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(6)
        self._save_profile_btn = QPushButton("Save Current Environment")
        set_button_role(self._save_profile_btn, "primary")
        self._save_profile_btn.clicked.connect(self._save_current_profile)
        self._remove_profile_btn = QPushButton("Remove Selected")
        set_button_role(self._remove_profile_btn, "danger")
        self._remove_profile_btn.clicked.connect(self._remove_selected_profile)
        button_layout.addWidget(self._save_profile_btn)
        button_layout.addWidget(self._remove_profile_btn)
        right_layout.addWidget(button_row)

        note = QLabel("这里保存的是 EIDORS 环境画像，不会修改用户原始 MATLAB 工程。")
        note.setWordWrap(True)
        set_hint_text(note)
        right_layout.addWidget(note)
        right_layout.addStretch(1)
        layout.addWidget(right, 1)
        return page

    def _refresh_manual_environment_state(self) -> None:
        matlab = self._matlab_edit.text().strip()
        startup = self._startup_edit.text().strip()

        self._status_labels["MATLAB"].setText("已指定" if matlab else "未指定")
        self._status_labels["EIDORS startup"].setText("已指定" if startup else "未指定")
        if self._status_labels["Bridge package"].text() == "未指定":
            self._status_labels["Bridge package"].setText("待生成")

    def _rebuild_profile_combo(self, profiles: list[EidorsEnvironment]) -> None:
        blocked = self._env_combo.blockSignals(True)
        self._env_combo.clear()
        self._env_combo.addItem("当前手动输入", None)
        for environment in profiles:
            self._env_combo.addItem(environment.name or "Saved EIDORS Environment", environment)
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
            name="Manual Environment",
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
            self._status_labels["Source"].setText("未选择")
            return
        source = Path(to_posix_path(source_text))
        if not source.exists():
            self._status_labels["Source"].setText("未找到")
            return
        suffix = source.suffix.lower() if source.is_file() else "<dir>"
        self._status_labels["Source"].setText(f"就绪 ({suffix})")

    def _reload_current_bundle(self) -> None:
        if self._loaded_bundle is None:
            self._import_status.setText("当前还没有已加载的 bridge 包。")
            return
        self._update_preview(self._loaded_bundle, self._importer.preview_loaded_package(self._loaded_bundle))

    def _generate_preview(self) -> None:
        source_text = self._source_edit.text().strip()
        if not source_text:
            QMessageBox.warning(self, "Interop Hub", "请先选择一个 EIDORS 脚本或 bridge 包来源。")
            return

        source = Path(source_text)
        if source.suffix.lower() == ".m":
            missing_parts: list[str] = []
            if not self._matlab_edit.text().strip():
                missing_parts.append("MATLAB")
            if not self._startup_edit.text().strip():
                missing_parts.append("startup.m")
            if missing_parts:
                QMessageBox.warning(
                    self,
                    "Interop Hub",
                    f"运行 EIDORS 脚本前，请先手动指定：{'、'.join(missing_parts)}。",
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
            QMessageBox.critical(self, "Interop Hub", f"生成预览失败：{exc}")
            self._import_status.setText(f"生成预览失败：{exc}")
            self._status_labels["Bridge package"].setText("Failed")
            return

        self._update_preview(loaded, preview)

    def _update_preview(self, loaded: LoadedBridgePackage, preview: EidorsImportPreview) -> None:
        self._loaded_bundle = loaded
        self._preview = preview
        self._apply_import_btn.setEnabled(True)
        self._run_smoke_btn.setEnabled(True)
        self._status_labels["Bridge package"].setText("就绪")

        source_rows: list[tuple[str, str]] = []
        mapping_rows: list[tuple[str, str]] = []
        for key, value in preview.geometry_summary.items():
            source_rows.append((f"Geometry · {key}", str(value)))
        for key, value in preview.measurement_summary.items():
            source_rows.append((f"Measurements · {key}", str(value)))
        for key, value in preview.recognized_fields.items():
            mapping_rows.append((f"Recognized · {key}", str(value)))
        for key, value in preview.inferred_fields.items():
            mapping_rows.append((f"Inferred · {key}", str(value)))
        for key in preview.missing_fields:
            mapping_rows.append((f"Missing · {key}", "需要用户补充或改用桥接模板包装脚本"))

        for table, rows in ((self._source_table, source_rows), (self._mapping_table, mapping_rows)):
            table.setRowCount(len(rows))
            for row_index, (left, right) in enumerate(rows):
                table.setItem(row_index, 0, QTableWidgetItem(left))
                table.setItem(row_index, 1, QTableWidgetItem(right))
            table.resizeColumnsToContents()

        self._preview_overview.setText(
            f"EIDORS -> PyEIDORS 映射预览：{preview.forward_model_config.display_dimension()}，"
            f"{preview.forward_model_config.n_elec} 电极/环，"
            f"{preview.forward_model_config.point_count()} 个边界电压点。"
        )
        self._preview_counts.setText(
            f"已准确识别 {len(preview.recognized_fields)} 项 | "
            f"已推断 {len(preview.inferred_fields)} 项 | "
            f"待补充 {len(preview.missing_fields)} 项"
        )

        warnings = list(preview.warnings) or ["未发现需要人工确认的高风险项。"]
        self._warnings_box.setPlainText("\n".join(f"- {item}" for item in warnings))
        self._validation_log.clear()

        self._import_status.setText(
            f"预览完成：{preview.forward_model_config.display_dimension()} | "
            f"{preview.forward_model_config.n_elec} 电极/环 | "
            f"{preview.forward_model_config.point_count()} 个边界电压点。"
        )

    def _apply_import(self) -> None:
        if self._loaded_bundle is None:
            return
        if self._apply_import_callback is None:
            QMessageBox.information(self, "Interop Hub", "当前窗口未接入导入回调。")
            return

        target = str(self._import_target_combo.currentData())
        try:
            message = self._apply_import_callback(target, self._loaded_bundle)
        except Exception as exc:
            QMessageBox.critical(self, "Interop Hub", f"导入失败：{exc}")
            self._import_status.setText(f"导入失败：{exc}")
            return

        smoke_message = ""
        if self._auto_smoke_check.isChecked() and self._smoke_validate_callback is not None:
            smoke_message = self._run_smoke_validation(show_dialog=False)

        self._import_status.setText(message)
        final_message = message if not smoke_message else f"{message}\n\n{smoke_message}"
        QMessageBox.information(self, "Interop Hub", final_message)

    def _run_smoke_validation(self, *, show_dialog: bool = True) -> str:
        if self._loaded_bundle is None:
            return "当前没有可用于烟测的 bridge 包。"
        if self._smoke_validate_callback is None:
            return "当前窗口未接入烟测回调。"
        try:
            message = self._smoke_validate_callback(self._loaded_bundle)
        except Exception as exc:
            message = f"烟测失败：{exc}"
            self._validation_log.appendPlainText(f"[FAIL] {message}")
            if show_dialog:
                QMessageBox.warning(self, "Interop Hub", message)
            return message
        self._validation_log.appendPlainText(f"[OK] {message}")
        if show_dialog:
            QMessageBox.information(self, "Interop Hub", message)
        return message

    def _generate_export(self) -> None:
        if self._export_snapshot_provider is None:
            QMessageBox.information(self, "Interop Hub", "当前窗口未接入导出数据提供器。")
            return
        snapshots = self._export_snapshot_provider()
        source_kind = str(self._export_source_combo.currentData())
        snapshot = snapshots.get(source_kind)
        if not snapshot:
            QMessageBox.warning(self, "Interop Hub", "当前来源暂时没有可导出的上下文。")
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
            QMessageBox.critical(self, "Interop Hub", f"导出失败：{exc}")
            self._export_log.appendPlainText(f"[ERROR] {exc}")
            return

        self._export_log.appendPlainText(f"[OK] Bridge 工程已生成：{root}")
        self._export_log.appendPlainText(f"      Source: {source_kind}")
        self._export_log.appendPlainText("")

    def _load_profiles_into_list(self) -> None:
        profiles = self._capture_service.load_profiles()
        self._profiles_list.clear()
        for profile in profiles:
            item = QListWidgetItem(profile.name or "Unnamed EIDORS Environment")
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
            name=self._profile_name_edit.text().strip() or "Custom EIDORS Environment",
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
        self._append_diag(f"已保存 profile：{profile.name}")

    def _remove_selected_profile(self) -> None:
        row = self._profiles_list.currentRow()
        if row < 0:
            return
        item = self._profiles_list.item(row)
        profile = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(profile, EidorsEnvironment):
            return
        profiles = [item for item in self._capture_service.load_profiles() if item.name != profile.name]
        self._capture_service.save_profiles(profiles)
        self._load_profiles_into_list()
        self._append_diag(f"已删除 profile：{profile.name}")
