"""Bridge v3 immutable model asset manager."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyeidors.interop import ModelRegistry


def _package_summary(registry: ModelRegistry, model_id: str) -> dict[str, Any]:
    from pyeidors.interop.geometry_exchange import (
        electrode_specs_from_exchange_payload,
    )

    registered = registry.get(model_id)
    package = registry.load_package(model_id)
    geometry = package.geometry
    protocol = package.protocol
    fields = package.fields
    nodes = np.asarray(geometry.get("nodes", []))
    elements = np.asarray(geometry.get("elems", []))
    electrode_specs = electrode_specs_from_exchange_payload(dict(geometry))
    electrode_details = [
        {
            "index": index + 1,
            "type": spec.kind,
            "source_nodes_1based": [node + 1 for node in spec.source_nodes],
            "pem_weights": list(spec.node_weights),
            "boundary_kind": spec.boundary_kind,
            "source_faces_1based": [
                [node + 1 for node in face] for face in spec.source_faces
            ],
            "z_contact": spec.contact_impedance,
            "z_contact_present": spec.contact_impedance_present,
            "z_contact_applicable": spec.contact_impedance_applicable,
        }
        for index, spec in enumerate(electrode_specs)
    ]
    return {
        **registered.to_mapping(),
        "dimension": int(package.model.get("dimension", nodes.shape[1])),
        "node_count": int(nodes.shape[0]) if nodes.ndim == 2 else 0,
        "element_count": int(elements.shape[0]) if elements.ndim == 2 else 0,
        "electrode_count": int(package.model.get("n_elec", 0)),
        "electrodes": electrode_details,
        "electrode_node_counts": np.asarray(geometry.get("electrode_node_counts", []))
        .reshape(-1)
        .tolist(),
        "contact_impedance": np.asarray(geometry.get("contact_impedance", []))
        .reshape(-1)
        .tolist(),
        "stimulation_matrix": np.asarray(protocol.get("stim_matrix", [])).tolist(),
        "measurement_counts": np.asarray(protocol.get("measurement_counts", []))
        .reshape(-1)
        .tolist(),
        "normalize_measurements": bool(
            np.asarray(protocol.get("normalize_measurements", False)).reshape(-1)[0]
        ),
        "background_element_count": int(
            np.asarray(fields.get("background_elem_data", [])).size
        ),
        "target_element_count": int(
            np.asarray(fields.get("target_elem_data", [])).size
        ),
        "coarse2fine_shape": list(
            np.asarray(fields.get("coarse2fine", np.empty((0, 0)))).shape
        ),
        "forward_blockers": list(
            package.manifest.get("forward_blockers", [])
            or package.model.get("forward_blockers", [])
            or []
        ),
    }


class ModelAssetManagerDialog(QDialog):
    """Inspect registered v3 assets and manage the three workflow bindings."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        registry: ModelRegistry | None = None,
    ) -> None:
        super().__init__(parent)
        self.registry = registry or ModelRegistry()
        self.setWindowTitle("Bridge v3 模型资产管理器")
        self.resize(1080, 700)

        root = QVBoxLayout(self)
        intro = QLabel(
            "已注册模型是经过完整性校验的只读副本。选择模型后可绑定到"
            "仿真、数据集或实时成像。"
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "名称",
                "model_id",
                "状态",
                "来源",
                "仿真",
                "数据集",
                "实时",
            ]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._show_selected)
        root.addWidget(self.table, 1)

        controls = QHBoxLayout()
        self.flow_combo = QComboBox()
        for flow, label in (
            ("simulation", "绑定到仿真"),
            ("dataset", "绑定到数据集"),
            ("realtime", "绑定到实时成像"),
        ):
            self.flow_combo.addItem(label, flow)
        bind_button = QPushButton("绑定所选流程")
        bind_button.clicked.connect(self._bind_selected_flow)
        apply_all_button = QPushButton("应用到全部")
        apply_all_button.clicked.connect(self._apply_selected_to_all)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh)
        controls.addWidget(self.flow_combo)
        controls.addWidget(bind_button)
        controls.addWidget(apply_all_button)
        controls.addStretch(1)
        controls.addWidget(refresh_button)
        root.addLayout(controls)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("选择一个模型查看完整语义、协议和阻断项。")
        root.addWidget(self.details, 1)
        self.refresh()

    def _selected_model_id(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            raise ValueError("请先选择一个模型资产。")
        item = self.table.item(row, 1)
        if item is None or not item.text().strip():
            raise ValueError("所选模型没有 model_id。")
        return item.text().strip()

    def refresh(self) -> None:
        models = self.registry.list_models()
        bindings = {
            flow: model.model_id for flow, model in self.registry.bindings().items()
        }
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            values = (
                model.display_name,
                model.model_id,
                model.status,
                model.source_path,
                "✓" if bindings.get("simulation") == model.model_id else "",
                "✓" if bindings.get("dataset") == model.model_id else "",
                "✓" if bindings.get("realtime") == model.model_id else "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column >= 4:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        if models:
            self.table.selectRow(0)
        else:
            self.details.setPlainText("尚未注册 Bridge v3 模型。")

    def _show_selected(self) -> None:
        try:
            summary = _package_summary(
                self.registry,
                self._selected_model_id(),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            self.details.setPlainText(f"模型加载失败：{exc}")
            return
        self.details.setPlainText(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        )

    def _bind_selected_flow(self) -> None:
        try:
            model_id = self._selected_model_id()
            flow = str(self.flow_combo.currentData())
            self.registry.bind(flow, model_id)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self.refresh()

    def _apply_selected_to_all(self) -> None:
        try:
            self.registry.apply_to_all(self._selected_model_id())
        except (KeyError, OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self.refresh()


__all__ = ["ModelAssetManagerDialog"]
