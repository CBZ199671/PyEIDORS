"""Serial/4G connection panel with guided discovery hints."""

import re

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QWidget

from eit_app.hardware.serial_port_discovery import SerialPortDescriptor, discover_serial_ports
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role, set_hint_text

_WINDOWS_COM_RE = re.compile(r"(COM\d+)", re.IGNORECASE)


class ConnectionPanel(QGroupBox):
    """Panel for configuring and establishing device connections.

    Signals:
        connect_requested: Emitted with (transport_type, config_dict).
        disconnect_requested: Emitted when user clicks disconnect.
    """

    connect_requested = Signal(str, dict)
    disconnect_requested = Signal()
    validation_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("1. Link & Verify", parent)
        self._serial_ports: list[SerialPortDescriptor] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(8, 10, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._layout = layout

        self._flow_hint = QLabel("Select the transport and verify the device link first.")
        self._flow_hint.setWordWrap(True)
        self._flow_hint.setText("Select the transport and verify the device link first.")
        set_hint_text(self._flow_hint)
        layout.addRow(self._flow_hint)

        # Transport type
        self._transport_combo = AutoCloseComboBox()
        self._transport_combo.addItems(["Serial", "4G Relay"])
        self._transport_combo.currentIndexChanged.connect(self._on_transport_changed)
        layout.addRow("Transport:", self._transport_combo)

        # Serial port
        self._port_combo = AutoCloseComboBox()
        self._port_combo.setEditable(True)
        self._port_combo.currentIndexChanged.connect(lambda _index: self._update_serial_hint())
        self._refresh_btn = QPushButton("Scan")
        self._refresh_btn.setToolTip("Refresh serial ports")
        self._refresh_btn.clicked.connect(self._refresh_ports)
        set_button_role(self._refresh_btn, "subtle")
        port_row = QHBoxLayout()
        port_row.setContentsMargins(0, 0, 0, 0)
        port_row.setSpacing(8)
        port_row.addWidget(self._port_combo, 1)
        port_row.addWidget(self._refresh_btn)
        self._port_widget = QWidget()
        self._port_widget.setLayout(port_row)
        layout.addRow("Port:", self._port_widget)

        self._port_hint = QLabel("")
        self._port_hint.setWordWrap(True)
        set_hint_text(self._port_hint)
        layout.addRow(self._port_hint)

        # Baud rate
        self._baud_combo = AutoCloseComboBox()
        self._baud_combo.addItems(["115200", "57600", "38400", "19200", "9600"])
        layout.addRow("Baud rate:", self._baud_combo)

        # Relay host/port (hidden by default)
        self._server_host = QLineEdit()
        self._server_host.setText("127.0.0.1")
        self._server_host.setPlaceholderText("127.0.0.1")
        layout.addRow("Server host:", self._server_host)

        self._server_port = QSpinBox()
        self._server_port.setRange(1, 65535)
        self._server_port.setValue(4555)
        layout.addRow("Server port:", self._server_port)

        self._board_id = QSpinBox()
        self._board_id.setRange(1, 255)
        self._board_id.setValue(1)
        layout.addRow("Board ID:", self._board_id)

        self._user_id = QSpinBox()
        self._user_id.setRange(1, 255)
        self._user_id.setValue(1)
        layout.addRow("User ID:", self._user_id)
        self._server_host.textChanged.connect(lambda _text: self._update_relay_hint())
        self._server_port.valueChanged.connect(lambda _value: self._update_relay_hint())

        self._transport_hint = QLabel("")
        self._transport_hint.setWordWrap(True)
        set_hint_text(self._transport_hint)
        layout.addRow(self._transport_hint)

        # Connect / Disconnect buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 2, 0, 0)
        btn_layout.setSpacing(6)
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setToolTip("Connect and verify the device link")
        self._connect_btn.clicked.connect(self._on_connect)
        set_button_role(self._connect_btn, "primary")
        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.clicked.connect(self.disconnect_requested)
        self._disconnect_btn.setEnabled(False)
        set_button_role(self._disconnect_btn, "danger")
        btn_layout.addWidget(self._connect_btn)
        btn_layout.addWidget(self._disconnect_btn)
        layout.addRow(btn_layout)

        self._on_transport_changed(0)
        self._refresh_ports()

    def _on_transport_changed(self, index: int) -> None:
        is_serial = index == 0
        is_relay = index == 1
        self._set_row_visible(self._port_widget, is_serial)
        self._set_row_visible(self._baud_combo, is_serial)
        self._set_row_visible(self._port_hint, is_serial)
        self._set_row_visible(self._server_host, is_relay)
        self._set_row_visible(self._server_port, is_relay)
        self._set_row_visible(self._board_id, is_relay)
        self._set_row_visible(self._user_id, is_relay)
        self._set_row_visible(self._transport_hint, is_relay)
        if is_serial:
            self._refresh_ports()
        else:
            self._update_relay_hint()

    def _refresh_ports(self) -> None:
        current_port = self.selected_serial_port()
        self._port_combo.clear()
        ports = discover_serial_ports()
        self._serial_ports = list(ports)
        for port in ports:
            self._port_combo.addItem(
                port.display_name,
                {
                    "device": port.device,
                    "display_name": port.display_name,
                    "source": port.source,
                },
            )

        if not ports:
            self._port_hint.setText(
                "未检测到可用串口。软件会自动检查本地串口和 Windows COM 口；"
                "请确认 USB 线、驱动和设备供电正常后再点 Scan。"
            )
            return

        selected_index = None
        for index, port in enumerate(ports):
            if current_port and port.device == current_port:
                selected_index = index
                break
        if selected_index is None:
            selected_index = 0
        self._port_combo.setCurrentIndex(selected_index)
        self._update_serial_hint()

    def _on_connect(self) -> None:
        transport_map = {0: "serial", 1: "relay"}
        transport = transport_map.get(self._transport_combo.currentIndex(), "serial")

        if transport == "serial":
            if not self.selected_serial_port():
                self._refresh_ports()
            if not self.selected_serial_port():
                self._port_hint.setText(
                    "仍未检测到可用串口，暂不发起连接。请检查 USB 连接、驱动和设备电源。"
                )
                self.validation_failed.emit("Connection failed: No serial port detected.")
                return

        port_value = self.selected_serial_port()
        port_display = self.selected_serial_display_name()
        config = {
            "port": port_value,
            "port_display": port_display,
            "baudrate": int(self._baud_combo.currentText()),
            "server_host": self._server_host.text(),
            "server_port": self._server_port.value(),
            "board_id": self._board_id.value(),
            "user_id": self._user_id.value(),
        }
        self.connect_requested.emit(transport, config)

    def set_connected(self, connected: bool) -> None:
        """Update button states based on connection status."""
        self._connect_btn.setEnabled(not connected)
        self._disconnect_btn.setEnabled(connected)
        self._transport_combo.setEnabled(not connected)
        self._port_combo.setEnabled(not connected)
        self._baud_combo.setEnabled(not connected)
        self._server_host.setEnabled(not connected)
        self._server_port.setEnabled(not connected)
        self._board_id.setEnabled(not connected)
        self._user_id.setEnabled(not connected)

    def selected_serial_port(self) -> str:
        current_data = self._port_combo.currentData()
        if isinstance(current_data, dict):
            return str(current_data.get("device", "")).strip()
        text = str(current_data or self._port_combo.currentText()).strip()
        if not text:
            return ""
        for port in self._serial_ports:
            if text == port.display_name:
                return port.device
            if text.startswith(f"{port.device} -"):
                return port.device
        match = _WINDOWS_COM_RE.search(text)
        if match is not None:
            return match.group(1).upper()
        if text.startswith("/dev/") and " -" in text:
            return text.split(" -", 1)[0].strip()
        return text

    def selected_serial_display_name(self) -> str:
        current_data = self._port_combo.currentData()
        if isinstance(current_data, dict):
            display_name = str(current_data.get("display_name", "")).strip()
            if display_name:
                return display_name
        return self._port_combo.currentText().strip() or self.selected_serial_port()

    def serial_port_count(self) -> int:
        return len(self._serial_ports)

    def refresh_serial_ports(self) -> None:
        self._refresh_ports()

    def set_serial_hint(self, text: str) -> None:
        self._port_hint.setText(text)

    def set_relay_hint(self, text: str) -> None:
        self._transport_hint.setText(text)

    def _update_serial_hint(self) -> None:
        count = len(self._serial_ports)
        current_data = self._port_combo.currentData()
        selected_name = self.selected_serial_display_name()
        source = ""
        if isinstance(current_data, dict):
            source = str(current_data.get("source", "")).strip()

        if count == 1:
            if source == "windows-com":
                self._port_hint.setText(
                    f"已自动选中唯一串口：{selected_name}。连接时会自动使用 Windows 主机串口桥接。"
                )
            else:
                self._port_hint.setText(f"已自动选中唯一串口：{selected_name}。")
            return

        if source == "windows-com":
            self._port_hint.setText(
                f"检测到 {count} 个串口，当前选择 {selected_name}。连接时会自动使用 Windows 主机串口桥接。"
            )
            return

        self._port_hint.setText(f"检测到 {count} 个串口，请确认并选择硬件对应端口。")

    def _update_relay_hint(self) -> None:
        host = self._server_host.text().strip() or "127.0.0.1"
        port = self._server_port.value()
        self._transport_hint.setText(
            f"4G Relay 将连接到 {host}:{port}；点击 Connect 前会先做服务器可达性检查。"
        )

    def _set_row_visible(self, field: QWidget, visible: bool) -> None:
        try:
            self._layout.setRowVisible(field, visible)
        except AttributeError:
            field.setVisible(visible)
