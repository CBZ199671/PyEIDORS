"""Serial/4G connection panel with old relay server fields."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QWidget

from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.theme import set_button_role, set_hint_text


class ConnectionPanel(QGroupBox):
    """Panel for configuring and establishing device connections.

    Signals:
        connect_requested: Emitted with (transport_type, config_dict).
        disconnect_requested: Emitted when user clicks disconnect.
    """

    connect_requested = Signal(str, dict)
    disconnect_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("1. Link & Verify", parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(10)
        self._layout = layout

        self._flow_hint = QLabel("Select the transport and verify the device link first.")
        self._flow_hint.setWordWrap(True)
        self._flow_hint.setText("Select the transport and verify the device link first.")
        set_hint_text(self._flow_hint)
        layout.addRow(self._flow_hint)

        # Transport type
        self._transport_combo = AutoCloseComboBox()
        self._transport_combo.addItems(["Serial", "4G Relay", "Simulator"])
        self._transport_combo.currentIndexChanged.connect(self._on_transport_changed)
        layout.addRow("Transport:", self._transport_combo)

        # Serial port
        self._port_combo = AutoCloseComboBox()
        self._port_combo.setEditable(True)
        self._refresh_btn = QPushButton("Refresh")
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

        # Connect / Disconnect buttons
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 2, 0, 0)
        btn_layout.setSpacing(8)
        self._connect_btn = QPushButton("Connect & Verify")
        self._connect_btn.clicked.connect(self._on_connect)
        set_button_role(self._connect_btn, "primary")
        self._disconnect_btn = QPushButton("Disconnect Link")
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
        self._set_row_visible(self._server_host, is_relay)
        self._set_row_visible(self._server_port, is_relay)
        self._set_row_visible(self._board_id, is_relay)
        self._set_row_visible(self._user_id, is_relay)

    def _refresh_ports(self) -> None:
        self._port_combo.clear()
        try:
            from serial.tools.list_ports import comports

            for info in comports():
                label = f"{info.device} - {info.description}"
                self._port_combo.addItem(label, info.device)
        except ImportError:
            self._port_combo.addItem("(pyserial not installed)")

    def _on_connect(self) -> None:
        transport_map = {0: "serial", 1: "relay", 2: "simulator"}
        transport = transport_map.get(self._transport_combo.currentIndex(), "simulator")
        config = {
            "port": self._port_combo.currentData() or self._port_combo.currentText(),
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

    def _set_row_visible(self, field: QWidget, visible: bool) -> None:
        try:
            self._layout.setRowVisible(field, visible)
        except AttributeError:
            field.setVisible(visible)
