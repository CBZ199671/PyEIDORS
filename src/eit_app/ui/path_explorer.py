"""Unified Qt path picker helpers for Linux, WSL, and Windows."""

from __future__ import annotations

from string import ascii_uppercase
from pathlib import Path

from PySide6.QtCore import QByteArray, QSettings, QSize, QUrl
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QFileIconProvider,
    QHeaderView,
    QListView,
    QTreeView,
    QWidget,
)

from eit_app.i18n import t
from eit_app.interop.environment import (
    running_in_wsl,
    running_on_windows,
    to_posix_path,
    to_windows_path,
)
from eit_app.ui.theme import current_theme_mode

_SETTINGS_GROUP = "visual_path_picker"
_GEOMETRY_KEY = "geometry"
_RECENT_DIRS_KEY = "recent_directories"
_MAX_RECENT_DIRS = 8


def _settings() -> QSettings:
    return QSettings("PyEIDORS", "EITWorkstation")


def _available_windows_drives() -> list[str]:
    drives: list[str] = []
    if running_on_windows():
        for letter in ascii_uppercase:
            drive = f"{letter}:\\"
            if Path(drive).exists():
                drives.append(drive)
        return drives

    mnt_root = Path("/mnt")
    if not mnt_root.exists():
        return drives
    for child in sorted(mnt_root.iterdir(), key=lambda item: item.name.lower()):
        if child.is_dir() and len(child.name) == 1 and child.name.isalpha():
            drives.append(str(child))
    return drives


def visual_path_roots() -> list[tuple[str, str]]:
    """Return `(display_label, path)` tuples for the sidebar shortcuts.

    Labels are resolved against the active UI language each time this
    function is called — the picker is rebuilt on every `pick_visual_path`
    invocation, so it always renders in the language the user chose.
    """
    roots: list[tuple[str, str]] = []
    home = str(Path.home())

    if running_in_wsl():
        roots.extend(
            [
                (t("path_picker.sidebar.wsl_home"), home),
                (t("path_picker.sidebar.wsl_root"), "/"),
            ]
        )
        for drive in _available_windows_drives():
            roots.append((f"Windows {Path(drive).name.upper()}:", drive))
        return roots

    if running_on_windows():
        roots.append((t("path_picker.sidebar.windows_home"), home))
        for drive in _available_windows_drives():
            roots.append((f"Windows {drive[0].upper()}:", drive))
        return roots

    return [
        (t("path_picker.sidebar.linux_home"), home),
        (t("path_picker.sidebar.linux_root"), "/"),
    ]


def _normalize_existing_directory(path: str | Path) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    normalized = Path(
        to_windows_path(raw) if running_on_windows() else to_posix_path(raw)
    )
    if normalized.exists():
        return str(normalized if normalized.is_dir() else normalized.parent)
    return ""


def _dedupe_existing_directories(paths: list[str | Path]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in paths:
        candidate = _normalize_existing_directory(item)
        if not candidate:
            continue
        key = candidate.replace("\\", "/").rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    return ordered


def best_visual_root_for_path(path: str | Path) -> str:
    raw = str(path).strip()
    roots = visual_path_roots()
    if not raw:
        return roots[0][1] if roots else str(Path.home())
    normalized = raw.replace("\\", "/").rstrip("/").lower()
    best_root = ""
    best_len = -1
    for _label, root in roots:
        candidate = str(root).replace("\\", "/").rstrip("/").lower()
        if normalized.startswith(candidate) and len(candidate) > best_len:
            best_root = root
            best_len = len(candidate)
    return best_root or (roots[0][1] if roots else str(Path.home()))


def _recent_directories() -> list[str]:
    settings = _settings()
    settings.beginGroup(_SETTINGS_GROUP)
    try:
        raw = settings.value(_RECENT_DIRS_KEY, [])
    finally:
        settings.endGroup()

    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple)):
        values = [str(item) for item in raw]
    else:
        values = []
    return _dedupe_existing_directories(values)


def _save_recent_directory(path: str | Path) -> None:
    current = _recent_directories()
    updated = _dedupe_existing_directories([path, *current])[:_MAX_RECENT_DIRS]
    settings = _settings()
    settings.beginGroup(_SETTINGS_GROUP)
    try:
        settings.setValue(_RECENT_DIRS_KEY, updated)
    finally:
        settings.endGroup()


def _restore_dialog_geometry(dialog: QFileDialog) -> None:
    settings = _settings()
    settings.beginGroup(_SETTINGS_GROUP)
    try:
        raw = settings.value(_GEOMETRY_KEY)
    finally:
        settings.endGroup()

    if isinstance(raw, QByteArray) and not raw.isEmpty():
        dialog.restoreGeometry(raw)
        return
    if isinstance(raw, (bytes, bytearray)) and raw:
        dialog.restoreGeometry(QByteArray(raw))
        return
    dialog.resize(1040, 720)


def _save_dialog_geometry(dialog: QFileDialog) -> None:
    settings = _settings()
    settings.beginGroup(_SETTINGS_GROUP)
    try:
        settings.setValue(_GEOMETRY_KEY, dialog.saveGeometry())
    finally:
        settings.endGroup()


def _existing_sidebar_urls(initial_path: str = "") -> list[QUrl]:
    urls: list[QUrl] = []
    sidebar_paths = _dedupe_existing_directories(
        [root for _label, root in visual_path_roots()]
        + _recent_directories()
        + [initial_path]
    )
    for root in sidebar_paths:
        urls.append(QUrl.fromLocalFile(str(root)))
    return urls


def _normalize_initial_directory(initial_path: str | Path) -> str:
    raw = str(initial_path).strip()
    if not raw:
        return best_visual_root_for_path("")
    if running_on_windows():
        path = Path(to_windows_path(raw))
    else:
        path = Path(to_posix_path(raw))
    if path.exists():
        return str(path if path.is_dir() else path.parent)
    return best_visual_root_for_path(raw)


def _selected_local_path(dialog: QFileDialog) -> str:
    selected = dialog.selectedFiles()
    if selected:
        return selected[0]
    return dialog.directory().absolutePath()


def _configure_dialog_appearance(dialog: QFileDialog) -> None:
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setViewMode(QFileDialog.ViewMode.Detail)
    dialog.setIconProvider(QFileIconProvider())
    dialog.setLabelText(QFileDialog.DialogLabel.LookIn, t("path_picker.label.look_in"))
    dialog.setLabelText(
        QFileDialog.DialogLabel.FileName, t("path_picker.label.file_name")
    )
    dialog.setLabelText(
        QFileDialog.DialogLabel.FileType, t("path_picker.label.file_type")
    )
    dialog.setLabelText(QFileDialog.DialogLabel.Accept, t("path_picker.label.accept"))
    dialog.setLabelText(QFileDialog.DialogLabel.Reject, t("path_picker.label.reject"))
    # Pick the palette at dialog-open time rather than subscribing
    # to theme_mode changes — these dialogs are short-lived (modal
    # exec loop), so the palette active when the user clicks "Pick…"
    # is good enough.  The global QApplication stylesheet already
    # handles every non-QFileDialog widget; these rules only exist
    # to tune the file browser chrome (tree/list selection colors,
    # button min sizes) which QFileDialog paints via its own sub-
    # stylesheet override.
    if current_theme_mode() == "dark":
        dialog_bg = "#1a1f26"
        view_bg = "#222831"
        view_border = "#3e4754"
        sel_bg = "#1e4870"
        sel_fg = "#ecf4fb"
    else:
        dialog_bg = "#f5f8fc"
        view_bg = "#ffffff"
        view_border = "#d5e0ee"
        sel_bg = "#dbe9fb"
        sel_fg = "#22364f"
    dialog.setStyleSheet(
        f"""
        QFileDialog {{
            background: {dialog_bg};
        }}
        QFileDialog QTreeView,
        QFileDialog QListView {{
            background: {view_bg};
            border: 1px solid {view_border};
            border-radius: 10px;
            padding: 2px;
            selection-background-color: {sel_bg};
            selection-color: {sel_fg};
        }}
        QFileDialog QLineEdit,
        QFileDialog QComboBox {{
            min-height: 32px;
        }}
        QFileDialog QPushButton {{
            min-height: 34px;
            min-width: 96px;
        }}
        """
    )

    for tree in dialog.findChildren(QTreeView):
        tree.setAlternatingRowColors(True)
        tree.setUniformRowHeights(True)
        tree.setSortingEnabled(True)
        tree.setIconSize(QSize(18, 18))
        header = tree.header()
        if header is not None:
            header.setStretchLastSection(False)
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)

    for view in dialog.findChildren(QListView):
        view.setUniformItemSizes(True)
        view.setIconSize(QSize(24, 24))


def pick_visual_path(
    parent: QWidget | None,
    *,
    title: str,
    mode: str,
    filter_spec: str = "All files (*)",
    initial_path: str = "",
) -> str:
    start = _normalize_initial_directory(initial_path)
    dialog = QFileDialog(parent, title, start, filter_spec)
    _configure_dialog_appearance(dialog)
    _restore_dialog_geometry(dialog)
    dialog.setSidebarUrls(_existing_sidebar_urls(start))
    dialog.setNameFilter(filter_spec)
    dialog.setHistory(_dedupe_existing_directories([start, *_recent_directories()]))
    initial_candidate = Path(start)
    if initial_candidate.exists() and initial_candidate.is_file():
        dialog.selectFile(str(initial_candidate))

    choose_current_dir = {"enabled": False}
    if mode == "file":
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
    elif mode == "directory":
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
    elif mode == "file_or_directory":
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        button_box = dialog.findChild(QDialogButtonBox)
        if button_box is not None:
            choose_dir_btn = button_box.addButton(
                t("path_picker.button.choose_current_folder"),
                QDialogButtonBox.ButtonRole.ActionRole,
            )

            def _choose_directory() -> None:
                choose_current_dir["enabled"] = True
                dialog.accept()

            choose_dir_btn.clicked.connect(_choose_directory)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    accepted = dialog.exec() == int(QFileDialog.DialogCode.Accepted)
    _save_dialog_geometry(dialog)

    if not accepted:
        return ""

    if choose_current_dir["enabled"]:
        selected_directory = dialog.directory().absolutePath()
        _save_recent_directory(selected_directory)
        return selected_directory

    resolved = _selected_local_path(dialog)
    _save_recent_directory(resolved)
    if mode == "directory":
        return resolved
    if mode == "file":
        return resolved if Path(resolved).is_file() else ""
    if Path(resolved).exists():
        return resolved
    return dialog.directory().absolutePath()
