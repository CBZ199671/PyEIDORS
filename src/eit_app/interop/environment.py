"""Cross-platform MATLAB + EIDORS environment detection helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from .models import EidorsEnvironment, InteropCapabilityReport

_SETTINGS_GROUP = "interop_hub"
_PROFILES_KEY = "profiles"
_LAST_ENV_KEY = "last_environment"
_LINUX_STARTUP_LIMIT = 24
_WINDOWS_STARTUP_LIMIT = 24


def _powershell_binary() -> str | None:
    for candidate in ("powershell.exe", "pwsh.exe", "powershell", "pwsh"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def running_in_wsl() -> bool:
    try:
        release = os.uname().release.lower()
    except AttributeError:
        return False
    return "microsoft" in release


def running_on_windows() -> bool:
    return os.name == "nt"


def _is_windows_style_path(raw: str | Path) -> bool:
    text = str(raw).strip()
    if len(text) >= 2 and text[1] == ":":
        return True
    return text.startswith("\\\\")


def _normalized_path_key(raw: str | Path) -> str:
    text = str(raw).strip().replace("\\", "/").rstrip("/")
    return text.lower()


def to_windows_path(path: str | Path) -> str:
    raw = str(path).strip()
    if not raw:
        return raw
    if _is_windows_style_path(raw):
        return raw.replace("/", "\\")
    try:
        result = subprocess.run(
            ["wslpath", "-w", raw],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or raw
    except Exception:
        return raw


def to_posix_path(path: str | Path) -> str:
    raw = str(path).strip()
    if not raw:
        return raw
    if raw.startswith("/"):
        return raw
    if raw.startswith("\\\\wsl.localhost\\") or raw.startswith("\\\\wsl$\\"):
        normalized = raw.replace("\\", "/")
        parts = normalized.split("/")
        if len(parts) >= 5:
            return "/" + "/".join(parts[4:])
    try:
        result = subprocess.run(
            ["wslpath", raw],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or raw
    except Exception:
        return raw


def _decode_process_bytes(raw: bytes) -> str:
    if not raw:
        return ""

    # Only treat the payload as UTF-16 when it actually looks like UTF-16.
    looks_utf16 = raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:8]
    encodings = ["utf-8"]
    if looks_utf16:
        encodings.extend(["utf-16", "utf-16-le", "utf-16-be"])
    encodings.extend(["gbk", "cp936", "latin-1"])

    for encoding in encodings:
        try:
            return raw.decode(encoding).strip()
        except Exception:
            continue
    return ""


def _run_powershell_capture(
    script: str, *, timeout: float | None = None
) -> tuple[int, str, str]:
    binary = _powershell_binary()
    if not binary:
        return 127, "", "PowerShell 不可用。"
    command = [binary, "-NoProfile", "-Command", script]
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return 124, "", "PowerShell 调用超时。"
    return (
        result.returncode,
        _decode_process_bytes(result.stdout),
        _decode_process_bytes(result.stderr),
    )


def _run_powershell_lines(script: str, *, timeout: float | None = None) -> list[str]:
    code, stdout, _ = _run_powershell_capture(script, timeout=timeout)
    if code != 0 or not stdout:
        return []
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def _run_command_capture(
    command: list[str], *, timeout: float | None = None
) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            command, capture_output=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return 124, "", "命令执行超时。"
    except Exception as exc:
        return 127, "", str(exc)
    return (
        result.returncode,
        _decode_process_bytes(result.stdout),
        _decode_process_bytes(result.stderr),
    )


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _runtime_label(runtime_kind: str) -> str:
    labels = {
        "wsl-bridged": "WSL 桥接",
        "linux-native": "Linux 原生",
        "windows-host": "Windows 原生",
    }
    return labels.get(runtime_kind, runtime_kind or "未标注")


def _guess_host_os_from_path(path: str | Path) -> str:
    raw = str(path).strip()
    if not raw:
        return ""
    if _is_windows_style_path(raw):
        return "windows"
    return "linux"


def matlab_command_for_execution(environment: EidorsEnvironment) -> str:
    raw = environment.matlab_command.strip()
    if not raw:
        return raw
    host_os = environment.matlab_host_os or _guess_host_os_from_path(raw)
    if host_os == "windows" and running_in_wsl():
        converted = to_posix_path(raw)
        return converted or raw
    return raw


def matlab_runtime_path(path: str | Path, environment: EidorsEnvironment) -> str:
    raw = str(path).strip()
    if not raw:
        return raw
    matlab_host_os = environment.matlab_host_os or _guess_host_os_from_path(
        environment.matlab_command
    )
    if matlab_host_os == "windows":
        return to_windows_path(raw)
    return to_posix_path(raw)


def infer_startup_from_source_path(source_path: str | Path) -> str:
    raw = str(source_path).strip()
    if not raw:
        return ""
    source = Path(to_posix_path(raw))
    anchor = source if source.is_dir() else source.parent
    candidates: list[Path] = []
    for directory in (anchor, *anchor.parents[:5]):
        candidates.extend(
            [
                directory / "startup.m",
                directory / "eidors" / "startup.m",
            ]
        )
        try:
            for child in directory.iterdir():
                if child.is_dir() and "eidors" in child.name.lower():
                    candidates.append(child / "startup.m")
        except Exception:
            continue
    seen: set[str] = set()
    for candidate in candidates:
        key = _normalized_path_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and "eidors" in str(candidate).lower():
            if _is_windows_style_path(raw):
                return to_windows_path(candidate)
            return str(candidate)
    return ""


def _path_affinity(left: str | Path, right: str | Path) -> int:
    left_parts = [part for part in _normalized_path_key(left).split("/") if part]
    right_parts = [part for part in _normalized_path_key(right).split("/") if part]
    score = 0
    for lhs, rhs in zip(left_parts, right_parts, strict=False):
        if lhs != rhs:
            break
        score += 2
    if left_parts and right_parts and left_parts[0] == right_parts[0]:
        score += 4
    return score


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = _normalized_path_key(text)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _limited_find_startups(root: Path, *, max_depth: int, limit: int) -> list[str]:
    if not root.exists():
        return []
    if root.is_file():
        if root.name == "startup.m" and "eidors" in str(root).lower():
            return [str(root)]
        return []

    root = root.resolve()
    base_depth = len(root.parts)
    results: list[str] = []
    for current_root, dirs, files in os.walk(root):
        current_path = Path(current_root)
        depth = len(current_path.parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        dirs[:] = [item for item in dirs if "onedrive" not in item.lower()]
        if "startup.m" not in files:
            continue
        candidate = current_path / "startup.m"
        if "eidors" not in str(candidate).lower():
            continue
        results.append(str(candidate))
        if len(results) >= limit:
            break
    return results


def _toolbox_startups(toolbox_root: Path, *, limit: int = 8) -> list[Path]:
    if not toolbox_root.exists():
        return []
    candidates: list[Path] = []
    for pattern in (
        "eidors/startup.m",
        "eidors*/startup.m",
        "*eidors*/startup.m",
        "*/eidors/startup.m",
        "*/eidors*/startup.m",
    ):
        candidates.extend(toolbox_root.glob(pattern))
    if len(candidates) < limit:
        candidates.extend(
            Path(item)
            for item in _limited_find_startups(toolbox_root, max_depth=4, limit=limit)
        )

    seen: set[str] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        key = _normalized_path_key(candidate)
        if key in seen or not candidate.exists():
            continue
        seen.add(key)
        ordered.append(candidate)
        if len(ordered) >= limit:
            break
    return ordered


def _detect_linux_candidates() -> dict[str, Any]:
    matlab_items: list[dict[str, str]] = []
    matlab_seen: set[str] = set()

    def add_matlab(executable: str | Path, root: str | Path | None = None) -> None:
        executable_text = str(executable).strip()
        if not executable_text:
            return
        key = _normalized_path_key(executable_text)
        if key in matlab_seen:
            return
        matlab_seen.add(key)
        root_path = str(root or Path(executable_text).parent.parent)
        name = Path(root_path).name or "MATLAB"
        matlab_items.append(
            {
                "name": name,
                "matlab_command": executable_text,
                "matlab_root": root_path,
            }
        )

    matlab_root = os.environ.get("MATLABROOT", "").strip()
    if matlab_root:
        executable = Path(matlab_root) / "bin" / "matlab"
        if executable.exists():
            add_matlab(executable, matlab_root)

    resolved_matlab = shutil.which("matlab")
    if resolved_matlab:
        add_matlab(resolved_matlab)

    for parent in (
        Path("/usr/local/MATLAB"),
        Path("/opt/MATLAB"),
        Path.home() / "MATLAB",
        Path.home() / "apps" / "MATLAB",
    ):
        if not parent.exists():
            continue
        if (parent / "bin" / "matlab").exists():
            add_matlab(parent / "bin" / "matlab", parent)
            continue
        try:
            for child in sorted(parent.iterdir(), reverse=True):
                executable = child / "bin" / "matlab"
                if executable.exists():
                    add_matlab(executable, child)
        except Exception:
            continue

    search_roots: list[Path] = []

    def add_root(candidate: str | Path) -> None:
        if not candidate:
            return
        root = Path(candidate).expanduser()
        if root.exists() and "onedrive" not in str(root).lower():
            search_roots.append(root)

    for env_name in ("EIDORS_HOME", "EIDORS_ROOT"):
        add_root(os.environ.get(env_name, ""))
    for item in os.environ.get("MATLABPATH", "").split(os.pathsep):
        if item and "eidors" in item.lower():
            add_root(item)
    for candidate in (
        Path.home() / "workspace",
        Path.home() / "src",
        Path.home() / "source",
        Path.home() / "projects",
        Path.home() / "repos",
        Path.home() / "GitHub",
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        Path.home() / "Documents",
    ):
        add_root(candidate)

    for base in (Path("/opt"), Path("/usr/local")):
        if not base.exists():
            continue
        for pattern in ("eidors*", "*eidors*"):
            for candidate in base.glob(pattern):
                add_root(candidate)

    toolbox_startups: list[str] = []
    for matlab in matlab_items:
        toolbox_root = Path(str(matlab.get("matlab_root", ""))) / "toolbox"
        for candidate in _toolbox_startups(toolbox_root):
            toolbox_startups.append(str(candidate))

    startups: list[str] = _unique_strings(toolbox_startups)
    startup_strategy = "toolbox" if startups else "manual"
    if not startups:
        for root in search_roots:
            direct = root / "startup.m"
            if direct.exists() and "eidors" in str(direct).lower():
                startups.append(str(direct))
            startups.extend(
                _limited_find_startups(root, max_depth=5, limit=_LINUX_STARTUP_LIMIT)
            )
        startups = _unique_strings(startups)
        startup_strategy = "broadened" if startups else "manual"

    octave_commands = [
        item for item in (shutil.which("octave"), shutil.which("octave-cli")) if item
    ]
    return {
        "matlab": matlab_items,
        "startups": startups,
        "searched_roots": _unique_strings([str(item) for item in search_roots]),
        "octave_commands": _unique_strings(octave_commands),
        "startup_strategy": startup_strategy,
    }


def _mounted_windows_drives() -> list[Path]:
    mnt_root = Path("/mnt")
    if not mnt_root.exists():
        return []
    drives: list[Path] = []
    for child in sorted(mnt_root.iterdir()):
        if child.is_dir() and len(child.name) == 1 and child.name.isalpha():
            drives.append(child)
    return drives


def _windows_user_home_from_wsl() -> Path | None:
    for script in (
        "[Environment]::GetFolderPath('UserProfile')",
        "$env:USERPROFILE",
    ):
        values = _run_powershell_lines(script)
        if not values:
            continue
        posix = to_posix_path(values[0])
        candidate = Path(posix)
        if candidate.exists():
            return candidate
    return None


def _detect_windows_candidates_from_wsl() -> dict[str, Any]:
    drives = _mounted_windows_drives()
    matlab_items: list[dict[str, str]] = []
    seen_matlab: set[str] = set()

    def add_matlab(executable: Path) -> None:
        if not executable.exists():
            return
        command = to_windows_path(executable)
        key = _normalized_path_key(command)
        if key in seen_matlab:
            return
        seen_matlab.add(key)
        matlab_root = to_windows_path(executable.parent.parent)
        matlab_items.append(
            {
                "name": executable.parent.parent.name or "MATLAB",
                "matlab_command": command,
                "matlab_root": matlab_root,
            }
        )

    for source in _run_powershell_lines(
        "Get-Command matlab.exe -All -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty Source -Unique"
    ):
        source_path = Path(to_posix_path(source))
        if source_path.exists():
            add_matlab(source_path)

    for drive in drives:
        for pattern in (
            "Program Files/MATLAB/*/bin/matlab.exe",
            "MATLAB/*/bin/matlab.exe",
            "Apps/MATLAB/*/bin/matlab.exe",
            "Tools/MATLAB/*/bin/matlab.exe",
        ):
            for executable in sorted(drive.glob(pattern), reverse=True):
                add_matlab(executable)

    search_roots: list[Path] = []

    def add_root(candidate: Path) -> None:
        if candidate.exists() and "onedrive" not in str(candidate).lower():
            search_roots.append(candidate)

    for env_name in ("EIDORS_HOME", "EIDORS_ROOT", "MATLABROOT"):
        raw = next(iter(_run_powershell_lines(f"$env:{env_name}")), "")
        if raw:
            add_root(Path(to_posix_path(raw)))

    user_home = _windows_user_home_from_wsl()
    if user_home is not None:
        for relative in (
            "workspace",
            "source",
            "src",
            "projects",
            "repos",
            "GitHub",
            "Desktop",
            "Downloads",
            "Documents/MATLAB",
            "Documents/GitHub",
            "Documents/Projects",
            "Documents/workspace",
        ):
            add_root(user_home / relative)

    for drive in drives:
        for relative in ("workspace", "source", "src", "projects", "repos", "GitHub"):
            add_root(drive / relative)

    ordered_roots = []
    seen_roots: set[str] = set()
    for root in search_roots:
        key = _normalized_path_key(root)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        ordered_roots.append(root)

    toolbox_startups: list[str] = []
    for matlab in matlab_items:
        toolbox_root = (
            Path(to_posix_path(str(matlab.get("matlab_root", "")))) / "toolbox"
        )
        for candidate in _toolbox_startups(toolbox_root):
            toolbox_startups.append(to_windows_path(candidate))

    startups: list[str] = _unique_strings(toolbox_startups)
    startup_strategy = "toolbox" if startups else "manual"
    if not startups:
        for root in ordered_roots:
            direct = root / "startup.m"
            if direct.exists() and "eidors" in str(direct).lower():
                startups.append(to_windows_path(direct))
            for candidate in _limited_find_startups(root, max_depth=5, limit=8):
                startups.append(to_windows_path(candidate))
            startups = _unique_strings(startups)
            if len(startups) >= 4:
                break
        startup_strategy = "broadened" if startups else "manual"

    return {
        "matlab": matlab_items,
        "startups": startups,
        "searched_roots": _unique_strings(
            [to_windows_path(item) for item in ordered_roots]
        ),
        "drive_roots": _unique_strings([to_windows_path(item) for item in drives]),
        "startup_strategy": startup_strategy,
    }


def _detect_windows_candidates() -> dict[str, Any]:
    if running_in_wsl():
        return _detect_windows_candidates_from_wsl()

    drive_roots = _unique_strings(
        _run_powershell_lines(
            "Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue | "
            "Select-Object -ExpandProperty Root"
        )
    )
    if not drive_roots:
        drive_roots = [r"C:\\"]

    matlab_items: list[dict[str, str]] = []
    seen_matlab: set[str] = set()

    def add_matlab(path: str) -> None:
        if not path or "onedrive" in path.lower():
            return
        key = _normalized_path_key(path)
        if key in seen_matlab:
            return
        seen_matlab.add(key)
        if "\\" in path:
            root = path.rsplit("\\", 2)[0]
        else:
            root = str(Path(path).parent.parent)
        if root.lower().endswith("\\bin") or root.lower().endswith("/bin"):
            root = str(Path(root).parent)
        name = Path(to_posix_path(root)).name or "MATLAB"
        matlab_items.append(
            {
                "name": name,
                "matlab_command": path,
                "matlab_root": root,
            }
        )

    for source in _run_powershell_lines(
        "Get-Command matlab.exe -All -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty Source -Unique"
    ):
        add_matlab(source)

    drive_literal = ", ".join(f"'{drive}'" for drive in drive_roots)
    matlab_scan_script = f"""
    $drives = @({drive_literal})
    foreach ($drive in $drives) {{
      foreach ($relative in @('Program Files\\MATLAB', 'MATLAB', 'Apps\\MATLAB', 'Tools\\MATLAB')) {{
        $base = Join-Path $drive $relative
        if (-not (Test-Path $base)) {{ continue }}
        Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
          Sort-Object Name -Descending |
          ForEach-Object {{
            $exe = Join-Path $_.FullName 'bin\\matlab.exe'
            if (Test-Path $exe) {{ $exe }}
          }}
      }}
    }}
    """
    for source in _run_powershell_lines(matlab_scan_script):
        add_matlab(source)

    search_roots: list[str] = []

    def add_search_root(raw: str) -> None:
        text = str(raw).strip()
        if not text or "onedrive" in text.lower():
            return
        search_roots.append(text)

    for raw in _run_powershell_lines(
        "@($env:EIDORS_HOME, $env:EIDORS_ROOT, $env:MATLABROOT) | "
        "Where-Object { $_ -and (Test-Path $_) -and ($_ -notmatch 'OneDrive') }"
    ):
        add_search_root(raw)

    user_home = next(
        iter(_run_powershell_lines("[Environment]::GetFolderPath('UserProfile')")), ""
    )
    if user_home:
        for relative in (
            "workspace",
            "source",
            "src",
            "projects",
            "repos",
            "GitHub",
            "Desktop",
            "Downloads",
            r"Documents\MATLAB",
            r"Documents\GitHub",
            r"Documents\Projects",
            r"Documents\workspace",
        ):
            add_search_root(f"{user_home}\\{relative}")

    search_roots = _unique_strings(search_roots)
    roots_literal = ", ".join(
        "'" + root.replace("'", "''") + "'" for root in search_roots
    )
    startup_literals: list[str] = []
    for matlab in matlab_items:
        toolbox_root = (
            Path(to_posix_path(str(matlab.get("matlab_root", "")))) / "toolbox"
        )
        for candidate in _toolbox_startups(toolbox_root):
            startup_literals.append(to_windows_path(candidate))
    startups = _unique_strings(startup_literals)
    startup_strategy = "toolbox" if startups else "manual"
    if not startups:
        startup_scan_script = f"""
        $roots = @({roots_literal}) | Where-Object {{ $_ -and (Test-Path $_) -and ($_ -notmatch 'OneDrive') }}
        foreach ($root in $roots) {{
          $direct = Join-Path $root 'startup.m'
          if (Test-Path $direct -and $direct -match 'eidors' -and $direct -notmatch 'OneDrive') {{
            $direct
          }}
          Get-ChildItem -Path $root -Filter startup.m -Recurse -ErrorAction SilentlyContinue |
            Where-Object {{ $_.FullName -match 'eidors' -and $_.FullName -notmatch 'OneDrive' }} |
            Select-Object -First {_WINDOWS_STARTUP_LIMIT} |
            Select-Object -ExpandProperty FullName
        }}
        """
        startups = _unique_strings(
            _run_powershell_lines(startup_scan_script) if search_roots else []
        )
        startup_strategy = "broadened" if startups else "manual"

    return {
        "matlab": matlab_items,
        "startups": startups,
        "searched_roots": search_roots,
        "drive_roots": drive_roots,
        "startup_strategy": startup_strategy,
    }


def _build_environment_name(
    matlab_name: str, runtime_kind: str, *, has_matlab: bool, has_startup: bool
) -> str:
    runtime = _runtime_label(runtime_kind)
    if has_matlab and has_startup:
        return f"{matlab_name} / EIDORS（{runtime}）"
    if has_matlab:
        return f"{matlab_name}（{runtime}，startup 待确认）"
    if has_startup:
        return f"EIDORS startup（{runtime}，MATLAB 待确认）"
    return f"EIDORS 环境（{runtime}）"


def _best_matching_startup(anchor: str, startup_items: list[str]) -> str:
    if not startup_items:
        return ""
    if not anchor:
        return startup_items[0]
    return max(startup_items, key=lambda candidate: _path_affinity(anchor, candidate))


def _pair_candidates(
    matlab_items: list[dict[str, str]],
    startup_items: list[str],
    *,
    host_os: str,
    runtime_kind: str,
    source: str,
) -> list[EidorsEnvironment]:
    environments: list[EidorsEnvironment] = []
    used_startups: set[str] = set()
    for matlab in matlab_items:
        command = str(matlab.get("matlab_command", "")).strip()
        root = str(matlab.get("matlab_root", "")).strip()
        startup = _best_matching_startup(root or command, startup_items)
        if startup:
            used_startups.add(_normalized_path_key(startup))
        matlab_name = (
            str(matlab.get("name", "")).strip()
            or Path(to_posix_path(root or command)).name
            or "MATLAB"
        )
        environments.append(
            EidorsEnvironment(
                name=_build_environment_name(
                    matlab_name,
                    runtime_kind,
                    has_matlab=bool(command),
                    has_startup=bool(startup),
                ),
                matlab_command=command,
                matlab_root=root,
                eidors_startup=startup,
                source=source,
                matlab_host_os=host_os,
                startup_host_os=host_os if startup else "",
                runtime_kind=runtime_kind,
            )
        )

    if not matlab_items:
        for startup in startup_items[:4]:
            environments.append(
                EidorsEnvironment(
                    name=_build_environment_name(
                        "EIDORS startup",
                        runtime_kind,
                        has_matlab=False,
                        has_startup=True,
                    ),
                    eidors_startup=startup,
                    source=source,
                    startup_host_os=host_os,
                    runtime_kind=runtime_kind,
                )
            )
        return environments

    for startup in startup_items:
        key = _normalized_path_key(startup)
        if key in used_startups:
            continue
        environments.append(
            EidorsEnvironment(
                name=_build_environment_name(
                    "EIDORS startup", runtime_kind, has_matlab=False, has_startup=True
                ),
                eidors_startup=startup,
                source=source,
                startup_host_os=host_os,
                runtime_kind=runtime_kind,
            )
        )
        if len(environments) >= len(matlab_items) + 2:
            break
    return environments


def _dedupe_environments(
    environments: list[EidorsEnvironment],
) -> list[EidorsEnvironment]:
    seen: set[tuple[str, str, str]] = set()
    ordered: list[EidorsEnvironment] = []
    for environment in environments:
        key = (
            _normalized_path_key(environment.matlab_command),
            _normalized_path_key(environment.eidors_startup),
            environment.runtime_kind,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(environment)
    return ordered


def _environment_sort_key(environment: EidorsEnvironment) -> tuple[int, int, str]:
    completeness = int(bool(environment.matlab_command)) + int(
        bool(environment.eidors_startup)
    )
    if running_in_wsl():
        runtime_priority = {
            "wsl-bridged": 0,
            "linux-native": 1,
            "windows-host": 2,
        }.get(environment.runtime_kind, 3)
    elif running_on_windows():
        runtime_priority = {
            "windows-host": 0,
            "linux-native": 1,
            "wsl-bridged": 2,
        }.get(environment.runtime_kind, 3)
    else:
        runtime_priority = {
            "linux-native": 0,
            "wsl-bridged": 1,
            "windows-host": 2,
        }.get(environment.runtime_kind, 3)
    return (-completeness, runtime_priority, environment.name.lower())


class InteropSettingsStore:
    """Persist interop profiles and recent paths using QSettings."""

    def __init__(self) -> None:
        self._settings = QSettings("PyEIDORS", "EITWorkstation")

    def load_profiles(self) -> list[EidorsEnvironment]:
        self._settings.beginGroup(_SETTINGS_GROUP)
        try:
            raw = self._settings.value(_PROFILES_KEY, "[]")
            payload = json.loads(str(raw or "[]"))
            if not isinstance(payload, list):
                return []
            profiles = []
            for item in payload:
                if isinstance(item, dict):
                    profiles.append(EidorsEnvironment.from_mapping(item))
            return profiles
        finally:
            self._settings.endGroup()

    def save_profiles(self, profiles: list[EidorsEnvironment]) -> None:
        self._settings.beginGroup(_SETTINGS_GROUP)
        try:
            self._settings.setValue(
                _PROFILES_KEY,
                json.dumps(
                    [profile.to_mapping() for profile in profiles], ensure_ascii=True
                ),
            )
        finally:
            self._settings.endGroup()

    def load_last_environment(self) -> EidorsEnvironment | None:
        self._settings.beginGroup(_SETTINGS_GROUP)
        try:
            raw = self._settings.value(_LAST_ENV_KEY, "")
            if not raw:
                return None
            payload = json.loads(str(raw))
            if not isinstance(payload, dict):
                return None
            return EidorsEnvironment.from_mapping(payload)
        except json.JSONDecodeError:
            return None
        finally:
            self._settings.endGroup()

    def save_last_environment(self, environment: EidorsEnvironment) -> None:
        self._settings.beginGroup(_SETTINGS_GROUP)
        try:
            self._settings.setValue(
                _LAST_ENV_KEY,
                json.dumps(environment.to_mapping(), ensure_ascii=True),
            )
        finally:
            self._settings.endGroup()


class EidorsEnvironmentDetector:
    """Detect MATLAB and EIDORS startup locations from the current runtime."""

    def __init__(self, settings_store: InteropSettingsStore | None = None) -> None:
        self._settings_store = settings_store or InteropSettingsStore()

    def load_profiles(self) -> list[EidorsEnvironment]:
        return self._settings_store.load_profiles()

    def save_profiles(self, profiles: list[EidorsEnvironment]) -> None:
        self._settings_store.save_profiles(profiles)

    def save_last_environment(self, environment: EidorsEnvironment) -> None:
        self._settings_store.save_last_environment(environment)

    def load_last_environment(self) -> EidorsEnvironment | None:
        return self._settings_store.load_last_environment()

    def infer_startup_from_source(self, source_path: str | Path) -> str:
        return infer_startup_from_source_path(source_path)

    def detect(self) -> tuple[list[EidorsEnvironment], InteropCapabilityReport]:
        report = InteropCapabilityReport()
        detected: list[EidorsEnvironment] = []

        windows_candidates = _detect_windows_candidates()
        linux_candidates = _detect_linux_candidates()

        detected.extend(
            _pair_candidates(
                windows_candidates.get("matlab", []),
                windows_candidates.get("startups", []),
                host_os="windows",
                runtime_kind="wsl-bridged" if running_in_wsl() else "windows-host",
                source="detected_windows",
            )
        )
        detected.extend(
            _pair_candidates(
                linux_candidates.get("matlab", []),
                linux_candidates.get("startups", []),
                host_os="linux",
                runtime_kind="linux-native",
                source="detected_linux",
            )
        )

        last = self.load_last_environment()
        if last is not None:
            detected.append(last)

        detected = _dedupe_environments(detected)
        detected.sort(key=_environment_sort_key)

        report.can_launch_matlab = any(bool(item.matlab_command) for item in detected)
        report.has_eidors_startup = any(bool(item.eidors_startup) for item in detected)
        report.can_capture_script = any(
            bool(item.matlab_command and item.eidors_startup) for item in detected
        )
        report.matlab_found_count = sum(1 for item in detected if item.matlab_command)
        report.startup_found_count = sum(1 for item in detected if item.eidors_startup)

        startup_strategies = {
            str(windows_candidates.get("startup_strategy", "")),
            str(linux_candidates.get("startup_strategy", "")),
        }
        startup_strategies.discard("")
        report.toolbox_startup_found = "toolbox" in startup_strategies
        report.broadened_search_used = any(
            item in startup_strategies for item in ("broadened", "manual")
        )
        report.broadened_startup_found = "broadened" in startup_strategies
        report.manual_browse_required = (
            report.can_launch_matlab and not report.has_eidors_startup
        )

        if windows_candidates.get("matlab") or windows_candidates.get("startups"):
            report.issues.append(
                "已扫描 Windows 侧 MATLAB/EIDORS 常见位置，包括所有文件系统盘符上的 Program Files/MATLAB、"
                "workspace/source/GitHub/Desktop/Downloads 等目录。"
            )
        if linux_candidates.get("matlab") or linux_candidates.get("startups"):
            report.issues.append(
                "已扫描 Linux/WSL 侧 MATLAB/EIDORS 常见位置，包括 $MATLABROOT、$EIDORS_HOME、"
                "~/workspace、~/src、/opt、/usr/local 等目录。"
            )
        if linux_candidates.get("octave_commands") and not linux_candidates.get(
            "matlab"
        ):
            report.issues.append(
                "检测到 Octave，但当前 v1 互通链仍以 MATLAB 为正式运行端；Octave 先作为后续扩展位保留。"
            )
        if "toolbox" in startup_strategies:
            report.issues.append(
                "已优先从已检测到的 MATLAB 安装目录下检索 toolbox 中的 EIDORS startup.m。"
            )
        if "broadened" in startup_strategies:
            report.issues.append(
                "MATLAB 已找到，但 toolbox 内未命中时，已自动扩大到常见工程目录继续搜索 startup.m。"
            )

        if not report.can_launch_matlab:
            report.issues.append(
                "未自动检测到 MATLAB 安装路径，可在 Profiles & Paths 中手动指定。"
            )
        if not report.has_eidors_startup:
            report.issues.append(
                "未自动检测到 EIDORS startup.m，请手动点击 Browse 指定；选择用户脚本后系统也会再尝试按脚本位置自动反推。"
            )
        if detected and not report.can_capture_script:
            report.issues.append(
                "已检测到部分环境信息，但还缺少 MATLAB 或 startup.m，需手动补全后才能采集脚本。"
            )
        if not detected:
            report.issues.append("当前未检测到任何可用候选环境。")

        return detected, report

    def test_matlab_launch(self, environment: EidorsEnvironment) -> tuple[bool, str]:
        if not environment.matlab_command:
            return False, "尚未配置 MATLAB 可执行文件。"
        command = matlab_command_for_execution(environment)
        code, stdout, stderr = _run_command_capture(
            [command, "-batch", "disp('PYEIDORS_MATLAB_OK');"],
            timeout=45,
        )
        if code == 127 and stderr:
            return False, f"启动 MATLAB 失败: {stderr}"
        if code != 0:
            return False, stderr.strip() or stdout.strip() or "MATLAB 返回非零退出码。"
        return "PYEIDORS_MATLAB_OK" in stdout, stdout.strip() or "MATLAB 已启动。"

    def test_eidors_startup(self, environment: EidorsEnvironment) -> tuple[bool, str]:
        if not environment.matlab_command or not environment.eidors_startup:
            return False, "需要同时配置 MATLAB 路径和 EIDORS startup.m。"
        command = matlab_command_for_execution(environment)
        startup = matlab_runtime_path(environment.eidors_startup, environment)
        escaped_startup = startup.replace("'", "''")
        expression = f"run('{escaped_startup}');disp(exist('eidors_default','file'));"
        code, stdout, stderr = _run_command_capture(
            [command, "-batch", expression], timeout=60
        )
        if code == 127 and stderr:
            return False, f"EIDORS 启动测试失败: {stderr}"
        if code != 0:
            return (
                False,
                stderr.strip() or stdout.strip() or "EIDORS startup 执行失败。",
            )
        return "2" in stdout.split(), stdout.strip() or "EIDORS startup 已运行。"
