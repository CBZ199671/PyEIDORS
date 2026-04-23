from __future__ import annotations

from pathlib import Path

import eit_app.interop.environment as environment_module
from eit_app.interop import EidorsEnvironment


def test_detector_merges_windows_and_linux_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        environment_module,
        "_detect_windows_candidates",
        lambda: {
            "matlab": [
                {
                    "name": "R2023b",
                    "matlab_command": r"D:\Program Files\MATLAB\R2023b\bin\matlab.exe",
                    "matlab_root": r"D:\Program Files\MATLAB\R2023b",
                }
            ],
            "startups": [r"C:\Users\tester\Desktop\eidors\startup.m"],
            "searched_roots": [r"C:\Users\tester\Desktop"],
        },
    )
    monkeypatch.setattr(
        environment_module,
        "_detect_linux_candidates",
        lambda: {
            "matlab": [
                {
                    "name": "R2024a",
                    "matlab_command": "/usr/local/MATLAB/R2024a/bin/matlab",
                    "matlab_root": "/usr/local/MATLAB/R2024a",
                }
            ],
            "startups": ["/opt/eidors/startup.m"],
            "searched_roots": ["/opt"],
            "octave_commands": [],
        },
    )
    detector = environment_module.EidorsEnvironmentDetector()
    monkeypatch.setattr(detector, "load_last_environment", lambda: None)

    environments, report = detector.detect()

    assert report.can_launch_matlab is True
    assert report.has_eidors_startup is True
    assert report.can_capture_script is True
    assert any(item.runtime_kind == "wsl-bridged" for item in environments)
    assert any(item.runtime_kind == "linux-native" for item in environments)
    assert any(item.matlab_host_os == "windows" for item in environments)
    assert any(item.matlab_host_os == "linux" for item in environments)


def test_matlab_command_for_execution_converts_windows_path_in_wsl(monkeypatch) -> None:
    environment = EidorsEnvironment(
        name="Windows MATLAB / EIDORS",
        matlab_command=r"D:\Program Files\MATLAB\R2023b\bin\matlab.exe",
        matlab_host_os="windows",
        runtime_kind="wsl-bridged",
    )
    monkeypatch.setattr(environment_module, "running_in_wsl", lambda: True)
    monkeypatch.setattr(
        environment_module,
        "to_posix_path",
        lambda raw: "/mnt/d/Program Files/MATLAB/R2023b/bin/matlab.exe",
    )

    command = environment_module.matlab_command_for_execution(environment)

    assert command == "/mnt/d/Program Files/MATLAB/R2023b/bin/matlab.exe"


def test_matlab_runtime_path_matches_environment_host(monkeypatch) -> None:
    windows_environment = EidorsEnvironment(
        name="Windows MATLAB / EIDORS",
        matlab_command=r"D:\Program Files\MATLAB\R2023b\bin\matlab.exe",
        matlab_host_os="windows",
        runtime_kind="wsl-bridged",
    )
    linux_environment = EidorsEnvironment(
        name="Linux MATLAB / EIDORS",
        matlab_command="/usr/local/MATLAB/R2024a/bin/matlab",
        matlab_host_os="linux",
        runtime_kind="linux-native",
    )
    monkeypatch.setattr(
        environment_module, "to_windows_path", lambda raw: r"C:\converted\startup.m"
    )
    monkeypatch.setattr(
        environment_module, "to_posix_path", lambda raw: "/converted/startup.m"
    )

    assert (
        environment_module.matlab_runtime_path(
            "/home/tom/eidors/startup.m", windows_environment
        )
        == r"C:\converted\startup.m"
    )
    assert (
        environment_module.matlab_runtime_path(
            r"C:\eidors\startup.m", linux_environment
        )
        == "/converted/startup.m"
    )


def test_infer_startup_from_source_path_finds_nearby_eidors_tree(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "sample_project"
    scripts_dir = project_root / "scripts"
    eidors_dir = project_root / "eidors"
    scripts_dir.mkdir(parents=True)
    eidors_dir.mkdir(parents=True)
    script = scripts_dir / "demo_forward.m"
    startup = eidors_dir / "startup.m"
    script.write_text("% demo", encoding="utf-8")
    startup.write_text("% eidors startup", encoding="utf-8")

    detected = environment_module.infer_startup_from_source_path(script)

    assert detected == str(startup)


def test_toolbox_startups_prioritize_eidors_inside_matlab_toolbox(
    tmp_path: Path,
) -> None:
    toolbox_root = tmp_path / "MATLAB" / "R2025a" / "toolbox"
    startup = toolbox_root / "eidors-v3.11" / "eidors" / "startup.m"
    startup.parent.mkdir(parents=True)
    startup.write_text("% toolbox startup", encoding="utf-8")

    candidates = environment_module._toolbox_startups(toolbox_root)

    assert startup in candidates


def test_run_command_capture_decodes_non_utf8_output(monkeypatch) -> None:
    class _FakeResult:
        returncode = 0
        stdout = "启动成功".encode("gbk")
        stderr = b""

    monkeypatch.setattr(
        environment_module.subprocess, "run", lambda *args, **kwargs: _FakeResult()
    )

    code, stdout, stderr = environment_module._run_command_capture(
        ["fake-matlab", "-batch", "disp('ok')"]
    )

    assert code == 0
    assert stdout == "启动成功"
    assert stderr == ""
