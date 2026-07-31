from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
EASY_INSTALL_DIR = ROOT / "scripts" / "release" / "easy-install"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_v770_easy_installer_templates_are_unsigned_and_keyless() -> None:
    required = (
        "outer-header.sh.in",
        "runtime-common.sh",
        "install.sh",
        "install-from-local-cache.sh",
        "start-pyeidors.sh",
    )
    for name in required:
        assert (EASY_INSTALL_DIR / name).is_file(), name

    cache_installer = (EASY_INSTALL_DIR / "install-from-local-cache.sh").read_text(
        encoding="utf-8"
    )
    assert "--no-check-sigs" in cache_installer
    assert "PYEIDORS_NIX_BIN" in cache_installer
    assert "command -v nix" not in cache_installer
    assert "extra-trusted-public-keys" not in cache_installer
    assert "/etc/nix/nix.conf" not in cache_installer

    cache_builder = _read("scripts/release/build_binary_cache_bundle.sh")
    assert "--no-check-sigs" in cache_builder
    assert "generate-binary-cache-key" not in cache_builder
    assert "store sign" not in cache_builder
    assert "'/^Sig:/d'" in cache_builder

    one_click_builder = _read("scripts/release/build_easy_installers.sh")
    for edition in ("cpu-universal", "nvidia-sm61", "nvidia-modern"):
        assert edition in one_click_builder
    assert "/nix/var/nix/profiles/default/bin/nix" in one_click_builder
    assert '"${nix_candidate%/*}/nix-store"' in one_click_builder


def test_v770_host_tools_and_nix_candidates_are_functionally_validated(
    tmp_path: Path,
) -> None:
    common = EASY_INSTALL_DIR / "runtime-common.sh"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_nix = fake_bin / "nix"
    fake_nix.write_text(
        "#!/bin/sh\nprintf 'nix (Nix) 99.0\\n'\n",
        encoding="utf-8",
    )
    fake_nix.chmod(0o755)

    command = f"""
set -euo pipefail
source {common}
resolved="$(PATH={fake_bin}:/usr/bin:/bin pyeidors_resolve_host_tool tar)"
[ "$resolved" = /usr/bin/tar ] || [ "$resolved" = /bin/tar ]
if pyeidors_validate_nix_candidate {fake_nix}; then
  echo "accepted nix without paired nix-store" >&2
  exit 1
fi
pyeidors_nix_version_at_least 2.4 2.4
pyeidors_nix_version_at_least 2.34.1 2.4
if pyeidors_nix_version_at_least 2.3.16 2.4; then
  echo "accepted unsupported old Nix" >&2
  exit 1
fi
PYEIDORS_ORIGINAL_PATH={fake_bin}:/usr/bin:/bin
export PYEIDORS_ORIGINAL_PATH
resolved_nix="$(pyeidors_find_nix)"
[ "$resolved_nix" = /nix/var/nix/profiles/default/bin/nix ]
"""
    subprocess.run(
        ["/bin/bash", "-c", command],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    outer = (EASY_INSTALL_DIR / "outer-header.sh.in").read_text(encoding="utf-8")
    for tool in ("tar", "zstd", "unzip", "curl", "sha256sum"):
        assert f'pyeidors_resolve_host_tool "{tool}"' in outer
    assert "pyeidors_probe_archive_tools" in outer


def test_v770_outer_installer_ignores_broken_path_archive_shims(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    for name in ("tar", "zstd", "unzip", "curl", "sha256sum"):
        path = fake_bin / name
        path.write_text(
            f"#!/bin/sh\necho BROKEN_PATH_{name} >&2\nexit 97\n",
            encoding="utf-8",
        )
        path.chmod(0o755)

    template = (EASY_INSTALL_DIR / "outer-header.sh.in").read_text(encoding="utf-8")
    common = (EASY_INSTALL_DIR / "runtime-common.sh").read_text(encoding="utf-8")
    common = common.removeprefix("#!/usr/bin/env bash\n")
    rendered = (
        template.replace("# @RUNTIME_COMMON@", common.rstrip())
        .replace("@BUNDLE_NAME@", "fixture")
        .replace("@EDITION_NAME_ZH@", "测试版")
        .replace("@VERSION@", "0")
        .replace("@PAYLOAD_SHA256@", "0" * 64)
        .replace("@MIN_TMP_GIB@", "1")
    )
    installer = tmp_path / "fixture.run"
    installer.write_text(rendered, encoding="utf-8")

    result = subprocess.run(
        ["/bin/bash", str(installer)],
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "PYEIDORS_PREREQ_ONLY": "1",
        },
    )
    assert "prerequisite-only" in result.stdout
    assert "BROKEN_PATH" not in result.stderr


def test_v771_launcher_and_nix_wrappers_isolate_host_python_cuda() -> None:
    launcher = (EASY_INSTALL_DIR / "start-pyeidors.sh").read_text(encoding="utf-8")
    common = (EASY_INSTALL_DIR / "runtime-common.sh").read_text(encoding="utf-8")
    assert "pyeidors_clean_runtime_environment" in launcher
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDACXX",
        "PETSC_DIR",
        "SLEPC_DIR",
        "LD_PRELOAD",
    ):
        assert re.search(rf"\bunset\b[^\n]*\b{name}\b", common), name

    flake = _read("flake.nix")
    assert re.search(r'"--set"\s+"PYTHONNOUSERSITE"\s+"1"', flake)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "LD_PRELOAD",
        "CMAKE_PREFIX_PATH",
    ):
        assert re.search(rf'"--unset"\s+"{name}"', flake), name
    for name in ("CUDA_HOME", "CUDA_PATH", "CUDACXX", "PETSC_DIR", "SLEPC_DIR"):
        set_pattern = rf'"--set"\s+"{name}"'
        default_pattern = rf'"--set-default"\s+"{name}"'
        assert re.search(set_pattern, flake), name
        assert not re.search(default_pattern, flake), name


def test_v771_installer_scrubs_host_environment_before_nix_probe(
    tmp_path: Path,
) -> None:
    outer = (EASY_INSTALL_DIR / "outer-header.sh.in").read_text(encoding="utf-8")
    installer = (EASY_INSTALL_DIR / "install.sh").read_text(encoding="utf-8")
    assert outer.index("\npyeidors_clean_runtime_environment\n") < outer.index(
        "\ninstall_or_repair_unpack_tools\n"
    )
    assert installer.index("pyeidors_clean_runtime_environment") < installer.index(
        "install_nix_if_needed"
    )

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    shutil.copy2(EASY_INSTALL_DIR / "install.sh", fixture)
    shutil.copy2(EASY_INSTALL_DIR / "runtime-common.sh", fixture)
    (fixture / "edition.conf").write_text(
        "\n".join(
            (
                "VERSION='2.0.0'",
                "EDITION_ID='cpu-universal'",
                "EDITION_NAME_ZH='CPU 通用版'",
                "MIN_HOME_GIB=1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["/bin/bash", str(fixture / "install.sh")],
        check=False,
        text=True,
        capture_output=True,
        env={
            "HOME": str(tmp_path / "home"),
            "USER": "test-user",
            "LOGNAME": "test-user",
            "PATH": "/usr/bin:/bin",
            "LD_PRELOAD": "/lib/x86_64-linux-gnu/libm.so.6",
        },
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "检测到兼容 Nix" in output
    assert "旧版或损坏的 Nix" not in output
    assert "install-from-local-cache.sh" in output


def test_v771_installed_launcher_actually_scrubs_host_environment(
    tmp_path: Path,
) -> None:
    install_root = tmp_path / "PyEIDORS"
    package = tmp_path / "package"
    bin_dir = package / "bin"
    install_root.mkdir()
    bin_dir.mkdir(parents=True)
    shutil.copy2(EASY_INSTALL_DIR / "runtime-common.sh", install_root)
    shutil.copy2(EASY_INSTALL_DIR / "start-pyeidors.sh", install_root)
    (install_root / "edition.conf").write_text(
        "\n".join(
            (
                "VERSION='2.0.0'",
                "EDITION_ID='cpu-universal'",
                "EDITION_NAME_ZH='CPU 通用版'",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (install_root / "installed-package-map.tsv").write_text(
        f"pyeidors\t{package}\n",
        encoding="utf-8",
    )
    fake_app = bin_dir / "eit-app"
    fake_app.write_text(
        """#!/bin/bash
set -euo pipefail
for name in PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CUDA_HOME \
CUDA_PATH CUDACXX PETSC_DIR SLEPC_DIR LD_PRELOAD QT_PLUGIN_PATH; do
  [ -z "${!name+x}" ] || { echo "$name leaked" >&2; exit 1; }
done
[ "${PYTHONNOUSERSITE:-}" = "1" ]
printf 'clean-runtime\\n'
""",
        encoding="utf-8",
    )
    fake_app.chmod(0o755)

    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "PYTHONHOME": "/tmp/host-python",
        "PYTHONPATH": "/tmp/host-site-packages",
        "VIRTUAL_ENV": "/tmp/host-venv",
        "CONDA_PREFIX": "/tmp/host-conda",
        "CUDA_HOME": "/tmp/cuda-11",
        "CUDA_PATH": "/tmp/cuda-11",
        "CUDACXX": "/tmp/cuda-11/bin/nvcc",
        "PETSC_DIR": "/tmp/host-petsc",
        "SLEPC_DIR": "/tmp/host-slepc",
        "LD_PRELOAD": "/lib/x86_64-linux-gnu/libm.so.6",
        "QT_PLUGIN_PATH": "/tmp/host-qt",
    }
    result = subprocess.run(
        ["/bin/bash", str(install_root / "start-pyeidors.sh"), "--real"],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    assert result.stdout.rstrip().endswith("clean-runtime")


def test_v770_v771_beginner_docs_cover_conflict_and_recovery_paths() -> None:
    zh = _read("docs/EASY_INSTALL_LINUX.zh.md")
    en = _read("docs/EASY_INSTALL_LINUX.en.md")
    for text in (zh, en):
        for term in (
            "zstd",
            "tar",
            "Nix",
            "PYTHONPATH",
            "CUDA_HOME",
            "PyTorch",
            "nvidia-smi",
            "TMPDIR",
            "SHA-256",
        ):
            assert term in text
