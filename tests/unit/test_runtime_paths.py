from __future__ import annotations

import os
from pathlib import Path

from eit_app.backend_worker_runtime import backend_worker_env
from pyeidors.cache import CacheManager, CachePolicy
from pyeidors.geometry.derived_cache import mesh_derived_cache_path
from pyeidors.inverse.greit_registry import greit_registry_dir
from pyeidors.inverse.greit_warmup import greit_common_config_dir
from pyeidors.runtime_paths import (
    pyeidors_cache_path,
    pyeidors_cache_root,
    pyeidors_data_path,
    pyeidors_data_root,
    pyeidors_output_path,
    pyeidors_output_root,
    resolve_pyeidors_cache_dir,
    resolve_pyeidors_mesh_dir,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_v620_packaged_cache_defaults_resolve_to_user_cache_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "user-cache"
    data_root = tmp_path / "user-data"
    output_root = tmp_path / "user-output"
    monkeypatch.setenv("PYEIDORS_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("PYEIDORS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PYEIDORS_OUTPUT_ROOT", str(output_root))
    monkeypatch.delenv("PYEIDORS_CACHE_REQUESTED_ROOT", raising=False)
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", raising=False)
    monkeypatch.delenv("PYEIDORS_GREIT_ARTIFACT_REGISTRY_DIR", raising=False)
    monkeypatch.delenv("PYEIDORS_GREIT_COMMON_CONFIG_DIR", raising=False)

    assert pyeidors_cache_root() == cache_root
    assert pyeidors_data_root() == data_root
    assert pyeidors_output_root() == output_root
    assert pyeidors_cache_path("gui_rm") == cache_root / "gui_rm"
    assert pyeidors_data_path("measurements") == data_root / "measurements"
    assert pyeidors_output_path("results") == output_root / "results"
    assert resolve_pyeidors_cache_dir(".pyeidors_cache/v2") == cache_root / "v2"
    assert resolve_pyeidors_mesh_dir("eit_meshes") == cache_root / "eit_meshes"
    assert (
        resolve_pyeidors_mesh_dir("eit_meshes/generated")
        == cache_root / "eit_meshes" / "generated"
    )
    assert resolve_pyeidors_mesh_dir("custom_meshes") == Path("custom_meshes")

    manager = CacheManager(
        scope="off",
        cache_dir=".pyeidors_cache/v2",
        policy=CachePolicy(disk_lifecycle="persistent"),
    )
    assert manager.requested_cache_dir == cache_root / "v2"
    assert (
        mesh_derived_cache_path(".pyeidors_cache/v2", "abc")
        == cache_root / "v2" / "mesh_derived" / "abc.h5"
    )
    assert greit_registry_dir() == cache_root / "greit_artifacts"
    assert greit_common_config_dir() == cache_root / "greit_common_configs"

    repo = tmp_path / "nix-store-like-source"
    repo.mkdir()
    env, cache = backend_worker_env(repo=repo, profile="complex64-cuda")

    assert cache.profile_root == (
        cache_root / "gui_backend_worker" / "v1" / "complex64-cuda"
    )
    assert env["PYEIDORS_CACHE_ROOT"] == str(cache_root)
    assert env["PYEIDORS_DATA_ROOT"] == str(data_root)
    assert env["PYEIDORS_OUTPUT_ROOT"] == str(output_root)
    assert env["PYEIDORS_GREIT_ARTIFACT_REGISTRY_DIR"] == str(
        cache_root / "greit_artifacts"
    )
    assert env["PYEIDORS_GREIT_COMMON_CONFIG_DIR"] == str(
        cache_root / "greit_common_configs"
    )
    assert env["XDG_CACHE_HOME"] == str(cache.profile_root / "xdg-cache")


def test_v620_source_session_cache_root_keeps_dev_shell_relative_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    requested = tmp_path / ".pyeidors_cache" / "v2"
    monkeypatch.delenv("PYEIDORS_CACHE_ROOT", raising=False)
    monkeypatch.setenv("PYEIDORS_CACHE_REQUESTED_ROOT", str(requested))

    assert pyeidors_cache_root() == tmp_path / ".pyeidors_cache"
    assert pyeidors_cache_path("v2") == requested


def test_v620_gui_default_output_paths_resolve_to_user_writable_roots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYEIDORS_DATA_ROOT", str(tmp_path / "data-root"))
    monkeypatch.setenv("PYEIDORS_OUTPUT_ROOT", str(tmp_path / "output-root"))
    monkeypatch.delenv("EIT_APP_DB_PATH", raising=False)

    from eit_app.ui.dialogs.batch_reconstruction_dialog import (
        _default_batch_results_dir,
    )
    from eit_app.ui.dialogs.interop_hub_dialog import (
        _default_interop_capture_dir,
        _default_interop_export_dir,
    )
    from eit_app.ui.dialogs.reconstruction_dialog import _default_results_dir
    from eit_app.ui.hardware.acquisition_panel import AcquisitionPanel
    from eit_app.ui.main_window import EITWorkstation
    from eit_app.ui.simulation.dataset_generator_panel import DatasetGeneratorPanel

    assert (
        AcquisitionPanel.default_output_dir()
        == pyeidors_data_path("measurements").resolve()
    )
    assert (
        DatasetGeneratorPanel.default_output_dir()
        == pyeidors_data_path("datasets").resolve()
    )
    assert EITWorkstation._default_db_path() == pyeidors_data_path("eit_frames.db")
    assert _default_results_dir() == pyeidors_output_path("reconstructions")
    assert _default_batch_results_dir() == pyeidors_output_path("batch_reconstructions")
    assert _default_interop_capture_dir() == pyeidors_output_path("interop")
    assert _default_interop_export_dir() == pyeidors_output_path("interop_export")


def test_v620_gui_artifact_relative_paths_resolve_to_user_cache_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache-root"
    monkeypatch.setenv("PYEIDORS_CACHE_ROOT", str(cache_root))
    monkeypatch.delenv("PYEIDORS_CACHE_REQUESTED_ROOT", raising=False)

    from eit_app.controllers.reconstruction_controller import (
        _cache_relative_or_absolute_path,
        _greit_registry_dir_from_meta,
        _resolve_rm_artifact_path,
    )

    assert (
        _cache_relative_or_absolute_path(
            "custom_artifacts", default=".pyeidors_cache/gui_rm"
        )
        == cache_root / "custom_artifacts"
    )
    assert (
        _cache_relative_or_absolute_path(None, default=".pyeidors_cache/gui_rm")
        == cache_root / "gui_rm"
    )
    assert _greit_registry_dir_from_meta({"greit_registry_dir": "registry"}) == (
        cache_root / "registry"
    )

    artifact = cache_root / "relative_artifact.h5"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"placeholder")
    assert _resolve_rm_artifact_path({"rm_artifact_path": "relative_artifact.h5"}) == (
        artifact
    )


def test_v620_petsc_cuda_probe_cache_uses_user_cache_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache-root"
    explicit = tmp_path / "explicit-probe-cache"
    monkeypatch.setenv("PYEIDORS_CACHE_ROOT", str(cache_root))
    monkeypatch.delenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR", raising=False)

    from pyeidors.perf.capabilities import _petsc_cuda_probe_disk_cache_dir

    assert _petsc_cuda_probe_disk_cache_dir() == cache_root / "capabilities"

    monkeypatch.setenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR", str(explicit))
    assert _petsc_cuda_probe_disk_cache_dir() == explicit


def test_v620_packaged_source_has_no_cwd_output_defaults() -> None:
    repo = Path(__file__).resolve().parents[2]
    source_roots = (repo / "src" / "eit_app", repo / "src" / "pyeidors")
    forbidden = (
        "Path.cwd()",
        'Path("outputs',
        "Path('outputs",
        "outputs/",
        'Path("results',
        "Path('results",
        "results/",
        'Path("data',
        "Path('data",
        'Path("reports',
        "Path('reports",
        "reports/",
    )
    offenders: list[str] = []
    for root in source_roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                if pattern in text:
                    rel = path.relative_to(repo).as_posix()
                    offenders.append(f"{rel}: {pattern}")
    assert offenders == []


def test_v620_user_scripts_have_no_cwd_output_defaults() -> None:
    repo = Path(__file__).resolve().parents[2]
    forbidden = (
        'Path("outputs',
        "Path('outputs",
        "outputs/",
        'Path("results',
        "Path('results",
        "results/",
        'Path("reports',
        "Path('reports",
        "reports/",
        "fullfile('outputs'",
        "fullfile('results'",
        "fullfile('reports'",
    )
    offenders: list[str] = []
    for root in (repo / "scripts",):
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".py", ".m"}:
                continue
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                for pattern in forbidden:
                    if pattern not in line:
                        continue
                    if pattern == "results/" and "test_results/" in line:
                        continue
                    rel = path.relative_to(repo).as_posix()
                    offenders.append(f"{rel}:{line_no}: {pattern}")
    assert offenders == []
