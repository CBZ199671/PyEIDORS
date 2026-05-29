from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "run_sharded_unit_tests.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_sharded_unit_tests", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_category_shards_cover_each_unit_file_once():
    module = _load_module()
    unit_files = module.discover_unit_tests()
    shards = module.build_category_shards(unit_files)

    covered = [path for shard in shards for path in shard.files]

    assert set(covered) == set(unit_files)
    assert len(covered) == len(set(covered))
    assert all(shard.files for shard in shards)


def test_refactor_smoke_shard_has_no_cov_nix_command():
    module = _load_module()
    shard = module.select_shards(["fp-refactor-smoke"])[0]
    command = module.emitted_shell_command(shard)

    assert command[:5] == ["nix", "develop", "-c", "uv", "run"]
    assert "--no-cov" in command
    assert "tests/unit/test_forward_solver_presets.py" in command
    assert "tests/unit/test_jacobian_linearization.py" in command
    assert command[-1] == "-q"


def test_optional_shards_emit_required_pytest_opt_ins():
    module = _load_module()
    shards = {
        shard.name: shard
        for shard in module.select_shards(["gui", "hardware", "perf-cuda"])
    }

    gui_command = module.emitted_shell_command(shards["gui"])
    hardware_command = module.emitted_shell_command(shards["hardware"])
    cuda_command = module.emitted_shell_command(shards["perf-cuda"])

    assert "--run-gui" in gui_command
    assert "--run-slow" in gui_command
    assert "--run-hardware" in hardware_command
    assert "--run-gpu" in cuda_command
    assert "--run-slow" in cuda_command


def test_dry_run_command_quotes_pytest_args_with_spaces():
    module = _load_module()
    shard = module.select_shards(["fp-refactor-smoke"])[0]
    rendered = module.format_shell_command(
        module.emitted_shell_command(shard, ["-k", "solver and not slow"])
    )

    assert "'solver and not slow'" in rendered


def test_all_selection_includes_gui_and_excludes_hardware_by_default():
    module = _load_module()
    args = module._parse_args(["--run", "--all"])
    shards = module._selected_shards(args)
    names = {shard.name for shard in shards}

    assert "gui" in names
    assert "hardware" not in names


def test_all_selection_can_include_hardware():
    module = _load_module()
    args = module._parse_args(["--run", "--all", "--include-hardware"])
    shards = module._selected_shards(args)
    names = {shard.name for shard in shards}

    assert "gui" in names
    assert "hardware" in names


def test_bare_run_selection_excludes_hardware_by_default_and_keeps_smoke():
    module = _load_module()
    args = module._parse_args(["--run"])
    shards = module._selected_shards(args)
    names = {shard.name for shard in shards}

    assert "gui" in names
    assert "hardware" not in names
    assert "fp-refactor-smoke" in names


def test_bare_dry_run_selection_can_include_hardware():
    module = _load_module()
    args = module._parse_args(["--dry-run", "--include-hardware"])
    shards = module._selected_shards(args)
    names = {shard.name for shard in shards}

    assert "gui" in names
    assert "hardware" in names
    assert "fp-refactor-smoke" in names


def test_gui_and_hardware_shards_keep_domains_separate():
    module = _load_module()
    shards = {
        shard.name: set(shard.relative_files)
        for shard in module.build_category_shards()
    }

    assert "tests/unit/test_eit_app_gui_smoke.py" in shards["gui"]
    assert "tests/unit/test_eit_app_interop_hub.py" in shards["gui"]
    assert "tests/unit/test_eit_app_serial_device.py" in shards["hardware"]
    assert "tests/unit/test_eit_app_relay_transport.py" in shards["hardware"]
    assert shards["gui"].isdisjoint(shards["hardware"])


def test_unknown_shard_reports_known_names():
    module = _load_module()

    with pytest.raises(ValueError, match="known shards"):
        module.select_shards(["does-not-exist"])


def test_relative_report_dir_is_normalized_to_repo():
    module = _load_module()

    report_dir = module._normalize_report_dir(Path("test_results/example"))

    assert report_dir == REPO_ROOT / "test_results" / "example"
    assert module._relative_to_repo(report_dir / "summary.json") == (
        "test_results/example/summary.json"
    )
