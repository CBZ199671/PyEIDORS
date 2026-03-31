from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_patch_helper():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "env"
        / "patch_pymfem_source.py"
    )
    spec = importlib.util.spec_from_file_location("patch_pymfem_source", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_source_tree_updates_expected_build_files(tmp_path: Path) -> None:
    patch_helper = _load_patch_helper()
    build_system = tmp_path / "_build_system"
    build_system.mkdir(parents=True)

    (build_system / "build_globals.py").write_text(
        "import os\nmetis_prefix = ''\nhypre_prefix = ''\n",
        encoding="utf-8",
    )
    (build_system / "build_pymfem.py").write_text(
        "    command = [python, 'setup.py', 'build_ext', '--inplace', '--parallel',\n"
        "               str(max((cpu_count() - 1, 1)))]\n",
        encoding="utf-8",
    )
    (build_system / "build_config.py").write_text(
        "    if self.hypre_prefix != '':\n"
        "        check = find_libpath_from_prefix('HYPRE', self.hypre_prefix)\n"
        "        assert check != '', \"libHYPRE.so is not found in the specified <path>/lib or lib64\"\n"
        "        hypre_prefix = os.path.expanduser(self.hypre_prefix)\n"
        "        build_hypre = False\n",
        encoding="utf-8",
    )

    patch_helper.patch_source_tree(tmp_path)

    globals_text = (build_system / "build_globals.py").read_text(encoding="utf-8")
    assert "PYEIDORS_HYPRE_PREFIX" in globals_text
    assert "build_py_done = False" in globals_text

    build_text = (build_system / "build_pymfem.py").read_text(encoding="utf-8")
    assert "PYEIDORS_PYMFEM_BUILD_JOBS" in build_text

    config_text = (build_system / "build_config.py").read_text(encoding="utf-8")
    assert "bglb.hypre_prefix = os.path.expanduser(self.hypre_prefix)" in config_text
