from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "persistence_format_guard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "persistence_format_guard", SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_guard_blocks_new_production_numpy_writer(tmp_path):
    module = _load_module()
    _write(
        tmp_path / "src" / "pkg" / "new_writer.py",
        "import numpy as np\n\ndef save(path):\n    np.savez(path, values=[1])\n",
    )

    findings = module.scan_repo(tmp_path)
    violations = module.guard_violations(findings)

    assert [item.kind for item in violations] == ["numpy_writer"]
    assert violations[0].classification == "production"
    assert violations[0].api == "np.savez"


def test_guard_blocks_numpy_writer_aliases(tmp_path):
    module = _load_module()
    _write(
        tmp_path / "src" / "pkg" / "aliased_writer.py",
        "from numpy import savez_compressed as write_archive\n\n"
        "def save(path):\n"
        "    write_archive(path, values=[1])\n",
    )

    violations = module.guard_violations(module.scan_repo(tmp_path))

    assert len(violations) == 1
    assert violations[0].api == "np.savez_compressed"


def test_guard_exempts_test_only_numpy_writer(tmp_path):
    module = _load_module()
    _write(
        tmp_path / "tests" / "unit" / "test_writer.py",
        "import numpy as np\n\n"
        "def test_save(tmp_path):\n"
        "    np.savez(tmp_path / 'fixture.npz', values=[1])\n",
    )

    findings = module.scan_repo(tmp_path)

    assert not module.guard_violations(findings)
    assert any(
        item.kind == "numpy_writer" and item.classification == "test-only"
        for item in findings
    )


def test_guard_exempts_in_memory_numpy_serializer(tmp_path):
    module = _load_module()
    _write(
        tmp_path / "src" / "pkg" / "hash_payload.py",
        "import io\n"
        "import numpy as np\n\n"
        "def digest(values):\n"
        "    buffer = io.BytesIO()\n"
        "    np.save(buffer, values, allow_pickle=True)\n"
        "    return buffer.getbuffer()\n",
    )

    findings = module.scan_repo(tmp_path)
    serializers = [item for item in findings if item.kind == "numpy_memory_serializer"]

    assert len(serializers) == 1
    assert serializers[0].classification == "production"
    assert not module.guard_violations(findings)


def test_guard_classifies_hdf5_mesh_and_legacy_readers(tmp_path):
    module = _load_module()
    _write(
        tmp_path / "src" / "pkg" / "formats.py",
        "import h5py\n"
        "import gmsh\n"
        "import numpy as np\n"
        "from dolfinx.io import XDMFFile\n\n"
        "def touch(comm):\n"
        "    np.load('legacy.npz')\n"
        "    gmsh.write('mesh.msh')\n"
        "    h5py.File('cache.h5', 'w')\n"
        "    XDMFFile(comm, 'mesh.xdmf', 'w')\n",
    )

    findings = module.scan_repo(tmp_path)
    keys = {(item.kind, item.api, item.classification) for item in findings}

    assert ("numpy_reader", "np.load", "production") in keys
    assert ("mesh_source_writer", "gmsh.write", "production") in keys
    assert ("hdf5_io", "h5py.File", "production") in keys
    assert ("hdf5_io", "XDMFFile", "production") in keys
    assert not module.guard_violations(findings)


def test_current_repo_legacy_allowlist_has_no_numpy_writer_violations():
    module = _load_module()

    findings = module.scan_repo(REPO_ROOT)

    assert module.guard_violations(findings) == []
    assert module.summary(findings).get("legacy-production:numpy_writer", 0) == 0
