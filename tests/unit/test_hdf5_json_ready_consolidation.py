"""T87: shared JSON-ready helper contracts for HDF5 writers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pyeidors.io import hdf5_artifacts
from pyeidors.io._json import json_ready


REPO_ROOT = Path(__file__).resolve().parents[2]
MESH_TOOLS = REPO_ROOT / "scripts" / "mesh_tools"
if str(MESH_TOOLS) not in sys.path:
    sys.path.insert(0, str(MESH_TOOLS))

import matlab_mesh_hdf5  # noqa: E402
from scripts.benchmarks import benchmark_dual_model_rm_v1  # noqa: E402
from scripts.benchmarks import benchmark_dynamic_tv_huber_sweep  # noqa: E402
from scripts.benchmarks import benchmark_dynamic_validation  # noqa: E402
from scripts.benchmarks import benchmark_lazy_48e_cuda_runtime  # noqa: E402
from scripts.benchmarks import review_dynamic_eidors_metrics  # noqa: E402
from scripts.diagnostics import gallery_shared  # noqa: E402


def test_hdf5_writers_reuse_shared_json_ready_alias() -> None:
    assert hdf5_artifacts._json_ready is json_ready
    assert matlab_mesh_hdf5._json_ready is json_ready


def test_script_json_ready_helpers_reuse_shared_alias() -> None:
    assert benchmark_dual_model_rm_v1._jsonable is json_ready
    assert benchmark_dynamic_tv_huber_sweep._json_ready is json_ready
    assert benchmark_dynamic_validation._json_ready is json_ready
    assert benchmark_lazy_48e_cuda_runtime._jsonable is json_ready
    assert review_dynamic_eidors_metrics._json_ready is json_ready
    assert gallery_shared.jsonable is json_ready


def test_json_ready_preserves_legacy_recursive_conversion_contract() -> None:
    payload = {
        "path": Path("mesh/out.h5"),
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "scalar": np.float64(1.25),
        "tuple": (Path("a"), np.int32(7)),
        3: {"nested": np.array([np.float32(0.5)], dtype=np.float32)},
    }

    ready = json_ready(payload)

    assert ready == {
        "path": "mesh/out.h5",
        "array": [[1, 2], [3, 4]],
        "scalar": 1.25,
        "tuple": ["a", 7],
        "3": {"nested": [0.5]},
    }
    assert json.dumps(ready, sort_keys=True)
