"""T4 — smoke for the forward KSP session reuse benchmark (G1 evidence).

Drives the bench script in-process on a tiny 2D unit-square mesh under
``auto`` and ``never`` regimes and asserts:

- HDF5 + JSON + Markdown artifact files are produced.
- summary.json schema records env_path, regime params, V13/V14/V52/V67
  cites and per-regime stats.
- ``auto`` regime reuses the PETSc KSP session at least once; ``never``
  regime never reuses it (V13 + V14 contract).
- ``auto`` cumulative PC setup time does not exceed ``never`` —
  i.e. G1 persistence does not regress setup cost.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "benchmarks" / "benchmark_forward_ksp_session_reuse.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_forward_ksp_session_reuse", SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_smoke(out_dir: Path, *, n_iter: int = 3, n_elec: int = 4) -> dict:
    module = _load_module()
    argv = [
        "--out-dir",
        str(out_dir),
        "--mesh-dim",
        "2",
        "--n-elec",
        str(n_elec),
        "--n-iter",
        str(n_iter),
        "--mesh-refinement",
        "3",
        "--regimes",
        "auto,never",
        "--solver-preset",
        "spd_hypre",
        "--petsc-device",
        "cpu",
        "--rtol",
        "1e-6",
        "--atol",
        "1e-9",
        "--max-it",
        "200",
        "--sigma-noise-scale",
        "0.02",
        "--seed",
        "7",
    ]
    rc = module.main(argv)
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return summary


def test_t4_bench_writes_artifact_with_v_cites_and_env_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "t4_smoke"
    summary = _run_smoke(out_dir)

    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "ksp_session_reuse_runs.h5").is_file()
    assert (out_dir / "summary.md").is_file()

    assert summary["task"] == "T4"
    assert summary["schema_version"] == 1
    assert summary["mesh_dim"] == 2
    assert summary["regimes"] == ["auto", "never"]
    assert sorted(summary["v_cites"]) == ["V13", "V14", "V52", "V67"]
    assert isinstance(summary["env_path"], str) and len(summary["env_path"]) > 0
    assert "per_regime" in summary
    for regime in ("auto", "never"):
        assert regime in summary["per_regime"]
        per = summary["per_regime"][regime]
        for key in (
            "n_calls",
            "n_reused",
            "n_refresh",
            "cumulative_setup_seconds",
            "first_call_setup_seconds",
            "iter_max_mean",
            "refresh_reasons",
        ):
            assert key in per, f"summary regime {regime!r} missing {key!r}"


def test_t4_bench_auto_reuses_session_never_does_not(tmp_path: Path) -> None:
    out_dir = tmp_path / "t4_contract"
    summary = _run_smoke(out_dir, n_iter=4)

    auto = summary["per_regime"]["auto"]
    never = summary["per_regime"]["never"]

    # V13: auto regime must reuse the PETSc KSP session at least once
    # after the initial setup. n_iter=4 → at least 1 reuse expected.
    assert auto["n_reused"] >= 1, "auto regime failed to reuse KSP session (V13)"
    # V14: never regime must dispose the session every call → no reuse
    assert never["n_reused"] == 0, "never regime should never reuse session (V14)"
    # never refreshes every call (or first-build), so n_refresh == n_calls
    assert never["n_refresh"] == never["n_calls"]

    # G1 evidence: warm setup must not regress vs cold baseline.
    # Allow tiny noise budget on tiny meshes where both are sub-ms.
    assert (
        auto["cumulative_setup_seconds"] <= never["cumulative_setup_seconds"] + 5e-3
    ), (
        "auto cumulative setup exceeds never beyond noise budget; "
        f"auto={auto['cumulative_setup_seconds']}, "
        f"never={never['cumulative_setup_seconds']}"
    )
    assert "g1_cumulative_setup_saved_seconds" in summary
    assert "g1_warm_cold_setup_ratio" in summary


def test_t4_bench_hdf5_arrays_and_metadata_ready(tmp_path: Path) -> None:
    out_dir = tmp_path / "t4_h5"
    summary = _run_smoke(out_dir, n_iter=3)
    h5_path = out_dir / "ksp_session_reuse_runs.h5"

    with h5py.File(h5_path, "r") as handle:
        meta = json.loads(handle.attrs["metadata_json"])
        assert meta["task"] == "T4"
        assert meta["schema_version"] == 1
        assert sorted(meta["v_cites"]) == ["V13", "V14", "V52", "V67"]
        arrays_group = handle["arrays"]
        for regime in ("auto", "never"):
            for field in (
                "iter_max_per_call",
                "iter_total_per_call",
                "setup_seconds",
                "session_reused",
                "refresh_triggered",
                "pc_session_total_setups",
            ):
                key = f"regime_{regime}_{field}"
                assert key in arrays_group, f"missing dataset {key!r}"
                arr = np.asarray(arrays_group[key])
                assert arr.shape == (summary["per_regime"][regime]["n_calls"],)

        names_attr = handle.attrs.get("array_names_json")
        assert names_attr is not None
        names_list = json.loads(
            names_attr.decode("utf-8") if isinstance(names_attr, bytes) else names_attr
        )
        assert "regime_auto_setup_seconds" in names_list
        assert "regime_never_setup_seconds" in names_list


def test_t4_bench_markdown_summary_mentions_v_cites(tmp_path: Path) -> None:
    out_dir = tmp_path / "t4_md"
    _ = _run_smoke(out_dir, n_iter=3)
    md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "T4" in md
    for cite in ("V13", "V14", "V52", "V67"):
        assert cite in md, f"summary.md missing V cite {cite!r}"
    # Renders the per-regime table header.
    assert "regime" in md and "cum_setup_s" in md
    assert "G1 cumulative setup" in md or "warm/cold setup ratio" in md


def test_t4_bench_rejects_unknown_regime(tmp_path: Path) -> None:
    module = _load_module()
    out_dir = tmp_path / "t4_invalid"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="unknown regimes"):
        module.main(
            [
                "--out-dir",
                str(out_dir),
                "--mesh-dim",
                "2",
                "--n-elec",
                "4",
                "--n-iter",
                "1",
                "--mesh-refinement",
                "2",
                "--regimes",
                "auto,bogus",
                "--solver-preset",
                "spd_gamg",
            ]
        )
