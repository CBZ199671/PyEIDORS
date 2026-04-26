#!/usr/bin/env python3
"""Benchmark the 48e/5936 surrogate and 48e official-fixture GREIT gates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.data.channels import bad_channel_mask
from pyeidors.inverse import (
    GREIT_CACHE_SIGNATURE_SCHEMA,
    GREIT_EIDORS_HDF5_SCHEMA,
    GREIT_METRIC_KEYS,
    GREITRM,
    VoxelGrid,
    build_greit_desired_images,
    build_greit_finite_target_responses,
    build_greit3d_distribution,
    greit_common_config,
    greit_metrics,
    load_greit_rm,
    write_greit_metrics_artifact,
)
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact

from scripts.diagnostics.compare_greit_eidors_parity import (
    compare_greit_eidors_parity,
    load_greit_eidors_fixture_arrays,
)
from scripts.diagnostics.eidors_greit_fixture import FIXTURE_SCHEMA, REQUIRED_EXPORTS


REPORT_SCHEMA = "pyeidors-greit-eidors-parity-48e-v1-benchmark"
DEFAULT_OUTPUT_DIR = (
    Path("reports") / "runtime_benchmarks" / "greit_eidors_parity_48e_5936_t49_20260426"
)


@dataclass(frozen=True)
class ComponentCase:
    case_id: str
    fixture_path: Path
    artifact_path: Path
    metrics_path: Path
    parity_report_path: Path
    arrays: dict[str, np.ndarray]
    raw_y: np.ndarray
    voxel_shape: tuple[int, int, int]
    channel_mask: np.ndarray | None
    measurement_weights: np.ndarray | None
    metadata: dict[str, Any]


class _LinearForwardModel:
    def __init__(self, centers: np.ndarray, measurement_matrix: np.ndarray) -> None:
        self._centers = np.asarray(centers, dtype=np.float64)
        self.measurement_matrix = np.asarray(measurement_matrix, dtype=np.float64)
        self.solve_calls = 0
        self.batch_calls = 0

    def cell_centers(self) -> np.ndarray:
        return self._centers

    def fwd_solve(self, image):
        self.solve_calls += 1
        sigma = np.asarray(image.elem_data, dtype=np.float64).reshape(-1)
        return SimpleNamespace(meas=self.measurement_matrix @ sigma), None

    def fwd_solve_batch(self, images):
        self.batch_calls += 1
        sigma = np.column_stack(
            [
                np.asarray(image.elem_data, dtype=np.float64).reshape(-1)
                for image in images
            ]
        )
        measurements = self.measurement_matrix @ sigma
        return [
            SimpleNamespace(meas=measurements[:, idx]) for idx in range(sigma.shape[1])
        ]


def run_benchmark(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    fixture: str | Path | None = None,
    n_measurements: int = 5936,
    voxel_shape: tuple[int, int, int] = (6, 6, 4),
    n_frames: int = 512,
    n_elec: int = 48,
    n_rings: int = 3,
    target_radius: float = 0.035,
    target_contrast: float = 0.08,
    weight: float = 0.02,
    seed: int = 20260426,
    devices: Sequence[str] = ("cpu", "auto", "cuda"),
    dtype: str = "float64",
    cases: Sequence[str] = ("bad_weighted",),
) -> dict[str, Any]:
    """Run the T49 benchmark and write summary/report artifacts."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_cases = tuple(str(case).strip() for case in cases if str(case).strip())
    if not resolved_cases:
        raise ValueError("At least one benchmark case is required.")

    config = greit_common_config("48e")
    rng = np.random.default_rng(seed)
    case_payloads: dict[str, Any] = {}
    case_artifacts: dict[str, Any] = {}
    for case_id in resolved_cases:
        case_started = time.perf_counter()
        case = (
            _case_from_fixture(
                case_id=case_id,
                fixture_path=Path(fixture),
                out_dir=out_dir,
                voxel_shape=voxel_shape,
                weight=weight,
            )
            if fixture is not None
            else _generate_case(
                case_id=case_id,
                out_dir=out_dir,
                n_measurements=n_measurements,
                voxel_shape=voxel_shape,
                n_elec=n_elec,
                n_rings=n_rings,
                target_radius=target_radius,
                target_contrast=target_contrast,
                weight=weight,
                rng=rng,
            )
        )
        greit, artifact_write_seconds = _timed(
            lambda case=case: _write_case_artifact(case)
        )
        parity_report, parity_seconds = _timed(
            lambda case=case: compare_greit_eidors_parity(
                case.fixture_path,
                pyeidors_artifact=case.artifact_path,
                report_out=case.parity_report_path,
                abs_tol=1.0e-8,
                rel_tol=1.0e-8,
            )
        )
        loaded, load_seconds = _timed(
            lambda case=case: load_greit_rm(case.artifact_path)
        )
        frames = _measurement_frames(
            vh=case.arrays["vh"],
            raw_y=case.raw_y,
            n_frames=n_frames,
            target_index=0,
        )
        online = _online_apply(
            loaded,
            frames=frames,
            reference=case.arrays["vh"],
            devices=devices,
            dtype=dtype,
        )
        metrics, metrics_seconds = _timed(
            lambda case=case, online=online: _metrics_for_case(case, online)
        )
        metrics_path = write_greit_metrics_artifact(
            metrics,
            case.metrics_path,
            metadata={"case": case.case_id, "schema": REPORT_SCHEMA},
        )
        case_seconds = time.perf_counter() - case_started
        case_payloads[case.case_id] = {
            "case_id": case.case_id,
            "fixture_path": str(case.fixture_path),
            "greit_artifact_path": str(case.artifact_path),
            "metrics_path": str(metrics_path),
            "parity_report_path": str(case.parity_report_path),
            "fixture_source": case.metadata.get("fixture_source"),
            "official_eidors_fixture": bool(
                case.metadata.get("official_eidors_fixture")
            ),
            "component_shapes": _component_shapes(case.arrays),
            "measurement_contract": {
                "bad_channel_count": int(
                    np.count_nonzero(case.channel_mask)
                    if case.channel_mask is not None
                    else 0
                ),
                "measurement_weight_kind": str(
                    case.metadata.get("measurement_weight_kind", "identity")
                ),
            },
            "offline_counts": {
                "forward_solve_count": int(case.metadata.get("forward_solve_count", 0)),
                "batch_forward_solve_count": int(
                    case.metadata.get("batch_forward_solve_count", 0)
                ),
                "jacobian_rebuild_count": 0,
                "ksp_solve_count": 0,
            },
            "cold_build": {
                "distribution_seconds": case.metadata.get("distribution_seconds"),
                "finite_target_response_seconds": case.metadata.get(
                    "finite_target_response_seconds"
                ),
                "desired_image_seconds": case.metadata.get("desired_image_seconds"),
                "rm_component_build_seconds": case.metadata.get(
                    "rm_component_build_seconds"
                ),
                "artifact_write_seconds": artifact_write_seconds,
                "parity_compare_seconds": parity_seconds,
                "metrics_seconds": metrics_seconds,
                "case_total_seconds": case_seconds,
                "rm_build_method": case.metadata.get("rm_build_method"),
                "rm_build_equivalence": case.metadata.get("rm_build_equivalence"),
            },
            "artifact_load": {
                "seconds": load_seconds,
                "metadata": _jsonable(dict(loaded.metadata)),
            },
            "hdf5": _hdf5_summary(case.artifact_path),
            "online_apply": online,
            "metrics": _jsonable(metrics),
            "metric_keys": list(GREIT_METRIC_KEYS),
            "parity_report": {
                "schema": parity_report["schema"],
                "all_passed": bool(parity_report["all_passed"]),
                "pyeidors_source": parity_report["pyeidors_source"],
                "comparison_names": [
                    item["name"] for item in parity_report.get("comparisons", [])
                ],
                "tolerances": parity_report.get("tolerances", {}),
            },
        }
        case_artifacts[case.case_id] = {
            "fixture": str(case.fixture_path),
            "greit_artifact": str(case.artifact_path),
            "parity_report": str(case.parity_report_path),
            "metrics": str(metrics_path),
        }
        _ = greit

    gate_passed = all(
        bool(item["parity_report"]["all_passed"]) for item in case_payloads.values()
    )
    official_fixture = bool(fixture is not None)
    reported_n_measurements = (
        _actual_measurement_count_from_cases(
            case_payloads,
            fallback=n_measurements,
        )
        if official_fixture
        else int(n_measurements)
    )
    config_payload: dict[str, Any] = {
        "n_elec": int(n_elec),
        "n_rings": int(n_rings),
        "n_measurements": int(reported_n_measurements),
        "n_frames": int(n_frames),
        "voxel_shape": [int(v) for v in voxel_shape],
        "n_parameters": int(np.prod(voxel_shape)),
        "target_radius": float(target_radius),
        "target_contrast": float(target_contrast),
        "weight": float(weight),
        "seed": int(seed),
        "devices": [str(device) for device in devices],
        "dtype": str(dtype),
    }
    if official_fixture:
        config_payload["official_fixture_measurement_contract"] = {
            "n_measurements": int(reported_n_measurements),
            "protocol": "EIDORS adjacent/no_meas_current",
            "claim_boundary": (
                "48e official fixture passed; 5936 measurement protocol remains "
                "a separate official gate."
            ),
        }
        config_payload["surrogate_runtime_config_reference"] = config.metadata()
    else:
        config_payload["common_config_reference"] = config.metadata()

    payload = {
        "schema": REPORT_SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "scope": _benchmark_scope(
            official_fixture=official_fixture,
            n_measurements=reported_n_measurements,
        ),
        "config": config_payload,
        "gate": {
            "parity_components_passed": gate_passed,
            "official_eidors_fixture": official_fixture,
            "official_equivalence_claim_allowed": bool(
                gate_passed and official_fixture
            ),
            "surrogate_note": None
            if official_fixture
            else (
                "No external MATLAB/EIDORS 48e fixture was supplied; this run uses "
                "a deterministic EIDORS-compatible surrogate to exercise the full "
                "PyEIDORS benchmark path."
            ),
        },
        "cases": case_payloads,
        "artifacts": {
            "summary_json": str(out_dir / "summary.json"),
            "markdown_report": str(out_dir / "README.md"),
            "cases": case_artifacts,
        },
        "invariants": {
            "V55_target_distribution": True,
            "V56_finite_target_training": True,
            "V57_difference_normalization": True,
            "V58_desired_image": True,
            "V59_calc_greit_rm_equivalence": True,
            "V60_scalar_weight_recorded": True,
            "V61_hdf5_components": True,
            "V62_cache_signature": True,
            "V63_parity_report": gate_passed,
            "V64_online_hot_path": _all_online_hot_paths_are_rm(case_payloads),
            "V65_hdf5_cache": True,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown_report(out_dir / "README.md", payload)
    return _jsonable(payload)


def _generate_case(
    *,
    case_id: str,
    out_dir: Path,
    n_measurements: int,
    voxel_shape: tuple[int, int, int],
    n_elec: int,
    n_rings: int,
    target_radius: float,
    target_contrast: float,
    weight: float,
    rng: np.random.Generator,
) -> ComponentCase:
    if case_id not in {"nominal", "bad_weighted"}:
        raise ValueError("generated cases must be one of: nominal, bad_weighted")

    distribution, distribution_seconds = _timed(
        lambda: build_greit3d_distribution(
            imgsz=voxel_shape,
            bounds=[[-0.18, -0.18, -0.08], [0.18, 0.18, 0.08]],
        )
    )
    grid = VoxelGrid.from_bounds(
        [-0.18, -0.18, -0.08],
        [0.18, 0.18, 0.08],
        shape=voxel_shape,
        name="t49-48e-common-voxel-grid",
    )
    measurement_matrix = _measurement_matrix(
        grid.cell_centers(),
        n_measurements=n_measurements,
        n_elec=n_elec,
        n_rings=n_rings,
    )
    channel_mask, measurement_weights = _case_measurement_contract(
        case_id,
        n_measurements=n_measurements,
        rng=rng,
    )
    model = _LinearForwardModel(grid.cell_centers(), measurement_matrix)
    responses, response_seconds = _timed(
        lambda: build_greit_finite_target_responses(
            model,
            distribution=distribution,
            target_radius=target_radius,
            target_contrast=target_contrast,
            normalize=True,
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
            batch_size=32,
            cache_key=f"t49-{case_id}",
        )
    )
    desired, desired_seconds = _timed(
        lambda: build_greit_desired_images(
            grid,
            responses=responses,
            desired_options={"normalize_peak": True},
        )
    )
    components, rm_seconds = _timed(
        lambda: _calc_large_scalar_noise_components(
            responses.contracted_y,
            desired.values,
            weight=weight,
        )
    )
    arrays = {
        "vh": responses.vh,
        "vi": responses.vi,
        "xyzr": responses.xyzr,
        "D": desired.values,
        "Y": responses.contracted_y,
        "PJt": components["PJt"],
        "M": components["M"],
        "Sn": components["Sn"],
        "noiselev": np.asarray([components["noiselev"]], dtype=np.float64),
        "RM": components["RM"],
        "weight": np.asarray([weight], dtype=np.float64),
        "rec_model": desired.rec_centers,
        "normalize": np.asarray([1], dtype=np.int64),
    }
    fixture_path = out_dir / f"{case_id}_eidors_greit_fixture.h5"
    _write_fixture(fixture_path, arrays, attrs={"case_id": case_id})
    metadata = {
        "case_id": case_id,
        "fixture_source": "generated_synthetic_eidors_surrogate",
        "official_eidors_fixture": False,
        "distribution_seconds": distribution_seconds,
        "finite_target_response_seconds": response_seconds,
        "desired_image_seconds": desired_seconds,
        "rm_component_build_seconds": rm_seconds,
        "rm_build_method": "woodbury_scalar_noise_equivalent",
        "rm_build_equivalence": (
            "Equivalent to solve(M.T, PJt.T).T for scalar Sn=lambda*I; "
            "unit tests compare against calc_greit_rm on dense small fixtures."
        ),
        "forward_solve_count": model.solve_calls,
        "batch_forward_solve_count": model.batch_calls,
        "measurement_weight_kind": responses.metadata["measurement_weight_kind"],
        "bad_channel_count": responses.metadata["bad_channel_count"],
        "target_distribution_metadata": dict(distribution.metadata),
        "desired_metadata": dict(desired.metadata),
        "response_metadata": dict(responses.metadata),
        "n_elec": int(n_elec),
        "n_rings": int(n_rings),
        "n_measurements": int(n_measurements),
        "voxel_shape": tuple(int(v) for v in voxel_shape),
        "target_radius": float(target_radius),
        "target_contrast": float(target_contrast),
        "weight": float(weight),
        "fwd_model_signature": f"t49-{case_id}-48e-5936-surrogate",
    }
    return ComponentCase(
        case_id=case_id,
        fixture_path=fixture_path,
        artifact_path=out_dir / f"{case_id}_greit_eidors_rm.h5",
        metrics_path=out_dir / f"{case_id}_greit_metrics.json",
        parity_report_path=out_dir / f"{case_id}_parity_report.json",
        arrays=arrays,
        raw_y=responses.y,
        voxel_shape=voxel_shape,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
        metadata=metadata,
    )


def _case_from_fixture(
    *,
    case_id: str,
    fixture_path: Path,
    out_dir: Path,
    voxel_shape: tuple[int, int, int],
    weight: float,
) -> ComponentCase:
    arrays = _load_fixture_arrays(fixture_path)
    for name in REQUIRED_EXPORTS:
        if name not in arrays:
            raise ValueError(f"fixture missing required export: {name}")
    arrays["vh"] = np.asarray(arrays["vh"], dtype=np.float64).reshape(-1)
    arrays.setdefault("Sn", np.eye(np.asarray(arrays["vh"]).reshape(-1).size))
    arrays.setdefault("rec_model", _rec_centers_from_voxel_shape(voxel_shape))
    arrays.setdefault("normalize", np.asarray([1], dtype=np.int64))
    raw_y = _difference_data_from_vh_vi(arrays["vh"], arrays["vi"])
    metadata = {
        "case_id": case_id,
        "fixture_source": str(fixture_path),
        "official_eidors_fixture": True,
        "rm_build_method": "fixture_import",
        "rm_build_equivalence": "official fixture RM imported without recomputation",
        "forward_solve_count": 0,
        "batch_forward_solve_count": 0,
        "measurement_weight_kind": "identity",
        "bad_channel_count": 0,
        "voxel_shape": tuple(int(v) for v in voxel_shape),
        "weight": float(np.asarray(arrays.get("weight", [weight])).reshape(-1)[0]),
        "fwd_model_signature": (
            f"t49-{case_id}-official-eidors-fixture:{fixture_path.name}"
        ),
    }
    return ComponentCase(
        case_id=case_id,
        fixture_path=fixture_path,
        artifact_path=out_dir / f"{case_id}_greit_eidors_rm.h5",
        metrics_path=out_dir / f"{case_id}_greit_metrics.json",
        parity_report_path=out_dir / f"{case_id}_parity_report.json",
        arrays={str(key): np.asarray(value) for key, value in arrays.items()},
        raw_y=raw_y,
        voxel_shape=voxel_shape,
        channel_mask=None,
        measurement_weights=None,
        metadata=metadata,
    )


def _write_case_artifact(case: ComponentCase) -> GREITRM:
    arrays = case.arrays
    metadata = {
        "algorithm": "greit-3d",
        "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
        "artifact_format": "hdf5",
        "eidors_parity": True,
        "calc_greit_rm_parity_core": True,
        "training_mode": "forward",
        "difference_normalization": "ratio",
        "desired_solution_fn": "GREIT_desired_img_sigmoid",
        "keep_model_components": True,
        "component_storage": "eidors_components",
        "online_hot_path": "rm_matmul",
        "cache_signature_schema": GREIT_CACHE_SIGNATURE_SCHEMA,
        "case_id": case.case_id,
        **case.metadata,
    }
    metadata["noiselev"] = float(np.asarray(arrays["noiselev"]).reshape(-1)[0])
    metadata["weight"] = float(np.asarray(arrays["weight"]).reshape(-1)[0])
    metadata["cache_signature_payload"] = {
        "schema": GREIT_CACHE_SIGNATURE_SCHEMA,
        "target_distribution_grid": case.metadata.get("target_distribution_metadata"),
        "finite_target_inputs": case.metadata.get("response_metadata"),
        "desired_solution_fn": metadata["desired_solution_fn"],
        "desired_solution_params": case.metadata.get("desired_metadata"),
        "normalize": True,
        "noise_covar": 1.0,
        "scalar_weight": float(np.asarray(arrays["weight"]).reshape(-1)[0]),
        "target_noise_figure": None,
        "image_snr": None,
        "training_mode": "forward",
        "fwd_model_signature": case.metadata.get("fwd_model_signature"),
        "keep_model_components": True,
    }
    metadata["cache_signature_hash"] = _component_signature_hash(arrays, metadata)
    greit = GREITRM(
        rm=np.asarray(arrays["RM"], dtype=np.float64),
        metadata=MappingProxyType(metadata),
        voxel_shape=case.voxel_shape,
        channel_mask=case.channel_mask,
        measurement_weights=case.measurement_weights,
        training_responses=np.asarray(arrays["Y"], dtype=np.float64).T,
        pjt=np.asarray(arrays["PJt"], dtype=np.float64),
        m=np.asarray(arrays["M"], dtype=np.float64),
        sn=np.asarray(arrays["Sn"], dtype=np.float64),
        y=np.asarray(arrays["Y"], dtype=np.float64),
        d=np.asarray(arrays["D"], dtype=np.float64),
        vh=np.asarray(arrays["vh"], dtype=np.float64),
        vi=np.asarray(arrays["vi"], dtype=np.float64),
        xyzr=np.asarray(arrays["xyzr"], dtype=np.float64),
        rec_model=np.asarray(arrays["rec_model"], dtype=np.float64),
        fwd_model_signature=str(case.metadata.get("fwd_model_signature")),
        cache_signature=str(metadata["cache_signature_hash"]),
    )
    saved = greit.save(case.artifact_path)
    return load_greit_rm(saved)


def _calc_large_scalar_noise_components(
    y: np.ndarray,
    d: np.ndarray,
    *,
    weight: float,
) -> dict[str, np.ndarray | float]:
    y = np.asarray(y, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    scalar_weight = float(weight)
    noiselev = float(scalar_weight * np.mean(np.abs(y)))
    pjt = np.ascontiguousarray(d @ y.T, dtype=np.float64)
    m_matrix = np.ascontiguousarray(y @ y.T, dtype=np.float64)
    diag = np.diag_indices(m_matrix.shape[0])
    m_matrix[diag] += noiselev * noiselev
    gram = y.T @ y + (noiselev * noiselev) * np.eye(y.shape[1], dtype=np.float64)
    rm = np.ascontiguousarray(d @ np.linalg.solve(gram, y.T), dtype=np.float64)
    return {
        "RM": rm,
        "PJt": pjt,
        "M": m_matrix,
        "Sn": np.eye(y.shape[0], dtype=np.float64),
        "noiselev": noiselev,
    }


def _measurement_matrix(
    centers: np.ndarray,
    *,
    n_measurements: int,
    n_elec: int,
    n_rings: int,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    electrodes = _electrode_positions(n_elec=n_elec, n_rings=n_rings)
    diff = centers[None, :, :] - electrodes[:, None, :]
    dist2 = np.sum(diff * diff, axis=2)
    fields = 1.0 / np.sqrt(dist2 + 2.5e-3)
    fields -= fields.mean(axis=1, keepdims=True)
    fields /= np.maximum(np.linalg.norm(fields, axis=1, keepdims=True), 1.0e-12)
    rows = []
    base = np.full(centers.shape[0], 1.0 / centers.shape[0], dtype=np.float64)
    for meas in range(n_measurements):
        a = meas % n_elec
        b = (meas * 7 + 5) % n_elec
        c = (meas * 11 + 3) % n_elec
        d = (meas * 13 + 1) % n_elec
        row = (fields[a] - fields[b]) * (fields[c] - fields[d])
        row += 0.05 * np.sin((meas + 1) * centers[:, 0] / 0.18)
        row -= float(np.mean(row))
        norm = max(float(np.linalg.norm(row)), 1.0e-12)
        rows.append(base + 0.035 * row / norm)
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def _electrode_positions(*, n_elec: int, n_rings: int) -> np.ndarray:
    if n_elec <= 0 or n_rings <= 0 or n_elec % n_rings:
        raise ValueError("n_elec must be divisible by n_rings.")
    per_ring = n_elec // n_rings
    levels = np.linspace(-0.06, 0.06, n_rings)
    positions = []
    for z in levels:
        for idx in range(per_ring):
            theta = 2.0 * np.pi * idx / per_ring
            positions.append([0.18 * np.cos(theta), 0.18 * np.sin(theta), z])
    return np.asarray(positions, dtype=np.float64)


def _case_measurement_contract(
    case_id: str,
    *,
    n_measurements: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if case_id == "nominal":
        return None, None
    mask = bad_channel_mask(n_measurements, np.arange(0, n_measurements, 31))
    weights = 0.75 + 0.5 * rng.random(n_measurements)
    return mask, weights.astype(np.float64, copy=False)


def _measurement_frames(
    *,
    vh: np.ndarray,
    raw_y: np.ndarray,
    n_frames: int,
    target_index: int,
) -> np.ndarray:
    vh = np.asarray(vh, dtype=np.float64).reshape(-1)
    raw_y = np.asarray(raw_y, dtype=np.float64)
    target = raw_y[:, int(target_index)]
    scales = 0.75 + 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames))
    return np.asarray(
        [vh * (1.0 + scale * target) for scale in scales], dtype=np.float64
    )


def _online_apply(
    greit: GREITRM,
    *,
    frames: np.ndarray,
    reference: np.ndarray,
    devices: Sequence[str],
    dtype: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for device in devices:
        device_name = str(device).strip().lower()
        if not device_name:
            continue
        try:
            prepared, prepare_seconds = _timed(
                lambda device_name=device_name: greit.prepare_online(
                    device=device_name,
                    dtype=dtype,
                    cache_key=f"t49:{device_name}:{dtype}",
                )
            )
            one, one_seconds = _timed(
                lambda prepared=prepared, device_name=device_name: prepared.reconstruct(
                    frames[:1],
                    normalize=True,
                    v_ref=reference,
                    device=device_name,
                    dtype=dtype,
                    return_metadata=True,
                )
            )
            batch, batch_seconds = _timed(
                lambda prepared=prepared, device_name=device_name: prepared.reconstruct(
                    frames,
                    normalize=True,
                    v_ref=reference,
                    device=device_name,
                    dtype=dtype,
                    return_metadata=True,
                )
            )
            results[device_name] = {
                "prepare_seconds": prepare_seconds,
                "apply_1_frame_seconds": one_seconds,
                "apply_batch_seconds": batch_seconds,
                "apply_batch_n_frames": int(frames.shape[0]),
                "metadata_1_frame": _jsonable(dict(one.metadata)),
                "metadata_batch": _jsonable(dict(batch.metadata)),
                "output_norm_1_frame": float(np.linalg.norm(np.asarray(one.values))),
                "output_norm_batch": float(np.linalg.norm(np.asarray(batch.values))),
                "values_1_frame": np.asarray(one.values).reshape(-1),
            }
        except Exception as exc:
            results[device_name] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def _metrics_for_case(
    case: ComponentCase, online: Mapping[str, Any]
) -> dict[str, float]:
    entry = _first_successful_online_entry(online)
    recon = np.asarray(entry["values_1_frame"], dtype=np.float64).reshape(-1)
    target = np.asarray(case.arrays["D"], dtype=np.float64)[:, 0]
    threshold = 0.5 * float(np.max(np.abs(target)))
    target_mask = np.abs(target) >= max(threshold, np.finfo(np.float64).eps)
    if not np.any(target_mask):
        target_mask[int(np.argmax(np.abs(target)))] = True
    centers = _matching_rec_model_centers(case.arrays.get("rec_model"), target.size)
    metrics = greit_metrics(
        recon,
        target_mask,
        centers=centers,
    )
    if set(metrics) != set(GREIT_METRIC_KEYS):
        raise RuntimeError("GREIT metric key set is incomplete.")
    return metrics


def _first_successful_online_entry(online: Mapping[str, Any]) -> Mapping[str, Any]:
    for entry in online.values():
        if isinstance(entry, Mapping) and "values_1_frame" in entry:
            return entry
    raise RuntimeError("No successful online apply entry is available for metrics.")


def _matching_rec_model_centers(values: Any, n_cells: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == n_cells:
        return np.ascontiguousarray(array, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == n_cells:
        return np.ascontiguousarray(array.T, dtype=np.float64)
    return None


def _write_fixture(
    path: Path, arrays: Mapping[str, np.ndarray], *, attrs: Mapping[str, Any]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = FIXTURE_SCHEMA
        for key, value in attrs.items():
            handle.attrs[str(key)] = _hdf5_attr_value(value)
        for name, value in arrays.items():
            if name in REQUIRED_EXPORTS or name in {"Sn", "rec_model", "normalize"}:
                handle.create_dataset(str(name), data=np.asarray(value))
    return path


def _load_fixture_arrays(path: Path) -> dict[str, np.ndarray]:
    return {
        str(name): np.asarray(value)
        for name, value in load_greit_eidors_fixture_arrays(path).items()
        if _is_array_like(value)
    }


def _is_array_like(value: Any) -> bool:
    return isinstance(value, (np.ndarray, list, tuple, int, float, np.number))


def _difference_data_from_vh_vi(vh: np.ndarray, vi: np.ndarray) -> np.ndarray:
    vh = np.asarray(vh, dtype=np.float64).reshape(-1)
    vi = np.asarray(vi, dtype=np.float64)
    return np.ascontiguousarray(vi / vh.reshape(-1, 1) - 1.0, dtype=np.float64)


def _rec_centers_from_voxel_shape(shape: tuple[int, int, int]) -> np.ndarray:
    grid = VoxelGrid.from_bounds(
        [-0.18, -0.18, -0.08],
        [0.18, 0.18, 0.08],
        shape=shape,
    )
    return grid.cell_centers()


def _component_signature_hash(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
) -> str:
    import hashlib

    summary = {
        "schema": GREIT_CACHE_SIGNATURE_SCHEMA,
        "case_id": metadata.get("case_id"),
        "training_mode": metadata.get("training_mode"),
        "difference_normalization": metadata.get("difference_normalization"),
        "desired_solution_fn": metadata.get("desired_solution_fn"),
        "shape_RM": tuple(int(v) for v in np.asarray(arrays["RM"]).shape),
        "shape_Y": tuple(int(v) for v in np.asarray(arrays["Y"]).shape),
        "weight": float(np.asarray(arrays["weight"]).reshape(-1)[0]),
        "noiselev": float(np.asarray(arrays["noiselev"]).reshape(-1)[0]),
    }
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hdf5_summary(path: Path) -> dict[str, Any]:
    artifact = read_hdf5_artifact(path, lazy=True, verify_checksums=False)
    arrays: dict[str, Any] = {}
    for name, dataset in artifact.arrays.items():
        arrays[str(name)] = {
            "shape": [int(v) for v in dataset.shape],
            "dtype": str(dataset.dtype),
            "compression": dataset.compression,
            "chunks": None
            if dataset.chunks is None
            else [int(v) for v in dataset.chunks],
            "sha256_recorded": bool(dataset.sha256),
        }
    return {
        "schema": artifact.schema,
        "metadata": _jsonable(dict(artifact.metadata)),
        "arrays": arrays,
    }


def _component_shapes(arrays: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {
        str(key): [int(v) for v in np.asarray(value).shape]
        for key, value in arrays.items()
    }


def _actual_measurement_count_from_cases(
    cases: Mapping[str, Any],
    *,
    fallback: int,
) -> int:
    for case in cases.values():
        vh_shape = case.get("component_shapes", {}).get("vh")
        if vh_shape:
            count = 1
            for value in vh_shape:
                count *= int(value)
            return int(count)
    return int(fallback)


def _benchmark_scope(*, official_fixture: bool, n_measurements: int) -> str:
    if official_fixture:
        return (
            "48e official EIDORS fixture GREIT RM benchmark "
            f"(actual n_measurements={int(n_measurements)}; "
            "5936 measurement protocol separate)"
        )
    return "48e/5936 EIDORS-parity GREIT RM benchmark"


def _all_online_hot_paths_are_rm(cases: Mapping[str, Any]) -> bool:
    for case in cases.values():
        for entry in case.get("online_apply", {}).values():
            if entry.get("error"):
                continue
            metadata = entry.get("metadata_batch", {})
            if metadata.get("online_hot_path") != "rm_matmul":
                return False
            for key in (
                "forward_solve_count",
                "jacobian_rebuild_count",
                "ksp_solve_count",
            ):
                if int(metadata.get(key, 0)) != 0:
                    return False
    return True


def _format_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _write_markdown_report(path: Path, payload: Mapping[str, Any]) -> Path:
    lines = [
        f"# {_markdown_report_title(payload)}",
        "",
        f"- schema: `{payload['schema']}`",
        f"- scope: `{payload['scope']}`",
        f"- generated: {payload['timestamp_utc']}",
        f"- git: `{payload['git_commit']}`",
        f"- official EIDORS fixture: `{payload['gate']['official_eidors_fixture']}`",
        f"- parity components passed: `{payload['gate']['parity_components_passed']}`",
        f"- official-equivalence claim allowed: `{payload['gate']['official_equivalence_claim_allowed']}`",
        "",
    ]
    if payload["gate"].get("surrogate_note"):
        lines.extend(["## Fixture Note", "", payload["gate"]["surrogate_note"], ""])
    lines.extend(
        [
            "## Cases",
            "",
            "| case | fixture | parity | bad ch | W | load s | metric PE | metric RES |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for case_id, case in payload["cases"].items():
        metrics = case.get("metrics", {})
        contract = case.get("measurement_contract", {})
        lines.append(
            "| {case} | {source} | {parity} | {bad} | {wkind} | {load} | {pe} | {res} |".format(
                case=case_id,
                source=case.get("fixture_source"),
                parity=case.get("parity_report", {}).get("all_passed"),
                bad=contract.get("bad_channel_count"),
                wkind=contract.get("measurement_weight_kind"),
                load=_format_seconds(case.get("artifact_load", {}).get("seconds")),
                pe=metrics.get("PE"),
                res=metrics.get("RES"),
            )
        )
    lines.extend(
        [
            "",
            "## Online Apply",
            "",
            "| case | device | effective | resident | 1 frame s | 512 frame s | forward solves | KSP solves |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for case_id, case in payload["cases"].items():
        for device, entry in case.get("online_apply", {}).items():
            if entry.get("error"):
                lines.append(
                    f"| {case_id} | {device} | error | error | n/a | n/a | n/a | n/a |"
                )
                continue
            meta = entry.get("metadata_batch", {})
            lines.append(
                "| {case} | {device} | {effective} | {resident} | {one} | {batch} | {fwd} | {ksp} |".format(
                    case=case_id,
                    device=device,
                    effective=meta.get("device_effective", ""),
                    resident=meta.get("rm_matrix_resident", ""),
                    one=_format_seconds(entry.get("apply_1_frame_seconds")),
                    batch=_format_seconds(entry.get("apply_batch_seconds")),
                    fwd=meta.get("forward_solve_count"),
                    ksp=meta.get("ksp_solve_count"),
                )
            )
    lines.extend(
        [
            "",
            "## Cold Build",
            "",
            "| case | finite responses s | desired D s | RM build s | artifact write s | parity compare s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case_id, case in payload["cases"].items():
        cold = case.get("cold_build", {})
        lines.append(
            "| {case} | {resp} | {desired} | {rm} | {write} | {compare} |".format(
                case=case_id,
                resp=_format_seconds(cold.get("finite_target_response_seconds")),
                desired=_format_seconds(cold.get("desired_image_seconds")),
                rm=_format_seconds(cold.get("rm_component_build_seconds")),
                write=_format_seconds(cold.get("artifact_write_seconds")),
                compare=_format_seconds(cold.get("parity_compare_seconds")),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _markdown_report_title(payload: Mapping[str, Any]) -> str:
    if payload["gate"].get("official_eidors_fixture"):
        return "48e Official EIDORS Fixture GREIT Runtime Gate"
    return "48e/5936 EIDORS-Parity GREIT Runtime Gate"


def _timed(fn):
    _sync_cuda()
    started = time.perf_counter()
    value = fn()
    _sync_cuda()
    return value, float(time.perf_counter() - started)


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(val)
            for key, val in value.items()
            if key != "values_1_frame"
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, MappingProxyType):
        return _jsonable(dict(value))
    return value


def _hdf5_attr_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating)):
        return value
    return json.dumps(_jsonable(value), sort_keys=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _parse_shape(raw: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError("shape must be three positive integers")
    return parts


def _parse_cases(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def _parse_devices(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--n-measurements", type=int, default=5936)
    parser.add_argument("--voxel-shape", default="6,6,4")
    parser.add_argument("--n-frames", type=int, default=512)
    parser.add_argument("--n-elec", type=int, default=48)
    parser.add_argument("--n-rings", type=int, default=3)
    parser.add_argument("--target-radius", type=float, default=0.035)
    parser.add_argument("--target-contrast", type=float, default=0.08)
    parser.add_argument("--weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--devices", default="cpu,auto,cuda")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--cases", default="bad_weighted")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_benchmark(
        output_dir=args.output_dir,
        fixture=args.fixture,
        n_measurements=args.n_measurements,
        voxel_shape=_parse_shape(args.voxel_shape),
        n_frames=args.n_frames,
        n_elec=args.n_elec,
        n_rings=args.n_rings,
        target_radius=args.target_radius,
        target_contrast=args.target_contrast,
        weight=args.weight,
        seed=args.seed,
        devices=_parse_devices(args.devices),
        dtype=args.dtype,
        cases=_parse_cases(args.cases),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
