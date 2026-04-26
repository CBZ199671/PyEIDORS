#!/usr/bin/env python3
"""Compare MATLAB EIDORS GREIT fixtures against PyEIDORS GREIT components."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyeidors.inverse.greit import GREIT_METRIC_KEYS, calc_greit_rm, greit_metrics
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


REPORT_SCHEMA = "pyeidors-greit-eidors-parity-report-v1"
REQUIRED_FIXTURE_FIELDS = (
    "vh",
    "vi",
    "xyzr",
    "D",
    "Y",
    "PJt",
    "M",
    "noiselev",
    "RM",
    "weight",
)
DEFAULT_COMPONENTS = ("Y", "D", "PJt", "M", "noiselev", "RM", "RM@dv", "metrics")


@dataclass(frozen=True)
class ComparisonResult:
    name: str
    passed: bool
    max_abs_error: float
    relative_error: float
    abs_tolerance: float | str
    rel_tolerance: float | str
    shape_eidors: tuple[int, ...] | Mapping[str, tuple[int, ...]]
    shape_pyeidors: tuple[int, ...] | Mapping[str, tuple[int, ...]]
    source: str

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "max_abs_error": _json_ready_float(self.max_abs_error),
            "relative_error": _json_ready_float(self.relative_error),
            "abs_tolerance": self.abs_tolerance,
            "rel_tolerance": self.rel_tolerance,
            "shape_eidors": _json_ready_shape(self.shape_eidors),
            "shape_pyeidors": _json_ready_shape(self.shape_pyeidors),
            "source": self.source,
        }


def compare_greit_eidors_parity(
    fixture_path: str | Path,
    *,
    pyeidors_artifact: str | Path | None = None,
    report_out: str | Path | None = None,
    abs_tol: float = 1.0e-9,
    rel_tol: float = 1.0e-9,
    dv_index: int = 0,
    normalize: bool | None = None,
) -> dict[str, Any]:
    """Build a GREIT parity drift report for one captured EIDORS fixture."""

    fixture = _load_fixture_arrays(fixture_path)
    _require_fixture_fields(fixture, fixture_path)
    official = _official_components(fixture)
    py = (
        _load_pyeidors_components(pyeidors_artifact)
        if pyeidors_artifact is not None
        else _compute_pyeidors_components_from_fixture(fixture, normalize=normalize)
    )
    if "Y" not in py:
        py["Y"] = _difference_data_from_vh_vi(
            official["vh"],
            official["vi"],
            normalize=_normalize_flag(fixture, normalize),
        )
    if "D" not in py:
        py["D"] = official["D"]

    target_index = _validate_target_index(dv_index, official["Y"])
    official_recon = _rm_times_dv(official["RM"], official["Y"], target_index)
    py_recon = _rm_times_dv(py["RM"], official["Y"], target_index)
    official_metrics = _metrics_for_reconstruction(
        official_recon,
        official["D"],
        fixture,
        target_index=target_index,
    )
    py_metrics = _metrics_for_reconstruction(
        py_recon,
        official["D"],
        fixture,
        target_index=target_index,
    )

    comparisons = [
        _compare_array("Y", official["Y"], py["Y"], abs_tol, rel_tol, source="vh/vi"),
        _compare_array("D", official["D"], py["D"], abs_tol, rel_tol, source="D"),
        _compare_array(
            "PJt", official["PJt"], py["PJt"], abs_tol, rel_tol, source="calc_GREIT_RM"
        ),
        _compare_array(
            "M", official["M"], py["M"], abs_tol, rel_tol, source="calc_GREIT_RM"
        ),
        _compare_array(
            "noiselev",
            official["noiselev"],
            py["noiselev"],
            abs_tol,
            rel_tol,
            source="calc_GREIT_RM",
        ),
        _compare_array(
            "RM",
            official["RM"],
            py["RM"],
            abs_tol,
            rel_tol,
            source="calc_GREIT_RM",
        ),
        _compare_array(
            "RM@dv",
            official_recon,
            py_recon,
            abs_tol,
            rel_tol,
            source=f"target_index={target_index}",
        ),
        _compare_metrics(
            official_metrics,
            py_metrics,
            abs_tol,
            rel_tol,
            source=f"greit_metrics target_index={target_index}",
        ),
    ]
    payload = {
        "schema": REPORT_SCHEMA,
        "fixture_path": str(Path(fixture_path)),
        "pyeidors_artifact_path": None
        if pyeidors_artifact is None
        else str(Path(pyeidors_artifact)),
        "component_order": list(DEFAULT_COMPONENTS),
        "tolerances": {
            name: {"abs": abs_tol, "rel": rel_tol} for name in DEFAULT_COMPONENTS
        },
        "tolerance_source": "cli-or-default",
        "target_index": target_index,
        "normalize": _normalize_flag(fixture, normalize),
        "all_passed": all(item.passed for item in comparisons),
        "comparisons": [item.as_json() for item in comparisons],
        "eidors_shapes": {
            name: list(map(int, np.asarray(official[name]).shape))
            for name in REQUIRED_FIXTURE_FIELDS
        },
        "pyeidors_source": "artifact"
        if pyeidors_artifact is not None
        else "computed_from_fixture_components",
        "metric_keys": list(GREIT_METRIC_KEYS),
    }
    if report_out is not None:
        write_parity_report(report_out, payload)
    return payload


def write_parity_report(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _load_fixture_arrays(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        return _load_hdf5_fixture(source)
    except OSError:
        return _load_matlab_v7_fixture(source)


def _load_hdf5_fixture(path: Path) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            if isinstance(handle[key], h5py.Dataset):
                arrays[str(key)] = _dataset_value(handle[key])
        for key, value in handle.attrs.items():
            arrays.setdefault(str(key), _attr_value(value))
    return arrays


def _load_matlab_v7_fixture(path: Path) -> dict[str, Any]:
    from scipy.io import loadmat

    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {
        str(key): value
        for key, value in payload.items()
        if not str(key).startswith("__")
    }


def _dataset_value(dataset: h5py.Dataset) -> Any:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return np.asarray(value)


def _attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_fixture_fields(fixture: Mapping[str, Any], path: str | Path) -> None:
    missing = [name for name in REQUIRED_FIXTURE_FIELDS if name not in fixture]
    if missing:
        raise ValueError(f"EIDORS GREIT fixture {path} missing: {', '.join(missing)}")


def _official_components(fixture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "vh": _as_vector(fixture["vh"], name="vh"),
        "vi": _as_matrix(fixture["vi"], name="vi"),
        "xyzr": _as_matrix(fixture["xyzr"], name="xyzr"),
        "D": _as_matrix(fixture["D"], name="D"),
        "Y": _as_matrix(fixture["Y"], name="Y"),
        "PJt": _as_matrix(fixture["PJt"], name="PJt"),
        "M": _as_matrix(fixture["M"], name="M"),
        "noiselev": _as_scalar_array(fixture["noiselev"], name="noiselev"),
        "RM": _as_matrix(fixture["RM"], name="RM"),
        "weight": _as_scalar_array(fixture["weight"], name="weight"),
    }


def _compute_pyeidors_components_from_fixture(
    fixture: Mapping[str, Any],
    *,
    normalize: bool | None,
) -> dict[str, np.ndarray]:
    official = _official_components(fixture)
    py_y = _difference_data_from_vh_vi(
        official["vh"],
        official["vi"],
        normalize=_normalize_flag(fixture, normalize),
    )
    py_d = official["D"]
    components = calc_greit_rm(
        py_y,
        py_d,
        weight=float(official["weight"].reshape(-1)[0]),
        noise_covar=_noise_covar_from_fixture(fixture),
    )
    return {
        "Y": py_y,
        "D": py_d,
        "PJt": components.pjt,
        "M": components.m,
        "noiselev": np.asarray([components.noiselev], dtype=np.float64),
        "RM": components.rm,
    }


def _load_pyeidors_components(path: str | Path) -> dict[str, np.ndarray]:
    artifact = read_hdf5_artifact(path)
    arrays = dict(artifact.arrays)
    return {
        "Y": _array_from_aliases(arrays, "Y", "y"),
        "D": _array_from_aliases(arrays, "D", "d"),
        "PJt": _array_from_aliases(arrays, "PJt", "pjt"),
        "M": _array_from_aliases(arrays, "M", "m"),
        "noiselev": _array_from_aliases(arrays, "noiselev"),
        "RM": _array_from_aliases(arrays, "RM", "rm"),
    }


def _array_from_aliases(arrays: Mapping[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name], dtype=np.float64)
    raise ValueError(f"PyEIDORS artifact missing any of: {', '.join(names)}")


def _difference_data_from_vh_vi(
    vh: np.ndarray, vi: np.ndarray, *, normalize: bool
) -> np.ndarray:
    if vi.shape[0] != vh.size:
        raise ValueError(f"vi rows {vi.shape[0]} do not match vh length {vh.size}.")
    if normalize:
        denom = vh.reshape(-1, 1)
        if np.any(np.abs(denom) <= np.finfo(np.float64).eps):
            raise ValueError(
                "vh contains zero entries; ratio normalization is undefined."
            )
        return np.ascontiguousarray(vi / denom - 1.0, dtype=np.float64)
    return np.ascontiguousarray(vi - vh.reshape(-1, 1), dtype=np.float64)


def _normalize_flag(fixture: Mapping[str, Any], override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    for key in ("normalize", "mdl_normalize", "difference_normalization"):
        if key in fixture:
            value = fixture[key]
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "ratio", "normalized"}
            return bool(np.asarray(value).reshape(-1)[0])
    return True


def _noise_covar_from_fixture(fixture: Mapping[str, Any]) -> Any:
    for key in ("noise_covar", "Sn"):
        if key in fixture:
            return np.asarray(fixture[key], dtype=np.float64)
    return 1.0


def _validate_target_index(index: int, y: np.ndarray) -> int:
    target_index = int(index)
    if target_index < 0 or target_index >= y.shape[1]:
        raise ValueError(
            f"target index {target_index} outside Y target count {y.shape[1]}."
        )
    return target_index


def _rm_times_dv(rm: np.ndarray, y: np.ndarray, target_index: int) -> np.ndarray:
    return np.ascontiguousarray(rm @ y[:, int(target_index)], dtype=np.float64)


def _metrics_for_reconstruction(
    recon: np.ndarray,
    d: np.ndarray,
    fixture: Mapping[str, Any],
    *,
    target_index: int,
) -> dict[str, float]:
    target = np.asarray(d[:, int(target_index)], dtype=np.float64).reshape(-1)
    abs_target = np.abs(target)
    threshold = 0.5 * float(np.max(abs_target))
    if threshold <= np.finfo(np.float64).eps:
        mask = abs_target > 0.0
    else:
        mask = abs_target >= threshold
    if not np.any(mask):
        mask[int(np.argmax(abs_target))] = True
    centers = _optional_centers(fixture, n_cells=target.size)
    return greit_metrics(
        recon,
        mask,
        centers=centers,
        target_values=target,
    )


def _optional_centers(fixture: Mapping[str, Any], *, n_cells: int) -> np.ndarray | None:
    for key in ("rec_centers", "rec_model", "centers"):
        if key not in fixture:
            continue
        array = np.asarray(fixture[key], dtype=np.float64)
        if array.ndim == 2 and array.shape[0] == n_cells:
            return array
        if array.ndim == 2 and array.shape[1] == n_cells:
            return array.T
    return None


def _compare_array(
    name: str,
    eidors: Any,
    pyeidors: Any,
    abs_tol: float,
    rel_tol: float,
    *,
    source: str,
) -> ComparisonResult:
    left = np.asarray(eidors, dtype=np.float64)
    right = np.asarray(pyeidors, dtype=np.float64)
    if left.shape != right.shape:
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs_error=float("inf"),
            relative_error=float("inf"),
            abs_tolerance=abs_tol,
            rel_tolerance=rel_tol,
            shape_eidors=tuple(int(v) for v in left.shape),
            shape_pyeidors=tuple(int(v) for v in right.shape),
            source=source,
        )
    delta = np.asarray(left - right, dtype=np.float64)
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    denom = max(1.0, float(np.max(np.abs(left))) if left.size else 0.0)
    rel = float(max_abs / denom)
    return ComparisonResult(
        name=name,
        passed=bool(max_abs <= abs_tol or rel <= rel_tol),
        max_abs_error=max_abs,
        relative_error=rel,
        abs_tolerance=abs_tol,
        rel_tolerance=rel_tol,
        shape_eidors=tuple(int(v) for v in left.shape),
        shape_pyeidors=tuple(int(v) for v in right.shape),
        source=source,
    )


def _compare_metrics(
    eidors: Mapping[str, float],
    pyeidors: Mapping[str, float],
    abs_tol: float,
    rel_tol: float,
    *,
    source: str,
) -> ComparisonResult:
    left = np.asarray(
        [float(eidors[key]) for key in GREIT_METRIC_KEYS], dtype=np.float64
    )
    right = np.asarray(
        [float(pyeidors[key]) for key in GREIT_METRIC_KEYS], dtype=np.float64
    )
    result = _compare_array("metrics", left, right, abs_tol, rel_tol, source=source)
    return ComparisonResult(
        name=result.name,
        passed=result.passed,
        max_abs_error=result.max_abs_error,
        relative_error=result.relative_error,
        abs_tolerance=result.abs_tolerance,
        rel_tolerance=result.rel_tolerance,
        shape_eidors={key: () for key in GREIT_METRIC_KEYS},
        shape_pyeidors={key: () for key in GREIT_METRIC_KEYS},
        source=result.source,
    )


def _as_vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_matrix(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty 2D matrix.")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_scalar_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar.")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _json_ready_float(value: float) -> float | str:
    if np.isfinite(value):
        return float(value)
    if np.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _json_ready_shape(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready_shape(val) for key, val in value.items()}
    return [int(v) for v in tuple(value)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--pyeidors-artifact", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--abs-tol", type=float, default=1.0e-9)
    parser.add_argument("--rel-tol", type=float, default=1.0e-9)
    parser.add_argument("--dv-index", type=int, default=0)
    parser.add_argument(
        "--normalize",
        choices=("auto", "ratio", "raw"),
        default="auto",
        help="Difference-data mode. auto reads fixture normalize if present, else ratio.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    normalize = None
    if args.normalize == "ratio":
        normalize = True
    elif args.normalize == "raw":
        normalize = False
    report = compare_greit_eidors_parity(
        args.fixture,
        pyeidors_artifact=args.pyeidors_artifact,
        report_out=args.report_out,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        dv_index=args.dv_index,
        normalize=normalize,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
