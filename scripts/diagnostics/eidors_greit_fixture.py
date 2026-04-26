#!/usr/bin/env python3
"""Validate EIDORS GREIT source maps and captured MATLAB fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_MAP_SCHEMA = "pyeidors-eidors-greit-source-map-v1"
FIXTURE_SCHEMA = "pyeidors-eidors-greit-fixture-v1"
MANIFEST_SCHEMA = "pyeidors-eidors-greit-fixture-manifest-v1"

REQUIRED_EXPORTS = (
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

REQUIRED_OFFICIAL_FUNCTIONS = (
    "GREIT3D_distribution",
    "mk_GREIT_model",
    "mk_GREIT_model/stim_targets",
    "simulate_movement",
    "calc_GREIT_RM",
    "calc_GREIT_RM/calc_PJt",
    "ng_mk_cyl_models",
)

REQUIRED_CASE_IDS = ("tiny_3d_cylinder", "reduced_48e_5936")


def default_source_map_path() -> Path:
    return Path(__file__).with_name("eidors_greit_source_map.json")


def load_source_map(path: str | Path | None = None) -> dict[str, Any]:
    source = default_source_map_path() if path is None else Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def validate_source_map(payload: Mapping[str, Any]) -> None:
    schema = str(payload.get("schema", ""))
    if schema != SOURCE_MAP_SCHEMA:
        raise ValueError(f"unexpected source-map schema {schema!r}")
    _require_all(
        payload.get("required_exports") or (),
        REQUIRED_EXPORTS,
        label="source-map required_exports",
    )
    official_ids = [
        str(item.get("id", ""))
        for item in payload.get("official_functions") or ()
        if isinstance(item, Mapping)
    ]
    _require_all(official_ids, REQUIRED_OFFICIAL_FUNCTIONS, label="official functions")
    for item in payload.get("official_functions") or ():
        if not isinstance(item, Mapping):
            raise ValueError("official_functions entries must be objects")
        url = str(item.get("official_url", ""))
        if not url.startswith("https://eidors3d.sourceforge.net/doc/eidors/"):
            raise ValueError(f"official URL is not an EIDORS doc URL: {url!r}")
        if not item.get("parity_fields"):
            raise ValueError(f"missing parity_fields for {item.get('id')!r}")

    case_ids = [
        str(item.get("case_id", ""))
        for item in payload.get("fixture_cases") or ()
        if isinstance(item, Mapping)
    ]
    _require_all(case_ids, REQUIRED_CASE_IDS, label="fixture cases")


def validate_fixture_hdf5(path: str | Path) -> dict[str, Any]:
    import h5py

    source = Path(path)
    with h5py.File(source, "r") as handle:
        names = set(str(name) for name in handle.keys())
        _require_all(names, REQUIRED_EXPORTS, label=f"{source} root datasets")
        schema = _read_hdf5_string(handle, "schema")
        if schema and schema != FIXTURE_SCHEMA:
            raise ValueError(f"unexpected fixture schema {schema!r}")
        shapes = {
            name: _hdf5_export_shape(handle[name])
            for name in REQUIRED_EXPORTS
            if name in handle
        }
    return {
        "schema": FIXTURE_SCHEMA,
        "path": str(source),
        "exports": list(REQUIRED_EXPORTS),
        "shapes": shapes,
    }


def build_manifest(fixture_paths: Sequence[str | Path]) -> dict[str, Any]:
    fixtures = [validate_fixture_hdf5(path) for path in fixture_paths]
    return {
        "schema": MANIFEST_SCHEMA,
        "source_map": str(default_source_map_path()),
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _require_all(values: Iterable[Any], required: Iterable[str], *, label: str) -> None:
    present = {str(value) for value in values}
    missing = [value for value in required if value not in present]
    if missing:
        raise ValueError(f"{label} missing: {', '.join(missing)}")


def _read_hdf5_string(handle: Any, key: str) -> str:
    if key in handle.attrs:
        value = handle.attrs[key]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        decoded = _decode_hdf5_string_value(value)
        if decoded:
            return decoded
        return str(value)
    if key not in handle:
        return ""
    dataset = handle[key]
    try:
        value = dataset[()]
    except Exception:
        return ""
    decoded = _decode_hdf5_string_value(value)
    if decoded:
        return decoded
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _hdf5_export_shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(v) for v in value.shape)
    if "MATLAB_sparse" in value.attrs and "jc" in value:
        return (int(value.attrs["MATLAB_sparse"]), int(value["jc"].shape[0]) - 1)
    return ()


def _decode_hdf5_string_value(value: Any) -> str:
    """Decode Python, h5py, or MATLAB v7.3 char-array string payloads."""

    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    try:
        import numpy as np
    except Exception:
        return ""

    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.str_):
        return str(value)
    if not isinstance(value, np.ndarray):
        return ""
    if value.dtype.kind in {"S", "U"}:
        return "".join(str(item) for item in value.reshape(-1)).strip()
    if value.dtype.kind not in {"i", "u"}:
        return ""
    codes = [int(item) for item in value.reshape(-1)]
    if not codes or any(code < 0 or code > 0x10FFFF for code in codes):
        return ""
    return "".join(chr(code) for code in codes if code).strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-map",
        type=Path,
        default=default_source_map_path(),
        help="EIDORS GREIT source-map JSON to validate.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        action="append",
        default=[],
        help="Captured MATLAB v7.3 fixture .mat/.h5 file. Repeatable.",
    )
    parser.add_argument("--manifest-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_source_map(load_source_map(args.source_map))
    if args.fixture:
        manifest = build_manifest(args.fixture)
        if args.manifest_out is not None:
            write_manifest(args.manifest_out, manifest)
        else:
            print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
