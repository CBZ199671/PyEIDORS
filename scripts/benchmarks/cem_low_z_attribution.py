#!/usr/bin/env python3
"""Cross low-z CEM assemblies and linear algebra backends."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_exact_extension_suite import (
    ATTRIBUTION_CASE_IDS,
    EXTENSION_CASES,
    FORMULATIONS,
    _sqrt_fraction_exact,
    exact_extension_accuracy_metrics,
    extension_case_cell_conductivities,
    extension_case_mesh,
    solve_exact_extension_case,
)
from scripts.benchmarks.cem_fair_common import (
    _classic_state,
    _robin_state,
    _solve_classic,
    _solve_robin,
)
from scripts.benchmarks.compare_cem_formulations import configure_fonts, write_json


ATTRIBUTION_SCHEMA = "cem-low-z-attribution-v1"
ASSEMBLIES = ("pyeidors", "eidors", "ngsolve")
BACKENDS = ("scipy_superlu", "matlab_sparse_lu")


def _update_array_digest(digest: Any, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))


def block_payload_sha256(
    robin_matrix: Any,
    coupling: Any,
    electrode_matrix: Any,
    currents: np.ndarray,
) -> str:
    """Hash canonical CSC block values and the exact float64 RHS payload."""

    digest = hashlib.sha256(b"cem-low-z-block-payload-v1")
    for matrix in (robin_matrix, coupling, electrode_matrix):
        canonical = csc_matrix(matrix, dtype=np.float64)
        canonical.sum_duplicates()
        canonical.sort_indices()
        _update_array_digest(digest, np.asarray(canonical.shape, dtype="<i8"))
        _update_array_digest(digest, canonical.indptr.astype("<i8", copy=False))
        _update_array_digest(digest, canonical.indices.astype("<i8", copy=False))
        _update_array_digest(digest, canonical.data.astype("<f8", copy=False))
    _update_array_digest(
        digest,
        np.ascontiguousarray(currents, dtype="<f8"),
    )
    return digest.hexdigest()


def _accumulate_contributions(
    contributions: Iterable[tuple[int, int, float]],
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float64)
    for row, column, value in contributions:
        result[int(row), int(column)] += float(value)
    return result


def assembly_order_sensitivity(
    contributions: list[tuple[int, int, float]],
    *,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Compare forward and reverse float64 accumulation of identical terms."""

    forward = _accumulate_contributions(contributions, shape)
    reverse = _accumulate_contributions(reversed(contributions), shape)
    delta = forward - reverse
    forward_digest = hashlib.sha256(
        np.ascontiguousarray(forward, dtype="<f8")
    ).hexdigest()
    reverse_digest = hashlib.sha256(
        np.ascontiguousarray(reverse, dtype="<f8")
    ).hexdigest()
    return {
        "forward_reverse_max_abs": float(np.max(np.abs(delta))),
        "forward_reverse_relative_frobenius": float(
            np.linalg.norm(delta) / max(np.linalg.norm(forward), np.finfo(float).tiny)
        ),
        "forward_sha256": forward_digest,
        "reverse_sha256": reverse_digest,
    }


def classify_attribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered three-of-four dominance rule."""

    if {str(record["case_id"]) for record in records} != set(ATTRIBUTION_CASE_IDS):
        raise ValueError("attribution requires exactly the four preregistered cases")
    labels = {
        "assembly_implementation_dominant": "assembly_effect_log10",
        "linear_backend_dominant": "backend_effect_log10",
        "pure_accumulation_order_dominant": "order_effect_log10",
    }
    support = {label: 0 for label in labels}
    per_case = []
    for record in records:
        effects = {label: abs(float(record[field])) for label, field in labels.items()}
        ordered = sorted(effects.items(), key=lambda item: item[1], reverse=True)
        winner, maximum = ordered[0]
        runner_up = ordered[1][1]
        noise = abs(float(record.get("noise_floor_log10", 0.0)))
        supported = maximum > runner_up + noise
        if supported:
            support[winner] += 1
        per_case.append(
            {
                "case_id": str(record["case_id"]),
                "winner": winner if supported else "mixed_or_inconclusive",
                "effects": effects,
            }
        )
    classification = "mixed_or_inconclusive"
    supporting_case_count = 0
    for label, count in support.items():
        if count >= 3:
            classification = label
            supporting_case_count = count
            break
    return {
        "classification": classification,
        "supporting_case_count": supporting_case_count,
        "required_supporting_case_count": 3,
        "support_counts": support,
        "per_case": per_case,
    }


def load_block_payload(path: Path) -> dict[str, Any]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    result = {
        "A_R": csc_matrix(payload["A_R"], dtype=np.float64),
        "C": csc_matrix(payload["C"], dtype=np.float64),
        "D": csc_matrix(payload["D"], dtype=np.float64),
        "currents": np.asarray(payload["currents"], dtype=np.float64),
    }
    result["sha256"] = block_payload_sha256(
        result["A_R"],
        result["C"],
        result["D"],
        result["currents"],
    )
    return result


def solve_scipy_blocks(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    classic_state = _classic_state(payload["A_R"], payload["C"], payload["D"])
    _, classic_voltage = _solve_classic(classic_state, payload["currents"])
    robin_state = _robin_state(payload["A_R"], payload["C"], payload["D"])
    _, robin_voltage = _solve_robin(robin_state, payload["currents"])
    return {
        "classic": np.asarray(classic_voltage, dtype=np.float64),
        "robin_transconductance": np.asarray(robin_voltage, dtype=np.float64),
    }


def _exact_local_contributions(case: Any) -> dict[str, list[tuple[int, int, float]]]:
    nodes, cells, edges, _, _ = extension_case_mesh(case)
    cell_sigma = extension_case_cell_conductivities(case, nodes, cells)
    a_r: list[tuple[int, int, float]] = []
    coupling: list[tuple[int, int, float]] = []
    electrode: list[tuple[int, int, float]] = []
    for triangle, conductivity in zip(cells, cell_sigma, strict=True):
        indices = [int(value) for value in triangle]
        x1, y1 = nodes[indices[0]]
        x2, y2 = nodes[indices[1]]
        x3, y3 = nodes[indices[2]]
        determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = abs(determinant) / 2
        b = (y2 - y3, y3 - y1, y1 - y2)
        c = (x3 - x2, x1 - x3, x2 - x1)
        for local_row, global_row in enumerate(indices):
            for local_column, global_column in enumerate(indices):
                value = (
                    conductivity
                    * (b[local_row] * b[local_column] + c[local_row] * c[local_column])
                    / (4 * area)
                )
                a_r.append((global_row, global_column, float(value)))
    for vertex_a, vertex_b, label in np.asarray(edges, dtype=np.int64):
        if int(label) <= 0:
            continue
        a = int(vertex_a)
        b_index = int(vertex_b)
        dx = nodes[a][0] - nodes[b_index][0]
        dy = nodes[a][1] - nodes[b_index][1]
        length_over_z = _sqrt_fraction_exact(dx * dx + dy * dy) / case.contact_impedance
        diagonal = float(length_over_z / 3)
        off_diagonal = float(length_over_z / 6)
        half = float(length_over_z / 2)
        total = float(length_over_z)
        electrode_index = int(label) - 1
        a_r.extend(
            (
                (a, a, diagonal),
                (b_index, b_index, diagonal),
                (a, b_index, off_diagonal),
                (b_index, a, off_diagonal),
            )
        )
        coupling.extend(
            (
                (a, electrode_index, -half),
                (b_index, electrode_index, -half),
            )
        )
        electrode.append((electrode_index, electrode_index, total))
    return {"A_R": a_r, "C": coupling, "D": electrode}


def controlled_order_payloads(case: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    nodes, _, _, _, _ = extension_case_mesh(case)
    contributions = _exact_local_contributions(case)
    currents = np.zeros((case.n_electrodes, case.n_electrodes), dtype=np.float64)
    for column in range(case.n_electrodes):
        currents[column, column] = 1.0
        currents[(column + case.drive_skip) % case.n_electrodes, column] = -1.0
    shapes = {
        "A_R": (len(nodes), len(nodes)),
        "C": (len(nodes), case.n_electrodes),
        "D": (case.n_electrodes, case.n_electrodes),
    }
    payloads = []
    for reverse in (False, True):
        matrices = {}
        for name in ("A_R", "C", "D"):
            terms = contributions[name]
            order = reversed(terms) if reverse else terms
            matrices[name] = csc_matrix(_accumulate_contributions(order, shapes[name]))
        matrices["currents"] = currents
        matrices["sha256"] = block_payload_sha256(
            matrices["A_R"],
            matrices["C"],
            matrices["D"],
            currents,
        )
        payloads.append(matrices)
    return payloads[0], payloads[1]


def _case_directory(suite_output: Path, case_id: str) -> Path:
    case = next(case for case in EXTENSION_CASES if case.case_id == case_id)
    return suite_output / "cases" / f"{case.case_id}_{case.label}"


def prepare_backend_cross_manifest(suite_output: Path) -> dict[str, Any]:
    records = []
    filenames = {
        "pyeidors": "pyeidors_assembled_blocks.mat",
        "eidors": "eidors_assembled_blocks.mat",
        "ngsolve": "ngsolve_assembled_blocks.mat",
    }
    for case_id in ATTRIBUTION_CASE_IDS:
        case_dir = _case_directory(suite_output, case_id)
        for assembly, filename in filenames.items():
            block_path = case_dir / filename
            payload = load_block_payload(block_path)
            records.append(
                {
                    "case_id": case_id,
                    "assembly": assembly,
                    "block_path": str(block_path.resolve()),
                    "block_sha256": payload["sha256"],
                }
            )
    manifest = {
        "schema": ATTRIBUTION_SCHEMA,
        "case_ids": list(ATTRIBUTION_CASE_IDS),
        "assemblies": list(ASSEMBLIES),
        "records": records,
        "matlab_output": str((suite_output / "matlab_backend_cross.json").resolve()),
    }
    write_json(suite_output / "backend_cross_manifest.json", manifest)
    return manifest


def _cross_metric_record(
    *,
    case: Any,
    assembly: str,
    backend: str,
    block_sha256: str,
    formulation: str,
    voltage: np.ndarray,
    reference: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "assembly": assembly,
        "backend": backend,
        "block_sha256": block_sha256,
        "formulation": formulation,
        **exact_extension_accuracy_metrics(voltage, reference),
    }


def _effect_summary(
    cross_metrics: list[dict[str, Any]],
    order_metrics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_effects = []
    for case_id in ATTRIBUTION_CASE_IDS:
        formulation_effects = {}
        for formulation in FORMULATIONS:
            selected = [
                item
                for item in cross_metrics
                if item["case_id"] == case_id and item["formulation"] == formulation
            ]
            log_error = {
                (item["assembly"], item["backend"]): math.log10(
                    item["truth_relative_l2"]
                )
                for item in selected
            }
            assembly_effect = float(
                np.mean(
                    [
                        abs(
                            log_error[("pyeidors", backend)]
                            - log_error[("eidors", backend)]
                        )
                        for backend in BACKENDS
                    ]
                )
            )
            all_three_assembly_range = float(
                np.mean(
                    [
                        max(log_error[(assembly, backend)] for assembly in ASSEMBLIES)
                        - min(log_error[(assembly, backend)] for assembly in ASSEMBLIES)
                        for backend in BACKENDS
                    ]
                )
            )
            backend_effect = float(
                np.mean(
                    [
                        abs(
                            log_error[(assembly, "scipy_superlu")]
                            - log_error[(assembly, "matlab_sparse_lu")]
                        )
                        for assembly in ("pyeidors", "eidors")
                    ]
                )
            )
            order_pair = {
                item["order"]: math.log10(item["truth_relative_l2"])
                for item in order_metrics
                if item["case_id"] == case_id and item["formulation"] == formulation
            }
            order_effect = abs(order_pair["forward"] - order_pair["reverse"])
            formulation_effects[formulation] = {
                "assembly_effect_log10": assembly_effect,
                "all_three_assembly_range_log10": all_three_assembly_range,
                "backend_effect_log10": backend_effect,
                "order_effect_log10": order_effect,
            }
        case_effects.append(
            {
                "case_id": case_id,
                "assembly_effect_log10": float(
                    np.mean(
                        [
                            formulation_effects[name]["assembly_effect_log10"]
                            for name in FORMULATIONS
                        ]
                    )
                ),
                "backend_effect_log10": float(
                    np.mean(
                        [
                            formulation_effects[name]["backend_effect_log10"]
                            for name in FORMULATIONS
                        ]
                    )
                ),
                "order_effect_log10": float(
                    np.mean(
                        [
                            formulation_effects[name]["order_effect_log10"]
                            for name in FORMULATIONS
                        ]
                    )
                ),
                "noise_floor_log10": 0.05,
                "all_three_assembly_range_log10": float(
                    np.mean(
                        [
                            formulation_effects[name]["all_three_assembly_range_log10"]
                            for name in FORMULATIONS
                        ]
                    )
                ),
                "by_formulation": formulation_effects,
            }
        )
    return case_effects, classify_attribution(case_effects)


def _write_cross_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "assembly",
        "backend",
        "block_sha256",
        "formulation",
        "truth_relative_l2",
        "exact_reduced_scaled_backward_residual",
        "voltage_gauge_relative_residual",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: record[field] for field in fields} for record in records
        )


def _plot_effects(case_effects: list[dict[str, Any]], path: Path) -> None:
    configure_fonts()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        }
    )
    labels = [item["case_id"] for item in case_effects]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for offset, (field, label, color) in enumerate(
        (
            ("assembly_effect_log10", "Assembly implementation", "#1f5a94"),
            ("backend_effect_log10", "Linear backend", "#c56a1a"),
            ("order_effect_log10", "Pure accumulation order", "#687a3c"),
        )
    ):
        axis.bar(
            x + (offset - 1) * width,
            [item[field] for item in case_effects],
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Paired effect on log10 exact truth error (decades)")
    axis.set_title("Low-contact-impedance CEM error attribution")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def compare_backend_cross(suite_output: Path) -> dict[str, Any]:
    manifest = json.loads(
        (suite_output / "backend_cross_manifest.json").read_text(encoding="utf-8")
    )
    matlab = json.loads(
        (suite_output / "matlab_backend_cross.json").read_text(encoding="utf-8")
    )
    matlab_lookup = {
        (str(item["case_id"]), str(item["assembly"])): item
        for item in matlab["records"]
    }
    cross_metrics: list[dict[str, Any]] = []
    order_metrics: list[dict[str, Any]] = []
    for case_id in ATTRIBUTION_CASE_IDS:
        case = next(case for case in EXTENSION_CASES if case.case_id == case_id)
        reference = solve_exact_extension_case(case)
        for entry in (
            item for item in manifest["records"] if item["case_id"] == case_id
        ):
            payload = load_block_payload(Path(entry["block_path"]))
            if payload["sha256"] != entry["block_sha256"]:
                raise RuntimeError(f"{case_id}/{entry['assembly']} block hash changed")
            scipy_voltages = solve_scipy_blocks(payload)
            matlab_record = matlab_lookup[(case_id, str(entry["assembly"]))]
            if str(matlab_record["block_sha256"]) != payload["sha256"]:
                raise RuntimeError(
                    f"{case_id}/{entry['assembly']} MATLAB hash mismatch"
                )
            for formulation in FORMULATIONS:
                cross_metrics.append(
                    _cross_metric_record(
                        case=case,
                        assembly=str(entry["assembly"]),
                        backend="scipy_superlu",
                        block_sha256=payload["sha256"],
                        formulation=formulation,
                        voltage=np.asarray(
                            scipy_voltages[formulation], dtype=np.float64
                        ),
                        reference=reference,
                    )
                )
                cross_metrics.append(
                    _cross_metric_record(
                        case=case,
                        assembly=str(entry["assembly"]),
                        backend="matlab_sparse_lu",
                        block_sha256=payload["sha256"],
                        formulation=formulation,
                        voltage=np.asarray(
                            matlab_record["raw_electrode_voltages"][formulation],
                            dtype=np.float64,
                        ),
                        reference=reference,
                    )
                )
        forward, reverse = controlled_order_payloads(case)
        for order, payload in (("forward", forward), ("reverse", reverse)):
            voltages = solve_scipy_blocks(payload)
            for formulation in FORMULATIONS:
                order_metrics.append(
                    {
                        "case_id": case_id,
                        "order": order,
                        "formulation": formulation,
                        "block_sha256": payload["sha256"],
                        **exact_extension_accuracy_metrics(
                            voltages[formulation],
                            reference,
                        ),
                    }
                )
    case_effects, conclusion = _effect_summary(cross_metrics, order_metrics)
    csv_path = suite_output / "cem_low_z_backend_cross.csv"
    plot_path = suite_output / "cem_low_z_attribution.png"
    _write_cross_csv(csv_path, cross_metrics)
    _plot_effects(case_effects, plot_path)
    report = {
        "schema": ATTRIBUTION_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_ids": list(ATTRIBUTION_CASE_IDS),
        "assemblies": list(ASSEMBLIES),
        "backends": list(BACKENDS),
        "effect_definition": {
            "assembly": "mean backend/formulation absolute PyEIDORS-assembly vs EIDORS-assembly log10 truth-error difference",
            "all_three_assembly_range": "secondary mean backend/formulation range across PyEIDORS, EIDORS and NGSolve assemblies",
            "backend": "mean PyEIDORS/EIDORS assembly and formulation absolute SciPy-vs-MATLAB log10 error difference",
            "order": "mean formulation absolute forward-vs-reverse log10 error difference",
            "dominance_margin_decades": 0.05,
            "required_cases": 3,
        },
        "cross_metrics": cross_metrics,
        "controlled_order_metrics": order_metrics,
        "case_effects": case_effects,
        "conclusion": conclusion,
        "artifacts": {"csv": csv_path.name, "plot": plot_path.name},
    }
    json.dumps(report, allow_nan=False)
    write_json(suite_output / "cem_low_z_attribution.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "compare"))
    parser.add_argument(
        "--suite-output",
        type=Path,
        default=ROOT / "output" / "cem_exact_extension",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite_output = args.suite_output.resolve()
    if args.command == "prepare":
        manifest = prepare_backend_cross_manifest(suite_output)
        print(f"Prepared {len(manifest['records'])} assembly/backend records")
        return 0
    report = compare_backend_cross(suite_output)
    print(
        "Low-z attribution: "
        f"{report['conclusion']['classification']} "
        f"({report['conclusion']['supporting_case_count']}/4)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
