from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from scripts.benchmarks.cem_low_z_attribution import (
    ATTRIBUTION_CASE_IDS,
    ASSEMBLIES,
    BACKENDS,
    FORMULATIONS,
    _effect_summary,
    assembly_order_sensitivity,
    block_payload_sha256,
    classify_attribution,
)


def test_v729_attribution_case_ids_match_preregistration() -> None:
    assert ATTRIBUTION_CASE_IDS == ("X05", "X13", "X33", "X21")


def test_v729_block_digest_is_sparse_format_independent_and_value_sensitive() -> None:
    a_r = csc_matrix(np.asarray([[2.0, -1.0], [-1.0, 2.0]]))
    coupling = csc_matrix(np.asarray([[-1.0, 0.0], [0.0, -1.0]]))
    electrode = csc_matrix(np.eye(2))
    currents = np.asarray([[1.0, -1.0], [-1.0, 1.0]])
    digest = block_payload_sha256(a_r, coupling, electrode, currents)
    assert digest == block_payload_sha256(
        a_r.tocsr(), coupling.tocsr(), electrode.tocsr(), currents.copy()
    )
    changed = a_r.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    assert digest != block_payload_sha256(changed, coupling, electrode, currents)


def test_v729_controlled_order_probe_measures_forward_reverse_accumulation() -> None:
    contributions = [
        (0, 0, 1.0),
        (0, 0, 2.0**-53),
        (0, 0, -1.0),
        (1, 1, 0.25),
    ]
    result = assembly_order_sensitivity(contributions, shape=(2, 2))
    assert result["forward_reverse_max_abs"] > 0.0
    assert result["forward_sha256"] != result["reverse_sha256"]


def test_v730_attribution_requires_three_of_four_consistent_cases() -> None:
    records = [
        {
            "case_id": case_id,
            "assembly_effect_log10": 0.8,
            "backend_effect_log10": 0.1,
            "order_effect_log10": 0.05,
            "noise_floor_log10": 0.01,
        }
        for case_id in ATTRIBUTION_CASE_IDS[:3]
    ]
    records.append(
        {
            "case_id": ATTRIBUTION_CASE_IDS[3],
            "assembly_effect_log10": 0.1,
            "backend_effect_log10": 0.7,
            "order_effect_log10": 0.05,
            "noise_floor_log10": 0.01,
        }
    )
    conclusion = classify_attribution(records)
    assert conclusion["classification"] == "assembly_implementation_dominant"
    assert conclusion["supporting_case_count"] == 3

    tied = [
        dict(record, assembly_effect_log10=0.2, backend_effect_log10=0.2)
        for record in records
    ]
    assert classify_attribution(tied)["classification"] == "mixed_or_inconclusive"


def test_v729_primary_assembly_effect_is_not_inflated_by_ngsolve_outlier() -> None:
    cross_metrics = []
    order_metrics = []
    error_by_assembly = {
        "pyeidors": 1.0e-15,
        "eidors": 2.0e-15,
        "ngsolve": 1.0e-9,
    }
    for case_id in ATTRIBUTION_CASE_IDS:
        for formulation in FORMULATIONS:
            for assembly in ASSEMBLIES:
                for backend in BACKENDS:
                    cross_metrics.append(
                        {
                            "case_id": case_id,
                            "formulation": formulation,
                            "assembly": assembly,
                            "backend": backend,
                            "truth_relative_l2": error_by_assembly[assembly],
                        }
                    )
            order_metrics.extend(
                (
                    {
                        "case_id": case_id,
                        "formulation": formulation,
                        "order": "forward",
                        "truth_relative_l2": 1.0e-15,
                    },
                    {
                        "case_id": case_id,
                        "formulation": formulation,
                        "order": "reverse",
                        "truth_relative_l2": 1.1e-15,
                    },
                )
            )

    case_effects, conclusion = _effect_summary(cross_metrics, order_metrics)

    assert conclusion["classification"] == "assembly_implementation_dominant"
    assert all(0.30 < item["assembly_effect_log10"] < 0.31 for item in case_effects)
    assert all(item["all_three_assembly_range_log10"] > 5.9 for item in case_effects)
