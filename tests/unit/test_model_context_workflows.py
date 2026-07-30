"""Bridge v3 simulation and dataset workflow binding gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eit_app.controllers.dataset_generator_controller import (
    DatasetGeneratorRequest,
    _DatasetGeneratorWorker,
)
from eit_app.controllers.forward_solver_controller import (
    ForwardSolverRequest,
    _bridge_model_context,
    _create_forward_system,
    _execute_forward_request_unlocked,
    _pattern_and_electrode_count,
    _resolve_forward_runtime,
    _setup_generated_forward_system,
)
from eit_app.models.forward_model_config import ForwardModelConfig
from eit_app.models.simulation_state import (
    DatasetGeneratorConfig,
    InhomogeneitySpec,
)
from pyeidors.interop import BridgeV3Package, ModelContextFactory, ModelRegistry
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


def _write_workflow_package(root: Path) -> BridgeV3Package:
    geometry = {
        "index_base": 1,
        "source_framework": "pyeidors",
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
        "nodes": np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        "elems": np.asarray([[1, 2, 3], [1, 3, 4]], dtype=np.int64),
        "boundary_facets": np.asarray(
            [[1, 2], [2, 3], [3, 4], [4, 1]],
            dtype=np.int64,
        ),
        "boundary_edges": np.asarray(
            [[1, 2], [2, 3], [3, 4], [4, 1]],
            dtype=np.int64,
        ),
        "electrode_nodes": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "electrode_node_counts": np.asarray([2, 2], dtype=np.int64),
        "electrode_model": ["cem_faces", "distributed_point"],
        "electrode_boundary_kind": ["exterior", "none"],
        "pem_node_weights": np.asarray([[0.0, 0.0], [0.25, 0.75]]),
        "cem_face_nodes": np.asarray([[1, 2]], dtype=np.int64),
        "cem_face_node_counts": np.asarray([2], dtype=np.int64),
        "cem_face_electrode": np.asarray([1], dtype=np.int64),
        "n_elec": 2,
        "background": 1.0,
        "truth_elem_data": np.asarray([1.25, 1.5]),
        "contact_impedance": np.asarray([0.02, np.nan]),
        "contact_impedance_present": np.asarray([True, False]),
        "contact_impedance_applicable": np.asarray([True, False]),
        "effective_gnd_node": 1,
        "normalize_measurements": False,
        "mesh_name": "workflow_mixed",
        "mesh_level": "unit",
        "scenario_name": "workflow",
    }
    protocol = {
        "stim_matrix": np.asarray([[1.0, -1.0]]),
        "stim_matrix_raw": np.asarray([[1.0, -1.0]]),
        "meas_matrices": np.asarray([[[1.0, -1.0]]]),
        "measurement_counts": np.asarray([1], dtype=np.int64),
        "stimulation_supported": True,
        "normalize_measurements": False,
    }
    fields = {
        "background": 1.0,
        "background_present": True,
        "background_elem_data": np.asarray([1.0, 2.0]),
        "target_elem_data": np.asarray([1.25, 1.5]),
    }
    return BridgeV3Package.write(
        root,
        model={
            "schema_version": 3,
            "name": "workflow model",
            "n_elec": 2,
            "dimension": 2,
            "potential_order": 1,
            "forward_ready": True,
        },
        geometry=geometry,
        protocol=protocol,
        fields=fields,
        capabilities={"forward_ready": True},
    )


def _bound_config(
    tmp_path: Path,
    monkeypatch,
    *,
    flows: tuple[str, ...],
    field_override_mode: str = "imported",
) -> tuple[ForwardModelConfig, ModelRegistry]:
    monkeypatch.setenv("PYEIDORS_DATA_ROOT", str(tmp_path / "data"))
    package = _write_workflow_package(tmp_path / "source")
    registry = ModelRegistry()
    registered = registry.register(package.root)
    for flow in flows:
        registry.bind(flow, registered.model_id)
    config = ForwardModelConfig.from_mapping(
        {
            "mesh_source": "interop",
            "mesh_dimension": 2,
            "n_elec": 2,
            "n_rings": 1,
            "electrode_model": "mixed",
            "measurement_protocol": "custom",
            "interop_semantics": {
                "model_id": registered.model_id,
                "forward_fingerprint": registered.forward_fingerprint,
                "protocol_layout_hash": registered.protocol_layout_hash,
                "protocol_physics_hash": registered.protocol_physics_hash,
                "field_override_mode": field_override_mode,
            },
        }
    )
    return config, registry


def test_v757_simulation_uses_bound_context_target_and_background(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, registry = _bound_config(
        tmp_path,
        monkeypatch,
        flows=("simulation",),
    )
    expected = ModelContextFactory(registry).for_flow("simulation")
    request = ForwardSolverRequest(
        mesh_dimension=2,
        n_electrodes=2,
        inhomogeneities=[
            InhomogeneitySpec(
                shape="rectangle",
                size_x=10.0,
                size_y=10.0,
                conductivity=99.0,
            )
        ],
        forward_model_config=config.to_mapping(),
    )

    result = _execute_forward_request_unlocked(request)

    np.testing.assert_allclose(
        result.ground_truth_conductivity,
        expected.target_local,
    )
    reference_system = expected.create_system(initialize_inverse=False)
    reference = reference_system.forward_solve(expected.background_local)
    np.testing.assert_allclose(
        result.homogeneous_voltages,
        np.asarray(reference.meas).reshape(-1),
    )
    assert result.forward_model_config["model_id"] == expected.registered.model_id
    assert result.forward_model_config["field_override_mode"] == "imported"


def test_v757_managed_context_is_the_only_interop_system_setup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, _registry = _bound_config(
        tmp_path,
        monkeypatch,
        flows=("simulation",),
    )
    pattern, total_electrodes = _pattern_and_electrode_count(config)
    runtime = _resolve_forward_runtime(config)

    system = _create_forward_system(
        forward_cfg=config,
        runtime=runtime,
        pattern=pattern,
        total_electrodes=total_electrodes,
    )
    _setup_generated_forward_system(
        system,
        forward_cfg=config,
        runtime=runtime,
    )

    assert _bridge_model_context(system) is not None
    assert system.mesh is _bridge_model_context(system).mesh


def test_v757_dataset_defaults_to_imported_reference_and_truth(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, registry = _bound_config(
        tmp_path,
        monkeypatch,
        flows=("dataset",),
    )
    expected = ModelContextFactory(registry).for_flow("dataset")
    output_dir = tmp_path / "dataset"
    request = DatasetGeneratorRequest(
        config=DatasetGeneratorConfig(
            n_samples=1,
            output_dir=str(output_dir),
            n_inhomogeneities_min=3,
            n_inhomogeneities_max=3,
            mesh_dimension=2,
            n_electrodes=2,
        ),
        forward_model_config=config.to_mapping(),
    )

    _DatasetGeneratorWorker(request).run()

    package = read_hdf5_artifact(output_dir / "sample_000000.h5")
    np.testing.assert_allclose(
        package.arrays["ground_truth"],
        expected.target_local,
    )
    np.testing.assert_allclose(
        package.arrays["background_conductivity"],
        expected.background_local,
    )
    assert int(package.arrays["n_inhomogeneities"]) == 0
