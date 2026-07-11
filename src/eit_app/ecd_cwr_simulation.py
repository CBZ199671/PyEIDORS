"""ECD-CWR CEM simulation request adapter for the backend worker."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import h5py
import numpy as np


REQUEST_SCHEMA = "ecd-cwr-cem-backend-request-v1"
RESULT_SCHEMA = "ecd-cwr-cem-simulation-result-v1"
COMMAND = "ecd_cwr_simulate_cem"
OUTPUT_LAYOUT = "row_major_16x16_256_complex"
BASE_CONTACT_IMPEDANCE = 0.01
OPEN_CONTACT_IMPEDANCE = 1.0e6
GLOBAL_OPEN_CONTACT_IMPEDANCE = 1.0e3


def _resolve_path(path: str | Path) -> Path:
    text = str(path)
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", text)
    if match:
        drive = match.group(1).lower()
        tail = match.group(2).replace("\\", "/")
        return Path("/mnt") / drive / tail
    return Path(text).expanduser()


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text in {"-inf", "-infinity"}:
        return -math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _stable_seed(scenario_id: str) -> int:
    acc = 2166136261
    for byte in str(scenario_id).encode("utf-8"):
        acc ^= byte
        acc = (acc * 16777619) & 0xFFFFFFFF
    return int(acc)


def _target_conductivity(pattern: str, index: int) -> float:
    token = str(pattern or "high").strip().lower()
    if token == "low":
        return 0.5
    if token == "mixed":
        return 2.0 if index % 2 == 0 else 0.5
    return 2.0


def _target_center(
    placement: str,
    *,
    index: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    token = str(placement or "center").strip().lower()
    if token == "boundary":
        angle = 2.0 * math.pi * index / max(count, 1)
        return 0.72 * math.cos(angle), 0.72 * math.sin(angle)
    if token == "random":
        radius = 0.75 * math.sqrt(float(rng.random()))
        angle = 2.0 * math.pi * float(rng.random())
        return radius * math.cos(angle), radius * math.sin(angle)

    if count <= 1:
        return 0.0, 0.0
    angle = 2.0 * math.pi * index / count
    return 0.22 * math.cos(angle), 0.22 * math.sin(angle)


def _inhomogeneities_from_scenario(scenario: dict[str, Any]) -> list[Any]:
    from eit_app.models.simulation_state import InhomogeneitySpec

    count = max(0, int(scenario.get("target_count", 0)))
    placement = str(scenario.get("target_placement", "center"))
    conductivity_pattern = str(scenario.get("conductivity_pattern", "high"))
    rng = np.random.default_rng(_stable_seed(str(scenario.get("scenario_id", ""))))
    targets: list[Any] = []
    for idx in range(count):
        cx, cy = _target_center(placement, index=idx, count=count, rng=rng)
        targets.append(
            InhomogeneitySpec(
                shape="circle",
                center_x=float(cx),
                center_y=float(cy),
                center_z=0.0,
                size_x=0.12,
                size_y=0.12,
                size_z=0.12,
                conductivity=_target_conductivity(conductivity_pattern, idx),
            )
        )
    return targets


def _fault_contact_impedance_vector(
    scenario: dict[str, Any], n_electrodes: int
) -> np.ndarray:
    z = np.full(max(1, int(n_electrodes)), BASE_CONTACT_IMPEDANCE, dtype=np.float64)
    fault_mode = str(scenario.get("fault_mode", "none")).strip().lower()
    zc = dict(scenario.get("contact_impedance") or {})
    multiplier = _coerce_float(zc.get("multiplier", 1.0), 1.0)
    if math.isinf(multiplier):
        value = (
            GLOBAL_OPEN_CONTACT_IMPEDANCE
            if fault_mode == "global"
            else OPEN_CONTACT_IMPEDANCE
        )
    else:
        value = BASE_CONTACT_IMPEDANCE * max(float(multiplier), 0.0)

    if fault_mode in {"", "none"}:
        return z
    if fault_mode == "global":
        z[:] = value
        return z

    for electrode in list(scenario.get("fault_electrodes") or []):
        idx = int(electrode) % z.size
        z[idx] = value
    return z


def _noise_std_from_snr(signal: np.ndarray, snr_db: float) -> float:
    if not math.isfinite(float(snr_db)):
        return 0.0
    arr = np.asarray(signal)
    power = float(np.mean(np.abs(arr) ** 2))
    if power <= 0.0:
        return 0.0
    return math.sqrt(power / (10.0 ** (float(snr_db) / 10.0)))


def _add_complex_noise(
    values: np.ndarray,
    *,
    snr_db: float,
    seed: int,
) -> np.ndarray:
    std = _noise_std_from_snr(values, snr_db)
    if std <= 0.0:
        return np.asarray(values)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        noise = rng.normal(0.0, std / math.sqrt(2.0), arr.shape) + 1j * rng.normal(
            0.0, std / math.sqrt(2.0), arr.shape
        )
        return arr + noise.astype(np.result_type(arr.dtype, np.complex64))
    return arr + rng.normal(0.0, std, arr.shape).astype(arr.dtype, copy=False)


def _retained_indices_208() -> np.ndarray:
    return np.asarray(
        [stim * 16 + k for stim in range(16) for k in range(2, 15)],
        dtype=np.int32,
    )


def _ensure_full_observation(values: Any, expected_count: int = 256) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size != expected_count:
        raise ValueError(
            "ECD-CWR CEM simulation requires full 16x16 observations; "
            f"expected {expected_count}, got {arr.size}."
        )
    return np.asarray(
        arr.reshape(16, 16), dtype=np.result_type(arr.dtype, np.complex64)
    )


def build_forward_request(
    payload: dict[str, Any],
    *,
    reference: bool = False,
    contact_impedance_override: np.ndarray | None = None,
    frequency_hz: float | None = None,
) -> Any:
    from eit_app.controllers.forward_solver_controller import ForwardSolverRequest

    if str(payload.get("schema_version", "")) != REQUEST_SCHEMA:
        raise ValueError(
            f"unsupported ECD-CWR request schema: {payload.get('schema_version')!r}"
        )
    if str(payload.get("command", "")) != COMMAND:
        raise ValueError(f"unsupported ECD-CWR command: {payload.get('command')!r}")

    model = dict(payload.get("model") or {})
    scenario = dict(payload.get("scenario") or {})
    if reference:
        scenario = {
            **scenario,
            "target_count": 0,
            "fault_mode": "none",
            "fault_electrodes": [],
            "contact_impedance": {"label": "zc_x1", "multiplier": 1.0},
        }
    n_electrodes = int(model.get("electrode_count", 16))
    contact_impedance = (
        np.asarray(contact_impedance_override, dtype=np.float64)
        if contact_impedance_override is not None
        else _fault_contact_impedance_vector(scenario, n_electrodes)
    )
    if contact_impedance.size != n_electrodes:
        raise ValueError(
            "contact_impedance_override length must match electrode_count; "
            f"expected {n_electrodes}, got {contact_impedance.size}."
        )

    forward_cfg = {
        "mesh_dimension": 2,
        "mesh_refinement": 0.12,
        "n_elec": n_electrodes,
        "n_rings": 1,
        "stim_pattern": str(model.get("stim_pattern", "{ad}")),
        "meas_pattern": str(model.get("measurement_pattern", "{ad}")),
        "measurement_protocol": "eidors_full_3d",
        "rotate_meas": bool(model.get("rotate_measurements", True)),
        "use_meas_current": True,
        "contact_impedance": contact_impedance.tolist(),
        "background_conductivity": 1.0,
        "noise_level": 0.0,
        "simulation_schema": REQUEST_SCHEMA,
        "simulation_scenario_id": str(
            payload.get("scenario_id", scenario.get("scenario_id", ""))
        ),
        "simulation_output_layout": OUTPUT_LAYOUT,
        "simulation_reference_frame": bool(reference),
    }
    if frequency_hz is not None:
        forward_cfg["simulation_frequency_hz"] = float(frequency_hz)
    return ForwardSolverRequest(
        mesh_dimension=2,
        mesh_refinement=0.12,
        n_electrodes=n_electrodes,
        background_conductivity=1.0,
        inhomogeneities=[] if reference else _inhomogeneities_from_scenario(scenario),
        noise_level=0.0,
        forward_model_config=forward_cfg,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_result_hdf5(
    path: Path,
    *,
    payload: dict[str, Any],
    full_observation: np.ndarray,
    reference_observation: np.ndarray,
    retained_indices: np.ndarray,
    contact_impedance: np.ndarray,
    ground_truth_conductivity: np.ndarray,
    node_coords: np.ndarray,
    cell_connectivity: np.ndarray,
    forward_metadata: dict[str, Any],
    contact_jacobian_208x16: np.ndarray | None = None,
    contact_jacobian_step: float | None = None,
    frequency_hz: list[float] | None = None,
    frequency_full_observations: np.ndarray | None = None,
    frequency_reference_observations: np.ndarray | None = None,
    frequency_contact_impedance: np.ndarray | None = None,
    frequency_reference_contact_impedance: np.ndarray | None = None,
    frequency_contact_impedance_multipliers: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    retained = full_observation.reshape(-1)[retained_indices]
    reference_retained = reference_observation.reshape(-1)[retained_indices]
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = RESULT_SCHEMA
        handle.attrs["scenario_id"] = str(payload.get("scenario_id", ""))
        handle.attrs["layout"] = OUTPUT_LAYOUT
        handle.attrs["metadata_json"] = json.dumps(
            _json_ready(
                {
                    "request_schema": payload.get("schema_version"),
                    "command": payload.get("command"),
                    "model": payload.get("model"),
                    "scenario": payload.get("scenario"),
                    "forward_model_config": forward_metadata,
                }
            ),
            sort_keys=True,
        )
        handle.create_dataset("raw_complex_256", data=full_observation)
        handle.create_dataset("reference_complex_256", data=reference_observation)
        handle.create_dataset("retained_complex_208", data=retained)
        handle.create_dataset("reference_retained_complex_208", data=reference_retained)
        handle.create_dataset("retained_indices_208", data=retained_indices)
        handle.create_dataset("contact_impedance", data=contact_impedance)
        if frequency_hz:
            frequency_full = np.asarray(frequency_full_observations)
            frequency_reference = np.asarray(frequency_reference_observations)
            if frequency_full.shape != (len(frequency_hz), 16, 16):
                raise ValueError(
                    "frequency_full_observations must have shape "
                    f"({len(frequency_hz)}, 16, 16); got {frequency_full.shape}."
                )
            if frequency_reference.shape != (len(frequency_hz), 16, 16):
                raise ValueError(
                    "frequency_reference_observations must have shape "
                    f"({len(frequency_hz)}, 16, 16); got {frequency_reference.shape}."
                )
            handle.attrs["frequency_count"] = int(len(frequency_hz))
            handle.create_dataset("frequency_hz", data=np.asarray(frequency_hz))
            handle.create_dataset("frequency_raw_complex_256", data=frequency_full)
            handle.create_dataset(
                "frequency_reference_complex_256",
                data=frequency_reference,
            )
            handle.create_dataset(
                "frequency_retained_complex_208",
                data=frequency_full.reshape(len(frequency_hz), -1)[:, retained_indices],
            )
            handle.create_dataset(
                "frequency_reference_retained_complex_208",
                data=frequency_reference.reshape(len(frequency_hz), -1)[
                    :, retained_indices
                ],
            )
            if frequency_contact_impedance_multipliers is not None:
                handle.create_dataset(
                    "frequency_contact_impedance_multipliers",
                    data=np.asarray(frequency_contact_impedance_multipliers),
                )
            if frequency_contact_impedance is not None:
                handle.create_dataset(
                    "frequency_contact_impedance_16",
                    data=np.asarray(frequency_contact_impedance),
                )
            if frequency_reference_contact_impedance is not None:
                handle.create_dataset(
                    "frequency_reference_contact_impedance_16",
                    data=np.asarray(frequency_reference_contact_impedance),
                )
        if contact_jacobian_208x16 is not None:
            handle.create_dataset(
                "contact_jacobian_208x16",
                data=np.asarray(contact_jacobian_208x16),
            )
            handle.attrs["contact_jacobian_step"] = float(contact_jacobian_step or 0.0)
        handle.create_dataset(
            "ground_truth_conductivity",
            data=np.asarray(ground_truth_conductivity),
        )
        handle.create_dataset("node_coords", data=np.asarray(node_coords))
        handle.create_dataset("cell_connectivity", data=np.asarray(cell_connectivity))


def _write_label_json(
    path: Path,
    *,
    payload: dict[str, Any],
    contact_impedance: np.ndarray,
    retained_indices: np.ndarray,
    frequency_hz: list[float] | None = None,
    frequency_contact_impedance_multipliers: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scenario = dict(payload.get("scenario") or {})
    label = {
        "schema_version": "ecd-cwr-cem-simulation-label-v1",
        "scenario_id": str(payload.get("scenario_id", scenario.get("scenario_id", ""))),
        "layout": OUTPUT_LAYOUT,
        "fault_mode": scenario.get("fault_mode", "none"),
        "fault_electrodes": list(scenario.get("fault_electrodes") or []),
        "contact_impedance": contact_impedance.tolist(),
        "retained_indices_208": retained_indices.tolist(),
        "frequency_hz": list(frequency_hz or []),
        "frequency_contact_impedance_multipliers": list(
            frequency_contact_impedance_multipliers or []
        ),
        "request": payload,
    }
    path.write_text(json.dumps(_json_ready(label), indent=2, sort_keys=True), "utf-8")


def _contact_jacobian_enabled(payload: dict[str, Any]) -> bool:
    model = dict(payload.get("model") or {})
    output = dict(payload.get("output") or {})
    value = model.get(
        "emit_contact_jacobian", output.get("emit_contact_jacobian", False)
    )
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _contact_jacobian_step(payload: dict[str, Any]) -> float:
    model = dict(payload.get("model") or {})
    raw = model.get("contact_jacobian_step", BASE_CONTACT_IMPEDANCE * 1.0e-3)
    step = _coerce_float(raw, BASE_CONTACT_IMPEDANCE * 1.0e-3)
    return max(float(step), BASE_CONTACT_IMPEDANCE * 1.0e-6)


def _frequency_hz_values(payload: dict[str, Any]) -> list[float]:
    model = dict(payload.get("model") or {})
    output = dict(payload.get("output") or {})
    raw = model.get("frequencies_hz", output.get("frequencies_hz", []))
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        raw_values = [raw]
    else:
        raw_values = list(raw)
    values = [float(_coerce_float(item, math.nan)) for item in raw_values]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("frequencies_hz values must be positive finite numbers.")
    return values


def _frequency_contact_impedance_multipliers(
    payload: dict[str, Any],
    frequency_count: int,
) -> list[float]:
    if frequency_count <= 0:
        return []
    model = dict(payload.get("model") or {})
    output = dict(payload.get("output") or {})
    raw = model.get(
        "frequency_contact_impedance_multipliers",
        output.get("frequency_contact_impedance_multipliers"),
    )
    if raw is None:
        raise ValueError(
            "frequency_contact_impedance_multipliers is required when "
            "frequencies_hz is provided."
        )
    if isinstance(raw, (str, bytes)):
        raw_values = [raw]
    else:
        raw_values = list(raw)
    if len(raw_values) != frequency_count:
        raise ValueError(
            "frequency_contact_impedance_multipliers length must match "
            f"frequencies_hz length; expected {frequency_count}, got {len(raw_values)}."
        )
    values = [float(_coerce_float(item, math.nan)) for item in raw_values]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(
            "frequency_contact_impedance_multipliers values must be finite and non-negative."
        )
    return values


def _compute_contact_jacobian_208x16(
    payload: dict[str, Any],
    *,
    base_contact_impedance: np.ndarray,
    reference_observation: np.ndarray,
    retained_indices: np.ndarray,
    step: float,
    runner: Callable[..., Any],
    progress_cb: Callable[[str], None] | None,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    base_retained = reference_observation.reshape(-1)[retained_indices]
    for electrode in range(base_contact_impedance.size):
        perturbed = np.asarray(base_contact_impedance, dtype=np.float64).copy()
        perturbed[electrode] += step
        request = build_forward_request(
            payload,
            reference=True,
            contact_impedance_override=perturbed,
        )
        if progress_cb is not None:
            progress_cb(
                f"Running ECD-CWR contact Jacobian perturbation {electrode + 1}/16..."
            )
        result = runner(request, progress_cb=progress_cb)
        observation = _ensure_full_observation(result.boundary_voltages)
        retained = observation.reshape(-1)[retained_indices]
        columns.append((retained - base_retained) / step)
    return np.stack(columns, axis=1)


def _compute_frequency_observations(
    payload: dict[str, Any],
    *,
    target_contact_impedance: np.ndarray,
    reference_contact_impedance: np.ndarray,
    snr_db: float,
    seed: int,
    runner: Callable[..., Any],
    progress_cb: Callable[[str], None] | None,
) -> tuple[
    list[float],
    list[float],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    frequency_hz = _frequency_hz_values(payload)
    if not frequency_hz:
        return [], [], None, None, None, None

    multipliers = _frequency_contact_impedance_multipliers(payload, len(frequency_hz))
    target_observations: list[np.ndarray] = []
    reference_observations: list[np.ndarray] = []
    target_contact_impedances: list[np.ndarray] = []
    reference_contact_impedances: list[np.ndarray] = []

    for index, (frequency, multiplier) in enumerate(
        zip(frequency_hz, multipliers, strict=True)
    ):
        target_zc = np.asarray(target_contact_impedance, dtype=np.float64) * multiplier
        reference_zc = (
            np.asarray(reference_contact_impedance, dtype=np.float64) * multiplier
        )
        if progress_cb is not None:
            progress_cb(
                "Running ECD-CWR multi-frequency target simulation "
                f"{index + 1}/{len(frequency_hz)} at {frequency:g} Hz..."
            )
        target_request = build_forward_request(
            payload,
            contact_impedance_override=target_zc,
            frequency_hz=frequency,
        )
        target_result = runner(target_request, progress_cb=progress_cb)
        target_observation = _ensure_full_observation(target_result.boundary_voltages)
        target_observation = _add_complex_noise(
            target_observation,
            snr_db=snr_db,
            seed=seed + 10007 * (index + 1),
        )

        if progress_cb is not None:
            progress_cb(
                "Running ECD-CWR multi-frequency reference simulation "
                f"{index + 1}/{len(frequency_hz)} at {frequency:g} Hz..."
            )
        reference_request = build_forward_request(
            payload,
            reference=True,
            contact_impedance_override=reference_zc,
            frequency_hz=frequency,
        )
        reference_result = runner(reference_request, progress_cb=progress_cb)
        reference_observation = _ensure_full_observation(
            reference_result.boundary_voltages
        )

        target_observations.append(target_observation)
        reference_observations.append(reference_observation)
        target_contact_impedances.append(target_zc)
        reference_contact_impedances.append(reference_zc)

    return (
        frequency_hz,
        multipliers,
        np.stack(target_observations, axis=0),
        np.stack(reference_observations, axis=0),
        np.stack(target_contact_impedances, axis=0),
        np.stack(reference_contact_impedances, axis=0),
    )


def run_ecd_cwr_simulation_request(
    payload: dict[str, Any],
    *,
    progress_cb: Callable[[str], None] | None = None,
    execute_forward: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    from eit_app.controllers.forward_solver_controller import execute_forward_request

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    request = build_forward_request(payload)
    reference_request = build_forward_request(payload, reference=True)
    output = dict(payload.get("output") or {})
    hdf5_path = _resolve_path(output.get("hdf5_path", "ecd-cwr-simulation.h5"))
    label_path = _resolve_path(
        output.get("label_json_path", "ecd-cwr-simulation.label.json")
    )

    emit("Running ECD-CWR CEM target forward simulation...")
    runner = execute_forward if execute_forward is not None else execute_forward_request
    result = runner(request, progress_cb=progress_cb)
    emit("Running ECD-CWR CEM healthy reference simulation...")
    reference_result = runner(reference_request, progress_cb=progress_cb)

    scenario = dict(payload.get("scenario") or {})
    snr_db = _coerce_float(
        dict(scenario.get("noise") or {}).get("snr_db", math.inf), math.inf
    )
    seed = _stable_seed(
        str(payload.get("scenario_id", scenario.get("scenario_id", "")))
    )
    full_observation = _ensure_full_observation(result.boundary_voltages)
    full_observation = _add_complex_noise(full_observation, snr_db=snr_db, seed=seed)
    reference_observation = _ensure_full_observation(reference_result.boundary_voltages)
    retained_indices = _retained_indices_208()
    contact_impedance = np.asarray(request.forward_model_config["contact_impedance"])
    reference_contact_impedance = np.asarray(
        reference_request.forward_model_config["contact_impedance"]
    )
    (
        frequency_hz,
        frequency_contact_impedance_multipliers,
        frequency_full_observations,
        frequency_reference_observations,
        frequency_contact_impedance,
        frequency_reference_contact_impedance,
    ) = _compute_frequency_observations(
        payload,
        target_contact_impedance=contact_impedance,
        reference_contact_impedance=reference_contact_impedance,
        snr_db=snr_db,
        seed=seed,
        runner=runner,
        progress_cb=progress_cb,
    )
    contact_jacobian = None
    jacobian_step = None
    if _contact_jacobian_enabled(payload):
        jacobian_step = _contact_jacobian_step(payload)
        contact_jacobian = _compute_contact_jacobian_208x16(
            payload,
            base_contact_impedance=np.asarray(
                reference_request.forward_model_config["contact_impedance"],
                dtype=np.float64,
            ),
            reference_observation=reference_observation,
            retained_indices=retained_indices,
            step=jacobian_step,
            runner=runner,
            progress_cb=progress_cb,
        )
    ground_truth_conductivity = np.asarray(
        getattr(result, "ground_truth_conductivity", np.array([]))
    )
    node_coords = np.asarray(getattr(result, "node_coords", np.empty((0, 0))))
    cell_connectivity = np.asarray(
        getattr(result, "cell_connectivity", np.empty((0, 0), dtype=np.int32))
    )

    _write_result_hdf5(
        hdf5_path,
        payload=payload,
        full_observation=full_observation,
        reference_observation=reference_observation,
        retained_indices=retained_indices,
        contact_impedance=contact_impedance,
        ground_truth_conductivity=ground_truth_conductivity,
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
        forward_metadata=dict(getattr(result, "forward_model_config", {}) or {}),
        contact_jacobian_208x16=contact_jacobian,
        contact_jacobian_step=jacobian_step,
        frequency_hz=frequency_hz,
        frequency_full_observations=frequency_full_observations,
        frequency_reference_observations=frequency_reference_observations,
        frequency_contact_impedance=frequency_contact_impedance,
        frequency_reference_contact_impedance=frequency_reference_contact_impedance,
        frequency_contact_impedance_multipliers=frequency_contact_impedance_multipliers,
    )
    _write_label_json(
        label_path,
        payload=payload,
        contact_impedance=contact_impedance,
        retained_indices=retained_indices,
        frequency_hz=frequency_hz,
        frequency_contact_impedance_multipliers=frequency_contact_impedance_multipliers,
    )
    emit("ECD-CWR CEM simulation complete.")
    return {
        "schema": RESULT_SCHEMA,
        "scenario_id": str(payload.get("scenario_id", "")),
        "hdf5_path": str(hdf5_path),
        "label_json_path": str(label_path),
        "layout": OUTPUT_LAYOUT,
        "full_observation_count": int(full_observation.size),
        "retained_observation_count": int(retained_indices.size),
        "frequency_count": int(len(frequency_hz)),
    }


def run_ecd_cwr_simulation_request_file(
    input_path: str | Path,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    payload = json.loads(_resolve_path(input_path).read_text("utf-8"))
    return run_ecd_cwr_simulation_request(payload, progress_cb=progress_cb)


__all__ = [
    "COMMAND",
    "OUTPUT_LAYOUT",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "build_forward_request",
    "run_ecd_cwr_simulation_request",
    "run_ecd_cwr_simulation_request_file",
]
