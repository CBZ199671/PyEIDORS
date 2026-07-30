"""两个演示共用的、可逐变量检查的经典/Robin CEM 代数。

Inspectable Classic/Robin CEM algebra shared by the walkthroughs.
中文：正式基准仍是权威记录；本模块把同一分块代数展开成具名中间变量，
便于在 VS Code 中检查每个矩阵、分解、右端项和解。
English: The production benchmark remains the source of record. This module
expands the same algebra into named intermediate objects for VS Code inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import csv
import json
import math
from pathlib import Path
from typing import Any

from mpmath import mp
import numpy as np
from scipy.io import loadmat
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import bmat, csc_matrix
from scipy.sparse.linalg import SuperLU, splu


FORMULATIONS = ("classic", "robin_transconductance")


@dataclass(frozen=True)
class CEMBlocks:
    """Preassembled CEM blocks and the applied electrode-current matrix."""

    robin_matrix: csc_matrix
    coupling: csc_matrix
    electrode_matrix: csc_matrix
    currents: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.robin_matrix.shape[0])

    @property
    def electrode_count(self) -> int:
        return int(self.electrode_matrix.shape[0])


@dataclass(frozen=True)
class ForwardFixture:
    """共享正问题输入及公平性元数据。 / Shared forward inputs and fairness metadata."""

    nodes: np.ndarray
    cells: np.ndarray
    tagged_edges: np.ndarray
    cell_conductivity: np.ndarray
    currents: np.ndarray
    electrode_count: int
    case_id: str
    mesh_fingerprint: str
    potential_order: int
    scalar_dtype: str
    conductivity_pattern: str
    contact_impedance_exact: str


@dataclass(frozen=True)
class ClassicDebugState:
    """Named objects created during the augmented Classic CEM setup."""

    system_matrix: csc_matrix
    factor: SuperLU
    node_count: int
    electrode_count: int


@dataclass(frozen=True)
class RobinDebugState:
    """Named objects created during the Robin/Schur-complement setup."""

    robin_matrix: csc_matrix
    body_factor: SuperLU
    electrode_basis: np.ndarray
    coupling_basis: np.ndarray
    response_basis: np.ndarray
    schur_action_basis: np.ndarray
    reduced_map: np.ndarray
    reduced_lu: np.ndarray
    reduced_pivots: np.ndarray


@dataclass(frozen=True)
class FormulationSolution:
    """Body and electrode potentials plus the formulation-specific state."""

    body_potential: np.ndarray
    electrode_voltage: np.ndarray
    state: ClassicDebugState | RobinDebugState
    lagrange_multiplier: np.ndarray | None = None


def _as_csc_float64(matrix: Any) -> csc_matrix:
    return csc_matrix(matrix, dtype=np.float64)


def load_assembled_blocks(path: Path) -> CEMBlocks:
    """Load the exact A_R/C/D/current payload written by any FEM runner."""

    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    robin_matrix = _as_csc_float64(payload["A_R"])
    coupling = _as_csc_float64(payload["C"])
    electrode_matrix = _as_csc_float64(payload["D"])
    currents = np.asarray(payload["currents"], dtype=np.float64)
    if currents.ndim == 1:
        currents = currents.reshape(-1, 1)
    expected = electrode_matrix.shape[0]
    if currents.shape[0] != expected:
        raise ValueError(
            f"current row count {currents.shape[0]} does not match "
            f"{expected} electrodes"
        )
    return CEMBlocks(
        robin_matrix=robin_matrix,
        coupling=coupling,
        electrode_matrix=electrode_matrix,
        currents=currents,
    )


def load_forward_fixture(
    mat_path: Path,
    metadata_path: Path,
) -> ForwardFixture:
    """加载共享网格、电导率和电流。 / Load the shared mesh, conductivity, and drives."""

    payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    cells = np.asarray(payload["elems"], dtype=np.int64)
    tagged_edges = np.asarray(payload["tagged_boundary_edges"], dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError(f"expected triangular cells, received {cells.shape}")
    if tagged_edges.ndim != 2 or tagged_edges.shape[1] != 3:
        raise ValueError(
            f"expected two edge vertices plus one label, received {tagged_edges.shape}"
        )
    if cells.size and int(cells.min()) >= 1:
        cells = cells - 1
    tagged_edges = tagged_edges.copy()
    if tagged_edges[:, :2].size and int(tagged_edges[:, :2].min()) >= 1:
        tagged_edges[:, :2] -= 1
    cell_conductivity = np.asarray(
        payload.get("truth_elem_data", payload["background"]),
        dtype=np.float64,
    ).reshape(-1)
    if cell_conductivity.size == 1:
        cell_conductivity = np.full(
            cells.shape[0],
            float(cell_conductivity[0]),
            dtype=np.float64,
        )
    if cell_conductivity.size != cells.shape[0]:
        raise ValueError(
            "cell conductivity count does not match the triangular-cell count"
        )
    currents = np.asarray(payload["current_patterns"], dtype=np.float64)
    if currents.ndim == 1:
        currents = currents.reshape(-1, 1)
    electrode_count = int(np.asarray(payload["n_elec"]).item())
    if currents.shape[0] != electrode_count:
        raise ValueError("current rows do not match the electrode count")
    return ForwardFixture(
        nodes=np.ascontiguousarray(nodes),
        cells=np.ascontiguousarray(cells),
        tagged_edges=np.ascontiguousarray(tagged_edges),
        cell_conductivity=np.ascontiguousarray(cell_conductivity),
        currents=np.ascontiguousarray(currents),
        electrode_count=electrode_count,
        case_id=str(metadata["case_id"]),
        mesh_fingerprint=str(metadata["mesh_fingerprint"]),
        potential_order=int(metadata["potential_order"]),
        scalar_dtype=str(metadata["scalar_dtype"]),
        conductivity_pattern=str(metadata["conductivity_pattern"]),
        contact_impedance_exact=str(metadata["contact_impedance_exact"]),
    )


def plot_forward_fixture(
    fixture: ForwardFixture,
    *,
    current_column: int = 0,
):
    """绘制公平正问题的网格/电导率/电流。 / Plot the fair forward setup."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.tri import Triangulation

    if not 0 <= current_column < fixture.currents.shape[1]:
        raise IndexError("current_column is outside the available drive patterns")
    triangulation = Triangulation(
        fixture.nodes[:, 0],
        fixture.nodes[:, 1],
        fixture.cells,
    )
    sigma_min = float(np.min(fixture.cell_conductivity))
    sigma_max = float(np.max(fixture.cell_conductivity))
    if sigma_max == sigma_min:
        normalization = Normalize(vmin=0.0, vmax=max(sigma_max, 1.0))
    else:
        normalization = Normalize(vmin=sigma_min, vmax=sigma_max)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        constrained_layout=True,
    )
    conductivity_plot = axes[0].tripcolor(
        triangulation,
        facecolors=fixture.cell_conductivity,
        cmap="cividis",
        norm=normalization,
        edgecolors="#D1D5DB",
        linewidth=0.75,
    )
    figure.colorbar(
        conductivity_plot,
        ax=axes[0],
        shrink=0.82,
        label=r"Cell conductivity $\sigma$",
    )
    axes[0].set_title("网格与单元电导率 / Mesh and cell conductivity")
    axes[1].triplot(
        triangulation,
        color="#D1D5DB",
        linewidth=0.75,
    )
    drive = fixture.currents[:, current_column]
    for vertex_a, vertex_b, label in fixture.tagged_edges:
        electrode = int(label) - 1
        if electrode < 0:
            continue
        x_values = fixture.nodes[[int(vertex_a), int(vertex_b)], 0]
        y_values = fixture.nodes[[int(vertex_a), int(vertex_b)], 1]
        midpoint = fixture.nodes[[int(vertex_a), int(vertex_b)]].mean(axis=0)
        axes[0].plot(x_values, y_values, color="#111827", linewidth=2.2)
        axes[0].text(
            midpoint[0] * 1.075,
            midpoint[1] * 1.075,
            str(electrode + 1),
            ha="center",
            va="center",
            fontsize=8,
            color="#111827",
        )
        current = float(drive[electrode])
        if current > 0:
            color = "#D97706"
            linewidth = 5.0
        elif current < 0:
            color = "#1D4ED8"
            linewidth = 5.0
        else:
            color = "#6B7280"
            linewidth = 2.0
        axes[1].plot(
            x_values,
            y_values,
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
        )
        axes[1].text(
            midpoint[0] * 1.075,
            midpoint[1] * 1.075,
            f"{electrode + 1}",
            ha="center",
            va="center",
            fontsize=8,
            color=color,
        )
    positive = [str(index + 1) for index, value in enumerate(drive) if value > 0]
    negative = [str(index + 1) for index, value in enumerate(drive) if value < 0]
    axes[1].set_title(
        f"边界注入电流：模式 {current_column + 1} / "
        f"Boundary drive pattern {current_column + 1}\n"
        f"+I: {', '.join(positive)}    −I: {', '.join(negative)}"
    )
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#D97706", linewidth=5, label="+I 注入 / inject"),
            Line2D([0], [0], color="#1D4ED8", linewidth=5, label="−I 回流 / return"),
            Line2D([0], [0], color="#6B7280", linewidth=2, label="0 未激励 / inactive"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
    )
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlim(-1.14, 1.14)
        axis.set_ylim(-1.14, 1.14)
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.grid(False)
    figure.suptitle(
        f"{fixture.case_id} 共享正问题 / Shared forward problem\n"
        f"Mesh fingerprint: {fixture.mesh_fingerprint[:16]}…  |  "
        f"N={fixture.nodes.shape[0]}, K={fixture.cells.shape[0]}, "
        f"L={fixture.electrode_count}, P={fixture.currents.shape[1]}"
    )
    return figure, axes


def plot_forward_solution(
    fixture: ForwardFixture,
    solutions: dict[str, Any],
    *,
    current_column: int = 0,
):
    """绘制 Classic/Robin 正问题解及差值。 / Plot both forward solutions and deltas."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.tri import Triangulation

    if not 0 <= current_column < fixture.currents.shape[1]:
        raise IndexError("current_column is outside the available drive patterns")
    try:
        classic = solutions["classic"]
        robin = solutions["robin_transconductance"]
    except KeyError as error:
        raise KeyError("solutions must contain Classic and Robin results") from error

    classic_body_all = np.asarray(classic.body_potential, dtype=np.float64)
    robin_body_all = np.asarray(robin.body_potential, dtype=np.float64)
    classic_voltage_all = np.asarray(classic.electrode_voltage, dtype=np.float64)
    robin_voltage_all = np.asarray(robin.electrode_voltage, dtype=np.float64)
    expected_body_shape = (
        fixture.nodes.shape[0],
        fixture.currents.shape[1],
    )
    expected_voltage_shape = (
        fixture.electrode_count,
        fixture.currents.shape[1],
    )
    if classic_body_all.shape != expected_body_shape:
        raise ValueError(
            f"Classic body potential has {classic_body_all.shape}, "
            f"expected {expected_body_shape}"
        )
    if robin_body_all.shape != expected_body_shape:
        raise ValueError(
            f"Robin body potential has {robin_body_all.shape}, "
            f"expected {expected_body_shape}"
        )
    if classic_voltage_all.shape != expected_voltage_shape:
        raise ValueError(
            f"Classic electrode voltage has {classic_voltage_all.shape}, "
            f"expected {expected_voltage_shape}"
        )
    if robin_voltage_all.shape != expected_voltage_shape:
        raise ValueError(
            f"Robin electrode voltage has {robin_voltage_all.shape}, "
            f"expected {expected_voltage_shape}"
        )

    classic_body = classic_body_all[:, current_column]
    robin_body = robin_body_all[:, current_column]
    body_delta = robin_body - classic_body
    classic_voltage = classic_voltage_all[:, current_column]
    robin_voltage = robin_voltage_all[:, current_column]
    voltage_delta = robin_voltage - classic_voltage
    triangulation = Triangulation(
        fixture.nodes[:, 0],
        fixture.nodes[:, 1],
        fixture.cells,
    )

    body_min = float(min(np.min(classic_body), np.min(robin_body)))
    body_max = float(max(np.max(classic_body), np.max(robin_body)))
    if body_min == body_max:
        body_padding = max(abs(body_min), 1.0) * 1e-12
        body_min -= body_padding
        body_max += body_padding
    body_norm = Normalize(vmin=body_min, vmax=body_max)
    body_delta_limit = max(
        float(np.max(np.abs(body_delta))),
        np.finfo(np.float64).eps,
    )
    body_delta_norm = Normalize(
        vmin=-body_delta_limit,
        vmax=body_delta_limit,
    )

    voltage_min = float(min(np.min(classic_voltage), np.min(robin_voltage)))
    voltage_max = float(max(np.max(classic_voltage), np.max(robin_voltage)))
    voltage_padding = max(voltage_max - voltage_min, 1.0) * 0.08
    shared_voltage_limits = (
        voltage_min - voltage_padding,
        voltage_max + voltage_padding,
    )
    voltage_delta_limit = max(
        float(np.max(np.abs(voltage_delta))),
        np.finfo(np.float64).eps,
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15.8, 9.4),
        constrained_layout=True,
    )
    body_maps = []
    for axis, values, title in (
        (axes[0, 0], classic_body, "Classic 体电势 / Classic body potential"),
        (axes[0, 1], robin_body, "Robin 体电势 / Robin body potential"),
    ):
        body_maps.append(
            axis.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap="cividis",
                norm=body_norm,
            )
        )
        axis.triplot(triangulation, color="#CBD5E1", linewidth=0.55, alpha=0.8)
        axis.set_title(title)
    body_delta_map = axes[0, 2].tripcolor(
        triangulation,
        body_delta,
        shading="gouraud",
        cmap="coolwarm",
        norm=body_delta_norm,
    )
    axes[0, 2].triplot(
        triangulation,
        color="#CBD5E1",
        linewidth=0.55,
        alpha=0.8,
    )
    axes[0, 2].set_title("体电势差值 Robin − Classic / Body-potential difference")
    for axis in axes[0, :]:
        for vertex_a, vertex_b, label in fixture.tagged_edges:
            if int(label) <= 0:
                continue
            axis.plot(
                fixture.nodes[[int(vertex_a), int(vertex_b)], 0],
                fixture.nodes[[int(vertex_a), int(vertex_b)], 1],
                color="#111827",
                linewidth=1.45,
            )
        axis.set_aspect("equal")
        axis.set_xlim(-1.08, 1.08)
        axis.set_ylim(-1.08, 1.08)
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.grid(False)
    figure.colorbar(
        body_maps[0],
        ax=axes[0, :2],
        shrink=0.78,
        label="Body potential",
    )
    figure.colorbar(
        body_delta_map,
        ax=axes[0, 2],
        shrink=0.78,
        label="Robin − Classic",
        format="%.1e",
    )

    electrode_indices = np.arange(1, fixture.electrode_count + 1)
    axes[1, 0].plot(
        electrode_indices,
        classic_voltage,
        color="#D97706",
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="Classic",
    )
    axes[1, 0].axhline(0.0, color="#94A3B8", linewidth=0.9)
    axes[1, 0].set_title("Classic 电极电压 / Classic electrode voltage")
    axes[1, 1].plot(
        electrode_indices,
        robin_voltage,
        color="#1D4ED8",
        marker="s",
        markerfacecolor="white",
        markersize=4.5,
        linewidth=1.8,
        linestyle="--",
        label="Robin",
    )
    axes[1, 1].axhline(0.0, color="#94A3B8", linewidth=0.9)
    axes[1, 1].set_title("Robin 电极电压 / Robin electrode voltage")
    axes[1, 2].plot(
        electrode_indices,
        voltage_delta,
        color="#7C3AED",
        marker="D",
        markerfacecolor="white",
        markersize=4.2,
        linewidth=1.6,
    )
    axes[1, 2].axhline(0.0, color="#475569", linewidth=1.0)
    axes[1, 2].set_ylim(
        -1.12 * voltage_delta_limit,
        1.12 * voltage_delta_limit,
    )
    axes[1, 2].set_title("电极电压差值 Robin − Classic / Electrode-voltage difference")
    for axis in axes[1, :2]:
        axis.set_ylim(*shared_voltage_limits)
    for axis in axes[1, :]:
        axis.set_xlabel("电极编号 / Electrode index")
        axis.set_ylabel("电压 / Voltage")
        axis.set_xticks(electrode_indices)
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)

    body_relative_delta = float(
        np.linalg.norm(body_delta)
        / max(np.linalg.norm(classic_body), np.finfo(np.float64).eps)
    )
    voltage_relative_delta = float(
        np.linalg.norm(voltage_delta)
        / max(np.linalg.norm(classic_voltage), np.finfo(np.float64).eps)
    )
    drive = fixture.currents[:, current_column]
    positive = [str(index + 1) for index, value in enumerate(drive) if value > 0]
    negative = [str(index + 1) for index, value in enumerate(drive) if value < 0]
    figure.suptitle(
        f"{fixture.case_id} 模式 {current_column + 1} 正问题求解结果 / "
        f"Forward results for drive {current_column + 1}\n"
        f"+I: {', '.join(positive)}  −I: {', '.join(negative)}  |  "
        f"relative Δu={body_relative_delta:.3e}, "
        f"relative ΔU={voltage_relative_delta:.3e}"
    )
    return figure, axes


def zero_sum_helmert_basis(electrode_count: int) -> np.ndarray:
    """Return deterministic orthonormal Q with Q.T@1=0 and Q.T@Q=I."""

    if electrode_count < 2:
        raise ValueError("at least two electrodes are required")
    basis = np.zeros((electrode_count, electrode_count - 1), dtype=np.float64)
    for column in range(electrode_count - 1):
        scale = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


def build_classic_state(blocks: CEMBlocks) -> ClassicDebugState:
    """分解经典 CEM 零均值增广矩阵。 / Factor the augmented Classic matrix."""

    node_count = blocks.node_count
    electrode_count = blocks.electrode_count
    gauge_column = csc_matrix(np.ones((electrode_count, 1), dtype=np.float64))
    system_matrix = bmat(
        [
            [blocks.robin_matrix, blocks.coupling, None],
            [blocks.coupling.T, blocks.electrode_matrix, gauge_column],
            [None, gauge_column.T, None],
        ],
        format="csc",
        dtype=np.float64,
    )
    factor = splu(system_matrix)
    return ClassicDebugState(
        system_matrix=system_matrix,
        factor=factor,
        node_count=node_count,
        electrode_count=electrode_count,
    )


def solve_classic(
    state: ClassicDebugState,
    currents: np.ndarray,
) -> FormulationSolution:
    """求解经典增广系统。 / Solve the Classic augmented block system."""

    current_matrix = np.asarray(currents, dtype=np.float64)
    rhs = np.zeros(
        (
            state.node_count + state.electrode_count + 1,
            current_matrix.shape[1],
        ),
        dtype=np.float64,
    )
    start = state.node_count
    stop = start + state.electrode_count
    rhs[start:stop, :] = current_matrix
    full_solution = state.factor.solve(rhs)
    return FormulationSolution(
        body_potential=full_solution[:start, :],
        electrode_voltage=full_solution[start:stop, :],
        state=state,
        lagrange_multiplier=full_solution[stop : stop + 1, :],
    )


def build_robin_state(blocks: CEMBlocks) -> RobinDebugState:
    """分解 A_R 与零和 Robin 约化映射。 / Factor A_R and the reduced map."""

    electrode_basis = zero_sum_helmert_basis(blocks.electrode_count)
    coupling_basis = np.asarray(
        blocks.coupling @ electrode_basis,
        dtype=np.float64,
    )
    body_factor = splu(blocks.robin_matrix)
    response_basis = np.asarray(
        body_factor.solve(coupling_basis),
        dtype=np.float64,
    )
    schur_action_basis = np.asarray(
        blocks.electrode_matrix @ electrode_basis - blocks.coupling.T @ response_basis,
        dtype=np.float64,
    )
    reduced_map = np.asarray(
        electrode_basis.T @ schur_action_basis,
        dtype=np.float64,
    )
    reduced_lu, reduced_pivots = lu_factor(reduced_map)
    return RobinDebugState(
        robin_matrix=blocks.robin_matrix,
        body_factor=body_factor,
        electrode_basis=electrode_basis,
        coupling_basis=coupling_basis,
        response_basis=response_basis,
        schur_action_basis=schur_action_basis,
        reduced_map=reduced_map,
        reduced_lu=reduced_lu,
        reduced_pivots=reduced_pivots,
    )


def solve_robin(
    state: RobinDebugState,
    currents: np.ndarray,
) -> FormulationSolution:
    """求解 T_r y=Q.T I，再恢复 U 与 u。 / Solve and recover U and u."""

    current_matrix = np.asarray(currents, dtype=np.float64)
    reduced_rhs = state.electrode_basis.T @ current_matrix
    coefficients = lu_solve(
        (state.reduced_lu, state.reduced_pivots),
        reduced_rhs,
    )
    electrode_voltage = state.electrode_basis @ coefficients
    body_potential = -(state.response_basis @ coefficients)
    return FormulationSolution(
        body_potential=body_potential,
        electrode_voltage=electrode_voltage,
        state=state,
    )


def solve_both_formulations(
    blocks: CEMBlocks,
) -> dict[str, FormulationSolution]:
    """Build independent states and solve both mathematically equivalent forms."""

    classic_state = build_classic_state(blocks)
    classic = solve_classic(classic_state, blocks.currents)
    robin_state = build_robin_state(blocks)
    robin = solve_robin(robin_state, blocks.currents)
    return {
        "classic": classic,
        "robin_transconductance": robin,
    }


def relative_frobenius(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return float(np.linalg.norm(candidate - reference) / denominator)


def scaled_backward_residual(
    matrix: np.ndarray | csc_matrix,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Return ||AX-B||F / (||A||F||X||F + ||B||F)."""

    residual = matrix @ solution - rhs
    matrix_norm = float(
        np.linalg.norm(matrix.data)
        if isinstance(matrix, csc_matrix)
        else np.linalg.norm(matrix)
    )
    denominator = matrix_norm * float(np.linalg.norm(solution)) + float(
        np.linalg.norm(rhs)
    )
    return float(np.linalg.norm(residual) / max(denominator, np.finfo(float).eps))


def formulation_diagnostics(
    blocks: CEMBlocks,
    solutions: dict[str, FormulationSolution],
) -> dict[str, float]:
    """Compute directly inspectable float64 parity and residual diagnostics."""

    classic = solutions["classic"]
    robin = solutions["robin_transconductance"]
    classic_state = classic.state
    robin_state = robin.state
    if not isinstance(classic_state, ClassicDebugState):
        raise TypeError("classic solution has the wrong state type")
    if not isinstance(robin_state, RobinDebugState):
        raise TypeError("Robin solution has the wrong state type")
    if classic.lagrange_multiplier is None:
        raise ValueError("Classic solution is missing its gauge multiplier")

    full_solution = np.vstack(
        (
            classic.body_potential,
            classic.electrode_voltage,
            classic.lagrange_multiplier,
        )
    )
    full_rhs = np.zeros_like(full_solution)
    start = blocks.node_count
    stop = start + blocks.electrode_count
    full_rhs[start:stop, :] = blocks.currents
    robin_coefficients = robin_state.electrode_basis.T @ robin.electrode_voltage
    robin_rhs = robin_state.electrode_basis.T @ blocks.currents
    return {
        "electrode_voltage_relative_l2": relative_frobenius(
            robin.electrode_voltage,
            classic.electrode_voltage,
        ),
        "body_potential_relative_l2": relative_frobenius(
            robin.body_potential,
            classic.body_potential,
        ),
        "classic_scaled_backward_residual": scaled_backward_residual(
            classic_state.system_matrix,
            full_solution,
            full_rhs,
        ),
        "robin_scaled_backward_residual": scaled_backward_residual(
            robin_state.reduced_map,
            robin_coefficients,
            robin_rhs,
        ),
        "classic_voltage_gauge_max_abs": float(
            np.max(np.abs(np.sum(classic.electrode_voltage, axis=0)))
        ),
        "robin_voltage_gauge_max_abs": float(
            np.max(np.abs(np.sum(robin.electrode_voltage, axis=0)))
        ),
    }


def load_portable_exact_reference(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mp_from_fraction_string(value: str):
    fraction = Fraction(value)
    return mp.mpf(fraction.numerator) / mp.mpf(fraction.denominator)


def _mp_matrix_from_fraction_strings(values: list[list[str]]):
    rows = len(values)
    columns = len(values[0]) if rows else 0
    result = mp.matrix(rows, columns)
    for row in range(rows):
        for column in range(columns):
            result[row, column] = _mp_from_fraction_string(values[row][column])
    return result


def _mp_matrix_from_float64(values: np.ndarray):
    array = np.asarray(values, dtype=np.float64)
    result = mp.matrix(array.shape[0], array.shape[1])
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            fraction = Fraction.from_float(float(array[row, column]))
            result[row, column] = mp.mpf(fraction.numerator) / mp.mpf(
                fraction.denominator
            )
    return result


def _mp_frobenius(matrix) -> Any:
    return mp.sqrt(mp.fsum(abs(value) ** 2 for value in matrix))


def exact_reference_metrics(
    candidate: np.ndarray,
    reference: dict[str, Any],
) -> dict[str, float]:
    """Reproduce the report metrics against the portable exact QQ reference."""

    with mp.workdps(100):
        candidate_mp = _mp_matrix_from_float64(candidate)
        truth_mp = _mp_matrix_from_fraction_strings(reference["voltage"])
        delta = candidate_mp - truth_mp
        relative_error = _mp_frobenius(delta) / _mp_frobenius(truth_mp)
        electrode_count = candidate_mp.rows
        gauge = mp.matrix(1, candidate_mp.cols)
        centered = mp.matrix(candidate_mp.rows, candidate_mp.cols)
        for column in range(candidate_mp.cols):
            mean = (
                mp.fsum(candidate_mp[row, column] for row in range(electrode_count))
                / electrode_count
            )
            gauge[0, column] = mean * electrode_count
            for row in range(electrode_count):
                centered[row, column] = candidate_mp[row, column] - mean
        coefficients = mp.matrix(electrode_count - 1, candidate_mp.cols)
        for row in range(electrode_count - 1):
            for column in range(candidate_mp.cols):
                coefficients[row, column] = centered[row, column]
        reduced_map = _mp_matrix_from_fraction_strings(reference["reduced_map"])
        reduced_rhs = _mp_matrix_from_fraction_strings(reference["reduced_rhs"])
        residual = reduced_map * coefficients - reduced_rhs
        backward = _mp_frobenius(residual) / (
            _mp_frobenius(reduced_map) * _mp_frobenius(coefficients)
            + _mp_frobenius(reduced_rhs)
        )
        gauge_residual = _mp_frobenius(gauge) / _mp_frobenius(candidate_mp)
        maximum = max((abs(value) for value in delta), default=mp.mpf("0"))
        reduced_float = np.asarray(
            [
                [float(Fraction(value)) for value in row]
                for row in reference["reduced_map"]
            ],
            dtype=np.float64,
        )
        return {
            "truth_relative_l2": float(relative_error),
            "truth_max_abs": float(maximum),
            "exact_reduced_scaled_backward_residual": float(backward),
            "voltage_gauge_relative_residual": float(gauge_residual),
            "reduced_condition_number_2_estimate": float(np.linalg.cond(reduced_float)),
        }


def load_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(array <= 0):
        raise ValueError("geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(array))))


def summarize_accuracy_records(
    records: list[dict[str, str]],
) -> dict[str, Any]:
    """Reproduce headline GM, win counts, and the Q4 strict ordering."""

    solvers = sorted({row["solver"] for row in records})
    formulations = sorted({row["formulation"] for row in records})
    case_ids = sorted({row["case_id"] for row in records})
    expected_rows = len(solvers) * len(formulations) * len(case_ids)
    if len(records) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} complete solver/formulation/case rows, "
            f"found {len(records)}"
        )

    geometric_means: dict[str, dict[str, float]] = {}
    win_counts: dict[str, dict[str, int]] = {}
    rankings: dict[str, list[str]] = {}
    for formulation in formulations:
        geometric_means[formulation] = {}
        win_counts[formulation] = {solver: 0 for solver in solvers}
        for solver in solvers:
            values = [
                float(row["truth_relative_l2"])
                for row in records
                if row["solver"] == solver and row["formulation"] == formulation
            ]
            geometric_means[formulation][solver] = _geometric_mean(values)
        for case_id in case_ids:
            selected = sorted(
                (
                    row
                    for row in records
                    if row["case_id"] == case_id and row["formulation"] == formulation
                ),
                key=lambda row: float(row["truth_relative_l2"]),
            )
            order = [row["solver"] for row in selected]
            rankings[f"{case_id}:{formulation}"] = order
            win_counts[formulation][order[0]] += 1

    q4_case_ids = sorted(
        {row["case_id"] for row in records if row["refinement_level_id"] == "Q4"}
    )
    q4_summary: dict[str, Any] = {}
    for formulation in formulations:
        observed = [rankings[f"{case_id}:{formulation}"] for case_id in q4_case_ids]
        same = all(order == observed[0] for order in observed[1:])
        q4_summary[formulation] = {
            "case_ids": q4_case_ids,
            "same_order_all_cases": same,
            "ordering": observed[0] if same else None,
        }

    return {
        "record_count": len(records),
        "case_count": len(case_ids),
        "solvers": solvers,
        "formulations": formulations,
        "geometric_means": geometric_means,
        "win_counts": win_counts,
        "q4_summary": q4_summary,
        "rankings": rankings,
    }


def summarize_timing_records(
    records: list[dict[str, str]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return matched-case Robin/Classic geometric-mean timing ratios."""

    solvers = sorted({row["solver"] for row in records})
    result: dict[str, dict[str, dict[str, float]]] = {}
    for solver in solvers:
        solver_rows = [row for row in records if row["solver"] == solver]
        case_ids = sorted({row["case_id"] for row in solver_rows})
        result[solver] = {}
        for phase, field in (
            ("cold", "cold_median_seconds"),
            ("setup", "setup_median_seconds"),
            ("warm_reuse", "warm_reuse_median_seconds"),
        ):
            ratios: list[float] = []
            for case_id in case_ids:
                selected = {
                    row["formulation"]: float(row[field])
                    for row in solver_rows
                    if row["case_id"] == case_id
                }
                ratios.append(selected["robin_transconductance"] / selected["classic"])
            result[solver][phase] = {
                "geometric_mean_robin_over_classic": _geometric_mean(ratios),
                "robin_faster_case_count": int(sum(value < 1.0 for value in ratios)),
                "case_count": len(ratios),
            }
    return result
