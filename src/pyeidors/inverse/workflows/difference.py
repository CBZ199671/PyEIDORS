"""差分成像流程封装。"""

from __future__ import annotations

from typing import Optional, Dict, Any

from typing import TYPE_CHECKING

from ...data.structures import EITData, EITImage
from .base import (
    ReconstructionResult,
    resolve_reconstruction_output,
    compute_residuals,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...core_system import EITSystem


def perform_difference_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    reference_data: EITData,
    initial_image: Optional[EITImage] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReconstructionResult:
    """执行差分成像。

    参数:
        eit_system: 已 setup 的 `EITSystem`。
        measurement_data: 目标帧 `EITData`。
        reference_data: 参考帧 `EITData`。
        initial_image: 初始导电率图像，可选。
        metadata: 附加信息（帧索引等）。
    """

    if not eit_system._is_initialized:  # pylint: disable=protected-access
        raise RuntimeError("EITSystem 尚未初始化，请先调用 setup()。")

    initial_guess = (
        initial_image.elem_data if initial_image is not None else None
    )

    reconstruction = eit_system.inverse_solve(
        data=measurement_data,
        reference_data=reference_data,
        initial_guess=initial_guess,
    )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(reconstruction, eit_system.fwd_model)
    )

    simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)

    measured_vector = measurement_data.meas - reference_data.meas
    simulated_vector = simulated_data.meas - reference_data.meas
    residual_vector, l2_error, rel_error, mse = compute_residuals(
        measured_vector, simulated_vector
    )

    result_metadata: Dict[str, Any] = {
        "reference_measured": reference_data.meas,
        "display_values": conductivity_values,
    }
    if metadata:
        result_metadata.update(metadata)

    return ReconstructionResult(
        mode="difference",
        conductivity=conductivity_values,
        conductivity_image=conductivity_image,
        measured=measured_vector,
        simulated=simulated_vector,
        residual=residual_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_history,
        metadata=result_metadata,
    )
