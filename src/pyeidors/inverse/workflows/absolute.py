"""绝对成像流程封装。"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from typing import TYPE_CHECKING

from ...data.structures import EITData, EITImage
from .base import (
    ReconstructionResult,
    resolve_reconstruction_output,
    compute_residuals,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...core_system import EITSystem


def perform_absolute_reconstruction(
    eit_system: "EITSystem",
    measurement_data: EITData,
    baseline_image: Optional[EITImage] = None,
    initial_image: Optional[EITImage] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ReconstructionResult:
    """执行绝对成像。

    参数:
        eit_system: 已完成 setup 的 `EITSystem` 实例。
        measurement_data: 当前帧测量数据 (`EITData`)，通常来自实测或仿真。
        baseline_image: 均匀导电率图像，用于生成前向基线及显示差值。
                        若未提供则自动使用 `create_homogeneous_image()`。
        initial_image: 求解初始化猜测，缺省使用 `baseline_image`。
        metadata: 附加信息（帧索引、频率等），会原样存入结果。
    """

    if not eit_system._is_initialized:  # pylint: disable=protected-access
        raise RuntimeError("EITSystem 尚未初始化，请先调用 setup()。")

    if baseline_image is None:
        baseline_image = eit_system.create_homogeneous_image()

    baseline_elem = baseline_image.elem_data.copy()
    initial_guess = (
        initial_image.elem_data if initial_image is not None else baseline_elem
    )

    reconstruction = eit_system.inverse_solve(
        data=measurement_data,
        reference_data=None,
        initial_guess=initial_guess,
    )

    conductivity_image, conductivity_values, residual_history, sigma_history = (
        resolve_reconstruction_output(reconstruction, eit_system.fwd_model)
    )

    simulated_data, _ = eit_system.fwd_model.fwd_solve(conductivity_image)

    measured_vector = measurement_data.meas
    simulated_vector = simulated_data.meas
    residual_vector, l2_error, rel_error, mse = compute_residuals(
        measured_vector, simulated_vector
    )

    result_metadata: Dict[str, Any] = {
        "display_values": conductivity_values,
        "baseline_used": baseline_elem,
    }
    if metadata:
        result_metadata.update(metadata)

    return ReconstructionResult(
        mode="absolute",
        conductivity=conductivity_values,
        conductivity_image=conductivity_image,
        measured=measured_vector,
        simulated=simulated_vector,
        residual=residual_vector,
        residual_history=residual_history,
        sigma_change_history=sigma_history,
        metadata=result_metadata,
    )
