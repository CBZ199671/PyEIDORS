"""实测测量数据集辅助工具。

该模块提供 `MeasurementDataset`，用于将符合规范的测量矩阵和元数据
转换为 PyEidors 内部使用的 `EITData` 对象。这样可以把硬件/上位机的
格式适配工作与逆问题流程解耦。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np

from .structures import PatternConfig, EITData
from ..electrodes.patterns import StimMeasPatternManager


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"无法解析布尔值: {value}")
    return bool(value)


def _parse_direction(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in {"cw", "ccw"}:
        raise ValueError(f"方向必须为 'cw' 或 'ccw'，收到: {value}")
    return text


@dataclass
class MeasurementDataset:
    """封装符合规范的测量矩阵与元数据。

    参数:
        measurements: 形状为 ``(n_frames, n_meas_total)`` 或 ``(n_meas_total,)`` 的数组。
        pattern_config: 用于生成激励/测量模式的配置。
        metadata: 原始元数据字典，主要用于追踪和调试。
        data_type: 传递给 ``EITData`` 的 ``type`` 标记，例如 ``"real"``/``"difference"``。
    """

    measurements: np.ndarray
    pattern_config: PatternConfig
    stim_matrix: np.ndarray
    n_elec: int
    n_stim: int
    n_meas_total: int
    n_meas_per_stim: Sequence[int]
    metadata: Mapping[str, Any]
    data_type: str = "real"

    # ---------------------------- 构造接口 ----------------------------
    @classmethod
    def from_metadata(
        cls,
        measurements: Union[np.ndarray, Sequence[Sequence[float]]],
        metadata: Mapping[str, Any],
        data_type: str = "real",
    ) -> "MeasurementDataset":
        """根据元数据构造测量数据集。

        该方法会:
        1. 构建 ``PatternConfig``;
        2. 创建 ``StimMeasPatternManager`` 并计算测量数量;
        3. 校验测量矩阵尺寸是否一致;
        4. 返回封装后的数据集对象。
        """

        measurements_array = cls._normalize_measurements(measurements)
        pattern_config = cls._pattern_config_from_metadata(metadata)
        pattern_manager = StimMeasPatternManager(pattern_config)

        expected_meas = pattern_manager.n_meas_total
        if measurements_array.shape[1] != expected_meas:
            raise ValueError(
                "测量矩阵列数与激励/测量模式不匹配："
                f"得到 {measurements_array.shape[1]} 列，"
                f"预期 {expected_meas} 列。"
            )

        expected_frames = metadata.get("n_frames")
        if expected_frames is not None and expected_frames != measurements_array.shape[0]:
            raise ValueError(
                "测量矩阵帧数与元数据不一致："
                f"元数据 n_frames={expected_frames}, 实际帧数={measurements_array.shape[0]}"
            )

        return cls(
            measurements=measurements_array,
            pattern_config=pattern_config,
            stim_matrix=pattern_manager.stim_matrix.copy(),
            n_elec=pattern_config.n_elec,
            n_stim=pattern_manager.n_stim,
            n_meas_total=pattern_manager.n_meas_total,
            n_meas_per_stim=tuple(pattern_manager.n_meas_per_stim),
            metadata=dict(metadata),
            data_type=data_type,
        )

    # ---------------------------- 公共 API ----------------------------
    def to_eit_data(self, frame_index: int = 0, data_type: Optional[str] = None) -> EITData:
        """将指定帧转换为 ``EITData`` 对象。

        参数:
            frame_index: 选择的帧索引，默认使用第一帧。
            data_type: 覆盖默认 ``data_type``，例如 ``"difference"``。
        """

        frame = self._get_frame(frame_index)
        return EITData(
            meas=frame.copy(),
            stim_pattern=self.stim_matrix.copy(),
            n_elec=self.n_elec,
            n_stim=self.n_stim,
            n_meas=self.n_meas_total,
            type=data_type or self.data_type,
        )

    def iter_frames(self, data_type: Optional[str] = None) -> Iterator[EITData]:
        """逐帧生成 ``EITData`` 对象。"""

        for idx in range(self.measurements.shape[0]):
            yield self.to_eit_data(frame_index=idx, data_type=data_type)

    def summary(self) -> Dict[str, Any]:
        """返回测量配置与数据规模的概要信息。"""

        return {
            "n_frames": int(self.measurements.shape[0]),
            "n_elec": self.n_elec,
            "n_stim": self.n_stim,
            "n_meas_total": self.n_meas_total,
            "n_meas_per_stim": list(self.n_meas_per_stim),
            "data_type": self.data_type,
        }

    # ---------------------------- 内部工具 ----------------------------
    @staticmethod
    def _normalize_measurements(
        measurements: Union[np.ndarray, Sequence[Sequence[float]]]
    ) -> np.ndarray:
        array = np.asarray(measurements, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(
                "measurements 必须是一维或二维数组，"
                f"当前维度为 {array.ndim}"
            )
        return array

    @staticmethod
    def _pattern_config_from_metadata(metadata: Mapping[str, Any]) -> PatternConfig:
        required_fields = ["n_elec", "stim_pattern", "meas_pattern"]
        missing = [field for field in required_fields if field not in metadata]
        if missing:
            raise KeyError(f"metadata 缺少必要字段: {', '.join(missing)}")

        return PatternConfig(
            n_elec=int(metadata["n_elec"]),
            n_rings=int(metadata.get("n_rings", 1)),
            stim_pattern=metadata.get("stim_pattern", "{ad}"),
            meas_pattern=metadata.get("meas_pattern", "{ad}"),
            amplitude=float(metadata.get("amplitude", 1.0)),
            use_meas_current=_parse_bool(metadata.get("use_meas_current"), False),
            use_meas_current_next=int(metadata.get("use_meas_current_next", 0)),
            rotate_meas=_parse_bool(metadata.get("rotate_meas"), True),
            stim_direction=_parse_direction(metadata.get("stim_direction"), "ccw"),
            meas_direction=_parse_direction(metadata.get("meas_direction"), "ccw"),
            stim_first_positive=_parse_bool(metadata.get("stim_first_positive"), False),
        )

    def _get_frame(self, frame_index: int) -> np.ndarray:
        if not 0 <= frame_index < self.measurements.shape[0]:
            raise IndexError(
                f"frame_index 超出范围: {frame_index}，"
                f"可用索引为 [0, {self.measurements.shape[0] - 1}]"
            )
        return self.measurements[frame_index]
