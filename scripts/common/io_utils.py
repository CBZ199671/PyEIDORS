"""数据输入输出工具函数。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("需要 PyYAML，请安装: pip install pyyaml") from exc


def load_metadata(path: Path) -> Dict[str, Any]:
    """加载 YAML 或 JSON 格式的元数据文件。
    
    Args:
        path: 元数据文件路径 (.yaml, .yml, .json)
        
    Returns:
        解析后的元数据字典
    """
    with path.open("r", encoding="utf-8") as fh:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(fh)
        if suffix == ".json":
            return json.load(fh)
        raise ValueError(f"不支持的元数据格式: {suffix}")


def load_csv_measurements(
    csv_path: Path,
    use_part: str = "real",
    measurement_gain: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载 EIT 测量 CSV 文件。
    
    期望 CSV 格式: 4列 (ref_real, ref_imag, target_real, target_imag)
    
    Args:
        csv_path: CSV 文件路径
        use_part: 使用哪部分数据 ("real", "imag", "mag")
        measurement_gain: 测量放大器增益，数据会除以此值
        
    Returns:
        (reference_frame, target_frame) 两帧测量数据
    """
    arr = np.loadtxt(csv_path, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"期望 4 列 CSV，实际形状: {arr.shape}")
    
    ref_re, ref_im, tgt_re, tgt_im = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    
    if use_part == "real":
        ref, tgt = ref_re, tgt_re
    elif use_part == "imag":
        ref, tgt = ref_im, tgt_im
    elif use_part == "mag":
        ref = np.abs(ref_re + 1j * ref_im)
        tgt = np.abs(tgt_re + 1j * tgt_im)
    else:
        raise ValueError(f"未知 use_part={use_part}，可选: real, imag, mag")
    
    if measurement_gain != 1.0:
        ref = ref / measurement_gain
        tgt = tgt / measurement_gain
    
    return ref, tgt


def load_single_frame(
    csv_path: Path,
    col_idx: int,
    measurement_gain: float = 1.0,
) -> np.ndarray:
    """从 CSV 加载单帧测量数据。
    
    Args:
        csv_path: CSV 文件路径
        col_idx: 列索引 (0-based)
        measurement_gain: 测量放大器增益
        
    Returns:
        单帧测量向量
    """
    raw = np.loadtxt(csv_path, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] <= col_idx:
        raise ValueError(f"CSV 形状 {raw.shape} 无法提取列 {col_idx}")
    
    frame = raw[:, col_idx].astype(float)
    if measurement_gain != 1.0:
        frame = frame / measurement_gain
    return frame


def save_reconstruction_results(
    output_dir: Path,
    conductivity: np.ndarray,
    measured: np.ndarray,
    predicted: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """保存重建结果到文件。
    
    Args:
        output_dir: 输出目录
        conductivity: 重建的电导率向量
        measured: 测量电压
        predicted: 预测电压
        metadata: 可选的运行元数据
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数值数据
    np.savez(
        output_dir / "result_arrays.npz",
        conductivity=conductivity,
        measured=measured,
        predicted=predicted,
        residual=np.asarray(predicted) - np.asarray(measured),
    )
    
    # 保存元数据
    if metadata is not None:
        with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)
