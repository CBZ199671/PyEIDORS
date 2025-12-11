"""网格相关工具函数。"""

from __future__ import annotations

import numpy as np


def cell_to_node(mesh, cell_values: np.ndarray) -> np.ndarray:
    """将单元值转换为节点值（通过平均相邻单元）。
    
    Args:
        mesh: FEniCS 网格对象
        cell_values: 每个单元的值，形状 (num_cells,)
        
    Returns:
        每个节点的值，形状 (num_vertices,)
    """
    node_vals = np.zeros(mesh.num_vertices())
    counts = np.zeros(mesh.num_vertices())
    
    for ci, cell in enumerate(mesh.cells()):
        for v in cell:
            node_vals[v] += cell_values[ci]
            counts[v] += 1
    
    counts[counts == 0] = 1  # 避免除零
    node_vals /= counts
    return node_vals


def get_mesh_info(mesh) -> dict:
    """获取网格基本信息。
    
    Args:
        mesh: FEniCS 网格对象
        
    Returns:
        包含网格信息的字典
    """
    return {
        "num_cells": mesh.num_cells(),
        "num_vertices": mesh.num_vertices(),
        "num_edges": mesh.num_edges(),
        "geometric_dimension": mesh.geometric_dimension(),
        "topology_dimension": mesh.topology().dim(),
    }
