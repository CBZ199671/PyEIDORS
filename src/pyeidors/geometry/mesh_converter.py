"""网格格式转换器"""

import numpy as np
import meshio
from pathlib import Path
from configparser import ConfigParser
from typing import Dict, Tuple, Any
import logging

# 设置日志
logger = logging.getLogger(__name__)

# 检查FEniCS可用性
try:
    from fenics import Mesh
    from dolfin import XDMFFile, MeshValueCollection
    from dolfin.cpp.mesh import MeshFunctionSizet
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS/dolfin不可用")


class MeshConverter:
    """网格格式转换器"""
    
    def __init__(self, mesh_file: str, output_dir: str):
        self.mesh_file = mesh_file
        self.output_dir = output_dir
        self.prefix = Path(mesh_file).stem
    
    def convert(self) -> Tuple[Any, Any, Dict[str, int]]:
        """转换MSH到FEniCS格式"""
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS不可用，无法转换网格")
            
        # 读取MSH文件
        msh = meshio.read(self.mesh_file)
        
        # 导出XDMF文件
        self._export_domain(msh)
        self._export_boundaries(msh)
        association_table = self._export_association_table(msh)
        
        # 导入到FEniCS
        return self._import_fenics_mesh(association_table)
    
    def _export_domain(self, msh):
        """导出域XDMF文件"""
        cell_type = "triangle"
        
        # 提取域单元
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            raise ValueError("未找到域物理组")
        
        data = np.concatenate([cell.data for cell in cells])
        domain_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]
        
        # 提取单元数据
        cell_data = {
            "subdomains": [
                np.concatenate([
                    msh.cell_data["gmsh:physical"][i]
                    for i, cell in enumerate(msh.cells)
                    if cell.type == cell_type
                ])
            ]
        }
        
        # 创建域网格
        domain = meshio.Mesh(
            points=msh.points[:, :2],
            cells=domain_cells,
            cell_data=cell_data
        )
        
        # 导出XDMF
        meshio.write(f"{self.output_dir}/{self.prefix}_domain.xdmf", domain)
    
    def _export_boundaries(self, msh):
        """导出边界XDMF文件"""
        cell_type = "line"
        
        # 提取边界单元
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            logger.warning("未找到边界物理组")
            return
        
        data = np.concatenate([cell.data for cell in cells])
        boundary_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]
        
        # 提取单元数据
        cell_data = {
            "boundaries": [
                np.concatenate([
                    msh.cell_data["gmsh:physical"][i]
                    for i, cell in enumerate(msh.cells)
                    if cell.type == cell_type
                ])
            ]
        }
        
        # 创建边界网格
        boundaries = meshio.Mesh(
            points=msh.points[:, :2],
            cells=boundary_cells,
            cell_data=cell_data
        )
        
        # 导出XDMF
        meshio.write(f"{self.output_dir}/{self.prefix}_boundaries.xdmf", boundaries)
    
    def _export_association_table(self, msh) -> Dict[str, int]:
        """导出关联表"""
        association_table = {}
        
        try:
            for label, arrays in msh.cell_sets.items():
                # 查找非空数组
                for i, array in enumerate(arrays):
                    if array.size != 0 and label != "gmsh:bounding_entities":
                        if i < len(msh.cell_data["gmsh:physical"]):
                            value = msh.cell_data["gmsh:physical"][i][0]
                            association_table[label] = int(value)
                        break
        except Exception as e:
            logger.warning(f"处理关联表时出错: {e}")
        
        # 保存关联表
        config = ConfigParser()
        config["ASSOCIATION TABLE"] = {k: str(v) for k, v in association_table.items()}
        
        with open(f"{self.output_dir}/{self.prefix}_association_table.ini", 'w') as f:
            config.write(f)
        
        return association_table
    
    def _import_fenics_mesh(self, association_table: Dict[str, int]) -> Tuple[Any, Any, Dict[str, int]]:
        """导入FEniCS网格"""
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS/dolfin不可用")
        
        # 导入域
        mesh = Mesh()
        with XDMFFile(f"{self.output_dir}/{self.prefix}_domain.xdmf") as infile:
            infile.read(mesh)
        
        # 导入边界
        boundaries_mvc = MeshValueCollection("size_t", mesh, dim=1)
        with XDMFFile(f"{self.output_dir}/{self.prefix}_boundaries.xdmf") as infile:
            infile.read(boundaries_mvc, 'boundaries')
        boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
        
        return mesh, boundaries_mf, association_table