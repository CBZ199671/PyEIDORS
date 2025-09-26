"""优化的网格生成器"""

import numpy as np
import gmsh
import meshio
import tempfile
import time
from pathlib import Path
from math import pi, cos, sin
from typing import Optional, Dict, Any, Union, Tuple
from contextlib import contextmanager
import logging

from ..data.structures import MeshConfig, ElectrodePosition
from .mesh_converter import MeshConverter

# 设置日志
logger = logging.getLogger(__name__)

# 检查FEniCS可用性
try:
    from fenics import Mesh
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS不可用，只能生成原始网格文件")


class MeshGenerator:
    """优化的网格生成器"""
    
    def __init__(self, config: MeshConfig, electrodes: ElectrodePosition):
        self.config = config
        self.electrodes = electrodes
        self.mesh_data = {}
        
    @contextmanager
    def gmsh_context(self, model_name: str = "EIT_Mesh"):
        """GMsh上下文管理器"""
        gmsh.initialize()
        gmsh.model.add(model_name)
        try:
            yield
        finally:
            gmsh.finalize()
    
    def generate(self, output_dir: Optional[Path] = None, 
                 use_fenics: bool = True) -> Union[Mesh, Dict[str, Any]]:
        """
        生成网格
        
        参数:
            output_dir: 输出目录
            use_fenics: 是否转换为FEniCS格式
            
        返回:
            FEniCS Mesh对象或网格数据字典
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        mesh_file = output_dir / f"mesh_{int(time.time() * 1e6) % 1000000}.msh"
        
        with self.gmsh_context():
            self._create_geometry()
            self._set_physical_groups()
            self._generate_mesh()
            gmsh.write(str(mesh_file))
            
            # 保存电极顶点数据
            self._extract_electrode_vertices()
        
        if use_fenics and FENICS_AVAILABLE:
            return self._convert_to_fenics(mesh_file, output_dir)
        else:
            return {
                'mesh_file': mesh_file,
                'radius': self.config.radius,
                'electrodes': self.electrodes,
                'vertex_data': self.mesh_data.get('electrode_vertices', [])
            }
    
    def _create_geometry(self):
        """创建几何形状"""
        positions = self.electrodes.positions
        n_in = self.config.electrode_vertices
        n_out = self.config.gap_vertices
        r = self.config.radius
        
        boundary_points = []
        electrode_ranges = []
        
        # 创建边界点
        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)
            
            # 电极点
            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)
            
            electrode_ranges.append((start_idx, len(boundary_points) - 1))
            
            # 间隙点
            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi
            
            gap_points = np.linspace(gap_start, gap_end, n_out + 2)[1:-1]
            for theta in gap_points:
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)
        
        # 创建边界线
        lines = []
        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            line = gmsh.model.occ.addLine(boundary_points[i], boundary_points[next_i])
            lines.append(line)
        
        # 创建表面
        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])
        
        # 添加内部控制点
        mesh_size_center = 0.095
        cp_distance = 0.1
        center_points = [
            gmsh.model.occ.addPoint(x, y, 0.0, meshSize=mesh_size_center)
            for x, y in [(-cp_distance, cp_distance), (cp_distance, cp_distance),
                         (-cp_distance, -cp_distance), (cp_distance, -cp_distance)]
        ]
        
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.embed(0, center_points, 2, surface)
        
        # 保存数据
        self.mesh_data['boundary_points'] = boundary_points
        self.mesh_data['electrode_ranges'] = electrode_ranges
        self.mesh_data['lines'] = lines
        self.mesh_data['surface'] = surface
    
    def _set_physical_groups(self):
        """设置物理组"""
        surface = self.mesh_data['surface']
        lines = self.mesh_data['lines']
        electrode_ranges = self.mesh_data['electrode_ranges']
        
        # 域
        gmsh.model.addPhysicalGroup(2, [surface], 1, name="domain")
        
        # 电极
        electrode_lines = []
        for i, (start, end) in enumerate(electrode_ranges):
            lines_for_electrode = []
            for j in range(start, end):
                line_idx = j % len(lines)
                lines_for_electrode.append(lines[line_idx])
            
            if lines_for_electrode:
                gmsh.model.addPhysicalGroup(1, lines_for_electrode, i + 2, 
                                          name=f"electrode_{i+1}")
                electrode_lines.extend(lines_for_electrode)
        
        # 间隙
        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            gmsh.model.addPhysicalGroup(1, gap_lines, self.electrodes.L + 2, name="gaps")
    
    def _generate_mesh(self):
        """生成网格"""
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
        gmsh.model.mesh.generate(2)
    
    def _extract_electrode_vertices(self):
        """提取电极顶点坐标"""
        positions = self.electrodes.positions
        r = self.config.radius
        n_in = self.config.electrode_vertices
        
        electrode_vertices = []
        for start, end in positions:
            vertices = []
            for theta in np.linspace(start, end, n_in):
                vertices.append([r * cos(theta), r * sin(theta)])
            electrode_vertices.append(vertices)
        
        self.mesh_data['electrode_vertices'] = electrode_vertices
    
    def _convert_to_fenics(self, mesh_file: Path, output_dir: Path):
        """转换为FEniCS网格"""
        converter = MeshConverter(str(mesh_file), str(output_dir))
        mesh, boundaries_mf, association_table = converter.convert()
        
        # 添加兼容性属性
        mesh.radius = self.config.radius
        mesh.vertex_elec = self.mesh_data.get('electrode_vertices', [])
        mesh.electrodes = self.electrodes
        mesh.boundaries_mf = boundaries_mf
        mesh.association_table = association_table
        
        return mesh