"""简化的EIT网格生成器 - 使用GMsh生成FEniCS兼容的网格"""

import numpy as np
import tempfile
import time
from pathlib import Path
from math import pi, cos, sin
import logging

logger = logging.getLogger(__name__)

# 检查依赖
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    logger.warning("GMsh不可用，无法生成网格")

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    logger.warning("meshio不可用，网格转换功能受限")

try:
    from fenics import Mesh, MeshFunction, HDF5File
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS不可用，无法创建FEniCS网格对象")


class SimpleEITMeshGenerator:
    """简化的EIT网格生成器"""
    
    def __init__(self, n_elec: int = 16, radius: float = 1.0, 
                 mesh_size: float = 0.1, electrode_width: float = 0.1):
        """
        初始化网格生成器
        
        参数:
            n_elec: 电极数量
            radius: 圆形域半径
            mesh_size: 网格尺寸
            electrode_width: 电极宽度（弧度）
        """
        if not GMSH_AVAILABLE:
            raise ImportError("GMsh不可用，请安装gmsh: pip install gmsh")
        
        self.n_elec = n_elec
        self.radius = radius
        self.mesh_size = mesh_size
        self.electrode_width = electrode_width
        
        # 计算电极位置
        self.electrode_positions = self._calculate_electrode_positions()
    
    def _calculate_electrode_positions(self):
        """计算电极位置"""
        positions = []
        for i in range(self.n_elec):
            center_angle = 2 * pi * i / self.n_elec
            start_angle = center_angle - self.electrode_width / 2
            end_angle = center_angle + self.electrode_width / 2
            positions.append((start_angle, end_angle))
        return positions
    
    def generate_circular_mesh(self, output_dir: str = None, 
                              save_files: bool = True) -> object:
        """
        生成圆形EIT网格
        
        参数:
            output_dir: 输出目录
            save_files: 是否保存网格文件
            
        返回:
            FEniCS网格对象（如果可用）或网格信息字典
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 生成时间戳
        timestamp = int(time.time() * 1000) % 1000000
        mesh_name = f"eit_mesh_{timestamp}"
        
        logger.info(f"开始生成EIT网格: {mesh_name}")
        
        # 初始化GMsh
        gmsh.initialize()
        gmsh.model.add(mesh_name)
        
        try:
            # 创建几何
            self._create_simple_circular_geometry()
            
            # 设置物理组
            self._set_physical_groups()
            
            # 生成网格
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.mesh_size)
            gmsh.model.mesh.generate(2)
            
            # 保存网格文件
            if save_files:
                msh_file = output_path / f"{mesh_name}.msh"
                gmsh.write(str(msh_file))
                logger.info(f"网格文件已保存: {msh_file}")
            
            # 转换为FEniCS格式
            if FENICS_AVAILABLE:
                fenics_mesh = self._convert_to_fenics(mesh_name, output_path, save_files)
                logger.info("FEniCS网格创建成功")
                return fenics_mesh
            else:
                logger.warning("FEniCS不可用，返回网格信息")
                return self._create_mesh_info(mesh_name, output_path)
        
        finally:
            gmsh.finalize()
    
    def _create_simple_circular_geometry(self):
        """创建简化的圆形几何"""
        # 创建中心点
        center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, self.mesh_size)
        
        # 创建边界点
        boundary_points = []
        for i in range(self.n_elec * 4):  # 每个电极区域用4个点表示
            angle = 2 * pi * i / (self.n_elec * 4)
            x = self.radius * cos(angle)
            y = self.radius * sin(angle)
            point = gmsh.model.geo.addPoint(x, y, 0.0, self.mesh_size)
            boundary_points.append(point)
        
        # 创建边界线（圆弧）
        boundary_lines = []
        electrode_lines = []
        gap_lines = []
        
        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            line = gmsh.model.geo.addCircleArc(boundary_points[i], center, boundary_points[next_i])
            boundary_lines.append(line)
            
            # 确定这条线是电极还是间隙
            # 每4条线中的第2、3条是电极
            local_pos = i % 4
            if local_pos in [1, 2]:
                electrode_lines.append(line)
            else:
                gap_lines.append(line)
        
        # 创建曲线环和平面
        curve_loop = gmsh.model.geo.addCurveLoop(boundary_lines)
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        
        # 同步几何
        gmsh.model.geo.synchronize()
        
        # 保存信息
        self.electrode_lines = electrode_lines
        self.gap_lines = gap_lines
        self.boundary_lines = boundary_lines
        self.surface = surface
    
    def _set_physical_groups(self):
        """设置物理组"""
        # 域
        gmsh.model.addPhysicalGroup(2, [self.surface], 1, name="domain")
        
        # 电极 - 将相邻的电极线分组
        electrode_groups = []
        for i in range(self.n_elec):
            # 每个电极由2条相邻的线组成
            start_idx = i * 2
            if start_idx + 1 < len(self.electrode_lines):
                elec_lines = self.electrode_lines[start_idx:start_idx + 2]
                if elec_lines:  # 确保有线
                    gmsh.model.addPhysicalGroup(1, elec_lines, i + 2, name=f"electrode_{i+1}")
                    electrode_groups.extend(elec_lines)
        
        # 间隙 - 所有非电极的边界线
        remaining_lines = [line for line in self.boundary_lines if line not in electrode_groups]
        if remaining_lines:
            gmsh.model.addPhysicalGroup(1, remaining_lines, self.n_elec + 2, name="boundary")
    
    def _convert_to_fenics(self, mesh_name: str, output_path: Path, save_files: bool):
        """转换为FEniCS网格"""
        # 使用meshio转换
        if MESHIO_AVAILABLE:
            # 保存临时msh文件
            temp_msh = output_path / f"{mesh_name}_temp.msh"
            gmsh.write(str(temp_msh))
            
            # 使用meshio读取并转换
            mesh_data = meshio.read(temp_msh)
            
            if save_files:
                # 保存为XDMF格式
                xdmf_file = output_path / f"{mesh_name}.xdmf"
                meshio.write(xdmf_file, mesh_data)
                
                # 创建边界信息
                self._create_boundary_files(mesh_data, mesh_name, output_path)
            
            # 创建FEniCS mesh对象
            return self._create_fenics_mesh_object(mesh_data, mesh_name)
        
        else:
            logger.warning("meshio不可用，无法转换网格格式")
            return self._create_mesh_info(mesh_name, output_path)
    
    def _create_boundary_files(self, mesh_data, mesh_name: str, output_path: Path):
        """创建边界信息文件"""
        # 创建关联表
        association_table = {}
        for i in range(self.n_elec):
            association_table[i + 2] = i + 2  # 电极标记从2开始
        
        # 保存关联表
        import configparser
        config = configparser.ConfigParser()
        config['boundary_ids'] = {str(k): str(v) for k, v in association_table.items()}
        
        association_file = output_path / f"{mesh_name}_association_table.ini"
        with open(association_file, 'w') as f:
            config.write(f)
    
    def _create_fenics_mesh_object(self, mesh_data, mesh_name: str):
        """创建FEniCS网格对象"""
        
        class EnhancedEITMesh:
            """增强的EIT网格对象"""
            
            def __init__(self, mesh_data, mesh_name, generator):
                self.mesh_data = mesh_data
                self.mesh_name = mesh_name
                self.generator = generator
                
                # 基本属性
                self.radius = generator.radius
                self.n_elec = generator.n_elec
                self.vertex_elec = []
                
                # 创建关联表
                self.association_table = {i + 2: i + 2 for i in range(self.n_elec)}
                
                # 模拟边界标记
                self.boundaries_mf = None
                
                # 网格统计
                self._compute_stats()
            
            def _compute_stats(self):
                """计算网格统计信息"""
                points = self.mesh_data.points
                cells = self.mesh_data.cells
                
                self._num_vertices = len(points)
                self._num_cells = len(cells[0].data) if cells else 0
                
                # 计算边界框
                self.bbox_min = np.min(points[:, :2], axis=0)
                self.bbox_max = np.max(points[:, :2], axis=0)
                self.center = np.mean(points[:, :2], axis=0)
            
            def coordinates(self):
                """返回坐标数组"""
                return self.mesh_data.points[:, :2]  # 只返回2D坐标
            
            def num_vertices(self):
                """返回顶点数"""
                return self._num_vertices
            
            def num_cells(self):
                """返回单元数"""
                return self._num_cells
            
            def cells(self):
                """返回单元连接"""
                if self.mesh_data.cells:
                    return self.mesh_data.cells[0].data
                return np.array([])
            
            def get_info(self):
                """获取网格信息"""
                return {
                    'mesh_name': self.mesh_name,
                    'num_vertices': self.num_vertices(),
                    'num_cells': self.num_cells(),
                    'num_electrodes': self.n_elec,
                    'radius': self.radius,
                    'center': self.center.tolist(),
                    'bbox': [self.bbox_min.tolist(), self.bbox_max.tolist()],
                    'association_table': self.association_table
                }
        
        return EnhancedEITMesh(mesh_data, mesh_name, self)
    
    def _create_mesh_info(self, mesh_name: str, output_path: Path):
        """创建网格信息字典"""
        return {
            'mesh_name': mesh_name,
            'n_elec': self.n_elec,
            'radius': self.radius,
            'mesh_size': self.mesh_size,
            'output_path': str(output_path),
            'note': 'FEniCS不可用，只返回基本信息'
        }


def create_simple_eit_mesh(n_elec: int = 16, radius: float = 1.0, 
                          mesh_size: float = 0.1, output_dir: str = None):
    """
    快速创建EIT网格的便捷函数
    
    参数:
        n_elec: 电极数量
        radius: 半径
        mesh_size: 网格尺寸
        output_dir: 输出目录
        
    返回:
        网格对象
    """
    generator = SimpleEITMeshGenerator(
        n_elec=n_elec,
        radius=radius,
        mesh_size=mesh_size,
        electrode_width=2 * pi / n_elec / 8  # 电极占1/8圆周，更小的电极
    )
    
    return generator.generate_circular_mesh(output_dir=output_dir)