"""优化的EIT网格生成器 - 基于参考实现的改进版本"""

import numpy as np
import tempfile
import time
from pathlib import Path
from math import pi, cos, sin
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Any

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
    from fenics import Mesh, MeshFunction, MeshValueCollection, XDMFFile
    from dolfin.cpp.mesh import MeshFunctionSizet
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS不可用，无法创建FEniCS网格对象")


@dataclass
class ElectrodePosition:
    """电极位置配置 - 基于参考实现"""
    L: int  # 电极数量
    coverage: float = 0.5  # 电极覆盖率
    rotation: float = 0.0  # 旋转角度
    anticlockwise: bool = True  # 逆时针方向
    
    def __post_init__(self):
        if not isinstance(self.L, int) or self.L <= 0:
            raise ValueError("电极数量必须是正整数")
        if not 0 < self.coverage <= 1:
            raise ValueError("覆盖率必须在(0, 1]范围内")
    
    @property
    def positions(self) -> List[Tuple[float, float]]:
        """计算每个电极的起始和结束角度"""
        electrode_size = 2 * pi / self.L * self.coverage
        gap_size = 2 * pi / self.L * (1 - self.coverage)
        
        # 计算第一个电极的中心位置应该在y轴正半轴 (π/2)
        # 因此第一个电极的起始位置应该是 π/2 - electrode_size/2
        first_electrode_center = pi / 2 + self.rotation
        first_electrode_start = first_electrode_center - electrode_size / 2
        
        positions = []
        for i in range(self.L):
            # 每个电极占用的总角度空间（电极 + 间隙）
            total_space_per_electrode = electrode_size + gap_size
            
            start = first_electrode_start + i * total_space_per_electrode
            end = start + electrode_size
            positions.append((start, end))
        
        if not self.anticlockwise:
            positions[1:] = positions[1:][::-1]
        
        return positions


@dataclass  
class OptimizedMeshConfig:
    """优化的网格配置参数"""
    radius: float = 1.0
    refinement: int = 8
    electrode_vertices: int = 6  # 每个电极的顶点数
    gap_vertices: int = 1       # 间隙区域的顶点数
    
    @property
    def mesh_size(self) -> float:
        """计算网格尺寸"""
        return self.radius / (self.refinement * 2)


class OptimizedMeshGenerator:
    """优化的网格生成器 - 基于参考实现"""
    
    def __init__(self, config: OptimizedMeshConfig, electrodes: ElectrodePosition):
        """
        初始化网格生成器
        
        参数:
            config: 网格配置
            electrodes: 电极位置配置
        """
        if not GMSH_AVAILABLE:
            raise ImportError("GMsh不可用，请安装gmsh: pip install gmsh")
        
        self.config = config
        self.electrodes = electrodes
        self.mesh_data = {}
    
    def generate(self, output_dir: Optional[Path] = None) -> object:
        """
        生成网格
        
        参数:
            output_dir: 输出目录
            
        返回:
            FEniCS网格对象或网格信息
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # 生成唯一的网格文件名
        timestamp = int(time.time() * 1e6) % 1000000
        mesh_file = output_dir / f"mesh_{timestamp}.msh"
        
        logger.info(f"开始生成EIT网格: {mesh_file.stem}")
        
        # 初始化GMsh
        gmsh.initialize()
        gmsh.model.add("EIT_Mesh")
        
        try:
            # 创建几何
            self._create_geometry()
            
            # 设置物理组
            self._set_physical_groups()
            
            # 生成网格
            self._generate_mesh()
            
            # 保存网格文件
            gmsh.write(str(mesh_file))
            
            # 提取电极顶点信息
            self._extract_electrode_vertices()
            
        finally:
            gmsh.finalize()
        
        # 转换为FEniCS格式
        return self._convert_to_fenics(mesh_file, output_dir)
    
    def _create_geometry(self):
        """创建几何 - 基于参考实现的逻辑"""
        positions = self.electrodes.positions
        n_in = self.config.electrode_vertices  # 电极顶点数
        n_out = self.config.gap_vertices       # 间隙顶点数
        r = self.config.radius
        
        boundary_points = []
        electrode_ranges = []
        
        # 为每个电极创建顶点
        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)
            
            # 在电极区域创建顶点
            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)
            
            # 记录电极范围
            electrode_ranges.append((start_idx, len(boundary_points) - 1))
            
            # 在间隙区域创建顶点
            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi
            
            # 间隙顶点（不包括端点，避免重复）
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
        
        # 创建曲线环和表面
        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])
        
        # 添加内部控制点以改善网格质量
        mesh_size_center = 0.095
        cp_distance = 0.1
        center_points = [
            gmsh.model.occ.addPoint(x, y, 0.0, meshSize=mesh_size_center)
            for x, y in [(-cp_distance, cp_distance), (cp_distance, cp_distance),
                         (-cp_distance, -cp_distance), (cp_distance, -cp_distance)]
        ]
        
        # 同步几何模型
        gmsh.model.occ.synchronize()
        
        # 嵌入控制点
        gmsh.model.mesh.embed(0, center_points, 2, surface)
        
        # 保存几何信息
        self.mesh_data['boundary_points'] = boundary_points
        self.mesh_data['electrode_ranges'] = electrode_ranges
        self.mesh_data['lines'] = lines
        self.mesh_data['surface'] = surface
    
    def _set_physical_groups(self):
        """设置物理组 - 用于边界条件"""
        surface = self.mesh_data['surface']
        lines = self.mesh_data['lines']
        electrode_ranges = self.mesh_data['electrode_ranges']
        
        # 设置域物理组
        gmsh.model.addPhysicalGroup(2, [surface], 1, name="domain")
        
        # 设置电极物理组
        electrode_lines = []
        for i, (start, end) in enumerate(electrode_ranges):
            lines_for_electrode = []
            for j in range(start, end):
                line_idx = j % len(lines)
                lines_for_electrode.append(lines[line_idx])
            
            if lines_for_electrode:
                # 电极编号从2开始（1是域）
                gmsh.model.addPhysicalGroup(1, lines_for_electrode, i + 2, 
                                          name=f"electrode_{i+1}")
                electrode_lines.extend(lines_for_electrode)
        
        # 设置间隙（非电极边界）物理组
        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            gmsh.model.addPhysicalGroup(1, gap_lines, self.electrodes.L + 2, name="gaps")
    
    def _generate_mesh(self):
        """生成网格"""
        # 设置网格尺寸
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
        
        # 生成2D网格
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
        """转换为FEniCS网格格式"""
        if not FENICS_AVAILABLE:
            logger.warning("FEniCS不可用，返回基本网格信息")
            return self._create_mesh_info(mesh_file.stem, output_dir)
        
        try:
            # 使用转换器转换网格
            converter = OptimizedMeshConverter(str(mesh_file), str(output_dir))
            mesh, boundaries_mf, association_table = converter.convert()
            
            # 添加EIT特定属性
            mesh.radius = self.config.radius
            mesh.vertex_elec = self.mesh_data.get('electrode_vertices', [])
            mesh.electrodes = self.electrodes
            mesh.boundaries_mf = boundaries_mf
            mesh.association_table = association_table
            
            logger.info("FEniCS网格转换成功")
            return mesh
            
        except Exception as e:
            logger.error(f"FEniCS网格转换失败: {e}")
            return self._create_mesh_info(mesh_file.stem, output_dir)
    
    def _create_mesh_info(self, mesh_name: str, output_dir: Path):
        """创建网格信息字典（FEniCS不可用时的备选方案）"""
        return {
            'mesh_name': mesh_name,
            'n_electrodes': self.electrodes.L,
            'radius': self.config.radius,
            'refinement': self.config.refinement,
            'output_dir': str(output_dir),
            'electrode_vertices': self.mesh_data.get('electrode_vertices', []),
            'note': 'FEniCS不可用，仅返回基本信息'
        }


class OptimizedMeshConverter:
    """优化的网格格式转换器 - 基于参考实现"""
    
    def __init__(self, mesh_file: str, output_dir: str):
        """
        初始化转换器
        
        参数:
            mesh_file: GMsh网格文件路径
            output_dir: 输出目录
        """
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio不可用，请安装meshio: pip install meshio")
        
        self.mesh_file = mesh_file
        self.output_dir = output_dir
        self.prefix = Path(mesh_file).stem
    
    def convert(self):
        """
        转换网格格式
        
        返回:
            (mesh, boundaries_mf, association_table) 元组
        """
        # 读取GMsh网格
        msh = meshio.read(self.mesh_file)
        
        # 导出域
        self._export_domain(msh)
        
        # 导出边界
        self._export_boundaries(msh)
        
        # 导出关联表
        association_table = self._export_association_table(msh)
        
        # 导入为FEniCS网格
        return self._import_fenics_mesh(association_table)
    
    def _export_domain(self, msh):
        """导出域到XDMF格式"""
        cell_type = "triangle"
        
        # 获取三角形单元
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            raise ValueError("未找到三角形单元")
        
        # 合并所有三角形单元数据
        data = np.concatenate([cell.data for cell in cells])
        domain_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]
        
        # 设置单元数据（子域标记）
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
            points=msh.points[:, :2],  # 只使用2D坐标
            cells=domain_cells,
            cell_data=cell_data
        )
        
        # 保存域网格
        domain_file = f"{self.output_dir}/{self.prefix}_domain.xdmf"
        meshio.write(domain_file, domain)
        logger.debug(f"域网格已保存: {domain_file}")
    
    def _export_boundaries(self, msh):
        """导出边界到XDMF格式"""
        cell_type = "line"
        
        # 获取线单元
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            logger.warning("未找到边界线单元")
            return
        
        # 合并所有线单元数据
        data = np.concatenate([cell.data for cell in cells])
        boundary_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]
        
        # 设置边界数据
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
            points=msh.points[:, :2],  # 只使用2D坐标
            cells=boundary_cells,
            cell_data=cell_data
        )
        
        # 保存边界网格
        boundaries_file = f"{self.output_dir}/{self.prefix}_boundaries.xdmf"
        meshio.write(boundaries_file, boundaries)
        logger.debug(f"边界网格已保存: {boundaries_file}")
    
    def _export_association_table(self, msh):
        """导出关联表到INI文件"""
        association_table = {}
        
        try:
            # 从GMsh物理组信息中提取关联表
            for label, arrays in msh.cell_sets.items():
                for i, array in enumerate(arrays):
                    if array.size != 0 and label != "gmsh:bounding_entities":
                        if i < len(msh.cell_data["gmsh:physical"]):
                            value = msh.cell_data["gmsh:physical"][i][0]
                            association_table[label] = int(value)
                        break
        except Exception as e:
            logger.warning(f"处理关联表时出错: {e}")
        
        # 保存关联表
        from configparser import ConfigParser
        config = ConfigParser()
        config["ASSOCIATION TABLE"] = {k: str(v) for k, v in association_table.items()}
        
        association_file = f"{self.output_dir}/{self.prefix}_association_table.ini"
        with open(association_file, 'w') as f:
            config.write(f)
        
        logger.debug(f"关联表已保存: {association_file}")
        return association_table
    
    def _import_fenics_mesh(self, association_table):
        """导入为FEniCS网格对象"""
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS不可用，无法导入网格")
        
        # 读取域网格
        mesh = Mesh()
        domain_file = f"{self.output_dir}/{self.prefix}_domain.xdmf"
        with XDMFFile(domain_file) as infile:
            infile.read(mesh)
        
        # 读取边界标记
        boundaries_mvc = MeshValueCollection("size_t", mesh, dim=1)
        boundaries_file = f"{self.output_dir}/{self.prefix}_boundaries.xdmf"
        
        try:
            with XDMFFile(boundaries_file) as infile:
                infile.read(boundaries_mvc, 'boundaries')
            boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
        except Exception as e:
            logger.warning(f"读取边界标记失败: {e}")
            # 创建空的边界标记
            boundaries_mf = MeshFunction("size_t", mesh, 1, 0)
        
        logger.info(f"FEniCS网格导入成功: {mesh.num_vertices()}个顶点, {mesh.num_cells()}个单元")
        
        return mesh, boundaries_mf, association_table


# 便捷函数
def create_eit_mesh(n_elec: int = 16, 
                   radius: float = 1.0, 
                   refinement: int = 6,
                   electrode_coverage: float = 0.5,
                   output_dir: str = None) -> object:
    """
    便捷函数：创建标准EIT网格
    
    参数:
        n_elec: 电极数量
        radius: 圆形域半径
        refinement: 网格细化级别
        electrode_coverage: 电极覆盖率
        output_dir: 输出目录
        
    返回:
        FEniCS网格对象
    """
    # 创建配置
    mesh_config = OptimizedMeshConfig(
        radius=radius,
        refinement=refinement,
        electrode_vertices=6,
        gap_vertices=1
    )
    
    electrode_config = ElectrodePosition(
        L=n_elec,
        coverage=electrode_coverage,
        rotation=0.0,
        anticlockwise=True
    )
    
    # 生成网格
    generator = OptimizedMeshGenerator(mesh_config, electrode_config)
    return generator.generate(output_dir=Path(output_dir) if output_dir else None)


def load_or_create_mesh(mesh_dir: str = "eit_meshes", 
                       mesh_name: str = None,
                       n_elec: int = 16,
                       **kwargs) -> object:
    """
    加载现有网格或创建新网格
    
    参数:
        mesh_dir: 网格目录
        mesh_name: 网格名称（如果为None则创建新网格）
        n_elec: 电极数量
        **kwargs: 传递给create_eit_mesh的其他参数
        
    返回:
        网格对象
    """
    if mesh_name is not None:
        # 尝试加载现有网格
        try:
            from .mesh_loader import MeshLoader
            loader = MeshLoader(mesh_dir)
            return loader.load_fenics_mesh(mesh_name)
        except Exception as e:
            logger.warning(f"加载网格失败: {e}, 将创建新网格")
    
    # 创建新网格
    return create_eit_mesh(n_elec=n_elec, output_dir=mesh_dir, **kwargs)