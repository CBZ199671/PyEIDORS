"""网格加载器 - 支持多种网格格式的加载功能"""

import numpy as np
import configparser
from pathlib import Path
import h5py
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# 检查FEniCS可用性
try:
    from fenics import Mesh, MeshFunction, HDF5File, XDMFFile
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS不可用，无法加载FEniCS网格格式")


class MeshLoader:
    """网格加载器类 - 支持加载现有的网格文件"""
    
    def __init__(self, mesh_dir: str = "eit_meshes"):
        """
        初始化网格加载器
        
        参数:
            mesh_dir: 网格文件目录
        """
        self.mesh_dir = Path(mesh_dir)
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"网格目录不存在: {mesh_dir}")
    
    def load_fenics_mesh(self, mesh_name: str = "mesh_506999") -> object:
        """
        加载FEniCS格式的网格
        
        参数:
            mesh_name: 网格名称（不包含扩展名）
            
        返回:
            包含网格、边界信息和关联表的网格对象
        """
        if not FENICS_AVAILABLE:
            raise ImportError("FEniCS不可用，无法加载网格")
        
        # 构建文件路径
        domain_file = self.mesh_dir / f"{mesh_name}_domain.h5"
        boundaries_file = self.mesh_dir / f"{mesh_name}_boundaries.h5" 
        association_file = self.mesh_dir / f"{mesh_name}_association_table.ini"
        
        # 检查文件是否存在
        for file_path in [domain_file, boundaries_file, association_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"网格文件不存在: {file_path}")
        
        logger.info(f"加载网格: {mesh_name}")
        
        # 加载主网格
        mesh = Mesh()
        with HDF5File(mesh.mpi_comm(), str(domain_file), "r") as hdf:
            hdf.read(mesh, "/mesh", False)
        
        # 加载边界标记
        boundaries_mf = MeshFunction("size_t", mesh, 1)
        with HDF5File(mesh.mpi_comm(), str(boundaries_file), "r") as hdf:
            hdf.read(boundaries_mf, "/boundaries")
        
        # 加载关联表
        association_table = self._load_association_table(association_file)
        
        # 创建扩展的网格对象
        enhanced_mesh = self._create_enhanced_mesh(mesh, boundaries_mf, association_table)
        
        logger.info(f"网格加载成功 - 节点数: {mesh.num_vertices()}, 单元数: {mesh.num_cells()}")
        
        return enhanced_mesh
    
    def _load_association_table(self, file_path: Path) -> Dict[int, int]:
        """加载关联表"""
        config = configparser.ConfigParser()
        config.read(file_path)
        
        association_table = {}
        if 'boundary_ids' in config:
            for key, value in config['boundary_ids'].items():
                association_table[int(key)] = int(value)
        
        return association_table
    
    def _create_enhanced_mesh(self, mesh, boundaries_mf, association_table) -> object:
        """创建增强的网格对象"""
        
        class EnhancedMesh:
            """增强的网格对象，包含所有必要的EIT属性"""
            
            def __init__(self, mesh, boundaries_mf, association_table):
                # 复制所有原始网格的属性和方法
                for attr in dir(mesh):
                    if not attr.startswith('_'):
                        try:
                            setattr(self, attr, getattr(mesh, attr))
                        except AttributeError:
                            pass
                
                # EIT特定属性
                self.boundaries_mf = boundaries_mf
                self.association_table = association_table
                
                # 默认几何参数
                self.radius = 1.0
                self.vertex_elec = []
                
                # 推断电极数量
                self.n_electrodes = len([k for k in association_table.keys() if k >= 2])
                
                # 计算网格统计信息
                self._compute_mesh_stats()
            
            def _compute_mesh_stats(self):
                """计算网格统计信息"""
                coords = mesh.coordinates()
                self.center = np.mean(coords, axis=0)
                self.bbox_min = np.min(coords, axis=0)
                self.bbox_max = np.max(coords, axis=0)
                
                # 估算半径
                distances = np.linalg.norm(coords - self.center, axis=1)
                self.radius = np.max(distances)
            
            def get_info(self) -> Dict[str, Any]:
                """获取网格信息"""
                return {
                    'num_vertices': mesh.num_vertices(),
                    'num_cells': mesh.num_cells(),
                    'num_electrodes': self.n_electrodes,
                    'radius': self.radius,
                    'center': self.center.tolist(),
                    'bbox': [self.bbox_min.tolist(), self.bbox_max.tolist()],
                    'association_table': self.association_table
                }
        
        return EnhancedMesh(mesh, boundaries_mf, association_table)
    
    def load_numpy_mesh(self, file_path: str) -> np.ndarray:
        """
        加载numpy格式的网格数据
        
        参数:
            file_path: numpy文件路径
            
        返回:
            网格数据数组
        """
        mesh_file = self.mesh_dir / file_path
        if not mesh_file.exists():
            raise FileNotFoundError(f"文件不存在: {mesh_file}")
        
        return np.load(mesh_file)
    
    def list_available_meshes(self) -> Dict[str, list]:
        """列出可用的网格文件"""
        meshes = {
            'fenics_h5': [],
            'xdmf': [],
            'msh': [],
            'numpy': []
        }
        
        for file_path in self.mesh_dir.glob("*"):
            if file_path.suffix == '.h5':
                base_name = file_path.stem
                if base_name.endswith('_domain'):
                    meshes['fenics_h5'].append(base_name[:-7])  # 移除 '_domain'
            elif file_path.suffix == '.xdmf':
                meshes['xdmf'].append(file_path.stem)
            elif file_path.suffix == '.msh':
                meshes['msh'].append(file_path.stem)
            elif file_path.suffix == '.npy':
                meshes['numpy'].append(file_path.stem)
        
        return meshes
    
    def get_default_mesh(self) -> object:
        """获取默认网格（如果可用）"""
        available = self.list_available_meshes()
        
        # 优先使用FEniCS H5格式
        if available['fenics_h5']:
            mesh_name = available['fenics_h5'][0]
            logger.info(f"使用默认FEniCS网格: {mesh_name}")
            return self.load_fenics_mesh(mesh_name)
        
        raise FileNotFoundError("没有找到可用的默认网格文件")


def create_simple_mesh_loader(mesh_dir: str = "eit_meshes") -> MeshLoader:
    """创建简单的网格加载器实例"""
    return MeshLoader(mesh_dir)