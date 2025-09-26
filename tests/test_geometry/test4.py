"""
标准EIT高斯牛顿重建器 - 最小化实现
基于标准符号约定的完整EIT正逆问题流程
（完整的EIT正逆问题，仿真和实测数据都可以用）
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# FEniCS相关库
from fenics import *
from dolfin import XDMFFile, Mesh, MeshValueCollection
from dolfin.cpp.mesh import MeshFunctionSizet
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Any
import time

# 导入网格生成相关模块
import gmsh
from math import pi, cos, sin
import meshio
from configparser import ConfigParser

# ==================== 数据结构定义 ====================

@dataclass
class PatternConfig:
    """激励和测量模式配置"""
    n_elec: int
    n_rings: int = 1
    stim_pattern: Union[str, List[int]] = '{ad}'
    meas_pattern: Union[str, List[int]] = '{ad}'
    amplitude: float = 1.0
    use_meas_current: bool = False
    use_meas_current_next: int = 0
    rotate_meas: bool = True

@dataclass
class EITData:
    """EIT数据容器"""
    meas: np.ndarray
    stim_pattern: np.ndarray
    n_elec: int
    n_stim: int
    n_meas: int
    type: str = 'real'

@dataclass
class EITImage:
    """EIT图像容器"""
    elem_data: np.ndarray
    fwd_model: Any
    type: str = 'conductivity'
    name: str = ''
    
    def get_conductivity(self) -> np.ndarray:
        if self.type == 'resistivity':
            return 1.0 / self.elem_data
        return self.elem_data

@dataclass
class ElectrodePosition:
    """电极位置配置"""
    L: int
    coverage: float = 0.5
    rotation: float = 0.0
    anticlockwise: bool = True
    
    def __post_init__(self):
        if not isinstance(self.L, int) or self.L <= 0:
            raise ValueError("电极数量必须是正整数")
        if not 0 < self.coverage <= 1:
            raise ValueError("覆盖率必须在(0, 1]范围内")
    
    @property
    def positions(self) -> List[Tuple[float, float]]:
        electrode_size = 2 * pi / self.L * self.coverage
        gap_size = 2 * pi / self.L * (1 - self.coverage)
        
        positions = []
        for i in range(self.L):
            start = electrode_size * i + gap_size * i + self.rotation
            end = electrode_size * (i + 1) + gap_size * i + self.rotation
            positions.append((start, end))
        
        if not self.anticlockwise:
            positions[1:] = positions[1:][::-1]
        
        return positions

@dataclass
class MeshConfig:
    """网格配置参数"""
    radius: float = 1.0
    refinement: int = 8
    electrode_vertices: int = 6
    gap_vertices: int = 1
    
    @property
    def mesh_size(self) -> float:
        return self.radius / (self.refinement * 2)

# ==================== 激励测量模式管理器 ====================

class StimMeasPatternManager:
    """激励和测量模式管理器"""
    
    def __init__(self, config: PatternConfig):
        self.config = config
        self.n_elec = config.n_elec
        self.n_rings = config.n_rings
        self.tn_elec = self.n_elec * self.n_rings
        
        self._parse_patterns()
        self._generate_patterns()
        self._compute_measurement_selector()
    
    def _parse_patterns(self):
        # 激励模式解析
        if isinstance(self.config.stim_pattern, str):
            if self.config.stim_pattern == '{ad}':
                self.inj_electrodes = [0, 1]
            elif self.config.stim_pattern == '{op}':
                self.inj_electrodes = [0, self.n_elec // 2]
            else:
                raise ValueError(f"Unknown stimulation pattern: {self.config.stim_pattern}")
        else:
            self.inj_electrodes = self.config.stim_pattern
            
        self.inj_weights = np.array([-1, 1]) if len(self.inj_electrodes) == 2 else np.array([1])
        
        # 测量模式解析
        if isinstance(self.config.meas_pattern, str):
            if self.config.meas_pattern == '{ad}':
                self.meas_electrodes = [0, 1]
            elif self.config.meas_pattern == '{op}':
                self.meas_electrodes = [0, self.n_elec // 2]
            else:
                raise ValueError(f"Unknown measurement pattern: {self.config.meas_pattern}")
        else:
            self.meas_electrodes = self.config.meas_pattern
            
        self.meas_weights = np.array([1, -1]) if len(self.meas_electrodes) == 2 else np.array([1])
    
    def _generate_patterns(self):
        self.stim_matrix = []
        self.meas_matrices = []
        self.meas_start_indices = []
        self.n_meas_total = 0
        self.n_meas_per_stim = []
        
        for ring in range(self.n_rings):
            for elec in range(self.n_elec):
                # 激励向量
                stim_vec = np.zeros(self.tn_elec)
                for i, inj_elec in enumerate(self.inj_electrodes):
                    idx = (inj_elec + elec) % self.n_elec + ring * self.n_elec
                    stim_vec[idx] = self.config.amplitude * self.inj_weights[i]
                
                # 测量矩阵
                meas_mat = self._make_meas_matrix(elec, ring)
                
                if not self.config.use_meas_current:
                    meas_mat = self._filter_measurements(meas_mat, elec, ring)
                
                if meas_mat.shape[0] > 0:
                    self.stim_matrix.append(stim_vec)
                    self.meas_matrices.append(meas_mat)
                    self.meas_start_indices.append(self.n_meas_total)
                    self.n_meas_per_stim.append(meas_mat.shape[0])
                    self.n_meas_total += meas_mat.shape[0]
        
        self.stim_matrix = np.array(self.stim_matrix)
        self.n_stim = len(self.stim_matrix)
    
    def _compute_measurement_selector(self):
        if self.config.use_meas_current:
            self.meas_selector = np.ones(self.n_elec * self.n_stim, dtype=bool)
            return

        selector = []
        for i in range(self.n_stim):
            elec = i % self.n_elec
            ring = i // self.n_elec
            
            full_meas_mat = self._make_meas_matrix(elec, ring)
            filtered_meas_mat = self.meas_matrices[i]
            
            full_set_hash = self._create_meas_hash(full_meas_mat)
            filtered_set_hash = self._create_meas_hash(filtered_meas_mat)
            
            frame_selector = np.isin(full_set_hash, filtered_set_hash)
            selector.append(frame_selector)
        
        self.meas_selector = np.concatenate(selector)
    
    def _create_meas_hash(self, meas_mat: np.ndarray) -> np.ndarray:
        if meas_mat.size == 0:
            return np.array([])
        
        pos_indices = np.argmax(meas_mat > 0, axis=1)
        neg_indices = np.argmax(meas_mat < 0, axis=1)
        
        pos_mask = np.any(meas_mat > 0, axis=1)
        neg_mask = np.any(meas_mat < 0, axis=1)
        
        hash_vals = (pos_indices * pos_mask) * 1e7 + (neg_indices * neg_mask)
        return hash_vals
    
    def _make_meas_matrix(self, elec: int, ring: int) -> np.ndarray:
        meas_list = []
        offset = elec if self.config.rotate_meas else 0
        
        for meas_idx in range(self.tn_elec):
            meas_vec = np.zeros(self.tn_elec)
            within_ring = meas_idx % self.n_elec
            ring_offset = (meas_idx // self.n_elec) * self.n_elec
            
            for i, meas_elec in enumerate(self.meas_electrodes):
                idx = (meas_elec + within_ring + offset) % self.n_elec + ring_offset
                meas_vec[idx] = self.meas_weights[i]
            
            meas_list.append(meas_vec)
        
        return np.array(meas_list)
    
    def _filter_measurements(self, meas_mat: np.ndarray, elec: int, ring: int) -> np.ndarray:
        stim_indices = []
        for inj_elec in self.inj_electrodes:
            idx = (inj_elec + elec) % self.n_elec + ring * self.n_elec
            stim_indices.append(idx)
        
        if self.config.use_meas_current_next > 0:
            extended = []
            for idx in stim_indices:
                base = idx % self.n_elec
                ring_base = idx - base
                for offset in range(-self.config.use_meas_current_next, 
                                  self.config.use_meas_current_next + 1):
                    extended.append((base + offset) % self.n_elec + ring_base)
            stim_indices = list(set(extended))
        
        mask = ~np.any(meas_mat[:, stim_indices] != 0, axis=1)
        return meas_mat[mask]
    
    def get_stim_matrix(self) -> np.ndarray:
        return self.stim_matrix
    
    def apply_meas_pattern(self, electrode_voltages: np.ndarray) -> np.ndarray:
        measurements = np.zeros(self.n_meas_total)
        
        for i, (start_idx, meas_mat) in enumerate(zip(self.meas_start_indices, self.meas_matrices)):
            n_meas = meas_mat.shape[0]
            measurements[start_idx:start_idx + n_meas] = meas_mat @ electrode_voltages[i]
        
        return measurements

# ==================== 网格生成器 ====================

class MeshGenerator:
    """网格生成器"""
    
    def __init__(self, config: MeshConfig, electrodes: ElectrodePosition):
        self.config = config
        self.electrodes = electrodes
        self.mesh_data = {}
    
    def generate(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        mesh_file = output_dir / f"mesh_{int(time.time() * 1e6) % 1000000}.msh"
        
        gmsh.initialize()
        gmsh.model.add("EIT_Mesh")
        
        try:
            self._create_geometry()
            self._set_physical_groups()
            self._generate_mesh()
            gmsh.write(str(mesh_file))
            self._extract_electrode_vertices()
        finally:
            gmsh.finalize()
        
        return self._convert_to_fenics(mesh_file, output_dir)
    
    def _create_geometry(self):
        positions = self.electrodes.positions
        n_in = self.config.electrode_vertices
        n_out = self.config.gap_vertices
        r = self.config.radius
        
        boundary_points = []
        electrode_ranges = []
        
        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)
            
            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)
            
            electrode_ranges.append((start_idx, len(boundary_points) - 1))
            
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
        
        lines = []
        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            line = gmsh.model.occ.addLine(boundary_points[i], boundary_points[next_i])
            lines.append(line)
        
        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])
        
        mesh_size_center = 0.095
        cp_distance = 0.1
        center_points = [
            gmsh.model.occ.addPoint(x, y, 0.0, meshSize=mesh_size_center)
            for x, y in [(-cp_distance, cp_distance), (cp_distance, cp_distance),
                         (-cp_distance, -cp_distance), (cp_distance, -cp_distance)]
        ]
        
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.embed(0, center_points, 2, surface)
        
        self.mesh_data['boundary_points'] = boundary_points
        self.mesh_data['electrode_ranges'] = electrode_ranges
        self.mesh_data['lines'] = lines
        self.mesh_data['surface'] = surface
    
    def _set_physical_groups(self):
        surface = self.mesh_data['surface']
        lines = self.mesh_data['lines']
        electrode_ranges = self.mesh_data['electrode_ranges']
        
        gmsh.model.addPhysicalGroup(2, [surface], 1, name="domain")
        
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
        
        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            gmsh.model.addPhysicalGroup(1, gap_lines, self.electrodes.L + 2, name="gaps")
    
    def _generate_mesh(self):
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
        gmsh.model.mesh.generate(2)
    
    def _extract_electrode_vertices(self):
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
        converter = MeshConverter(str(mesh_file), str(output_dir))
        mesh, boundaries_mf, association_table = converter.convert()
        
        mesh.radius = self.config.radius
        mesh.vertex_elec = self.mesh_data.get('electrode_vertices', [])
        mesh.electrodes = self.electrodes
        mesh.boundaries_mf = boundaries_mf
        mesh.association_table = association_table
        
        return mesh

class MeshConverter:
    """网格格式转换器"""
    
    def __init__(self, mesh_file: str, output_dir: str):
        self.mesh_file = mesh_file
        self.output_dir = output_dir
        self.prefix = Path(mesh_file).stem
    
    def convert(self):
        msh = meshio.read(self.mesh_file)
        
        self._export_domain(msh)
        self._export_boundaries(msh)
        association_table = self._export_association_table(msh)
        
        return self._import_fenics_mesh(association_table)
    
    def _export_domain(self, msh):
        cell_type = "triangle"
        
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            raise ValueError("未找到域物理组")
        
        data = np.concatenate([cell.data for cell in cells])
        domain_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]
        
        cell_data = {
            "subdomains": [
                np.concatenate([
                    msh.cell_data["gmsh:physical"][i]
                    for i, cell in enumerate(msh.cells)
                    if cell.type == cell_type
                ])
            ]
        }
        
        domain = meshio.Mesh(
            points=msh.points[:, :2],
            cells=domain_cells,
            cell_data=cell_data
        )
        
        meshio.write(f"{self.output_dir}/{self.prefix}_domain.xdmf", domain)
    
    def _export_boundaries(self, msh):
        cell_type = "line"
        
        cells = [cell for cell in msh.cells if cell.type == cell_type]
        if not cells:
            return
        
        data = np.concatenate([cell.data for cell in cells])
        boundary_cells = [meshio.CellBlock(cell_type=cell_type, data=data)]
        
        cell_data = {
            "boundaries": [
                np.concatenate([
                    msh.cell_data["gmsh:physical"][i]
                    for i, cell in enumerate(msh.cells)
                    if cell.type == cell_type
                ])
            ]
        }
        
        boundaries = meshio.Mesh(
            points=msh.points[:, :2],
            cells=boundary_cells,
            cell_data=cell_data
        )
        
        meshio.write(f"{self.output_dir}/{self.prefix}_boundaries.xdmf", boundaries)
    
    def _export_association_table(self, msh):
        association_table = {}
        
        try:
            for label, arrays in msh.cell_sets.items():
                for i, array in enumerate(arrays):
                    if array.size != 0 and label != "gmsh:bounding_entities":
                        if i < len(msh.cell_data["gmsh:physical"]):
                            value = msh.cell_data["gmsh:physical"][i][0]
                            association_table[label] = int(value)
                        break
        except Exception as e:
            print(f"处理关联表时出错: {e}")
        
        config = ConfigParser()
        config["ASSOCIATION TABLE"] = {k: str(v) for k, v in association_table.items()}
        
        with open(f"{self.output_dir}/{self.prefix}_association_table.ini", 'w') as f:
            config.write(f)
        
        return association_table
    
    def _import_fenics_mesh(self, association_table):
        mesh = Mesh()
        with XDMFFile(f"{self.output_dir}/{self.prefix}_domain.xdmf") as infile:
            infile.read(mesh)
        
        boundaries_mvc = MeshValueCollection("size_t", mesh, dim=1)
        with XDMFFile(f"{self.output_dir}/{self.prefix}_boundaries.xdmf") as infile:
            infile.read(boundaries_mvc, 'boundaries')
        boundaries_mf = MeshFunctionSizet(mesh, boundaries_mvc)
        
        return mesh, boundaries_mf, association_table

# ==================== EIT前向模型 ====================

class EITForwardModel:
    """EIT前向模型 - 基于完全电极模型"""
    
    def __init__(self, n_elec: int, pattern_config: PatternConfig, z: np.ndarray, mesh):
        self.n_elec = n_elec
        self.z = np.array(z)
        self.mesh = mesh
        
        # 创建激励测量模式管理器
        self.pattern_manager = StimMeasPatternManager(pattern_config)
        
        # 设置边界条件
        self.boundaries_mf = mesh.boundaries_mf
        self.association_table = mesh.association_table
        self.ds_electrodes = Measure("ds", domain=mesh, subdomain_data=self.boundaries_mf)
        
        # 计算电极长度
        self.electrode_len = assemble(1 * self.ds_electrodes(2))
        
        # 创建函数空间
        self.V = FunctionSpace(mesh, "Lagrange", 1)  # 电位
        self.V_sigma = FunctionSpace(mesh, "DG", 0)  # 导电率
        
        # 获取自由度数量
        u_sol = Function(self.V)
        self.dofs = u_sol.vector().size()
        
        # 定义变分形式
        self.u = TrialFunction(self.V)
        self.phi = TestFunction(self.V)
        
        # 组装系统矩阵
        self.M = self._assemble_electrode_matrix()
        
        # 设置雅可比计算
        self._setup_jacobian_computation()
    
    def _assemble_electrode_matrix(self):
        """组装电极相关的系统矩阵"""
        b = 0
        for i in range(self.n_elec):
            electrode_tag = i + 2
            if electrode_tag in self.association_table.values():
                b += 1 / self.z[i] * inner(self.u, self.phi) * self.ds_electrodes(electrode_tag)

        B = assemble(b)
        
        row, col, val = as_backend_type(B).mat().getValuesCSR()
        M = csr_matrix((val, col, row))
        
        M.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        M_lil = lil_matrix(M)

        for i in range(self.n_elec):
            electrode_tag = i + 2
            if electrode_tag in self.association_table.values():
                c = -1 / self.z[i] * self.phi * self.ds_electrodes(electrode_tag)
                C_i = assemble(c).get_local()
                M_lil[self.dofs + i, :self.dofs] = C_i
                M_lil[:self.dofs, self.dofs + i] = C_i
                
                M_lil[self.dofs + i, self.dofs + i] = 1 / self.z[i] * self.electrode_len
                M_lil[self.dofs + self.n_elec, self.dofs + i] = 1
                M_lil[self.dofs + i, self.dofs + self.n_elec] = 1

        return csr_matrix(M_lil)
    
    def _setup_jacobian_computation(self):
        """设置雅可比计算所需的属性"""
        self.L = self.n_elec
        self.omega = self.mesh
        
        self.DG0 = FunctionSpace(self.mesh, "DG", 0)
        self.test_func_dg = TestFunction(self.DG0)
        
        self.cell_areas = assemble(self.test_func_dg * dx).get_local()
        self.n_elements = len(self.cell_areas)
        
        self.Q_DG = VectorFunctionSpace(self.mesh, "DG", 0)
    
    def create_full_matrix(self, sigma: Function):
        """构建包含导电率的完整系统矩阵"""
        a = inner(sigma * grad(self.u), grad(self.phi)) * dx
        A = assemble(a)
        
        row, col, val = as_backend_type(A).mat().getValuesCSR()
        scipy_A = csr_matrix((val, col, row))
        scipy_A.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)

        return scipy_A + self.M
    
    def forward_solve(self, sigma: Function, current_patterns=None):
        """前向求解"""
        if current_patterns is None:
            # 使用默认激励模式
            M_complete = self.create_full_matrix(sigma)
            solver = factorized(M_complete)
            
            u_all = []
            U_all = np.zeros((self.pattern_manager.n_stim, self.n_elec))
            
            stim_matrix = self.pattern_manager.stim_matrix
            
            for i in range(self.pattern_manager.n_stim):
                rhs = np.zeros(self.dofs + self.n_elec + 1)
                for j in range(self.n_elec):
                    rhs[self.dofs + j] = stim_matrix[i, j]
                
                sol = solver(rhs)
                u_all.append(sol[:self.dofs])
                U_all[i, :] = sol[self.dofs:-1]
            
            return u_all, U_all
        else:
            # 使用指定的电流模式
            M_complete = self.create_full_matrix(sigma)
            solver = factorized(M_complete)
            
            n_patterns = current_patterns.shape[1]
            u_all = []
            U_all = np.zeros((n_patterns, self.n_elec))
            
            for i in range(n_patterns):
                rhs = np.zeros(self.dofs + self.n_elec + 1)
                for j in range(self.n_elec):
                    rhs[self.dofs + j] = current_patterns[j, i]
                
                sol = solver(rhs)
                u_all.append(sol[:self.dofs])
                U_all[i, :] = sol[self.dofs:-1]
            
            return u_all, U_all
    
    def fwd_solve(self, img: EITImage):
        """前向求解接口"""
        sigma = Function(self.V_sigma)
        sigma.vector()[:] = img.get_conductivity()
        
        u_all, U_all = self.forward_solve(sigma)
        
        # 应用测量模式
        meas = self.pattern_manager.apply_meas_pattern(U_all)
        
        # 创建EIT数据对象
        data = EITData(
            meas=meas,
            stim_pattern=self.pattern_manager.stim_matrix,
            n_elec=self.n_elec,
            n_stim=self.pattern_manager.n_stim,
            n_meas=self.pattern_manager.n_meas_total,
            type='simulated'
        )
        
        return data, U_all
    
    def calc_measurement_jacobian(self, sigma: Function, u_all=None) -> np.ndarray:
        """计算测量雅可比矩阵 - 使用验证过的方法"""
        if u_all is None:
            u_all, _ = self.forward_solve(sigma)

        # 构造用于雅可比计算的电流模式
        I2_all = []
        for i in range(self.L):
            I2 = np.zeros(self.L)
            I2[i] = 1
            I2_all.append(I2)
        I2_all = np.array(I2_all).T

        bu_all, _ = self.forward_solve(sigma, I2_all)

        # 创建梯度函数空间
        Q_DG = VectorFunctionSpace(self.omega, "DG", 0)
        DG0 = FunctionSpace(self.omega, "DG", 0)

        v = TestFunction(DG0)
        cell_area = assemble(v * dx).get_local()

        # 计算u的梯度
        list_grad_u = []
        for u in u_all:
            u_fun = Function(self.V)
            u_fun.vector()[:] = u

            grad_u = project(grad(u_fun), Q_DG)
            grad_u_vec = grad_u.vector().get_local().reshape(-1, 2)
            list_grad_u.append(grad_u_vec)

        # 计算bu的梯度
        list_grad_bu = []
        for bu in bu_all:
            bu_fun = Function(self.V)
            bu_fun.vector()[:] = bu

            grad_bu = project(grad(bu_fun), Q_DG)
            grad_bu_vec = grad_bu.vector().get_local().reshape(-1, 2)
            list_grad_bu.append(grad_bu_vec)

        # 组装雅可比矩阵
        Jacobian_all = None
        for h in range(len(u_all)):
            derivative = []
            for j in range(self.L):
                row = np.sum(list_grad_bu[j] * list_grad_u[h], axis=1) * cell_area
                derivative.append(row)

            Jacobian = np.array(derivative)
            if h == 0:
                Jacobian_all = Jacobian
            else:
                Jacobian_all = np.concatenate((Jacobian_all, Jacobian), axis=0)

        # 转换为测量雅可比矩阵
        measurement_jacobian = self._convert_electrode_to_measurement_jacobian(Jacobian_all)
        
        return measurement_jacobian
    
    def _convert_electrode_to_measurement_jacobian(self, electrode_jacobian: np.ndarray) -> np.ndarray:
        """将电极电势雅可比转换为测量雅可比"""
        n_stim = self.pattern_manager.n_stim
        n_elec = self.n_elec
        n_elements = electrode_jacobian.shape[1]
        
        expected_rows = n_elec * n_stim
        if electrode_jacobian.shape[0] != expected_rows:
            raise ValueError(f"电极雅可比矩阵行数不匹配: 期望{expected_rows}, 实际{electrode_jacobian.shape[0]}")
        
        measurement_jacobian_blocks = []
        
        for stim_idx in range(n_stim):
            elec_start = stim_idx * n_elec
            elec_end = (stim_idx + 1) * n_elec
            electrode_jac_for_stim = electrode_jacobian[elec_start:elec_end, :]
            
            meas_matrix = self.pattern_manager.meas_matrices[stim_idx]
            
            meas_jacobian_for_stim = meas_matrix @ electrode_jac_for_stim
            
            measurement_jacobian_blocks.append(meas_jacobian_for_stim)
        
        measurement_jacobian = np.vstack(measurement_jacobian_blocks)
        
        return measurement_jacobian

# ==================== 标准高斯牛顿重建器 ====================

class StandardGaussNewtonReconstructor:
    """PyTorch加速的高斯牛顿EIT重建器"""
    
    def __init__(self, 
                 fwd_model,
                 max_iterations: int = 15,
                 convergence_tol: float = 1e-4,
                 regularization_param: float = 0.01,
                 line_search_steps: int = 8,
                 clip_values: Tuple[float, float] = (1e-6, 10.0),
                 device: str = 'cuda:0',
                 verbose: bool = True):
        """
        初始化PyTorch加速的高斯牛顿重建器
        
        参数:
            device: 计算设备 ('cuda:0', 'cuda:1', 'cpu')
            其他参数与标准版本相同
        """
        self.fwd_model = fwd_model
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.regularization_param = regularization_param
        self.line_search_steps = line_search_steps
        self.clip_values = clip_values
        self.verbose = verbose
        
        # 设置计算设备
        if device.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device(device)
            if self.verbose:
                print(f"使用GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device('cpu')
            if self.verbose:
                print("使用CPU计算")
        
        self.n_elements = len(Function(fwd_model.V_sigma).vector()[:])
        self.n_measurements = fwd_model.pattern_manager.n_meas_total
        
        # 预计算正则化矩阵并转换为torch张量
        self.R_torch = None
        
        if self.verbose:
            print(f"PyTorch加速高斯牛顿重建器初始化:")
            print(f"  元素数: {self.n_elements}")
            print(f"  测量数: {self.n_measurements}")
            print(f"  最大迭代数: {max_iterations}")
            print(f"  正则化参数: {regularization_param}")
            print(f"  计算设备: {self.device}")
    
    def reconstruct(self, 
                   measured_data: Union[object, np.ndarray],
                   initial_conductivity: float = 1.0,
                   regularization_matrix: Optional[np.ndarray] = None):
        """
        执行PyTorch加速的高斯牛顿重建
        """
        
        # 处理输入数据并转换为torch张量
        if hasattr(measured_data, 'meas'):
            meas_vector = measured_data.meas
        else:
            meas_vector = measured_data.flatten()
        
        if len(meas_vector) != self.n_measurements:
            raise ValueError(f"测量数据长度不匹配: {len(meas_vector)} vs {self.n_measurements}")
        
        # 转换测量数据到GPU
        meas_torch = torch.from_numpy(meas_vector).float().to(self.device)
        
        # 设置正则化矩阵
        if regularization_matrix is None:
            if self.R_torch is None:
                R_np = self._create_smoothness_matrix()
                self.R_torch = torch.from_numpy(R_np).float().to(self.device)
        else:
            self.R_torch = torch.from_numpy(regularization_matrix).float().to(self.device)
        
        # 初始化导电率分布
        sigma_current = Function(self.fwd_model.V_sigma)
        sigma_current.vector()[:] = initial_conductivity
        
        # 记录收敛历史
        residual_history = []
        sigma_change_history = []
        
        # 预分配torch张量以减少内存分配开销
        sigma_torch = torch.zeros(self.n_elements, device=self.device, dtype=torch.float32)
        
        if self.verbose:
            print(f"\n开始PyTorch加速高斯牛顿重建...")
        
        with tqdm(total=self.max_iterations, disable=not self.verbose) as pbar:
            for iteration in range(self.max_iterations):
                
                # 1. 前向求解 (仍需在CPU上进行，因为fwd_model基于FEniCS)
                img_current = self._create_eit_image(sigma_current.vector()[:])
                data_simulated, _ = self.fwd_model.fwd_solve(img_current)
                
                # 2. 转换到GPU并计算残差
                data_sim_torch = torch.from_numpy(data_simulated.meas).float().to(self.device)
                residual_torch = data_sim_torch - meas_torch
                residual_norm = torch.norm(residual_torch).item()
                residual_history.append(residual_norm)
                
                # 3. 计算雅可比矩阵并转换到GPU
                measurement_jacobian_np = -self.fwd_model.calc_measurement_jacobian(sigma_current)
                J_torch = torch.from_numpy(measurement_jacobian_np).float().to(self.device)
                
                # 4. GPU上构建高斯牛顿系统: (J^T J + λR) Δσ = -J^T r
                JTJ = torch.mm(J_torch.t(), J_torch)
                JTr = torch.mv(J_torch.t(), residual_torch)
                
                A = JTJ + self.regularization_param * self.R_torch
                b = -JTr  # 标准形式的负号
                
                # 5. GPU上求解线性系统
                try:
                    delta_sigma_torch = torch.linalg.solve(A, b)
                except RuntimeError:
                    # 增加正则化重试
                    A_regularized = JTJ + (self.regularization_param * 10) * self.R_torch
                    delta_sigma_torch = torch.linalg.solve(A_regularized, b)
                
                # 6. 线搜索确定最优步长
                optimal_step_size = self._line_search_torch(
                    sigma_current, delta_sigma_torch, meas_torch, residual_norm
                )
                
                # 7. 更新导电率分布
                sigma_old_values = sigma_current.vector()[:].copy()
                delta_sigma_np = delta_sigma_torch.cpu().numpy()
                sigma_current.vector()[:] += optimal_step_size * delta_sigma_np
                
                # 8. 应用约束
                if self.clip_values is not None:
                    sigma_current.vector()[:] = np.clip(
                        sigma_current.vector()[:], self.clip_values[0], self.clip_values[1]
                    )
                
                # 9. 计算相对变化 (在GPU上进行)
                sigma_new_torch = torch.from_numpy(sigma_current.vector()[:]).float().to(self.device)
                sigma_old_torch = torch.from_numpy(sigma_old_values).float().to(self.device)
                
                sigma_change = torch.norm(sigma_new_torch - sigma_old_torch).item()
                relative_change = sigma_change / (torch.norm(sigma_new_torch).item() + 1e-12)
                sigma_change_history.append(relative_change)
                
                # 10. 检查收敛
                if relative_change < self.convergence_tol:
                    if self.verbose:
                        print(f"\n收敛达到! 迭代 {iteration}, 相对变化: {relative_change:.2e}")
                    break
                
                # 更新进度条
                if self.verbose:
                    pbar.set_description(
                        f"残差: {residual_norm:.2e} | 相对变化: {relative_change:.2e} | 步长: {optimal_step_size:.3f}"
                    )
                    pbar.update(1)
        
        # 构建结果
        results = {
            'conductivity': sigma_current,
            'residual_history': residual_history,
            'sigma_change_history': sigma_change_history,
            'iterations': len(residual_history),
            'converged': relative_change < self.convergence_tol,
            'final_residual': residual_history[-1],
            'final_relative_change': relative_change
        }
        
        if self.verbose:
            print(f"\n重建完成:")
            print(f"  迭代次数: {results['iterations']}")
            print(f"  最终残差: {results['final_residual']:.2e}")
            print(f"  最终相对变化: {results['final_relative_change']:.2e}")
            print(f"  收敛状态: {'是' if results['converged'] else '否'}")
        
        return results
    
    def _line_search_torch(self, sigma_current, delta_sigma_torch, meas_target_torch, current_residual_norm):
        """PyTorch加速的线搜索算法"""
        step_candidates = torch.linspace(0.1, 1.0, self.line_search_steps, device=self.device)
        best_step = step_candidates[0].item()
        best_residual = float('inf')
        
        # 转换到CPU进行前向求解
        delta_sigma_np = delta_sigma_torch.cpu().numpy()
        meas_target_np = meas_target_torch.cpu().numpy()
        
        for step_size in step_candidates:
            try:
                # 测试该步长
                sigma_test = sigma_current.copy(deepcopy=True)
                sigma_test.vector()[:] += step_size.item() * delta_sigma_np
                
                # 应用约束
                if self.clip_values is not None:
                    sigma_test.vector()[:] = np.clip(
                        sigma_test.vector()[:], self.clip_values[0], self.clip_values[1]
                    )
                
                # 前向求解
                img_test = self._create_eit_image(sigma_test.vector()[:])
                data_test, _ = self.fwd_model.fwd_solve(img_test)
                
                # 转换到GPU计算残差
                data_test_torch = torch.from_numpy(data_test.meas).float().to(self.device)
                residual_torch = data_test_torch - meas_target_torch
                residual_norm = torch.norm(residual_torch).item()
                
                # 更新最佳步长
                if residual_norm < best_residual:
                    best_residual = residual_norm
                    best_step = step_size.item()
                    
            except Exception:
                continue
        
        return best_step
    
    def _create_eit_image(self, elem_data):
        """创建EIT图像对象的辅助函数"""
        # 这里需要根据你的EITImage类的具体实现进行调整
        try:
            # from your_eit_module import EITImage  # 替换为实际的导入
            return EITImage(elem_data=elem_data, fwd_model=self.fwd_model)
        except ImportError:
            # 如果没有EITImage类，创建一个简单的替代
            class SimpleEITImage:
                def __init__(self, elem_data, fwd_model):
                    self.elem_data = elem_data
                    self.fwd_model = fwd_model
            return SimpleEITImage(elem_data, self.fwd_model)
    
    def _create_smoothness_matrix(self) -> np.ndarray:
        """创建平滑性正则化矩阵"""
        mesh = self.fwd_model.mesh
        n_cells = mesh.num_cells()
        
        rows, cols, data = [], [], []
        row_idx = 0
        
        # 基于网格拓扑构建拉普拉斯算子
        for edge in edges(mesh):
            adjacent_cells = []
            for cell in cells(edge):
                adjacent_cells.append(cell.index())
            
            if len(adjacent_cells) == 2:
                cell1, cell2 = adjacent_cells
                rows.extend([row_idx, row_idx])
                cols.extend([cell1, cell2])
                data.extend([1.0, -1.0])
                row_idx += 1
        
        L = csr_matrix((data, (rows, cols)), shape=(row_idx, n_cells))
        return (L.T @ L).toarray()
    
    def get_memory_usage(self):
        """获取GPU内存使用情况"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(self.device) / 1024**3      # GB
            return f"GPU内存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB"
        else:
            return "使用CPU计算"
    
    def clear_cache(self):
        """清理GPU缓存"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.verbose:
                print("GPU缓存已清理")

# ==================== 合成数据生成 ====================

def create_synthetic_data(fwd_model: EITForwardModel,
                         inclusion_conductivity: float = 2.5,
                         background_conductivity: float = 1.0,
                         noise_level: float = 0.02,
                         center: Tuple[float, float] = (0.2, 0.2),
                         radius: float = 0.3):
    """创建合成EIT测试数据"""
    
    # 创建真实导电率分布
    sigma_true = Function(fwd_model.V_sigma)
    sigma_true.vector()[:] = background_conductivity
    
    # 添加圆形异常
    for cell in cells(fwd_model.mesh):
        cell_center = cell.midpoint()
        x, y = cell_center.x(), cell_center.y()
        if (x - center[0])**2 + (y - center[1])**2 < radius**2:
            sigma_true.vector()[cell.index()] = inclusion_conductivity
    
    # 生成清洁测量数据
    img_true = EITImage(elem_data=sigma_true.vector()[:], fwd_model=fwd_model)
    data_clean, _ = fwd_model.fwd_solve(img_true)
    
    # 添加高斯白噪声
    np.random.seed(42)  # 确保可重复性
    noise = noise_level * np.std(data_clean.meas) * np.random.randn(len(data_clean.meas))
    data_noisy = EITData(
        meas=data_clean.meas + noise,
        stim_pattern=data_clean.stim_pattern,
        n_elec=data_clean.n_elec,
        n_stim=data_clean.n_stim,
        n_meas=data_clean.n_meas,
        type='simulated_noisy'
    )
    
    snr_db = 20 * np.log10(np.std(data_clean.meas) / np.std(noise))
    
    return {
        'sigma_true': sigma_true,
        'data_clean': data_clean,
        'data_noisy': data_noisy,
        'noise': noise,
        'snr_db': snr_db
    }

# ==================== 可视化 ====================

def visualize_reconstruction_results(fwd_model: EITForwardModel,
                                   sigma_true: Function,
                                   results: dict,
                                   title: str = "标准EIT高斯牛顿重建结果"):
    """可视化重建结果"""
    
    # 创建三角剖分
    mesh = fwd_model.mesh
    coordinates = mesh.coordinates()
    triangles = []
    for cell in cells(mesh):
        triangles.append([vertex.index() for vertex in vertices(cell)])
    triangles = np.array(triangles)
    
    from matplotlib.tri import Triangulation
    tri = Triangulation(coordinates[:, 0], coordinates[:, 1], triangles)
    
    # 创建图形
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 真实导电率分布
    im1 = ax1.tripcolor(tri, sigma_true.vector()[:], shading='flat', cmap='jet', vmin=1.0, vmax=2.5)
    ax1.set_title('真实导电率分布', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 重建导电率分布
    sigma_reconstructed = results['conductivity']
    im2 = ax2.tripcolor(tri, sigma_reconstructed.vector()[:], shading='flat', cmap='jet', vmin=1.0, vmax=2.5)
    ax2.set_title('重建导电率分布', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 残差收敛曲线
    ax3.semilogy(results['residual_history'], 'b-o', markersize=6, linewidth=2)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('残差范数', fontsize=12)
    ax3.set_title('残差收敛曲线', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 相对变化曲线
    ax4.semilogy(results['sigma_change_history'], 'r-s', markersize=6, linewidth=2)
    ax4.set_xlabel('迭代次数', fontsize=12)
    ax4.set_ylabel('相对变化', fontsize=12)
    ax4.set_title('参数相对变化', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加收敛线
    if results['converged']:
        ax4.axhline(y=results['final_relative_change'], color='k', linestyle='--', alpha=0.5, 
                   label=f'收敛值: {results["final_relative_change"]:.2e}')
        ax4.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 计算并显示重建质量指标
    relative_error = np.linalg.norm(sigma_reconstructed.vector()[:] - sigma_true.vector()[:]) / np.linalg.norm(sigma_true.vector()[:])
    correlation = np.corrcoef(sigma_true.vector()[:], sigma_reconstructed.vector()[:])[0, 1]
    
    print(f"\n重建质量评估:")
    print(f"  相对L2误差: {relative_error:.4f}")
    print(f"  相关系数: {correlation:.4f}")
    print(f"  最终残差: {results['final_residual']:.2e}")
    print(f"  收敛状态: {'是' if results['converged'] else '否'}")

# ==================== 主程序 ====================

def main():
    """标准EIT重建演示程序"""
    
    print("="*70)
    print("标准EIT高斯牛顿重建演示")
    print("="*70)
    
    # 1. 系统配置
    n_elec = 16
    print(f"\n1. 系统配置:")
    print(f"   电极数量: {n_elec}")
    print(f"   激励模式: 相邻电极")
    print(f"   测量模式: 相邻电极")
    
    # 2. 创建网格
    print(f"\n2. 生成网格...")
    mesh_config = MeshConfig(radius=1.0, refinement=6, electrode_vertices=4)
    electrode_config = ElectrodePosition(L=n_elec, coverage=0.5)
    
    generator = MeshGenerator(mesh_config, electrode_config)
    mesh = generator.generate()
    
    print(f"   网格生成完成: {mesh.num_cells()}个单元, {mesh.num_vertices()}个顶点")
    
    # 3. 创建前向模型
    print(f"\n3. 创建前向模型...")
    pattern_config = PatternConfig(
        n_elec=n_elec,
        stim_pattern='{ad}',
        meas_pattern='{ad}',
        amplitude=1.0,
        use_meas_current=False
    )
    
    z = np.full(n_elec, 1e-6)  # 接触阻抗
    fwd_model = EITForwardModel(n_elec, pattern_config, z, mesh)
    
    print(f"   前向模型创建完成")
    print(f"   激励数: {fwd_model.pattern_manager.n_stim}")
    print(f"   测量数: {fwd_model.pattern_manager.n_meas_total}")
    
    # 4. 生成合成数据
    print(f"\n4. 生成合成测试数据...")
    synthetic_data = create_synthetic_data(
        fwd_model=fwd_model,
        inclusion_conductivity=2.5,
        background_conductivity=1.0,
        noise_level=0.02,
        center=(-0.3, 0.1),
        radius=0.3
    )
    
    print(f"   合成数据生成完成")
    print(f"   信噪比: {synthetic_data['snr_db']:.1f} dB")
    print(f"   测量数据范围: [{np.min(synthetic_data['data_noisy'].meas):.3f}, {np.max(synthetic_data['data_noisy'].meas):.3f}]")
    
    # 5. 创建重建器
    print(f"\n5. 创建标准高斯牛顿重建器...")
    reconstructor = StandardGaussNewtonReconstructor(
        fwd_model=fwd_model,
        max_iterations=250,
        convergence_tol=1e-5,
        regularization_param=0.001,
        line_search_steps=10,
        clip_values=(0.1, 5.0),
        verbose=True
    )
    
    # 6. 执行重建
    print(f"\n6. 执行高斯牛顿重建...")
    results = reconstructor.reconstruct(
        measured_data=synthetic_data['data_noisy'],
        initial_conductivity=1.0
    )
    
    # 7. 可视化结果
    print(f"\n7. 可视化重建结果...")
    visualize_reconstruction_results(
        fwd_model=fwd_model,
        sigma_true=synthetic_data['sigma_true'],
        results=results,
        title="标准EIT高斯牛顿重建结果"
    )
    
    print(f"\n" + "="*70)
    print("标准EIT重建演示完成!")
    print("="*70)
    
    return fwd_model, synthetic_data, results


if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 或者您安装的其他中文字体，例如 'Noto Sans CJK SC'
    plt.rcParams['font.family'] = 'sans-serif' # 通常默认就是 sans-serif
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 运行主程序
    fwd_model, synthetic_data, results = main()