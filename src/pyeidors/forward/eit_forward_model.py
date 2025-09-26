"""EIT前向模型 - 基于完全电极模型（简化版）"""

import numpy as np
import warnings
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

# FEniCS imports
from fenics import *
from dolfin import as_backend_type

from ..data.structures import PatternConfig, EITData, EITImage
from ..electrodes.patterns import StimMeasPatternManager


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