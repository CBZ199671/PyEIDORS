"""模块化的PyTorch加速高斯牛顿EIT重建器"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from tqdm import tqdm
from fenics import Function

from ...data.structures import EITData, EITImage
from ..jacobian.direct_jacobian import DirectJacobianCalculator
from ..regularization.smoothness import SmoothnessRegularization


class ModularGaussNewtonReconstructor:
    """模块化的PyTorch加速高斯牛顿EIT重建器"""
    
    def __init__(self, 
                 fwd_model,
                 jacobian_calculator=None,
                 regularization=None,
                 max_iterations: int = 15,
                 convergence_tol: float = 1e-4,
                 regularization_param: float = 0.01,
                 line_search_steps: int = 8,
                 clip_values: Tuple[float, float] = (1e-6, 10.0),
                 device: str = 'cuda:0',
                 verbose: bool = True):
        """
        初始化模块化高斯牛顿重建器
        
        参数:
            fwd_model: 前向模型
            jacobian_calculator: 雅可比计算器（可选）
            regularization: 正则化对象（可选）
            其他参数同之前版本
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
        
        # 设置雅可比计算器
        if jacobian_calculator is None:
            self.jacobian_calculator = DirectJacobianCalculator(fwd_model)
        else:
            self.jacobian_calculator = jacobian_calculator
        
        # 设置正则化
        if regularization is None:
            self.regularization = SmoothnessRegularization(fwd_model, alpha=1.0)
        else:
            self.regularization = regularization
        
        self.n_elements = len(Function(fwd_model.V_sigma).vector()[:])
        self.n_measurements = fwd_model.pattern_manager.n_meas_total
        
        # 预计算正则化矩阵
        self.R_torch = None
        
        if self.verbose:
            print(f"模块化PyTorch高斯牛顿重建器初始化:")
            print(f"  元素数: {self.n_elements}")
            print(f"  测量数: {self.n_measurements}")
            print(f"  雅可比计算器: {type(self.jacobian_calculator).__name__}")
            print(f"  正则化方法: {type(self.regularization).__name__}")
            print(f"  计算设备: {self.device}")
    
    def reconstruct(self, 
                   measured_data: Union[object, np.ndarray],
                   initial_conductivity: float = 1.0,
                   jacobian_method: str = 'efficient'):
        """执行模块化的高斯牛顿重建"""
        
        # 处理输入数据
        if hasattr(measured_data, 'meas'):
            meas_vector = measured_data.meas
        else:
            meas_vector = measured_data.flatten()
        
        if len(meas_vector) != self.n_measurements:
            raise ValueError(f"测量数据长度不匹配: {len(meas_vector)} vs {self.n_measurements}")
        
        # 转换测量数据到GPU
        meas_torch = torch.from_numpy(meas_vector).float().to(self.device)
        
        # 获取正则化矩阵
        if self.R_torch is None:
            R_np = self.regularization.get_regularization_matrix()
            self.R_torch = torch.from_numpy(R_np).float().to(self.device)
        
        # 初始化导电率分布
        sigma_current = Function(self.fwd_model.V_sigma)
        sigma_current.vector()[:] = initial_conductivity
        
        # 记录收敛历史
        residual_history = []
        sigma_change_history = []
        
        if self.verbose:
            print(f"\n开始模块化高斯牛顿重建...")
            print(f"使用雅可比方法: {jacobian_method}")
        
        with tqdm(total=self.max_iterations, disable=not self.verbose) as pbar:
            for iteration in range(self.max_iterations):
                
                # 1. 前向求解
                img_current = EITImage(elem_data=sigma_current.vector()[:], fwd_model=self.fwd_model)
                data_simulated, _ = self.fwd_model.fwd_solve(img_current)
                
                # 2. 计算残差
                data_sim_torch = torch.from_numpy(data_simulated.meas).float().to(self.device)
                residual_torch = data_sim_torch - meas_torch
                residual_norm = torch.norm(residual_torch).item()
                residual_history.append(residual_norm)
                
                # 3. 使用模块化雅可比计算器
                measurement_jacobian_np = -self.jacobian_calculator.calculate(
                    sigma_current, method=jacobian_method
                )
                J_torch = torch.from_numpy(measurement_jacobian_np).float().to(self.device)
                
                # 4. 构建高斯牛顿系统
                JTJ = torch.mm(J_torch.t(), J_torch)
                JTr = torch.mv(J_torch.t(), residual_torch)
                
                A = JTJ + self.regularization_param * self.R_torch
                b = -JTr
                
                # 5. 求解线性系统
                try:
                    delta_sigma_torch = torch.linalg.solve(A, b)
                except RuntimeError:
                    A_regularized = JTJ + (self.regularization_param * 10) * self.R_torch
                    delta_sigma_torch = torch.linalg.solve(A_regularized, b)
                
                # 6. 线搜索
                optimal_step_size = self._line_search_torch(
                    sigma_current, delta_sigma_torch, meas_torch, residual_norm
                )
                
                # 7. 更新导电率
                sigma_old_values = sigma_current.vector()[:].copy()
                delta_sigma_np = delta_sigma_torch.cpu().numpy()
                sigma_current.vector()[:] += optimal_step_size * delta_sigma_np
                
                # 8. 应用约束
                if self.clip_values is not None:
                    sigma_current.vector()[:] = np.clip(
                        sigma_current.vector()[:], self.clip_values[0], self.clip_values[1]
                    )
                
                # 9. 检查收敛
                sigma_new_torch = torch.from_numpy(sigma_current.vector()[:]).float().to(self.device)
                sigma_old_torch = torch.from_numpy(sigma_old_values).float().to(self.device)
                
                sigma_change = torch.norm(sigma_new_torch - sigma_old_torch).item()
                relative_change = sigma_change / (torch.norm(sigma_new_torch).item() + 1e-12)
                sigma_change_history.append(relative_change)
                
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
            'final_relative_change': relative_change,
            'jacobian_method': jacobian_method,
            'regularization_type': type(self.regularization).__name__
        }
        
        if self.verbose:
            print(f"\n重建完成:")
            print(f"  迭代次数: {results['iterations']}")
            print(f"  最终残差: {results['final_residual']:.2e}")
            print(f"  雅可比方法: {jacobian_method}")
            print(f"  正则化类型: {results['regularization_type']}")
        
        return results
    
    def _line_search_torch(self, sigma_current, delta_sigma_torch, meas_target_torch, current_residual_norm):
        """线搜索算法"""
        step_candidates = torch.linspace(0.1, 1.0, self.line_search_steps, device=self.device)
        best_step = step_candidates[0].item()
        best_residual = float('inf')
        
        delta_sigma_np = delta_sigma_torch.cpu().numpy()
        
        for step_size in step_candidates:
            try:
                sigma_test = sigma_current.copy(deepcopy=True)
                sigma_test.vector()[:] += step_size.item() * delta_sigma_np
                
                if self.clip_values is not None:
                    sigma_test.vector()[:] = np.clip(
                        sigma_test.vector()[:], self.clip_values[0], self.clip_values[1]
                    )
                
                img_test = EITImage(elem_data=sigma_test.vector()[:], fwd_model=self.fwd_model)
                data_test, _ = self.fwd_model.fwd_solve(img_test)
                
                data_test_torch = torch.from_numpy(data_test.meas).float().to(self.device)
                residual_torch = data_test_torch - meas_target_torch
                residual_norm = torch.norm(residual_torch).item()
                
                if residual_norm < best_residual:
                    best_residual = residual_norm
                    best_step = step_size.item()
                    
            except Exception:
                continue
        
        return best_step
    
    def set_regularization(self, regularization):
        """动态设置正则化方法"""
        self.regularization = regularization
        self.R_torch = None  # 重置缓存
        
        if self.verbose:
            print(f"正则化方法更新为: {type(regularization).__name__}")
    
    def set_jacobian_calculator(self, jacobian_calculator):
        """动态设置雅可比计算器"""
        self.jacobian_calculator = jacobian_calculator
        
        if self.verbose:
            print(f"雅可比计算器更新为: {type(jacobian_calculator).__name__}")