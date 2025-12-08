"""模块化的PyTorch加速高斯牛顿EIT重建器"""

import numpy as np
import torch
from typing import Tuple, Optional, Union
from tqdm import tqdm
from fenics import Function

from ...data.structures import EITImage
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
                 verbose: bool = True,
                 use_measurement_weights: bool = False,
                 weight_floor: float = 1e-9,
                 measurement_weight_strategy: str = "none",
                 max_step: float = 1.0,
                 min_step: float = 0.1,
                 negate_jacobian: bool = True,
                 scaling_factor: float = 1.0):
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
        self.measurement_weight_strategy = measurement_weight_strategy
        self.use_measurement_weights = use_measurement_weights or measurement_weight_strategy != "none"
        self.weight_floor = weight_floor
        self._meas_weight_sqrt: Optional[torch.Tensor] = None
        self._baseline_measurement: Optional[np.ndarray] = None
        self._measured_vector: Optional[np.ndarray] = None
        self.negate_jacobian = negate_jacobian
        self.max_step = max_step
        self.min_step = min_step
        self.model_scale: float = 1.0
        self.step_schedule: Optional[list[float]] = None
        # 全链路缩放因子（用于同时放大预测/雅可比和调整正则）
        self.scaling_factor = scaling_factor
        
        # 设置计算设备
        if device.startswith('cuda') and torch.cuda.is_available():
            self.device = torch.device(device)
            if self.verbose:
                print(f"使用GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device('cpu')
            if self.verbose:
                print("使用CPU计算")

        self._torch_dtype = torch.float64
        
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

        self._meas_weight_sqrt = None
        self._baseline_measurement = None
        
        # 处理输入数据
        if hasattr(measured_data, 'meas'):
            meas_vector = measured_data.meas
        else:
            meas_vector = measured_data.flatten()
        
        if len(meas_vector) != self.n_measurements:
            raise ValueError(f"测量数据长度不匹配: {len(meas_vector)} vs {self.n_measurements}")
        
        # 转换测量数据到目标设备
        meas_torch = torch.from_numpy(meas_vector).to(self.device, dtype=self._torch_dtype)
        self._measured_vector = meas_vector.copy()
        
        # 获取正则化矩阵
        if self.R_torch is None:
            R_np = self.regularization.get_regularization_matrix()
            self.R_torch = torch.from_numpy(R_np).to(self.device, dtype=self._torch_dtype)
        
        model_scale = getattr(self, "model_scale", 1.0)

        # 初始化导电率分布
        if initial_conductivity is None:
            initial_conductivity = 1.0
        sigma_current = Function(self.fwd_model.V_sigma)
        sigma_current.vector()[:] = initial_conductivity
        self._ensure_measurement_weights(sigma_current)
        
        # 记录收敛历史
        residual_history = []
        sigma_change_history = []
        k_scale = float(self.scaling_factor)
        if self.verbose:
            print(f"[INFO] scaling_factor k={k_scale:.3e}, lambda={self.regularization_param:.3e}")
        
        if self.verbose:
            print(f"\n开始模块化高斯牛顿重建...")
            print(f"使用雅可比方法: {jacobian_method}")
        
        with tqdm(total=self.max_iterations, disable=not self.verbose) as pbar:
            for iteration in range(self.max_iterations):
                
                # 1. 前向求解
                img_current = EITImage(elem_data=sigma_current.vector()[:], fwd_model=self.fwd_model)
                data_simulated, _ = self.fwd_model.fwd_solve(img_current)
                
                # 2. 计算残差
                data_sim_torch = torch.from_numpy(data_simulated.meas).to(self.device, dtype=self._torch_dtype)
                if model_scale != 1.0:
                    data_sim_torch = data_sim_torch * model_scale
                if k_scale != 1.0:
                    data_sim_torch = data_sim_torch * k_scale
                residual_torch = data_sim_torch - meas_torch
                if self._meas_weight_sqrt is not None:
                    weighted_residual_torch = residual_torch * self._meas_weight_sqrt
                    residual_norm_weighted = torch.norm(weighted_residual_torch).item()
                else:
                    weighted_residual_torch = residual_torch
                    residual_norm_weighted = torch.norm(weighted_residual_torch).item()
                residual_norm = torch.norm(residual_torch).item()
                residual_max = torch.max(torch.abs(residual_torch)).item()
                residual_history.append(residual_norm)
                
                # 3. 使用模块化雅可比计算器
                measurement_jacobian_np = self.jacobian_calculator.calculate(
                    sigma_current, method=jacobian_method
                )
                if self.negate_jacobian:
                    measurement_jacobian_np = -measurement_jacobian_np
                if model_scale != 1.0:
                    measurement_jacobian_np = measurement_jacobian_np * model_scale
                if k_scale != 1.0:
                    measurement_jacobian_np = measurement_jacobian_np * k_scale
                J_torch = torch.from_numpy(measurement_jacobian_np).to(self.device, dtype=self._torch_dtype)
                if self._meas_weight_sqrt is not None:
                    J_weighted = J_torch * self._meas_weight_sqrt.unsqueeze(1)
                else:
                    J_weighted = J_torch
                
                # 4. 构建高斯牛顿系统
                JTJ = torch.mm(J_weighted.t(), J_weighted)
                JTr = torch.mv(J_weighted.t(), weighted_residual_torch)
                
                lambda_eff = self.regularization_param * (k_scale ** 2)
                A = JTJ + lambda_eff * self.R_torch
                b = -JTr
                if self.verbose and iteration == 0:
                    jtr_norm = torch.norm(JTr).item()
                    meas_norm = torch.norm(meas_torch).item()
                    meas_max = torch.max(torch.abs(meas_torch)).item()
                    pred_norm = torch.norm(data_sim_torch).item()
                    pred_max = torch.max(torch.abs(data_sim_torch)).item()
                    print(
                        f"[DEBUG i0] model_scale={model_scale:.3e}, "
                        f"meas_norm={meas_norm:.3e}, pred_norm={pred_norm:.3e}, "
                        f"meas_max={meas_max:.3e}, pred_max={pred_max:.3e}, "
                        f"res_norm={residual_norm:.3e}, res_max={residual_max:.3e}, "
                        f"JTr_norm={jtr_norm:.3e}"
                    )
                
                # 5. 求解线性系统
                try:
                    delta_sigma_torch = torch.linalg.solve(A, b)
                except RuntimeError:
                    A_regularized = JTJ + (self.regularization_param * 10) * self.R_torch
                    delta_sigma_torch = torch.linalg.solve(A_regularized, b)
                delta_norm = torch.norm(delta_sigma_torch).item()

                # 6. 线搜索
                if self.step_schedule is not None and iteration < len(self.step_schedule):
                    optimal_step_size = float(self.step_schedule[iteration])
                else:
                    optimal_step_size = self._line_search_torch(
                        sigma_current,
                        delta_sigma_torch,
                        meas_torch,
                        residual_norm_weighted,
                        self._meas_weight_sqrt,
                    )
                    if self.min_step is not None and optimal_step_size < self.min_step:
                        optimal_step_size = self.min_step
                if self.verbose:
                    jtr_norm = torch.norm(JTr).item()
                    desc = (
                        f"[DEBUG i{iteration}] "
                        f"model_scale={model_scale:.3e}, "
                        f"meas_norm={meas_norm:.3e}, pred_norm={pred_norm:.3e}, "
                        f"meas_max={meas_max:.3e}, pred_max={pred_max:.3e}, "
                        f"res_norm={residual_norm:.3e}, res_max={residual_max:.3e}, "
                        f"JTr_norm={jtr_norm:.3e}, delta_norm={delta_norm:.3e}, step={optimal_step_size:.3e}, lambda_eff={lambda_eff:.3e}, k={k_scale:.3e}"
                    )
                    print(desc)
                
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
                sigma_new_torch = torch.from_numpy(sigma_current.vector()[:]).to(self.device, dtype=self._torch_dtype)
                sigma_old_torch = torch.from_numpy(sigma_old_values).to(self.device, dtype=self._torch_dtype)
                
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
                        f"残差: {residual_norm:.2e} | 加权残差: {residual_norm_weighted:.2e} | 相对变化: {relative_change:.2e} | 步长: {optimal_step_size:.3f}"
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
        if self._baseline_measurement is not None:
            results['baseline_measurement'] = self._baseline_measurement.copy()
        if self._meas_weight_sqrt is not None:
            results['measurement_weight'] = (self._meas_weight_sqrt.detach().cpu().numpy() ** 2)
        
        if self.verbose:
            print(f"\n重建完成:")
            print(f"  迭代次数: {results['iterations']}")
            print(f"  最终残差: {results['final_residual']:.2e}")
            print(f"  雅可比方法: {jacobian_method}")
            print(f"  正则化类型: {results['regularization_type']}")
        
        return results
    
    def _ensure_measurement_weights(self, sigma_function: Function) -> None:
        """基于基线前向解计算测量加权系数（EIDORS `calc_meas_icov` 的简化版本）。"""
        strategy = self.measurement_weight_strategy
        if not self.use_measurement_weights or strategy == "none":
            self._meas_weight_sqrt = None
            self._baseline_measurement = None
            return

        img = EITImage(elem_data=sigma_function.vector()[:], fwd_model=self.fwd_model)
        baseline_data, _ = self.fwd_model.fwd_solve(img)
        baseline_vector = baseline_data.meas.astype(np.float64)
        self._baseline_measurement = baseline_vector.copy()

        if strategy == "baseline":
            reference_vector = baseline_vector
        elif strategy == "scaled_baseline":
            reference_vector = self._scale_baseline_to_measured(baseline_vector)
        elif strategy == "difference":
            reference_vector = self._difference_with_baseline(baseline_vector)
        else:
            reference_vector = baseline_vector

        weights = reference_vector ** 2
        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.maximum(weights, self.weight_floor)
        median = np.median(weights)
        if median > 0:
            weights = weights / median

        self._meas_weight_sqrt = torch.from_numpy(np.sqrt(weights)).to(self.device, dtype=self._torch_dtype)

    def _scale_baseline_to_measured(self, baseline_vector: np.ndarray) -> np.ndarray:
        """线性拉伸基线测量以匹配当前实测，供权重估计使用。"""
        if self._measured_vector is None:
            return baseline_vector

        x = baseline_vector
        y = self._measured_vector
        denom = np.dot(x, x)
        if denom < 1e-18:
            return baseline_vector
        scale = np.dot(y, x) / denom
        if abs(scale) < 1e-12:
            scale = 1.0 if scale >= 0 else -1.0
        bias = y.mean() - scale * x.mean()
        return scale * baseline_vector + bias

    def _difference_with_baseline(self, baseline_vector: np.ndarray) -> np.ndarray:
        """模仿 EIDORS 的差分归一化：使用与基线差值的幅度来构造权重。"""
        if self._measured_vector is None:
            return baseline_vector
        diff = self._measured_vector - baseline_vector
        diff_abs = np.abs(diff)
        return np.where(diff_abs > self.weight_floor, diff_abs, self.weight_floor)

    def _line_search_torch(
        self,
        sigma_current,
        delta_sigma_torch,
        meas_target_torch,
        current_weighted_residual,
        weight_vector=None,
    ):
        """单调回溯线搜索：尝试 max_step, max_step/2, ... 直至 min_step。"""
        step = float(self.max_step)
        delta_sigma_np = delta_sigma_torch.cpu().numpy()
        best_residual = current_weighted_residual
        best_step = 0.0

        for _ in range(max(1, self.line_search_steps)):
            sigma_test = sigma_current.copy(deepcopy=True)
            sigma_test.vector()[:] += step * delta_sigma_np
            if self.clip_values is not None:
                sigma_test.vector()[:] = np.clip(sigma_test.vector()[:], self.clip_values[0], self.clip_values[1])

            img_test = EITImage(elem_data=sigma_test.vector()[:], fwd_model=self.fwd_model)
            data_test, _ = self.fwd_model.fwd_solve(img_test)
            data_test_torch = torch.from_numpy(data_test.meas).to(self.device, dtype=self._torch_dtype)
            residual_torch = data_test_torch - meas_target_torch
            if weight_vector is not None:
                residual_norm = torch.norm(residual_torch * weight_vector).item()
            else:
                residual_norm = torch.norm(residual_torch).item()

            improvement_tol = max(best_residual * 1e-6, 1e-12)
            if residual_norm < best_residual - improvement_tol:
                best_residual = residual_norm
                best_step = step
                break

            step *= 0.5
            if step < self.min_step:
                break

        return float(best_step if best_step > 0 else self.min_step)
    
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
