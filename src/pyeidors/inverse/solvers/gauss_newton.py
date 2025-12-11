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
    """模块化的PyTorch加速高斯牛顿EIT重建器
    
    实现 EIDORS 风格的 Gauss-Newton 迭代算法：
    
    更新公式: dx = -(J'WJ + λ²RtR)⁻¹ (J'W·dv + λ²RtR·de)
    
    其中:
        dv = 测量误差 = f(σ) - y_measured
        de = 先验误差 = σ - σ_prior
        W  = 测量逆协方差矩阵
        RtR = 空间正则化矩阵
        λ  = 超参数
    """
    
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
                 min_iterations: int = 1,
                 use_prior_term: bool = True):
        """
        初始化模块化高斯牛顿重建器
        
        参数:
            fwd_model: 前向模型
            jacobian_calculator: 雅可比计算器（可选）
            regularization: 正则化对象（可选）
            use_prior_term: 是否在更新中包含先验误差项 λ²RtR·de (EIDORS 风格)
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
        self.step_schedule: Optional[list[float]] = None
        self.min_iterations = int(max(1, min_iterations))
        # EIDORS 风格: 是否使用先验误差项 λ²RtR·de
        self.use_prior_term = use_prior_term
        # 先验导电率 (用于计算 de = σ - σ_prior)
        self._prior_data: Optional[np.ndarray] = None
        if self.verbose:
            print(
                f"[INFO] GN config: lambda={self.regularization_param:.3e}, "
                f"use_prior_term={self.use_prior_term}"
            )
        
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
                   jacobian_method: str = 'efficient',
                   prior_data: Optional[np.ndarray] = None):
        """执行模块化的高斯牛顿重建
        
        参数:
            measured_data: 测量数据
            initial_conductivity: 初始导电率 (标量或数组)
            jacobian_method: 雅可比计算方法
            prior_data: 先验导电率分布 (用于计算 de = σ - σ_prior)
                        如果为 None，则使用 initial_conductivity
        """

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

        meas_norm = torch.norm(meas_torch).item()
        meas_max = torch.max(torch.abs(meas_torch)).item()
        meas_weighted_norm = None
        if self._meas_weight_sqrt is not None:
            meas_weighted_norm = torch.norm(meas_torch * self._meas_weight_sqrt).item()

        model_scale = getattr(self, "model_scale", 1.0)

        # 初始化导电率分布
        if initial_conductivity is None:
            initial_conductivity = 1.0
        sigma_current = Function(self.fwd_model.V_sigma)
        if np.isscalar(initial_conductivity):
            sigma_current.vector()[:] = initial_conductivity
        else:
            sigma_current.vector()[:] = np.asarray(initial_conductivity).flatten()
        self._ensure_measurement_weights(sigma_current)
        
        # 设置先验数据 (用于 EIDORS 风格的 de = σ - σ_prior)
        if prior_data is not None:
            self._prior_data = np.asarray(prior_data).flatten()
        elif np.isscalar(initial_conductivity):
            self._prior_data = np.full(self.n_elements, initial_conductivity)
        else:
            self._prior_data = np.asarray(initial_conductivity).flatten()
        prior_torch = torch.from_numpy(self._prior_data).to(self.device, dtype=self._torch_dtype)
        
        # 记录收敛历史
        residual_history = []
        sigma_change_history = []
        iteration_logs = []
        
        if self.verbose:
            print(f"[INFO] lambda={self.regularization_param:.3e}")
            print(f"\n开始模块化高斯牛顿重建...")
            print(f"使用雅可比方法: {jacobian_method}")

        prev_residual = None

        with tqdm(total=self.max_iterations, disable=not self.verbose) as pbar:
            for iteration in range(self.max_iterations):
                
                # 1. 前向求解
                img_current = EITImage(elem_data=sigma_current.vector()[:], fwd_model=self.fwd_model)
                data_simulated, _ = self.fwd_model.fwd_solve(img_current)
                
                # 2. 计算残差 (dv = f(σ) - y_meas)
                data_sim_torch = torch.from_numpy(data_simulated.meas).to(self.device, dtype=self._torch_dtype)
                residual_torch = data_sim_torch - meas_torch  # dv
                if self._meas_weight_sqrt is not None:
                    weighted_residual_torch = residual_torch * self._meas_weight_sqrt
                    residual_norm_weighted = torch.norm(weighted_residual_torch).item()
                else:
                    weighted_residual_torch = residual_torch
                residual_norm_weighted = torch.norm(weighted_residual_torch).item()
                residual_norm = torch.norm(residual_torch).item()
                residual_max = torch.max(torch.abs(residual_torch)).item()
                
                # 计算 EIDORS 风格的完整目标函数: 0.5*dv'*W*dv + 0.5*de'*λ²RtR*de
                sigma_vec_torch = torch.from_numpy(sigma_current.vector()[:]).to(
                    self.device, dtype=self._torch_dtype
                )
                de_current = sigma_vec_torch - prior_torch  # de = σ - σ_prior
                meas_misfit = 0.5 * torch.dot(weighted_residual_torch, weighted_residual_torch).item()
                
                lambda_sq = self.regularization_param
                RtR_de = torch.mv(self.R_torch, de_current)
                prior_misfit = 0.5 * lambda_sq * torch.dot(de_current, RtR_de).item()
                total_objective = meas_misfit + prior_misfit
                
                residual_history.append(residual_norm)
                res_drop = None if prev_residual is None else prev_residual - residual_norm
                
                # 3. 使用模块化雅可比计算器
                measurement_jacobian_np = self.jacobian_calculator.calculate(
                    sigma_current, method=jacobian_method
                )
                if self.negate_jacobian:
                    measurement_jacobian_np = -measurement_jacobian_np
                J_torch = torch.from_numpy(measurement_jacobian_np).to(self.device, dtype=self._torch_dtype)
                if self._meas_weight_sqrt is not None:
                    J_weighted = J_torch * self._meas_weight_sqrt.unsqueeze(1)
                else:
                    J_weighted = J_torch
                
                # 4. 构建高斯牛顿系统
                # EIDORS 风格: dx = -(J'WJ + λ²RtR)⁻¹ (J'W·dv + λ²RtR·de)
                JTJ = torch.mm(J_weighted.t(), J_weighted)
                JTr = torch.mv(J_weighted.t(), weighted_residual_torch)
                
                lambda_eff = self.regularization_param
                
                # 计算先验误差项 de = σ_current - σ_prior
                sigma_current_torch = torch.from_numpy(sigma_current.vector()[:]).to(
                    self.device, dtype=self._torch_dtype
                )
                de_torch = sigma_current_torch - prior_torch
                
                A = JTJ + lambda_eff * self.R_torch
                
                # EIDORS 风格的完整 RHS: -(J'W·dv + λ²RtR·de)
                if self.use_prior_term:
                    RtR_de = torch.mv(self.R_torch, de_torch)
                    b = -(JTr + lambda_eff * RtR_de)
                else:
                    b = -JTr

                pred_norm = torch.norm(data_sim_torch).item()
                pred_max = torch.max(torch.abs(data_sim_torch)).item()
                jtr_norm = torch.norm(JTr).item()
                rel_residual = residual_norm / (meas_norm + 1e-12)
                rel_residual_weighted = (
                    residual_norm_weighted / (meas_weighted_norm + 1e-12) if meas_weighted_norm else None
                )

                # 5. 求解线性系统
                try:
                    delta_sigma_torch = torch.linalg.solve(A, b)
                except RuntimeError:
                    A_regularized = JTJ + (self.regularization_param * 10) * self.R_torch
                    delta_sigma_torch = torch.linalg.solve(A_regularized, b)
                delta_norm = torch.norm(delta_sigma_torch).item()

                # 6. 线搜索 (EIDORS 风格: 使用完整目标函数)
                if self.step_schedule is not None and iteration < len(self.step_schedule):
                    optimal_step_size = float(self.step_schedule[iteration])
                else:
                    optimal_step_size = self._line_search_torch(
                        sigma_current,
                        delta_sigma_torch,
                        meas_torch,
                        residual_norm_weighted,
                        self._meas_weight_sqrt,
                        prior_torch=prior_torch,
                        lambda_eff=lambda_eff,
                    )
                    if self.min_step is not None and optimal_step_size < self.min_step:
                        optimal_step_size = self.min_step

                
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

                # 如果残差上升，则回退到上一步（模仿 EIDORS 的 bad step rollback）
                if prev_residual is not None and residual_norm > prev_residual:
                    if self.verbose:
                        print(
                            f"[WARN] residual increased ({residual_norm:.3e} > {prev_residual:.3e}), "
                            "rolling back step"
                        )
                    sigma_current.vector()[:] = sigma_old_values
                    residual_history[-1] = prev_residual
                    sigma_change_history[-1] = 0.0
                    continue
                sigma_change_history.append(relative_change)

                iteration_logs.append(
                    {
                        "iteration": iteration,
                        "residual": residual_norm,
                        "residual_weighted": residual_norm_weighted,
                        "relative_residual": rel_residual,
                        "relative_residual_weighted": rel_residual_weighted,
                        "residual_max": residual_max,
                        "meas_norm": meas_norm,
                        "pred_norm": pred_norm,
                        "meas_max": meas_max,
                        "pred_max": pred_max,
                        "JTr_norm": jtr_norm,
                        "delta_norm": delta_norm,
                        "step": optimal_step_size,
                        "lambda_eff": lambda_eff,
                        "relative_change": relative_change,
                        "res_drop": res_drop,
                        # EIDORS 风格的目标函数分解
                        "meas_misfit": meas_misfit,
                        "prior_misfit": prior_misfit,
                        "total_objective": total_objective,
                    }
                )
                prev_residual = residual_norm
                
                if relative_change < self.convergence_tol:
                    if self.verbose:
                        print(f"\n收敛达到! 迭代 {iteration}, 相对变化: {relative_change:.2e}")
                    if iteration + 1 >= self.min_iterations:
                        break
                
                # 更新进度条
                if self.verbose:
                    pbar.set_postfix_str(
                        f"残差={residual_norm:.2e}, 步长={optimal_step_size:.3f}, Δσ={relative_change:.2e}"
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
            'regularization_type': type(self.regularization).__name__,
            'iteration_logs': iteration_logs,
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
        if self.verbose:
            finite_weights = weights[np.isfinite(weights)]
            w_min = finite_weights.min() if finite_weights.size else float("nan")
            w_max = finite_weights.max() if finite_weights.size else float("nan")
            w_med = np.median(finite_weights) if finite_weights.size else float("nan")
            print(
                f"[INFO] measurement weights ({strategy}): min={w_min:.3e}, med={w_med:.3e}, max={w_max:.3e}"
            )

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
        prior_torch=None,
        lambda_eff=None,
        retry: int = 0,
    ):
        """EIDORS 风格的线搜索 (line_search_onm2 完整移植)
        
        完整实现 EIDORS 的线搜索逻辑，包括：
        1. 数值稳定性检查 (calc_perturb)
        2. 多点采样 + 选择最小残差
        3. 自适应调整搜索范围
        4. 递归重试机制 (最多5次)
        
        目标函数: residual = 0.5*dv'*W*dv + 0.5*de'*(λ²RtR)*de
        
        参数:
            sigma_current: 当前导电率 (FEniCS Function)
            delta_sigma_torch: 搜索方向
            meas_target_torch: 目标测量值
            current_weighted_residual: 当前加权残差
            weight_vector: 测量权重 (sqrt(W))
            prior_torch: 先验导电率 (用于计算 de)
            lambda_eff: 有效正则化参数 λ²
            retry: 当前重试次数 (内部使用)
        """
        delta_sigma_np = delta_sigma_torch.cpu().numpy()
        current_residual = float(current_weighted_residual)
        x = sigma_current.vector()[:].copy()
        
        # 初始化 perturb (EIDORS 默认采样点)
        if not hasattr(self, '_line_search_perturb') or self._line_search_perturb is None:
            base_perturb = np.array([0, 1/16, 1/8, 1/4, 1/2, 1])
            self._line_search_perturb = base_perturb * self.max_step
        
        # 计算数值稳定的搜索范围 (类似 EIDORS calc_perturb)
        perturb = self._calc_perturb_limits(x, delta_sigma_np)
        
        # 多点采样评估目标函数
        mlist = np.full(len(perturb), np.nan)
        dv0 = None  # 保存 alpha=0 时的 dv，用于后续
        
        for i, alpha in enumerate(perturb):
            if i == 0:
                # alpha=0 时直接使用当前残差
                mlist[i] = current_residual ** 2 * 0.5
                continue
            
            # 计算新的导电率
            sigma_test_np = x + alpha * delta_sigma_np
            if self.clip_values is not None:
                sigma_test_np = np.clip(sigma_test_np, self.clip_values[0], self.clip_values[1])
            
            # 前向求解
            img_test = EITImage(elem_data=sigma_test_np, fwd_model=self.fwd_model)
            try:
                data_test, _ = self.fwd_model.fwd_solve(img_test)
            except Exception:
                mlist[i] = np.inf
                continue
            
            data_test_torch = torch.from_numpy(data_test.meas).to(self.device, dtype=self._torch_dtype)
            
            # 计算 dv (测量误差)
            dv_torch = data_test_torch - meas_target_torch
            if weight_vector is not None:
                weighted_dv = dv_torch * weight_vector
            else:
                weighted_dv = dv_torch
            
            # 测量误差项: 0.5 * dv' * W * dv
            meas_misfit = 0.5 * torch.dot(weighted_dv, weighted_dv).item()
            
            # 先验误差项: 0.5 * de' * (λ²RtR) * de
            prior_misfit = 0.0
            if self.use_prior_term and prior_torch is not None and lambda_eff is not None:
                sigma_test_torch = torch.from_numpy(sigma_test_np).to(self.device, dtype=self._torch_dtype)
                de_torch = sigma_test_torch - prior_torch
                RtR_de = torch.mv(self.R_torch, de_torch)
                prior_misfit = 0.5 * lambda_eff * torch.dot(de_torch, RtR_de).item()
            
            # 完整目标函数
            total_objective = meas_misfit + prior_misfit
            
            # 检查数值有效性 (EIDORS: NaN/Inf -> +Inf)
            if np.isnan(total_objective) or np.isinf(total_objective):
                mlist[i] = np.inf
            else:
                mlist[i] = total_objective
            
            # 早停：如果目标函数暴涨，跳过后续点 (EIDORS 风格)
            if mlist[i] / mlist[0] > 1e10:
                break
        
        # 选择最优步长
        valid_idx = np.where(np.isfinite(mlist))[0]
        if len(valid_idx) == 0:
            chosen_step = 0.0
            best_objective = mlist[0] if np.isfinite(mlist[0]) else np.inf
        else:
            best_idx = valid_idx[np.argmin(mlist[valid_idx])]
            chosen_step = perturb[best_idx]
            best_objective = mlist[best_idx]
        
        # EIDORS 风格的 perturb 自适应调整
        self._update_perturb_eidors_style(chosen_step, perturb, mlist, valid_idx)
        
        # EIDORS 风格递归重试：如果 alpha=0 是最优，尝试新的 perturb
        if chosen_step == 0 and retry < 5:
            return self._line_search_torch(
                sigma_current, delta_sigma_torch, meas_target_torch,
                current_weighted_residual, weight_vector, prior_torch, lambda_eff,
                retry=retry + 1
            )
        
        # 返回步长 (允许返回 0，由调用方决定如何处理)
        return float(chosen_step)
    
    def _calc_perturb_limits(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        """计算数值稳定的搜索范围 (类似 EIDORS calc_perturb)
        
        检查浮点数精度限制，避免数值溢出。
        
        参数:
            x: 当前导电率值
            dx: 搜索方向
        
        返回:
            perturb: 经过数值稳定性调整的采样点数组
        """
        perturb = self._line_search_perturb.copy()
        
        # 确保 perturb[0] = 0
        if perturb[0] != 0:
            perturb = np.concatenate([[0], perturb])
        
        # 计算 alpha 的数值稳定上限
        # 避免 x + alpha*dx 溢出或变为非法值
        eps_machine = np.finfo(np.float64).eps
        
        # 上限: 防止溢出到 inf
        realmax = np.finfo(np.float64).max / 2
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # (realmax - x) / dx for dx > 0
            au_pos = (realmax - x) / dx
            au_pos[dx <= 0] = np.inf
            au_pos[~np.isfinite(au_pos)] = np.inf
            
            # (-realmax - x) / dx for dx < 0
            au_neg = (-realmax - x) / dx
            au_neg[dx >= 0] = np.inf
            au_neg[~np.isfinite(au_neg)] = np.inf
            
            max_alpha = min(np.min(au_pos), np.min(au_neg))
        
        # 下限: 防止步长太小以至于没有变化
        with np.errstate(divide='ignore', invalid='ignore'):
            al = eps_machine * np.abs(x) / np.abs(dx)
            al[~np.isfinite(al)] = 0
            min_alpha = np.max(al) if len(al) > 0 else 0
        
        # 限制 max_alpha 不超过 1.0 (完整牛顿步)
        max_alpha = min(max_alpha, 1.0)
        
        # 如果 perturb 超出范围，在对数空间重新缩放 (EIDORS 风格)
        if perturb[-1] > max_alpha or (len(perturb) > 1 and perturb[1] < min_alpha):
            # 过滤掉 0 和太小的值
            p_nonzero = perturb[perturb > eps_machine]
            if len(p_nonzero) == 0:
                # fallback
                return np.array([0, max_alpha / 4, max_alpha / 2, max_alpha])
            
            # 在对数空间重新缩放
            log_p = np.log10(p_nonzero)
            log_max = np.log10(max_alpha) if max_alpha > eps_machine else -10
            log_min = np.log10(min_alpha) if min_alpha > eps_machine else -100
            
            # 缩放到有效范围
            p_range = log_p[-1] - log_p[0] if len(log_p) > 1 else 1
            target_range = log_max - log_min
            
            if p_range > target_range and target_range > 0:
                log_p = log_p * (target_range / p_range)
            
            if log_p[-1] > log_max:
                log_p = log_p - (log_p[-1] - log_max)
            elif log_p[0] < log_min:
                log_p = log_p + (log_min - log_p[0])
            
            perturb = np.concatenate([[0], 10 ** log_p])
            
        return perturb
    
    def _update_perturb_eidors_style(
        self, chosen_step: float, perturb: np.ndarray, 
        mlist: np.ndarray, valid_idx: np.ndarray
    ) -> None:
        """EIDORS 风格的 perturb 自适应调整
        
        根据本次线搜索结果调整下次的搜索范围。
        """
        goodi = valid_idx
        dtol = self.convergence_tol  # 用作判断阈值
        
        if chosen_step == 0:  # bad step: alpha=0 is best
            if len(goodi) > 1 and mlist[0] * 1.05 < mlist[goodi[-1]]:
                # 解发散了，缩小搜索范围
                self._line_search_perturb = self._line_search_perturb / 10
            elif perturb[-1] > 1.0 - 1e-9:
                # 已经在 alpha=1 附近，放弃
                pass
            elif perturb[-1] * 10 > 1.0 - 1e-9:
                # 扩大但限制在 1.0
                self._line_search_perturb = self._line_search_perturb / perturb[-1]
            else:
                # 扩大搜索范围
                self._line_search_perturb = self._line_search_perturb * 10
        else:  # good step
            # 检查是否有明显改进
            all_similar = len(goodi) > 0 and np.all(mlist[goodi] / mlist[0] - 1 > -10 * dtol)
            
            if all_similar and perturb[-1] * 10 < 1.0 + 1e-9:
                # 改进不大，扩大范围
                self._line_search_perturb = self._line_search_perturb * 10
            else:
                # 以最优点为中心重新调整
                if chosen_step > 0 and perturb[-1] > 0:
                    scale = (chosen_step / perturb[-1]) * 2
                    new_perturb = self._line_search_perturb * scale
                    # 限制在 1.0 以内
                    if new_perturb[-1] > 1.0 - 1e-9:
                        new_perturb = new_perturb / new_perturb[-1]
                    self._line_search_perturb = new_perturb
        
        # EIDORS 风格: 添加 1% 随机扰动，避免卡在局部
        jiggle = np.exp(np.random.randn(len(self._line_search_perturb)) * 0.01)
        self._line_search_perturb = self._line_search_perturb * jiggle
        
        # 确保不超过 1.0
        if self._line_search_perturb[-1] > 1.0 - 1e-9:
            self._line_search_perturb = self._line_search_perturb / self._line_search_perturb[-1]
    
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
