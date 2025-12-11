"""PyEidors核心系统类

这是整个EIT系统的主要接口，集成了前向模型、逆问题求解器和数据处理功能。
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from fenics import Function

from .data.structures import EITData, EITImage, PatternConfig, MeshConfig
from .forward.eit_forward_model import EITForwardModel
from .inverse.solvers.gauss_newton import ModularGaussNewtonReconstructor
from .inverse.jacobian.direct_jacobian import DirectJacobianCalculator
from .inverse.regularization.smoothness import SmoothnessRegularization, NOSERRegularization, TikhonovRegularization
from .inverse import (
    perform_absolute_reconstruction,
    perform_difference_reconstruction,
    ReconstructionResult,
)
from .electrodes.patterns import StimMeasPatternManager
from .geometry.mesh_loader import MeshLoader
from .geometry.simple_mesh_generator import create_simple_eit_mesh


class EITSystem:
    """PyEidors核心系统类
    
    集成了EIT系统的所有主要功能：
    - 网格生成和管理
    - 前向问题求解
    - 逆问题重建
    - 数据处理和可视化
    """
    
    def __init__(
        self,
        n_elec: int = 16,
        pattern_config: Optional[PatternConfig] = None,
        mesh_config: Optional[MeshConfig] = None,
        contact_impedance: Optional[np.ndarray] = None,
        base_conductivity: float = 1.0,
        regularization_type: str = "noser",
        regularization_alpha: float = 1.0,
        noser_exponent: float = 0.5,
        noser_floor: float = 1e-12,
        **kwargs,
    ):
        """
        初始化EIT系统
        
        参数:
            n_elec: 电极数量
            pattern_config: 激励测量模式配置
            mesh_config: 网格配置
            contact_impedance: 接触阻抗
            base_conductivity: 基线导电率
            regularization_type: 正则化类型 ("noser", "tikhonov", "smoothness")
            regularization_alpha: 正则化系数
            noser_exponent: NOSER 正则化的指数 (EIDORS 默认 0.5)
            noser_floor: NOSER 对角线元素的最小值
            **kwargs: 其他配置参数
        """
        self.n_elec = n_elec
        
        # 设置默认配置
        if pattern_config is None:
            pattern_config = PatternConfig(
                n_elec=n_elec,
                stim_pattern='{ad}',
                meas_pattern='{ad}',
                amplitude=1.0
            )
        self.pattern_config = pattern_config
        
        if mesh_config is None:
            mesh_config = MeshConfig(radius=1.0, refinement=8)
        self.mesh_config = mesh_config
        
        # 设置接触阻抗
        if contact_impedance is None:
            contact_impedance = np.ones(n_elec) * 0.01
        self.contact_impedance = contact_impedance

        self.base_conductivity = base_conductivity
        self.regularization_type = regularization_type.lower()
        self.regularization_alpha = regularization_alpha
        self.noser_exponent = noser_exponent
        self.noser_floor = noser_floor
        
        # 初始化组件
        self.mesh = None
        self.fwd_model = None
        self.reconstructor = None
        self._is_initialized = False
        
    def setup(self, mesh=None):
        """设置EIT系统
        
        参数:
            mesh: 可选的外部网格，如果不提供则需要手动设置
        """
        # 设置网格
        if mesh is not None:
            self.mesh = mesh
        else:
            # 首先尝试加载现有网格
            try:
                mesh_loader = MeshLoader()
                self.mesh = mesh_loader.get_default_mesh()
                print(f"已加载现有网格: {self.mesh.get_info()}")
            except Exception as load_error:
                # 如果加载失败，尝试生成新网格
                print(f"加载现有网格失败: {load_error}")
                print("正在生成新的EIT网格...")
                try:
                    self.mesh = create_simple_eit_mesh(
                        n_elec=self.n_elec,
                        radius=1.0,
                        mesh_size=0.1
                    )
                    print(f"新网格生成成功: {self.mesh.get_info()}")
                except Exception as gen_error:
                    raise RuntimeError(f"无法生成网格: {gen_error}。请检查GMsh安装或提供网格对象。")
        
        # 初始化前向模型
        self.fwd_model = EITForwardModel(
            n_elec=self.n_elec,
            pattern_config=self.pattern_config,
            z=self.contact_impedance,
            mesh=self.mesh
        )
        
        # 初始化重建器
        jacobian_calculator = DirectJacobianCalculator(self.fwd_model)
        if self.regularization_type == "noser":
            regularization = NOSERRegularization(
                self.fwd_model,
                jacobian_calculator,
                base_conductivity=self.base_conductivity,
                alpha=self.regularization_alpha,
                exponent=self.noser_exponent,
                floor=self.noser_floor,
            )
        elif self.regularization_type == "tikhonov":
            regularization = TikhonovRegularization(self.fwd_model, alpha=self.regularization_alpha)
        else:
            regularization = SmoothnessRegularization(self.fwd_model, alpha=self.regularization_alpha)
        
        self.reconstructor = ModularGaussNewtonReconstructor(
            fwd_model=self.fwd_model,
            jacobian_calculator=jacobian_calculator,
            regularization=regularization
        )
        
        self._is_initialized = True
        
    def forward_solve(self, conductivity: Union[np.ndarray, Function, EITImage]) -> EITData:
        """前向求解
        
        参数:
            conductivity: 导电率分布
            
        返回:
            EIT测量数据
        """
        if not self._is_initialized:
            raise RuntimeError("系统未初始化，请先调用setup()方法")
        
        # 处理不同类型的导电率输入
        if isinstance(conductivity, np.ndarray):
            img = EITImage(elem_data=conductivity, fwd_model=self.fwd_model)
        elif isinstance(conductivity, EITImage):
            img = conductivity
        else:
            raise ValueError("不支持的导电率输入类型")
            
        # 执行前向求解
        data, _ = self.fwd_model.fwd_solve(img)
        return data
        
    def inverse_solve(self, data: EITData, 
                     reference_data: Optional[EITData] = None,
                     initial_guess: Optional[np.ndarray] = None) -> EITImage:
        """逆问题重建
        
        参数:
            data: 测量数据
            reference_data: 参考数据（可选）
            initial_guess: 初始猜测（可选）
            
        返回:
            重建的导电率分布
        """
        if not self._is_initialized:
            raise RuntimeError("系统未初始化，请先调用setup()方法")
        
        # 处理差分测量
        if reference_data is not None:
            diff_data = EITData(
                meas=data.meas - reference_data.meas,
                stim_pattern=data.stim_pattern,
                n_elec=data.n_elec,
                n_stim=data.n_stim,
                n_meas=data.n_meas,
                type='difference'
            )
        else:
            diff_data = data
        
        # 执行重建
        result = self.reconstructor.reconstruct(diff_data, initial_guess)
        
        return result

    def absolute_reconstruct(
        self,
        measurement_data: EITData,
        baseline_image: Optional[EITImage] = None,
        initial_image: Optional[EITImage] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionResult:
        """执行绝对成像重建的便捷入口。"""

        if baseline_image is None and self._is_initialized:
            baseline_image = self.create_homogeneous_image()

        return perform_absolute_reconstruction(
            eit_system=self,
            measurement_data=measurement_data,
            baseline_image=baseline_image,
            initial_image=initial_image,
            metadata=metadata,
        )

    def difference_reconstruct(
        self,
        measurement_data: EITData,
        reference_data: EITData,
        initial_image: Optional[EITImage] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReconstructionResult:
        """执行差分成像重建的便捷入口。"""

        return perform_difference_reconstruction(
            eit_system=self,
            measurement_data=measurement_data,
            reference_data=reference_data,
            initial_image=initial_image,
            metadata=metadata,
        )
        
    def create_homogeneous_image(self, conductivity: Optional[float] = None) -> EITImage:
        """创建均匀导电率图像
        
        参数:
            conductivity: 导电率值
        
        返回:
            均匀导电率图像
        """
        if not self._is_initialized:
            raise RuntimeError("系统未初始化，请先调用setup()方法")

        if conductivity is None:
            conductivity = self.base_conductivity

        n_elements = len(Function(self.fwd_model.V_sigma).vector()[:])
        elem_data = np.ones(n_elements) * conductivity
        
        return EITImage(elem_data=elem_data, fwd_model=self.fwd_model)
        
    def add_phantom(self, base_conductivity: float = 1.0, 
                   phantom_conductivity: float = 2.0,
                   phantom_center: tuple = (0.3, 0.3),
                   phantom_radius: float = 0.2) -> EITImage:
        """添加圆形幻影
        
        参数:
            base_conductivity: 背景导电率
            phantom_conductivity: 幻影导电率
            phantom_center: 幻影中心坐标
            phantom_radius: 幻影半径
            
        返回:
            包含幻影的导电率图像
        """
        if not self._is_initialized:
            raise RuntimeError("系统未初始化，请先调用setup()方法")
        
        # 获取网格中心坐标
        V_sigma = self.fwd_model.V_sigma
        dof_coordinates = V_sigma.tabulate_dof_coordinates()
        
        # 创建基础导电率分布
        elem_data = np.ones(len(dof_coordinates)) * base_conductivity
        
        # 添加圆形幻影
        for i, coord in enumerate(dof_coordinates):
            x, y = coord[0], coord[1]
            distance = np.sqrt((x - phantom_center[0])**2 + (y - phantom_center[1])**2)
            if distance <= phantom_radius:
                elem_data[i] = phantom_conductivity
        
        return EITImage(elem_data=elem_data, fwd_model=self.fwd_model)
        
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息
        
        返回:
            系统配置信息字典
        """
        info = {
            'n_elec': self.n_elec,
            'pattern_config': self.pattern_config,
            'mesh_config': self.mesh_config,
            'initialized': self._is_initialized
        }
        
        if self._is_initialized:
            info.update({
                'n_elements': len(Function(self.fwd_model.V_sigma).vector()[:]),
                'n_nodes': self.fwd_model.V.dim(),
                'n_measurements': self.fwd_model.pattern_manager.n_meas_total,
                'n_stimulation_patterns': self.fwd_model.pattern_manager.n_stim
            })
        
        return info
