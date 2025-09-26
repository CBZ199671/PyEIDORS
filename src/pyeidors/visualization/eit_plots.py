"""EIT可视化模块 - 提供网格、导电率分布和测量数据的可视化功能"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from typing import Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

# 检查可选依赖
try:
    from fenics import Function, plot as fenics_plot
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EITVisualizer:
    """EIT可视化器 - 提供多种EIT相关的可视化功能"""
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        初始化可视化器
        
        参数:
            style: matplotlib样式
            figsize: 默认图像尺寸
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib不可用，无法进行可视化")
        
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            logger.warning(f"样式 {style} 不可用，使用默认样式")
    
    def plot_mesh(self, mesh, title: str = "网格结构", 
                  show_electrodes: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制网格结构
        
        参数:
            mesh: FEniCS网格对象
            title: 图像标题
            show_electrodes: 是否显示电极位置
            save_path: 保存路径（可选）
            
        返回:
            matplotlib图像对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 获取网格坐标和连接关系
        coordinates = mesh.coordinates()
        cells = mesh.cells()
        
        # 创建三角网格
        triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)
        
        # 绘制网格
        ax.triplot(triangulation, 'k-', alpha=0.3, linewidth=0.5)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=1, c='blue', alpha=0.6)
        
        # 如果有电极信息，绘制电极位置
        if show_electrodes and hasattr(mesh, 'vertex_elec') and mesh.vertex_elec:
            self._plot_electrodes(ax, mesh.vertex_elec)
        
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_conductivity(self, mesh, conductivity: Union[Function, np.ndarray], 
                         title: str = "导电率分布", colormap: str = 'viridis',
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制导电率分布
        
        参数:
            mesh: FEniCS网格对象
            conductivity: 导电率分布（Function对象或numpy数组）
            title: 图像标题
            colormap: 颜色映射
            save_path: 保存路径（可选）
            
        返回:
            matplotlib图像对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 处理不同类型的导电率输入
        if isinstance(conductivity, Function):
            conductivity_values = conductivity.vector()[:]
        else:
            conductivity_values = np.array(conductivity)
        
        # 获取网格信息
        coordinates = mesh.coordinates()
        cells = mesh.cells()
        
        # 创建三角网格
        triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)
        
        # 如果导电率是单元值，需要插值到节点
        if len(conductivity_values) == mesh.num_cells():
            # 单元中心值插值到节点
            node_values = self._interpolate_cell_to_node(mesh, conductivity_values)
        else:
            node_values = conductivity_values
        
        # 绘制导电率分布
        im = ax.tripcolor(triangulation, node_values, cmap=colormap, shading='gouraud')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('导电率 (S/m)', fontsize=12)
        
        # 绘制网格轮廓
        ax.triplot(triangulation, 'k-', alpha=0.2, linewidth=0.3)
        
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_measurements(self, data, title: str = "测量数据", 
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制测量数据
        
        参数:
            data: EITData对象或测量数组
            title: 图像标题
            save_path: 保存路径（可选）
            
        返回:
            matplotlib图像对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # 提取测量数据
        if hasattr(data, 'meas'):
            measurements = data.meas
        else:
            measurements = np.array(data)
        
        # 绘制测量数据时序图
        ax1.plot(measurements, 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_title('测量序列', fontweight='bold')
        ax1.set_xlabel('测量索引')
        ax1.set_ylabel('电压 (V)')
        ax1.grid(True, alpha=0.3)
        
        # 绘制测量数据直方图
        ax2.hist(measurements, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('测量数据分布', fontweight='bold')
        ax2.set_xlabel('电压 (V)')
        ax2.set_ylabel('概率密度')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(measurements)
        std_val = np.std(measurements)
        ax2.axvline(mean_val, color='red', linestyle='--', 
                   label=f'均值: {mean_val:.4f}')
        ax2.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7,
                   label=f'±标准差: {std_val:.4f}')
        ax2.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
        ax2.legend()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_reconstruction_comparison(self, mesh, true_conductivity, reconstructed_conductivity,
                                     title: str = "重建结果对比", save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制真实分布与重建分布的对比
        
        参数:
            mesh: FEniCS网格对象
            true_conductivity: 真实导电率分布
            reconstructed_conductivity: 重建导电率分布
            title: 图像标题
            save_path: 保存路径（可选）
            
        返回:
            matplotlib图像对象
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 处理输入数据
        if isinstance(true_conductivity, Function):
            true_values = true_conductivity.vector()[:]
        else:
            true_values = np.array(true_conductivity)
            
        if isinstance(reconstructed_conductivity, Function):
            recon_values = reconstructed_conductivity.vector()[:]
        else:
            recon_values = np.array(reconstructed_conductivity)
        
        # 获取网格信息
        coordinates = mesh.coordinates()
        cells = mesh.cells()
        triangulation = tri.Triangulation(coordinates[:, 0], coordinates[:, 1], cells)
        
        # 确定颜色范围
        vmin = min(np.min(true_values), np.min(recon_values))
        vmax = max(np.max(true_values), np.max(recon_values))
        
        # 绘制真实分布
        im1 = axes[0].tripcolor(triangulation, true_values, cmap='viridis', 
                               vmin=vmin, vmax=vmax, shading='gouraud')
        axes[0].set_title('真实分布', fontweight='bold')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # 绘制重建分布
        im2 = axes[1].tripcolor(triangulation, recon_values, cmap='viridis', 
                               vmin=vmin, vmax=vmax, shading='gouraud')
        axes[1].set_title('重建分布', fontweight='bold')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # 绘制误差分布
        error = np.abs(true_values - recon_values)
        im3 = axes[2].tripcolor(triangulation, error, cmap='hot', shading='gouraud')
        axes[2].set_title('绝对误差', fontweight='bold')
        axes[2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # 计算重建误差指标
        relative_error = np.linalg.norm(error) / np.linalg.norm(true_values)
        
        fig.suptitle(f'{title} (相对误差: {relative_error:.4f})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_electrodes(self, ax, electrode_vertices):
        """绘制电极位置"""
        for i, electrode in enumerate(electrode_vertices):
            if electrode:  # 检查电极是否有顶点
                electrode_array = np.array(electrode)
                ax.plot(electrode_array[:, 0], electrode_array[:, 1], 'ro-', 
                       markersize=6, linewidth=2, label=f'电极 {i+1}' if i < 5 else "")
        
        if len(electrode_vertices) <= 5:
            ax.legend()
    
    def _interpolate_cell_to_node(self, mesh, cell_values):
        """将单元中心值插值到节点"""
        node_values = np.zeros(mesh.num_vertices())
        node_counts = np.zeros(mesh.num_vertices())
        
        for cell_idx, cell in enumerate(mesh.cells()):
            for vertex_idx in cell:
                node_values[vertex_idx] += cell_values[cell_idx]
                node_counts[vertex_idx] += 1
        
        # 避免除零
        node_counts[node_counts == 0] = 1
        node_values /= node_counts
        
        return node_values
    
    def plot_convergence(self, iterations, errors, title: str = "收敛曲线", 
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制算法收敛曲线
        
        参数:
            iterations: 迭代次数数组
            errors: 对应的误差值
            title: 图像标题
            save_path: 保存路径（可选）
            
        返回:
            matplotlib图像对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(iterations, errors, 'b-o', linewidth=2, markersize=6)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('误差 (对数坐标)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def create_visualizer(style: str = 'seaborn') -> EITVisualizer:
    """创建EIT可视化器实例"""
    return EITVisualizer(style=style)