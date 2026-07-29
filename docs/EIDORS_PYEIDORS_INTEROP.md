# EIDORS ↔ PyEIDORS 模型迁移指南

这是一条面向新用户的正式迁移路径。它可以交换：

- 2D 三角形或 3D 四面体网格；
- 边界边/面与电极节点；
- 每个电极的接触阻抗；
- EIDORS 实际使用的 stimulation / measurement 矩阵；
- 均匀、目标和差分边界电压；
- 可选的重构默认参数。

新格式由两个层次组成：

- `Bridge Package v2`：一个可复制的目录，入口是 `manifest.json`；
- `Geometry v2`：目录内的 `geometry.mat`，格式标识为
  `eidors_pyeidors_geometry_v2`。

旧的 `eidors_pyeidors_bridge_v1` 2D MAT 仍可读取。

## 最短路径：从 EIDORS 切换到 PyEIDORS

先进入 PyEIDORS 的标准运行环境：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#complex64-cuda
```

准备一个普通 EIDORS 脚本。脚本无需调用 PyEIDORS，只要执行结束后工作区中
存在以下常见变量之一即可：

- 模型：`fmdl`、`imdl.fwd_model` 或 `img.fwd_model`；
- 图像（可选）：`img_truth`、`img` 或 `img_bg`；
- 电压（可选）：`vh` / `vhom` 与 `vi` / `vtarget`。

仓库提供可直接运行的示例：

- `examples/interop/eidors_2d_quickstart.m`
- `examples/interop/eidors_3d_quickstart.m`

在 WSL2 中执行捕获命令（路径按本机实际位置替换）：

```bash
pyeidors-interop capture examples/interop/eidors_3d_quickstart.m \
  --output output/my_eidors_3d_bridge \
  --matlab '/mnt/d/Program Files/MATLAB/R2023b/bin/matlab.exe' \
  --eidors-startup '/mnt/d/Program Files/MATLAB/R2023b/toolbox/eidors-v3.12-ng/eidors/startup.m'
```

捕获器运行原脚本，不要求用户改写模型构造方式。生成后依次验证、查看和导入：

```bash
pyeidors-interop validate output/my_eidors_3d_bridge
pyeidors-interop inspect output/my_eidors_3d_bridge
pyeidors-interop import-geometry output/my_eidors_3d_bridge --forward-smoke
```

也可以不用安装的命令名：

```bash
python -m pyeidors.interop validate output/my_eidors_3d_bridge
```

`import-geometry --forward-smoke` 会：

1. 将 MAT 的 1-based connectivity 转为 DOLFINx；
2. 建立真实的 2D/3D `EITMesh`；
3. 逐个匹配边界实体和电极标签；
4. 使用捕获到的 stimulation / measurement 矩阵；
5. 在导入网格上运行一次均匀介质 CEM 前向求解。

命令返回 JSON；`valid=true` 或 `forward_smoke=passed` 才表示对应门禁通过。

## GUI 路径

1. 启动 `./eit-gui --gpu`（无 GPU 时使用 `./eit-gui --cpu`）。
2. 打开“工具 → EIDORS 互操作”。
3. 选择 EIDORS `.m` 脚本、已有 Bridge 目录或单个 `geometry.mat`。
4. 首次使用 `.m` 时选择 MATLAB 可执行文件和 EIDORS `startup.m`。
5. 点击预览，检查维度、节点、单元、电极和测量点数。
6. 应用到“仿真”或“数据集”。

应用到仿真/数据集后，配置中的 `mesh_source` 为 `interop`。后台 worker 直接执行
`EITSystem.setup(mesh=imported_eit_mesh)`，不会重新生成近似圆形或圆柱网格。
如果移动了包目录，请重新加载包；错误信息会指出缺失的 `geometry.mat`。

## 从 PyEIDORS 导出到 EIDORS

GUI 中先完成一次仿真，然后：

1. 打开“工具 → EIDORS 互操作”；
2. 切换到“导出到 EIDORS”；
3. 选择当前 Simulation；
4. 选择输出目录并导出；
5. 在 MATLAB 中运行输出目录内的 `run_in_eidors.m`。

PyEIDORS 的前向结果会携带实际边界 facets 和实际电极节点；导出器优先使用这些
标签，不再仅根据角度猜测电极。`run_in_eidors.m` 会重建 `fmdl`，并在存在协议
矩阵时逐个重建 `fmdl.stimulation(i).stim_pattern` 和
`meas_pattern`。若包中包含测量数据，还会生成 `vh` 与 `vi`。

无 GUI 的最小 3D 导出示例：

```bash
python examples/interop/pyeidors_3d_export.py \
  --output output/pyeidors_3d_bridge \
  --eidors-startup '/mnt/d/path/to/eidors/startup.m'
```

在 MATLAB/EIDORS 中可用随仓库提供的验收函数复核精确边界、电极、刺激/测量
矩阵及一次有限前向求解：

```matlab
addpath('examples/interop');
validate_bridge_in_eidors( ...
    'output/pyeidors_3d_bridge/run_in_eidors.m');
```

它会在包目录写出 `eidors_import_report.json`；任一计数、协议或前向有限性
检查失败都会抛出错误。

## Python API

直接导入单个 Geometry v1/v2 MAT：

```python
from pathlib import Path

from pyeidors import EITSystem
from pyeidors.interop import build_mesh_from_exchange_mat

mesh, metadata = build_mesh_from_exchange_mat(Path("bridge/geometry.mat"))

system = EITSystem(n_elec=mesh.n_electrodes)
system.setup(mesh=mesh, initialize_inverse=False)
```

读取完整 Bridge Package：

```python
from eit_app.interop import InteropBundleImporter, validate_bridge_package

report = validate_bridge_package("bridge")
if not report["valid"]:
    raise RuntimeError(report["errors"])

loaded, preview = InteropBundleImporter().preview_package("bridge")
config = preview.forward_model_config
assert config.mesh_source == "interop"
```

## Bridge Package v2 目录

```text
bridge/
├── manifest.json
├── geometry.mat
├── config.json
├── measurements.csv              # 可选
├── reconstruction_preset.json    # 可选
├── run_capture_from_eidors.m      # EIDORS 捕获包
└── run_in_eidors.m                # PyEIDORS 导出包
```

正式 schema：

- `schemas/interop/eidors_pyeidors_bridge_v2.schema.json`
- `schemas/interop/eidors_pyeidors_geometry_v2.schema.json`

所有 manifest 文件路径必须为安全的相对路径。包可以整体移动或压缩，不允许把
开发者本机的 MATLAB/EIDORS/项目绝对路径当作协议的一部分。

## Geometry v2 关键字段

| 字段 | 规则 |
| --- | --- |
| `exchange_format` | `eidors_pyeidors_geometry_v2` |
| `schema_version` | `2` |
| `index_base` | `1` |
| `dimension` | `2` 或 `3` |
| `cell_type` | 2D=`triangle`，3D=`tetrahedron` |
| `nodes` | `N×2` 或 `N×3` |
| `elems` | 2D `M×3`、3D `M×4`，1-based |
| `boundary_facets` | 2D `K×2`、3D `K×3`，1-based |
| `boundary_edges` | 兼容别名；若同时存在，必须与 `boundary_facets` 表示同一集合 |
| `electrode_nodes` | 每行一个电极的 padded 1-based 节点 |
| `electrode_node_counts` | 每行有效节点数 |
| `contact_impedance` | 标量或每电极一个值 |
| `stim_matrix` | 可选，`n_stim×n_electrodes` |
| `meas_matrices` | 可选，padded `n_stim×max_n_meas×n_electrodes` |
| `measurement_counts` | 可选，每次 stimulation 的有效 measurement 行数 |

导入器会验证索引范围、数组宽度、维度/单元类型组合、边界别名一致性、电极数量，
并要求每个源边界 facet 在 DOLFINx 网格中恰好匹配。

若 EIDORS 模型使用单节点点电极（例如部分 `mk_common_model` 3D 模型），
PyEIDORS 会把每个点显式投影到相邻 boundary facets，并在报告中写出
`electrode_projection=incident_boundary_facets`。这是为了进入正面积 CEM 的
离散转换，不属于 exact-surface parity；surface-electrode 模型的报告为
`exact_surface_nodes`。

## 能声明什么，不能声明什么

通过 `validate`、精确网格导入和双向脚本运行，可以声明：

- 包结构、网格 topology/geometry、电极和协议可迁移；
- 2D 三角形与 3D 四面体都进入真实 PyEIDORS 计算链；
- 两端可以在同一离散模型上运行。

这不自动证明两个框架的逆问题结果逐元素相等。正则化、归一化、先验、求解器和
浮点精度仍会影响重构。任何“数值等价”结论必须另行使用相同测量、算法和容差
报告验证，不能由格式 roundtrip 推导。
