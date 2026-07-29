# EIDORS ↔ PyEIDORS 模型迁移指南

PyEIDORS v2 使用 `Bridge Package v2` 作为双方共同协议。它不是去猜任意
MATLAB 源代码的含义，而是：

1. 在真实 MATLAB/EIDORS 环境中执行用户脚本；
2. 从执行后的工作区发现标准 EIDORS 对象；
3. 保存对象中的网格、电极、刺激、测量和材料参数；
4. 为每个值记录“源字段、推导值、EIDORS 运行时默认、缺失或不支持”；
5. 将“格式可以导入”与“可以做等价 PyEIDORS 正演”分开验证。

因此，常规 EIDORS 脚本通常可以直接迁移，但不能承诺自动理解所有任意
MATLAB 程序、闭包、外部仪器对象或自定义求解器。遇到多个模型或多个图像时，
捕获器会要求用户明确选择，不会静默猜测。

## 可以交换的内容

- 2D 三角形或 3D 四面体网格；
- 边界边/面、每个电极的节点和可选 `electrode.faces`；
- CEM、单节点 PEM 和分布式点电极的源模型分类；
- 每个电极的 `z_contact`、存在性和按维度解释的单位；
- 原始与有效 `stim_pattern`、`meas_pattern`；
- `current_density` 及每个刺激的实际电流摘要；
- `volt_pattern`、`interior_sources` 和刺激标签（保存后可能阻止 PyEIDORS 正演）；
- 背景/目标图像的逐单元 conductivity；
- resistivity、log conductivity 等 EIDORS 图像参数化的标准换算结果；
- `gnd_node`、`normalize_measurements`、求解器、system matrix、Jacobian 等来源信息；
- 均匀、目标和差分边界电压；
- 可选的 PyEIDORS 前向配置和重构预设。

新格式由两个层次组成：

- `Bridge Package v2`：可复制的目录，入口为 `manifest.json`；
- `Geometry v2`：目录内的 `geometry.mat`，格式标识为
  `eidors_pyeidors_geometry_v2`。

旧 `eidors_pyeidors_bridge_v1` 2D MAT 仍可读取。

## 最短路径：EIDORS → PyEIDORS

进入项目标准运行环境：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#complex64-cuda
```

捕获一个普通 EIDORS 脚本：

```bash
pyeidors-interop capture examples/interop/eidors_3d_quickstart.m \
  --output output/my_eidors_bridge \
  --matlab '/mnt/d/Program Files/MATLAB/R2023b/bin/matlab.exe' \
  --eidors-startup '/mnt/d/Program Files/MATLAB/R2023b/toolbox/eidors-v3.12-ng/eidors/startup.m'
```

然后运行三个门禁：

```bash
pyeidors-interop validate output/my_eidors_bridge
pyeidors-interop inspect output/my_eidors_bridge
pyeidors-interop import-geometry output/my_eidors_bridge --forward-smoke
```

没有安装命令入口时可使用：

```bash
python -m pyeidors.interop validate output/my_eidors_bridge
```

`valid=true` 只表示协议和几何合法。`forward_ready=true` 才表示没有已知的
等价正演阻塞项；`forward_smoke=passed` 表示已在导入网格上实际完成一次有限
PyEIDORS 正演。

## 捕获器如何找到模型

捕获器按 EIDORS 对象本身发现内容，不依赖固定变量名：

- `type='fwd_model'` 或具有标准 `nodes`、`elems` 的结构；
- `type='image'`、带 `fwd_model` 及 EIDORS 图像参数字段的结构；
- `type='data'` 或具有 `meas` 的结构；
- `imdl.fwd_model`、`img.fwd_model` 等嵌套标准模型。

若工作区只有一个标准模型，会自动选择。图像角色可由常见变量名
`img_bg`、`img_truth`、`vh`、`vi` 等推断，并在元数据中标记为 `inferred`。
若候选不唯一，命令会列出候选路径并失败。此时明确指定：

```bash
pyeidors-interop capture my_model.m \
  --output output/my_bridge \
  --matlab '<MATLAB executable>' \
  --eidors-startup '<EIDORS startup.m>' \
  --fwd-model-var imdl.fwd_model \
  --background-image-var img_bg \
  --target-image-var img_truth \
  --homogeneous-data-var vh \
  --target-data-var vi
```

这套机制比正则表达式解析 `.m` 源码可靠，因为循环、函数调用和条件分支只有执行
后才有确定结果。但下列情况不能无条件自动迁移：

- 脚本结束前清空或隐藏了所有标准 EIDORS 对象；
- 模型只存在于无法访问的嵌套函数/闭包或外部进程；
- 电极是仪器通道字符串而非数值网格节点；
- 自定义 solver 改写了标准 `fwd_model` 语义；
- 同一工作区有多个候选且用户没有选择；
- 图像参数化无法通过 EIDORS `data_mapper` 和 `convert_img_units` 转成逐单元
  conductivity。

## CEM、PEM 和电极坐标

EIDORS 电极的标准判定与迁移策略如下。

| EIDORS 源对象 | 捕获分类 | 几何能否导入 | 等价 PyEIDORS 正演 |
| --- | --- | --- | --- |
| `electrode(i).faces` | `cem_faces` | 是，保存 faces 和节点并集 | 可以 |
| 多节点且覆盖完整边界 facet | `cem` | 是，保存精确表面节点 | 可以 |
| 单节点 | `point` | 是，保存精确点坐标/节点 | 默认阻止 |
| 多节点但不包含完整边界 facet | `distributed_point` | 是 | 默认阻止 |

PyEIDORS 当前正演使用正面积 CEM。把 PEM 点电极映射到相邻 boundary facets
会改变电极模型，因此不会默认发生。若用户明确接受这一近似，可运行：

```bash
pyeidors-interop import-geometry output/pem_bridge \
  --forward-smoke \
  --allow-point-electrode-projection
```

报告会写出：

- 精确表面电极：`electrode_projection=exact_surface_nodes`；
- 明确接受的点电极近似：`electrode_projection=incident_boundary_facets`。

`z_contact` 不存在通用 EIDORS 默认值。捕获器保存每个电极的原值以及
`contact_impedance_present`；缺失时写 `NaN + false` 并阻止正演，不会补
`0.01`。按 EIDORS 一阶系统矩阵约定，其量纲为：

- 2D：`Ω·源长度单位`；
- 3D：`Ω·源长度单位²`。

若模型没有声明坐标单位，报告只写 `source_length_unit`，不会假装是米。

## 注入电流到底保存什么

EIDORS 的权威输入是每个 `fmdl.stimulation(i).stim_pattern`。捕获器保存：

- `stim_matrix_raw`：模型对象中的原始刺激矩阵；
- `current_density`、`current_density_present`；
- `stim_matrix`：标准 EIDORS 一阶求解器实际使用的有效刺激矩阵；
- `stim_positive_current` / `stim_negative_current`；
- `stim_net_current` / `stim_max_abs_current` / `stim_balanced`。

当 `fwd_model.current_density` 是有限正标量时，EIDORS 在求解时使用：

```text
effective stim_pattern = raw stim_pattern / current_density
```

例如示例中的原始幅值为 `0.02 A`，`current_density=2`，导入 PyEIDORS 的
有效幅值必须为 `0.01 A`。

`mk_stim_patterns(..., 0.01)` 中的 `0.01 A` 只是该生成器第六参数省略时的
默认值，不能由此推断任意现有模型都是 `0.01 A`。捕获器没有刺激时也不会
自行补这个数值。

电流模式是 PyEIDORS 当前可执行的等价路线。若源刺激包含：

- `volt_pattern`；
- `interior_sources`；
- 复数刺激/测量矩阵；
- 缺失或非标准刺激字段；

这些数据仍会保存在 MAT 中，但 `stimulation_supported=false`，等价
PyEIDORS 正演会被阻止，而不是丢字段后继续计算。

## 背景和目标电导率

捕获器不使用“中位数”“全 1”或其他猜测来补图像。它按 EIDORS 标准路径处理：

1. 执行图像自身的 `params_mapping.function`（若存在）；
2. 在适用时执行 `fwd_model.coarse2fine` 和 `fwd_model.background`；
3. 调用 EIDORS `data_mapper`；
4. 调用 `convert_img_units(..., 'conductivity')`；
5. 保存每个 forward element 的 conductivity。

因此 `conductivity`、`resistivity`、`log_conductivity`、
`log10_conductivity`、`log_resistivity` 和 `log10_resistivity` 会按
EIDORS 自己的转换语义进入协议。均匀图像另外保存 scalar `background`；
非均匀背景仍保存完整 `background_elem_data`，但当前 GUI 的单标量背景正演会
以 `background_is_nonuniform_and_not_gui_scalar_compatible` 阻止，避免取中值。

背景/目标角色若由变量名推断，元数据状态为 `derived`；由用户 selector 指定时
为 `exact`；缺失或转换失败分别为 `missing` / `unsupported`。

## 常见正演参数的处理

| 参数 | 处理 |
| --- | --- |
| `nodes` / `elems` / `boundary` | 精确保存，MATLAB 1-based |
| 电极数和节点/坐标 | 从实际 `fmdl.electrode` 保存，不从脚本文本猜 |
| `electrode.faces` | 保存 face connectivity 和节点并集 |
| `z_contact` | 每电极值、存在性、单位 |
| `stimulation` | 原始/有效刺激、测量矩阵和电流摘要 |
| `current_density` | 保存并按标准 EIDORS 语义应用一次 |
| `gnd_node` | 保存源值；一阶 solver 缺失时记录中心节点运行时推导 |
| `normalize_measurements` | 保存源存在性以及调用 `mdl_normalize` 得到的有效值 |
| `solve/system_mat/jacobian` | 保存声明值和可解析的 EIDORS 默认值 |
| `measured_quantity` / `units` | 有则保存，无则明确 `unspecified` |
| `coarse2fine` / `background` | 在图像映射中按 EIDORS 顺序应用并记录 |
| `model_reduction` | 记录存在性、类型/函数和尺寸；不把性能优化当作新物理参数 |
| `potential_order` | EIDORS 标准模型无统一可移植字段，当前不猜测 |

缺失 `gnd_node` 在标准 `fwd_solve_1st_order` 下可按 EIDORS 源码记录中心最近
节点；自定义 solver 下行为未知则阻止。缺失 `normalize_measurements` 时保存
捕获当时 `eidors_default` 的有效值和来源，不把它伪装成源模型显式设置。

## 示例矩阵

- `examples/interop/eidors_2d_quickstart.m`：2D CEM、均匀背景和目标；
- `examples/interop/eidors_3d_quickstart.m`：3D surface CEM；
- `examples/interop/eidors_3d_point_electrode_quickstart.m`：3D PEM、
  resistivity→conductivity、`current_density`；
- `examples/interop/eidors_missing_fields_semantics.m`：故意缺少
  `z_contact/gnd_node/normalize_measurements`，验证不伪造默认值；
- `examples/interop/eidors_multiple_models_requires_selector.m`：两个模型候选，
  验证歧义失败及 `--fwd-model-var` 明确选择；
- `examples/interop/pyeidors_3d_export.py`：PyEIDORS → EIDORS。

维护者可运行真实 MATLAB/EIDORS 语义验收：

```bash
python scripts/interop/run_eidors_source_semantics_acceptance.py \
  --output output/eidors_source_semantics_acceptance \
  --matlab '/mnt/d/Program Files/MATLAB/R2023b/bin/matlab.exe' \
  --eidors-startup '/mnt/d/Program Files/MATLAB/R2023b/toolbox/eidors-v3.12-ng/eidors/startup.m'
```

它同时验证 2D/3D CEM、PEM、缺失字段、`0.02/2=0.01 A`、resistivity 转换、
默认 PEM 阻止和显式 PEM 投影正演。

本次真实环境结果记录在
`reports/interop/eidors_source_semantics_acceptance_20260730.md`。

## GUI 路径

1. 启动 `./eit-gui --gpu`。
2. 打开“工具 → EIDORS 互操作”。
3. 选择 EIDORS `.m`、Bridge 目录或单个 `geometry.mat`。
4. 首次使用 `.m` 时选择 MATLAB 和 EIDORS `startup.m`。
5. 预览节点、单元、电极、测量数以及 `forward_blockers`。
6. 只有无阻塞项的模型才应用到等价仿真/数据集正演。

导入后的 `mesh_source=interop`。worker 直接使用导入的 `EITMesh`，不会重新
生成圆形或圆柱近似网格。

## PyEIDORS → EIDORS

GUI 中完成仿真后打开“工具 → EIDORS 互操作”，选择当前 Simulation 和输出
目录，再在 MATLAB 中运行包内 `run_in_eidors.m`。

脚本会恢复：

- 实际网格、boundary facets 和 CEM 电极节点/faces；
- 接触阻抗及其存在性门禁；
- 明确记录的 ground/normalize 语义；
- 自定义刺激/测量矩阵，或由明确 PyEIDORS 配置生成的协议；
- CSV 实数测量或 MAT 复数测量。

无 GUI 示例：

```bash
python examples/interop/pyeidors_3d_export.py \
  --output output/pyeidors_3d_bridge \
  --eidors-startup '/mnt/d/path/to/eidors/startup.m'
```

真实 EIDORS 复核：

```matlab
addpath('examples/interop');
validate_bridge_in_eidors( ...
    'output/pyeidors_3d_bridge/run_in_eidors.m');
```

## Python API

导入 Geometry MAT：

```python
from pathlib import Path

from pyeidors.interop import build_mesh_from_exchange_mat

mesh, metadata = build_mesh_from_exchange_mat(Path("bridge/geometry.mat"))
print(metadata["n_elec"])
```

读取完整包并检查正演就绪性：

```python
from eit_app.interop import InteropBundleImporter, validate_bridge_package

report = validate_bridge_package("bridge")
if not report["valid"]:
    raise RuntimeError(report["errors"])

loaded, preview = InteropBundleImporter().preview_package("bridge")
config = preview.forward_model_config
config.require_interop_forward_ready()
```

PEM 投影必须显式开启：

```python
config = config.with_overrides(allow_interop_approximations=True)
config.require_interop_forward_ready()
```

## Bridge Package v2

```text
bridge/
├── manifest.json
├── geometry.mat
├── config.json
├── measurements.csv              # 可选，实数
├── measurements.mat              # 可选，复数或非 CSV 数据
├── reconstruction_preset.json    # 可选
├── capture_report.json           # EIDORS 捕获选择/阻塞报告
├── run_capture_from_eidors.m
└── run_in_eidors.m
```

正式 schema：

- `schemas/interop/eidors_pyeidors_bridge_v2.schema.json`
- `schemas/interop/eidors_pyeidors_geometry_v2.schema.json`

manifest 只允许安全相对路径；本机 MATLAB/EIDORS 绝对路径属于运行配置，不是
可移植几何协议。

## Geometry v2 核心字段

| 字段 | 规则 |
| --- | --- |
| `nodes` / `elems` | 2D/3D simplex，1-based connectivity |
| `boundary_facets` | 2D edge 或 3D triangle |
| `electrode_nodes/counts` | 每电极 padded 节点及有效长度 |
| `electrode_faces/counts` | 可选 EIDORS face 定义 |
| `electrode_model` | `cem/cem_faces/point/distributed_point` |
| `contact_impedance/present/unit` | 源值、存在性和量纲 |
| `stim_matrix_raw` | EIDORS 模型中的原始电流模式 |
| `stim_matrix` | PyEIDORS 应采用的有效电流模式 |
| `meas_matrices/counts` | 每次 stimulation 的测量矩阵 |
| `current_density*` | 原值、存在性、是否已应用 |
| `stim_*_current` | 每个刺激的电流幅值和平衡摘要 |
| `background_elem_data` | 映射后的逐单元背景 conductivity |
| `target_elem_data` | 映射后的逐单元目标 conductivity |
| `*_present` | 区分源值、缺失和空数组 |
| `capture_metadata_json` | 选择方式、来源、solver 和阻塞项 |

## 能声明什么，不能声明什么

协议门禁可以声明：

- 网格拓扑/坐标、电极和刺激测量矩阵已被确定地迁移；
- 原始值、EIDORS 有效值、运行时默认和缺失值没有混为一谈；
- 2D/3D CEM 可进入真实 PyEIDORS 计算链；
- PEM 只有经明确同意才转换为 incident-facet CEM 近似。

它不能仅凭格式 roundtrip 声明两个框架的逆问题结果逐元素相等。正则化、
先验、归一化、solver、离散空间和浮点精度仍需单独的数值等价报告。
