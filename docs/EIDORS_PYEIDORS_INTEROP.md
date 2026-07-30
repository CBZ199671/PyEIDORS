# EIDORS ↔ PyEIDORS Bridge v3 新手指南

Bridge v3 是双方共同的、可校验的模型交换协议，不是一个只保存节点和单元的
临时 MAT 文件。PyEIDORS v2 只接受
`eidors_pyeidors_bridge_v3`；Bridge v1/v2 和独立 `geometry.mat` 会明确报错，
不会静默升级或猜测。

## 最简单的方法：GUI 一键加载

1. 启动 EIT Workstation。
2. 选择“工具 → 加载 EIDORS 模型…”（`Ctrl+I`）。
3. 选择 Bridge v3 包目录，或选择一个普通 EIDORS `.m` 脚本。
4. `.m` 脚本需要 MATLAB 和 EIDORS `startup.m`。PyEIDORS 会启动全新 MATLAB
   进程，在临时工作目录中捕获标准对象；脚本仍可能产生其自行编写的外部副作用，
   GUI 会在首次执行前确认。
5. 查看预览中的电极类型/节点/面/权重、接触阻抗、刺激电流、测量数、背景/目标
   场、哈希和阻断项。
6. 选择“仿真”“数据集”“实时成像”或“应用到全部”，然后点击加载。

加载成功后，包会被复制到只读模型资产库，并以 `model_id` 去重。原包被移动或删除
不会影响运行。“工具 → Bridge v3 模型资产管理”可以检查资产和三个流程的当前
绑定。

多模型脚本不会由名称猜测。错误信息会列出完整候选路径；将所需路径填入 GUI 的
“模型变量/完整路径”“背景图像变量/完整路径”等选择框后重试。

## 五分钟：EIDORS → PyEIDORS

### GUI

直接选择示例：

```text
examples/interop/eidors_3d_point_electrode_quickstart.m
```

设置 MATLAB 和 `<EIDORS>/startup.m`，预览后“应用到仿真”。这是原生 PEM
模型；导入后仍是 PEM，不会投影成 CEM。

### CLI

所有运行命令都从项目的 Nix 环境执行：

```bash
nix develop .#complex64-cuda --command pyeidors-interop capture \
  examples/interop/eidors_3d_point_electrode_quickstart.m \
  --output output/eidors_pem_v3 \
  --matlab matlab \
  --eidors-startup /path/to/eidors/startup.m

nix develop .#complex64-cuda --command pyeidors-interop validate \
  output/eidors_pem_v3
nix develop .#complex64-cuda --command pyeidors-interop inspect \
  output/eidors_pem_v3
nix develop .#complex64-cuda --command pyeidors-interop register \
  output/eidors_pem_v3 --name "EIDORS PEM example"
```

若脚本有多个候选，使用 `--fwd-model-var`、`--background-image-var`、
`--target-image-var`、`--homogeneous-data-var`、`--target-data-var` 明确选择。

### 在 MATLAB 中直接导出标准对象

把仓库的 `matlab/` 目录加入 MATLAB 路径：

```matlab
addpath('/path/to/PyEIDORS/matlab');
package_dir = pyeidors_export_v3(inv_model, 'output/my_model_v3', ...
    'Background', img_h, ...
    'Target', img_i, ...
    'EidorsStartup', '/path/to/eidors/startup.m');
```

`source` 可以是 `fwd_model`、`inv_model` 或 `image`。背景和目标变量是显式的，
因此不会误选同一工作区中的其他图像。

## 五分钟：PyEIDORS → EIDORS

先生成一个包含 CEM 与加权 PEM 的 v3 包：

```bash
nix develop .#complex64-cuda --command python \
  examples/interop/pyeidors_mixed_export.py \
  --output output/pyeidors_mixed_v3 \
  --eidors-startup /path/to/eidors/startup.m

nix develop .#complex64-cuda --command pyeidors-interop validate \
  output/pyeidors_mixed_v3
```

然后在 MATLAB/EIDORS 中：

```matlab
run('/path/to/eidors/startup.m');
addpath('/path/to/PyEIDORS/matlab');
imported = pyeidors_import_v3('/path/to/output/pyeidors_mixed_v3');
fmdl = imported.fwd_model;
img_h = imported.background_image;
img_i = imported.target_image;
vh = imported.homogeneous_data;
vi = imported.target_data;
```

也可以运行包中的 `run_in_eidors.m`。加权 PEM 在 EIDORS 结构中会展开为等价的
物理点电极组合，但 Bridge v3 中的逻辑电极顺序、权重和回读语义保持不变。

## v3 包里有什么

| 文件 | 权威内容 |
|---|---|
| `manifest.json` | 文件大小/SHA-256、来源、能力、阻断项、四类身份哈希 |
| `model.json` | 维度、P1、ground、单位、归一化、可运行性 |
| `geometry.mat` | 节点、单元、外部/内部 CEM 面、逐电极类型、PEM 权重、z |
| `protocol.mat` | raw/effective stimulation、measurement、N2E/QQ/VV/v2meas、选择器 |
| `fields.mat` | 背景/目标逐单元电导率、参数化来源、`coarse2fine` |
| `measurements.mat` | 可选参考/目标/差分测量，保留形状和复数 dtype |

身份的含义：

- `model_id`：整个语义包的不可变身份；
- `forward_fingerprint`：网格、电极、ground、背景和绝对刺激物理；
- `protocol_layout_hash`：忽略幅值的刺激/测量拓扑和顺序；
- `protocol_physics_hash`：包含有效注入电流。

## 电极语义

### PEM

单节点 PEM 和多节点加权 PEM 都使用 EIDORS 的精确 `N2E` 语义：

```text
Q += Wᵀ I
U = W u
```

没有 incident-facet 投影，也不需要用户同意近似。PEM 上的 `z_contact` 只保存
来源；EIDORS 的 PEM 一阶系统不使用它，PyEIDORS 也不把它放进矩阵或缓存。

### CEM

CEM 使用原始外部边界面或 `system_mat_fields.CEM_boundary` 内部面，以及每个电极
自己的 `z_contact`。CEM 缺失 `z_contact` 时导入会阻断，因为 EIDORS 没有一个
可移植的通用默认值。

### mixed

每个逻辑电极独立声明 `cem` 或 `pem`。CEM 电流进入电极未知量，PEM 电流进入
节点右端项，结果仍按原逻辑电极顺序输出。

## 刺激、测量和实时数据

捕获器调用 EIDORS `fwd_model_parameters`，保存实际运行时的 `N2E`、`QQ`、
`VV`、`v2meas`、`normalize`、`meas_select` 和 `current_density`，同时保留 raw
刺激。注入电流不会由一个模糊的 GUI 数字猜测。

实时/历史数据只有在刺激行和测量行可以唯一证明为排列及符号变化时才自动映射。
重复、缺失或歧义立即阻断。实际电流优先级是：

```text
逐帧元数据 > 会话元数据 > 设备配置
```

每个刺激行必须是模型行的有限、非零实数倍。PyEIDORS 用实际电流重建刺激与
Jacobian，并记录原值、实际值、比例和运行时指纹。数据库参考/目标若使用不同
映射或不同实际电流也会阻断。

## 各业务流程的默认行为

- 仿真：使用包内背景场和目标场；只有显式选择均匀场、绘制目标或随机目标才覆盖。
- 数据集：包内背景是参考，包内目标是默认真值。
- 数据库：旧会话保持“未绑定”；完成协议验证后才保存 `model_id` 和通道证明。
- 实时：使用绑定模型的正问题、协议和当前帧实际电流，不退回生成网格。

`meas_select → difference/normalize → inverse`、参考背景和 `coarse2fine` 会迁移；
具体 EIDORS 重建算法和超参数不会被暗中映射。

## 会被保留但阻断等价正演的情况

- voltage stimulation；
- interior source；
- instrument/extra nodes；
- 非 P1 空间；
- 无法复现的自定义 solver、system matrix 或 Jacobian。

这些信息会留在审计/预览中，但 `forward_ready=false`。PyEIDORS 不会删掉字段后
假装模型等价。

## 常见错误

- `Only eidors_pyeidors_bridge_v3...`：所选是 v1/v2 或独立 MAT；请从源对象重新
  导出 v3。
- `Multiple EIDORS ... objects`：填写完整候选变量路径，不要只写一个模糊短名。
- `contact_impedance_missing_no_eidors_default`：至少一个 CEM 电极缺少 z。
- `protocol ... ambiguous/missing`：硬件通道无法唯一对应模型，不能按点数硬套。
- `integrity ... failed`：托管包被修改或损坏，必须从可信源重新注册。
