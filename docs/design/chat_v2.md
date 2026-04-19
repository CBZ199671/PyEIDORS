# Chat

_Started 2026-04-19 08:30 UTC_

---

## User



## Assistant

I'll start by exploring the codebase to understand what we're working with.

_[tool: update_todos]_

_[tool: local_ls]_

This is a PyQt desktop application. Let me explore the key UI files.

_[tool: local_read]_

_[tool: local_read]_

_[tool: local_ls]_

_[tool: local_read]_

_[tool: local_read]_

_[tool: local_read]_

I have a solid grasp of the app now. Let me snip older exploration and continue building.

_[tool: snip]_

_[tool: update_todos]_

_[tool: set_project_title]_

_[tool: local_ls]_

_[tool: local_ls]_

I have enough context from the codebase: a PyQt6 desktop app called "EIT 工作站" (EIT Workstation) for Electrical Impedance Tomography simulation and reconstruction. It has 4 tabs (实测/仿真/数据集/数据库), light/dark themes, bilingual zh/en, blue accent palette (#1f5d8b), Times New Roman + Microsoft YaHei fonts, and a step-based workflow design.

Let me build the design system now.

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: update_todos]_

I'll build compact preview cards and the UI kit efficiently.

_[tool: snip]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: register_assets]_

_[tool: snip]_

_[tool: update_todos]_

Let me check what I have in context about the EIT app structure before building the UI kit.

_[tool: read_file]_

_[tool: list_files]_

_[tool: read_file]_

Now building the UI kit for the EIT Workstation.

_[tool: snip]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: write_file]_

Now the index.html and CSS for the UI kit:

_[tool: snip]_

_[tool: write_file]_

_[tool: write_file]_

_[tool: register_assets]_

_[tool: write_file]_

_[tool: update_todos]_

_[tool: snip]_

_[tool: done]_

_[tool: snip]_

_[tool: fork_verifier_agent]_

Design system complete. All four tabs of the EIT Workstation, full token set, preview cards, and SKILL.md are in place.

**Caveats**
- **Fonts**: No licensed Segoe UI / Microsoft YaHei / Times New Roman files — substituted Noto Sans SC + Noto Serif SC from Google Fonts. Drop real font files into `fonts/` and update `colors_and_type.css` when you can.
- **Logo**: The source app has no logo (window title *is* the brand). I synthesized a simple wordmark slug at `assets/logo.svg` — feel free to replace.
- **Plots**: The UI-kit plot views are stylized SVG stand-ins for pyqtgraph/matplotlib output. They show the palette and layout correctly but are not real EIT reconstructions.

**Please help me iterate** — I'd love your feedback on:
1. Does the steel-blue accent `#1f5d8b` match the real app? I lifted it from `theme.py` but haven't seen a screenshot.
2. Are the four tab names (实测 / 仿真 / 数据集 / 数据库) and step names (连接 / 会话 / 采集, etc.) the actual strings you use? If not, tell me what to swap in.
3. If you have a real logo, any screenshots of the running app, or a preferred webfont for the UI chrome, drop them in and I'll rebuild the affected cards.

## User

Regenerate "README": 有一些字体乱码了

## Assistant

_[tool: snip]_

_[tool: grep]_

The README file's Chinese characters render fine in my tools (the grep shows them correctly). The "乱码" (garbled text) the user sees is likely because the README is being rendered somewhere without UTF-8 declaration or with a font that lacks CJK glyphs. Let me check how it's being shown in the preview pane — the asset is registered as path "README.md" which renders as raw text. I'll create a proper HTML preview wrapper for the README so it displays correctly with a CJK-capable font.

_[tool: snip]_

_[tool: read_file]_

The README file itself is fine (proper UTF-8). The "乱码" is because the review pane renders .md as plain text without a CJK-capable font. I'll create an HTML preview wrapper that renders the README content with proper fonts, and register it as the README asset.

_[tool: snip]_

_[tool: write_file]_

_[tool: register_assets]_

_[tool: show_to_user]_

乱码问题已修复 — README 现在作为 HTML 渲染，使用 Noto Sans SC 字体，中文字符可以正常显示。原始 `README.md` 源文件保留不变。

## User

Regenerate "README": https://github.com/CBZ199671/PyEIDORS，这是我第一代版本的readme，请你参考

## Assistant

_[tool: snip]_

_[tool: web_fetch]_

_[tool: list_files]_

_[tool: web_fetch]_

_[tool: web_fetch]_

_[tool: web_search]_

I can't access the GitHub repo directly. Let me ask the user to share the README content — I'll check if maybe it was attached as a file or if GitHub connector is available.

## Assistant

## User

Regenerate "Components / Cards": 背景颜色不一样，需要统一

## Assistant

## User

Regenerate "UI Kit / EIT Workstation": 整体上不错，但是功能不要丢失，比如数据集生成中，我的原版本中有很多可以设置的地方，包括实测数据tab页最后实时采集、实时成像的界面上成像框图有4个，并且还能根据历史采集会话，选择参考等。总之就是不要流失功能，但是你现在的逻辑规划等很好，请你继续沿用！

## Assistant

## User

Regenerate "Components / Table": 实时采集的每一个session中间的文件名要包含每一次的频率

## Assistant

## User

Regenerate "Brand / Logo": 商标能不能做的再优美一些？立体3D一些，还可以复杂一点！

## Assistant

## Assistant

## Assistant

## Assistant

_[tool: snip]_

_[tool: read_file]_

Let me craft a 3D, dimensional logo — using gradients, radial layering, isometric perspective on the electrode ring to suggest a real EIT sensor, plus depth on the wordmark.

_[tool: write_file]_

_[tool: write_file]_

_[tool: register_assets]_

_[tool: done]_

新商标更有立体感了 — 做了这些改动：

- **传感器环**改为等轴透视（椭圆化），用径向渐变画出金属质感的厚度和内腔
- **电极**从 8 颗扁平圆点升级到 16 颗立体铬珠（EIT 标准电极数），每颗带高光和接触阴影
- **内腔**展示了真实的 viridis σ 场 + 等势线 + 网格 + 偏心的琥珀色非均匀块
- **激励弧**（橙色）连接一对电极，暗示当前注入通道
- **文字标**做了浮雕处理：白色高光 + 深蓝阴影描边，serif "EIT" 与 sans "Workstation" 之间加了细分割线
- 整体加了**接地投影**让标识"立"在页面上

预览卡片也同时展示了 light/dark 底色上的表现，以及 180 / 120 / 64 px 三种尺寸的收缩行为 — 需要再调的话告诉我（比如换配色、去掉 σ 场、改用横版等）。

## User

Regenerate "Components / Table": 实时采集的每一个session中间的文件名要包含每一次的频率；Regenerate "UI Kit / EIT Workstation": 整体上不错，但是功能不要丢失，比如数据集生成中，我的原版本中有很多可以设置的地方，包括实测数据tab页最后实时采集、实时成像的界面上成像框图有4个，并且还能根据历史采集会话，选择参考等。总之就是不要流失功能，但是你现在的逻辑规划等很好，请你继续沿用！；Regenerate "Components / Cards": 背景颜色不一样，需要统一
Components / Cards
UI Kit / EIT Workstation

## Assistant

收到了3个反馈，我来依次处理。先探索原版本了解功能细节。

_[tool: update_todos]_

_[tool: local_ls]_

_[tool: local_grep]_

[The user needs to re-grant access — waiting for them to click Allow in the permission banner.]

## User

Access re-granted for "eit_app" — continue.

## Assistant

Now let me explore the real app to understand all the functionality I need to preserve.

_[tool: local_ls]_

_[tool: local_grep]_

_[tool: local_grep]_

_[tool: snip]_

_[tool: local_read]_

_[tool: local_ls]_

Great — filename pattern is `{session}_frame_{NNNN}_{HZ}Hz.csv/.yaml`. Now let me look at the reconstruction widget (the 4 images) and dataset-gen dialog, plus examine my current kit.

_[tool: local_read]_

_[tool: local_ls]_

_[tool: list_files]_

Let me look at the dataset generator (simulation) and the current UI kit files.

_[tool: local_ls]_

_[tool: list_files]_

_[tool: list_files]_

Let me see the current kit structure and key previews.

_[tool: list_files]_

_[tool: read_file]_

_[tool: read_file]_

_[tool: snip]_

_[tool: read_file]_

_[tool: list_files]_

Let me see the current state of HardwareTab and DatasetTab to understand what functionality was lost.

_[tool: read_file]_

_[tool: read_file]_

_[tool: snip]_

_[tool: read_file]_

_[tool: read_file]_

_[tool: snip]_

_[tool: read_file]_

Let me peek at the dataset generator panel and reconstruction widget title (4-image context) to understand settings.

_[tool: local_grep]_

_[tool: local_grep]_

local_grep needs a directory path. Let me search differently.

_[tool: local_grep]_

_[tool: local_read]_

_[tool: snip]_

_[tool: local_read]_

_[tool: local_read]_

I have enough now. Key findings:

- **Hardware tab layout**: left = 3-step toolbox (Connection/Control/Acquisition), center = **2×2 grid** (LivePlot | BoundaryVoltage on top, Reconstruction | Equipotential on bottom) — these are the "4 imaging frames", right = **FrameBrowser** (select reference from history).
- **Filename pattern**: `{session}_frame_{NNNN}_{HZ}Hz.csv`
- **Dataset generator has TWO panels**: RandomizationPanel (shapes, count min/max, pos range, size range, conductivity range, bg conductivity, noise) + RunPanel (samples, output dir, progress).

_[tool: snip]_

Now let me fix the 3 items. Starting with **Cards** (unified background) and **Table** (filename with freq):

_[tool: write_file]_

_[tool: write_file]_

Got it — dropping the Chinese subtitle.

_[tool: str_replace_edit]_

_[tool: show_to_user]_

_[tool: snip]_

_[tool: read_file]_

_[tool: read_file]_

_[tool: str_replace_edit]_

_[tool: snip]_

Now I need to also tighten the preview viewBox since the Chinese row is gone, and continue with the UI Kit rebuild. Let me also trim the preview viewport height.

_[tool: read_file]_

_[tool: str_replace_edit]_

Now let me tackle the big piece — rebuilding the UI Kit to restore the 4-image live imaging grid, reference-selection frame browser, and full dataset-generator config. I'll split into focused files.

_[tool: snip]_

Now I'll rewrite HardwareTab to add the 2×2 imaging grid + frame browser with reference selection, and DatasetTab to expose the full RandomizationPanel + RunPanel fields. First, I need a few more CSS tweaks.

_[tool: str_replace_edit]_

Now I'll rewrite HardwareTab with the 2×2 imaging grid and frame browser with reference selection:

_[tool: snip]_

_[tool: write_file]_

I need to update CSS for the 3-column hardware layout, rewrite DatasetTab with full config, and verify.

_[tool: snip]_

_[tool: str_replace_edit]_

_[tool: write_file]_

_[tool: done]_

_[tool: snip]_

_[tool: fork_verifier_agent]_

三个反馈都已处理完成：

- **Brand / Logo** — 去掉了中文描述行，SVG viewBox 也收紧到 `360 × 96`
- **Components / Cards** — 两张卡用同一底色（`--panel`），summary 变体只靠边框色和标题色区分，带虚线分隔的行表
- **Components / Table** — 文件名用 `sess..._frame_NNNN_{HZ}Hz.csv` 格式，跟 `recording_controller.py` 一致；还加了彩色频率徽章方便扫频会话识别
- **UI Kit · 实测 Tab** — 恢复了左侧 3-step toolbox（Connection · Control · Acquisition），中央 **2×2 成像网格**（边界电压波形 / 拟合 / σ 重构 / 等势线），右侧 **Frame Browser** 可跨会话选参考帧；Control 步骤有频率/幅度/测量模式，Acquisition 有扫频配置（起止频率、步数、log/linear）
- **UI Kit · 数据集 Tab** — 暴露了完整的随机化参数面板：形状池、N 范围、位置范围、尺寸范围、σ_inh 范围、σ_bg 范围、噪声；以及运行面板的种子、网格、电极数、正问题求解器

## User

Regenerate "README": # PyEIDORS

<p align="center">
  <img src="pictures/Fig.%204.%20fig_absolute_vs_difference.png" alt="PyEIDORS banner" width="900" />
</p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
![License](https://img.shields.io/badge/license-MIT-green)
![Backend](https://img.shields.io/badge/backend-FEniCS-orange)
![Accel](https://img.shields.io/badge/accel-PyTorch-red)

A Python-first EIT framework with a FEniCS Complete Electrode Model (CEM) backend and PyTorch acceleration, designed to be familiar to EIDORS users while enabling modern GPU/differentiable workflows.

SoftwareX manuscript in preparation; citation info will be added after acceptance.

## Affiliation, Laboratory, and Funding

| Item | Details |
|---|---|
| Laboratory | 455 Lab |
| Location | Beijing, China |
| Affiliation | College of Information and Electrical Engineering, China Agricultural University |
| Lab Head | Prof. Lan Huang, Prof. Zhong-Yi Wang, and Dr. Lifeng Fan |

455 Lab focuses on plant electrophysiological phenotyping, crop root phenotyping, and the monitoring of crop water status. Building on Electrical Impedance Tomography (EIT), the lab develops methods for real-time, in situ, and non-destructive monitoring of electrical conductivity distributions in root zones, ears of corn, and related plant tissues. PyEIDORS is developed within this research context to support advanced EIT modeling, reconstruction, and experimental analysis.

**Funding Statement.** This work was supported by the National Natural Science Foundation of China [Grant Number 62271488].

## Why PyEIDORS

- **Numerical Consistency**: Matches EIDORS-style workflows with verified simulation parity.
- **Modern Architecture**: Hybrid FEniCS (FEM) + PyTorch (Inverse/Accel) design.
- **Modular & Extensible**: `EITSystem` coordinator makes it easy to replace geometry, forward models, or solvers.
- **Research Ready**: End-to-end scripts for absolute & difference reconstruction, real-time mesh generation, and benchmarking.

---

## Quick Start

0. **Install Docker (required)**:
   - **Windows**: install WSL2 first, then install Docker Desktop (WSL2 backend).
   - **macOS/Linux**: install Docker Desktop (or Docker Engine on Linux).
   - After installation, open a terminal and verify:
     ```bash
     docker --version
     ```

1. **Download this repository and enter the folder**:
   ```bash
   git clone https://github.com/CBZ199671/PyEIDORS.git
   cd PyEIDORS
   ```
   You can move the `PyEIDORS` folder anywhere you like; just make sure you run commands inside it.

2. **Start the Docker Environment** (this mounts the current folder into the container):
   ```bash
   docker run -ti -v "$(pwd):/root/shared" -w /root/shared --name pyeidors ghcr.io/cbz199671/pyeidors-env:latest
   ```

   If you have an NVIDIA GPU with CUDA set up, add `--gpus all`:
   ```bash
   docker run -ti --gpus all -v "$(pwd):/root/shared" -w /root/shared --name pyeidors ghcr.io/cbz199671/pyeidors-env:latest
   ```

   If you use a different local path, replace `$(pwd)` with your full folder path. The `-v` flag binds your local repo into `/root/shared` inside the container.

3. **Install the Package**:
   ```bash
   pip install -e .
   ```

4. **Run a Synthetic Demo** (Paper parity example):
   ```bash
   python scripts/run_synthetic_parity.py --output-root results/simulation_parity/run03 --mode both --difference-solver single-step --gn-regularization 1e-11
   ```

5. **Run a Real-Data Demo** (Tank difference imaging):
   ```bash
   python scripts/run_single_step_diff_realdata.py --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv --background-sigma 0.008 --lambda 0.9
   ```

The repository includes pre-generated demo outputs under `results/` (see the `COMMAND.md` files inside each demo directory).

---

## Gallery & Validation

### Modern Architecture: FEniCS + PyTorch

<p align="center">
  <img src="pictures/Fig.%201.%20pyeidors_architecture.png" alt="Architecture: FEniCS + PyTorch" width="900" />
</p>

### Gauss-Newton Absolute Reconstruction

<p align="center">
  <img src="pictures/reconstruction_iterations.gif" alt="Gauss-Newton absolute reconstruction iterations" width="600" />
</p>

*Absolute-mode voltage RMSE reaches **8.23×10⁻⁸ V** in this synthetic parity example, supporting numerical consistency of the underlying FEM implementation and EIDORS-style reconstruction workflow rather than serving as an analytical-solution validation.*

Synthetic parity metrics for absolute and difference reconstruction modes:

| Metric | Absolute mode | Difference mode |
|---|---:|---:|
| RMSE (V) | 8.23×10⁻⁸ | 7.68×10⁻⁵ |
| MAE (V) | 5.59×10⁻⁸ | 6.43×10⁻⁵ |
| Max. absolute error (V) | 2.60×10⁻⁷ | 1.47×10⁻⁴ |
| Pearson correlation | >0.9999 | 0.991 |
| Measurements | 208 | 208 |

### Simulation Parity with EIDORS

No closed-form analytical reference solution is used in this example; the reported voltage agreement is intended only as a synthetic numerical-consistency / cross-platform parity check.

<p align="center">
  <img src="pictures/Fig.%203.%20Simulation%20parity_combined.png" alt="Simulation parity with EIDORS" width="900" />
</p>

Comparison of conductivity reconstructions and voltage predictions between PyEIDORS and MATLAB/EIDORS. (a) Ground-truth. (b) PyEIDORS single-step difference. (c) MATLAB/EIDORS raw. (d) EIDORS (aligned). (e) Differential voltage traces.

*Note: The larger residuals observed in EIDORS' voltage predictions compared to PyEIDORS do not imply superior reconstruction performance by PyEIDORS. This discrepancy arises primarily from numerical implementation differences in forward modeling. Since the synthetic measurement data in this experiment was generated using PyEIDORS' forward model, the PyEIDORS inverse solver benefits from inherent modeling consistency. Conversely, EIDORS incurs inevitable modeling bias when reconstructing from this data due to subtle differences in mesh discretization, finite element interpolation orders, and Complete Electrode Model (CEM) boundary handling.*

### Experimental Validation: Tank Data

<p align="center">
  <img src="pictures/Fig.%205.%20compare_tank.png" alt="Tank data validation" width="900" />
</p>

Verification using tank measurement data. PyEIDORS (b, c) demonstrates consistent performance with MATLAB/EIDORS benchmarks (d, e). Panels (f, g) show the original EIDORS reconstructions before affine alignment.

Parameter settings for the forward modeling and inverse solution corresponding to the tank comparison panels:

| Category | Parameter | b | c | d | e |
|---|---|---:|---:|---:|---:|
| Fwd. | Background conductivity (S/m) | 0.008 | 0.008 | 0.008 | 0.008 |
| Fwd. | Stimulation amplitude | 1.0 | 5e-5 | 1.0 | 5e-5 |
| Fwd. | Measurement gain | 10 | 10 | 10 | 10 |
| Fwd. | Contact impedance | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| Fwd. | Mesh radius | 0.025 | 0.025 | 0.025 | 0.025 |
| Fwd. | Number of electrodes | 16 | 16 | 16 | 16 |
| Fwd. | Number of nodes & elements | 1976 & 3806 | 1976 & 3806 | 1122 & 2130 | 1122 & 2130 |
| Inv. | Regularization lambda | 1.5 | 0.9 | 1.5 | 0.9 |
| Inv. | Regularization type | NOSER | NOSER | NOSER | NOSER |

### Experimental Validation: Bio-impedance (Corn Stalk)

<p align="center">
  <img src="pictures/Fig.%206.%20corn_stem_reconstruction.png" alt="Corn stalk reconstruction" width="900" />
</p>

Reconstruction of a corn stalk sample, capturing electrical heterogeneity in biological tissue.

---

## System Architecture

PyEIDORS is designed as a modular pipeline:

```
Mesh Loading/Generation ──► Forward Model (Complete Electrode Model)
                                  │
                                  ▼
                      Jacobian Computation & Regularization
                                  │
                                  ▼
                      Modular Gauss-Newton Reconstruction
                                  │
                                  ▼
            Visualization · Synthetic Data · Result Analysis
```

This architecture is intentionally modular so you can swap meshes, forward models, Jacobian calculators, or priors without rewriting the full pipeline.

Highlights:

- Designed for research and engineering practice, covering the complete pipeline of mesh generation, forward modeling, Jacobian computation, regularization, and Gauss-Newton reconstruction.
- Modular design with `EITSystem` as the core coordinator for geometry, forward, and inverse problem components.
- Supports GMsh+meshio+FEniCS mesh workflow, with built-in stimulation/measurement pattern manager, synthetic data generation, and visualization tools.
- Provides examples, tests, and reports to help verify electrode layouts, mesh quality, and end-to-end reconstruction pipelines.

### Key Components

- **Geometry**: GMsh-based mesh generation (`mesh_generator.py`) and cached loading (`mesh_loader.py`).
- **Forward Model**: Complete Electrode Model (CEM) implemented in FEniCS (`eit_forward_model.py`).
- **Inverse Solver**:
    - Adaptive Gauss-Newton (PyTorch-accelerated).
    - EIDORS-style single-step difference imaging.
    - Sparse Bayesian learning workflows.
- **Visualization**: `EITVisualizer` for meshes, conductivity maps, and measurement error plots.

For a file-level map of the codebase, see `FILE_ORGANIZATION.md`.

---

## Performance Benchmarks

End-to-end **single-step difference reconstruction** timing (Warm Start).
PyEIDORS' **measurement-space solve** is derived via Woodbury to avoid forming the large n&times;n system (n = mesh elements). The standard form solves
(J<sup>T</sup>J + &lambda;R) &delta; = J<sup>T</sup>d (parameter space), while the measurement-space form solves
(J R<sup>-1</sup> J<sup>T</sup> + &lambda;I) y = d and then &delta; = R<sup>-1</sup> J<sup>T</sup> y.
Here m = number of measurements, so the inner solve is m&times;m (typically 208), which is much smaller than n&times;n for dense meshes. With NOSER, R is diagonal, so R<sup>-1</sup> is just elementwise inversion, keeping the solution algebraically equivalent but far cheaper in memory and runtime.

<p align="center">
  <img src="pictures/benchmark_difference_runtime.png" alt="PyEIDORS baseline difference benchmark" width="900" />
</p>
<p align="center">
  <img src="pictures/benchmark_difference_runtime_measurement_6_24.png" alt="PyEIDORS measurement-space difference benchmark" width="900" />
</p>

For reference, we include the EIDORS timing curve (cold vs cached). EIDORS caching is extremely strong; the cold (no cache) curve is the fairer comparison point. PyEIDORS caching is currently limited and does not yet match EIDORS' cache behavior.

<p align="center">
  <img src="pictures/benchmark_difference_runtime_eidors.png" alt="EIDORS difference benchmark" width="900" />
</p>

Bench scripts:
- PyEIDORS: `python scripts/benchmarks/benchmark_difference_runtime.py`
- EIDORS: `compare_with_Eidors/benchmark_jacobian_runtime.m` (set `benchmark_mode = 'difference'`)

Accuracy check (parameter-space vs measurement-space, refinement=12):
- `delta_rel=8.66e-09`, `rmse_param=4.503e-01`, `rmse_meas=4.503e-01`, `pred_rel=5.10e-10`
- Reproduce with: `python scripts/benchmarks/benchmark_difference_runtime.py --refinements 12 --compare-solvers --single-step-space measurement`

| Elements | Baseline (s) | Measurement-Space (s) | Speedup |
|---:|---:|---:|---:|
| 2,650 | 1.873 | 1.336 | **1.40×** |
| 5,702 | 5.848 | 2.356 | **2.48×** |
| 10,584 | 20.722 | 4.003 | **5.18×** |
| 14,650 | 46.465 | 5.313 | **8.74×** |
| 18,474 | 85.128 | 7.082 | **12.02×** |

---

## Advanced Usage

### 1. Synthetic Simulation Comparison
Automatically generate simulation data and compute error statistics vs. EIDORS:

```bash
python scripts/run_synthetic_parity.py \
  --output-root results/simulation_parity/run01 \
  --mode both --save-forward-csv \
  --difference-solver single-step \
  --eidors-csv path/to/eidors_voltages.csv
```

The script will:

- Run forward simulation with given circular phantom parameters, saving baseline/anomaly boundary voltages.
- Execute absolute and difference imaging reconstruction sequentially, outputting residuals, RMSE, correlation coefficients, and other metrics (`metrics.json`).
- Optionally load EIDORS-generated voltage vectors via `--eidors-csv` to automatically compute differences with PyEIDORS.

### 2. Real Measurement Data
After data normalization (see `docs/MEASUREMENT_DATA_SPEC.md`), run reconstruction:

```bash
python scripts/run_real_measurement_reconstruction.py \
  --csv data/measurements/sample.csv \
  --metadata data/measurements/sample.yaml \
  --use-cols 0 2
```

The script validates the measurement matrix, builds `EITSystem`, and performs difference inverse problem reconstruction. Output measurement curves and conductivity images are saved in `results/real_measurements/`.

### 3. Sparse Bayesian Learning
Run the advanced sparse Bayesian solver (supports GPU):

```bash
python scripts/run_sparse_bayesian_reconstruction.py \
  --csv data/measurements/sample.csv \
  --mode both --solver fista --use-gpu
```

The repository also includes a pre-generated tank sparse Bayesian demo under:
`results/tank_final_results/sparse_bayesian_physical_bg0008_v1_0/` (see `COMMAND.md` inside).

Results are written to `results/sparse_bayesian/` by default. For a full list of options, run `python scripts/run_sparse_bayesian_reconstruction.py --help`.

## Data, Visualization, and Testing

- Synthetic data: `create_synthetic_data` supports setting noise level, anomaly position and conductivity, returning clean/noisy data with SNR metrics.
- Real measurement data: `MeasurementDataset` helper class builds `EITData` from normalized measurement matrices and metadata, see `docs/MEASUREMENT_DATA_SPEC.md`.
- Visualization: `EITVisualizer` includes built-in plotting for mesh, conductivity, measurements, reconstruction comparison, and convergence curves, with PNG report output.
- Testing: `tests/unit/test_complete_eit_system.py` provides end-to-end pipeline validation, `tests/unit/test_optimized_mesh_generator.py` covers geometry and electrode layout.
- Examples: `examples/basic_usage.py` demonstrates module structure, environment checking, and system initialization steps.

---

## Documentation

- **File Structure**: `FILE_ORGANIZATION.md`
- **Data Specs**: `docs/MEASUREMENT_DATA_SPEC.md`
- **Electrode Setup**: `docs/ELECTRODE_Y_AXIS_POSITIONING.md`
- **Docker Notes**: `docs/DOCKER.md`

## Acknowledgments

PyEIDORS is developed in the research environment of 455 Lab, College of Information and Electrical Engineering, China Agricultural University. This work was supported by the National Natural Science Foundation of China [Grant Number 62271488].

## Docker Environment Setup

The recommended workflow is to use the prebuilt Docker image. It already contains FEniCS, CUQIpy, and PyTorch. See `docs/DOCKER.md` for the most up-to-date commands.

### Option A: Prebuilt Image (Recommended)

```bash
docker pull ghcr.io/cbz199671/pyeidors-env:latest

docker run -ti \
  -v "$(pwd):/root/shared" \
  -w /root/shared \
  --name pyeidors \
  ghcr.io/cbz199671/pyeidors-env:latest
```
*Note: Add `--gpus all` for GPU support.*

Notes:

- The `-v ...:/root/shared` path should point to your local clone of this repository.
- On Windows, install WSL2 + Docker Desktop (WSL2 backend) first; see `docs/DOCKER.md`.
- For GPU support, add `--gpus all` (requires NVIDIA Container Toolkit).
- Do not set `--cpus` / `--memory` unless you want to limit Docker. By default, Docker can use the available host resources.
- If you hit shared-memory issues (e.g., `/dev/shm` errors), add `--shm-size=2g` (or larger). On Linux, `--ipc=host` can also help.
- If you omit `--gpus all`, PyTorch will run on CPU; it will not error unless a script explicitly forces CUDA.
- If you downloaded an offline image archive (e.g., `pyeidors-ultra22.tar.zst`), see `docs/DOCKER.md` for `docker load` instructions.

### Option B: Manual Build

```bash
# Start container
docker run -ti \
  -v "$(pwd):/root/shared" \
  -w /root/shared \
  --name pyeidors \
  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

# (Optional) Install CJK font dependencies
apt-get update && apt-get install -y fonts-wqy-zenhei

# Install CUQIpy and CUQIpy-FEniCS
pip install cuqipy cuqipy-fenics

# Create virtual environment
python3 -m venv /opt/final_venv --system-site-packages
source /opt/final_venv/bin/activate

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Environment Components
- **FEniCS**: 2024-05-30 (Official Image)
- **CUQIpy**: 1.3.0
- **CUQIpy-FEniCS**: 0.8.0
- **PyTorch**: 2.7.1+cu128
- **Python**: 3.10+

do like this

## Assistant

_[tool: snip]_

This is a sister project's README — the user wants me to replace this project's README with it, styled using our EIT Workstation Design System. Let me check what we have and explore.

_[tool: read_file]_

_[tool: read_file]_

Now let me check if there's an existing README.html file and see the structure:

_[tool: list_files]_

_[tool: read_file]_

I understand the user wants the README redone in the style/format of the sample PyEIDORS README they pasted — i.e. a proper project-README look (centered figure, badges, tables, code blocks, sections) but for *our* EIT Workstation Design System.

Let me rebuild README.html in that style using our design tokens.

_[tool: read_file]_

I'll rebuild README.html in the style of the PyEIDORS README sample — centered banner image, badges, section structure, tables, code blocks, etc. — but describing *our* EIT Workstation Design System.

_[tool: snip]_

_[tool: list_files]_

_[tool: list_files]_

_[tool: snip]_

Now I'll rebuild README.html in the PyEIDORS style — banner, badges, structured sections, tables with right-aligned numbers, code blocks, ASCII diagrams, centered figures. All describing our EIT Workstation Design System.

_[tool: write_file]_

_[tool: snip]_

_[tool: done]_

Rebuilt README.html in the PyEIDORS project-README format — centered SVG banner showing the four tabs + a mini reconstruction plot + a ready-state banner, a row of metadata badges (platform · languages · theme · compute · typography · icons), an Affiliation/Funding table, Quick Start, Products table, Content Fundamentals, Visual Foundations (palette with inline swatches, type, spacing, interaction states), ASCII architecture diagrams, Iconography, File Index, and a font-substitution note — all styled with our design tokens from `colors_and_type.css`.

## User

Regenerate "UI Kit / EIT Workstation": 我觉得你的这个设计有些地方比我的原本设计要好，功能逻辑更好，但是有些地方又功能缺失，比我原版的要差，而且有些界面超出了范围太多，比如数据集的左侧，不如就用我现在的分阶段设置。数据库中的成像单帧和批处理怎么选都没有，但是你添加的类型和来源包括标签等构想非常好。实测界面有两个设为参考有些重复。

## Assistant

_[tool: snip]_

_[tool: list_files]_

_[tool: list_files]_

I'll look at the current flagged files to address the feedback: (1) dataset tab left panel should use the user's original staged approach, not my expanded one; (2) database tab needs a single-frame vs batch reconstruction selector; (3) hardware tab has duplicate "设为参考" buttons.

_[tool: read_file]_

_[tool: read_file]_

_[tool: read_file]_

_[tool: snip]_

Now let me check the other files I need to understand context:

_[tool: read_file]_

Good — SimulationTab is fine; the user said the staged approach there is good. Now let me fix the three issues:

1. **HardwareTab**: Remove duplicate "设为参考" — there's one in the center toolbar AND one in FrameBrowser. Keep only the FrameBrowser one.
2. **DatasetTab**: Replace the huge left rail with a staged (StepList) approach like SimulationTab.
3. **DatabaseTab**: Add single-frame vs batch reconstruction selector.

_[tool: str_replace_edit]_

Now rebuild DatasetTab with a staged left rail, and DatabaseTab with single-frame vs batch reconstruction selector.

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: done]_

三处都改好了：

- **实测 Tab** — 移除了中央工具条上的「设为参考」按钮，只保留右侧 Frame Browser 里的那个
- **数据集 Tab** — 左侧改成分阶段 StepList（形状 → 随机范围 → 运行），和仿真 Tab 一致，不再超出边界
- **数据库 Tab** — 详情区新增反演模式选择器：**单帧 Single frame**（帧号 + 参考 + 算法 + λ）vs **批处理 Batch**（起止帧 + 步长 + 参考策略 + 输出格式），按钮文案随模式切换；原本的类型/来源/标签结构保留

## User

Regenerate "Brand / Logo": 显示有些不完整，请你重新生成，面向全球用户，因此只需要有英文即可！可以借鉴一下FENICSX的图标的设计灵感！

## Assistant

_[tool: snip]_

_[tool: read_file]_

_[tool: read_file]_

_[tool: snip]_

I'll redesign the logo inspired by FEniCSx's iconography — they use a clean geometric mark (the stylized "X" with a mesh/wave motif) alongside bold lowercase wordmark. For EIT Workstation, I'll create a similar pairing: a geometric electrode-ring mark + English-only wordmark.

FEniCSx style references:
- Clean flat geometric mark (no 3D gradients, no photorealism)
- Distinct monogram symbol (their X has a concave triangle/mesh cut)
- Paired with bold, slightly condensed lowercase wordmark
- Limited palette (dark blue + single accent)
- Descriptor line underneath in small caps or regular weight

_[tool: write_file]_

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: show_to_user]_

_[tool: snip]_

Logo redesigned — FEniCSx-inspired: flat 2-color electrode ring mark (16 nodes with crossing injection chords, amber highlight on active pair, small σ kernel at center) paired with a bold English-only "EIT | Workstation" wordmark and a tracked descriptor line "ELECTRICAL IMPEDANCE TOMOGRAPHY".

## User

Regenerate "Brand / Logo": icon图标不太好，有很多没有意义的线条，而且EIT全称的英文超出了，太长了！

## Assistant

_[tool: snip]_

_[tool: write_file]_

_[tool: snip]_

_[tool: show_to_user]_

_[tool: snip]_

Logo cleaned up — removed the noisy chord lines, simplified to 8 electrodes with a single clean injection arc between the amber-highlighted pair, and shortened the descriptor to "IMPEDANCE TOMOGRAPHY" so it no longer overflows.

## User

Regenerate "Brand / Logo": 不如第一版，给我重新设计一下！

## Assistant

<response>

_[tool: snip]_

Let me design a completely fresh take — cleaner, more distinctive, with more visual interest than the simplified version but tighter than the busy first attempt. I'll draw inspiration from FEniCSx's iconic monogram style: a solid, confident geometric mark with strong negative space that reads as a single symbol, not a diagram.

_[tool: write_file]_

<response>

_[tool: show_to_user]_

Fresh take — the circular mark now has weight and meaning: a solid tomographic cross-section showing two conductivity bands (a prominent dark upper band and a softer lower one, with a curved boundary between them — the kind of shape you actually see in a σ reconstruction), ringed by 8 electrodes with the active pair highlighted in amber. Reads as a single confident symbol rather than a thin diagram, while still being unmistakably "EIT".

