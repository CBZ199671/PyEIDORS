# Resolved bug history

Append-only resolved-bug registry. New bug intake starts in root `SPEC.md` and moves here after resolution.

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | 2026-04-20 | `ForwardKSPSession` applied `setReusePreconditioner(True)` uniformly; for `ksp_type==preonly` + `pc_type∈{lu,cholesky,qr}` this silently reuses stale LU/Cholesky/QR factorisation across sigma updates, solving `A(σ_new) x = b` with `A(σ_old)^{-1}`. No Krylov iteration to correct the error (unlike iterative+AMG where reuse is a staleness penalty). | V24,T14 |
| B2 | 2026-04-21 | CUDA/dense `matSolve` auto override ignored explicit `mat_solve_mode="off"`; CUDA profile had `petsc_amgx=false` / PCAMGX absent, but `cuda_amgx` proceeded past setup toward solve | V42 |
| B3 | 2026-04-21 | `/check §V` found V2 drift: CUDA auto `matSolve` branch (`effective_device=="cuda"` & `petsc_cuda_dense`) bypasses `performance_mode=="aggressive"` and `forward_mat_solve_min_patterns`; V2 says exact iff formula | V2 |
| B4 | 2026-04-21 | `spd_hypre + petsc_device=cuda` forward benchmark SIGSEGV in PETSc/Hypre CUDA, even with `--forward-mat-solve off`; current measured safe CUDA forward route = `spd_gamg + petsc_device=cuda` | V44 |
| B5 | 2026-04-21 | AmgX cannot enable in current CUDA shell because PETSc has CUDA Mat/Vec/Dense (`aijcusparse`, `cuda`, `densecuda`) but PCAMGX is not compiled/registered: `PETSc.PC.Type.AMGX` is absent and `PCSetType("amgx")` returns error code 86 `Unable to find requested PC type amgx`; `flake.nix` CUDA PETSc override only adds CUDA/cuBLAS/cuSPARSE/cuSOLVER flags, with no AmgX package or `--with-amgx` / `--download-amgx` configure path; FEniCSx/DOLFINx latest release does not ship AmgX by itself | T33 |
| B6 | 2026-04-21 | GUI 3D CUDA policy downgraded unavailable AmgX/Hypre CUDA to `spd_gamg` but left `forward_mat_solve=auto`; PETSc `KSPMatSolve` on current `spd_gamg + cuda` 48-electrode/5936-measurement config failed after a long solve attempt with negative convergence reason `-10`. Stable measured route is `spd_gamg + petsc_device=cuda + forward_mat_solve=off` | V45 |
| B7 | 2026-04-22 | WSLg GUI bootstrap pinned Qt to XCB to protect embedded VTK, so Windows HiDPI scaled the whole app through XWayland blur; launcher also repeated full uv/import sync and PETSc CUDA probe on every GUI start, then imported heavy pyeidors/PETSc/Torch/CUQI paths and synchronously queried Windows COM ports before first paint | V51 |
| B8 | 2026-04-22 | User PATH shadows coreutils `env`: `/home/tom/.local/bin/env` intercepted `env EIT_APP_AUTO_QUIT_MS=5000 bash scripts/gui/run_eit_app.sh --gpu`, returned success without running payload, and produced bogus `real 0.00` GUI timing. `/usr/bin/env ...` executed correctly | V52,T37 |
| B9 | 2026-04-22 | GUI 3D `single_step_cached` accepted unconstrained `sigma_bg + alpha * delta_sigma`; calibration candidate failure swallowed then `alpha=1.0`; `cuda_structured` correctly rejected nonphysical FEM top-left diagonal (`NaN/Inf` or `<=0`) during forward validation | V53,T38 |
| B10 | 2026-04-22 | GUI 3D PyVista offscreen drag path rendered at 60% physical size for responsiveness but set the pixmap DPR to the widget DPR; Qt therefore displayed drag frames as smaller logical images, then snapped back on the idle full-resolution frame | V54,T39 |
| B11 | 2026-04-22 | SPEC/bench text overclaimed 3D GREIT as official-aligned; code is linearized GREIT-RM v0 (`Y=T@J.T`, `D≈T`) and lacks EIDORS finite-target `vh/vi`, `desired_solution_fn`, NF weight search, HDF5 model-component parity artifacts | V50,V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65,T40,T41,T42,T43,T44,T45,T46,T47,T48,T49,T50 |
| B12 | 2026-04-22 | Project persistence still mixed: mesh cache already prefers XDMF/HDF5 but freshness tied to source `.msh`; many production writers still emit `.npz/.npy` (`greit_rm.npz`, `one_step_*_rm.npz`, `outputs.npz`, `result_arrays.npz`, dataset `mesh_info.npz`/`sample_*.npz`, GUI "NumPy archive"). This conflicts with FEniCSx-aligned HDF5-unified cache/save target | V65,V66,V67,V68,T51,T52,T53,T54,T55,T56,T57 |
| B13 | 2026-04-22 | After fixing 3D offscreen drag size jitter, the interaction path still defaulted to ~30 fps and 0.6× drag framebuffer scale; on high-resource GPU/WSLg machines that made rotation feel visibly choppy and low-fidelity | V69,T58 |
| B14 | 2026-04-26 | Full `tests/unit -q --no-cov` exceeded the 10min local gate: GN validation and reconstruction CLI validation spawned one Python process per case; GN diff cache tests used oversized/cold mesh fixtures; GUI theme tests repolished the whole QApplication for local palette/listener assertions | V72 |
| B15 | 2026-04-27 | `DirectJacobianCalculator` (历史 `direct_jacobian.py:367` sign=+1.0, `:481` 无 minus) 与 `AdjointJacobianCalculator` (历史 `adjoint_jacobian.py:104` sign=-1.0, `:189` 显式 `-np.sum`) 返回 J 整体反号. `linearized.py:127,159` matvec/rmatvec 应用 `self.sign` → 反号传递. GN runtime `gauss_newton_runtime.py:952` `rhs=-jtr` 仅与 Direct 自洽；切到 Adjoint (`set_jacobian_calculator`) 后 δσ 静默反号. Production 默认 Direct 故 V21/V38 通过；无 cross-method parity 测试覆盖此切换. 修：T73 加 V73 sign-parity contract test, T75 Path C 抽 `_core`, Adapter sign 由 `_assemble_numpy` 后置 `-jacobian` 收敛, sign 不再分散在两 calculator 内（旧文件行号已重排, 详 commits `cfa2976` / `0f849cf` / `bb27df0` / `3f8d6d6` / `2945bcf`） | V73,T73,T75 |
| B16 | 2026-04-27 | `prior/tv_irls.py:358` `tv_irls_objective` 中 `_effective_beta(beta, beta)` 第二参写错（应 `beta_floor`）. 函数签名 (`:440`) 取 `max(beta, beta_floor)`；其他调用 (`:51`, `:119`) 都正确传 `beta_floor`. β<beta_floor 时 objective 评估丢失下限保护，与外层 IRLS 路径用的 `RtR` 不一致 → objective 与梯度分歧、单调下降假设破坏 | V74,T74 |
| B17 | 2026-04-29 | §I 声明 `pyeidors.inverse.jacobian.compute_sigma_fingerprint`, 但 `jacobian/__init__.py` 未 re-export / `__all__` 未列出；T93 import-surface gate 首次覆盖即失败 | V77,T93 |
| B18 | 2026-04-29 | T4 benchmark compared `auto` vs `never` with per-regime RNG seed offset, so reported setup saving mixed PC refresh policy with different σ trajectories; artifact also lacked command argv / mesh provenance needed to audit `3D 16e ref5 hypre` claim | V80,T4 |
| B19 | 2026-04-29 | T6 persistent Jacobian key omitted calculator identity/sign convention, so Direct-filled cache could be reused after `set_jacobian_calculator(EidorsJacobianAdapter(...))`; `_last_persistent_jacobian_lookup` also persisted across runs, so dense cache key could leak into later operator/`linearized` diagnostics | V81,T6 |
| B20 | 2026-04-29 | GUI 2D `eidors_one_step_noser` could reuse stale single-step context / disk semantic Jacobian after Jacobian sign/projection implementation drift because cache signatures lacked calculator/sign/projection/operator semantic axes; error metrics also compared forward truth and inverse reconstruction by raw array index, so different mesh orders could report negative correlation and make a centered anomaly look wrong | V82 |
| B21 | 2026-04-29 | GUI `--gpu` 2D kept `petsc_device=auto`; CUDA shell routed cached 2D inverse forward model to PETSc CUDA, producing context `base_meas` norm 1.047 vs homogeneous reference 0.116, so α calibration chose `6.96e-06` and NOSER stayed near background/wrong location | V83 |
| B22 | 2026-04-29 | GUI simulation inverse UI/route drifted behind SPEC v1: method `eidors_one_step_noser` still maps to fine-mesh dense `single_step_cached` cold context (`2094` unknowns vs `208` meas in user run), not dual-model/coarse RM hot path; method list omits NOSER/Laplace/GREIT RM routes; α widget hidden by canonical `difference_lambda=1e-2`; result can be spatially fragmented/discrete despite formula parity | V84,T99,T100,T101,T102,T103,T104 |
| B23 | 2026-04-30 | Forward PETSc KSP setup injected `pyeidors_forward_*_pc_hypre_type=boomeramg` into global options DB, including failed/fallback KSP attempts and non-HYPRE PCs, then never removed it; full unit exit printed PETSc `unused database options` warning | V85 |
| B24 | 2026-04-30 | GUI 3D mesh panel let `electrode_area_m2_override` imply `electrode_height_ratio=0.2` even when dense ring counts (`n_rings>=5`) have level spacing <0.2; forward mesh build reached `Cylinder3DMeshConfig` and failed `electrode windows overlap` instead of clamping invalid GUI geometry | V86 |
| B25 | 2026-04-30 | GUI 3D RM routes keyed artifacts by electrode/protocol labels but not effective measurement vector length; 16e×3ring ad/ad produced 2160 samples while cached/known 48e GREIT artifact had 5936 columns, so hot path reached `rm_matmul` with `delta_v` 2160 vs RM 5936 instead of rebuilding/declaring protocol mismatch | V87 |
| B26 | 2026-05-01 | GUI 3D `laplace_rm` auto-build inherited `jacobian_representation=auto→linearized` from fast runtime; cached context returned `_CustomLinearOperator` as `ctx["J"]`, but RM artifact build requires dense matrix and crashed at `np.asarray(ctx["J"], dtype=float)` | V88 |
| B27 | 2026-05-01 | GUI 3D `laplace_rm` could keep loading pre-fix RM artifact because signature missed dense-J build semantics; cached RM artifact also omitted fit J, so warm/disk-hit reconstruction had `simulated=None` or flat red overlay while result metadata risked carrying private `_inmem_jacobian` scratch | V89 |
| B28 | 2026-05-01 | GUI 3D RM artifacts rebuilt but still showed flat red curve: `laplace_rm` graph prior had no edges for hexa cells (shared face=4 verts, old simplex rule required 7) → RM all zero; `noser_rm` used raw volt/J under canonical λ so `JRM` projected only ~0.4% of measured diff | V90 |
| B29 | 2026-05-01 | GUI 3D smooth/GREIT routes still produced misleading images after V90: local `graph_ltl` was exactly `D.T@D == graph_laplacian`, so `curvature_rm` collapsed to `laplace_rm`; `graph_laplacian` missed EIDORS `prior_laplace` 2× face contribution; `greit3d_rm` auto-selected/used deterministic `fixture_only=true,eidors_parity=false` common-config artifacts by measurement count alone, including wrong ring geometry, and rendered full voxel cuboids with no model-component voltage fit | V91,T101,T102 |
| B30 | 2026-05-01 | After T105, 3D `noser_rm` improved but `laplace_rm`/`curvature_rm` stayed noisy and `greit3d_rm` still hid the inner inclusion. Root cause split: smooth routes cold-built singular EIDORS graph priors in measurement form (`P≈RtR⁻¹`); artifact diagnostics showed Laplace/Curvature systems at `cond≈1e21`, and same-J synthetic inclusion test gave current Laplace corr `-0.055` vs official param-form positive corr, Curvature `0.077→0.619`. GREIT native builder treated EIDORS `target_size=0.20` as absolute 0.20m radius in a 0.18m tank, so training targets covered nearly the whole volume; display hexa sizing used whole-cloud median distance, inflating masked centers into cuboids | V93,T106 |
| B31 | 2026-05-05 | GUI 2D `noser_rm` still built/applied raw Δv/J. Local 16e/208 centered-circle diagnostic: raw RM peak `1.106`, fit corr `0.945`; normalized RM peak `1.466`, fit corr `0.9996`. Raw voltage scale over-weighted boundary channels and visually flattened/shifted the inclusion | V94,T100 |
| B32 | 2026-05-05 | GUI 2D `noser_rm` after V94 could still load/build a normalized HDF5 artifact whose persisted J/RM came from stale pre-fix semantic cache. Artifact metadata said `difference_mode=normalized`, but same-geometry cold recompute showed persisted J/RM large relative drift and user-visible long-strip/crescent. Root split: RM artifact signature changed, but auto-build still called `build_shared_context(cache_scope="both")`, allowing disk/session `calc_jacobian` to feed the new HDF5; console also showed mesh-cache lines but not RM build/cache status | V95,T100 |
| B33 | 2026-05-05 | GUI could keep `_last_fwd_result` after user edited the nonuniform target/mesh/noise, so the left truth plot reflected new inputs while inverse still consumed old `boundary_voltages`/`homogeneous_voltages`; official EIDORS and direct PyEIDORS NOSER both centered the same-circle case, isolating the GUI state/provenance drift | V96 |
| B34 | 2026-05-06 | GUI 2D mesh panel displayed derived electrode length `2πR*cov/n` but returned it as explicit `electrode_length_m_override=0.19635`; the old centered experiment used coverage-only semantics (`override=None`, artifact cond≈554) while the new default built/loaded a distinct explicit-length RM (`cond≈2100`) that stretched the centered circle into a crescent despite matching voltage fit | V97 |
| B35 | 2026-05-06 | GUI close could leave CPU-bound work alive: forward/reconstruction `shutdown()` only called `QThread.quit()+wait()`, which does not interrupt long Gmsh/PETSc/Jacobian/RM calls; meanwhile forward-result electrode overlay cleanup caught AttributeError/ValueError but Matplotlib raised `NotImplementedError: cannot remove artist`, so UI update crashed before a clean stop path completed | V98 |
| B36 | 2026-05-06 | WSLg/RDP reported non-origin multi-screen geometry and can transiently report bogus terminal/screen dimensions; GUI startup used primary-screen size/placement blindly and logged nothing after `_retranslate()`, so `./eit-gui --gpu` could enter event loop with the main window off visible desktop or collapsed by bad geometry, appearing "stuck" | V99 |
| B37 | 2026-05-08 | GUI 2D conductivity widget used `Figure` tight-layout + `colorbar(ax=...)`; each pane's colorbar tick/title layout could steal different space, so truth/reconstruction with different value ranges rendered the same unit domain at different visual sizes | V100 |
| B38 | 2026-05-09 | GUI 3D sphere truth used only cell-centroid inclusion tests; with current `height=0.16`, `mesh_size=0.10`, hex z stages around electrode windows, the sphere boundary skipped/overfilled whole vertical layers and looked like uneven stacked blocks instead of a rounded ball | V101 |
| B39 | 2026-05-09 | GUI simulation inverse overwrote the absolute boundary-voltage plot with raw/normalized difference-space RM fit, so 3D RM looked like a flat red line against the forward voltage curve; geomv2 hex z-layer count allocation also rounded symmetric three-ring windows then removed/added layers via first-`argmax`, giving top/bottom non-mirrored layers; the 3D highlight mask treated tiny near-background RM noise as a real inclusion, producing a large fake shell | V102,V103,V104 |
| B40 | 2026-05-09 | Simulation Step4 stale-input status used one long QLabel line without word wrap; the text became the left form's minimum width and forced a horizontal scrollbar in the simulation control column | V105 |
| B41 | 2026-05-09 | 3D tetra simulation NOSER RM auto-built an inverse mesh finer than the displayed/forward mesh (`r=0.18,h=0.1 -> h_inv=0.03`, 3967 cells for 208 measurements), so the one-step solve preferred boundary-shell artifacts; the 3D colorbar then min/max-stretched near-background σ noise into high-contrast saturated patches | V106,V107 |
| B42 | 2026-05-09 | GUI Step2 showed 3D inhomogeneity “高” as the internal half-height/radius used by `_paint_shape`; in a `height=0.16 m` cylinder, entering `0.1` already meant half-height `0.1 > 0.08`, so changing it to `0.5` produced the same full-height inclusion and looked like height information was lost | V108 |
| B43 | 2026-05-09 | GUI Step2 still allowed 3D sphere rows to carry independent L/W/H values, and `_paint_shape(shape="circle", mesh_dimension=3)` used only `size_x` as the radius; stale or edited sphere rows could therefore ignore the visible height and render as a clipped cylinder-like inclusion | V109 |
| B44 | 2026-05-09 | GUI 3D hex forward mesh kept a fixed low Z-layer count (`max(6, refinement*3)`) regardless of cylinder height. With `height=0.5 m` and a `0.1 m` sphere, central layers were too thick, so volume-fraction painting changed whole tall cells and the truth view looked like a stretched cylinder; generator revision was bumped to `g3d4` so old cached `g3d3` meshes are not reused | V110 |
| B45 | 2026-05-11 | CUDA CEM gauge fix zeroed both the reference-electrode row/column and the global constraint row/column, deleting one electrode balance equation and changing the boundary model instead of only selecting a voltage gauge. `spd_gamg`/`3d_gamg` could still report convergence under exact EIDORS stim/meas replay but produce visually wrong boundary voltages and inverse input | V111 |
| B46 | 2026-05-11 | 2D CUDA fair comparison reused the CPU projected Jacobian because the shared EIDORS-style Jacobian path projected `grad(u)` through DOLFINx `Expression.interpolate`; with a PETSc-CUDA 2D fwd_model that interpolation could fail with `RuntimeError: Cells lists have different lengths`. For P1 simplex elements the gradient is cell-constant, so the DOLFINx interpolation was unnecessary on the failing path | V112 |
| B47 | 2026-05-11 | Simulation boundary-voltage visualization could look empty or one-sided: the fit outline was drawn above the ground-truth curve, covering it when curves overlapped; the plot did not explicitly reset Y range after 2D/3D or amplitude changes; and auto-built RM caches missing their persisted fit Jacobian could still be loaded, yielding a reconstructed image without a boundary-voltage fit overlay | V113 |
| B48 | 2026-05-11 | GUI 3D conductivity controls were laid out as one long horizontal row. Maximized windows hid the issue, but default launch width gave each simulation pane too little horizontal space, so display-mode buttons, opacity controls, toggles, and reset text clipped or spilled across the splitter | V114 |
| B49 | 2026-05-11 | 工具→计算精度只影响 ADC/正问题边界数组；仿真逆问题请求未写 `rm_dtype`，RM auto-build/load 又硬转 `float64`，且 NOSER measurement-form 先把对角 RtR 展开成 `75188×75188` dense 矩阵，∴ Float32 菜单仍报 `float64` 与相同 GiB 分配 | V115 |
| B50 | 2026-05-11 | 仿真 Step4 把 `max_iterations` 始终暴露给所有差分/RM 路由，且算法清单只保留差分/调试路由；`eidors_abs_gn` 旧配置还被归一到 `debug_full_gn`，导致新手以为差分成像可调迭代次数，同时 GUI 无法显式选择绝对成像 | V116 |
| B51 | 2026-05-12 | 数据库单帧/批量重构仍停在旧算法清单：默认“single-step” 实际未带 RM/single-step 元数据，NOSER/Laplace/Curvature RM 入口缺失，差分路径继续显示 `max_iterations`，导致数据库重构与仿真 Step4 的生产语义漂移 | V118 |
| B52 | 2026-05-12 | 偏心 3D 球体诊断被误读：Step2 显示直径但人工按半径解释；GUI 也无越界提示，3D 点云仅按绝对 σ 偏离高亮，负伪影可抢过正异常目标 | V119 |
| B53 | 2026-05-12 | 3D 点云/体高亮仍按“中位数偏离后的宽阈值”选出大量单元，且没有空间连通筛选；偏心球体的一步 RM/NOSER 射线伪影被整片高亮，视觉上盖过真实紧凑异常团 | V120 |
| B54 | 2026-05-15 | GUI GREIT 用同一套 rec-model 中心 hexa 扩展显示 2D/3D artifact：2D GREIT 被当作薄 3D 体素/菱形网格渲染，坐标观感与真值不对应；3D GREIT 默认 `8×8×5` 被圆柱 mask 后只剩约百级体素；Step4 同时显示 generic `Artifact weight` 与 GREIT `weight/NF`，控件语义重复 | V125 |
| B55 | 2026-05-22 | GUI tests used real app-data DB, so pytest temp session paths persisted into database tab; temp cleanup made preview/single-frame reconstruction load dead `/tmp/nix-shell...pytest...csv` paths. Frame regex also missed frequency-suffixed recorder files | V126 |
| B56 | 2026-05-22 | 数据库重构隐藏使用当前硬件/2D 默认 `n_elec`、网格、几何、stim/meas、drive、solver 参数；历史采集/仿真数据协议不同会静默错配正问题与逆问题 | V127 |
| B57 | 2026-05-22 | EIDORS 互操作 Profiles 页保存按钮显示 profile 表单路径，但实现读取 Import 页路径；用户在二级菜单配置页编辑 MATLAB/startup/source/output 后会保存错 profile | V128 |
| B58 | 2026-05-22 | 使用用户真实 MATLAB R2023b/EIDORS 3.12 联调时，采集阶段被 Windows 中文输出 UTF-8 解码错误中断；修复后又发现 capture 模板在局部函数作用域找 `fmdl`，导出脚本还固定 `mk_common_gridmdl('backproj')` 导致 40 点数据与 256 点模型不匹配 | V129 |
| B59 | 2026-05-25 | Plain pytest ran every integration/CUDA/GUI/hardware-facing test by default: one CUDA 3D parity case took ~13min, GUI full-button walk launched dataset generation in a background DOLFINx JIT thread and aborted during teardown, and hardware-facing tests were not hidden behind opt-in flags | V130 |
| B60 | 2026-05-25 | `complex-cuda` PETSc build set complex scalar while still linking default real-scalar Hypre; configure stopped with `HYPRE scalar numbers configuration is different than the requested type complex` before GPU complex CEM smoke could run | V131 |
| B61 | 2026-05-25 | GUI/launcher exposed implementation matrix (`cpu`/`cuda`/`complex`/`complex64`/`complex-cuda`/`complex64-cuda`) as user choices; one-click `--gpu` still meant real-only CUDA, so users had to understand PETSc scalar immutability before using frequency-domain complex admittance | V132 |
| B62 | 2026-05-26 | 复值仿真 `noser_rm` 请求避开 real single-step 后仍把 GUI RM route label 当作 `EITSystem.difference_preset`; core preset 校验拒绝 `noser_rm`，原生复值 GN 进入前崩溃 | V134 |
| B63 | 2026-05-26 | `complex64-cuda` GUI 中实值 3D 细网格仍按复值 PETSc 分配；CUDA CEM 非 direct KSP 强制 dense LU fallback，`138240` 单元触发 `MatSetUp(densecuda)` 显存 OOM | V135 |
| B64 | 2026-05-26 | GUI 进程本身运行在 `complex64-cuda` 时，用户输入实值参数也继承复值 PETSc scalar，导致前向/逆问题无法按最省内存的 real CUDA runtime 执行；前后端未分离使 GUI runtime 选择泄漏到数值求解路线 | V136 |
| B65 | 2026-05-26 | 前后端分离后 worker 每次请求都使用临时 `XDG_CACHE_HOME`，DOLFINx/FFCx JIT 与 profile 级 setup cache 无法复用；新正问题反复冷编译，速度比原同进程路线慢 | V137 |
| B66 | 2026-05-26 | 即使 profile cache 已持久化，GUI 每次正问题仍重新启动 Python/Nix worker，Python import、DOLFINx runtime、进程内 EITSystem/RM cache 都被丢弃；新设置正问题仍比原单进程路线多固定启动成本 | V138 |
| B67 | 2026-05-26 | 常驻 worker 优化后仍把最简单 2D 实值正问题强制跨进程；第一次点击要先启动/握手后端，固定开销超过实际 2D forward 计算，体验慢于前后端未拆分时的进程内路线 | V139 |
| B68 | 2026-05-26 | 2D real forward 已恢复进程内路线后，首次点击仍需用户等待 DOLFINx/FFCx JIT 和 runtime 初始化；没有 GUI idle prewarm，冷启动成本仍暴露在交互路径上 | V140 |
| B69 | 2026-05-26 | `ForwardSolverController.solve()` 无 accepted bool / `is_busy` 契约；GUI prewarm 启动 worker 后被当作未接受，预热状态机掉忙碌标记，first-click 仍可重启同一 cold solve | V140 |
| B70 | 2026-05-26 | GUI Step3 正问题请求仍走完整 `EITSystem.setup()`；仅算 forward 也初始化 `DirectJacobianCalculator`、regularization、`GaussNewtonReconstructor`，冷启动把逆问题成本压到第一次正问题点击 | V141 |
| B71 | 2026-05-26 | 启动预热先等 `900ms`，再走普通 input debounce `500ms`；用户进入仿真页后前 `~1.4s` 点击仍落入 cold forward 路径 | V140 |
| B72 | 2026-05-26 | FFCx 持久 cache 中可留下 `.c.cached` + `.so` 但缺 `.c` marker；下次进程认为 cache miss 并重编译，最后 `open(ready_name, "x")` 命中旧 `.c.cached` 抛 `FileExistsError`，破坏持久 cache 加速 | V142 |
| B73 | 2026-05-26 | 2D generated forward 每次用临时 mesh 文件；mesh disk/process cache 与 forward static setup cache key 无法跨请求命中，简单 2D 热路径仍重复 mesh/setup 工作 | V143 |
| B74 | 2026-05-26 | GUI prewarm 完成后只保存 ready signature，用户点击仍启动一次热 forward；虽然比 cold 快，但第一次交互仍做重复计算且不是真正零等待 | V140 |
| B75 | 2026-05-27 | V143 稳定 2D generated mesh cache 只按几何参数命名；GUI 运行在 `complex64` profile 时复用旧 `mesh_16e_r1_ref5_cov0p5` 的 `Mesh_float64`，DOLFINx 组装 `Form_complex64` 要求 `Mesh_float32`，导致 Step3 forward/prewarm 连续抛 `incompatible function arguments` | V144 |
| B76 | 2026-05-27 | 2D `complex64-cuda` GUI 启动预热与用户点击/后续预热可在同一进程同时进入 FFCx JIT；backend worker 有 profile lock，但 in-process smart route 没用同一锁，第二个编译任务等待同名 cache marker 超时并向 GUI 报 `JIT compilation timed out` | V145 |
| B77 | 2026-05-27 | 上一次 JIT 编译实际生成了 `.c/.o/.so`，但缺少 FFCx 用作 ready marker 的 `.c.cached`；后续同 profile 看到 `.c` 已存在后只等待 `.c.cached`，即使 `.so` 已可导入也最终 timeout | V142 |
| B78 | 2026-05-27 | 3D 仿真/预热切换时 GUI `shutdown()` 直接 `QThread.terminate()` 正在 `proc.stdout.readline()` 中等待 persistent backend 的 forward 线程；Python `BufferedReader` 被异步打断后，下一次复用该 worker 管道触发 `RuntimeError: reentrant call inside <_io.BufferedReader>` | V146 |
| B79 | 2026-05-27 | 3D 设置变更触发外部 `cuda` persistent worker 内部 FFCx 编译失败并留下新的 0 字节 `libffcx_forms_*.c`; 该错误作为 `BackendWorkerRequestError` 返回，父进程只把它当求解失败，不驱逐 worker、不即时清理新锁、不重试，于是 GUI 直接显示 `JIT compilation timed out` | V147 |
| B80 | 2026-05-27 | GUI 把 2D 的完整 forward 预热策略原样用于 3D；用户调整三维参数时后台自动排队/运行昂贵 3D forward，和手动运行争抢 CPU/GPU/内存，导致首次三维交互看起来更慢且内存占用过高 | V148 |
| B81 | 2026-05-27 | 3D persistent worker 热缓存保留 Python/DOLFINx/PETSc 大堆；大型 solve 完成后进程常驻使 WSL/GPU 会话继续高 RSS，占内存但没有自动回收阈值 | V187 |
| B82 | 2026-05-27 | 大 3D 网格已自动切点云但仍把全部 cell centers 交给 PyVista/Matplotlib；十万级点云首次展示与交互仍产生 UI 渲染内存尖峰 | V188 |
| B83 | 2026-05-27 | WSLg 禁用嵌入式 VTK 后，大 3D 点云仍先尝试 PyVista offscreen；仅导入/初始化 VTK 就可造成首次展示慢与额外内存峰值，即使最终只是点云显示 | V189 |
| B84 | 2026-05-27 | 点云渲染虽已抽样，但抽样前为保留异常区域仍对全量 cell centers 跑空间连通筛选；大网格上 `cKDTree` 成为新的显示前 CPU/内存尖峰 | V190 |
| B85 | 2026-05-27 | 3D viewer 入口无条件 `coords→float64`、`cells→int64`、`sigma→float64`；CUDA/worker 已给 `float32/int32` 大数组时，显示前先复制并放大内存 | V191 |
| B86 | 2026-05-27 | V191 保住 display entrypoint dtype 后，`array_geometry_cache` 仍在 cell-center 派生前把整数连通性统一转 `intp`；Linux 上 `int32` 大网格再次扩成 `int64` | V192 |
| B87 | 2026-05-27 | `array_geometry_cache` 用 `coords[cells].mean(axis=1)` 派生 cell centers；大 3D 网格先构造 `(n_cells,verts,dims)` 临时数组，峰值显著高于最终 centers | V193 |
| B88 | 2026-05-27 | 3D worker-only prewarm 只启动 Python 子进程；DOLFINx/PETSc/PyEIDORS forward 栈仍在首次真实请求中懒导入，用户点击路径继续承担 heavy import/runtime 初始化延迟 | V194 |
| B89 | 2026-05-27 | backend worker HDF5 协议对所有非空数组用 gzip；三维结果返回路径会为大坐标/连通性/电导率数组支付高 CPU 压缩成本，拖慢首次 solve 完成后的 GUI 接收 | V196 |
| B90 | 2026-05-27 | 仿真 metrics 面板即使真值/重构使用同一网格，也总是对 cell centers 建最近邻索引；同时在采样阶段把 `float32/int32` 三维数组扩成 `float64/int64`，增加 solve 后 UI 指标计算峰值内存 | V197 |
| B91 | 2026-05-27 | 2D/投影显示的 cell→node 平均用 `np.repeat` 展开每个单元值到所有顶点；大网格显示时额外分配按连通性展开的临时数组，并把 float32 电导率提升到 float64 | V198 |
| B92 | 2026-05-27 | `backend_worker_protocol` 顶层导入 forward/reconstruction controller；forward-only worker 在读写协议时也会加载 reconstruction 栈和更多数值依赖，增加冷启动和常驻内存 | V199 |
| B93 | 2026-05-27 | GUI 边界三角/2D 投影 helper 先把整块 `cell_connectivity` 转成 `int64`，即使输入已经是 `int32`；有效三角网格还会经 boolean filter 复制一次，放大大网格显示内存峰值 | V200 |
| B94 | 2026-05-27 | Matplotlib 3D fallback 的 point-scalar 分支用 `sigma[cells].mean(axis=1)`；大三维网格会先构造按单元顶点展开的标量矩阵，并把 float32 显示值提升到 float64 | V201 |
| B95 | 2026-05-27 | 3D 点云虽已设置显示点数上限，但抽样 fallback 仍先构造 `np.arange(n_points)`，背景补样仍构造 `flatnonzero(~anomaly_mask)`；百万级网格为选少量显示点仍分配全量索引数组 | V202 |
| B96 | 2026-05-27 | 点云异常检测入口对 `cell_sigma` 使用 `dtype=float`/`float64`，即使 worker/GUI 已传入 float32 显示数组，也会在抽样前复制放大一整份电导率向量 | V203 |
| B97 | 2026-05-27 | 点云抽样调用 `_cell_anomaly_mask(..., cell_centers=None)` 已明确跳过空间连通筛选，但 `_spatially_coherent_anomaly_mask` 仍先 `flatnonzero(mask)`，为不会使用的空间候选分配全量异常索引 | V204 |
| B98 | 2026-05-27 | `_cell_anomaly_mask` 在知道是否拥挤前先构造 `finite_score_values` 与 `finite_scores`；稀疏点云高亮只需峰值/计数却复制整条 score 向量 | V205 |
| B99 | 2026-05-27 | 3D 显示颜色范围计算对整条 `cell_sigma` 先做 `float64` 拷贝，再复制 finite 子集；大三维结果即使全有限也为 min/max/median 额外分配整向量 | V206 |
| B100 | 2026-05-27 | PyVista 点云 actor 构建先执行 `np.asarray(centers, dtype=float)` / `np.asarray(cell_sigma, dtype=float)` 再取 `sample_idx`；已设置点数上限时仍为全量点云扩成 float64 | V207 |
| B101 | 2026-05-27 | `_cell_center_sigma` 已有低峰值 `_cell_mean_values` 可用，但 point-scalar 分支仍执行 `values[cells].mean(axis=1)`；大点标量三维网格渲染前会分配展开标量矩阵 | V208 |
| B102 | 2026-05-27 | 空间连通高亮已只在显示抽样集运行，但 `_spatially_coherent_anomaly_mask` 仍先把整块 sampled centers 转 float64，再切候选中心；候选很少时复制范围过大 | V209 |
| B103 | 2026-05-27 | RM 矩阵与 worker/display 路线已支持 float32，但 RM artifact loader 与 reconstruction controller 对 `node_coords` 仍强制 `float64`；三维重构显示/缓存热路径会额外复制并放大几何数组 | V210 |
| B104 | 2026-05-27 | `CellMesh.cell_centers()` 仍用 `coordinates[cells].mean(axis=1)`；3D dual-mesh/RM/GREIT 几何派生会先分配按单元顶点展开的坐标块 | V211 |
| B105 | 2026-05-27 | mesh-derived HDF5/process cache 修过 `EITMesh.cell_centers()` 复用，但底层 `_cell_centers` / `_cell_measures` 仍用 `coords[cells]`/`points=coords[cells]`；首次派生大 3D mesh 时仍产生完整 cell-point 临时数组 | V212 |
| B106 | 2026-05-27 | `dual_mesh._cell_centers(mesh)` generic fallback 仍用 `coords[cells].mean(axis=1)`；非 `CellMesh` 适配对象走 coarse2fine 时会展开完整坐标块 | V213 |
| B107 | 2026-05-27 | `_locate_points_in_cell_mesh` 为 containment 先构造 `cell_vertices = coords[cells]` 再取 min/max/候选 simplex；大 coarse mesh 定位会产生完整 cell-vertices 临时数组 | V214 |
| B108 | 2026-05-27 | `VoxelGrid.cell_centers()` 用 `np.meshgrid` 为每个轴生成完整网格后再 `stack`；3D GREIT/RM 体素中心生成会产生 dim 份全尺寸中间数组 | V215 |
| B109 | 2026-05-27 | `build_greit3d_distribution` 先为 x/y/z 构造 3 个完整 `meshgrid`，再 `stack` 候选中心；大 3D 训练分布首次生成时额外占用多份全尺寸网格 | V216 |
| B110 | 2026-05-27 | GREIT `_metric_centers` fallback 用 `np.meshgrid` 生成指标坐标；无显式 centers 的大体素指标计算会分配 dim 份中间网格 | V217 |
| B111 | 2026-05-27 | `_gauss_reference_offsets` 为 Gauss offsets/weights 各构造一组 `meshgrid` 再 stack/product；GREIT desired-image 采样 helper 仍有不必要网格中间数组 | V218 |
| B112 | 2026-05-27 | GREIT `_default_radius` 用 `centers[:, None, :] - centers[None, :, :]` 构造全量两两距离矩阵；三维训练点变多时首次 RM/GREIT 初始化会出现 O(n²) 内存峰值 | V219 |
| B113 | 2026-05-27 | GREIT `_nearest_center_distance` 在 `_infer_center_spacing` 兜底路径中同样构造全量两两距离矩阵；不规则/适配网格的三维 desired-image extents 推断会出现 O(n²) 冷启动峰值 | V220 |
| B114 | 2026-05-27 | `_greit_sigmoid_adaptive_gauss` 先保存完整 `center_distances = norm(rec_centers[:,None]-targets[None])`，随后循环 target 只用一列；adaptive desired-image 冷启动多占一整份 `n_cells×n_targets` 矩阵 | V221 |
| B115 | 2026-05-27 | `_greit_sigmoid_average_over_samples` 对每个 target 执行 `flat_samples - target.reshape(1,3)` 后再求范数；Gauss/Sobol desired-image 采样会为距离计算临时分配完整 sample-coordinate 差分矩阵 | V222 |
| B116 | 2026-05-27 | GUI GREIT center-cloud hexa/quad display geometry 强制 `centers→float64`，用广播表达式展开角点，用 Python nested list 构造 cells，并在 2D rec-model padding 中用 `np.column_stack`；三维重构显示会放大 float32 几何并产生对象/拼接临时 | V223 |
| B117 | 2026-05-27 | GREIT desired-image 的 gauss/sobol/adaptive base/fine 路径先构造完整 `samples = rec_centers[:,None,:] + offsets[None,:,:] * extents[:,None,:]`；三维大网格会额外持有 `n_cells×n_samples×3` 临时张量 | V224 |
| B118 | 2026-05-27 | GREIT center desired-image 路径用 `rec_centers[:,None,:] - xyz_matrix.T[None,:,:]` 一次性构造中心-目标差分张量，再生成全目标距离矩阵；center/fast 模式会多持有大块中间数组 | V225 |
| B119 | 2026-05-27 | GeomV2 hex O-grid 3D mesh seed core 用 `np.meshgrid(core_axis, core_axis)` 构造两块完整 core 坐标数组；冷生成三维网格时有可避免的 core-grid 中间分配 | V226 |
| B120 | 2026-05-27 | GUI metrics `_nearest_resample` 在 SciPy/cKDTree 不可用或失败时用 `target[:,None]-source[None,:,:]` 构造全量距离张量；三维大网格指标兜底路径会出现 O(n_target×n_source) 内存峰值 | V227 |
| B121 | 2026-05-27 | Lazy adjoint Hessian diag chunk 中 `np.real(conj(contrib)*contrib) * weights[:,None] * cell_areas[None,:]**2` 为每个 chunk 额外构造权重/面积广播矩阵；3D NOSER/fast diag 路径峰值内存被放大 | V228 |
| B122 | 2026-05-27 | Eager `JacobianLinearization.to_dense()` 每个 block 先生成 `einsum` 结果再乘 `cell_areas[None,start:end]` 后赋值；显式 Jacobian/RM 构造路径为每块多分配一份临时敏感度矩阵 | V229 |
| B123 | 2026-05-27 | Fast GN runtime 对 dense Jacobian 和测量权重先执行 `measurement_jacobian_np * meas_weight_np[:,None]`，再传入 fast solver；大三维 dense J/RM 路径会立刻复制一整份加权 Jacobian | V230 |
| B124 | 2026-05-27 | Fast GN Woodbury 路径先构造 `ja_inv = J_weighted_dense_np * inv_diag[None,:]` 再算小系统；参数量很大时多持有一整份 dense Jacobian 大小的临时矩阵 | V231 |
| B125 | 2026-05-27 | Hardware reconstruction widget `_prepare_grid_cache` 用 `np.meshgrid(x_coords,y_coords)` 构造两块 256×256 网格，再 `column_stack` 成采样点；GUI 缓存刷新多分配两块中间数组 | V232 |
| B126 | 2026-05-27 | GREIT common-config warmup fixture RM 用整块 `rows[:,None]`/`cols[None,:]` 广播生成 sin/cos 和最终矩阵；首次预计算/缓存写入时峰值内存约为多份 RM 大小 | V233 |
| B127 | 2026-05-27 | `project_measurement_jacobian` 归一化先生成整块 `projected / reference[:,None]`，反向 orientation 再 `-projected`；大三维 Jacobian 差分投影会多持有一到两份完整矩阵 | V234 |
| B128 | 2026-05-27 | `build_difference_frames` 先构造 diff，再 `diff / safe` 和 `-diff` 额外复制整帧矩阵；normalization 即使无需 clamp 也复制完整 reference batch | V235 |
| B129 | 2026-05-27 | VoxelGrid/GREIT cartesian centers 每个轴用 `np.tile(np.repeat(...))` 生成完整列临时数组再赋值；大体素网格中心生成会多分配一条 `n_cells` 列/轴 | V236 |
| B130 | 2026-05-27 | GREIT metric `_weighted_centroid` 用 `coords * weights[:,None]` 构造 `n_cells×dim` 临时矩阵；指标计算在大三维网格上多分配一块中心矩阵大小临时数组 | V237 |
| B131 | 2026-05-27 | `cuda_structured` 组装 sigma state 时先在 CPU/NumPy 构造 `(1.0 / diag)[:, None]` 再传 torch；三维大系统预条件器初始化多一次 CPU 临时和形状扩展拷贝 | V238 |
| B132 | 2026-05-27 | synthetic circle phantom 与 core helper phantom 用 `centers[:, :dim] - center[None,:]`/`dof_coordinates[:, :2] - center[None,:]` 构造 `n_points×dim` 临时矩阵；大网格异常体掩码多一次二维距离临时 | V239 |
| B133 | 2026-05-27 | digit metrics 与 holdout fallback parameter points 各自用 `np.meshgrid(xs,ys)` 构造两块 `side×side` 网格再 `column_stack`；扫参/holdout 数据工具存在重复网格临时和重复实现 | V240 |
| B134 | 2026-05-27 | holdout/bucket `_weighted_structure` 重复实现 centroid/covariance，均用 `coords * weights[:,None]` 和 `centered * weights[:,None]` 构造二维临时；扫参结构指标多分配两块矩阵且逻辑重复 | V241 |
| B135 | 2026-05-27 | bucket dense `_source_gradient` 先构造 `diff = points-electrode`，再用 `r2[:,None]` 广播除法产生返回矩阵；每次电极/单元梯度计算多持有一份 `n_points×2` 临时 | V242 |
| B136 | 2026-05-27 | bucket dense `_source_potential` 先构造 `diff = points-electrode` 再求 `r2`；每次 source voltage 计算多持有一份 `n_points×2` 临时 | V243 |
| B137 | 2026-05-27 | EIDORS noise `_broadcast_v2` 行参考电压广播分支先建 `v2[:,None]` 视图再 `broadcast_to(...).astype(copy=True)`；可直接写最终输出矩阵减少一次广播路径开销 | V244 |
| B138 | 2026-05-27 | CacheManager miss 后直接 `compute_fn()`；GUI/backend 同 key 首轮并发请求会重复构造昂贵 mesh/Jacobian/RM artifact，放大首次加载 CPU/内存/JIT 压力 | V245 |
| B139 | 2026-05-27 | cache key `hash_path` 用 `Path.read_bytes()`、`hash_array` 用 `a.tobytes()`；仅计算 key/signature 时会为大 mesh/RM/Jacobian 依赖额外复制整块文件或数组 payload | V246 |
| B140 | 2026-05-27 | 复值 sigma fingerprint 拆分后 `linearized.py` 与 `gauss_newton_startup_cache.py` 各新增 SHA256 调用；T90 audit doc/test baseline 未同步，hash audit gate 报 inventory drift | V247 |
| B141 | 2026-05-27 | linearized/direct-Jacobian/GN startup sigma hash 仍局部拼接 `sigma_values.tobytes()`；大三维 sigma fingerprint 只为 cache key 又复制整条参数向量 | V248 |
| B142 | 2026-05-27 | `_JacobianActionBundle` 已新增 `hessian_diag` 以支持低峰值对角路径，但 `test_gn_runtime_contract_freeze` 仍锁旧字段集合，组合验证失败 | V249 |
| B143 | 2026-05-27 | GN fast linear-system sparse regularization/ROM cache hash 仍在多处 `np.ascontiguousarray(...).tobytes()`；大 CSR/J/RM/snapshot 只为 digest 多复制整块 payload | V250 |
| B144 | 2026-05-27 | CUDA structured/reduced snapshot/sparse Bayesian/GREIT registry 剩余 digest 点仍直接 `arr.tobytes()`；结构化 forward、ROM、SVD/GREIT registry 热路径为 hash 多复制数组 payload | V251 |
| B145 | 2026-05-27 | 大 3D 结果已自动切到 point-cloud 后，WSLg/headless 默认仍到 60000 cells 才跳过 PyVista offscreen；常见 12000+ cells 首帧仍先 import/初始化 VTK plotter，拖慢首次显示并抬高内存 | V252 |
| B146 | 2026-05-27 | GUI array-geometry cache 只有条目数 LRU，没有 resident byte budget；连续查看多个大 3D 结果时 `cell_centers` 可在 GUI 进程内按条目数累积 | V253 |
| B147 | 2026-05-27 | `DiskCacheStore` 写入先 `pickle.dumps`/`gzip.compress` 再 `Path.write_bytes`，读取先 `Path.read_bytes`/`gzip.decompress`；大 3D Jacobian/RM 缓存命中/落盘会额外持有整块序列化 payload | V254 |
| B148 | 2026-05-27 | GUI reconstruction `_array_pair_hash` 为 RM/GREIT mesh signature 对 `node_coords/cell_connectivity` 直接 `.tobytes()`；大 3D 结果只为签名复制整块几何/连通性 payload | V255 |
| B149 | 2026-05-27 | `scripts/common/gn_difference_runner` 构造 `sigma_hash` 时对三维背景 `sigma_bg` 直接 `.tobytes()`；CLI/benchmark/shared-context 冷启动只为 cache key 复制整条参数向量 | V256 |
| B150 | 2026-05-27 | `ProcessCacheStore.put` 对未知/对象型 cache value 通过 `pickle.dumps` 估 resident size；稀疏矩阵、SimpleNamespace 等包大数组时，L1 插入会额外序列化复制整块 payload | V257 |
| B151 | 2026-05-27 | `EITForwardModel` mesh content hash 与 sigma/z/pattern scalar hash 直接拼接 `.tobytes()`；forward setup/factor cache key 生成会为大三维 mesh/sigma 再复制整块 payload | V258 |
| B152 | 2026-05-27 | RtR prior、TV-IRLS、RM signature、GREIT signature 剩余 helper 仍用 `.tobytes()` 拼 digest payload；大三维 sparse/dense prior、RM/GREIT artifact 签名只为 cache key 多复制整块数组 | V259 |
| B153 | 2026-05-27 | `FrameRingBuffer.write` 把 real/imag 转整帧 `.tobytes()` 后写共享内存，`_read_slot` 先 `bytes(buf[slice])` 再 copy；实时采集帧传递多一次整帧 bytes 分配 | V260 |
| B154 | 2026-05-27 | synthetic parity / difference-runtime / KSP-session / mesh-IO benchmark 脚本仍在大数组 hash 处调用 `.tobytes()`；诊断/benchmark 冷跑会为 NOSER diag、sigma sequence、mesh tag pairs 多复制 payload | V261 |
| B155 | 2026-05-27 | GUI `array_geometry_cache` signature 对 dims/shape 仍用 `.tobytes()`，payload 直接单次 `memoryview` update；签名路径与其他 cache key streaming helper 不统一 | V262 |
| B156 | 2026-05-27 | EIDORS noise、RM reference frames、TV-IRLS initial batch 的 vector→frame 扩展仍用 `np.broadcast_to(...).copy()`/`.astype(copy=True)`；批量帧准备多一层 broadcast view/copy 路径且与 V244/V235 低峰值约束漂移 | V263 |
| B157 | 2026-05-27 | GREIT desired-image cell spacing/extents 多处用 `np.broadcast_to(extent,(n_cells,3)).copy()`；三维 GREIT 冷启动为重复 extent 矩阵走额外 broadcast-copy 路径 | V264 |
| B158 | 2026-05-27 | DynamicMeasurementSequence 1D bad-channel mask 先 `np.broadcast_to` 成只读 frame view，再 `np.ascontiguousarray` 复制；长序列 mask 准备与低峰值广播约束漂移 | V265 |
| B159 | 2026-05-27 | GUI forward solver 与 dataset generator 为 3D 体积分数绘制先构造 `node_coords[cell_connectivity]`，`_paint_shape` 再一次性生成所有单元 sample points；三维仿真/批量数据生成启动前多持有完整 cell-vertex/sample 张量 | V266 |
| B160 | 2026-05-27 | GN difference CLI / synthetic parity / difference-runtime benchmark 的 measurement-space 单步求解先构造 `jacobian * inv_reg_diag[None,:]` 或 `jw_scaled`; 大三维诊断/benchmark 会多持有完整 scaled dense Jacobian | V267 |
| B161 | 2026-05-27 | Hardware reconstruction widget T192 去掉 `meshgrid` 后仍用 `np.tile/np.repeat` 构造完整 x/y 采样列临时数组；网格缓存刷新仍多持有两条 full-grid 中间向量 | V268 |
| B162 | 2026-05-27 | synthetic parity / difference-runtime benchmark 在测量加权时先构造 `jacobian_weighted = jacobian * sqrt_weights[:,None]`；大三维脚本会在原始 dense J 之外再持有一整份 weighted J | V269 |
| B163 | 2026-05-27 | `DiskCacheStore` put/get 已改为流式 pickle/gzip IO，但类里残留未使用 `_serialize/_deserialize`，未来调用会重新引入整对象 `pickle.dumps` 缓冲 | V270 |
| B164 | 2026-05-27 | GN difference ROM reduced-RM 构建 `UᵀRU` 时先创建 `r_diag[:,None] * basis`；大三维 reduced basis 会在 basis 外再持有一整份 scaled basis | V271 |
| B165 | 2026-05-27 | 多个 3D benchmark/diagnostic phantom/ROI mask 使用 `np.linalg.norm(coords - center[None,:], axis=1)`；大三维脚本每次 mask 多分配 `n_points×dim` 差值矩阵 | V272 |
| B166 | 2026-05-27 | GUI planar quad→triangle projection 用 `np.repeat(np.arange(n_cells),2)` 构造 source indices；大网格重绘时 source 映射走 repeat 展开路径 | V273 |
| B167 | 2026-05-27 | Hardware reconstruction grid cache 计算 barycentric weights 后用 `np.column_stack((bary,1-sum))` 构造完整 valid-point 权重临时矩阵再赋回最终 `weights` | V274 |
| B168 | 2026-05-27 | Native complex normal-step 正则化 `LinearOperator` 转 dense 时先构造完整 `np.eye(n)`，再 `column_stack` 每列 matvec 输出；大参数正则算子会多持有一块 n×n eye 和列列表 | V275 |
| B169 | 2026-05-27 | GN ROM synthetic snapshot 构建用 `np.column_stack([rhs,de,...])` 拼接多列参数向量；大三维 ROM 冷启动会走通用 column-stack 拼接路径而非直接填最终矩阵 | V276 |
| B170 | 2026-05-27 | Reduced `SnapshotBank.matrix` / `select_snapshot_matrix` 用 `np.column_stack` 拼接 bank/synthetic/cached snapshot block 和去重列；大三维 ROM snapshot bank 合并会多走通用拼接临时路径 | V277 |
| B171 | 2026-05-27 | Reduced `merge_orthonormal_bases` 用 `np.column_stack(valid_blocks)` 合并 POD basis；大三维 ROM 多个 basis 合流到 QR 前会额外分配一份完整拼接临时矩阵 | V278 |
| B172 | 2026-05-27 | Reduced GN `build_reduced_operator` 对每个 basis 列应用正则化后用 `np.column_stack` 拼接 `R(U)`；大三维 reduced rank 增大时会多走一轮列表加完整拼接临时矩阵 | V279 |
| B173 | 2026-05-27 | Sparse Bayesian coarse matrix 先生成 grouped Jacobian 列列表再 `np.column_stack`；大三维粗层分组时会额外保留列列表和完整拼接临时矩阵 | V280 |
| B174 | 2026-05-27 | Dual-mesh `to_dense()` 为每个 coarse cell 构造完整 `np.eye(n)` 再 `np.column_stack` matvec 列；调试/缓存误用到大三维 dual mesh 时会多持有 eye 和列列表 | V281 |
| B175 | 2026-05-27 | GREIT finite-target `vi` / contracted `Y` response 构建先收集所有目标列再 `np.column_stack`；三维 RM 冷构建训练目标多时会额外持有列列表和完整拼接临时矩阵 | V282 |
| B176 | 2026-05-27 | GREIT finite-target conductivity 生成对每个 target 计算 `np.linalg.norm(fwd_centers-center)` 并 append sigma 行；三维训练目标多时反复分配 `n_cells×3` 差值矩阵和行列表 | V283 |
| B177 | 2026-05-27 | GREIT spherical/blob target 生成对每个 target 计算 `np.linalg.norm(cell_centers-center)` 并 append target/mask；等效球 mask 也用 `coords-center` 排序，大三维目标集会反复分配二维差值矩阵 | V284 |
| B178 | 2026-05-27 | GUI 3D hex 体积分数采样每次 `_cell_volume_sample_points` 都用 `np.column_stack` 重建 64×8 插值权重，matplotlib 3D 电极 patch 也用 `np.column_stack` 构造 lower/upper；粗网格/电极刷新会反复分配相同形状临时矩阵 | V285 |
| B179 | 2026-05-27 | GREIT `_as_xyz_points` 对 2D 点补 z 轴时用 `np.column_stack([points, zeros])`；几何入口会多构造一份拼接临时矩阵 | V286 |
| B180 | 2026-05-27 | Traditional Jacobian P1 affine cache用 `np.column_stack`，传统 electrode Jacobian 和 measurement projection 用 block 列表再 `np.vstack`；三维电极/测量多时会额外保留块列表和完整拼接临时矩阵 | V287 |
| B181 | 2026-05-27 | `RtRPrior.as_RtR(dense=True)` 对 matrix-free/callable prior 先构造完整 `np.eye(n)` 再 `column_stack` 每列 apply；大参数 prior materialization 会多持有 eye 和列列表 | V288 |
| B182 | 2026-05-27 | FEMx `cell_midpoints` 用 `coords[c2v].mean(axis=1)` 展开完整 cell×vertex×dim 坐标块，`estimate_radius` 用 `coords-center` 差值矩阵；三维网格初始化/数据生成会产生不必要峰值内存 | V289 |
| B183 | 2026-05-27 | `JacobianLinearization.__post_init__` 对每个 stim 的 adjoint gradients 使用 `np.stack` 生成 block；三维 matrix-free 线性化初始化会走通用堆叠路径并多保留输入 tuple 到输出复制过程 | V290 |
| B184 | 2026-05-27 | `EidorsJacobianAdapter` Torch 组装路径用 `np.stack` 构造每个 stim 或全量 adjoint gradient block；三维 GPU Jacobian 组装会多走通用堆叠临时路径 | V291 |
| B185 | 2026-05-27 | Dynamic Kalman predicted/filtered/smoothed states、projected observations 和 spatiotemporal GN rowwise baseline 都通过行列表再 `np.vstack` 拼接；三维动态窗口会额外保留行列表和完整拼接临时矩阵 | V292 |
| B186 | 2026-05-27 | Temporal TV postprocess 对每帧 TV refine 后用 `np.vstack(refined_rows)` 汇总；长时序三维 ROI 后处理会多持有行列表和完整拼接临时矩阵 | V293 |
| B187 | 2026-05-27 | `solve_tv_irls_batch` 对每帧 TV-IRLS result values 先建列表再 `np.vstack`；三维动态批量 IRLS 会额外分配拼接临时矩阵 | V294 |
| B188 | 2026-05-27 | GREIT finite-target `xyzr` 和 auto distribution bounds 使用 `np.vstack` 拼固定尺寸小矩阵；冷构建路径仍走通用堆叠并不利于后续禁止 stack 临时的回归约束 | V295 |
| B189 | 2026-05-27 | Mesh-derived tetra volume 每个四面体用 `np.vstack` 构造 3x3 行列式矩阵；三维派生几何缓存构建在 cell 多时会重复走通用堆叠 | V296 |
| B190 | 2026-05-27 | Cross-layer/hybrid 3D 测量协议将 same-layer 与 cross-layer 矩阵通过 `np.vstack` 拼接；多层电极配置初始化会多走一次通用拼接临时路径 | V297 |
| B191 | 2026-05-27 | Bucket-domain `_circle_nodes` 用 `np.column_stack` 生成边界点、再用 `np.vstack` 合并 boundary/interior；密集 bucket 审计会额外分配拼接临时矩阵 | V298 |
| B192 | 2026-05-27 | Bucket dense 电极中心用 `np.column_stack`，sensitivity 先 append 每个测量行再 `np.vstack`；密集 208/256 测量 sweep 会多持有行列表和完整拼接临时矩阵 | V299 |
| B193 | 2026-05-27 | Geometry exchange boundary edge 收集 edge 数组列表后 `np.vstack`；大边界导出会额外保留列表和拼接临时矩阵 | V300 |
| B194 | 2026-05-27 | Electrode label overlay 对每个 tag 的 facet segment 点使用 `np.vstack` 求 centroid；电极边界多 segment 时会多分配临时点矩阵 | V301 |
| B195 | 2026-05-27 | Frame CSV writer 用 `np.column_stack([real,imag])` 构造两列临时矩阵；高频采集逐帧写盘时会产生不必要拼接分配 | V302 |
| B196 | 2026-05-27 | GUI electrode overlay arc/patch 采样用 `np.column_stack` 构造 segment/lower/upper 并用 `np.vstack` 合并 patch 点；三维电极 overlay 构建会多保留 patch 点列表和拼接临时矩阵 | V303 |
| B197 | 2026-05-27 | `StimMeasPatternManager` 多环电极长度用 `np.tile` 展开、measurement selector 先收集每个 stim 的 bool 向量再 `np.concatenate`；三维多层电极初始化会多走一轮通用拼接/平铺临时数组 | V304 |
| B198 | 2026-05-27 | `assemble_sigma_contact_normal_system` 用 `np.concatenate` 拼接 sigma/contact RHS；大三维联合反演 normal system 构建会在两个子块之外再分配完整 RHS 拼接临时 | V305 |
| B199 | 2026-05-27 | 动态 spatiotemporal GN 与 TV/Huber IRLS 将逐帧 RHS block 列表通过 `np.concatenate` 拼最终右端；三维动态窗口会额外持有 block 列表和完整拼接临时 | V306 |
| B200 | 2026-05-27 | GUI forward/reconstruction contact-impedance helper 对“输入长度整除总电极数”的多环配置用 `np.tile` 展开；三维设置切换会多分配平铺临时数组且两处逻辑重复漂移 | V307 |
| B201 | 2026-05-27 | GUI 3D point-cloud 采样将 anomaly/background 两段索引用 `np.concatenate` 合并；大体素首次显示会额外分配 selected-index 拼接临时 | V308 |
| B202 | 2026-05-27 | measurement moving-average resume 路径先 `np.concatenate([prior_tail,batch])` 再累计和/截 history tail；在线滤波长窗口会额外持有历史+新帧拼接矩阵 | V309 |
| B203 | 2026-05-27 | BoundaryVoltagePlotWidget y-range 计算收集各 series 有限值后 `np.concatenate` 求全局 min/max；GUI 电压拟合重绘会额外分配完整合并曲线临时 | V310 |
| B204 | 2026-05-27 | GN line-search、GREIT extent、dual-mesh barycentric 小向量路径仍用 `np.concatenate`/`np.repeat`；核心 inverse 源码 stack-budget 扫描残留通用拼接/重复调用 | V311 |
| B205 | 2026-05-27 | holdout/bucket dense comparison plotters 为 sigma 色标范围把所有字段 `np.concatenate` 成一条长向量；数据 sweep 出图会额外持有完整字段拼接临时 | V312 |
| B206 | 2026-05-27 | cache array hash fallback 仍含 `.tobytes()`、process cache 未知对象 size fallback 用 `pickle.dumps`、GeomV2 hex core seed 用 `np.broadcast_to`；三维冷启动/cache bookkeeping 仍有整对象复制/序列化风险 | V313 |
| B207 | 2026-05-27 | GUI 3D cell-center cache miss fallback 用 `coords[cells,:3].mean(axis=1)` 展开完整 cell×vertex×dim 块；大体素 point-cloud 首显会多持有中心计算临时 | V314 |
| B208 | 2026-05-27 | `_apply_volume_fraction_streaming` 虽然按 chunk 处理 connectivity，但每个 chunk 仍先构造完整 `vertices` 和 `(chunk,n_samples,3)` sample tensor；三维 inhomogeneity painting 峰值内存仍随 chunk×sample 数放大 | V315 |
| B209 | 2026-05-27 | persistent backend warm 只报告 RSS 而不按预算回收，也没有把已 prime 的 worker 状态带进 forward/reconstruction 结果；三维首载诊断难以区分 worker/import 预热、JIT cache 与真正求解耗时，warm 后也可能静默保留超预算进程 | V316 |
| B210 | 2026-05-27 | GUI 默认 3D worker-only prewarm 在后台启动/prime 后只写 debug log；用户看不到当前是在预热 worker、已经热好、还是 warm 失败，首个三维点击仍像无状态卡顿 | V317 |
| B211 | 2026-05-27 | ProcessCacheStore 在写入超预算或低优先级大对象时先插入再统一 eviction；一个马上会被淘汰的三维大对象可能先冲掉已有热缓存，再把自己也淘汰，造成 L1 抖动和短时内存峰值 | V318 |
| B212 | 2026-05-27 | 三维 forward 首次加载只显示总失败/完成状态；缺少 import、mesh/JIT setup、target solve、homogeneous solve、worker HDF5 transport 与 GUI visualization 更新的阶段计时，无法判断下一步该优化预编译、缓存、求解器还是渲染 | V319 |
| B213 | 2026-05-27 | 3D 默认 worker 预热只能导入模块/prime runtime；若用户想提前支付 mesh + CEM static setup/JIT 冷启动成本，只能启用 full solve prewarm，导致后台预热在“太轻”和“太重”之间缺少中间档 | V320 |
| B214 | 2026-05-27 | setup-prime 只能通过 GUI 环境变量触发，运维侧 `eit-cache warm` 仍只会 import-only warm；用户无法用可重复 JSON 报告确认某个三维 forward request 的 setup/JIT 预热耗时与缓存命中 | V321 |
| B215 | 2026-05-27 | GUI setup-prime warm 若只按 profile 去重，改三维网格/电极/协议后可能跳过应有 setup/JIT prime；若按完整 simulation-input 签名去重，改异常体/噪声又会重复 prime 静态 setup | V322 |
| B216 | 2026-05-27 | `prime_forward_setup_request` 直接准备 profile cache 并触发 DOLFINx/FFCx setup，但没拿 normal forward 使用的 profile lock；setup-prime 与 one-shot/in-process/CLI warm 并发时可能争同一 `libffcx_*.c`，放大 JIT timeout | V323 |
| B217 | 2026-05-27 | V319 有阶段计时、V321 有 `eit-cache warm --forward-request`，但缺少一条可复用命令直接生成 GUI-style 3D request 并对 setup-prime/full solve 输出同一 JSON；三维优化仍需手写临时 heredoc/脚本，证据难复现 | V324 |
| B218 | 2026-05-27 | 实测 3D setup-prime 第二次 mesh/JIT 已命中后，`configure_system≈4.3s` 仍被 PETSc CUDA capability probe 等 runtime 检测占住；该 probe 只有进程内 LRU，CLI/GUI worker 新进程会重复试建 PETSc CUDA Mat/Vec | V325 |
| B219 | 2026-05-27 | 三维 forward 进入 conductivity painting 时先 `cell_midpoints(fwd.mesh)` 遍历 topology，再为 GUI payload 第二次遍历 connectivity；同时 `configure_system` 只有聚合耗时，无法继续区分 pattern/runtime/system 构造瓶颈 | V326 |
| B220 | 2026-05-27 | 默认 GUI 3D worker prewarm 只 import 重模块，不触发 PETSc CUDA capability probe；后续 setup/solve 即使进程已热，仍可能把 probe 的秒级开销留在 `configure.runtime` 前台路径 | V327 |
| B221 | 2026-05-28 | V327 把 capability probe 移入 worker prewarm 后，诊断 CLI 仍只能单独测 setup/solve；无法用一条命令复现“GUI 已做 worker warm 后用户首次 3D 点击”的同 worker 前台耗时 | V328 |
| B222 | 2026-05-28 | GUI 已经记录 worker warm report，但用户只能看到 pid/RSS/prime 耗时；无法从状态栏或 report 直接判断 PETSc CUDA capability probe 是否在后台命中/完成 | V329 |
| B223 | 2026-05-28 | `eit-cache doctor/stats` 只能看到 backend profile 和 FFCx cache，不扫描 profile-local `pyeidors-capabilities`；运维侧无法确认 PETSc CUDA capability probe cache 是否已由 worker warm 落盘 | V330 |
| B224 | 2026-05-28 | 三维 warm/benchmark 进度回调直接 append 到列表；长 JIT/mesh/worker 输出会把诊断 JSON 和父进程内存随日志行数线性放大 | V331 |
| B225 | 2026-05-28 | WSLg/headless 下 embedded VTK 禁用后，PyVista offscreen 若导入/建 plotter/首帧截图失败，后续三维刷新仍会重复慢失败探测；3D entrypoint/point-cloud actor 中还残留重复数组处理语句 | V332 |
| B226 | 2026-05-28 | backend worker HDF5 IPC 虽已从 gzip 改成 lzf，但未启用 shuffle；大三维 float32/int32 坐标、连通性、电导率和电压 payload 的重复字节模式不能被轻量压缩充分利用 | V333 |
| B227 | 2026-05-28 | backend worker result IPC 对缺失的 homogeneous/measured/simulated 仍写空 HDF5 dataset 并在读取时无条件打开；三维 forward/recon 错误或最小结果会多创建/读取无效对象 | V334 |
| B228 | 2026-05-28 | backend worker HDF5 IPC 主数组依赖 HDF5 自动 chunk；大三维坐标/连通性/电导率 payload 的 chunk 形状不可控，后续局部读取/懒加载没有稳定行边界 | V335 |
| B229 | 2026-05-28 | backend worker HDF5 IPC 读取路径对数组使用 `dataset[()]` 再 `np.asarray`；大三维结果读回语义依赖 h5py 默认分配，缺少显式最终 buffer 边界 | V336 |
| B230 | 2026-05-28 | GUI `channel_values()` 对所有显示通道强制 `dtype=np.float64`；三维 `float32/complex64` 电导率结果进入真值/重构视图时仅为可视化多持有一份双精度数组 | V337 |
| B231 | 2026-05-28 | GUI `has_complex_component()` 用 `imag[np.isfinite(imag)]` 判断复数模式；大三维 `complex64` 电导率仅为启用通道控件就复制整块有限虚部 | V338 |
| B232 | 2026-05-28 | GUI composite 通道先生成整块 `np.abs(arr)`、`np.angle(arr)` 再相乘；大三维 `complex64` 结果切到复合幅相视图时会同时持有 magnitude、phase 和输出三块数组 | V339 |
| B233 | 2026-05-28 | 3D anomaly mask 先 `residual = values - median`，再为 negative/absolute 模式分配整块 `-residual` 或 `np.abs(residual)` score；点云采样/异常高亮会多持有一份 cell 数组 | V340 |
| B234 | 2026-05-28 | 3D point-cloud 采样在异常点数超过显示上限时仍先 `np.flatnonzero(anomaly_mask)` 生成完整异常索引，再均匀截取；大面积异常体会多持有整块 int64 index 数组 | V341 |
| B235 | 2026-05-28 | 3D point-cloud 高亮在 PyVista/Matplotlib 路径对同一个 `inhom_mask` 多次布尔索引；Matplotlib 还按 x/y/z/sigma 重复扫描 mask 并创建临时列数组 | V342 |
| B236 | 2026-05-28 | 3D 色阶和 anomaly mask 初始化对常见全 finite 电导率数组先创建整块 `np.isfinite(values)` 布尔数组；大三维显示每次刷新会为只需判定 all-finite 的路径多占一份 cell 级 bool 内存 | V343 |
| B237 | 2026-05-28 | 3D anomaly mask 先构造候选 bool mask 统计峰值，再新建最终 threshold mask；非 finite 路径还会在已有 finite mask 后再次 `np.isfinite(score)`，多占/多扫一份 cell 级 bool 数据 | V344 |
| B238 | 2026-05-28 | 将 `_score_count_peak_above_floor` 直接改成三返回值破坏了现有直接调用单测/兼容契约；mask 复用应由 `_cell_anomaly_mask` opt-in，而默认 helper 仍返回 `(count, peak)` | V344 |
| B239 | 2026-05-28 | PyVista 体网格和电极面片构建时对刚创建的 C-order buffer 调用 `.flatten()`；大三维 volume grid 会额外复制一份 int64 VTK connectivity/face buffer | V345 |
| B240 | 2026-05-28 | 共享 `cell_to_node_average()` 在可视化 cell→node 映射后用 `node_values[np.isfinite(node_values)]` 复制有限子集求均值，再用 `np.where` 复制整块数组填 orphan/NaN 节点；大显示网格会多持有整块节点数组/有限子集 | V346 |
| B241 | 2026-05-28 | 硬件等势面 PyVista 渲染仍用 `faces.flatten()` 复制 face buffer，warp scale 又用 `node_values[np.isfinite(node_values)]` 复制有限节点值；大重构网格会多持有 face/finite 子集临时数组 | V347 |
| B242 | 2026-05-28 | 边界电压曲线 y-range 对每条序列使用 `arr[np.isfinite(arr)]` 复制有限值子集再求 min/max；长测量序列或复合曲线刷新会为坐标轴范围多持有一份有限值数组 | V348 |
| B243 | 2026-05-28 | PyVista 体高亮提取异常 cell 时使用 `np.where(inhom_mask)[0]`；相比直接 `np.flatnonzero` 多构造 tuple/索引包装，且隐藏了“只要一维索引”的意图 | V349 |
| B244 | 2026-05-28 | GREIT rec center-cloud 几何轴间距估计先对已排序的 `np.unique` 输出再次 `np.sort`，再复制 `diffs[np.isfinite(diffs)&(diffs>0)]`；finite centers 已保证 diff 有限且 unique diff 为正 | V350 |
| B245 | 2026-05-28 | 3D spatial anomaly 连通筛选对 KDTree 最近邻距离使用 `nearest[np.isfinite(nearest)&(nearest>0)]` 复制有限正距离子集；候选异常点多时会多持有一份 nearest-distance 数组 | V351 |
| B246 | 2026-05-28 | 边界电压 y-range 即使输入已经是 `float32` 显示曲线，也对每条序列 `np.asarray(..., dtype=np.float64)`；仅为坐标轴缩放多复制/加宽整条曲线 | V352 |
| B247 | 2026-05-28 | 2D 电导率真值/重构图像把 `float32/complex64` conductivity 通过 `np.asarray(..., dtype=float)` 升宽到 float64，3D 投影分支又把 `z` 坐标 `dtype=float`；仅为显示多复制/加宽结果与坐标数组 | V353 |
| B248 | 2026-05-28 | 边界电压 reconstructed overlay 已由通道投影得到 real float32 后，`_set_reconstructed_overlay()` 又 `np.asarray(..., dtype=np.float64)`；仅为 pyqtgraph 曲线输入多复制/升宽整条 overlay | V354 |
| B249 | 2026-05-28 | Matplotlib 3D surface fallback 对 cell-centered `float32` sigma 执行 `sigma.astype(float)`，随后用 `dtype=float` 构造 face values；WSLg/PyVista 不可用时三维 surface 显示会多持有一份 float64 cell/face 数据 | V355 |
| B250 | 2026-05-28 | 硬件等势面 `update_reconstruction()` 入口对 `node_coords` 和 `conductivity` 无条件 `dtype=np.float64`；硬件/重构 float32 payload 仅为进入 widget 就多复制/升宽 coords、sigma、node_values | V356 |
| B251 | 2026-05-28 | 硬件 2D 重构图像 `update_reconstruction()` 入口强制 `node_coords/conductivity` 为 float64，`_to_node_values()` 又用 float64 累加器；float32 重构结果显示前被复制/升宽两次 | V357 |
| B252 | 2026-05-28 | simulation metrics 先构造 `finite = isfinite(gt)&isfinite(rc)`，再复制 `gt[finite]`/`rc[finite]` 后用 `np.linalg.norm/np.corrcoef/mean`；大三维指标计算会多持有两份有限样本子集和后续差分临时 | V358 |
| B253 | 2026-05-28 | metrics 最近邻重采样对常见全 finite source/target 也先构造 `source_finite`/`target_finite` 全量 bool，再复制 `source_pos[source_finite]`、`source_values[source_finite]`、`target_pos[target_finite]`；不同网格大三维指标会多持有多份几何/值子集 | V359 |
| B254 | 2026-05-28 | 批量重构 `_save_outputs()` 为离线 PNG/电压拟合图把 `conductivity/node_coords/measured/simulated` 全部 `dtype=float`、connectivity `dtype=int`；float32/int32 结果仅为报告输出多复制/升宽整块数组 | V360 |
| B255 | 2026-05-28 | 主窗口单次保存重构图/电压拟合图路径与 batch 输出一样对 `conductivity/node_coords/measured/simulated` 用 `dtype=float`、connectivity `dtype=int`；float32/int32 结果仅为手动导出 PNG 多复制/升宽 | V361 |
| B256 | 2026-05-28 | 主窗口自动/手动硬件电压曲线和录制导出 payload 仍对 `measured/simulated/frame` 数组用 `dtype=float`/`np.float64`；float32 电压结果仅为显示或 interop snapshot 多复制/升宽 | V362 |
| B257 | 2026-05-28 | 共享 mesh 显示平均和 3D 显示 helper 对整数/非浮点电导率 fallback 使用 `np.float64`；分段/标签式大三维显示会仅为渲染多占一倍值数组内存 | V363 |
| B258 | 2026-05-28 | 硬件等势面 PyVista 相机对齐/重置函数把 `float32` 坐标轴切片 `dtype=float` 转成 float64；每次 3D 视角对齐仅为中心/直径计算多复制坐标列 | V364 |
| B259 | 2026-05-28 | 3D conductivity viewer 的 `_cell_centers()` 在 array-geometry cache miss/失效 fallback 中把 `float32` coords `dtype=float` 转成 float64；首次点云/Matplotlib fallback 仅为派生 centers 多复制/升宽整块坐标 | V365 |
| B260 | 2026-05-28 | 3D spatial anomaly 连通筛选在已抽取候选后仍用 `np.asarray(score[candidate_idx], dtype=np.float64)`；float32 score 仅为 component mass 排序多复制/升宽候选向量 | V366 |
| B261 | 2026-05-28 | 3D anomaly mask 在 positive/negative 模式计算 robust MAD 时调用 `np.nanmedian(np.abs(score))`；每次异常高亮/点云采样仅为阈值估计多分配一整块 abs-score 临时数组 | V367 |
| B262 | 2026-05-28 | 3D electrode overlay `default_patch_quads()` 已直填 point buffer，但 triangle indices 仍先 append Python tuple 列表再 `np.asarray`；首帧 overlay 几何构建多保留 tuple list 和转换路径 | V368 |
| B263 | 2026-05-28 | Matplotlib 3D fallback 在 `cmap(norm(...))` 生成 facecolor 数组后，又 `.copy()` 一份给 opacity slider 缓存；大边界面/高亮面首帧多保留一份 RGBA facecolor payload | V369 |
| B264 | 2026-05-28 | Matplotlib 3D 电极 fallback collection 用 `float64` lower/upper 坐标并为每个 quad 调 `np.array(..., dtype=float)`；首帧 fallback overlay 多走升精度和逐片 array 构造路径 | V370 |
| B265 | 2026-05-28 | Matplotlib 3D surface fallback 的 point-data face value 分支为每个边界面构造 `np.asarray(face)` 和 `point_sigma[...]` 子数组再 `np.nanmean`；大边界面首帧重复分配小数组 | V371 |
| B266 | 2026-05-28 | Matplotlib 3D surface fallback 为每个 boundary/highlight face 用 `np.asarray(face)` 构造整数索引数组再 gather `coords[..., :3]`；大边界面首帧重复分配小索引数组 | V372 |
| B267 | 2026-05-28 | 3D spatial anomaly component mass 排名先构造 `candidate_scores = score_values[candidate_idx]`，再对每个 component 建 `candidate_scores[component]` 子数组后 `np.nansum`；候选点多/碎片多时重复分配分数子数组 | V373 |
| B268 | 2026-05-28 | 3D volume-fraction streaming 已避免全量 cell/sample tensor，但入口仍 `np.asarray(node_coords, dtype=np.float64)`；float32 CUDA/generated mesh 在异常体绘制阶段仅为采样坐标多持有一份双精度节点坐标和 sample buffers | V374 |
| B269 | 2026-05-28 | Dataset generator setup 后仍用 `cell_midpoints(fwd.mesh)` 求 centers，再用 `[cells_conn.links(i) for i in range(n_cells)]` 二次遍历 topology 构造 connectivity；3D 数据集生成启动重复几何遍历和 Python 列表中间对象 | V375 |
| B270 | 2026-05-28 | simulation metrics 最近邻重采样的 all-finite 常见路径先构造完整 `mapped_values = valid_source_values[idx]`，再复制到最终 `mapped`；不同网格大三维指标计算多持有一份映射值向量 | V376 |
| B271 | 2026-05-28 | simulation metrics KDTree 不可用时的 brute-force fallback 无条件 `np.asarray(..., dtype=np.float64)` 并分配 float64 `distances/work`；float32 三维几何仅为 fallback 最近邻搜索被升宽一倍 | V377 |
| B272 | 2026-05-28 | ConductivityImageWidget 等比例坐标轴范围计算先建全量 `finite` mask，再复制 `x[finite]` / `y[finite]`；大 2D/3D 投影图刷新仅为 min/max 多持有两份坐标子集 | V378 |
| B273 | 2026-05-28 | 3D spatial anomaly 连通筛选在提取候选中心后强制 `np.ascontiguousarray(..., dtype=np.float64)`，并对 KDTree nearest 视图再 `np.asarray(..., dtype=np.float64)`；float32 点云高亮仅为 coherent-blob 过滤多保留升宽坐标/距离副本 | V379 |
| B274 | 2026-05-28 | 3D volume-fraction legacy fallback 中 `_paint_shape(cell_vertices=...)` 先 `dtype=float`，`_cell_volume_sample_points()` 再 `dtype=np.float64`；float32 顶点张量仅为回退采样路径被复制/升宽 | V380 |
| B275 | 2026-05-28 | 3D volume-fraction streaming 每 chunk 用 float64 `inside_counts`，legacy `_apply_volume_fraction()` 的 bool mean 也默认 float64；float32 采样点仅为体积分数比例缓冲多占双精度内存 | V381 |
| B276 | 2026-05-28 | ConductivityImageWidget tetra 投影去内部面后先构造 `kept` payload list，再用两个 list comprehension + `np.asarray` 生成 triangles/sources；大 tetra 边界投影多走两份中间列表 | V382 |
| B277 | 2026-05-28 | `extract_boundary_triangles()` tetra path 去内部面后先构造 kept payload list，再用 list comprehension + `np.asarray` 生成 triangles/sources；共享 3D 边界渲染多走中间列表 | V383 |
| B278 | 2026-05-28 | Matplotlib 3D `_boundary_faces()` 去内部面后先构造 `kept` payload list，再用 list comprehension + `np.asarray` 生成 faces/source_cells；fallback surface 渲染多走中间列表 | V384 |
| B279 | 2026-05-28 | Matplotlib 3D `_render_matplotlib_scene()` 对 `_boundary_faces()` 输出再构造全量 `valid_face_payload` 并用 `np.fromiter` 复制 source indices；正常合法网格 fallback surface 渲染多持有一份 face/source payload | V385 |
| B280 | 2026-05-28 | Matplotlib 3D `_render_matplotlib_scene()` 通过 list comprehension/append 对每个 boundary/highlight face 单独调用 `_face_vertices()`，为大 surface 生成大量小顶点数组 | V386 |
| B281 | 2026-05-28 | Matplotlib 3D anomaly highlight 先累积 `highlight_faces` / `highlight_values` 列表，再把 values 转数组着色；大异常区域 fallback highlight 多持有 face/value Python 列表 | V387 |
| B282 | 2026-05-28 | Matplotlib 3D anomaly highlight 先 `np.flatnonzero(mask)` 构造整块异常 cell index 数组，再遍历生成 highlight；大异常区域 fallback 多持有一份 int64 索引数组 | V388 |
| B283 | 2026-05-28 | Point-cloud highlight helper 先 `np.flatnonzero(inhom_mask)` 构造高亮 index，再对 centers/sigma 各 `np.take` 一次；大高亮点云多持有一份 int64 索引数组 | V389 |
| B284 | 2026-05-28 | spatial anomaly coherent filter 先 `np.flatnonzero(mask)` 再 `centers[candidate_idx, :3]` 高级索引复制候选中心；大候选区域多走索引构造和高级索引 gather 路径 | V390 |
| B285 | 2026-05-28 | spatial anomaly coherent filter 最后用 `coherent[candidate_idx[keep]] = True`，为保留候选再构造一份 kept global-index 子数组 | V391 |
| B286 | 2026-05-28 | `_sample_background_indices()` 在所有背景点都能保留时仍用 `np.flatnonzero(~anomaly_mask)`，会先构造整块反相 bool mask 再生成背景索引 | V392 |
| B287 | 2026-05-28 | `_sample_true_indices()` 和 `_point_cloud_sample_indices()` 在 true/anomaly indices 已能完整保留时仍用 `np.flatnonzero(...)`；小于显示预算但仍可能较大的 anomaly set 多走整块索引构造路径 | V393 |
| B288 | 2026-05-28 | mesh-derived `_cell_measures()` 为 2D/3D cells 先构造面积/体积 Python list 再 `np.asarray`；大三维派生缓存冷构建时多持有一份 cell-count 级 Python 临时列表 | V394 |
| B289 | 2026-05-28 | mesh-derived tetra `cell_measures` 虽已直填最终向量，但每个 4-vertex cell 仍走 `_polyhedron_volume(coords[cell])`，重复分配小型 gathered point array 并转 float64 | V395 |
| B290 | 2026-05-28 | mesh-derived 规则 8-vertex hexa `cell_measures` 也走 `_polyhedron_volume(coords[cell])`，轴对齐体素冷构建为每个 cell 触发 gathered point array 与通用 ConvexHull 路径 | V396 |
| B291 | 2026-05-28 | mesh-derived artifact 冷构建先提取 cells/coords 推导 centers/measures，随后 `mesh_derived_signature_payload()` 和 `mesh_derived_signature()` 又重复提取/遍历 topology 来生成 metadata；大 DOLFINx mesh 冷缓存多走数次 connectivity links | V397 |
| B292 | 2026-05-28 | FEMx `mesh_cell_vertices()` / `mesh_facet_vertices()` 仍用 `[connectivity.links(...)]` 构造完整 Python links 列表再转 `np.array`；共享三维几何 fallback 调用会多保留一份 links list | V398 |
| B293 | 2026-05-28 | dual-mesh `_locate_points_in_cell_mesh()` 每个 fine point 用 `(point >= mins) & (point <= maxs)` + `np.all(axis=1)` 生成 cell×dim bool 临时矩阵；coarse cell 多时映射循环反复分配候选筛选缓冲 | V399 |
| B294 | 2026-05-28 | dual-mesh locator 已复用 bbox candidate mask 后仍 `np.flatnonzero(candidate_mask)` 为每个 fine point 构造候选 cell index 数组；候选较多时又多一份 int64 临时 | V400 |
| B295 | 2026-05-28 | dual-mesh locator 候选筛选已复用 mask 后，simplex 检查仍为每个 candidate 执行 `coords[cells[cell_idx]]`，反复分配小型顶点数组 | V401 |
| B296 | 2026-05-28 | GREIT finite-target conductivity target loop 每次用 `dist2 <= radius2` 分配 bool mask，随后 `sigma[mask] = sigma[mask] + contrast` 又构造 masked conductivity 子数组；三维冷训练目标多时重复分配 | V402 |
| B297 | 2026-05-28 | Sparse MAP linear warm-start 对奇异值 mask 使用 `numerator[mask] / s_k[mask]`、`coeff[mask] /= s[mask]` 和 `coeff[~mask]`；高维 warm-start 会多构造 masked 子数组/反相 mask | V403 |
| B298 | 2026-05-28 | simulation metrics `_nearest_resample()` 非全有限 target 路径先 `mapped_values = np.take(...)` 再 `mapped[target_mask] = mapped_values`；大 3D 指标网格含少量坏点时多持有一份有效 target 临时向量 | V404 |
| B299 | 2026-05-28 | GN `ensure_measurement_weights()` 用 `np.where(np.isfinite(weights), weights, 0.0)` 和无 `out` 的 `np.maximum` 生成整块替换数组，verbose 又 `weights[np.isfinite(weights)]` 复制有限子集；三维测量通道多时权重准备多次全量分配 | V405 |
| B300 | 2026-05-28 | GN difference-mode weighting 用 `np.where(diff > floor, diff, floor)` 为刚生成的 diff 再分配整块替换数组；测量通道多时多一次全量拷贝 | V406 |
| B301 | 2026-05-28 | `_sanitize_preconditioner_diag()` 用 `np.where(np.isfinite(arr), ...)` 和无 `out` 的 `np.maximum` 生成替换数组；matrix-free PC diag 大时多分配且原地优化需避免改调用者输入 | V407 |
| B302 | 2026-05-28 | GN line-search best-step path 用 `np.where(np.isfinite(mlist))[0]` 再 `mlist[valid_idx]` / `mlist[goodi]` 挑最佳和更新扰动；trial 多或重试频繁时多分配 finite index/objective subset 临时数组 | V408 |
| B303 | 2026-05-28 | Dynamic TV/Huber robust helpers 用 `np.where(...)` 和 `arr * arr` 构造权重/penalty；帧数×参数数矩阵大时多持有 replacement/square 临时数组 | V409 |
| B304 | 2026-05-28 | Dynamic temporal ROI 默认全参数也通过 `weights[:, roi_mask]` / `temporal_diffs[:, roi_mask]` 布尔列索引复制整块子矩阵；长序列三维参数多时多持有 frame×param 临时数组 | V410 |
| B305 | 2026-05-28 | TV regularization `create_matrix()` 用 `np.square(grad_ref)` / `np.sqrt(...)` 临时和 `weights = weights / median_weight` 替换数组准备权重；三维参数多时正则矩阵冷构建多分配 | V411 |
| B306 | 2026-05-28 | GN `_finite_summary()` 用 `values[np.isfinite(values)]` 复制有限子集再 `np.linalg.norm(finite)`；大残差/步长向量诊断或异常路径会多持有整块 finite 临时数组 | V412 |
| B307 | 2026-05-28 | shared numeric `_finite_summary()` 用 `values[np.isfinite(values)]` 和 `np.abs(finite)` 格式化 `safe_dot` 诊断；大矩阵/向量非有限报错路径多复制有限值子集 | V413 |
| B308 | 2026-05-28 | electrode pattern `_finite_summary()` 用 `values[np.isfinite(values)]` 和 `np.abs(finite)` 格式化投影诊断；大测量矩阵/复数通道报错路径多复制有限值子集 | V414 |
| B309 | 2026-05-28 | GN regularization validation 用 `matrix.data[np.isfinite(matrix.data)]` / `dense[np.isfinite(dense)]` 为非有限错误格式化 min/max；大三维正则矩阵报错路径会复制有限矩阵 payload | V415 |
| B310 | 2026-05-28 | 3D point-cloud `_sample_true_indices()` 下采样分支每个 chunk 调 `np.flatnonzero(chunk)` 构造局部 true-index 数组；大异常区域会反复分配 chunk 级索引临时数组 | V416 |
| B311 | 2026-05-28 | PETSc CEM 电极矩阵装配每个电极用 `np.flatnonzero(c_i)` 后再 `c_i[nz]` 复制耦合值；三维 DOF 多时按电极反复生成索引和值临时数组 | V417 |
| B312 | 2026-05-28 | GREIT blob target generation 用 `target[~mask] = 0.0` 构造反相 mask/slice 来清零半径外值；三维 target×parameter 多时多分配 bool 临时路径 | V418 |
| B313 | 2026-05-28 | measurement-channel contract 用 `out[mask]` / `masked[mask, :]` / `masked[:, mask]` 清零坏通道；大 Jacobian/完整权重矩阵会走布尔高级赋值索引路径 | V419 |
| B314 | 2026-05-28 | RM frame-batch online contract 用 `out[:, mask] = 0.0` 清零坏测量列；长序列/三维帧批量会走布尔高级列索引路径 | V420 |
| B315 | 2026-05-28 | Temporal RM online path 先 `prepare_measurement_contract()` 只为 metadata，再调用帧 contract 应用；对角/identity 权重会额外构造密集对角 contract，长三维序列冷/热应用多分配 | V421 |
| B316 | 2026-05-28 | Dynamic temporal robust normal 用 `weights[:, ~roi_mask] = 0.0` 构造反相 ROI mask 并布尔列清零；长序列×三维参数权重矩阵多走高级索引临时路径 | V422 |
| B317 | 2026-05-28 | Measurement contract preparation 对一维/identity 权重用 `np.diag(weights)` 和 `np.diag(np.sqrt(weights))` 物化 O(n²) 对角矩阵；三维大测量数离线 RM/Jacobian 加权内存峰值被放大 | V423 |
| B318 | 2026-05-28 | Full measurement sqrt transform 用 `np.diag(np.sqrt(clipped)) @ eigenvectors.T` 先建 dense 对角矩阵；完整噪声权重路径多持有一个 O(n²) 临时矩阵 | V424 |
| B319 | 2026-05-28 | RM frame-batch diagonal contract 用 `np.sqrt(weights).reshape(...)` 额外分配 sqrt 权重向量；长时序/三维批量在线应用每次多持有 O(n_measurements) 临时数组 | V425 |
| B320 | 2026-05-28 | Matplotlib 3D cell-scalar surface 用 `face_values = np.take(cell_sigma, source_indices)` 生成 gather 返回数组；大三维边界面刷新时无法复用/控制最终输出分配 | V426 |
| B321 | 2026-05-28 | 3D point-cloud display arrays 用 `center_values[sample]` / `sigma_values[sample]` 高级索引生成采样数组；大点云刷新时输出分配不受控且非浮点中心需二次 cast | V427 |
| B322 | 2026-05-28 | 3D point-cloud sampling 合并 anomaly/background 后 `return np.sort(sampled)` 复制采样索引数组；大点云每次刷新多分配一份 index payload | V428 |
| B323 | 2026-05-28 | 3D finite scan helper 每个 chunk 用 `np.isfinite(chunk).all()`，发现非有限后又用 `np.isfinite(chunk/tail)` 生成 chunk 级 bool 临时；大三维非有限路径重复分配扫描 buffer | V429 |
| B324 | 2026-05-28 | 3D spatial anomaly 最近邻距离已复用 `nearest_valid` mask 后仍用 `nearest[nearest_valid] = np.nan` 布尔左值赋值；候选异常多时走高级索引写入路径 | V430 |
| B325 | 2026-05-28 | mesh-derived `_cell_measures()` fallback 仍用 `coords[cell]` 构造每格顶点数组；非规则 hexa / generic 3D 冷构建与 2D polygon measure 反复分配小型 gather payload | V431 |
| B326 | 2026-05-28 | graph-prior `cell_volumes()` 每个 simplex cell 用 `coords[cell]`、`vertices[1:] - vertices[0]`、`basis.T @ basis` 重建小数组；三维 graph Laplace / LTL prior volume weighting 冷构建重复分配 | V432 |
| B327 | 2026-05-28 | 3D anomaly crowded percentile 分支仍用 `score[finite_values]` 复制有限 score 子集后求 percentile；大面积异常且含 NaN payload 时多持有一份有限分数数组 | V433 |
| B328 | 2026-05-28 | simulation metrics 最近邻重采样非全有限路径仍用 `source_pos[source_mask]`、`source_values[source_mask]`、`target_pos[target_mask]` 布尔索引构造紧凑 source/query；大三维几何含少量坏点时多走高级索引复制 | V434 |
| B329 | 2026-05-28 | metrics `_finite_row_mask_or_none()` 每个 chunk 用 `np.isfinite(pos[start:stop]).all(axis=1)` 生成二维 bool 临时再行归约；大三维 source/target geometry 扫描有限性时多分配 chunk×dim mask | V435 |
| B330 | 2026-05-28 | metrics `_finite_pair_stats()` 每个 chunk 用 `np.isfinite(gt_chunk) & np.isfinite(rc_chunk)` 生成临时有限性 mask；大三维指标统计多次 chunk 扫描会重复分配 bool 数组 | V436 |
| B331 | 2026-05-28 | `_paint_shape()` centroid fallback 用 `values[mask] = conductivity` / `values[dist2 < ...] = conductivity` 写入异常体；未走体积分数路径的 2D/3D 绘制仍使用布尔左值高级索引 | V437 |
| B332 | 2026-05-28 | `cell_to_node_average()` 已避免 finite subset / `np.where` 后，orphan/NaN 填补仍用 `node_values[~touched]` 与 `node_values[nan_mask]` 布尔左值赋值；大显示网格节点补值仍走高级索引写入路径 | V438 |
| B333 | 2026-05-28 | GN `calc_perturb_limits()` 对 `au_pos/au_neg/al` 清理 sign 与非有限项时用 `au_pos[...] = np.inf`、`au_neg[...] = np.inf`、`al[...] = 0` 布尔左值写入；大参数向量线搜索扰动限制准备会反复走高级索引写入路径 | V439 |
| B334 | 2026-05-28 | `greit_metrics()` 用 `weights[qmi]`、`signed_image[qmi] * weights[qmi]`、`weights[qmi & ~equivalent_ball]`、`signed_image[opposite] * weights[opposite]` 计算 RES/SD/RNG；大三维 voxel/cell 指标评估会为掩码区域复制多份子数组 | V440 |
| B335 | 2026-05-28 | 硬件 `ReconstructionWidget._interpolate_to_rgba()` 每帧用 `self._grid_vertices[self._grid_valid_mask]`、`self._grid_weights[self._grid_valid_mask]`、`interpolated[self._grid_valid_mask]` 和 `rgba[~mask]`；大图像网格刷新会重复复制有效采样行与有限值子集 | V441 |
| B336 | 2026-05-28 | 硬件 `ReconstructionWidget._prepare_grid_cache()` 构建插值缓存时用 `simplex[valid_mask]`、`sample_points[valid_mask]`、`vertices[valid_mask]`、`weights[valid_mask]`；首次网格缓存构建会复制有效采样行并走布尔左值写入 | V442 |
| B337 | 2026-05-28 | `greit_metrics(target_values=None)` 仍通过 `_as_target_values()` 把 bool `target_mask` 转成整块 float64 target，再用于积分和目标质心；默认三维指标评估会多持有一份 cell-count target 数组 | V443 |
| B338 | 2026-05-28 | `greit_metrics()` 无论 target integral 正负都用 `signed_image = signal_sign * image` 复制整幅 image；默认正目标路径随后还要构造 centroid 权重缓冲，峰值多一份 image payload | V444 |
| B339 | 2026-05-28 | `VoxelGrid.locate_points()` 用 `np.all((scaled >= 0) & (scaled < upper), axis=1)` 构造 n_points×dim bool 临时，并用 `scaled[inside]`、`pts[~inside]`、`indices[~inside]` 复制内外点行；三维 dual-mesh 映射点多时峰值和拷贝放大 | V445 |
| B340 | 2026-05-28 | `VoxelGrid.locate_points()` 用 `np.floor((pts - self.origin) / self.spacing).astype(np.int64)` 计算 scaled indices；大点集映射时减法/除法/floor 链路产生多份全量浮点临时数组 | V446 |
| B341 | 2026-05-28 | GUI 3D `_conductivity_color_limits()` 在 values 含 NaN/Inf 时用 `np.median(values[finite_mask])` 复制有限值子集；大三维 cell_sigma 色阶统计含少量坏值时多持有一份近全量数组 | V447 |
| B342 | 2026-05-28 | `build_greit3d_distribution()` 已直填候选中心后仍用 `candidate_centers[inside_mask]` 复制内部 target centers；GREIT 3D 训练分布冷构建会额外走一次 bool 行高级索引 | V448 |
| B343 | 2026-05-28 | TV regularization `create_matrix()` all-finite 快路径已优化，但非有限权重 fallback 仍用 `finite_weights = weights[np.isfinite(weights)]` 复制有限权重子集求 median；大三维正则矩阵边界分支会多持有一份权重数组 | V449 |
| B344 | 2026-05-28 | 3D point-cloud `_point_cloud_sample_indices()` 已构造完整 anomaly indices 后，`_sample_background_indices()` 仍 `np.count_nonzero(mask_arr)` 重扫整块 mask 推导 background count；异常很少的大点云首次显示多一次 O(n) mask pass | V450 |
| B345 | 2026-05-28 | `VoxelGrid.locate_points()` outside 分支先 `np.any(outside_mask)` 判定，再由 `_compact_rows_where()` `np.count_nonzero(mask)` 计数分配；三维 coarse/fine outside-nearest fallback 对同一 mask 多扫一次 | V451 |
| B346 | 2026-05-28 | GREIT `_infer_center_spacing()` 对 `np.unique` 已排序坐标再次 `np.sort(coords)`，再 `np.diff` 并用 `diffs[diffs > eps]` 复制正间距子集求 median；三维 desired-image extent 冷推断多走排序/子集临时 | V452 |
| B347 | 2026-05-28 | `build_greit3d_distribution()` 先 `np.any(inside_mask)` 验证非空，再 `_compact_centers_by_mask()` 内 `np.count_nonzero(keep)` 计数分配；GREIT3D target-center 冷构建对同一 inside mask 多扫一次 | V453 |
| B348 | 2026-05-28 | GREIT `_build_finite_target_conductivities()` 每个 target 应用 contrast 后用 `np.any(sigma <= 0.0)` 检查正性；三维 finite-target 冷训练每行多分配一条 cell-count bool 向量 | V454 |
| B349 | 2026-05-28 | GREIT `_resolve_background_conductivity()` 用 `np.any(array <= 0.0)` 校验背景正性；三维 finite-target 背景向量只为正性检查多分配一条 cell-count bool 向量 | V455 |
| B350 | 2026-05-28 | GREIT `_as_cell_volumes()` 用 `np.any(volumes <= 0.0)` 校验 cell weights 正性；大三维 metrics 评估只为正性检查多分配一条 cell-count bool 向量 | V456 |
| B351 | 2026-05-28 | GREIT `_resolve_measurement_order()` 用 `(order < 0) \| (order >= n)` 构造两个 bool 向量和 OR 结果做范围检查；48e/5936 measurement-order 冷解析多分配测量数级临时 | V457 |
| B352 | 2026-05-28 | GREIT `vh` 归一化守卫用 `np.any(np.abs(vh) <= eps)`；finite-target ratio Y 与 EIDORS NF 路径只为零参考通道检查多分配 full abs 向量和 bool 向量 | V458 |
| B353 | 2026-05-28 | GREIT `_resolve_measurement_order()` 用 `np.unique(order)` 排序/复制 int order 检查 permutation，并为 provided identity 再构造 `np.arange` 比较；48e/5936 measurement-order 冷解析多走 int 排序/identity 临时 | V459 |
| B354 | 2026-05-28 | GREIT desired-image offset 解析用 `np.any(extents > eps, axis=0)` 构造 `n_cells x 3` bool 矩阵；三维 desired image 冷构建只为活跃轴检测多分配随 cell 数增长的临时数组 | V460 |
| B355 | 2026-05-28 | GREIT `_inside_mask_from_model_nodes()` Delaunay fallback 用 `(centers[:, :3] >= lower) & (centers[:, :3] <= upper)` + `np.all(axis=1)` 构造 `n_centers x 3` bool 矩阵；三维 distribution fallback 多一份候选中心级临时 | V461 |
| B356 | 2026-05-28 | Dynamic `_roi_mask()` integer index 校验用 `(indices < 0) \| (indices >= n_parameters)` 构造两个 bool 向量和 OR 结果；长三维 ROI 索引列表只为范围检查多分配索引数级临时 | V462 |
| B357 | 2026-05-28 | Dynamic `_restrict_difference_rows_to_roi()` 每个 CSR 行用 `roi_mask[cols]` 后 `np.all` 判断是否保留；长 ROI/高维 temporal difference 会按行重复分配小型索引子集 | V463 |
| B358 | 2026-05-28 | GREIT `_as_desired_cell_extents()` 用 `np.any(array < 0.0)` 校验非负；三维 desired extents 矩阵只为负值检查多分配 `n_cells x 3` bool 临时 | V464 |
| B359 | 2026-05-28 | GREIT `_as_xyz_points()` / `_model_nodes()` 用 `np.isfinite(points).all()` / `np.isfinite(nodes).all()` 校验大三维点云；模型节点和 target center 入口只为有限性检查多分配完整 bool 矩阵 | V465 |
| B360 | 2026-05-28 | GREIT finite-target measurement vector 和 training response `Y` 用 `np.isfinite(...).all()` 校验；目标数×测量数矩阵只为非有限检查多分配完整 bool payload | V466 |
| B361 | 2026-05-28 | Dynamic `_frame_batch()` / `_initial_dynamic_state()` 用 `np.isfinite(arr).all()` 校验 frame×parameter 矩阵；长三维 temporal window 只为有限性检查多分配完整 bool 矩阵 | V467 |
| B362 | 2026-05-28 | Dynamic RM/transition/initial-state/covariance 校验仍用 `np.isfinite(...).all()`；大 observation/state-space 矩阵只为有限性检查多分配完整 bool payload | V468 |
| B363 | 2026-05-28 | Dynamic `_temporal_weighted_normal()` 每个参数列用 `np.any(column_weights > 0.0)` 判断是否全零；长时序×参数列循环会反复分配列长 bool 临时 | V469 |
| B364 | 2026-05-28 | GREIT imported RM artifact 和 EIDORS NF helpers 多处用 `np.isfinite(...).all()` 校验 `Y/D/noise/PJt/vh/volume_weights`；大 artifact 载入或 NF 评估只为有限性检查多分配完整 bool payload | V470 |
| B365 | 2026-05-28 | GUI 3D `_spatially_coherent_anomaly_mask()` 用 `np.isfinite(candidate_centers).all()` 校验 KDTree 候选中心；大异常区域只为有限性检查多分配 `candidate_count x 3` bool 矩阵 | V471 |
| B366 | 2026-05-28 | GUI shared mesh `_all_finite()` 每个 chunk 用 `np.isfinite(chunk).all()` 分配 chunk 级 bool 临时；cell-to-node fallback 大数组扫描时重复分配 | V472 |
| B367 | 2026-05-28 | GUI PyVista volume highlight 对同一 `inhom_mask` 先 `np.any` 后 `np.flatnonzero`；有高亮 cell 时异常 mask 多走一次全量扫描 | V473 |
| B368 | 2026-05-28 | GUI `_simulation_reconstructed_voltage_fit()` 用 `np.isfinite(reconstructed).all()` 校验重构绝对电压；长测量向量只为有限性检查多分配完整 bool payload | V474 |
| B369 | 2026-05-28 | backend routing、forward setup key、CEM scalar coercion 和 Gauss-Newton scalar diagnostics 多处用 `np.abs(np.imag(...)) > tol` 判断复数输入；大接触阻抗/测量/参数数组只为虚部扫描多分配完整临时数组 | V475 |
| B370 | 2026-05-28 | forward `_coerce_scalar_array()` 和 CUDA structured `_build_sigma_state()` 用完整 `np.isfinite(...).all()` 校验大 conductivity/diagonal 数组；三维首轮装配后只为有限性检查额外分配 bool payload | V476 |
| B371 | 2026-05-28 | reconstruction controller 的 RM fit Jacobian、center-cloud 几何、streamed RM 输出和 simulated voltage fit 多处仍用完整 finite bool；contact impedance 复数判断仍分配完整 imag/abs/compare 临时 | V477 |
| B372 | 2026-05-28 | `cell_to_node_average()` 最终 NaN 填充阶段先分配完整 `nan_mask`，再分配完整 `finite_mask` 计算均值；三维大节点数组显示时额外占用两份 bool payload | V478 |
| B373 | 2026-05-28 | conductivity image `_finite_xy_bounds()` 每个 chunk 用 `np.isfinite(x_chunk) & np.isfinite(y_chunk)` 生成多个 bool 临时；大节点坐标缩放时重复分配 chunk mask | V479 |
| B374 | 2026-05-28 | reconstruction-matrix helpers 多处用完整 `np.isfinite(...).all()` 校验 dv/RM/frame/J/RtR/RM 输出；大 RM artifact 和批量帧路径只为有限性检查额外分配完整 bool payload | V480 |
| B375 | 2026-05-28 | sigma/contact block-system helpers 多处用完整 `np.isfinite(...).all()` 校验向量、CSR data、normal RHS、solver output 和 contact update；三维联合反演路径只为有限性检查多分配 bool payload | V481 |
| B376 | 2026-05-28 | matrix-free GN 和 dual-mesh helper 仍用完整 `np.isfinite(...).all()` 校验 vector、regularization data、weight diag 和 matrix action；三维 matrix-free 反演路径有多余 bool payload | V482 |
| B377 | 2026-05-28 | measurement-channel contract helpers 用完整 `np.isfinite(...).all()` 校验权重、measurement vector 和 Jacobian array；坏通道/RM 合同路径只为有限性检查多分配 bool payload | V483 |
| B378 | 2026-05-28 | temporal filtering/core helpers 用完整 `np.isfinite(...).all()` 校验 frame batch、hook output、state tail/last 和 timestamps；在线多帧测量过滤只为有限性检查多分配 bool payload | V484 |
| B379 | 2026-05-28 | RM matmul kernels 用完整 `np.isfinite(...).all()` 校验 RM matrix、batched delta_v 和输出 values；在线 RM CPU/GPU 路径只为有限性检查多分配 bool payload | V485 |
| B380 | 2026-05-28 | inverse temporal/TV postprocess helpers 用完整 `np.isfinite(...).all()` 校验 EMA initial、TV seed 和 TV difference vector；三维后处理只为有限性检查多分配 bool payload | V486 |
| B381 | 2026-05-28 | dynamic sequence 和 EIDORS noise 入口用完整 `np.isfinite(...).all()` / `np.any(values < 0.0)` 校验批量测量、dt、weights、frequency 和 noise signal；大 frame batch 只为验证多分配 bool payload | V487 |
| B382 | 2026-05-28 | RtR prior 与 TV-IRLS prior 多处用完整 `np.isfinite(...).all()` 校验 apply/diag/payload、graph gradient、weights、difference data、state、measurement、frames 和 initial；大三维先验/批处理只为有限性检查多分配 bool payload | V488 |
| B383 | 2026-05-28 | GN regularization readiness 对 RtRPrior probe/diag/dense、sparse data、LinearOperator probe 和 dense matrix 用完整 `np.isfinite(...).all()`；三维正则准备只为有限性检查多分配 bool payload | V489 |
| B384 | 2026-05-28 | GN linear-system helper 对 runtime arrays、native complex delta、matrix-free diag/probe、PMAT sparse/dense data 和 fused reduced delta 用完整 `np.isfinite(...).all()`；三维 fast/matrix-free solve 只为有限性检查多分配 bool payload | V490 |
| B385 | 2026-05-28 | reduced `SnapshotBank.add()` 用完整 `np.isfinite(vec).all()` 校验参数快照；大三维 reduced GN 快照入口只为有限性检查多分配 bool payload | V491 |
| B386 | 2026-05-28 | dual-mesh `CellMesh`/`VoxelGrid` validators 用完整 finite/negative/positive bool payload 校验坐标、cell index、origin 和 spacing；三维 dual/coarse mesh 入口有多余扫描分配 | V492 |
| B387 | 2026-05-28 | GUI reconstruction single-step sigma update 和 voxel-bounds parsing 用完整 `np.all(np.isfinite(...))` 校验 sigma/delta/raw estimate/bounds；大三维单步重构只为有限性检查多分配 bool payload | V493 |
| B388 | 2026-05-28 | dynamic inverse helpers 对 sparse difference data、timestamps、Jacobian stack、spatial prior data 和 block solve output 用完整 `np.isfinite(...).all()`；长时序/三维动态反演只为有限性检查多分配 bool payload | V494 |
| B389 | 2026-05-28 | GREIT desired-image、3D distribution、finite-target、RM、rec-model、metric 和 NF helpers 仍有多处完整 `np.isfinite(...).all()`；3D GREIT 冷构建/artifact/指标路径只为有限性检查多分配 bool payload | V495 |
| B390 | 2026-05-28 | ADC/digit/holdout data experiment validators 用完整 `np.all(np.isfinite(...))` 校验电压、向量、矩阵和 spline prediction；批量 sweep/report 路径只为有限性检查多分配 bool payload | V496 |
| B391 | 2026-05-28 | factor/voltage sweep、bucket domain truth 和 dense bucket reference/sensitivity builders 用完整 `np.all(np.isfinite(...))` 校验向量/矩阵；批量 sweep 与 dense bucket 实验只为有限性检查多分配 bool payload | V497 |
| B392 | 2026-05-28 | unit consistency、cached 3D CEM measure validation、TV smoothness weights 和 measurement projection 仍用完整 finite bool payload；三维缓存验证/投影/正则热路径有可避免分配 | V498 |
| B393 | 2026-05-28 | channels、block-system、matrix-free GN、electrode length、holdout area 和 circle-bucket radius guards 用 `np.any(arr <... )`/`np.nonzero` 生成比较 bool payload；大批量检查有可避免分配 | V499 |
| B394 | 2026-05-28 | GREIT extents/bounds/imgsz/axis/downsample/radius/steepness validators 用 `np.any(<comparison>)` 生成比较 bool payload；3D GREIT artifact 构建仍有可避免分配 | V500 |
| B395 | 2026-05-28 | geometry_exchange、graph_core 和 GUI forward complex detection 仍用比较/abs bool payload；interop 批量导入和 GUI 复数测量判定有可避免分配 | V501 |
| B396 | 2026-05-28 | electrode `_create_meas_hash()` 对同一 measurement matrix 重复执行 `meas_mat > 0` 和 `meas_mat < 0` 扫描；测量 selector/hash 构建有重复比较成本 | V502 |
| B397 | 2026-05-28 | normalized difference `_safe_reference()` / frame batch floor clamp 用 `safe[small]`、`tiny[nonzero]`、`signs[...]` 布尔子集复制；RM/GREIT 差分批量入口在近零 reference 时多分配子数组 | V503 |
| B398 | 2026-05-28 | measurement-form one-step RM 和 GREIT scalar/vector measurement regularisation/noise covariance 先物化 dense identity/diag，再通过 dense 加法形成系统矩阵；三维测量数大时额外持有一份 n_meas² 临时 | V504 |
| B399 | 2026-05-28 | GREIT `_nearest_unique_center_distance()` 用 `np.isfinite(nearest_values) & (nearest_values > 0.0)` 构造 mask 并复制 `positive_nearest` 子数组；3D target/center 冷半径推断有可避免距离子集分配 | V505 |
| B400 | 2026-05-28 | GN fast linear-system dense regularization metadata 用 `dense - np.diag(diag_vec)` 构造 offdiag 临时，Woodbury jitter 用 `jitter * np.eye(n_meas)` 再相加；三维 GN 快路径有额外 dense 临时 | V506 |
| B401 | 2026-05-28 | matrix-free GN `_sqrt_measurement_weights()` 用 `np.diag(np.diag(weights))` 判定 dense 权重矩阵是否对角，并再 `np.diag(weights)` 提取 diagonal；大测量数权重路径重复构造 dense 对角矩阵 | V507 |
| B402 | 2026-05-29 | Smoothness/Tikhonov/TV regularization empty-difference fallbacks 用 `np.eye(self.n_elements)` 构造 dense identity，再包装成 CSR 或直接返回；三维 n_elements 大时单位正则冷构建内存峰值放大 | V508 |
| B403 | 2026-05-29 | PyVista offscreen/embedded volume highlight 已有 `inhom_mask` 后仍无条件 `np.flatnonzero(inhom_mask)` 构造 int64 cell index；大三维高亮区域额外持有一份 cell-count index payload | V509 |
| B404 | 2026-05-29 | native complex GN normal-step solve 对默认 identity/一维 diagonal regularization 先物化 dense `reg`，再通过 `J_h @ J + lambda * reg` 和 `reg @ prior` 产生 dense 临时；复杂三维反演内存峰值可避免 | V510 |
| B405 | 2026-05-29 | matrix-free PMAT dense preconditioner setup 用 `dense + np.eye(n) * shift` 形成 shifted matrix；大 PMAT 路径在必要 dense copy 之外又构造一份 identity 临时 | V511 |
| B406 | 2026-05-29 | GUI reconstruction controller native-complex route 在没有 `R_matrix/R_diag` 时先返回 `np.eye(n_param)`，即使 solver 已支持 lazy identity；复杂三维 GUI 反演 dispatch 前多分配 dense identity | V512 |
| B407 | 2026-05-29 | sparse Bayesian IRLS/coarse/block correction paths 用 `np.diag(weights/group_sizes)` 或 `np.eye(stop-start)` 添加对角正则；高维 sparse MAP refine 额外产生 dense 临时 | V513 |
| B408 | 2026-05-29 | dynamic Kalman 默认 transition/noise/covariance 与 RM-observation H stack 反复用 `np.eye`/`scale*np.eye` 构造 dense identity，Joseph update 两次计算 `identity_state-kh`；多帧大状态滤波峰值内存偏高 | V514 |
| B409 | 2026-05-29 | TV nonlinear term 用 `self.alpha * np.diag(weights)` 先构造 dense diagonal 再生成 scaled dense copy；大参数正则辅助路径多一份 dense 临时 | V515 |
| B410 | 2026-05-29 | digit metric surrogate inverse 与 measurement-space RM ridge path 用 `normal + ridge*np.eye(...)` / `lhs + lambda²*np.eye(...)` 产生 identity 临时和第二份 dense 系统 | V516 |
| B411 | 2026-05-29 | measurement-channel dense compatibility conversion 与 dynamic vector covariance path 用 `np.diag(vector)` 构造 dense diagonal；大测量/状态兼容转换可直接填充最终矩阵 | V517 |
| B412 | 2026-05-29 | GN runtime 初始化电导率与 prior data 预处理用 `.flatten()` 强制复制输入；大三维初始场/先验数组即使可视图展平也会多一份 payload | V518 |
| B413 | 2026-05-29 | RtRPrior/GN regularization/POD dense diagonal extraction 用 `np.diag(dense)` 通用入口；已有 dense ndarray 可直接 `.diagonal()` 取视图再按需转 dtype | V519 |
| B414 | 2026-05-29 | GN linear-system native-complex 兼容 regularization 与 Woodbury small system 仍用 `np.eye`/`np.diag` 构造最终 dense identity/diagonal；可统一直接填充减少构造开销 | V520 |
| B415 | 2026-05-29 | GN difference runner single-step operator builder 用 `+ lam*np.eye(...)`、`+ lam*np.diag(reg_diag)` 和 `np.diag(system_matrix)` 生成对角临时；benchmark/cache warm 路径可复用 in-place 对角加法 | V521 |
| B416 | 2026-05-29 | 3D overview 诊断渲染用 3D `np.where` cylinder mask、`np.corrcoef`、`coords[mask]` 与 `recon_sigma[mask]` 子集；报告生成大网格时峰值内存偏高 | V522 |
| B417 | 2026-05-29 | common GN absolute/difference plotting runner 用 `np.corrcoef(measured,predicted)` 构造 2×N 临时；长测量向量报告/绘图可直接归约相关系数 | V523 |
| B418 | 2026-05-29 | gallery shared truth/consistency metrics 用 `np.corrcoef`、`truth[roi]`、`recon[roi/background_mask]` 和 `background_mask &= ~roi`；真实重建 gallery 大三维数组指标路径有多余子集复制 | V524 |
| B419 | 2026-05-29 | 多个诊断脚本 safe_corr/safe_pearson 先压缩有限样本 `a[mask]`/`b[mask]` 再 `np.corrcoef`；长边界电压/图像向量 parity 评估多持有 compact copy | V525 |
| B420 | 2026-05-29 | 8e/16e、scaled-boundary 和 gallery 诊断均值指标用 `np.mean(arr[mask])` 复制 ROI/背景子集；大三维诊断指标可直接 `where=` 归约 | V526 |
| B421 | 2026-05-29 | `benchmark_difference_runtime.build_measurement_weights()` 用 `np.where(np.isfinite(weights), weights, 0)` 清洗权重，额外生成同尺寸 dense 临时；可对私有权重向量原地清洗/取 floor | V527 |
| B422 | 2026-05-29 | holdout raw/fitted voltage RMSE 用 `full_diff[holdout_indices]`、`v_true[holdout_indices]` 和 `fit_true[holdout_indices]` 复制被留出通道子集；长序列 holdout 评估可 chunked take 归约 | V528 |
| B423 | 2026-05-29 | dynamic validation benchmark 用 list comprehension + `np.vstack` 生成 truth frames / measurement Jacobian，并用 `np.mean([record[key]...])` 聚合空间指标；大帧数/大网格验证可预分配逐行填充 | V529 |
| B424 | 2026-05-29 | real reconstruction gallery 2D/3D slice samplers 用 `np.column_stack([...])` 和 `np.full(x_grid.size, constant)` 构造 query 点；高分辨率切片插值会多持有列临时 | V530 |
| B425 | 2026-05-29 | tank realdata holdout compare 脚本用 `np.corrcoef`、`fit_residual[holdout_indices]`、`np.vstack` 和字段 `np.concatenate` 生成指标/输出；实测 holdout 报告在高分辨率逆网格下多持有临时矩阵 | V531 |
| B426 | 2026-05-29 | 8e/16e 与 scaled-boundary 诊断 `sample_to_grid()` 用 `np.column_stack([xg.ravel(), yg.ravel()])` 和 `(xg**2 + yg**2) > radius**2` 构造整网格临时；高分辨率对比图峰值内存偏高 | V532 |
| B427 | 2026-05-29 | synthetic parity 脚本残留 `np.corrcoef`、`np.where(np.isfinite(...))`、`hp²*np.eye`、`np.diag(noser_diag)` 和 forward CSV `np.column_stack`；单步差分/报告路径仍有可避免 dense 临时 | V533 |
| B428 | 2026-05-29 | benchmark difference runtime 的单步 measurement/parameter 空间仍用 `hp²*np.eye` 和 `np.diag(noser_diag)` 构造 dense 对角临时，权重清洗也保留完整 finite bool mask | V534 |
| B429 | 2026-05-29 | prior travelling-wave benchmark 用 truth/Jacobian 行列表再 `np.vstack`，peak-time 误差用 `peak_time_recon[peak_mask]`/`peak_time_truth[peak_mask]` 子集；动态 benchmark 扩展帧/网格时多持有临时矩阵 | V535 |
| B430 | 2026-05-29 | dual-model RM benchmark 用 `np.vstack` 构造 fine centers/Jacobian rows，并用 `np.diag(1/counts)` 做 coarse-to-fine 列缩放；48e/5936 synthetic benchmark 冷构建多持有 dense 临时 | V536 |
| B431 | 2026-05-29 | common method runners 在 paired GN、sparse-Bayes selected columns 和 frame-mode reference/target 中用 `np.vstack` 构造测量帧矩阵；批量 CLI case 运行有重复拼接临时 | V537 |
| B432 | 2026-05-29 | GREIT parity benchmark batch forward、fixture fallback Sn、large scalar-noise Gram/Sn 和 synthetic measurement rows 分别用 `np.column_stack`、`np.eye`、`np.eye`、`np.vstack`；48e/5936 parity 构建仍有 dense 临时 | V538 |
| B433 | 2026-05-29 | 3D inverse overview 渲染圆柱上下环和电极 marker 用 `np.column_stack`/`np.full_like` 构造三列点矩阵；报告渲染路径有小型但重复的点阵临时 | V539 |
| B434 | 2026-05-29 | fair EIDORS/PyEIDORS 诊断导出 3D boundary facets、measurement starts、measurement matrix concat 分别用行列表 `np.vstack`、`np.concatenate` 和 `np.vstack(pm.meas_matrices)`；MATLAB payload 构建有可避免拼接临时 | V540 |
| B435 | 2026-05-29 | EIDORS forward parity gate 验证 pattern manager 时用 `np.vstack(manager.meas_matrices)` 重新拼接测量矩阵；自定义 protocol parity 检查有可避免拼接临时 | V541 |
| B436 | 2026-05-29 | mesh IO format benchmark 的 tag entity-pair hash 用 `np.column_stack((index_array,value_array))` 构造 pair 矩阵；大标签表比较时有可避免列拼接临时 | V542 |
| B437 | 2026-05-29 | all-mode bucket noise sweep 出重构/误差网格图时用 `np.concatenate(values)` 汇总所有字段求色标范围；多 SNR×多方法扫参图会额外持有完整字段拼接临时 | V543 |
| B438 | 2026-05-29 | point-status、electrode-tag 和 fair EIDORS render 小诊断脚本仍用 `np.column_stack`、`np.vstack(segs)`、sigma `np.concatenate` 构造绘图辅助数组；批量报告生成有可避免拼接临时 | V544 |
| B439 | 2026-05-29 | GN difference linearized LSMR fallback 在每次 matvec 和 RHS 构造中用 `np.concatenate` 拼接测量残差和正则项；matrix-free fallback 长参数向量下多分配增广临时 | V545 |
| B440 | 2026-05-29 | direct Jacobian traditional path 用 `np.eye(n_elec)` 构造 electrode identity drive matrix；虽然维度小，但核心源码仍残留 dense helper 构造 | V546 |
| B441 | 2026-05-29 | GREIT parity、forward KSP、dynamic validation 与 3D 空间异常连通分量路径仍有 `np.isfinite(...).all()/any()` 或 `np.any(values < floor)` 全量 bool guard；批量诊断和三维 UI 路径有可避免峰值分配 | V547 |
| B442 | 2026-05-29 | block-system、electrode measurement filtering、dynamic sweep、GREIT parity ratio、GUI/CLI single-step floor flags 残留 `np.any(<comparison>)` / `np.diff` 全量比较临时；大参数/测量路径有可避免峰值分配 | V548 |
| B443 | 2026-05-29 | GN difference、tank holdout、3D runtime benchmark 仍有 `np.all(np.isfinite(...))` 写法；长向量/电极电压报告路径只为有限性检查多分配完整 bool payload | V549 |
| B444 | 2026-05-29 | GUI 和 GN-difference 单步 sigma-floor 限幅仍先构造 `delta < 0` 全量 mask，再复制 `sigma[negative_update]` / `delta[negative_update]` 子集求最大 alpha；三维参数场更新峰值内存偏高 | V550 |
| B445 | 2026-05-29 | holdout-fit 与 bucket dense 结构指标用 `weights_raw[mask]`、`areas[mask]`、`points[mask]`、`contrast[outside]` 和 `mask & outside` 复制 ROI/outside 子集；大网格指标报告峰值内存偏高 | V551 |
| B446 | 2026-05-29 | TV postprocess PDHG 循环用 `x_new[roi]`、`previous[roi]` 和 `y[not_roi]` 复制 ROI / 非 ROI 子集；三维 ROI TV 后处理每轮迭代峰值内存偏高 | V552 |
| B447 | 2026-05-29 | electrode pattern manager 的 stimulation-current measurement filter 用 `meas_mat[mask]` 高级索引复制过滤行；forward setup 协议构建有可避免临时 | V553 |
| B448 | 2026-05-29 | temporal frame validation、measurement/inverse moving-average 和 TV-PDHG 后处理无条件升宽到 `float64`，且 moving-average 用整帧 denominator/sliced-difference 表达式；三维 `float32` 多帧后处理会多占约一倍内存并增加首轮延迟 | V554 |
| B449 | 2026-05-29 | RM batch/temporal online apply 对预投影 `float32` 电压帧先 `_as_measurement_frames`/measurement contract 升宽到 `float64`，metadata wrapper 又把 `rm_matmul(dtype=float32)` 输出升宽；GPU/RM 在线三维多帧路径多一次宽窄转换和整块副本 | V555 |
| B450 | 2026-05-29 | GUI single-step sigma-floor 约束把 `float32` 背景场和更新量转为 `float64` 后再求 raw/floored sigma 与 display delta；三维单步重构结果在二次 forward/display 前多持有双精度参数向量 | V556 |
| B451 | 2026-05-29 | measurement-channel contract helper 对 Jacobian/residual/diagonal weights 全部 `_as_vector/_as_2d_array/_DiagonalMatrix` 强制 float64；float32 RM/GREIT/dynamic 入口先扩成双精度再被下游计算 dtype 拉回 | V557 |
| B452 | 2026-05-29 | difference projection `_complex_or_float_dtype` 对所有 real voltage/Jacobian 输入返回 float64；float32 normalized ΔV/RM online 上游在 contract/matmul 前已被升宽 | V558 |
| B453 | 2026-05-29 | hardware equipotential PyVista path 用 `np.zeros((n_pts,3), dtype=np.float64)` 构造 surface points；float32 reconstruction coords 进入 3D surface 显示时多持有双精度点坐标 | V559 |
| B454 | 2026-05-29 | acquisition ring-buffer 写入前 `_frame_component` 先把每个输入分量转成 float64，GUI poll 又对 `read_latest()` 已经脱离共享内存的数组再 copy；实时/硬件帧路径每帧多两处整帧临时 | V560 |
| B455 | 2026-05-29 | GUI 复杂电压阈值扫描固定分配 float64 abs 工作块，GREIT 2D rec-model 进入 3D hexa fallback 时用 float64 padded centers；float32 显示/路由边缘仍有双精度残留 | V561 |
| B456 | 2026-05-29 | normalized difference 的 reference floor 检查/钳制虽然输出已保持 float32，但 abs 工作块仍固定 float64；三维 RM normalized ΔV 首次/批量路径还有一次块级升宽临时 | V562 |
| B457 | 2026-05-29 | FrameData complex measurement vector 用 `real.copy() + 1j*imag.copy()`，mag/amplitude 先构造复数或平方临时；实时/批量重构提取测量向量时多持有整帧中间数组 | V563 |
| B458 | 2026-05-29 | HDF5 artifact checksum verification 对每个 dataset 调用 `np.asarray(dataset)` 再哈希；大型 RM/GREIT 缓存校验会把整块数组一次性读入内存，抵消 lazy/chunked artifact 的峰值优势 | V564 |
| B459 | 2026-05-29 | HDF5 legacy manifest fallback 在 dataset 缺少 `sha256` attrs 时用 `_array_digest(np.asarray(dataset))` 生成 artifact key；旧/迁移缓存的 lazy 读仍可能整块载入大型 RM/GREIT 数据 | V565 |
| B460 | 2026-05-29 | hardware reconstruction grid cache 在准备阶段就分配 interpolated/abs/normalized 三个 float64 display work buffers；float32 重构帧显示前多占三份双精度网格缓冲 | V566 |
| B461 | 2026-05-29 | 3D conductivity display 对 float64 坐标/电导率输入保持双精度进入 PyVista/Matplotlib scene；可视化首屏会把显示负载和后续派生数组放大约一倍 | V567 |
| B462 | 2026-05-29 | WSLg/Wayland embedded VTK 被禁用后仍默认尝试 PyVista offscreen；首次导入/初始化 VTK 可能长时间阻塞 GUI，并且失败前会重复进入慢路径 | V568 |
| B463 | 2026-05-29 | forward result geometry 抽取已避免 `node_coords[cell_connectivity]` 展开，但中心计算仍分配 `n_cells×dim` work buffer；3D/hex 首次 forward 返回前多持有一份坐标级临时 | V569 |
| B464 | 2026-05-29 | HDF5 artifact `_array_digest()` 对非连续 numeric 视图先 `ascontiguousarray` 整块拷贝再哈希；大型 3D/RM/GREIT 缓存写入 artifact key 时可能多占一份完整数组内存 | V570 |
| B465 | 2026-05-29 | cache-key `update_digest_with_array_payload()` / `hash_array()` 虽避免 `.tobytes()`，但仍先 `ascontiguousarray(np.asarray(...))`；非连续大型 mesh/RM/Jacobian 视图生成 key 时可能多占一份完整数组内存 | V571 |
| B466 | 2026-05-29 | GUI reconstruction `_array_pair_hash()` 在调用 streaming digest helper 前仍本地 `ascontiguousarray` 坐标/连通性数组；非连续三维 rec-model mesh signature 会额外复制一次 | V572 |
| B467 | 2026-05-29 | GREIT cache signature `_array_signature()` / `_array_digest()` 在调用 streaming digest helper 前仍本地 contiguous 化 numeric array；大型 3D GREIT 训练/导入签名生成可能额外复制一次 | V573 |
| B468 | 2026-05-29 | forward mesh-content hash、TV-IRLS digest 与 GN dense regularization hash 仍在共享 streaming helper 之前本地 contiguous 化；非连续 mesh/reg/state 视图生成 key 时额外复制一次 | V574 |
| B469 | 2026-05-29 | forward KSP session-reuse benchmark 的 `_array_sha256()` / `_sigma_sequence_hash()` 仍在 streaming helper 前本地 contiguous 化；大 sigma 序列视图跑报告时额外复制一次 | V575 |
| B470 | 2026-05-29 | `_forward_mesh_geometry_arrays()` 对规则 DOLFINx connectivity 仍逐 cell 调 `links(i)` 并构造小数组；三维 forward 结果返回前 Python 调用和小分配随 cell 数增长 | V576 |
| B471 | 2026-05-29 | forward simulation 和 dataset generator 即使 `noise_level=0` 也无条件 `data.meas.copy()` / `data_homog.meas.copy()`；默认无噪声 3D 输出路径多持有测量向量副本 | V577 |
| B472 | 2026-05-29 | EIDORS noise `_extract_measurements()` 和 same-shape `_broadcast_v2()` 在只读输入上先复制 v1/v2；批量噪声注入会在最终 noisy 输出之外多持有测量向量副本 | V578 |
| B473 | 2026-05-29 | Sparse Bayesian absolute/difference workflow 把 baseline/reference 大数组复制进 metadata；三维大网格重构结果会额外保留一份只用于展示/诊断的数组 | V579 |
| B474 | 2026-05-29 | `difference_measurement()` 在构造 difference `EITData` 时先复制 target/reference 原始电压，而 GN runtime 随后还会为内部状态再复制；差分重构入口多持有一层测量向量快照 | V580 |
| B475 | 2026-05-29 | 单帧 normalized `build_difference_vector()` 即使 reference 没有近零值也构造 safe reference 副本，并用非原地除法/取负产生临时数组；默认差分投影路径多一次 reference/diff 级内存峰值 | V581 |
| B476 | 2026-05-29 | normalized `project_measurement_jacobian()` 总是构造 safe reference 副本；大型 Jacobian 投影虽必须输出新矩阵，但 reference 向量无需在无 floor 命中时复制 | V582 |
| B477 | 2026-05-29 | 在线 RM frame measurement contract 对非连续输入先 `ascontiguousarray` 再 `.copy()`；batch 重构前处理可能短暂持有两份完整 frame payload | V583 |
| B478 | 2026-05-29 | batch normalized difference/online RM 对 1D reference 先展开成 `n_frames×n_meas` 矩阵；多帧重构时 reference payload 随帧数重复占用内存 | V584 |
| B479 | 2026-05-29 | `test_rm_reference_frames_broadcasts_vector_without_broadcast_to_copy` 仍断言 `_reference_frames()` 返回展开后的 `(n_frames,n_meas)` 副本，与 V584 的单行 reference 广播契约冲突 | V584 |
| B480 | 2026-05-29 | `cache.object_signature._normalize_for_signature()` 在调用已支持 streaming 的 `hash_array()` 前仍本地 contiguous 化 ndarray；语义缓存对象含大型非连续视图时多一次完整复制 | V585 |
| B481 | 2026-05-29 | GREIT finite-target `_apply_target_plane_offset()` 在无 `target_plane/target_offset` 的默认训练路径仍复制 target centers；大量训练目标时多保留一份中心坐标数组 | V586 |
| B482 | 2026-05-29 | persistent GN Jacobian cache 命中后仍 `np.array(cached, copy=True)` 复制完整 dense Jacobian；三维重复重构会在缓存命中路径多占一份 Jacobian 级内存峰值 | V587 |
| B483 | 2026-05-29 | RM signature `_digest_value()` 在调用 streaming array hash 前仍 `np.ascontiguousarray(array)`；三维 `coarse2fine`/noise/mask 等签名输入为大型非连续视图时会多一次完整 payload 复制 | V588 |
| B484 | 2026-05-29 | GN runtime 最终 forward-fit 只需要读取已拥有的 `sigma_final_array`，但 `EITImage(elem_data=sigma_final_array.copy())` 又复制了一份完整 element-data；三维 GN/调试拟合路径峰值内存偏高 | V589 |
| B485 | 2026-05-29 | backend worker 每次准备 profile FFCx cache 时对每个 `libffcx_*.c` 单独 glob 同 stem `.so`；编译表单多的三维 cache 首载会反复扫描目录 | V590 |
| B486 | 2026-05-29 | GUI array geometry cache 在哈希/派生 cell centers 前对 floating coords 和 integer cells 使用 `np.ascontiguousarray`；三维显示拿到切片/视图输入时会先复制完整几何 payload | V591 |
| B487 | 2026-05-29 | mesh-derived HDF5 cache 构建 cell connectivity 时逐 cell 调 `connectivity.links(i)`；DOLFINx 已提供 flat topology array/offsets 的情况下三维首载仍承担大量 Python 循环调用 | V592 |
| B488 | 2026-05-29 | CUDA structured 后端 `_stable_hash()` 在 sigma 状态复用键上对 float64 非连续视图先 `ascontiguousarray`；三维批量/切片 sigma 首次求解会多一次完整 sigma staging copy | V593 |
| B489 | 2026-05-29 | absolute-startup GN 与 direct-Jacobian 语义缓存键在 sigma 哈希前先 `ascontiguousarray`；三维非连续 sigma 视图进入缓存判断时会多一次完整 sigma staging copy | V594 |
| B490 | 2026-05-29 | matrix-free sigma fingerprint 与 ROM snapshot 去重在 hash 前复制非连续 sigma/column 视图；三维 GN/ROM 缓存与去重阶段会多保留完整 staging payload | V595 |
| B491 | 2026-05-29 | GREIT artifact registry 在规范化 ndarray 签名字段时先复制为 contiguous 再哈希；大型非连续 protocol/grid 视图进入 artifact lookup 时多一次完整 payload copy | V596 |
| B492 | 2026-05-29 | Sparse-Bayesian Jacobian cache key 在 baseline 哈希前强制 contiguous copy；非连续 baseline 视图进入缓存判断时多一次完整 baseline staging copy | V597 |
| B493 | 2026-05-29 | GN linear-system 的正则签名、ROM basis/operator payload 在哈希前强制 contiguous copy；三维 ROM/低秩缓存键构建会多保留完整 dense staging payload | V598 |
| B494 | 2026-05-29 | RtR prior `_signature_for_payload()` 在已有 dense/sparse payload 上再次 contiguous 化只为哈希；大型自定义正则签名会多一次完整 payload staging copy | V599 |
| B495 | 2026-05-29 | GUI 3D 空间异常过滤为最近邻距离分配完整 `nearest_valid` bool mask；大型候选异常集合估计半径时多一份候选数级布尔缓冲 | V600 |
| B496 | 2026-05-29 | GUI 3D 点云背景采样把只读 `ranks` 复制为初始 `candidates`；大点云下每次背景采样多一份 `max_points` 级整数缓冲 | V601 |
| B497 | 2026-05-29 | 边界电压 y 轴范围计算对每条曲线调用 `np.isfinite(arr)` 分配完整 bool mask；长帧曲线显示时多一份 series 级临时缓冲 | V602 |
| B498 | 2026-05-29 | 硬件页 equipotential 曲面高度范围虽已分块扫描，但每块仍 `np.isfinite(chunk)` 新建 bool mask；大节点数组显示时重复分配 chunk 级缓冲 | V603 |
| B499 | 2026-05-29 | GUI 复数通道判定虽已分块扫描 imaginary view，但每块仍新建 `isfinite`/`abs` 临时数组；大三维 `complex64` 结果切换通道时重复分配 chunk 级缓冲 | V604 |
| B500 | 2026-05-29 | 持久 dense Jacobian 进程缓存只按条目数淘汰，默认最多保留 4 个数组；三维大 Jacobian 命中优化可能在同一 worker 内累积多份高驻留内存 | V605 |
| B501 | 2026-05-29 | 生成/加载 mesh 的进程缓存只按 8 个条目淘汰；连续查看多个大三维网格时，完整 DOLFINx mesh、facet/cell tags 与派生数组可能在 GUI/worker 进程内累积驻留 | V606 |
| B502 | 2026-05-29 | forward static setup 进程缓存只按 8 个条目淘汰；三维多配置切换时 CEM electrode CSR matrix 与电极长度等静态 setup payload 可能在同一进程内累积驻留 | V607 |
| B503 | 2026-05-29 | GUI 重建 `_SYSTEM_CACHE` 与 `_FAST_CONTEXT_CACHE` 仍只按 4 个条目淘汰；三维 full-GN/单步上下文中 EITSystem、dense/linearized operator、LU factor 与显示几何会跨配置累积驻留 | V608 |
| B504 | 2026-05-29 | RM fit-Jacobian 进程缓存虽按单数组 512MiB 拒绝超大项，但仍可保留 2 个接近上限的 Jacobian；三维 RM artifact 热路径可能累积约双倍预算驻留 | V609 |
| B505 | 2026-05-29 | GUI 3D 默认预热仍是 import-only worker warm；实测未预热 setup prime 约 10.4s、预热 runtime 后 setup prime 约 0.53s，导致首次点击仍承担 mesh/static setup/JIT 路径 | V610 |
| B506 | 2026-05-29 | TV regularization 非有限权重 median fallback 同时分配全长 finite bool mask 与全长 NaN work copy；大 3D TV/curvature 调参时非有限保护路径多一份全量布尔驻留 | V611 |
| B507 | 2026-05-29 | GUI/数据集正问题 rectangle/cuboid fallback painting 为每个坐标轴创建 `np.abs(centers[:,axis]-c)` 全长临时数组并合成 full mask；大型三维长方体异常涂色多份 n_cells 级临时驻留 | V612 |
| B508 | 2026-05-29 | GN `_require_finite()` 在 complex 数组失败路径先 `np.abs(arr)` 再摘要；大型 complex admittance/GN 残差或矩阵一旦含 NaN，会额外分配完整实数 magnitude 数组 | V613 |
| B509 | 2026-05-29 | GN difference measurement-weight reference 用 `np.abs(measured-baseline)`；实数大测量向量会保留减法临时和 abs 输出两份完整数组 | V614 |
| B510 | 2026-05-29 | GN line-search 扰动限制用 `eps_machine*np.abs(x)/np.abs(dx)` 构造完整 lower-alpha 数组；三维参数场 full line-search 会额外保留多份 n_elem 级临时 | V615 |
| B511 | 2026-05-29 | GN matrix-free 预条件对角线清洗用 `(~np.isfinite(arr)) | (arr <= floor)` 构造完整 bad_mask；三维参数场只为 clamp reason 多分配 n_param 级布尔数组 | V616 |
| B512 | 2026-05-29 | GN line-search 溢出上界用 full `au_pos`/`au_neg` 数组再 mask 为 `inf`；三维 full line-search 在参数场外额外持有两份 n_elem 级临时 | V617 |
| B513 | 2026-05-31 | GUI 复值仿真中实部、虚部、幅值和相位缺少通道语义；相位图沿用 `S/m`/顺序色图语义，未固定到 `[-π, π]` 弧度范围 | V618 |
| B514 | 2026-05-31 | 将真值/重构物理通道强制共享色标后，重构结果的大幅异常值会把真值图色标拉到千级，导致用户设置的背景 `1+2j` 和目标 `2+3j` 在真值图上不再对应到 `Re=1..2`、`Im=2..3`、`|σ|≈2.236..3.606` 的实际范围 | V618 |
| B515 | 2026-05-31 | 复导纳成像异常仅从 GUI 图像判断会混淆色标、差分方向、复数共轭/符号、协议和求解器误差；缺少同一网格/同一协议下 EIDORS 与 PyEIDORS 的逐步公平对比产物 | V619 |
| B516 | 2026-06-03 | 纯 Nix 打包 GUI/脚本从只读安装/包路径运行时，默认 `.pyeidors_cache`、`eit_meshes`、GUI 默认 `data/`、`results/`、`eit_recordings/` 和脚本/benchmark 默认报告输出可能被解析到当前目录或源码/包路径，导致 forward/GREIT/RM/mesh 预热或用户输出写入只读路径并触发 PermissionError | V620 |
| B517 | 2026-06-04 | 纯 Nix `nix run .#eit-app-complex64-cuda` 中 simulation GN difference 快速路径动态导入 `scripts.common.gn_difference_runner`，但 package 只安装 `src/`，导致自动构建 RM artifact 时 `ModuleNotFoundError: No module named 'scripts'` | V621 |
| B518 | 2026-06-04 | `./eit-gui` 在 WSLg 下会默认启用 `EIT_APP_3D_WSLG_PYVISTA_OFFSCREEN=1`，但纯 Nix `nix run .#eit-app-complex64-cuda` wrapper 未设置该默认值，导致同一 3D 仿真结果因启动通道不同而一个走 PyVista/VTK offscreen、一个回落 Matplotlib 3D | V622 |
| B519 | 2026-06-04 | 纯 Nix PyVista 0.46.4 的 `DataSetFilters.extract_surface()` 不接受 `algorithm` 关键字，3D 仿真前向/逆向结果刷新进入 PyVista offscreen 后在边界轮廓提取处抛 `TypeError`，导致界面显示 `No data` 且逆问题结果刷新被误报为重建失败 | V623 |
| B520 | 2026-06-04 | 纯 Nix complex/complex64 CPU/GPU GUI wrapper 固定或等效走当前 Python worker 时，会使本应外部切到 real `default`/`cuda` 的纯实值 forward 仍在 complex 当前进程执行；同时 hex+GPU 复导纳请求会被自动提升到 real-only `cuda_structured` 并报 `cuda_structured forward backend is real-only` | V139,V624 |
| B521 | 2026-06-04 | 纯 Nix GUI 的仿真真值/重构结果 Matplotlib plot 区域文字和占位符比 `./eit-gui` 模糊；Qt 高 DPI 环境变量可能被平台早返回跳过，FigureCanvasQTAgg 也未在绘制/resize 前显式同步 Qt DPR/逻辑 DPI，导致 plot raster 被 WSLg/Windows compositor 放大 | V625 |
| B522 | 2026-06-05 | 仿真激励电流控件下限设为 `1e-12` 但只显示 6 位小数，Qt/旧配置可把激励恢复或量化为 `0.0`；纯 Nix backend worker 收到 `drive_value=0.0` 后触发 `drive_value must be positive` 并中断正问题 | V626 |
| B523 | 2026-06-05 | 伪三维 `pseudo3d_noser_rm` 被加入仿真 Step4 的静态方法列表，2D 正问题也能选择该 2D→3D 显示路线；3D 伪三维请求还继承 3D/CUDA 元数据，折叠成 2D 逆问题后仍可能触发 PETSc CUDA Mat/Vec 能力错误 | V627 |
| B524 | 2026-06-05 | 三维真值/重构/伪三维显示在 WSLg、PyVista offscreen 失败或大点云分支仍可创建并切到 Matplotlib 3D fallback，导致 GUI 出现 Matplotlib 坐标轴式三维成像而不是项目要求的 PyVista/VTK 三维视图 | V628 |
| B525 | 2026-06-05 | 伪三维插值路线把两层 3D 边界电压压扁成一个 16 电极二维环做单次 RM，而不是先按源电极层分别做 8 电极二维反演再沿 z 插值；右侧 3D 重构手动切到 `体` 时又被大网格自动点云逻辑立即切回 `点云` | V629 |
| B526 | 2026-06-10 | app-level quit path bypassed main-window `closeEvent`, leaving `DeviceController` QThread alive and causing `QThread: Destroyed while thread is still running` abort | V630 |
| B527 | 2026-06-10 | several hardware/simulation/interop/reconstruction `QFormLayout` rows kept default `DontWrapRows`, causing narrow-pane horizontal scroll while sibling panels wrapped | V631 |
| B528 | 2026-06-11 | `setup_generated_mesh` `dimension=3` branch called `create_cylinder_3d_eit_mesh` directly instead of `load_or_create_mesh`, bypassing eit_meshes disk/process cache: every fresh process re-ran gmsh (~24 s, msh written to /tmp) and GUI `runtime_diagnostics.mesh_cache_hit` stayed null for 3D while 2D reported true | V632 |
| B529 | 2026-06-11 | complex PETSc runtimes produce zero-imag complex arrays even for real-valued routes; `model_signature_from_forward_model` cast contact impedance `z` to `float64` (warning/error and lost nonzero imaginary signature content), and GN difference base-measurement cache cast complex voltages to `float64` via implicit warning | V76 |
| B530 | 2026-06-11 | PETSc CUDA vec-loop CEM forward solved in reference-electrode gauge (`U_0=0`) but returned raw electrode voltages without recentering to SciPy's zero-mean gauge; measurement projection stayed equivalent, while backend parity tests saw a constant per-pattern offset | V633 |
| B531 | 2026-06-11 | Matrix-free PETSc KSP in complex64 runtime passed zero-imag complex work/result vectors directly into real-valued SciPy `LinearOperator` callbacks and final real casts, triggering `ComplexWarning`/PETSc shell failure or fallback despite a real mathematical operator | V634 |
| B532 | 2026-06-11 | full `complex64-cuda` unit mix exposed real-valued GN regularization/runtime/line-search, workflow residual metrics, digit metrics, and visualization interpolation paths that implicitly cast zero-imag complex arrays to real, tripping strict `ComplexWarning` despite mathematically real payloads | V635 |
| B533 | 2026-06-11 | phase/NOSER diagnostic sweep scripts still wrote bulk arrays via `np.savez_compressed`, bypassing the repository HDF5 artifact contract and failing the production persistence guard | V636 |
| B534 | 2026-06-12 | pytest-cov used coverage precision 0, so `86.72%` rounded to `87%` for fail-under and full suite exited 0 while exact total sat below `--cov-fail-under=87` | V637 |
| B535 | 2026-06-12 | default `complex64-cuda` dev shell still relied on legacy uv `.venv-complex64-cuda` semantics while the pure Nix package/app closure already carried Torch/CUQI/Qt/pyqtgraph; `uv run` could create a separate `.venv`, manifest verification looked for the wrong/no profile lock, and the integration contract skipped until `.venv` existed | V638 |
| B536 | 2026-06-14 | GUI backend worker cross-profile fallback still wrapped `eit_app.backend_worker` in `uv run` inside `nix develop`, so forward/reconstruction worker isolation could recreate/use `.venv` despite pure Nix default route | V638 |
| B537 | 2026-06-14 | lightweight GUI helper tests imported full `eit_app.ui.main_window`, and GUI smoke collection legitimately imported it; both paths load `pyqtgraph`/PyOpenGL, whose EGL probe leaks `/proc/cpuinfo` under warnings-as-errors and can fail unrelated tests before timing/prewarm behavior is checked | V639 |
| B538 | 2026-06-14 | `safe_dot` ran `np.dot` outside local `errstate`; overflow warning escaped under warnings-as-errors before project nonfinite-result guard raised `FloatingPointError` | V640 |
| B539 | 2026-06-14 | `_disable_tf32` wrote PyTorch 2.9 legacy `allow_tf32`; deprecation `UserWarning` escalated to unit-test failure before device policy assertions | V641 |
| B540 | 2026-06-14 | public API lazy-import subprocess tests replaced Nix `PYTHONPATH` with bare repo `src`; accessing lazy exports dropped Nix NumPy/SciPy deps and failed with `ModuleNotFoundError` | V642 |
| B541 | 2026-06-14 | complex64-cuda integration passed Python float `1.0` to DOLFINx `fem.Constant` and cast zero-imag assemble/allreduce/sigma arrays via `float`/`astype(float)`; CEM scripts, electrode-label helpers, and 3D diagnostics failed under complex PETSc | V643 |
| B542 | 2026-06-14 | GN difference warm cache hit skipped base forward solve, so `fwd_model` backend diagnostics stayed pre-solve; jacobian/operator cache keys used different `backend_signature` than cold path and `operator_A` missed | V644 |
| B543 | 2026-06-14 | mm/cm/m integration fixture assigned square boundary facets using unrounded normalized float coords; `side_length=0.2` produced bin-boundary values below cm/mm, shifting right/bottom electrode tags and making voltage vectors look mismatched | V645 |
| B544 | 2026-06-14 | complex64-cuda integration parity gates kept double-precision `1e-6` vector/RMSE tolerance and fast-path whitelist missed `native-complex`, so valid single-precision native-complex solves failed | V646 |
| B545 | 2026-06-14 | targeted complex64-cuda unit rerun exposed test isolation drift: optimized mesh tests patched `opt_mesh_module.ufl.Measure` while lazy `ufl` could still be `None`, and FEM `Constant` fakes used `float(value)`, raising `ComplexWarning` for valid zero-imag PETSc scalar constants | V647 |
| B546 | 2026-06-14 | running GitNexus analyze inside `complex64-cuda` Nix shell resolved `node` to `/usr/bin/node` while `LD_LIBRARY_PATH` pointed at Nix GCC libs, causing `GLIBC_2.36/2.38 not found` before the knowledge graph could rebuild; clean WSL CLI worked because it had no Nix library-path contamination | V648 |
| B547 | 2026-06-14 | sharded validation dry-run still emitted `nix develop -c uv run pytest`, omitting `.#complex64-cuda` and reintroducing legacy uv/default-shell split despite pure Nix project contract | V649 |
| B548 | 2026-06-15 | `nix run .#eit-app-complex64-cuda` failed with `ModuleNotFoundError: eit_app.ui.forward_prewarm` because tracked `main_window.py` imported new lightweight helper modules still untracked; Git flake source excluded them from Nix store package | V650 |
| B549 | 2026-06-15 | GUI simulation control labeled target generator length as `网格尺寸`, implying exact generated mesh size though code converts value to integer `refinement` and actual cell sizes differ | V651 |
| B550 | 2026-06-15 | GUI simulation fixed default target length made relative mesh density depend on physical radius; 3D default radius `0.18 m` kept `h=0.1 m` (~D/3.6) while 16-electrode EIT should default near `D/15..D/20` | V652 |
| B551 | 2026-06-15 | `目标特征长度` still read like object/anomaly feature length in Chinese and kept users editing a nonhuman scale value; mesh UI needed density-first ordinary/engineering/advanced layers | V651,V653 |
| B552 | 2026-06-15 | GUI mesh density slider tick labels were placed by text-row geometry instead of Qt slider handle centers, so the ruler ticks and coarse/medium/fine/very-fine captions drifted vertically out of correspondence | V654 |
| B553 | 2026-06-15 | GUI mesh-density summary displayed `预计单元数≈15.1k` for 3D hex `D/33/refinement≈8`, but the geomv2 hex generator actually creates `100,224` elements (`4176` base O-grid quads × `24` z layers); the estimate used an obsolete volume heuristic and mixed 单元/元素 wording | V655 |
| B554 | 2026-06-15 | After hex estimate was fixed, 2D and 3D tetra still used obsolete density/volume heuristics: 2D default showed `预估元素数≈518` while Gmsh generated `2,034` elements, and 3D tetra default showed `≈9.2k` while Gmsh generated about `31.2k` elements | V656 |
| B555 | 2026-06-15 | `nix run .#eit-app-complex64-cuda` 3D forward solve hit PETSc CUDA KSP reason `-10`; dense LU fallback was correctly memory-gated (`30.40GiB>2.00GiB`) but code raised instead of CPU sparse fallback, so GUI aborted despite valid pure Nix CUDA runtime | V657 |
| B556 | 2026-06-15 | `ForwardProblemPanel._status_label` lacked word wrap / zero minimum width / ignored horizontal size policy; long PETSc error text widened Step3 form, stretched `求解正问题` button, and exposed horizontal scroll | V658 |
| B557 | 2026-06-15 | GUI GPU profile honored tetra selection but routed generated 3D forward to experimental DOLFINx/PETSc CUDA `spd_gamg`; tetra CUDA auto solver should instead use `3d_gamg` (`fgmres+gamg`) before AmgX work | V659 |
| B558 | 2026-06-15 | attempted stable-GPU fix silently rewrote GUI `mesh_family="tetra"` to `hex`, making tetra/hex simulations use identical hex mesh and identical element counts | V659 |
| B559 | 2026-06-16 | explicit `cuda_amgx` 3D tetra forward benchmark fell back to gmres/none successfully, but the failed temporary PCAMGX setup candidate was left for Python finalization and printed `PCDestroy_AMGX s_rsrc == NULL` at process exit | V660 |
| B560 | 2026-06-16 | explicit `cuda_amgx` 3D CEM route still bypassed PCAMGX: old V111 dense-direct gate treated every non-direct CUDA iterative CEM solve as dense fallback, and preset used CG without stable AMGX options | V661 |
| B561 | 2026-06-16 | explicit `cuda_amgx` setup fail-fast existed, but solve-stage negative KSP convergence still entered generic CUDA dense/CPU fallback via `petsc_ksp_failed:*`, hiding failed PCAMGX configs as successful benchmark/GUI results | V662 |
| B562 | 2026-06-17 | reconstruction controller recomputed 3D CUDA forward policy without `mesh_family`, so tetra inverse runtime used non-tetra auto downgrade path (`spd_gamg`) while simulation forward runtime used tetra `3d_gamg`; debug/legacy inverse routes and Hypre CUDA presets also stayed in default GUI choices, making diagnostic routes look mainline | V663 |
| B563 | 2026-06-17 | complex block-real AmgX smoke first compared reference-electrode-gauge candidate against recentered complex64-cuda runtime output, creating false ~1.0 relL2 error; default `block_jacobi/AGGREGATION` profile also failed PETSc/AmgX setup with error 77, so parity harness needed direct reference + gauge match + setup-proven default profile | V664 |
| B564 | 2026-06-17 | 3D complex GPU-only `16e_ref8` skipped dense fallback (`30.37GiB>2.00GiB`) and native sparse `3d_gamg` reported convergence while exported-system true residual was ~6.24; block-real AmgX true residual was ~1e-8 but route-vs-route error looked huge because native route was invalid truth | V665 |
| B565 | 2026-06-18 | 3D complex `noser_rm` very-fine auto-build had complex NOSER diagonal RtR; measurement form compact branch only accepted positive real diag, so `_prior_to_dense_matrix()` tried `51536×51536 complex64` dense prior (~19.8GiB) | V666 |
| B566 | 2026-06-20 | measurement form 对 graph Laplace/curvature RtR 直接 dense/pinv R^-1 J^H；奇异/近奇异 sparse prior 既 OOM 又放大数值误差，GUI 因旧 guard 默认 param | V668 |
| B567 | 2026-06-20 | 三维 RM 热路径在完整 fit Jacobian 超过预算时仍复用旧 artifact；图像可由 RM@ΔV 重建，但 `simulated` 为空，GUI 边界电压拟合只剩真值曲线 | V669 |
| B568 | 2026-07-07 | `backend_doctor._run_command` let `TimeoutExpired` escape; worker/nix/nvidia timeout → traceback before JSON report | V670 |
| B569 | 2026-07-07 | sm61 runtime + complex128 GPU complex route returned nonexistent `complex-cuda`/sm61 package path instead of usable CPU complex fallback | V671 |
| B570 | 2026-07-07 | ECD-CWR CEM simulation module existed in source/dev tests but pure Nix flake source omitted new untracked module, so packaged backend worker hit `ModuleNotFoundError: No module named 'eit_app.ecd_cwr_simulation'` | V672 |
| B571 | 2026-07-10 | packaged wrapper exported `gcc` runtime libs only; no compiler/binutils on `PATH`, so host without gcc failed first FEniCSx/FFCx JIT | V673 |
| B572 | 2026-07-10 | V622 static test coupled offscreen value directly to next `PATH` token; inserting valid `CC/CXX` wrapper args caused false regression | V622 |
| B573 | 2026-07-12 | packaged wrapper init executes `mkdir`, but common wrapper `PATH` omitted `coreutils`; stripped-host `nix run` failed before doctor/compiler probes despite bundled GCC | V680 |
| B574 | 2026-07-12 | post-T583 default suite passed 2017 assertions but `src/pyeidors` coverage remained 86.01%<87%; T584 `src/eit_app` acceptance coverage cannot repair unrelated package-wide debt | V637,T585 |
| B575 | 2026-07-12 | realtime `auto` preferred measurement-domain diagonal state without NOSER spatial anchor/guard; sustained real-water shift amplified spatial modes and polluted later frames after static NOSER recovered | V677,V682,T586 |
| B576 | 2026-07-19 | CEM benchmark mixed PyEIDORS complex64 with float64 peers, independent meshes/orders, asymmetric timing/cache scopes → cross-FEM + speed claims confounded | V687,V688,T588 |
| B577 | 2026-07-19 | Windows EIDORS batch first used assumed `C:\Program Files\MATLAB\R2023b\bin\matlab.exe`; actual runtime was on `D:` → launch failed before benchmark | V689,T588 |
| B578 | 2026-07-19 | Existing Robin PETSc parity test configured GMRES `rtol=1e-8` but float64 branch asserted direct-LU parity at `rtol=1e-10`; observed `1.79e-7` max abs mismatch despite benchmark SciPy-LU equivalence at ~1e-15 | V690 |
| B579 | 2026-07-19 | Repo-wide Ruff check traversed untouched untracked author notebook `CEM-via-Robin-Boundary/NGSolveEIT/CEM-via-Robin.ipynb` and requested reformat, blocking project verification | V691,T588 |
| B580 | 2026-07-19 | Full `complex64-cuda` pytest was wrapped in 120 s outer timeout; harness killed buffered run with exit 124, then PETSc reported broken pipe/MPI abort | V692,T588 |
| B581 | 2026-07-19 | Two read-only Nix probes used nested PowerShell/WSL/bash here-doc quoting; command parse failed before Python, initially obscuring dependency check | V697,T589 |
| B582 | 2026-07-19 | Absolute-accuracy fan fixture import matched source boundary ids directly against reordered DOLFINx local topology ids: 2/32 direct vs 32/32 through `geometry.input_global_indices`; prepare failed before multiprecision solve | V698,T589 |
| B583 | 2026-07-19 | Targeted validation stopped before pytest because Ruff format-check found the new V698 regression test unformatted; existing formatting gate is sufficient, then format and rerun | T589 |
| B584 | 2026-07-19 | PyEIDORS imported the same fan geometry/topology with reordered vertices, but v1 mesh fingerprint hashed raw node order and falsely rejected it; fingerprint must canonicalize the global vertex permutation | V699,T589 |
| B585 | 2026-07-19 | A post-run JSON probe repeated nested PowerShell/WSL/bash quoting and failed before reading the valid artifact; switched to direct PowerShell UNC JSON/CSV reads | V697,T589 |
| B586 | 2026-07-19 | First ephemeral NGSolve probe inherited the repo/Nix Python 3.13 and could not load system `libstdc++.so.6`; forcing system Python inside the project then hit `requires-python ==3.13.*`; `--no-project --python /usr/bin/python3` isolated the solver correctly | V700,T589 |
| B587 | 2026-07-19 | Full complex64-cuda suite executed 2487 tests with 2039 passed/448 skipped and no test failure, but command exited 1 because repository-wide coverage was 86.21% below the pre-existing 87% gate; retain result and rerun identical suite with `--no-cov` for functional exit status | T589 |
| B588 | 2026-07-19 | First staging attempt met a transient `.git/index.lock`; an initial diagnostic also let PowerShell expand bash `$()`; direct single-process WSL checks then found the lock already absent, so no lock file was deleted | V697,T589 |
| B589 | 2026-07-19 | T590 新增精确圆形 CEM 基准脚本在首次验证前尚未经过 Ruff 自动排版，`ruff format --check` 按既有格式门禁失败；无需新增不变量，先格式化再继续验证 | T590 |
| B590 | 2026-07-19 | T590 精确圆形 CEM 基准脚本保留未使用的 `scipy.io.loadmat` 导入，Ruff F401 在执行实验前拦截；既有静态检查门禁足够，删除死导入 | T590 |
| B591 | 2026-07-19 | 圆形 fixture 生成器把 0-based `edges` 与 1-based `electrode_nodes` 混在同一内存返回值中，拓扑语义不清且单测无法逐项核对；统一内存表示为 0-based | V701,V707,T590 |
| B592 | 2026-07-19 | 统一内存拓扑为 0-based 后，MAT 保存路径漏为 `electrode_nodes` 加一，读回再次减一使电极整体左移、PyEIDORS 电极测度为零；新增保存后 1-based/导入测度回归门禁 | V707,T590 |
| B593 | 2026-07-19 | 首次插入 V707 回归时补丁锚在 V702 函数中段，把其余断言误归入新测试；现有格式/pytest 门禁足够，恢复两个独立测试函数后再执行失败→修复循环 | T590 |
| B594 | 2026-07-19 | T590 首次 exact-suite 认证命令把 `/usr/bin/time` 格式串嵌入 PowerShell→WSL→bash 双引号，shell 在 Python 前报未闭合引号；沿用 V697，去掉外层计时包装并以单一字面命令重跑 | V697,T590 |
| B595 | 2026-07-19 | exact-suite 2×2 静态证据图首次导出把底部说明用绝对 `figure.text(y=.005)` 放置，与下排 x 轴标签重叠；V706 渲染检查足够，改用 constrained-layout 管理的 `supxlabel` 后复查 | V706,T590 |
| B596 | 2026-07-19 | MCP report 读取最终认证 JSON 时在 validator 调用前被严格 `JSON.parse` 拒绝；聚合器将 PyEIDORS 未提供的 assembly/import timing 写成 Python 非标准 `NaN`，改为 JSON `null` 并加 strict-serialization gate | V708,T590 |
| B597 | 2026-07-20 | T590 全量 `complex64-cuda` pytest 首次运行被外层 120 s 命令预算终止，PETSc/MPI 随后因 stdout 管道关闭报告 `Broken Pipe`；这不是测试断言失败，无需新增产品不变量，改用足够的验证时限原样重跑 | T590 |
| B598 | 2026-07-20 | T590 报告只突出 `Robin/Classic` 同阶段比值且把单次 state population 标成“热态建态” → 读者把跨阶段比值误读为 PyEIDORS 冷态快于热态；原始绝对值实际显示 warm reuse 快 12–33×；setup 单样本也不足以支持严谨比较 | V709,T591 |
| B599 | 2026-07-20 | T591 NGSolve 批处理让 PowerShell 抢先展开 Bash `$case_dir`，聚合预检又让 PowerShell 抢先执行 `$()` → runner/聚合均未启动且后者外层假报 exit 0；改用 7 条显式路径 + 无命令替换检查 | V697,T591 |
| B600 | 2026-07-20 | T592 Gmsh 实测 `h_max` 比例约 `1.33/1.69/1.68`，初版 FEM 外推却按 target-h 固定比例 2 使用 `2^p` → 非均匀网格 Richardson 阶数/极限有偏 | V716,T592 |
| B601 | 2026-07-20 | T593 精确求解优化探针让系数矩阵推断为 `QQ`、整数 RHS 推断为 `ZZ`，`DomainMatrix.lu_solve` 拒绝混合域 | V718,T593 |
| B602 | 2026-07-20 | T593 refinement 汇总误将可选 `nodes/cells` 当必填，破坏 V705 最小排序夹具；缺省时改由权威 exact case 网格推导 | V705,V717,V720,T593 |
| B603 | 2026-07-20 | T596/T597 测试先行首次收集因预注册扩展套件与低 z 归因模块尚未创建而 `ModuleNotFoundError`；这是预期红灯，V726/V729 已定义所需能力，无需新增不变量 | V726,V729,T596,T597 |
| B604 | 2026-07-20 | T596 首轮实现通过 8/9 扩展/归因门禁，唯一失败为预期的外部 NGSolve/EIDORS extension runner 文件尚未创建；V727 已要求其 σ/矩阵身份字段，无需新增不变量 | V727,T596 |
| B605 | 2026-07-20 | T596/T597 首次 Ruff 门禁发现 5 个新文件未格式化、`loadmat/_mp_from_sympy/Fraction` 死导入及低 z 汇总漏导入 `math`；既有 Ruff 门禁足够，无需新增不变量 | T596,T597 |
| B606 | 2026-07-20 | T596 首次 Q4 QQ 可行性探针误把 shell 命令预算设为 1 s，包装器以 exit 124 杀死进程；不能据此作性能/可行性结论，沿用 V692 以长预算和短周期 yield 原样重跑 | V692,T596 |
| B607 | 2026-07-20 | T596 Q4 `530×530` Classic + `513×513` Robin SymPy `DomainMatrix.lu_solve` 探针 28 min 未完成且无持久化中间成果；Q0..Q3 路径的“多 RHS”不足以保证 Q4 可恢复扩展，需 compiled QQ 后端/等价认证重构 + 原子 truth cache | V731,T596 |
| B608 | 2026-07-20 | T596 FLINT 0.6.0 小型 CEM smoke 的 exact solve 已完成，但结果提取使用不受支持的 `fmpq_mat[row_slice,col_slice]` → `TypeError`; 改为显式复制电极子矩阵，V731 既有 exact residual/identity gate 足够 | V731,T596 |
| B609 | 2026-07-20 | T596 Q4 FLINT helper 从 Nix dev shell 启动时继承 `LD_LIBRARY_PATH`，系统 Python 3.10 错载 Nix expat 并报 `GLIBC_2.36/2.38 not found`; 外部 exact backend 必须清理动态链接/Python 路径并校验固定版本 | V732,T596 |
| B610 | 2026-07-20 | T596 NGSolve 完成 16 个均匀 case 后，首个异质 X17 导入材料顺序为 `sigma_2,sigma_1` 而非 Gmsh 声明序；按域索引直接取 conductivity 会交换左右 σ，必须按材料名映射并重建逐单元 digest | V727,T596 |
| B611 | 2026-07-20 | T596 X17 材料名映射修正后仍用 `CoefficientFunction(tuple)`，NGSolve 将 tuple 解释为向量并拒绝非标量 `SymbolicBFI`; 域标量必须由 `mesh.MaterialCF(name→value)` 构造 | V727,T596 |
| B612 | 2026-07-20 | T596 X17 MAT 有正确 heterogeneous `truth_elem_data`/digest，但漏写 `conductivity_pattern`; EIDORS 数值使用异质 σ，报告却默认 `uniform`，QQ 聚合身份门禁拒绝；MAT 必须同时携带 pattern 证据字段 | V727,V728,T596 |
| B613 | 2026-07-20 | T597 MATLAB backend-cross 用 `repmat(struct(),N,1)` 预分配无字段结构，首条带字段 record 赋值报“在不同结构体之间进行下标赋值”；需预声明同构字段，矩阵/求解尚未进入结果聚合 | V729,T597 |
| B614 | 2026-07-20 | T596 extension timing gate 唯一违规为 X11 NGSolve Robin：cold median `1.372 ms` < warm `1.725 ms`，warm 11 样本跨 `15.8 us..6.44 ms` 显示调度抖动；按同一 11×16 协议重跑该完整三求解器 case，⊥ 发布失真聚合 | V709,T596 |
| B615 | 2026-07-20 | T596 X11 NGSolve 单例重跑命令在 PowerShell→WSL 边界使用 Bash `$case_dir`，变量到 Python 前为空并请求 `/common_mesh/...`; 沿用 V697 改用完整字面路径，不能记作求解失败 | V697,T596 |
| B616 | 2026-07-20 | T596/T597 最终真值元数据与归因防离群测试补丁后，Ruff 门禁发现 3 个新增 benchmark 文件需机械重排；执行 Ruff formatter 后同一 format/check 门禁通过，现有格式不变量足够 | T596,T597 |
| B617 | 2026-07-29 | v2 交付前真实互操作验收发现 export/import CLI 仍依赖已删除 benchmark/solver 内部符号，并使用过期 `EITSystem.setup`、DOLFINx `Function.vector()`/`DofMap()` API；MAT 导入又固定 `float64` 坐标，与 `complex64-cuda` 的 float32 DOLFINx/PETSc 运行时不匹配；Bridge 烟测仅计数却强转 complex conductivity 为 real 并告警 | V733,T598 |
| B618 | 2026-07-30 | Geometry v2 将边界实体从 edge 泛化为 facet 时，集合推导循环变量已改名但表达式残留 `edge`，导致 2D/3D 导入在建标签前统一 `NameError`；属机械重命名疏漏，由现有 V735 2D+3D 实网格测试直接覆盖 | V735,T599 |
| B619 | 2026-07-30 | Geometry v2 同时保留 canonical `boundary_facets` 与 legacy `boundary_edges`，但未约束二者一致；更新旧字段而遗留新字段时会合法校验却只导入部分边界，造成电极/边界标签静默漂移 | V738,T599 |
| B620 | 2026-07-30 | 真实 MATLAB/EIDORS 3D `a3cr` 捕获保存 `electrode_nodes` 为 `16×1`，SciPy `loadmat(...,squeeze_me=True)` 压成 `(16,)`；通用验证器用 `atleast_2d` 误恢复成 `1×16`，拒绝合法的每电极单节点模型 | V739,T599 |
| B621 | 2026-07-30 | 修复 B620 后真实 `a3cr` 网格可导入，但 16 个单节点 EIDORS 点电极没有任何完整三角 facet 子集；PyEIDORS CEM 初始化产生 16 个 zero-measure warnings 后拒绝非正 electrode length。协议必须区分精确 surface 电极与显式 point→incident-facet 投影 | V740,T599 |
| B622 | 2026-07-30 | T599 新增 Geometry v2/CLI/三维示例后首次聚焦门禁在 pytest 前发现 5 个 Python 文件未按 Ruff 标准布局；既有格式门禁足够，机械格式化后必须原样重跑检查与测试 | T599 |
| B623 | 2026-07-30 | 原生 PyEIDORS→EIDORS 三维新手示例首次真实运行给圆柱生成器传入单个 `electrode_level_fractions=(0.5,)`，违反该生成器至少双层高度的既有契约并在建网格前失败；示例必须采用项目默认双层 `(0.25,0.75)` 并由真实运行门禁覆盖 | T599 |
| B624 | 2026-07-30 | `complex64-cuda` 原生三维前向结果虽为数学实数，Bridge CSV 保存仍用 `np.asarray(..., dtype=float)` 直接丢弃复数 dtype 并发出 `ComplexWarning`；零虚部测量必须显式安全归一化，非零虚部必须拒绝静默降格 | V733,T599 |
| B625 | 2026-07-30 | 首次真实 PyEIDORS→EIDORS 验收启动命令把 MATLAB 代码嵌入 PowerShell→WSL→bash 三层引号，bash 在 MATLAB 启动前报未闭合双引号；沿用 V697，对 Windows MATLAB 直接使用 PowerShell + WSL UNC 字面路径 | V697,T599 |
| B626 | 2026-07-30 | MATLAB/EIDORS 已加载 PyEIDORS 三维包与 40 个测量后，验收报告调用不存在的 `eidors_version` 函数而在收尾失败；EIDORS 3.12 的版本 API 实为 `eidors_obj('eidors_version')` | T599 |
| B627 | 2026-07-30 | T599 全改动发布门禁发现 4 个早期互操作修改文件尚未按 Ruff 标准布局，尽管 `git diff --check` 已通过；既有格式门禁足够，机械排版后须对全部改动 Python 文件原样重跑 format/check | T599 |
| B628 | 2026-07-30 | 完整 unit suite 发现 4 个旧 CEM 基准 payload 通过公共 `STANDARD_INTEROP_FORMAT` 常量自称 Geometry v2，却仍只提供 v1 字段，严格验证器据此拒绝；既有 v1 生成器必须显式使用 `LEGACY_INTEROP_FORMAT`，不得削弱 v2 必填字段门禁 | V734,V735,T599 |
| B629 | 2026-07-30 | 发布验收的临时 JSON Schema 命令再次因 PowerShell→WSL→bash→Python 多层引号在 Python 启动前失败；沿用 V697，将三包验证、公开 CLI 前向烟测、真实 EIDORS 报告和 schema 语法检查固化为可复跑的单入口 Python 验收脚本 | V697,V737,T599 |
| B630 | 2026-07-30 | `nix build .#pyeidors-complex64-cuda` 完成后安装态 `pyeidors-interop` 报 `ModuleNotFoundError: pyeidors.interop.cli`；Git flake source 排除了尚未跟踪的新 CLI/示例/schema 文件。发布门禁必须在精确暂存本任务新增文件后重建并执行安装态 CLI | V650,V734,T599 |
| B631 | 2026-07-30 | PEM/resistivity 真实捕获脚本直接把 `mk_image(...,'resistivity')` 的嵌套参数字段交给顶层 `fwd_solve`；EIDORS 在 solver 内部映射前先读取 `img.elem_data` 统计帧数，脚本因此失败。参数化图像需先 `data_mapper`，捕获仍必须保存原 `current_params` 并按标准 `convert_img_units` 得到有效 conductivity | V745,T600 |
| B632 | 2026-07-30 | 新增真实 source-semantics 验收脚本首轮 Ruff 门禁发现未使用的 CEM geometry 局部变量；既有静态检查足够，删除冗余绑定并原样重跑格式、lint 和聚焦测试 | T600 |
| B633 | 2026-07-30 | 真实 CEM/PEM/missing-field 捕获与正演均执行后，验收报告的 `np.allclose/isclose` 返回 `numpy.bool_`，严格 `json.dumps(...,allow_nan=False)` 拒绝非原生 bool；报告边界统一归一化所有 check 为 Python `bool`，换新目录完整重跑 | V744,V746,T600 |
| B634 | 2026-07-30 | 非均匀背景回归夹具用单单元 `1×2` 多帧数组，MAT `loadmat(...,squeeze_me=True)` 将其压成长度 2，验证器按一单元空间数据正确拒绝；夹具改为明确两单元、两不同 conductivity，继续验证不取中值的 V745 语义 | V745,T600 |
| B635 | 2026-07-30 | Bridge 复杂测量原计划在非零虚部时回退 `measurements.mat`，但 `real_array_if_zero_imaginary` 抛 `TypeError` 而保存器只捕获 `ValueError`，导致 MAT 分支不可达；仅捕获这两类预期实数降格异常并用复数 roundtrip 测试固定 | V744,T600 |
| B636 | 2026-07-30 | 全仓发布门禁首次把 Python 单引号列表嵌入 PowerShell→WSL→`bash -lc` 单引号命令，测试启动前即引号未闭合；沿用 V697/B629，schema 改用无内嵌脚本的 `python -m json.tool` 串行校验后原样重跑 | V697,T600 |
| B637 | 2026-07-30 | 默认 coverage 门禁完整执行 2493 项并得到 2099 passed/394 skipped/0 test failures，但全仓总覆盖率 `85.88% < 87%`；缺口横跨既有 CUDA/CLI/realtime/实验模块，T600 不扩张范围伪补，保留默认门禁证据并以同套件 `--no-cov` 零失败作为功能回归门禁 | T600 |
| B638 | 2026-07-30 | 原生 PEM 首轮聚焦测试对缺省 `gnd_node/effective_gnd_node` 的 `object(None)` 直接调用 `np.isfinite`，使本应可检查的旧 CEM/legacy 几何在建网格收尾失败；ground 解析必须先判空/数值转换 | V746,V752,T601 |
| B639 | 2026-07-30 | 原生 PEM 通过 `locate_dofs_topological(V,0,vertex)` 建精确 N2E 前未创建 DOLFINx `0→tdim` 连通性，首次点电极设置报 `Missing dims 0->2`；P1 source-vertex→DOF 映射必须显式建拓扑连通性 | V747,V752,T601 |
| B640 | 2026-07-30 | 真实 EIDORS→PyEIDORS PEM 异质场验收中，均匀背景 relL2 为 `4.92e-7`，但首单元异常场为 `2.58e-3`；DOLFINx 建网格后重排了 768 个源单元（源单元 0 变为本地 182），导入器却把源顺序 `target_elem_data` 直接当本地顺序求解。逐单元场必须按 `topology.original_cell_index` 自动映射到运行时本地单元序 | V747,V750,V751,V752,T601 |
| B641 | 2026-07-30 | PyEIDORS→EIDORS 原生 PEM 反向验收的几何、单节点、协议、无投影及 z 不变性全部通过，但 MATLAB 验证器把 SciPy MAT 中的 Unicode 电极模型数组误解析为非 PEM，报告 `electrode_model=cem`；模型分类必须稳健处理 cell/string/char MAT 表示 | V750,V751,V752,T601 |
| B642 | 2026-07-30 | B640 改为使用建网格后自动重排的 local geometry 后，验收函数残留未使用的 source geometry 绑定并触发 Ruff F841；既有静态门禁足够，删除死绑定后原样重跑 | T601 |
| B643 | 2026-07-30 | B640 单元序映射首轮回归忽略 SciPy `squeeze_me=True` 会把单单元逐元场压成标量，导致合法单四面体包的 `background/target/truth_elem_data` 被映射器拒绝；单源单元标量必须恢复成长度 1 的本地场 | V750,V752,T601 |
| B644 | 2026-07-30 | 原生 PEM 完整回归中 32 个旧 PETSc/KSP 分支测试用 `__new__`/bare fixture 构造不经过构造器的 CEM 假模型；新增分支直接读取 `electrode_model`，统一报 `AttributeError`。兼容对象缺省必须视为既有 CEM，不能让新增 PEM 属性破坏旧求解策略测试 | V748,V752,T601 |
| B645 | 2026-07-30 | B644 兼容修复后 69/70 聚焦分支通过，剩余 cache-payload bare fixture 同样缺少新增 `contact_impedance_applicable`；缺省 CEM 必须继续把 z 纳入缓存身份 | V748,V752,T601 |
| B646 | 2026-07-30 | Bridge v3 单刺激×单测量 `meas_matrices=(1,1,n_elec)` 经 `loadmat(...,squeeze_me=True)` 压成 `(n_elec,)`；导入器只恢复二维/三维，误拒绝合法精确协议 | V763,T604 |
| B647 | 2026-07-30 | PEM-only v3 包的空 `cem_face_nodes=(0,dim)` 经 MAT roundtrip 压成 `(0,)`；导入器误 reshape 为 `(1,0)` 并与空 count/id 表判为长度不一致 | V763,T604 |
| B648 | 2026-07-30 | 单通道复数 measurements 经 MAT squeeze 解码成 Python `complex` 标量；Bridge v3 语义哈希只规范 ndarray/np.generic，`json.dumps` 在生成 model_id 时失败 | V764,T604 |
| B649 | 2026-07-30 | 真实 EIDORS 混合反向验收中 CEM 同时恢复 `electrode.nodes` 与 `electrode.faces`；`system_mat_fields` 进入 EIDORS 的双定义 warning 分支并因其日志 level 参数误用在装配前中止。faces 权威的 CEM 必须写 `nodes=[]` | V755,T604 |
| B651 | 2026-07-30 | ModelRegistry 首轮测试在 pytest warning gate 下出现成组 `ResourceWarning: unclosed database`；`sqlite3.Connection` 上下文只提交/回滚而不关闭，资产库每次操作必须由自有 contextmanager 在事务结束后 `finally: close()` | V765,T605 |
| B652 | 2026-07-30 | V765 新回归把既有 `list_models()` 空结果错误断言成 tuple，实际公共契约为 list；连接追踪断言应只依赖空值和全部连接已关闭，⊥ 无关容器类型假设 | V765,T605 |
| B653 | 2026-07-30 | T605 CLI 注册测试使用 `capsys` 读取不到 `_emit(..., stream=sys.stdout)` 定义时绑定的原始流，尽管 fd 层已捕获完整 JSON；CLI 端到端输出验收必须使用 `capfd` | T605 |
| B654 | 2026-07-30 | CLI 模块在 pytest collection 前已保存 stdout 对象，当前全局捕获层下 `capfd.readouterr()` 仍为空；安装态语义不应依赖进程内流替换，改为真实 `python -m ... register` 子进程验收参数/退出码/stdout/落盘 | T605 |
| B655 | 2026-07-30 | T606 工作流测试夹具漏写 geometry 既有必需字段 `background/truth_elem_data/mesh_name/mesh_level/scenario_name`，三流程均在建模前按预期 fail closed；补齐合法夹具，⊥ 放宽几何校验 | V753,V757,T606 |
| B656 | 2026-07-30 | T606 首次受管流程解析错误地读取不存在的 `BridgeV3Package.protocol_layout_hash/protocol_physics_hash` 属性，仿真/数据集均在求解前中止；三个请求身份值统一取已验证的 `RegisteredModel` 索引字段 | V756,V757,T606 |
| B657 | 2026-07-30 | T606 数据集已成功写出，但新测试把 artifact 数组误当裸 HDF5 根 dataset 读取；项目权威布局必须经 `read_hdf5_artifact` 校验 schema/checksum 后访问 `arrays` | V757,T606 |
| B658 | 2026-07-30 | 102 项扩展回归仅旧 V736 测试仍要求预览所得临时 `geometry.mat` 直接 `system.setup`；v3-only 资产契约下必须改为断言未注册/未绑定输入阻断且 setup 零调用 | V753,V756,V757,T606 |
| B659 | 2026-07-30 | 首次绑定数据库重建已进入 ModelContext 路径，却因 `MeasurementDataset` 公共元数据契约仍要求 `stim_pattern/meas_pattern` 而中止；补齐描述字段，custom stim/meas 数组继续作为权威算子 | V758,V760,T607 |
| B650 | 2026-07-30 | fresh EIDORS capture 在 MATLAB 已生成 staging `geometry.mat` 后、正式 manifest/哈希尚未写入前调用 v3 严格包 loader，必然报 missing manifest；捕获必须先拆分 staging geometry/protocol/fields/measurements，再一次性写正式包 | V755,T604 |
| B660 | 2026-07-30 | 教授复现实验的 README 要求从 `.#default` 启动 Jupyter/VS Code，但该 Nix profile 未提供 `ipykernel`/JupyterLab；裸 `/nix/store/.../bin/python` 又缺少 dev-shell 包路径，导致 VS Code 无法选择或启动真实 float64 Kernel | V766 |
| B661 | 2026-07-30 | 教授演示 Notebook 用 `\[`/`\]` 包围块公式，VS Code Markdown 先把反斜杠当转义，导致经典 CEM 方程后半段显示为纯文本；审计同时发现 Robin 说明在定义 `R=A_R^{-1}CQ` 后误写维度不相容的 `Qᵀ(D-CᵀR)Q`，且全部讲解仅英文 | V767 |
| B681 | 2026-07-30 | T609 预览详情扩展重复声明 `contact_impedance` 字典键，Ruff F601 在测试前阻断；逐电极原值必须用独立 `contact_impedance_per_electrode` 键，且资产管理器不得保留未使用导入 | V761,T609 |
| B662 | 2026-07-30 | T609 完整 GUI 互操作回归仅旧 V129 用例仍手工构造 v1 单 `geometry.mat` 并期待 loader 过滤 SciPy 私有键后接受；v3-only 必须改为断言缺 `model/protocol/fields` 明确 fail closed，⊥ 恢复旧包读取 | V753,V761,T609 |
| B663 | 2026-07-30 | T610 真实反向验收把 v3 逐电极 `contact_impedance_applicable` 向量直接用于标量 `&&`，MATLAB 在完成模型回建后拒绝继续；验收器必须对全部 PEM 电极执行向量归约，⊥ 把逐电极 schema 降回标量 | V755,V762,T610 |
| B664 | 2026-07-30 | T610 真实 2D/3D CEM 捕获的权威 `N2E` 为 `n_elec×(n_nodes+n_cem)`，v3 校验器却只接受 PEM 的 `n_elec×n_nodes`，导致四类合法包统一 invalid；`N2E` 必须保留完整系统未知量列并仅要求覆盖全部源节点，⊥ 截断 CEM 电极未知量 | V755,V762,T610 |
| B665 | 2026-07-30 | T610 缺失 CEM `z_contact` 的审计包已带明确 presence=false 与 forward blocker，但 `ElectrodeSpec` 构造仍把它判为结构非法，阻断资产预览/注册；v3 必须允许“可审计、不可正演”，正演入口继续 fail closed，⊥ 猜测或填充阻抗 | V753,V754,V755,V762,T610 |
| B666 | 2026-07-30 | 教授演示只输出 `A_R_shape/C_shape/...` 等变量名而未解释物理意义/维度，且仅加载已认证分数 JSON，未展示有理输入→精确组装→QQ LU→满秩/零残差/双路径同解/哈希认证；各框架也未把网格、电导率和边界电流可视化，无法凭演示直接证明正问题内容与公平设置 | V768 |
| B667 | 2026-07-30 | 教授演示虽已画共享网格、电导率和注流，却只用数值字典汇报求解结果；缺少 Classic/Robin 体电势、电极电压与差值图，无法直观看到正问题“计算出了什么”以及两条等价路径的舍入级差异 | V769 |
| B668 | 2026-07-30 | V769 首轮绘图回归夹具只构造一个解列却沿用含 16 个注流模式的 X01 输入，正确触发解数组维度门禁；回归数据必须为全部 $P$ 个模式提供 $N×P/L×P$ 解，并让 Notebook 正文显式出现“体电势差值”讲解 | V769 |
| B682 | 2026-07-30 | T610 全仓门禁发现旧前向分支测试通过 `EITForwardModel.__new__` 构造全 CEM 假模型，未经过新构造器因而没有 `electrode_specs/cem_electrode_indices`；兼容测试桩必须缺省为全外部 CEM，正常构造路径仍严格要求逐电极语义 | V748,V752,T610 |
| B669 | 2026-07-30 | T610 全仓门禁发现新增 Bridge v3 通道映射运行时指纹使用一个 canonical-JSON `sha256`，但 T90 每文件审计基线尚未登记；协议证明哈希必须作为 C 类 schema-locked digest 同步写入审计文档与回归清单 | V760,V762,T610 |
| B670 | 2026-07-30 | T610 全仓门禁发现四个内部 CEM 精度/连续体基准仍生成 `eidors_pyeidors_bridge_v1`，被 v3-only 导入器正确拒绝；基准 writer 必须直接输出 Geometry v3 的维度、索引、单元和权威边界字段，⊥ 恢复 v1 reader | V753,V762,T610 |
| B671 | 2026-07-31 | 一键包仅用 `command -v` 判断 `tar/zstd/nix` 存在，PATH 前置损坏/旧 shim 被直接执行；Nix cache 子脚本又独立重选 Nix，导致已存在正确系统 Nix 仍可被假 `nix` 劫持 | V770,T611 |
| B672 | 2026-07-31 | packaged launcher继承用户 `PYTHONPATH/PYTHONHOME`，Nix CUDA wrapper又以 `--set-default` 接受外部 `CUDA_HOME/CUDACXX`；用户旧 venv/PyTorch/CUDA 可注入或覆盖封闭运行时 | V771,T611 |
| B673 | 2026-07-31 | 一键安装仅在导入缓存之后、运行 doctor 之前清理 host 环境；兼容 host Bash 的旧 `LD_PRELOAD` 可在 Nix 版本探测阶段注入不兼容 glibc，导致正确 Nix 被误报为损坏。外层与内层入口必须在首个 host tool/Nix 子进程前清理环境 | V771,T611 |
| B674 | 2026-07-31 | V317 GUI warm smoke retained pre-AmgX `cuda` expectation after real+GPU route became `cuda-amgx` | V317,T612 |
| B675 | 2026-07-31 | Bridge v3 GUI import smoke kept incomplete `SimpleNamespace` after managed registry began requiring valid package root/manifest/identity | V756,V757,T612 |
| B676 | 2026-07-31 | Runtime diagnostic smoke hard-coded Chinese text under ambient locale; English Nix/WSL locale deterministically failed | V772,T612 |
| B677 | 2026-07-31 | Advertised legacy runner looked for three moved modules at repo root and executed pytest modules as plain Python, yielding missing/false-pass results | V773,T612 |
| B678 | 2026-07-31 | Generic cache `_normalize` passed Python/NumPy complex scalars through to `json.dumps`, raising `TypeError` under complex profiles | V774,T612 |
| B679 | 2026-07-31 | atexit pool cleanup called lock-taking `worker.shutdown()` while request could hold `_lock` blocked in `stdout.readline()` | V146,V775,T612 |
| B680 | 2026-07-31 | `BridgeV3Package.write` reused existing directory without deleting omitted optional artifacts, leaving stale measurements/reconstruction outside new manifest | V753,V776,T612 |
