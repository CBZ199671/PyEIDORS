"""Chinese (Simplified) translation dictionary.

Keys mirror :mod:`eit_app.i18n.en`.  Missing keys fall back to English
automatically (handled by :class:`eit_app.i18n.translator.Translator`).

文案规范
--------
* 终端用户可见的动词短语使用祈使式 ("开始采集" 而非 "开始采集一帧")
* 菜单 / 按钮的快捷键用 ``(&X)`` 后缀, ``X`` 为大写字母
* 避免冗余副词 ("已经完成" -> "已完成")
* Tab 标签和步骤名统一为 2-4 字的名词, 与英文版并列结构保持对齐
"""

from __future__ import annotations

TRANSLATIONS: dict[str, str] = {
    # ==================================================================
    # Application chrome
    # ==================================================================
    "app.title": "EIT \u5de5\u4f5c\u7ad9",  # EIT 工作站
    # ------------------------------------------------------------------
    # Tab labels
    # ------------------------------------------------------------------
    "tab.hardware": "\u5b9e\u6d4b",  # 实测
    "tab.simulation": "\u4eff\u771f",  # 仿真
    "tab.dataset": "\u6570\u636e\u96c6",  # 数据集
    "tab.database": "\u6570\u636e\u5e93",  # 数据库
    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------
    "menu.file": "\u6587\u4ef6(&F)",  # 文件(&F)
    "menu.file.open_recordings": "\u6253\u5f00\u91c7\u96c6\u6587\u4ef6\u5939(&R)",  # 打开采集文件夹(&R)
    "menu.file.open_output": "\u6253\u5f00\u91cd\u6784\u8f93\u51fa\u6587\u4ef6\u5939(&O)",  # 打开重构输出文件夹(&O)
    "menu.file.exit": "\u9000\u51fa(&X)",  # 退出(&X)
    # ------------------------------------------------------------------
    # Tools menu
    # ------------------------------------------------------------------
    "menu.view": "\u89c6\u56fe(&V)",  # 视图(&V)
    "menu.view.theme_light": "\u6d45\u8272\u4e3b\u9898(&L)",  # 浅色主题(&L)
    "menu.view.theme_dark": "\u6df1\u8272\u4e3b\u9898(&D)",  # 深色主题(&D)
    # 计算精度从"视图"挪到"工具"菜单 —— 它影响计算行为而非视觉，原归类不准确。
    "menu.tools.precision": "\u8ba1\u7b97\u7cbe\u5ea6(&P)",  # 计算精度(&P)
    "menu.tools.precision_float32": "Float32 (\u5feb\u901f, AD 7-bit \u5145\u88d5)",  # Float32 (快速, AD 7-bit 充裕)
    "menu.tools.precision_float64": "Float64 (\u9ad8\u7cbe\u5ea6)",  # Float64 (高精度)
    "main.status.precision_changed": "\u8ba1\u7b97\u7cbe\u5ea6\u5df2\u5207\u6362\u4e3a {mode}\uff0c\u4e0b\u4e00\u6b21\u91c7\u96c6 / \u6c42\u89e3\u751f\u6548\u3002",  # 计算精度已切换为 {mode}，下一次采集 / 求解生效。
    "menu.tools": "\u5de5\u5177(&T)",  # 工具(&T)
    "menu.tools.interop_hub": "EIDORS \u4e92\u64cd\u4f5c(&I)\u2026",  # EIDORS 互操作(&I)…
    "menu.tools.cache_telemetry": "\u7f13\u5b58\u9065\u6d4b(&C)\u2026",  # 缓存遥测(&C)…
    "main.status.recon_running": "\u6b63\u5728\u8fd0\u884c {method}\u2026",  # 正在运行 {method}…
    "main.status.recon_failed": "\u91cd\u6784\u5931\u8d25\uff1a{error}",  # 重构失败：{error}
    "main.status.recon_complete": "\u91cd\u6784\u5b8c\u6210\uff1a{method}",  # 重构完成：{method}
    "main.status.recon_save_ok": "\u8f93\u51fa\u5df2\u4fdd\u5b58\u5230 {folder}",  # 输出已保存到 {folder}
    "main.status.recon_save_failed": "\u4fdd\u5b58\u5931\u8d25\uff1a{error}",  # 保存失败：{error}
    "main.popup.recon_complete.title": "\u91cd\u6784\u5b8c\u6210",  # 重构完成
    "main.popup.recon_complete.text": "\u91cd\u6784\u5df2\u6210\u529f\u4fdd\u5b58\u3002",  # 重构已成功保存。
    "main.popup.recon_complete.informative": "\u8f93\u51fa\u6587\u4ef6\u5939\uff1a\n{folder}",  # 输出文件夹：\n{folder}
    "main.popup.recon_complete.open_folder": "\u6253\u5f00\u6587\u4ef6\u5939",  # 打开文件夹
    "main.popup.recon_complete.close": "\u5173\u95ed",  # 关闭
    "cache.telemetry.title": "\u7f13\u5b58\u9065\u6d4b",  # 缓存遥测
    "cache.telemetry.refresh": "\u5237\u65b0",  # 刷新
    "cache.telemetry.gc": "\u6e05\u7406\u7f13\u5b58",  # 清理缓存
    "cache.telemetry.max_size": "\u76ee\u6807\u5927\u5c0f",  # 目标大小
    "cache.telemetry.include_worker": "Worker \u7f13\u5b58",  # Worker 缓存
    "cache.telemetry.include_legacy": "\u65e7 npz/npy",  # 旧 npz/npy
    "cache.telemetry.idle": "\u7f13\u5b58\u9065\u6d4b\u5df2\u5c31\u7eea\u3002",  # 缓存遥测已就绪。
    "cache.telemetry.refreshing": "\u6b63\u5728\u5237\u65b0\u7f13\u5b58\u9065\u6d4b\u2026",  # 正在刷新缓存遥测…
    "cache.telemetry.gc_running": "\u6b63\u5728\u540e\u53f0\u6e05\u7406\u7f13\u5b58\u2026",  # 正在后台清理缓存…
    "cache.telemetry.summary": (
        "\u78c1\u76d8\u6761\u76ee\uff1a{disk_items} ({disk_mib:.1f} MiB) | "
        "\u5df2\u7d22\u5f15\uff1a{indexed} | Worker\uff1a{workers} | "
        "\u8c03\u5ea6\u5668 \u8fd0\u884c/\u7b49\u5f85\uff1a{active}/{pending}"
    ),
    # ==================================================================
    # Loading / busy overlay messages (shared across plots)
    # ==================================================================
    "hw.live_plot.loading_overlay": "\u7b49\u5f85\u8bbe\u5907\u6570\u636e\u5e27\u2026",  # 等待设备数据帧…
    "hw.reconstruction.loading_overlay": "\u91cd\u6784\u4e2d\u2026",  # 重构中…
    "voltage_plot.loading_overlay": "\u8ba1\u7b97\u8fb9\u754c\u7535\u538b\u4e2d\u2026",  # 计算边界电压中…
    "sim.results.ground_truth_loading": "\u6b63\u95ee\u9898\u6c42\u89e3\u4e2d\u2026",  # 正问题求解中…
    "sim.results.reconstruction_loading": "\u91cd\u6784\u4e2d\u2026",  # 重构中…
    "sim.results.viewer3d_no_data": "\u6682\u65e0 3D \u6570\u636e",  # 暂无 3D 数据
    "sim.results.viewer3d_loading": "\u6e32\u67d3 3D \u573a\u4e2d\u2026",  # 渲染 3D 场中…
    "sim.results.viewer3d_unavailable": "\u672a\u5b89\u88c5 PyVista / VTK\uff0c\u65e0\u6cd5\u663e\u793a 3D \u7f51\u683c",  # 未安装 PyVista / VTK，无法显示 3D 网格
    "sim.results.viewer3d_embedded_disabled": "\u5f53\u524d\u8fd0\u884c\u73af\u5883\u5df2\u7981\u7528\u5d4c\u5165\u5f0f PyVista / VTK\uff0c\u4ee5\u907f\u514d Qt/OpenGL \u5d29\u6e83\uff1b\u6709 3D \u6570\u636e\u65f6\u5c06\u5c1d\u8bd5 PyVista \u79bb\u5c4f\u6e32\u67d3\u3002",  # 当前运行环境已禁用嵌入式 PyVista / VTK，以避免 Qt/OpenGL 崩溃；有 3D 数据时将尝试 PyVista 离屏渲染。
    "sim.results.viewer3d_bad_mesh": "\u7f51\u683c\u4e0d\u662f\u652f\u6301\u7684\u4e09\u7ef4\u56db\u9762\u4f53/\u516d\u9762\u4f53\u4f53\u7f51\u683c",  # 网格不是支持的三维四面体/六面体体网格
    "sim.results.viewer3d_size_mismatch": "\u7535\u5bfc\u7387\u957f\u5ea6\u4e0e\u7f51\u683c\u4e0d\u5339\u914d",  # 电导率长度与网格不匹配
    "sim.results.viewer3d_display": "\u663e\u793a",  # 显示
    "sim.results.viewer3d_display_volume": "\u4f53\u7f51\u683c\u6e32\u67d3",  # 体网格渲染
    "sim.results.viewer3d_display_volume_short": "\u4f53",  # 体
    "sim.results.viewer3d_display_points": "\u70b9\u4e91\u6e32\u67d3",  # 点云渲染
    "sim.results.viewer3d_display_points_short": "\u70b9\u4e91",  # 点云
    "sim.results.viewer3d_anomaly_mode": "\u5f02\u5e38",  # 异常
    "sim.results.viewer3d_anomaly_positive": "\u4ec5\u7a81\u51fa\u9ad8\u4e8e\u80cc\u666f\u4e2d\u503c\u7684\u6b63\u5f02\u5e38",  # 仅突出高于背景中值的正异常
    "sim.results.viewer3d_anomaly_positive_short": "\u6b63",  # 正
    "sim.results.viewer3d_anomaly_negative": "\u4ec5\u7a81\u51fa\u4f4e\u4e8e\u80cc\u666f\u4e2d\u503c\u7684\u8d1f\u5f02\u5e38",  # 仅突出低于背景中值的负异常
    "sim.results.viewer3d_anomaly_negative_short": "\u8d1f",  # 负
    "sim.results.viewer3d_anomaly_absolute": "\u7a81\u51fa\u6b63\u8d1f\u53cc\u5411\u7edd\u5bf9\u504f\u79bb",  # 突出正负双向绝对偏离
    "sim.results.viewer3d_anomaly_absolute_short": "\u7edd\u5bf9",  # 绝对
    "sim.results.viewer3d_opacity": "\u900f\u660e\u5ea6",  # 透明度
    "sim.results.viewer3d_opacity_short": "\u900f\u660e",  # 透明
    "sim.results.viewer3d_highlight": "\u7a81\u51fa\u9009\u5b9a\u5f02\u5e38",  # 突出选定异常
    "sim.results.viewer3d_highlight_short": "\u7a81\u51fa",  # 突出
    "sim.results.viewer3d_wireframe": "\u8f6e\u5ed3\u7ebf",  # 轮廓线
    "sim.results.viewer3d_wireframe_short": "\u8f6e\u5ed3",  # 轮廓
    "sim.results.viewer3d_reset": "\u590d\u4f4d\u89c6\u89d2",  # 复位视角
    "sim.results.viewer3d_reset_short": "\u590d\u4f4d",  # 复位
    # 仿真显示：电极轮廓显示开关。2D 渲染线段，3D 渲染圆柱面贴片。
    "sim.results.electrodes_toggle": "\u663e\u793a\u7535\u6781",  # 显示电极
    "sim.results.electrodes_toggle_short": "\u7535\u6781",  # 电极
    # ------------------------------------------------------------------
    # Help menu + About dialog.
    # ------------------------------------------------------------------
    "menu.help": "\u5e2e\u52a9(&H)",  # 帮助(&H)
    "menu.help.about": "\u5173\u4e8e EIT \u5de5\u4f5c\u7ad9(&A)",  # 关于 EIT 工作站(&A)
    "about.title": "\u5173\u4e8e EIT \u5de5\u4f5c\u7ad9",  # 关于 EIT 工作站
    "about.brand_headline": "EIT \u5de5\u4f5c\u7ad9 \u00b7 \u7535\u963b\u6297\u65ad\u5c42\u6210\u50cf",  # EIT 工作站 · 电阻抗断层成像
    "about.version_line": "\u7248\u672c {version} \u00b7 {build}",  # 版本 {version} · {build}
    "about.body": "\u8de8\u5e73\u53f0 PySide6 \u684c\u9762\u5e94\u7528\uff0c\u8986\u76d6 EIT \u4ece\u786c\u4ef6\u91c7\u96c6\u3001\u4eff\u771f\u3001\u6570\u636e\u96c6\u751f\u6210\u5230\u91cd\u6784\u7684\u5168\u6d41\u7a0b\u3002\u4e2d\u82f1\u53cc\u8bed\uff0c\u9ed8\u8ba4 PyVista \u4e09\u7ef4\u53ef\u89c6\u5316\u3001PETSc / dolfinx \u6c42\u89e3\u3002",  # 跨平台 PySide6 桌面应用，覆盖 EIT 从硬件采集、仿真、数据集生成到重构的全流程。中英双语，默认 PyVista 三维可视化、PETSc / dolfinx 求解。
    "about.credit": "",
    "about.close": "\u5173\u95ed",  # 关闭
    # ------------------------------------------------------------------
    # Language menu
    # ------------------------------------------------------------------
    "menu.language": "\u8bed\u8a00(&L)",  # 语言(&L)
    "menu.language.zh": "\u4e2d\u6587",  # 中文
    "menu.language.en": "English",
    "menu.language.tooltip": "\u5728\u4e2d\u6587\u548c\u82f1\u6587\u4e4b\u95f4\u5207\u6362",  # 在中文和英文之间切换
    # ==================================================================
    # Hardware tab — Step labels on the left QToolBox
    # ==================================================================
    "hw.step.link": "\u6b65\u9aa4\u4e00 \u00b7 \u8fde\u63a5",  # 步骤一 · 连接
    "hw.step.setup": "\u6b65\u9aa4\u4e8c \u00b7 \u8bbe\u7f6e",  # 步骤二 · 设置
    "hw.step.acquire": "\u6b65\u9aa4\u4e09 \u00b7 \u91c7\u96c6",  # 步骤三 · 采集
    # ==================================================================
    # Hardware tab — Step 1 Connection panel
    # ==================================================================
    "hw.connection.title": "1. \u8fde\u63a5\u4e0e\u9a8c\u8bc1",  # 1. 连接与验证
    "hw.connection.flow_hint": "\u8bf7\u5148\u9009\u62e9\u4f20\u8f93\u65b9\u5f0f\u5e76\u9a8c\u8bc1\u8bbe\u5907\u8fde\u63a5\u3002",  # 请先选择传输方式并验证设备连接。
    "hw.connection.transport_label": "\u4f20\u8f93\u65b9\u5f0f\uff1a",  # 传输方式：
    "hw.connection.transport.serial": "\u4e32\u53e3",  # 串口
    "hw.connection.transport.relay_4g": "4G \u4e2d\u7ee7",  # 4G 中继
    "hw.connection.port_label": "\u7aef\u53e3\uff1a",  # 端口：
    "hw.connection.scan_button": "\u626b\u63cf",  # 扫描
    "hw.connection.scan_button_tooltip": "\u5237\u65b0\u4e32\u53e3\u5217\u8868",  # 刷新串口列表
    "hw.connection.baud_label": "\u6ce2\u7279\u7387\uff1a",  # 波特率：
    "hw.connection.host_label": "\u670d\u52a1\u5668\u5730\u5740\uff1a",  # 服务器地址：
    "hw.connection.port_spin_label": "\u670d\u52a1\u5668\u7aef\u53e3\uff1a",  # 服务器端口：
    "hw.connection.board_id_label": "\u677f\u5361 ID\uff1a",  # 板卡 ID：
    "hw.connection.user_id_label": "\u7528\u6237 ID\uff1a",  # 用户 ID：
    "hw.connection.connect_button": "\u8fde\u63a5",  # 连接
    "hw.connection.connect_button_tooltip": "\u5efa\u7acb\u8fde\u63a5\u5e76\u9a8c\u8bc1\u8bbe\u5907",  # 建立连接并验证设备
    "hw.connection.disconnect_button": "\u65ad\u5f00",  # 断开
    "hw.connection.port_hint.no_ports": "\u672a\u68c0\u6d4b\u5230\u53ef\u7528\u4e32\u53e3\u3002\u542f\u52a8\u5668\u4f1a\u81ea\u52a8\u68c0\u67e5\u672c\u5730 Linux \u4e32\u53e3\u548c Windows COM \u6865\u63a5\uff1b\u8bf7\u786e\u8ba4 USB \u7ebf\u3001\u9a71\u52a8\u548c\u8bbe\u5907\u4f9b\u7535\u540e\u518d\u70b9\u626b\u63cf\u3002",  # 未检测到可用串口。启动器会自动检查本地 Linux 串口和 Windows COM 桥接；请确认 USB 线、驱动和设备供电后再点扫描。
    "hw.connection.port_hint.scan_prompt": "\u4e3a\u4e86\u52a0\u5feb GUI \u542f\u52a8\uff0c\u4e32\u53e3\u6539\u4e3a\u6309\u9700\u626b\u63cf\u3002\u70b9\u201c\u626b\u63cf\u201d\uff0c\u6216\u70b9\u201c\u8fde\u63a5\u201d\u65f6\u5148\u626b\u63cf\u518d\u6253\u5f00\u8bbe\u5907\u3002",  # 为了加快 GUI 启动，串口改为按需扫描。点“扫描”，或点“连接”时先扫描再打开设备。
    "hw.connection.port_hint.still_no_ports": "\u4ecd\u672a\u68c0\u6d4b\u5230\u53ef\u7528\u4e32\u53e3\uff0c\u6682\u4e0d\u53d1\u8d77\u8fde\u63a5\u3002\u8bf7\u68c0\u67e5 USB \u8fde\u63a5\u3001\u9a71\u52a8\u548c\u8bbe\u5907\u7535\u6e90\u3002",  # 仍未检测到可用串口，暂不发起连接。请检查 USB 连接、驱动和设备电源。
    "hw.connection.port_hint.single_port_bridge": "\u5df2\u81ea\u52a8\u9009\u4e2d\u552f\u4e00\u4e32\u53e3\uff1a{port}\u3002\u8fde\u63a5\u65f6\u4f1a\u81ea\u52a8\u4f7f\u7528 Windows \u4e3b\u673a\u4e32\u53e3\u6865\u63a5\u3002",  # 已自动选中唯一串口：{port}。连接时会自动使用 Windows 主机串口桥接。
    "hw.connection.port_hint.single_port": "\u5df2\u81ea\u52a8\u9009\u4e2d\u552f\u4e00\u4e32\u53e3\uff1a{port}\u3002",  # 已自动选中唯一串口：{port}。
    "hw.connection.port_hint.multi_port_bridge": "\u68c0\u6d4b\u5230 {count} \u4e2a\u4e32\u53e3\uff0c\u5f53\u524d\u9009\u62e9 {port}\u3002\u8fde\u63a5\u65f6\u4f1a\u81ea\u52a8\u4f7f\u7528 Windows \u4e3b\u673a\u4e32\u53e3\u6865\u63a5\u3002",  # 检测到 {count} 个串口，当前选择 {port}。连接时会自动使用 Windows 主机串口桥接。
    "hw.connection.port_hint.multi_port": "\u68c0\u6d4b\u5230 {count} \u4e2a\u4e32\u53e3\uff0c\u8bf7\u786e\u8ba4\u5e76\u9009\u62e9\u786c\u4ef6\u5bf9\u5e94\u7aef\u53e3\u3002",  # 检测到 {count} 个串口，请确认并选择硬件对应端口。
    "hw.connection.relay_hint.dynamic": "4G \u4e2d\u7ee7\u5c06\u8fde\u63a5\u5230 {host}:{port}\uff1b\u70b9\u51fb\u8fde\u63a5\u524d\u4f1a\u5148\u505a\u670d\u52a1\u5668\u53ef\u8fbe\u6027\u68c0\u67e5\u3002",  # 4G 中继将连接到 {host}:{port}；点击连接前会先做服务器可达性检查。
    # ==================================================================
    # Hardware tab — Step 2 Control panel
    # ==================================================================
    "hw.control.title": "2. \u53c2\u6570\u8bbe\u7f6e\u4e0e\u8bca\u65ad",  # 2. 参数设置与诊断
    "hw.control.power_header": "\u6d4b\u91cf\u7535\u6e90",  # 测量电源
    "hw.control.power_on_button": "\u5f00\u542f\u7535\u6e90",  # 开启电源
    "hw.control.power_off_button": "\u5173\u95ed\u7535\u6e90",  # 关闭电源
    "hw.control.power_hint": "\u7535\u6e90\u5f00\u5173\u76f4\u63a5\u63a7\u5236\u677f\u5361\u4f9b\u7535\uff1b\u5355\u70b9\u6d4b\u8bd5\u4ec5\u7528\u4e8e\u529f\u80fd\u9a8c\u8bc1\u3002",  # 电源开关直接控制板卡供电；单点测试仅用于功能验证。
    "hw.control.layout_header": "\u786c\u4ef6\u5e03\u5c40",  # 硬件布局
    "hw.control.rotate_meas_check": "\u6d4b\u91cf\u968f\u6fc0\u52b1\u65cb\u8f6c",  # 测量随激励旋转
    "hw.control.use_meas_current_check": "\u6d4b\u91cf\u6fc0\u52b1\u76f8\u5173\u7535\u6781",  # 测量激励相关电极
    "hw.control.setup_header": "\u6d4b\u91cf\u53c2\u6570",  # 测量参数
    "hw.control.frequency_label": "\u9891\u7387",  # 频率
    "hw.control.freq_apply_button": "\u5e94\u7528",  # 应用
    "hw.control.stim_amp_label": "\u6fc0\u52b1\u7535\u6d41",  # 激励电流
    "hw.control.stim_apply_button": "\u5e94\u7528",  # 应用
    "hw.control.voltage_gain_label": "\u7535\u538b\u589e\u76ca",  # 电压增益
    "hw.control.vamp_apply_button": "\u5e94\u7528",  # 应用
    "hw.control.diag_header": "\u8bca\u65ad",  # 诊断
    "hw.control.spt_button": "\u5355\u70b9\u6d4b\u8bd5",  # 单点测试
    "hw.control.impedance_button": "\u963b\u6297\u6d4b\u91cf",  # 阻抗测量
    "hw.control.layout_grid.mode": "\u6a21\u5f0f",  # 模式
    "hw.control.layout_grid.elec_ring": "\u6bcf\u73af\u7535\u6781\u6570",  # 每环电极数
    "hw.control.layout_grid.rings": "\u73af\u6570",  # 环数
    "hw.control.layout_grid.stim_pattern": "\u6fc0\u52b1\u6a21\u5f0f",  # 激励模式
    "hw.control.layout_grid.meas_pattern": "\u6d4b\u91cf\u6a21\u5f0f",  # 测量模式
    "hw.control.layout_grid.extra_neighbors": "\u989d\u5916\u6392\u9664\u90bb\u5c45\u6570",  # 额外排除邻居数
    "hw.control.cem_grid.radius": "\u534a\u5f84",  # 半径
    "hw.control.cem_grid.elec_length": "\u7535\u6781\u957f\u5ea6",  # 电极长度
    "hw.control.cem_grid.contact_z": "\u63a5\u89e6\u963b\u6297",  # 接触阻抗
    # ==================================================================
    # Hardware tab — Step 3 Acquisition panel
    # ==================================================================
    "hw.acquisition.title": "3. \u91c7\u96c6\u4e0e\u5f55\u5236",  # 3. 采集与录制
    "hw.acquisition.flow_hint": "\u8bf7\u8bbe\u7f6e\u4fdd\u5b58\u8def\u5f84\u548c\u91c7\u96c6\u8ba1\u5212\uff0c\u7136\u540e\u542f\u52a8\u91c7\u96c6\u3002",  # 请设置保存路径和采集计划，然后启动采集。
    "hw.acquisition.record_header": "\u5f55\u5236\u8bbe\u7f6e",  # 录制设置
    "hw.acquisition.save_to_label": "\u4fdd\u5b58\u5230\uff1a",  # 保存到：
    "hw.acquisition.dir_placeholder": "\u8f93\u51fa\u76ee\u5f55\u2026",  # 输出目录…
    "hw.acquisition.browse_button": "\u6d4f\u89c8\u2026",  # 浏览…
    "hw.acquisition.record_check": "\u5f55\u5236\u5230\u672c\u5730",  # 录制到本地
    "hw.acquisition.plan_header": "\u91c7\u96c6\u8ba1\u5212",  # 采集计划
    "hw.acquisition.timed_interval_check": "\u5b9a\u65f6\u95f4\u9694",  # 定时间隔
    "hw.acquisition.interval_label": "\u95f4\u9694\uff1a",  # 间隔：
    "hw.acquisition.count_label": "\u91c7\u96c6\u6b21\u6570\uff1a",  # 采集次数：
    "hw.acquisition.count_continuous": "\u8fde\u7eed",  # 连续
    "hw.acquisition.freq_step_check": "\u8de8\u9891\u626b\u9891\u91c7\u96c6",  # 跨频扫频采集
    "hw.acquisition.start_freq_label": "\u8d77\u59cb\u9891\u7387\uff1a",  # 起始频率：
    "hw.acquisition.end_freq_label": "\u7ed3\u675f\u9891\u7387\uff1a",  # 结束频率：
    "hw.acquisition.plan_hint": "\u91c7\u96c6\u6b21\u6570 = 0 \u8868\u793a\u65e0\u9650\u8fde\u7eed\u91c7\u96c6\uff1b\u8bbe\u4e3a\u5927\u4e8e 0 \u65f6\uff0c\u5c06\u6267\u884c\u6709\u9650\u6b21\u91c7\u96c6\u5e76\u5728\u5b8c\u6210\u540e\u81ea\u52a8\u505c\u6b62\u3002",  # 采集次数 = 0 表示无限连续采集；设为大于 0 时，将执行有限次采集并在完成后自动停止。
    "hw.acquisition.action_header": "\u91c7\u96c6\u64cd\u4f5c",  # 采集操作
    "hw.acquisition.start_button": "\u5f00\u59cb",  # 开始
    "hw.acquisition.start_button_tooltip": "\u6309\u5f53\u524d\u8ba1\u5212\u5f00\u59cb\u91c7\u96c6",  # 按当前计划开始采集
    "hw.acquisition.single_frame_button": "\u5355\u5e27\u91c7\u96c6",  # 单帧采集
    "hw.acquisition.single_frame_button_tooltip": "\u91c7\u96c6\u4e00\u5e27\u6570\u636e",  # 采集一帧数据
    "hw.acquisition.stop_button": "\u505c\u6b62",  # 停止
    "hw.acquisition.stop_button_tooltip": "\u505c\u6b62\u5f53\u524d\u91c7\u96c6",  # 停止当前采集
    "hw.acquisition.frames_acquired_label": "\u5df2\u91c7\u96c6\u5e27\u6570\uff1a",  # 已采集帧数：
    "hw.acquisition.file_dialog_title": "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55",  # 选择输出目录
    # ==================================================================
    # Hardware tab — Session summary footer
    # ==================================================================
    "hw.summary.title": "\u4f1a\u8bdd\u6458\u8981",  # 会话摘要
    "hw.summary.field.identity": "\u8eab\u4efd\uff1a",  # 身份：
    "hw.summary.field.transport": "\u4f20\u8f93\uff1a",  # 传输：
    "hw.summary.field.layout": "\u5e03\u5c40\uff1a",  # 布局：
    "hw.summary.field.drive": "\u6fc0\u52b1\uff1a",  # 激励：
    "hw.summary.field.record": "\u5f55\u5236\u8def\u5f84\uff1a",  # 录制路径：
    "hw.summary.field.plan": "\u8ba1\u5212\uff1a",  # 计划：
    "hw.summary.indicator.link": "\u8fde\u63a5",  # 连接
    "hw.summary.indicator.power": "\u7535\u6e90",  # 电源
    "hw.summary.indicator.record": "\u5f55\u5236",  # 录制
    "hw.summary.indicator.acq": "\u91c7\u96c6",  # 采集
    # Banner — per-state title / detail / action
    "hw.summary.banner.link_down.title": "\u672a\u8fde\u63a5",  # 未连接
    "hw.summary.banner.link_down.detail": "\u6682\u65e0\u5df2\u9a8c\u8bc1\u7684\u8bbe\u5907\u8fde\u63a5\u3002",  # 暂无已验证的设备连接。
    "hw.summary.banner.link_down.action": "\u9009\u62e9\u4f20\u8f93\u65b9\u5f0f\uff0c\u7136\u540e\u70b9\u51fb\u8fde\u63a5\u5e76\u9a8c\u8bc1\u3002",  # 选择传输方式，然后点击连接并验证。
    "hw.summary.banner.fault.title": "\u94fe\u8def\u6545\u969c",  # 链路故障
    "hw.summary.banner.fault.detail": "\u94fe\u8def\u5904\u4e8e\u9519\u8bef\u72b6\u6001\uff0c\u9700\u8981\u64cd\u4f5c\u5458\u5904\u7406\u3002",  # 链路处于错误状态，需要操作员处理。
    "hw.summary.banner.fault.action": "\u65ad\u5f00\u8fde\u63a5\uff0c\u68c0\u67e5\u4f20\u8f93\u8bbe\u7f6e\u540e\u91cd\u65b0\u9a8c\u8bc1\u3002",  # 断开连接，检查传输设置后重新验证。
    "hw.summary.banner.verifying.title": "\u6b63\u5728\u9a8c\u8bc1\u94fe\u8def",  # 正在验证链路
    "hw.summary.banner.verifying.detail": "\u5de5\u4f5c\u7ad9\u6b63\u5728\u63a2\u6d4b\u8bbe\u5907\u5e76\u8bfb\u53d6\u534f\u8bae\u80fd\u529b\u3002",  # 工作站正在探测设备并读取协议能力。
    "hw.summary.banner.verifying.action": "\u8bf7\u7b49\u5f85\u94fe\u8def\u9a8c\u8bc1\u5b8c\u6210\u3002",  # 请等待链路验证完成。
    "hw.summary.banner.acquiring.title": "\u6b63\u5728\u91c7\u96c6",  # 正在采集
    "hw.summary.banner.acquiring.detail": "\u6b63\u5728\u4ece\u5f53\u524d\u4f20\u8f93\u94fe\u8def\u91c7\u96c6\u5e27\u3002",  # 正在从当前传输链路采集帧。
    "hw.summary.banner.acquiring.action": "\u67e5\u770b\u5b9e\u65f6\u56fe\u6216\u5b8c\u6210\u540e\u505c\u6b62\u91c7\u96c6\u3002",  # 查看实时图或完成后停止采集。
    "hw.summary.banner.acquiring_recording.title": "\u6b63\u5728\u91c7\u96c6 + \u5f55\u5236",  # 正在采集 + 录制
    "hw.summary.banner.acquiring_recording.detail": "\u5e27\u6b63\u5728\u91c7\u96c6\u5e76\u5199\u5165\u5f53\u524d\u4f1a\u8bdd\u3002",  # 帧正在采集并写入当前会话。
    "hw.summary.banner.acquiring_recording.action": "\u67e5\u770b\u5165\u6765\u5e27\u6216\u5b8c\u6210\u540e\u505c\u6b62\u91c7\u96c6\u3002",  # 查看入来帧或完成后停止采集。
    "hw.summary.banner.ready_simulator.title": "\u5373\u53ef\u91c7\u96c6",  # 即可采集
    "hw.summary.banner.ready_simulator.detail": "\u6a21\u62df\u5668\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u53ef\u7acb\u5373\u5f00\u59cb\u751f\u6210\u5e27\u3002",  # 模拟器链路已验证，可立即开始生成帧。
    "hw.summary.banner.ready_simulator.action": "\u5f00\u59cb\u8fde\u7eed\u91c7\u96c6\u6216\u5355\u5e27\u91c7\u96c6\u3002",  # 开始连续采集或单帧采集。
    "hw.summary.banner.ready_record_armed.title": "\u5c31\u7eea + \u5f85\u5f55\u5236",  # 就绪 + 待录制
    "hw.summary.banner.ready_record_armed.detail": "\u8bbe\u5907\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u6d4b\u91cf\u7535\u6e90\u5df2\u5f00\uff0c\u4e0b\u4e00\u6b21\u91c7\u96c6\u5c06\u4fdd\u5b58\u3002",  # 设备链路已验证，测量电源已开，下一次采集将保存。
    "hw.summary.banner.ready_record_armed.action": "\u5f00\u59cb\u91c7\u96c6\u5e76\u5f55\u5236\u4e0b\u4e00\u4f1a\u8bdd\u3002",  # 开始采集并录制下一会话。
    "hw.summary.banner.ready.title": "\u5373\u53ef\u91c7\u96c6",  # 即可采集
    "hw.summary.banner.ready.detail": "\u8bbe\u5907\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u6d4b\u91cf\u7535\u6e90\u5df2\u5f00\u3002",  # 设备链路已验证，测量电源已开。
    "hw.summary.banner.ready.action": "\u5f00\u59cb\u8fde\u7eed\u91c7\u96c6\u6216\u5355\u5e27\u91c7\u96c6\u3002",  # 开始连续采集或单帧采集。
    "hw.summary.banner.link_verified_armed.title": "\u94fe\u8def\u5df2\u9a8c\u8bc1",  # 链路已验证
    "hw.summary.banner.link_verified_armed.detail": "\u94fe\u8def\u5df2\u9a8c\u8bc1\u4e14\u5f55\u5236\u5df2\u5f85\u547d\uff0c\u4f46\u6d4b\u91cf\u7535\u6e90\u5c1a\u672a\u786e\u8ba4\u5f00\u542f\u3002",  # 链路已验证且录制已待命，但测量电源尚未确认开启。
    "hw.summary.banner.link_verified_armed.action": "\u786e\u8ba4\u786c\u4ef6\u5c31\u7eea\u540e\u5f00\u542f\u6d4b\u91cf\u7535\u6e90\uff0c\u7136\u540e\u5f00\u59cb\u91c7\u96c6\u3002",  # 确认硬件就绪后开启测量电源，然后开始采集。
    "hw.summary.banner.link_verified.title": "\u94fe\u8def\u5df2\u9a8c\u8bc1",  # 链路已验证
    "hw.summary.banner.link_verified.detail": "\u8bbe\u5907\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u7b49\u5f85\u6d4b\u91cf\u7535\u6e90\u6216\u4e0b\u4e00\u6b65\u8bbe\u7f6e\u3002",  # 设备链路已验证，等待测量电源或下一步设置。
    "hw.summary.banner.link_verified.action": "\u786e\u8ba4\u786c\u4ef6\u5c31\u7eea\u540e\u5f00\u542f\u6d4b\u91cf\u7535\u6e90\uff0c\u7136\u540e\u5f00\u59cb\u91c7\u96c6\u3002",  # 确认硬件就绪后开启测量电源，然后开始采集。
    # Indicator short chips
    "hw.summary.chip.link.down": "\u65ad\u5f00",  # 断开
    "hw.summary.chip.link.check": "\u9a8c\u8bc1\u4e2d",  # 验证中
    "hw.summary.chip.link.ok": "\u5df2\u8fde",  # 已连
    "hw.summary.chip.link.fault": "\u6545\u969c",  # 故障
    "hw.summary.chip.link.unk": "\u672a\u77e5",  # 未知
    "hw.summary.chip.power.unk": "\u672a\u77e5",  # 未知
    "hw.summary.chip.power.off": "\u5173",  # 关
    "hw.summary.chip.power.on": "\u5f00",  # 开
    "hw.summary.chip.record.off": "\u5173",  # 关
    "hw.summary.chip.record.arm": "\u5f85\u5f55",  # 待录
    "hw.summary.chip.record.rec": "\u5f55\u5236",  # 录制
    "hw.summary.chip.acq.idle": "\u7a7a\u95f2",  # 空闲
    "hw.summary.chip.acq.run": "\u8fde\u7eed",  # 连续
    "hw.summary.chip.acq.sch": "\u5b9a\u65f6",  # 定时
    "hw.summary.chip.acq.fin": "\u6709\u9650",  # 有限
    "hw.summary.chip.acq.step": "\u626b\u9891",  # 扫频
    "hw.summary.chip.acq.1fr": "\u5355\u5e27",  # 单帧
    # Legacy state keys kept for the default first-run indicator states.
    "hw.summary.state.down": "\u65ad\u5f00",  # 断开
    "hw.summary.state.unknown": "\u672a\u77e5",  # 未知
    "hw.summary.state.off": "\u5173",  # 关
    "hw.summary.state.idle": "\u7a7a\u95f2",  # 空闲
    # ==================================================================
    # Hardware tab — Right-side Frame browser
    # ==================================================================
    "hw.frame_browser.title": "\u5df2\u5f55\u5236\u7684\u5e27",  # 已录制的帧
    "hw.frame_browser.hint": "\u6bcf\u6b21\u91c7\u96c6\u7684\u9996\u5e27\u81ea\u52a8\u4f5c\u4e3a\u53c2\u8003\u3002\u70b9\u51fb\u4efb\u4e00\u5e27\u5e76\u6309\u201c\u8bbe\u4e3a\u53c2\u8003\u201d\u53ef\u8986\u76d6\uff1b\u6700\u65b0\u91c7\u96c6\u7684\u5e27\u59cb\u7ec8\u4f5c\u4e3a\u76ee\u6807\u3002",  # 每次采集的首帧自动作为参考。点击任一帧并按"设为参考"可覆盖；最新采集的帧始终作为目标。
    "hw.frame_browser.count_label": "\u5df2\u5f55\u5236\u5e27\u6570\uff1a{count}",  # 已录制帧数：{count}
    "hw.frame_browser.column.index": "\u5e8f\u53f7",  # 序号
    "hw.frame_browser.column.timestamp": "\u65f6\u95f4\u6233",  # 时间戳
    "hw.frame_browser.column.file": "\u6587\u4ef6",  # 文件
    "hw.frame_browser.set_ref_button": "\u8bbe\u4e3a\u53c2\u8003",  # 设为参考
    "hw.frame_browser.clear_button": "\u6e05\u7a7a\u5217\u8868",  # 清空列表
    # ==================================================================
    # Hardware tab — Live measurement plot
    # ==================================================================
    "hw.live_plot.title": "\u5b9e\u65f6\u6d4b\u91cf\u901a\u9053",  # 实时测量通道
    "hw.live_plot.y_label": "\u7535\u538b (V)",  # 电压 (V)
    "hw.live_plot.x_label_dynamic": "\u6d4b\u91cf\u5e8f\u53f7 (1-{count})",  # 测量序号 (1-{count})
    "hw.live_plot.curve.real": "\u5b9e\u90e8",  # 实部
    "hw.live_plot.curve.imag": "\u865a\u90e8",  # 虚部
    "hw.live_plot.empty_overlay": "\u8fd8\u6ca1\u6709\u5b9e\u65f6\u5e27\u3002\n\u542f\u52a8\u91c7\u96c6\u540e\u5c06\u663e\u793a\u5b9e\u90e8\u4e0e\u865a\u90e8\u3002",  # 还没有实时帧。\n启动采集后将显示实部与虚部。
    # ==================================================================
    # Hardware tab — Reconstruction widget
    # ==================================================================
    "hw.reconstruction.title": "\u91cd\u6784\u56fe\u50cf",  # 重构图像
    "hw.reconstruction.empty_overlay": "\u6682\u65e0\u91cd\u6784",  # 暂无重构
    "hw.reconstruction.error.expect_2d_triangles": "\u5feb\u901f\u91cd\u6784\u89c6\u56fe\u76ee\u524d\u4ec5\u652f\u6301 2D \u4e09\u89d2\u7f51\u683c",  # 快速重构视图目前仅支持 2D 三角网格
    # ------------------------------------------------------------------
    # Hardware tab — Equipotential contour widget
    # ------------------------------------------------------------------
    "hw.equipotential.title": "\u7b49\u52bf\u56fe",  # 等势图
    "hw.equipotential.empty_overlay": "\u6682\u65e0\u7b49\u52bf\u6570\u636e",  # 暂无等势数据
    "hw.equipotential.no_surface": "\u4e09\u7ef4\u7f51\u683c\u8868\u9762\u63d0\u53d6\u5931\u8d25",  # 三维网格表面提取失败
    "hw.equipotential.bad_coords": "\u7f51\u683c\u5750\u6807\u65e0\u6548",  # 网格坐标无效
    "hw.equipotential.size_mismatch": "\u7535\u5bfc\u7387\u4e0e\u7f51\u683c\u4e0d\u5339\u914d\uff1a\u03c3={sigma}, cells={cells}, nodes={nodes}",  # 电导率与网格不匹配：σ={sigma}, cells={cells}, nodes={nodes}
    "hw.equipotential.height_label": "\u9ad8\u5ea6\u7f29\u653e",  # 高度缩放
    "hw.equipotential.reset_button": "\u590d\u4f4d\u89c6\u89d2",  # 复位视角
    # ==================================================================
    # Hardware tab — Boundary voltage fit plot
    # ==================================================================
    "hw.boundary.title": "\u8fb9\u754c\u7535\u538b\u62df\u5408",  # 边界电压拟合
    "hw.boundary.y_label": "\u7535\u538b (V)",  # 电压 (V)
    "hw.boundary.phase_y_label": "\u76f8\u4f4d (deg)",  # 相位 (deg)
    "hw.boundary.x_label_dynamic": "\u8fb9\u754c\u7535\u538b\u5e8f\u53f7 (1-{count})",  # 边界电压序号 (1-{count})
    "hw.boundary.primary.measured": "\u5b9e\u6d4b",  # 实测
    "hw.boundary.primary.ground_truth": "\u771f\u503c",  # 真值
    "hw.boundary.secondary": "\u91cd\u6784\u62df\u5408",  # 重构拟合
    "hw.boundary.empty.hardware": "\u91cd\u6784\u66f4\u65b0\u540e\u5c06\u663e\u793a\u5b9e\u6d4b\u4e0e\u62df\u5408\u7684\u8fb9\u754c\u7535\u538b\u66f2\u7ebf\u3002",  # 重构更新后将显示实测与拟合的边界电压曲线。
    "hw.boundary.empty.simulation": "\u6b63\u95ee\u9898\u6216\u9006\u95ee\u9898\u66f4\u65b0\u540e\u5c06\u663e\u793a\u771f\u503c\u4e0e\u62df\u5408\u7684\u8fb9\u754c\u7535\u538b\u66f2\u7ebf\u3002",  # 正问题或逆问题更新后将显示真值与拟合的边界电压曲线。
    # ==================================================================
    # Shared plot-legend overlay (used across Hardware and Simulation)
    # ==================================================================
    "plot_legend.drag_tooltip": "\u53ef\u62d6\u62fd\u8c03\u6574\u56fe\u4f8b\u4f4d\u7f6e",  # 可拖拽调整图例位置
    # ==================================================================
    # Simulation tab — Step labels and Run Guide footer
    # ==================================================================
    "sim.step.mesh": "\u6b65\u9aa4\u4e00 \u00b7 \u7f51\u683c\u4e0e\u7535\u6781",  # 步骤一 · 网格与电极
    "sim.step.inhom": "\u6b65\u9aa4\u4e8c \u00b7 \u975e\u5747\u5300\u4f53",  # 步骤二 · 非均匀体
    "sim.step.inhom_2d": "\u6b65\u9aa4\u4e8c \u00b7 \u975e\u5747\u5300\u9762",  # 步骤二 · 非均匀面
    "sim.step.inhom_3d": "\u6b65\u9aa4\u4e8c \u00b7 \u975e\u5747\u5300\u4f53",  # 步骤二 · 非均匀体
    "sim.step.forward": "\u6b65\u9aa4\u4e09 \u00b7 \u6b63\u95ee\u9898",  # 步骤三 · 正问题
    "sim.step.inverse": "\u6b65\u9aa4\u56db \u00b7 \u9006\u95ee\u9898",  # 步骤四 · 逆问题
    "sim.runguide.title": "\u6d41\u7a0b\u6307\u5f15",  # 流程指引
    "sim.runguide.step1": "\u5148\u914d\u7f6e\u7f51\u683c\u4e0e\u7535\u6781\uff0c\u518d\u7ef4\u62a4\u5f02\u5e38\u4f53\u5217\u8868\u3002",  # 先配置网格与电极，再维护异常体列表。
    "sim.runguide.step2": "\u8fd0\u884c\u6b63\u95ee\u9898\u540e\u67e5\u770b\u8fb9\u754c\u7535\u538b\u4e0e\u771f\u503c\u56fe\u50cf\u3002",  # 运行正问题后查看边界电压与真值图像。
    "sim.runguide.step3": "\u8fd0\u884c\u9006\u95ee\u9898\u540e\u5728\u53f3\u4fa7\u67e5\u770b\u91cd\u6784\u56fe\u50cf\u4e0e\u8bef\u5dee\u6307\u6807\u3002",  # 运行逆问题后在右侧查看重构图像与误差指标。
    "sim.runguide.hint": "\u4e2d\u592e\u533a\u57df\u7528\u4e8e\u56fe\u50cf\u4e0e\u66f2\u7ebf\u5bf9\u6bd4\u3002",  # 中央区域用于图像与曲线对比。
    # ==================================================================
    # Simulation tab — Step 1 Mesh & Electrodes
    # ==================================================================
    "sim.mesh.title": "\u7f51\u683c\u4e0e\u7535\u6781",  # 网格与电极
    "sim.mesh.hint": "\u914d\u7f6e\u4eff\u771f\u7f51\u683c\u548c\u7535\u6781\u5e03\u5c40\u3002",  # 配置仿真网格和电极布局。
    "sim.mesh.dim.2d": "2D",
    "sim.mesh.dim.3d": "3D",
    "sim.mesh.dimension_label": "\u7ef4\u5ea6\uff1a",  # 维度：
    "sim.mesh.family_label": "3D \u5355\u5143\u7c7b\u578b\uff1a",  # 3D 单元类型：
    "sim.mesh.family.tetra": "\u56db\u9762\u4f53\uff084 \u8282\u70b9\uff09",  # 四面体（4 节点）
    "sim.mesh.family.hex": "\u516d\u9762\u4f53\uff088 \u8282\u70b9\uff0cGPU \u5feb\u901f\uff09",  # 六面体（8 节点，GPU 快速）
    "sim.mesh.size_label": "\u7f51\u683c\u5c3a\u5bf8\uff1a",  # 网格尺寸：
    "sim.mesh.refinement_tooltip": "\u6570\u503c\u8d8a\u5c0f\uff0c\u7f51\u683c\u8d8a\u7ec6\uff08\u5355\u5143\u66f4\u591a\uff09",  # 数值越小，网格越细（单元更多）
    "sim.mesh.radius_label": "\u534a\u5f84\uff1a",  # 半径：
    "sim.mesh.radius_tooltip": "2D \u5706\u5f62\u57df / 3D \u5706\u67f1\u4f53\u7684\u534a\u5f84\uff0c\u5355\u4f4d\u4e3a\u7c73 (m)",  # 2D 圆形域 / 3D 圆柱体的半径，单位为米 (m)
    "sim.mesh.height_label": "\u9ad8\u5ea6\uff1a",  # 高度：
    "sim.mesh.height_tooltip": "3D \u5706\u67f1\u4f53\u7684\u9ad8\u5ea6\uff08m\uff09\uff1b\u7535\u6781\u73af\u4f1a\u81ea\u52a8\u5747\u5300\u5206\u5e03\u5728\u8be5\u9ad8\u5ea6\u533a\u95f4\u5185 (15%-85%)\u3002",  # 3D 圆柱体的高度（m）；电极环会自动均匀分布在该高度区间内 (15%-85%)。
    "sim.mesh.electrodes_label": "\u6bcf\u73af\u7535\u6781\u6570\uff1a",  # 每环电极数：
    "sim.mesh.rings_label": "\u73af\u6570/\u5c42\u6570\uff1a",  # 环数/层数：
    "sim.mesh.electrode_length_label": "2D 电极长度：",
    "sim.mesh.electrode_length_tooltip": "二维边界电极弧长，单位为米；会同步换算为网格电极覆盖率。",
    "sim.mesh.electrode_area_label": "3D 电极面积：",
    "sim.mesh.electrode_area_tooltip": "三维圆柱侧壁单个电极贴片面积，单位为平方米；会同步换算为电极高度比例。",
    "sim.mesh.electrode_layout_label": "3D 电极编号：",
    "sim.mesh.electrode_layout.ring_major": "Ring-major（EIDORS 标准）",
    "sim.mesh.electrode_layout.zigzag": "Zigzag（旧版兼容）",
    "sim.mesh.conductivity_label": "背景 γ/σ：",
    "sim.mesh.contact_impedance_label": "接触阻抗 z：",
    "sim.mesh.complex_admittivity_tooltip": "可输入实数或复导纳，例如 1.0 或 1+0.25j S/m。",
    "sim.mesh.complex_impedance_tooltip": "可输入实数或复接触阻抗，例如 0.01 或 0.01+0.002j Ω·m²。",
    "sim.mesh.patterns_header": "\u6fc0\u52b1\u4e0e\u6d4b\u91cf\u6a21\u5f0f",  # 激励与测量模式
    "sim.mesh.patterns_hint": "\u63a7\u5236\u6b63\u95ee\u9898\u6c42\u89e3\u5668\u5982\u4f55\u751f\u6210\u6fc0\u52b1/\u6d4b\u91cf\u5bf9\u3002\u9006\u95ee\u9898\u91cd\u6784\u590d\u7528\u540c\u4e00\u6a21\u5f0f\u2014\u2014\u8bf7\u4e0e\u786c\u4ef6\u677f\u4fdd\u6301\u4e00\u81f4\u3002",  # 控制正问题求解器如何生成激励/测量对。逆问题重构复用同一模式——请与硬件板保持一致。
    "sim.mesh.drive_value_2d_label": "2D 激励线电流密度：",
    "sim.mesh.drive_value_2d_tooltip": "二维模型使用线电流密度 A/m；边界电压会随该值线性缩放。",
    "sim.mesh.drive_value_3d_label": "3D 总激励电流：",
    "sim.mesh.drive_value_3d_tooltip": "三维模型使用总注入电流，界面单位为 uA，内部按 A 传给求解器。",
    "sim.mesh.measurement_protocol_label": "3D 协议（激励→测量）：",
    "sim.mesh.measurement_protocol.eidors_full_3d": "同层激励 → 全层测量（标准 3D）",
    "sim.mesh.measurement_protocol.layer_local_2p5d": "逐层 2D → 切片/插值 3D（2.5D）",
    "sim.mesh.measurement_protocol.cross_layer_full": "仅跨层激励 → 全层+跨层测量",
    "sim.mesh.measurement_protocol.hybrid_full_3d": "同层+跨层激励 → 全层+跨层测量",
    "sim.mesh.measurement_protocol.custom": "自定义激励/测量矩阵",
    "sim.mesh.measurement_protocol_hint.eidors_full_3d": "在每一层内部按激励模式轮换注入电流；每次激励后，所有层都参与同层电压差测量。适合 EIDORS 标准多层 3D 方案。",
    "sim.mesh.measurement_protocol_hint.layer_local_2p5d": "每一层只使用本层电极完成二维激励和二维测量；不同层结果作为切片，再用于三维显示或插值。不会直接使用跨层电压。",
    "sim.mesh.measurement_protocol_hint.cross_layer_full": "只在相邻层同角度电极之间注入电流；测量包含各层同层电压差，并额外加入上下层电压差。适合复现实验中的纯层间激励硬件，但对小球体的横向定位通常不如混合协议。",
    "sim.mesh.measurement_protocol_hint.hybrid_full_3d": "同时包含每层内部激励和相邻层跨层激励；测量包含各层同层电压差和上下层电压差。信息覆盖更完整，更适合三维小体积内含物重构，但测量数更多、速度更慢。",
    "sim.mesh.measurement_protocol_hint.custom": "手动提供 stim_matrix 和 meas_matrices，用于复现真实硬件接线、固定某层激励或任意特殊采集协议。",
    "sim.mesh.stim_pattern_label": "\u6fc0\u52b1\u6a21\u5f0f\uff1a",  # 激励模式：
    "sim.mesh.meas_pattern_label": "\u6d4b\u91cf\u6a21\u5f0f\uff1a",  # 测量模式：
    "sim.mesh.rotate_meas_check": "\u6d4b\u91cf\u968f\u6fc0\u52b1\u65cb\u8f6c",  # 测量随激励旋转
    "sim.mesh.use_meas_current_check": "\u5305\u542b\u6fc0\u52b1\u76f8\u5173\u7535\u6781",  # 包含激励相关电极
    "sim.mesh.extra_neighbors_label": "\u989d\u5916\u6392\u9664\u90bb\u5c45\u6570\uff1a",  # 额外排除邻居数：
    "sim.mesh.custom_pattern_label": "自定义矩阵 JSON：",
    "sim.mesh.custom_pattern_placeholder": '{"stim_matrix": [[1, -1, 0, 0]], "meas_matrices": [[1, 0, -1, 0]]}',
    "sim.mesh.point_count_hint": "\u9884\u8ba1\u8fb9\u754c\u91c7\u6837\u70b9\u6570\uff1a{count}",  # 预计边界采样点数：{count}
    "sim.results.mode_real": "实值 EIT",
    "sim.results.mode_complex": "复导纳 EIT",
    "sim.results.channel_label": "视图：",
    "sim.results.channel.real": "实部 Re",
    "sim.results.channel.imag": "虚部 Im",
    "sim.results.channel.magnitude": "幅值 |·|",
    "sim.results.channel.phase": "相位 ∠",
    "sim.results.channel.composite": "复合幅相",
    # ==================================================================
    # Simulation tab — Step 2 Inhomogeneities
    # ==================================================================
    "sim.inhom.title": "\u975e\u5747\u5300\u4f53",  # 非均匀体
    "sim.inhom.title_2d": "\u975e\u5747\u5300\u9762",  # 非均匀面
    "sim.inhom.title_3d": "\u975e\u5747\u5300\u4f53",  # 非均匀体
    "sim.inhom.col.shape": "\u5f62\u72b6",  # 形状
    # Single-character header labels keep the table readable inside the
    # ~280 px context pane.  The "(m)" / "(S/m)" units moved to a
    # dedicated hint line above the table — see sim.inhom.units_hint.
    "sim.inhom.col.x": "X",
    "sim.inhom.col.y": "Y",
    "sim.inhom.col.z": "Z",
    # X / Y / Z 三轴的尺寸列。中文工程语境里：X 轴 = 长、Y 轴 = 宽、
    # Z 轴 = 高（2D 只用 长 + 宽，3D 三者全用），与"宽 / 高 / 深"
    # 的旧标签相比更符合阅读习惯。
    "sim.inhom.col.sizex": "\u957f",  # 长
    "sim.inhom.col.sizey": "\u5bbd",  # 宽
    "sim.inhom.col.sizez": "\u9ad8",  # 高
    "sim.inhom.col.conductivity": "\u03c3",  # σ
    "sim.inhom.units_hint": "\u5750\u6807 / \u5b8c\u6574\u5c3a\u5bf8\u5355\u4f4d\uff1am\uff0c\u03c3 \u5355\u4f4d\uff1aS/m",  # 坐标 / 完整尺寸单位：m，σ 单位：S/m
    "sim.inhom.boundary_warning": "\u7b2c {rows} \u9879\u975e\u5747\u5300\u4f53\u8d85\u51fa\u5f85\u6d4b\u57df\uff1b\u6b63\u95ee\u9898\u4ec5\u4f7f\u7528\u57df\u5185\u90e8\u5206\uff0c\u9006\u95ee\u9898\u53ef\u80fd\u51fa\u73b0\u660e\u663e\u4f2a\u5f71\u3002",  # 第 {rows} 项非均匀体超出待测域；正问题仅使用域内部分，逆问题可能出现明显伪影。
    "sim.inhom.add_circle": "+ \u5706\u5f62",  # + 圆形
    "sim.inhom.add_ellipse": "+ \u692d\u5706",  # + 椭圆
    "sim.inhom.add_rectangle": "+ \u77e9\u5f62",  # + 矩形
    "sim.inhom.add_sphere": "+ \u7403\u4f53",  # + 球体
    "sim.inhom.add_ellipsoid": "+ \u692d\u7403",  # + 椭球
    "sim.inhom.add_box": "+ \u957f\u65b9\u4f53",  # + 长方体
    "sim.inhom.remove_button": "\u5220\u9664",  # 删除
    # ==================================================================
    # Simulation tab — Step 3 Forward Problem
    # ==================================================================
    "sim.forward.title": "\u6b63\u95ee\u9898",  # 正问题
    "sim.forward.hint": "\u4ece\u7535\u5bfc\u7387\u5206\u5e03\u8ba1\u7b97\u8fb9\u754c\u7535\u538b\u3002",  # 从电导率分布计算边界电压。
    "sim.forward.noise_label": "\u566a\u58f0\u6c34\u5e73\uff1a",  # 噪声水平：
    "sim.forward.noise_tooltip": "\u76f8\u5bf9\u566a\u58f0\u6c34\u5e73 (0 = \u65e0\u566a\u58f0)",  # 相对噪声水平 (0 = 无噪声)
    "sim.forward.solve_button": "\u6c42\u89e3\u6b63\u95ee\u9898",  # 求解正问题
    "sim.forward.status_solving": "\u6c42\u89e3\u4e2d\u2026",  # 求解中…
    # ==================================================================
    # Simulation tab — Step 4 Inverse Problem
    # ==================================================================
    "sim.inverse.title": "\u9006\u95ee\u9898",  # 逆问题
    "sim.inverse.hint": "\u4ece\u8fb9\u754c\u7535\u538b\u91cd\u6784\u7535\u5bfc\u7387\u5206\u5e03\u3002",  # 从边界电压重构电导率分布。
    "sim.inverse.method_label": "\u65b9\u6cd5\uff1a",  # 方法：
    "sim.inverse.alpha_label": "\u6b63\u5219\u5316 \u03b1\uff1a",  # 正则化 α：
    "sim.inverse.alpha_tooltip": "\u7edd\u5bf9 GN \u4e0e\u8c03\u8bd5 full-GN \u8def\u7531\u4f7f\u7528\u8be5\u503c\uff1bRM/\u5355\u6b65\u8def\u7531\u4e0d\u4f7f\u7528\u3002",  # 绝对 GN 与调试 full-GN 路由使用该值；RM/单步路由不使用。
    "sim.inverse.lambda_eff_locked_label": "\u03bb_eff\uff08\u9501\u5b9a\uff09\uff1a",  # λ_eff（锁定）：
    "sim.inverse.lambda_eff_locked_tooltip": "\u5355\u6b65/RM \u8def\u7531\u9ed8\u8ba4\u56fa\u5b9a \u03bb_eff=1e-2\uff1b\u516c\u5f0f\u4f7f\u7528 hp^2 RtR\uff0chp=0.1\u3002\u5982\u9700\u4fee\u6539\uff0c\u8bf7\u52fe\u9009\u9ad8\u7ea7\u81ea\u5b9a\u4e49\u5e76\u91cd\u5efa RM\u3002",  # 单步/RM 路由默认固定 λ_eff=1e-2；公式使用 hp^2 RtR，hp=0.1。如需修改，请勾选高级自定义并重建 RM。
    "sim.inverse.lambda_eff_custom_label": "\u03bb_eff\uff08\u81ea\u5b9a\u4e49\uff09\uff1a",  # λ_eff（自定义）：
    "sim.inverse.custom_lambda_check": "\u9ad8\u7ea7\uff1a\u81ea\u5b9a\u4e49 \u03bb_eff",  # 高级：自定义 λ_eff
    "sim.inverse.custom_lambda_tooltip": "\u52fe\u9009\u540e\u4f7f\u7528\u8f93\u5165\u7684 \u03bb_eff \u6784\u5efa/\u52a0\u8f7d\u72ec\u7acb RM artifact\uff1b\u9996\u6b21\u8fd0\u884c\u4f1a\u51b7\u6784\u5efa\uff0c\u901f\u5ea6\u660e\u663e\u66f4\u6162\u3002",  # 勾选后使用输入的 λ_eff 构建/加载独立 RM artifact；首次运行会冷构建，速度明显更慢。
    "sim.inverse.artifact_weight_label": "Artifact weight\uff1a",  # Artifact weight：
    "sim.inverse.artifact_weight_tooltip": "GREIT \u6743\u91cd\u5b58\u5728 HDF5 artifact \u4e2d\uff1b\u8be5\u503c\u4e0d\u4f5c\u4e3a \u03b1 \u4f7f\u7528\u3002",  # GREIT 权重存在 HDF5 artifact 中；该值不作为 α 使用。
    "sim.inverse.artifact_nf1_tooltip": "GREIT \u4f7f\u7528 EIDORS NF=1 \u81ea\u52a8\u641c\u7d22 weight\uff1b\u8be5\u503c\u4e0d\u4f5c\u4e3a \u03b1 \u4f7f\u7528\uff0c\u51b7\u6784\u5efa\u4f1a\u66f4\u6162\u3002",  # GREIT 使用 EIDORS NF=1 自动搜索 weight；该值不作为 α 使用，冷构建会更慢。
    "sim.inverse.greit.group_title": "GREIT \u9ad8\u7ea7\u53c2\u6570",  # GREIT 高级参数
    "sim.inverse.greit.desired_label": "desired image\uff1a",  # desired image：
    "sim.inverse.greit.desired.center": "\u4e2d\u5fc3\u91c7\u6837",  # 中心采样
    "sim.inverse.greit.desired.gauss": "\u9ad8\u65af\u79ef\u5206",  # 高斯积分
    "sim.inverse.greit.desired.adaptive_gauss": "\u81ea\u9002\u5e94\u79ef\u5206",  # 自适应积分
    "sim.inverse.greit.desired.sobol_qmc": "Sobol-QMC",  # Sobol-QMC
    "sim.inverse.greit.target_count_label": "\u8bad\u7ec3\u76ee\u6807\u6570\uff1a",  # 训练目标数：
    "sim.inverse.greit.target_count_tooltip": "0 \u8868\u793a\u81ea\u52a8\u6839\u636e\u7535\u6781\u6570\u9009\u62e9 rec grid\uff1b\u6570\u91cf\u8d8a\u5927\uff0c\u51b7\u6784\u5efa\u8d8a\u6162\u3002",  # 0 表示自动根据电极数选择 rec grid；数量越大，冷构建越慢。
    "sim.inverse.greit.target_size_label": "\u76ee\u6807\u534a\u5f84 (R\u6bd4\u4f8b)\uff1a",  # 目标半径 (R比例)：
    "sim.inverse.greit.target_size_tooltip": "EIDORS/GREIT target_size \u8bed\u4e49\uff1a\u6309\u5f85\u6d4b\u57df\u534a\u5f84 R \u7684\u6bd4\u4f8b\u7ed9\u51fa\uff0c\u4f8b\u5982 0.20 \u4ee3\u8868 0.2R\u3002",  # EIDORS/GREIT target_size 语义：按待测域半径 R 的比例给出，例如 0.20 代表 0.2R。
    "sim.inverse.greit.weight_strategy_label": "\u6743\u91cd\u7b56\u7565\uff1a",  # 权重策略：
    "sim.inverse.greit.weight_strategy.fixed": "\u56fa\u5b9a weight",  # 固定 weight
    "sim.inverse.greit.weight_strategy.eidors_nf1": "EIDORS NF=1 \u81ea\u52a8\u641c\u7d22",  # EIDORS NF=1 自动搜索
    "sim.inverse.greit.weight_strategy_tooltip": "\u56fa\u5b9a weight \u76f4\u63a5\u4f7f\u7528\u4e0b\u65b9\u6570\u503c\uff1bEIDORS NF=1 \u4f1a\u5728\u51b7\u6784\u5efa RM \u65f6\u641c\u7d22 weight\uff0c\u4f7f\u566a\u58f0\u56fe\u6307\u6807\u63a5\u8fd1 1\u3002",  # 固定 weight 直接使用下方数值；EIDORS NF=1 会在冷构建 RM 时搜索 weight，使噪声图指标接近 1。
    "sim.inverse.greit.weight_label": "weight / NF\uff1a",  # weight / NF：
    "sim.inverse.greit.weight_tooltip": "\u7528\u4e8e GREIT RM \u8bad\u7ec3\u7684\u6743\u91cd/\u6b63\u5219\u5f3a\u5ea6\uff1b\u6539\u53d8\u5b83\u4f1a\u9009\u62e9\u6216\u6784\u5efa\u53e6\u4e00\u4e2a artifact\u3002",  # 用于 GREIT RM 训练的权重/正则强度；改变它会选择或构建另一个 artifact。
    "sim.inverse.greit.use_cache_check": "\u4f7f\u7528\u7f13\u5b58 RM",  # 使用缓存 RM
    "sim.inverse.greit.cache_tooltip": "\u52fe\u9009\u65f6\u4f18\u5148\u590d\u7528\u7b7e\u540d\u5b8c\u5168\u4e00\u81f4\u7684 GREIT RM artifact\u3002",  # 勾选时优先复用签名完全一致的 GREIT RM artifact。
    "sim.inverse.greit.rebuild_check": "\u91cd\u5efa RM",  # 重建 RM
    "sim.inverse.greit.rebuild_tooltip": "\u5ffd\u7565\u5df2\u6709 artifact \u5e76\u51b7\u6784\u5efa\u5f53\u524d GREIT RM\uff1b\u8017\u65f6\u660e\u663e\u589e\u52a0\u3002",  # 忽略已有 artifact 并冷构建当前 GREIT RM；耗时明显增加。
    "sim.inverse.greit.cold_build_hint": "\u63d0\u9192\uff1a\u6539\u53d8 desired image\u3001\u76ee\u6807\u6570\u3001\u76ee\u6807\u534a\u5f84\u3001\u6743\u91cd\u7b56\u7565\u6216 weight/NF \u4f1a\u6539\u53d8 GREIT \u7b7e\u540d\uff1b\u9996\u6b21\u8fd0\u884c\u9700\u8981\u51b7\u6784\u5efa RM\uff0cNF=1 \u81ea\u52a8\u641c\u7d22\u4f1a\u66f4\u6162\u3002",  # 提醒：改变 desired image、目标数、目标半径、权重策略或 weight/NF 会改变 GREIT 签名；首次运行需要冷构建 RM，NF=1 自动搜索会更慢。
    "sim.inverse.iterations_label": "\u6700\u5927\u8fed\u4ee3\u6b21\u6570\uff1a",  # 最大迭代次数：
    "sim.inverse.iterations_tooltip": "\u4ec5\u7edd\u5bf9\u6210\u50cf GN \u8def\u7531\u4f7f\u7528\uff1b\u5dee\u5206/RM \u7b97\u6cd5\u662f\u5355\u6b65\u6216\u7f13\u5b58\u77e9\u9635\u8def\u7531\uff0c\u4e0d\u63a5\u53d7\u8be5\u53c2\u6570\u3002",  # 仅绝对成像 GN 路由使用；差分/RM 算法是单步或缓存矩阵路由，不接受该参数。
    "sim.inverse.reconstruct_button": "\u91cd\u6784",  # 重构
    "sim.inverse.save_button": "\u4fdd\u5b58\u7ed3\u679c",  # 保存结果
    "sim.inverse.status_reconstructing": "\u91cd\u6784\u4e2d\u2026",  # 重构中…
    "sim.inverse.method.debug_fine_mesh_noser.tooltip": "\u8c03\u8bd5\u57fa\u7ebf\uff1a\u5728\u5f53\u524d\u7ec6\u7f51\u683c\u4e0a\u51b7\u6784\u5efa dense NOSER \u4e0a\u4e0b\u6587\uff0c\u6bd4 v1 RM \u70ed\u8def\u5f84\u66f4\u6162\u4e14\u66f4\u6613\u788e\u7247\u5316\u3002",  # 调试基线：在当前细网格上冷构建 dense NOSER 上下文，比 v1 RM 热路径更慢且更易碎片化。
    "sim.inverse.method.noser_rm.tooltip": "NOSER RM \u9ed8\u8ba4\u8def\u7531\uff1a\u51b7\u6784\u5efa\u6216\u590d\u7528 HDF5 \u7c97\u9006\u6a21\u578b artifact\uff0c\u7136\u540e\u7528 RM @ dv \u70ed\u8def\u5f84\u91cd\u6784\u3002",  # NOSER RM 默认路由：冷构建或复用 HDF5 粗逆模型 artifact，然后用 RM @ dv 热路径重构。
    "sim.inverse.method.laplace_rm.tooltip": "Laplace RM \u5e73\u6ed1\u8def\u7531\uff1a\u51b7\u6784\u5efa\u6216\u590d\u7528 HDF5 graph-Laplacian artifact\uff0c\u7136\u540e\u7528 RM @ dv \u70ed\u8def\u5f84\u91cd\u6784\u3002",  # Laplace RM 平滑路由：冷构建或复用 HDF5 graph-Laplacian artifact，然后用 RM @ dv 热路径重构。
    "sim.inverse.method.curvature_rm.tooltip": "Curvature RM \u5e73\u6ed1\u8def\u7531\uff1a\u51b7\u6784\u5efa\u6216\u590d\u7528 HDF5 graph-LtL artifact\uff0c\u7136\u540e\u7528 RM @ dv \u70ed\u8def\u5f84\u91cd\u6784\u3002",  # Curvature RM 平滑路由：冷构建或复用 HDF5 graph-LtL artifact，然后用 RM @ dv 热路径重构。
    "sim.inverse.method.pseudo3d_noser_rm.tooltip": "\u4f2a\u4e09\u7ef4\u8def\u7531\uff1a\u5148\u5c06\u7535\u6781\u5e03\u5c40\u6298\u53e0\u4e3a 2D NOSER RM \u5dee\u5206\u91cd\u6784\uff0c\u518d\u628a 2D \u7535\u5bfc\u7387\u56fe\u6324\u51fa\u4e3a 3D \u56db\u9762\u4f53\u663e\u793a\u7f51\u683c\uff1b\u5b83\u4e0d\u662f\u771f\u6b63\u7684 3D CEM \u9006\u95ee\u9898\u6c42\u89e3\u3002",  # 伪三维路线：先将电极布局折叠为 2D NOSER RM 差分重构，再把 2D 电导率图挤出为 3D 四面体显示网格；它不是真正的 3D CEM 逆问题求解。
    "sim.inverse.method.greit.tooltip": "GREIT \u8def\u7531\uff1a\u6839\u636e\u5f53\u524d 2D/3D \u7f51\u683c\u3001\u7535\u6781\u3001\u534f\u8bae\u548c\u9ad8\u7ea7\u8bad\u7ec3\u53c2\u6570\u6784\u5efa/\u590d\u7528 HDF5 artifact\uff0c\u518d\u7528 RM @ dv \u70ed\u8def\u5f84\u91cd\u6784\u3002",  # GREIT 路由：根据当前 2D/3D 网格、电极、协议和高级训练参数构建/复用 HDF5 artifact，再用 RM @ dv 热路径重构。
    "sim.inverse.method.greit3d_rm.tooltip": "GREIT \u65e7\u540d\u517c\u5bb9\u8def\u7531\uff1bGUI \u73b0\u5728\u7edf\u4e00\u663e\u793a\u4e3a greit\u3002",  # GREIT 旧名兼容路由；GUI 现在统一显示为 greit。
    "sim.inverse.method.absolute_gn.tooltip": "\u7edd\u5bf9\u6210\u50cf\uff1a\u76f4\u63a5\u4ece\u76ee\u6807\u8fb9\u754c\u7535\u538b\u4f30\u8ba1\u7edd\u5bf9\u7535\u5bfc\u7387\uff0c\u4e0d\u4f9d\u8d56\u53c2\u8003\u5e27\uff1b\u8fed\u4ee3 full GN \u51b7\u8def\u5f84\uff0c\u9002\u5408\u5c0f\u7f51\u683c\u5bf9\u7167\u3002",  # 绝对成像：直接从目标边界电压估计绝对电导率，不依赖参考帧；迭代 full GN 冷路径，适合小网格对照。
    "sim.inverse.method.debug_full_gn.tooltip": "\u8c03\u8bd5\u57fa\u7ebf\uff1a\u8fed\u4ee3 full GN \u51b7\u8def\u5f84\uff0c\u9002\u5408\u5bf9\u7167\uff0c\u4e0d\u662f\u5b9e\u65f6 RM \u8def\u7531\u3002",  # 调试基线：迭代 full GN 冷路径，适合对照，不是实时 RM 路由。
    # ==================================================================
    # Simulation tab — Right-side Metrics panel
    # ==================================================================
    "sim.metrics.title": "\u7f51\u683c\u4e0e\u8bef\u5dee\u6307\u6807",  # 网格与误差指标
    "sim.metrics.truth_mesh_label": "\u771f\u503c\u7f51\u683c\uff1a",  # 真值网格：
    "sim.metrics.recon_mesh_label": "\u91cd\u6784\u7f51\u683c\uff1a",  # 重构网格：
    "sim.metrics.mesh_value": "\u8282\u70b9 {nodes} / \u5143\u7d20 {elements}",  # 节点 {nodes} / 元素 {elements}
    "sim.metrics.l2_label": "\u76f8\u5bf9 L2 \u8bef\u5dee\uff1a",  # 相对 L2 误差：
    "sim.metrics.correlation_label": "\u76f8\u5173\u7cfb\u6570\uff1a",  # 相关系数：
    "sim.metrics.rmse_label": "\u5747\u65b9\u6839\u8bef\u5dee\uff1a",  # 均方根误差：
    # ==================================================================
    # Simulation tab — Centre results widget
    # ==================================================================
    "sim.results.ground_truth_title": "\u771f\u503c",  # 真值
    "sim.results.reconstruction_title": "\u91cd\u6784\u7ed3\u679c",  # 重构结果
    "sim.results.save_dialog_title": "\u4fdd\u5b58\u4eff\u771f\u7ed3\u679c",  # 保存仿真结果
    "sim.results.save_dialog_filter": "HDF5 \u5305 (*.h5 *.hdf5)",  # HDF5 包 (*.h5 *.hdf5)
    "sim.results.save_status": "\u5df2\u4fdd\u5b58 HDF5 \u5305\u5230 {path}",  # 已保存 HDF5 包到 {path}
    # ==================================================================
    # Dataset Generator tab — Step labels
    # ==================================================================
    "dataset.step.mesh": "\u6b65\u9aa4\u4e00 \u00b7 \u7f51\u683c\u4e0e\u7535\u6781",  # 步骤一 · 网格与电极
    "dataset.step.ranges": "\u6b65\u9aa4\u4e8c \u00b7 \u968f\u673a\u5316\u8303\u56f4",  # 步骤二 · 随机化范围
    "dataset.step.run": "\u6b65\u9aa4\u4e09 \u00b7 \u8f93\u51fa\u4e0e\u8fd0\u884c",  # 步骤三 · 输出与运行
    # ==================================================================
    # Dataset Generator tab — Central workspace blocks
    # ==================================================================
    "dataset.hero.title": "\u6279\u91cf\u6570\u636e\u96c6\u6d41\u6c34\u7ebf",  # 批量数据集流水线
    "dataset.hero.title_text": "\u6309\u6b65\u9aa4\u751f\u6210\u7f51\u683c\u611f\u77e5\u7684\u7535\u5bfc\u7387\u76ee\u6807\u4e0e\u8fb9\u754c\u7535\u538b\u914d\u5bf9\u3002",  # 按步骤生成网格感知的电导率目标与边界电压配对。
    "dataset.hero.hint": "\u4f7f\u7528\u5de6\u4fa7\u7684\u6b65\u9aa4\u5b9a\u4e49\u7f51\u683c\u3001\u968f\u673a\u5316\u8303\u56f4\u548c\u6279\u91cf\u8f93\u51fa\u76ee\u6807\u3002\u53f3\u4fa7\u7684\u6458\u8981\u9762\u677f\u4f1a\u540c\u6b65\u5f53\u524d\u8fd0\u884c\u72b6\u6001\u3002",  # 使用左侧的步骤定义网格、随机化范围和批量输出目标。右侧的摘要面板会同步当前运行状态。
    "dataset.artifacts.title": "\u751f\u6210\u7684\u4ea7\u7269",  # 生成的产物
    "dataset.artifacts.item1": "mesh_info.h5\uff1a\u8282\u70b9\u5750\u6807\u3001\u5355\u5143\u62d3\u6251\u4e0e\u5747\u5300\u7535\u538b",  # mesh_info.h5：节点坐标、单元拓扑与均匀电压
    "dataset.artifacts.item2": "sample_000000.h5\uff1a\u6bcf\u6837\u672c\u7684\u7535\u5bfc\u7387\u4e0e\u8fb9\u754c\u7535\u538b\u914d\u5bf9",  # sample_000000.h5：每样本的电导率与边界电压配对
    "dataset.artifacts.item3": "\u8f93\u51fa\u76ee\u5f55\u6210\u4e3a\u81ea\u5305\u542b\u7684 HDF5 \u6570\u636e\u96c6\u5305",  # 输出目录成为自包含的 HDF5 数据集包
    "dataset.notes.title": "\u4f7f\u7528\u8bf4\u660e",  # 使用说明
    "dataset.notes.item1": "\u6b64\u5904\u7684\u7f51\u683c\u8bbe\u7f6e\u72ec\u7acb\u4e8e\u4ea4\u4e92\u4eff\u771f Tab\u3002",  # 此处的网格设置独立于交互仿真 Tab。
    "dataset.notes.item2": "\u5f62\u72b6\u5f00\u5173\u5b9a\u4e49\u968f\u673a\u5bb6\u65cf\u6c60\uff1b\u5982\u679c\u672a\u52fe\u9009\u5219\u9ed8\u8ba4\u4f7f\u7528\u5706\u5f62\u3002",  # 形状开关定义随机家族池；如果未勾选则默认使用圆形。
    "dataset.notes.item3": "\u566a\u58f0\u5728\u6b63\u95ee\u9898\u6c42\u89e3\u540e\u6dfb\u52a0\uff0c\u786e\u4fdd\u7535\u538b\u6270\u52a8\u4e0e\u6279\u91cf\u8303\u56f4\u5339\u914d\u3002",  # 噪声在正问题求解后添加，确保电压扰动与批量范围匹配。
    # ==================================================================
    # Dataset Generator tab — Step 2 Randomization panel
    # ==================================================================
    "dataset.random.title": "\u968f\u673a\u5316\u8303\u56f4",  # 随机化范围
    "dataset.random.hint": "\u9009\u62e9\u8981\u91c7\u6837\u7684\u5f62\u72b6\uff0c\u4ee5\u53ca\u7528\u4e8e\u7ed8\u5236\u5408\u6210\u7535\u5bfc\u7387\u76ee\u6807\u7684\u6570\u503c\u8303\u56f4\u3002",  # 选择要采样的形状，以及用于绘制合成电导率目标的数值范围。
    "dataset.random.header.shapes": "\u5f62\u72b6\u5bb6\u65cf",  # 形状家族
    "dataset.random.header.count": "\u76ee\u6807\u6570\u91cf",  # 目标数量
    "dataset.random.header.spatial": "\u7a7a\u95f4\u8303\u56f4",  # 空间范围
    "dataset.random.header.conductivity": "\u7535\u5bfc\u7387\u8303\u56f4",  # 电导率范围
    "dataset.random.shape.circle": "\u5706\u5f62",  # 圆形
    "dataset.random.shape.ellipse": "\u692d\u5706",  # 椭圆
    "dataset.random.shape.rectangle": "\u77e9\u5f62",  # 矩形
    "dataset.random.shapes_label": "\u5f62\u72b6\uff1a",  # 形状：
    "dataset.random.n_label": "\u5f02\u5e38\u4f53\u6570\u91cf\uff1a",  # 异常体数量：
    "dataset.random.position_label": "\u4f4d\u7f6e\uff1a",  # 位置：
    "dataset.random.size_label": "\u5c3a\u5bf8\uff1a",  # 尺寸：
    "dataset.random.conductivity_label": "\u03c3 \u8303\u56f4\uff1a",  # σ 范围：
    "dataset.random.background_label": "\u80cc\u666f \u03c3\uff1a",  # 背景 σ：
    "dataset.random.noise_label": "\u566a\u58f0\u6c34\u5e73\uff1a",  # 噪声水平：
    # ==================================================================
    # Dataset Generator tab — Step 3 Output & Run panel
    # ==================================================================
    "dataset.run.title": "\u8f93\u51fa\u4e0e\u8fd0\u884c",  # 输出与运行
    "dataset.run.hint": "\u9009\u62e9\u6570\u636e\u96c6\u5199\u5165\u4f4d\u7f6e\uff0c\u786e\u8ba4\u7f51\u683c\u548c\u8303\u56f4\u540e\u542f\u52a8\u6279\u91cf\u4efb\u52a1\u3002",  # 选择数据集写入位置，确认网格和范围后启动批量任务。
    "dataset.run.samples_label": "\u6837\u672c\u6570\uff1a",  # 样本数：
    "dataset.run.save_to_label": "\u4fdd\u5b58\u5230\uff1a",  # 保存到：
    "dataset.run.dir_placeholder": "\u8f93\u51fa\u76ee\u5f55\u2026",  # 输出目录…
    "dataset.run.browse_button": "\u6d4f\u89c8\u2026",  # 浏览…
    "dataset.run.progress_header": "\u6267\u884c\u8fdb\u5ea6",  # 执行进度
    "dataset.run.status.ready": "\u51c6\u5907\u751f\u6210\u3002",  # 准备生成。
    "dataset.run.status.progress": "\u5df2\u751f\u6210 {current} / {total} \u4e2a\u6837\u672c\u3002",  # 已生成 {current} / {total} 个样本。
    "dataset.run.generate_button": "\u751f\u6210\u6570\u636e\u96c6",  # 生成数据集
    "dataset.run.cancel_button": "\u53d6\u6d88",  # 取消
    "dataset.run.file_dialog_title": "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55",  # 选择输出目录
    # ==================================================================
    # Dataset Generator tab — Right-side Summary panel
    # ==================================================================
    "dataset.summary.title": "\u751f\u6210\u6458\u8981",  # 生成摘要
    "dataset.summary.hint": "\u5728\u542f\u52a8\u751f\u6210\u5668\u524d\u68c0\u67e5\u5f53\u524d\u6279\u91cf\u914d\u7f6e\u3002",  # 在启动生成器前检查当前批量配置。
    "dataset.summary.progress": "\u8fdb\u5ea6\uff1a{current} / {total}",  # 进度：{current} / {total}
    "dataset.summary.state.idle": "\u7a7a\u95f2",  # 空闲
    "dataset.summary.state.generating": "\u751f\u6210\u4e2d",  # 生成中
    "dataset.summary.state.complete": "\u5df2\u5b8c\u6210",  # 已完成
    "dataset.summary.field.output": "\u8f93\u51fa\uff1a",  # 输出：
    "dataset.summary.field.samples": "\u6837\u672c\uff1a",  # 样本：
    "dataset.summary.field.shapes": "\u5f62\u72b6\uff1a",  # 形状：
    "dataset.summary.field.mesh": "\u7f51\u683c\uff1a",  # 网格：
    "dataset.summary.field.electrodes": "\u7535\u6781\u6570\uff1a",  # 电极数：
    "dataset.summary.field.status": "\u72b6\u6001\uff1a",  # 状态：
    # ==================================================================
    # Database tab — Left filter panel
    # ==================================================================
    "db.filters.title": "\u7b5b\u9009",  # 筛选
    "db.filters.hint": "\u6309\u540d\u79f0\u3001\u9891\u7387\u3001\u7535\u6781\u6570\u3001\u6fc0\u52b1\u7535\u6d41\u6216\u65e5\u671f\u68c0\u7d22\u5386\u53f2\u8bb0\u5f55\u3002",  # 按名称、频率、电极数、激励电流或日期检索历史记录。
    "db.filters.name_label": "\u540d\u79f0\uff1a",  # 名称：
    "db.filters.name_placeholder": "tank, test_for_gui \u2026",
    "db.filters.freq_label": "\u9891\u7387 (Hz)\uff1a",  # 频率 (Hz)：
    "db.filters.freq_min_placeholder": "\u6700\u5c0f",  # 最小
    "db.filters.freq_max_placeholder": "\u6700\u5927",  # 最大
    "db.filters.date_any": "\u5168\u90e8",  # 全部
    "db.filters.date_from_label": "\u8d77\u59cb\u65e5\u671f\uff1a",  # 起始日期：
    "db.filters.date_to_label": "\u7ed3\u675f\u65e5\u671f\uff1a",  # 结束日期：
    "db.filters.n_elec_label": "\u7535\u6781\u6570\uff1a",  # 电极数：
    "db.filters.n_elec_min_placeholder": "\u6700\u5c0f",  # 最小
    "db.filters.n_elec_max_placeholder": "\u6700\u5927",  # 最大
    "db.filters.stim_amp_label": "\u6fc0\u52b1\u7535\u6d41 (\u00b5A)\uff1a",  # 激励电流 (µA)：
    "db.filters.stim_amp_min_placeholder": "\u6700\u5c0f",  # 最小
    "db.filters.stim_amp_max_placeholder": "\u6700\u5927",  # 最大
    "db.filters.apply_button": "\u5e94\u7528\u7b5b\u9009",  # 应用筛选
    "db.filters.clear_button": "\u6e05\u7a7a",  # 清空
    "db.filters.refresh_button": "\u5237\u65b0",  # 刷新
    "db.stats.count": "{count} \u4e2a\u4f1a\u8bdd",  # {count} 个会话
    "db.stats.ready": "\u5c31\u7eea",  # 就绪
    "db.stats.backfill_progress": "\u56de\u586b\u4e2d\uff1a{current}/{total}",  # 回填中：{current}/{total}
    "db.stats.backfill_done": "\u56de\u586b\u5b8c\u6210\uff1a\u5171\u5bfc\u5165 {count} \u4e2a\u4f1a\u8bdd\u3002",  # 回填完成：共导入 {count} 个会话。
    # ==================================================================
    # Database tab — Central sessions / frames section
    # ==================================================================
    "db.sessions.title": "\u4f1a\u8bdd",  # 会话
    "db.sessions.col.id": "ID",
    "db.sessions.col.name": "\u540d\u79f0",  # 名称
    "db.sessions.col.started": "\u5f00\u59cb\u65f6\u95f4",  # 开始时间
    "db.sessions.col.n_elec": "\u7535\u6781\u6570",  # 电极数
    "db.sessions.col.frequency": "\u9891\u7387",  # 频率
    "db.sessions.col.stim": "\u6fc0\u52b1 (µA)",  # 激励 (µA)
    "db.sessions.col.gain": "\u589e\u76ca",  # 增益
    "db.sessions.col.frames": "\u5e27\u6570",  # 帧数
    "db.sessions.open_folder_button": "\u6253\u5f00\u6587\u4ef6\u5939",  # 打开文件夹
    "db.sessions.batch_recon_button": "\u6279\u91cf\u91cd\u6784\u2026",  # 批量重构…
    "db.frames.title": "\u5e27",  # 帧
    "db.frames.col.index": "\u5e8f\u53f7",  # 序号
    "db.frames.col.timestamp": "\u65f6\u95f4\u6233",  # 时间戳
    "db.frames.col.file": "\u6587\u4ef6",  # 文件
    "db.frames.selection_hint": "\u9009\u62e9\u4e00\u5e27\uff0c\u518d\u70b9\u51fb\u201c\u8bbe\u4e3a\u53c2\u8003\u201d\u6216\u201c\u8bbe\u4e3a\u76ee\u6807\u201d\u3002",  # 选择一帧，再点击"设为参考"或"设为目标"。
    "db.frames.selection_role.reference": "\u53c2\u8003",  # 参考
    "db.frames.selection_role.target": "\u76ee\u6807",  # 目标
    "db.frames.selection_unset": "{role}\uff1a<\u672a\u9009\u62e9>",  # {role}：<未选择>
    "db.frames.selection_set": "{role}\uff1a#{index}",  # {role}：#{index}
    "db.frames.set_ref_button": "\u8bbe\u4e3a\u53c2\u8003",  # 设为参考
    "db.frames.set_tgt_button": "\u8bbe\u4e3a\u76ee\u6807",  # 设为目标
    "db.frames.reconstruct_button": "\u91cd\u6784\u2026",  # 重构…
    "db.frames.recon_settings_button": "\u91cd\u6784\u53c2\u6570\u2026",  # 重构参数…
    "db.frames.recon_settings_tip": "\u8bbe\u7f6e\u5355\u5e27\u3001\u5dee\u5206\u548c\u6279\u91cf\u91cd\u6784\u5171\u7528\u7684\u6b63/\u9006\u95ee\u9898\u53c2\u6570\u3002",  # 设置单帧、差分和批量重构共用的正/逆问题参数。
    "db.frames.recon_settings_custom_tip": "\u5df2\u4e3a\u6570\u636e\u5e93\u91cd\u6784\u542f\u7528\u81ea\u5b9a\u4e49\u6b63/\u9006\u95ee\u9898\u53c2\u6570\u3002",  # 已为数据库重构启用自定义正/逆问题参数。
    "db.frames.clear_button": "\u6e05\u9664",  # 清除
    # ==================================================================
    # Database tab — Right-side preview panel
    # ==================================================================
    "db.preview.title": "\u5e27\u9884\u89c8",  # 帧预览
    "db.preview.hint": "\u70b9\u51fb\u4efb\u610f\u4e00\u5e27\u5728\u6b64\u9884\u89c8\u5176\u6ce2\u5f62\u3002",  # 点击任意一帧在此预览其波形。
    # ==================================================================
    # Main window — transient status-bar flash messages (Chinese)
    # ==================================================================
    # Connection / transport
    "main.status.port_not_found_scan": "\u672a\u68c0\u6d4b\u5230\u53ef\u7528\u4e32\u53e3\u3002\u8bf7\u68c0\u67e5 USB \u7ebf\u3001\u9a71\u52a8\u4e0e\u8bbe\u5907\u4f9b\u7535\uff0c\u7136\u540e\u70b9\u51fb\u626b\u63cf\u91cd\u8bd5\u3002",
    "main.status.relay_host_empty": "4G Relay \u670d\u52a1\u5668\u5730\u5740\u4e3a\u7a7a\uff0c\u8bf7\u5148\u586b\u5199\u53ef\u8bbf\u95ee\u7684 host\u3002",
    "main.status.verifying.windows_bridge": "\u6b63\u5728\u901a\u8fc7 Windows \u4e3b\u673a\u4e32\u53e3 {port} \u9a8c\u8bc1\u8bbe\u5907\u94fe\u8def\uff0c\u6ce2\u7279\u7387 {baud}\u3002",
    "main.status.verifying.serial": "\u6b63\u5728\u9a8c\u8bc1\u4e32\u53e3\u94fe\u8def\uff1a{port} @ {baud}",
    "main.status.verifying.relay": "\u6b63\u5728\u9a8c\u8bc1 4G Relay \u94fe\u8def\uff1a{host}:{port}",
    "main.status.verifying.generic": "\u6b63\u5728\u9a8c\u8bc1\u8bbe\u5907\u94fe\u8def\u3002",
    "main.status.link_verified": "\u94fe\u8def\u8fde\u63a5\u4e0e\u534f\u8bae\u9a8c\u8bc1\u5df2\u5b8c\u6210\uff0c\u53ef\u6309\u9700\u5f00\u542f\u6d4b\u91cf\u7535\u6e90\u5e76\u5f00\u59cb\u91c7\u96c6\u3002",
    # Acquisition + recording
    "main.error.connection_required": "\u8bf7\u5148\u5b8c\u6210\u8bbe\u5907\u8fde\u63a5\u9a8c\u8bc1\u3002",
    "main.error.port_release_failed": "\u542f\u52a8\u91c7\u96c6\u524d\u672a\u80fd\u91ca\u653e\u63a7\u5236\u4e32\u53e3\uff0c\u8bf7\u91cd\u8bd5\u6216\u91cd\u65b0\u8fde\u63a5\u8bbe\u5907\u3002",
    "main.error.acq_count_zero": "\u6709\u9650\u6b21\u91c7\u96c6\u6216\u5b9a\u65f6\u91c7\u96c6\u9700\u8981\u5c06\u91c7\u96c6\u6b21\u6570\u8bbe\u7f6e\u4e3a\u5927\u4e8e 0\u3002",
    "main.status.single_frame_started": "\u5355\u5e27\u91c7\u96c6\u5df2\u542f\u52a8\uff0c\u91c7\u5230 1 \u5e27\u540e\u5c06\u81ea\u52a8\u505c\u6b62\u3002",
    "main.status.single_frame_done": "\u5355\u5e27\u91c7\u96c6\u5b8c\u6210\u3002",
    "main.status.continuous_started": "\u8fde\u7eed\u91c7\u96c6\u5df2\u542f\u52a8\u3002",
    "main.status.plan_stopped": "\u8ba1\u5212\u91c7\u96c6\u5df2\u505c\u6b62\u3002",
    "main.status.plan_step_done": "\u7b2c {current}/{total} \u6b21\u91c7\u96c6\u5b8c\u6210\uff0c{interval:.1f}s \u540e\u5f00\u59cb\u4e0b\u4e00\u6b21\u3002",
    "main.status.recording_started": "\u5f00\u59cb\u5f55\u5236\uff1a{dir}",
    "main.status.recording_stopped": "\u5f55\u5236\u5df2\u505c\u6b62\uff0c\u5171\u4fdd\u5b58 {count} \u5e27\u3002",
    "main.status.frames_cleared": "\u5df2\u6e05\u7a7a\u5f55\u5236\u5e27\u5217\u8868\u3002",
    "main.status.record_enabled": "\u5df2\u542f\u7528\u5f55\u5236\uff0c\u5f00\u59cb\u91c7\u96c6\u540e\u5c06\u4fdd\u5b58\u5230 {dir}\u3002",
    "main.status.record_path_pending": "\u5f53\u524d\u5f55\u5236\u5df2\u5f00\u59cb\uff0c\u65b0\u4fdd\u5b58\u8def\u5f84\u5c06\u5728\u4e0b\u6b21\u91c7\u96c6\u65f6\u751f\u6548\u3002",
    # Reconstruction pre-warm
    "main.status.prewarming": "\u6b63\u5728\u9884\u70ed\u5b9e\u65f6\u91cd\u6784\u4e0a\u4e0b\u6587\u2026",
    "main.status.prewarm_done": "\u5b9e\u65f6\u91cd\u6784\u4e0a\u4e0b\u6587\u5df2\u9884\u70ed\uff0c\u540e\u7eed\u91c7\u96c6\u5c06\u76f4\u63a5\u8d70\u70ed\u542f\u52a8\u3002",
    "main.status.prewarm_failed": "\u5b9e\u65f6\u91cd\u6784\u9884\u70ed\u5931\u8d25\uff0c\u5c06\u5728\u9700\u8981\u65f6\u91cd\u8bd5\uff1a{reason}",
    "main.status.sim_backend_warm_start": "\u6b63\u5728\u9884\u70ed\u4e09\u7ef4\u540e\u7aef worker\uff08{profile}\uff09\u2026",
    "main.status.sim_backend_warm_done": "\u4e09\u7ef4\u540e\u7aef worker \u5df2\u9884\u70ed\uff1aprofile={profile}, pid={pid}, RSS={rss}, prime={prime_ms:.0f} ms\uff0cPETSc probe={probe}\u3002",
    "main.status.sim_backend_warm_failed": "\u4e09\u7ef4\u540e\u7aef worker \u9884\u70ed\u5931\u8d25\uff08{profile}\uff09\uff1a{reason}",
    "main.status.sim_backend_setup_start": "\u6b63\u5728\u9884\u70ed\u4e09\u7ef4\u6b63\u95ee\u9898 setup\uff08{profile}\uff09\u2026",
    "main.status.sim_backend_setup_done": "\u4e09\u7ef4\u6b63\u95ee\u9898 setup \u5df2\u9884\u70ed\uff1aprofile={profile}, pid={pid}, RSS={rss}, setup={prime_ms:.0f} ms\uff0cPETSc probe={probe}\u3002",
    "main.status.sim_backend_setup_failed": "\u4e09\u7ef4\u6b63\u95ee\u9898 setup \u9884\u70ed\u5931\u8d25\uff08{profile}\uff09\uff1a{reason}",
    # Frame browser / reference / target
    "main.status.reference_updated": "\u53c2\u8003\u5e27\u5df2\u66f4\u65b0\uff1a#{index}",
    "main.status.reference_selected": "\u53c2\u8003\u5e27\u5df2\u9009\u62e9\uff1a#{index}",
    "main.status.target_selected": "\u76ee\u6807\u5e27\u5df2\u9009\u62e9\uff1a#{index}",
    "main.status.frame_preview": "\u663e\u793a\u5e27 #{index} \u7684\u6ce2\u5f62\u6570\u636e",
    # Layout + protocol + power + diagnostics
    "main.status.layout_updated": "\u786c\u4ef6\u5e03\u5c40\u5df2\u66f4\u65b0\uff1a{points} \u4e2a\u8fb9\u754c\u7535\u538b\u70b9\u3002",
    "main.status.protocol_caps": "\u534f\u8bae\u80fd\u529b\uff1a{version}",
    "main.status.spt_result": "\u5355\u70b9\u6d4b\u8bd5\u8fd4\u56de\uff1areal={real:.4f} V, imag={imag:.4f} V",
    "main.status.power_on": "\u6d4b\u91cf\u7535\u6e90\u5df2\u5207\u6362\u4e3a ON\u3002",
    "main.status.power_off": "\u6d4b\u91cf\u7535\u6e90\u5df2\u5207\u6362\u4e3a OFF\u3002",
    "main.status.power_sent": "\u6d4b\u91cf\u7535\u6e90\u547d\u4ee4\u5df2\u53d1\u9001\u3002",
    "main.status.command_sent": "\u547d\u4ee4\u5df2\u53d1\u9001\uff1a{name}",
    "main.status.impedance_done": "\u63a5\u89e6\u963b\u6297\u6d4b\u91cf\u5b8c\u6210\u3002",
    "main.status.impedance_result": "\u63a5\u89e6\u963b\u6297\uff1a{values}",
    # Plan + frequency sweep
    "main.status.plan_started": "\u8ba1\u5212\u91c7\u96c6\u5df2\u542f\u52a8\uff0c\u5171 {count} \u6b21\u3002",
    "main.status.plan_sweep_note": "\u53d8\u9891\u91c7\u96c6\u5df2\u542f\u52a8\uff1a\u5c06\u6309\u4ea4\u9891\u5dee\u5b9e\u65f6\u66f4\u65b0\u6ce2\u5f62\u3001\u8fb9\u754c\u7535\u538b\u4e0e\u91cd\u6784\u663e\u793a\u3002",
    "main.status.plan_step_start": "\u5f00\u59cb\u7b2c {current}/{total} \u6b21\u91c7\u96c6\uff1a{hz} Hz",
    "main.status.plan_complete": "\u8ba1\u5212\u91c7\u96c6\u5b8c\u6210\uff0c\u5171 {count} \u6b21\u3002",
    # Interop hub bridge results
    "main.interop.geometry_generate_failed": "\u65e0\u6cd5\u81ea\u52a8\u751f\u6210 simulation geometry.mat\uff1a{error}",
    "main.interop.export_note_hw_real": "\u5f53\u524d\u5f55\u5236\u5bfc\u51fa\u9ed8\u8ba4\u4f7f\u7528\u5b9e\u90e8\u8fb9\u754c\u7535\u538b\uff0c\u4ee5\u4fbf\u4e0e EIDORS \u5e38\u89c1\u5dee\u5206\u5de5\u4f5c\u6d41\u5151\u5bb9\u3002",
    "main.interop.export_note_hw_no_geom": "\u786c\u4ef6\u9875\u5f53\u524d\u9ed8\u8ba4\u5bfc\u51fa\u5e03\u5c40\u6a21\u677f\uff1b\u82e5\u9700\u8981\u51e0\u4f55\uff0c\u8bf7\u5148\u4ece\u4eff\u771f\u7ed3\u679c\u6216 bridge \u5305\u5bfc\u5165 geometry \u8d44\u4ea7\u3002",
    "main.interop.applied_to_hw": "\u5df2\u5c06 bridge \u914d\u7f6e\u5bfc\u5165\u5230\u786c\u4ef6\u9875\uff1a{dim} | {n_elec} \u7535\u6781/\u73af | {points} \u70b9\u3002",
    "main.interop.applied_to_sim": "\u5df2\u5c06 bridge \u914d\u7f6e\u5bfc\u5165\u5230\u4eff\u771f\u9875\uff1a{dim} | {n_elec} \u7535\u6781/\u73af | {points} \u70b9\u3002",
    "main.interop.applied_to_dataset": "\u5df2\u5c06 bridge \u914d\u7f6e\u5bfc\u5165\u5230\u6570\u636e\u96c6\u9875\uff1a{dim} | {n_elec} \u7535\u6781/\u73af | {points} \u70b9\u3002",
    "main.interop.no_voltage_data": "\u8fd9\u4e2a bridge \u5305\u91cc\u6ca1\u6709\u53ef\u5bfc\u5165\u7684\u8fb9\u754c\u7535\u538b\u6570\u636e\u3002",
    "main.interop.voltage_cached": "\u5df2\u7f13\u5b58\u8fb9\u754c\u7535\u538b\u6570\u636e\u8d44\u4ea7\uff0c\u540e\u7eed\u53ef\u7528\u4e8e\u5bfc\u51fa\u3001\u5bf9\u7167\u6216\u91cd\u6784\u70df\u6d4b\u3002",
    "main.interop.no_geometry": "\u8fd9\u4e2a bridge \u5305\u91cc\u6ca1\u6709 geometry.mat\u3002",
    "main.interop.geometry_cached": "\u5df2\u7f13\u5b58 geometry \u8d44\u4ea7\uff0c\u540e\u7eed\u5bfc\u51fa\u5230 EIDORS \u65f6\u53ef\u76f4\u63a5\u590d\u7528\u3002",
    "main.interop.unknown_target": "\u672a\u77e5\u5bfc\u5165\u76ee\u6807\uff1a{target}",
    "main.interop.smoke_done": "\u4e92\u901a\u70df\u6d4b\u5df2\u5b8c\u6210\u3002",
    # humanize_error_message branches
    "main.hw_error.no_serial_ports": "\u672a\u68c0\u6d4b\u5230\u53ef\u7528\u4e32\u53e3\u3002\u8bf7\u68c0\u67e5 USB \u8fde\u63a5\u3001\u9a71\u52a8\u548c\u8bbe\u5907\u4f9b\u7535\u540e\u91cd\u65b0\u626b\u63cf\u3002",
    "main.hw_error.port_access_denied": "\u4e32\u53e3\u8bbf\u95ee\u88ab\u62d2\u7edd\uff0c\u53ef\u80fd\u88ab\u5176\u4ed6\u7a0b\u5e8f\u5360\u7528\u3002\u8bf7\u5173\u95ed\u5360\u7528\u8fdb\u7a0b\u540e\u91cd\u8bd5\u3002",
    "main.hw_error.windows_port_invalid": "\u4e32\u53e3\u65e0\u6cd5\u914d\u7f6e\u3002\u5f53\u524d\u73af\u5883\u4e2d\u8be5\u7aef\u53e3\u4e0d\u53ef\u7528\uff1b\u8bf7\u4f18\u5148\u4ece\u4e0b\u62c9\u6846\u9009\u62e9\u81ea\u52a8\u68c0\u6d4b\u5230\u7684 COM \u53e3\uff0c\u4e0d\u8981\u624b\u52a8\u586b\u5199 /dev/ttyS*\u3002",
    "main.hw_error.windows_bridge_port_busy": "Windows \u4e32\u53e3\u6865\u63a5\u5931\u8d25\uff1a\u8be5 COM \u53e3\u53ef\u80fd\u4ecd\u88ab\u5176\u4ed6\u7a0b\u5e8f\u5360\u7528\uff1b\u5982\u679c\u4f60\u521a\u5173\u95ed\u672c\u8f6f\u4ef6\uff0c\u8bf7\u7b49\u5f85 1-2 \u79d2\u540e\u91cd\u8bd5\u3002",
    "main.hw_error.windows_bridge_port_missing": "Windows \u4e32\u53e3\u6865\u63a5\u5931\u8d25\uff1a\u5f53\u524d\u627e\u4e0d\u5230\u8fd9\u4e2a COM \u53e3\uff0c\u8bf7\u91cd\u65b0\u63d2\u62d4\u8bbe\u5907\u540e\u518d\u626b\u63cf\u3002",
    "main.hw_error.windows_bridge_generic": "Windows \u4e3b\u673a\u4e32\u53e3\u6865\u63a5\u542f\u52a8\u5931\u8d25\uff0c\u8bf7\u91cd\u65b0\u626b\u63cf\u540e\u91cd\u8bd5\u3002",
    "main.hw_error.relay_host_empty": "4G Relay \u670d\u52a1\u5668\u5730\u5740\u4e3a\u7a7a\uff0c\u8bf7\u586b\u5199\u53ef\u8bbf\u95ee\u7684 host\u3002",
    "main.hw_error.relay_refused": "4G Relay \u670d\u52a1\u5668\u62d2\u7edd\u8fde\u63a5\uff0c\u8bf7\u68c0\u67e5 host/port \u662f\u5426\u6b63\u786e\u4ee5\u53ca\u670d\u52a1\u662f\u5426\u5df2\u542f\u52a8\u3002",
    "main.hw_error.relay_timeout": "4G Relay \u8fde\u63a5\u8d85\u65f6\uff0c\u8bf7\u68c0\u67e5\u7f51\u7edc\u3001\u670d\u52a1\u5668\u5730\u5740\u548c\u76ee\u6807\u8bbe\u5907\u662f\u5426\u5728\u7ebf\u3002",
    # ==================================================================
    # Bottom status bar (persistent chips + FPS / frame counters)
    # ==================================================================
    "status.fps": "\u5e27\u7387\uff1a--",  # 帧率：--
    "status.fps_value": "\u5e27\u7387\uff1a{value:.1f}",  # 帧率：{value:.1f}
    "status.frames": "\u5e27\u6570\uff1a0",  # 帧数：0
    "status.frames_value": "\u5e27\u6570\uff1a{count}",  # 帧数：{count}
    "status.mode.hardware": "\u6a21\u5f0f\uff1a\u5b9e\u6d4b",  # 模式：实测
    "status.mode.simulation": "\u6a21\u5f0f\uff1a\u4eff\u771f",  # 模式：仿真
    "status.mode.dataset": "\u6a21\u5f0f\uff1a\u6570\u636e\u96c6",  # 模式：数据集
    "status.mode.database": "\u6a21\u5f0f\uff1a\u6570\u636e\u5e93",  # 模式：数据库
    "status.mode.other": "\u6a21\u5f0f\uff1a{index}",  # 模式：{index}
    "status.link.connected": "\u94fe\u8def\uff1a\u5df2\u9a8c\u8bc1",  # 链路：已验证
    "status.link.connecting": "\u94fe\u8def\uff1a\u8fde\u63a5\u4e2d",  # 链路：连接中
    "status.link.disconnected": "\u94fe\u8def\uff1a\u65ad\u5f00",  # 链路：断开
    "status.link.error": "\u94fe\u8def\uff1a\u51fa\u9519",  # 链路：出错
    "status.link.other": "\u94fe\u8def\uff1a{status}",  # 链路：{status}
    "status.power.on": "\u7535\u6e90\uff1aON",  # 电源：ON
    "status.power.off": "\u7535\u6e90\uff1aOFF",  # 电源：OFF
    "status.power.unknown": "\u7535\u6e90\uff1a\u672a\u77e5",  # 电源：未知
    "status.power.other": "\u7535\u6e90\uff1a{status}",  # 电源：{status}
    "status.acq.idle": "\u91c7\u96c6\uff1a\u7a7a\u95f2",  # 采集：空闲
    "status.acq.continuous": "\u91c7\u96c6\uff1a\u8fde\u7eed",  # 采集：连续
    "status.acq.scheduled": "\u91c7\u96c6\uff1a\u5b9a\u65f6",  # 采集：定时
    "status.acq.finite_run": "\u91c7\u96c6\uff1a\u6709\u9650\u6b21\u6570",  # 采集：有限次数
    "status.acq.stepped_run": "\u91c7\u96c6\uff1a\u626b\u9891",  # 采集：扫频
    "status.acq.single_shot": "\u91c7\u96c6\uff1a\u5355\u5e27",  # 采集：单帧
    "status.acq.other": "\u91c7\u96c6\uff1a{mode}",  # 采集：{mode}
    "status.record.off": "\u5f55\u5236\uff1a\u5173",  # 录制：关
    "status.record.armed": "\u5f55\u5236\uff1a\u5c31\u7eea",  # 录制：就绪
    "status.record.recording": "\u5f55\u5236\u4e2d\u2026",  # 录制中…
    "status.record.other": "\u5f55\u5236\uff1a{status}",  # 录制：{status}
    # ==================================================================
    # Dialog — Difference Reconstruction
    # ==================================================================
    "dlg.difference.title": "\u5dee\u5206\u91cd\u6784",  # 差分重构
    "dlg.difference.frame_group": "\u5e27\u9009\u62e9",  # 帧选择
    "dlg.difference.ref_label": "\u53c2\u8003\u5e27\uff1a",  # 参考帧：
    "dlg.difference.tgt_label": "\u76ee\u6807\u5e27\uff1a",  # 目标帧：
    "dlg.difference.settings_group": "\u8bbe\u7f6e",  # 设置
    "dlg.difference.mode_label": "\u5dee\u5206\u6a21\u5f0f\uff1a",  # 差分模式：
    "dlg.difference.orient_label": "\u65b9\u5411\uff1a",  # 方向：
    "dlg.difference.part_label": "\u4f7f\u7528\u5206\u91cf\uff1a",  # 使用分量：
    "dlg.difference.warn_same_frame": "\u53c2\u8003\u5e27\u548c\u76ee\u6807\u5e27\u4e0d\u80fd\u76f8\u540c\u3002",  # 参考帧和目标帧不能相同。
    # ==================================================================
    # Dialog — Single-session Reconstruct
    # ==================================================================
    "dlg.reconstruction.title": "\u91cd\u6784",  # 重构
    "dlg.reconstruction.heading": "\u4ece\u5f55\u5236\u5e27\u91cd\u6784",  # 从录制帧重构
    "dlg.reconstruction.cancel_button": "\u53d6\u6d88",  # 取消
    "dlg.reconstruction.run_button": "\u8fd0\u884c\u91cd\u6784",  # 运行重构
    "dlg.reconstruction.selected_frames_group": "\u5df2\u9009\u5e27",  # 已选帧
    "dlg.reconstruction.ref_label": "\u53c2\u8003\uff1a",  # 参考：
    "dlg.reconstruction.tgt_label": "\u76ee\u6807\uff1a",  # 目标：
    "dlg.reconstruction.algo_params_group": "\u7b97\u6cd5\u4e0e\u53c2\u6570",  # 算法与参数
    "dlg.reconstruction.method_label": "\u65b9\u6cd5\uff1a",  # 方法：
    "dlg.reconstruction.part_label": "\u4f7f\u7528\u5206\u91cf\uff1a",  # 使用分量：
    "dlg.reconstruction.alpha_label": "\u6b63\u5219\u5316 \u03b1\uff1a",  # 正则化 α：
    "dlg.reconstruction.lambda_eff_label": "\u03bb_eff\uff1a",  # λ_eff：
    "dlg.reconstruction.custom_lambda_check": "\u9ad8\u7ea7\uff1a\u7528\u81ea\u5b9a\u4e49 \u03bb_eff \u91cd\u65b0\u6784\u5efa RM\uff08\u8f83\u6162\uff09",  # 高级：用自定义 λ_eff 重新构建 RM（较慢）
    "dlg.reconstruction.custom_lambda_tip": "\u4f7f\u7528\u8f93\u5165\u7684 \u03bb_eff \u51b7\u6784\u5efa\u6216\u52a0\u8f7d\u72ec\u7acb\u7684 RM \u6587\u4ef6\u3002",  # 使用输入的 λ_eff 冷构建或加载独立的 RM 文件。
    "dlg.reconstruction.lambda_locked_tip": "\u5355\u6b65/RM \u5dee\u5206\u8def\u5f84\u9ed8\u8ba4\u4f7f\u7528\u56fa\u5b9a \u03bb_eff=1e-2\u3002",  # 单步/RM 差分路径默认使用固定 λ_eff=1e-2。
    "dlg.reconstruction.iter_label": "\u6700\u5927\u8fed\u4ee3\u6b21\u6570\uff1a",  # 最大迭代次数：
    "dlg.reconstruction.output_group": "\u8f93\u51fa\uff08\u53ef\u9009\uff09",  # 输出（可选）
    "dlg.reconstruction.output_placeholder": "\u7559\u7a7a\u5219\u4ec5\u663e\u793a\u7ed3\u679c\uff08\u4e0d\u4fdd\u5b58\uff09",  # 留空则仅显示结果（不保存）
    "dlg.reconstruction.browse_button": "\u6d4f\u89c8\u2026",  # 浏览…
    "dlg.reconstruction.output_folder_label": "\u8f93\u51fa\u6587\u4ef6\u5939\uff1a",  # 输出文件夹：
    "dlg.reconstruction.save_image_check": "\u4fdd\u5b58\u91cd\u6784\u56fe\u50cf (PNG)",  # 保存重构图像 (PNG)
    "dlg.reconstruction.save_voltage_check": "\u4fdd\u5b58\u8fb9\u754c\u7535\u538b\u62df\u5408\u56fe (PNG)",  # 保存边界电压拟合图 (PNG)
    "dlg.reconstruction.not_selected": "<\u672a\u9009\u62e9>",  # <未选择>
    "dlg.reconstruction.absolute_no_ref_tip": "\u7edd\u5bf9\u65b9\u6cd5\u4e0d\u9700\u8981\u53c2\u8003\u5e27\u3002",  # 绝对方法不需要参考帧。
    "dlg.recon_settings.dialog_title": "\u91cd\u6784\u53c2\u6570",  # 重构参数
    "dlg.recon_settings.dialog_heading": "\u6b63/\u9006\u95ee\u9898\u53c2\u6570",  # 正/逆问题参数
    "dlg.recon_settings.reset_button": "\u91cd\u8f7d\u5f53\u524d\u6570\u636e",  # 重载当前数据
    "dlg.recon_settings.cancel_button": "\u53d6\u6d88",  # 取消
    "dlg.recon_settings.apply_button": "\u5e94\u7528",  # 应用
    "dlg.recon_settings.toggle_show": "\u663e\u793a\u6b63/\u9006\u95ee\u9898\u53c2\u6570",  # 显示正/逆问题参数
    "dlg.recon_settings.toggle_hide": "\u9690\u85cf\u6b63/\u9006\u95ee\u9898\u53c2\u6570",  # 隐藏正/逆问题参数
    "dlg.recon_settings.tab_mesh": "\u7f51\u683c\u4e0e\u7535\u6781",  # 网格与电极
    "dlg.recon_settings.tab_protocol": "\u6fc0\u52b1\u4e0e\u6d4b\u91cf",  # 激励与测量
    "dlg.recon_settings.tab_solver": "\u6c42\u89e3\u5668\u4e0e\u8fd0\u884c\u65f6",  # 求解器与运行时
    "dlg.recon_settings.mesh_dimension": "\u5f85\u6d4b\u57df\uff1a",  # 待测域：
    "dlg.recon_settings.mesh_refinement": "\u7f51\u683c\u5c3a\u5bf8/\u7ec6\u5316\uff1a",  # 网格尺寸/细化：
    "dlg.recon_settings.rm_inverse_mesh": "RM \u9006\u7f51\u683c\u5c3a\u5bf8\uff1a",  # RM 逆网格尺寸：
    "dlg.recon_settings.n_elec": "\u6bcf\u73af\u7535\u6781\u6570\uff1a",  # 每环电极数：
    "dlg.recon_settings.n_rings": "\u7535\u6781\u73af\u6570\uff1a",  # 电极环数：
    "dlg.recon_settings.electrode_layout": "\u7535\u6781\u6392\u5217\uff1a",  # 电极排列：
    "dlg.recon_settings.radius": "\u534a\u5f84\uff1a",  # 半径：
    "dlg.recon_settings.height": "\u9ad8\u5ea6\uff1a",  # 高度：
    "dlg.recon_settings.geometry_scale": "\u51e0\u4f55\u5230\u7c73\u7684\u7f29\u653e\uff1a",  # 几何到米的缩放：
    "dlg.recon_settings.electrode_coverage": "\u7535\u6781\u8986\u76d6\u7387\uff1a",  # 电极覆盖率：
    "dlg.recon_settings.electrode_length": "\u7535\u6781\u957f\u5ea6\uff1a",  # 电极长度：
    "dlg.recon_settings.electrode_area": "\u7535\u6781\u9762\u79ef\uff1a",  # 电极面积：
    "dlg.recon_settings.electrode_height_ratio": "\u7535\u6781\u9ad8\u5ea6\u6bd4\uff1a",  # 电极高度比：
    "dlg.recon_settings.stim_pattern": "\u6fc0\u52b1\u6a21\u5f0f\uff1a",  # 激励模式：
    "dlg.recon_settings.meas_pattern": "\u6d4b\u91cf\u6a21\u5f0f\uff1a",  # 测量模式：
    "dlg.recon_settings.measurement_protocol": "\u6d4b\u91cf\u534f\u8bae\uff1a",  # 测量协议：
    "dlg.recon_settings.rotate_meas": "\u65cb\u8f6c\u6d4b\u91cf\u987a\u5e8f",  # 旋转测量顺序
    "dlg.recon_settings.use_meas_current": "\u5305\u542b\u6d4b\u91cf\u7535\u6d41\u7535\u6781",  # 包含测量电流电极
    "dlg.recon_settings.use_meas_current_next": "\u6d4b\u91cf\u7535\u6d41 next\uff1a",  # 测量电流 next：
    "dlg.recon_settings.stim_direction": "\u6fc0\u52b1\u65b9\u5411\uff1a",  # 激励方向：
    "dlg.recon_settings.meas_direction": "\u6d4b\u91cf\u65b9\u5411\uff1a",  # 测量方向：
    "dlg.recon_settings.stim_first_positive": "\u7b2c\u4e00\u4e2a\u6fc0\u52b1\u7535\u6781\u4e3a\u6b63",  # 第一个激励电极为正
    "dlg.recon_settings.drive_mode": "\u9a71\u52a8\u6a21\u5f0f\uff1a",  # 驱动模式：
    "dlg.recon_settings.drive_value": "\u9a71\u52a8\u503c\uff1a",  # 驱动值：
    "dlg.recon_settings.contact_impedance": "\u63a5\u89e6\u963b\u6297\uff1a",  # 接触阻抗：
    "dlg.recon_settings.solver_mode": "\u9006\u95ee\u9898\u6c42\u89e3\u6a21\u5f0f\uff1a",  # 逆问题求解模式：
    "dlg.recon_settings.linear_solver": "\u7ebf\u6027\u6c42\u89e3\u5668\uff1a",  # 线性求解器：
    "dlg.recon_settings.preconditioner": "\u9884\u6761\u4ef6\uff1a",  # 预条件：
    "dlg.recon_settings.jacobian_representation": "Jacobian \u8868\u793a\uff1a",  # Jacobian 表示：
    "dlg.recon_settings.forward_solver_preset": "\u6b63\u95ee\u9898\u6c42\u89e3\u9884\u8bbe\uff1a",  # 正问题求解预设：
    "dlg.recon_settings.forward_mat_solve": "\u6b63\u95ee\u9898 mat-solve\uff1a",  # 正问题 mat-solve：
    "dlg.recon_settings.petsc_device": "PETSc \u8bbe\u5907\uff1a",  # PETSc 设备：
    "dlg.recon_settings.runtime_device": "\u8fd0\u884c\u65f6\u8bbe\u5907\uff1a",  # 运行时设备：
    "dlg.recon_settings.acceleration_profile": "\u52a0\u901f profile\uff1a",  # 加速 profile：
    # ==================================================================
    # Dialog — Batch Reconstruct
    # ==================================================================
    "dlg.batch.title": "\u6279\u91cf\u91cd\u6784",  # 批量重构
    "dlg.batch.heading": "\u6279\u91cf\u91cd\u6784",  # 批量重构
    "dlg.batch.close_button": "\u5173\u95ed",  # 关闭
    "dlg.batch.open_output_button": "\u6253\u5f00\u8f93\u51fa\u6587\u4ef6\u5939",  # 打开输出文件夹
    "dlg.batch.cancel_button": "\u53d6\u6d88\u4efb\u52a1",  # 取消任务
    "dlg.batch.run_button": "\u8fd0\u884c\u6279\u91cf",  # 运行批量
    "dlg.batch.folders_group": "\u6587\u4ef6\u5939",  # 文件夹
    "dlg.batch.input_placeholder": "\u5305\u542b\u5e27 CSV \u7684\u6587\u4ef6\u5939",  # 包含帧 CSV 的文件夹
    "dlg.batch.browse_button": "\u6d4f\u89c8\u2026",  # 浏览…
    "dlg.batch.input_label": "\u8f93\u5165\u6587\u4ef6\u5939\uff1a",  # 输入文件夹：
    "dlg.batch.output_placeholder": "\u7528\u4e8e\u5199\u5165\u91cd\u6784\u56fe\u50cf\u7684\u6587\u4ef6\u5939",  # 用于写入重构图像的文件夹
    "dlg.batch.output_label": "\u8f93\u51fa\u6587\u4ef6\u5939\uff1a",  # 输出文件夹：
    "dlg.batch.algo_params_group": "\u7b97\u6cd5\u4e0e\u53c2\u6570",  # 算法与参数
    "dlg.batch.method_label": "\u65b9\u6cd5\uff1a",  # 方法：
    "dlg.batch.part_label": "\u4f7f\u7528\u5206\u91cf\uff1a",  # 使用分量：
    "dlg.batch.alpha_label": "\u6b63\u5219\u5316 \u03b1\uff1a",  # 正则化 α：
    "dlg.batch.lambda_eff_label": "\u03bb_eff\uff1a",  # λ_eff：
    "dlg.batch.custom_lambda_check": "\u9ad8\u7ea7\uff1a\u7528\u81ea\u5b9a\u4e49 \u03bb_eff \u91cd\u65b0\u6784\u5efa RM\uff08\u8f83\u6162\uff09",  # 高级：用自定义 λ_eff 重新构建 RM（较慢）
    "dlg.batch.custom_lambda_tip": "\u4f7f\u7528\u8f93\u5165\u7684 \u03bb_eff \u4e3a\u6279\u91cf\u4efb\u52a1\u51b7\u6784\u5efa\u6216\u52a0\u8f7d\u72ec\u7acb\u7684 RM \u6587\u4ef6\u3002",  # 使用输入的 λ_eff 为批量任务冷构建或加载独立的 RM 文件。
    "dlg.batch.lambda_locked_tip": "\u5355\u6b65/RM \u5dee\u5206\u8def\u5f84\u9ed8\u8ba4\u4f7f\u7528\u56fa\u5b9a \u03bb_eff=1e-2\u3002",  # 单步/RM 差分路径默认使用固定 λ_eff=1e-2。
    "dlg.batch.iter_label": "\u6700\u5927\u8fed\u4ee3\u6b21\u6570\uff1a",  # 最大迭代次数：
    "dlg.batch.ref_browse_button": "\u6d4f\u89c8\u2026",  # 浏览…
    "dlg.batch.ref_label": "\u53c2\u8003\u5e27\uff1a",  # 参考帧：
    "dlg.batch.outputs_group": "\u8f93\u51fa",  # 输出
    "dlg.batch.save_image_check": "\u4fdd\u5b58\u91cd\u6784\u56fe\u50cf (PNG)",  # 保存重构图像 (PNG)
    "dlg.batch.progress_group": "\u8fdb\u5ea6",  # 进度
    "dlg.batch.ready": "\u5c31\u7eea\u3002",  # 就绪。
    "dlg.batch.cancelling": "\u53d6\u6d88\u4e2d\u2026",  # 取消中…
    "dlg.batch.progress_default": "{current}/{total}",
    "dlg.batch.progress_with_eta": "{current}/{total}  \u00b7  \u5269\u4f59 {eta}",  # {current}/{total} · 剩余 {eta}
    "dlg.batch.eta_seconds": "{seconds} \u79d2",  # {seconds} 秒
    "dlg.batch.eta_minutes": "{minutes} \u5206 {seconds} \u79d2",  # {minutes} 分 {seconds} 秒
    "dlg.batch.eta_hours": "{hours} \u5c0f\u65f6 {minutes} \u5206",  # {hours} 小时 {minutes} 分
    "dlg.batch.error": "\u2715  \u9519\u8bef\uff1a{message}",  # ✕  错误：{message}
    "dlg.batch.subtitle": "\u91cd\u6784\u8f93\u5165\u6587\u4ef6\u5939\u4e2d\u7684\u6bcf\u4e00\u4e2a\u5e27 CSV\u3002\u5bf9\u4e8e\u5dee\u5206\u65b9\u6cd5\uff0c\u53c2\u8003\u5e27\u4f1a\u5e94\u7528\u5230\u6240\u6709\u76ee\u6807\uff0c\u82e5\u53c2\u8003\u5e27\u4f4d\u4e8e\u540c\u4e00\u6587\u4ef6\u5939\u4e2d\u4f1a\u81ea\u52a8\u6392\u9664\u3002",  # 重构输入文件夹中的每一个帧 CSV。对于差分方法，参考帧会应用到所有目标，若参考帧位于同一文件夹中会自动排除。
    "dlg.batch.ref_placeholder": "\u4f5c\u4e3a\u53c2\u8003\u7684 CSV \u6587\u4ef6\uff08\u5dee\u5206\u65b9\u6cd5\u5fc5\u586b\uff09",  # 作为参考的 CSV 文件（差分方法必填）
    "dlg.batch.save_voltage_check": "\u4fdd\u5b58\u8fb9\u754c\u7535\u538b\u62df\u5408\u56fe (PNG)",  # 保存边界电压拟合图 (PNG)
    "dlg.batch.file_dialog.input": "\u9009\u62e9\u8f93\u5165\u6587\u4ef6\u5939",  # 选择输入文件夹
    "dlg.batch.file_dialog.output": "\u9009\u62e9\u8f93\u51fa\u6587\u4ef6\u5939",  # 选择输出文件夹
    "dlg.batch.file_dialog.ref": "\u9009\u62e9\u53c2\u8003\u5e27 CSV",  # 选择参考帧 CSV
    "dlg.batch.file_dialog.csv_filter": "CSV \u6587\u4ef6 (*.csv)",  # CSV 文件 (*.csv)
    "dlg.batch.finished_ok": "\u2713  \u5b8c\u6210 \u2014 \u6210\u529f\uff1a{succeeded}\uff0c\u5931\u8d25\uff1a{failed}",  # ✓  完成 — 成功：{succeeded}，失败：{failed}
    "dlg.batch.finished_mixed": "\u26a0  \u5b8c\u6210 \u2014 \u6210\u529f\uff1a{succeeded}\uff0c\u5931\u8d25\uff1a{failed}",  # ⚠  完成 — 成功：{succeeded}，失败：{failed}
    "dlg.batch.finished_fail": "\u2715  \u5b8c\u6210 \u2014 \u6210\u529f\uff1a{succeeded}\uff0c\u5931\u8d25\uff1a{failed}",  # ✕  完成 — 成功：{succeeded}，失败：{failed}
    # Reconstruction dialog — subtitle copy
    "dlg.reconstruction.subtitle": "\u9009\u62e9\u7b97\u6cd5\uff0c\u8bbe\u7f6e\u6b63\u5219\u5316\u53c2\u6570\uff0c\u7136\u540e\u8fd0\u884c\u3002\u5dee\u5206\u65b9\u6cd5\u9700\u8981\u53c2\u8003\u5e27\u548c\u76ee\u6807\u5e27\uff1b\u7edd\u5bf9\u65b9\u6cd5\u4ec5\u9700\u8981\u76ee\u6807\u5e27\u3002",  # 选择算法，设置正则化参数，然后运行。差分方法需要参考帧和目标帧；绝对方法仅需要目标帧。
    # ==================================================================
    # Dialog — Interop Hub (EIDORS ↔ PyEIDORS 迁移工作台)
    # ==================================================================
    "dlg.interop.title": "\u4e92\u64cd\u4f5c\u4e2d\u5fc3",  # 互操作中心
    "dlg.interop.intro": "\u5728\u8fd9\u91cc\u6211\u4eec\u628a EIDORS \u4e0e PyEIDORS \u4e4b\u95f4\u7684\u8fc1\u79fb\u505a\u6210\u4e00\u6761\u53ef\u89c6\u5316\u3001\u53ef\u786e\u8ba4\u3001\u53ef\u56de\u6eda\u7684\u5de5\u4f5c\u6d41\u3002",  # 在这里我们把 EIDORS 与 PyEIDORS 之间的迁移做成一条可视化、可确认、可回滚的工作流。
    # Tab labels
    "dlg.interop.tabs.import": "\u4ece EIDORS \u5bfc\u5165",  # 从 EIDORS 导入
    "dlg.interop.tabs.export": "\u5bfc\u51fa\u5230 EIDORS",  # 导出到 EIDORS
    "dlg.interop.tabs.profiles": "\u73af\u5883\u753b\u50cf\u4e0e\u8def\u5f84",  # 环境画像与路径
    # Shared — path pick button
    "dlg.interop.path_pick_button": "\u9009\u62e9\u2026",  # 选择…
    # Manual status panel (top of Import tab)
    "dlg.interop.status.title": "\u5f53\u524d\u624b\u52a8\u6307\u5b9a\u72b6\u6001",  # 当前手动指定状态
    "dlg.interop.status.unspecified": "\u672a\u6307\u5b9a",  # 未指定
    "dlg.interop.status.pending": "\u5f85\u751f\u6210",  # 待生成
    "dlg.interop.status.specified": "\u5df2\u6307\u5b9a",  # 已指定
    "dlg.interop.status.not_selected": "\u672a\u9009\u62e9",  # 未选择
    "dlg.interop.status.not_found": "\u672a\u627e\u5230",  # 未找到
    "dlg.interop.status.ready_fmt": "\u5c31\u7eea\uff08{suffix}\uff09",  # 就绪({suffix})
    "dlg.interop.status.ready": "\u5c31\u7eea",  # 就绪
    "dlg.interop.status.failed": "\u5931\u8d25",  # 失败
    # Step 1 — Environment
    "dlg.interop.env.title": "\u7b2c 1 \u6b65 \u00b7 \u6307\u5b9a\u73af\u5883",  # 第 1 步 · 指定环境
    "dlg.interop.env.hint": "\u8bf7\u70b9\u51fb\u201c\u9009\u62e9\u2026\u201d\u624b\u52a8\u6307\u5b9a MATLAB \u4e0e startup.m \u8def\u5f84\u3002\u7edf\u4e00\u6587\u4ef6\u6d4f\u89c8\u5668\u4f1a\u6309\u5f53\u524d\u73af\u5883\u663e\u793a\u53ef\u8bbf\u95ee\u7684 Linux / WSL / Windows \u4f4d\u7f6e\uff1b\u73af\u5883\u753b\u50cf\u53ef\u5728\u201c\u73af\u5883\u753b\u50cf\u4e0e\u8def\u5f84\u201d\u9875\u7ba1\u7406\u3002",  # 请点击"选择…"手动指定 MATLAB 与 startup.m 路径。统一文件浏览器会按当前环境显示可访问的 Linux / WSL / Windows 位置；环境画像可在"环境画像与路径"页管理。
    "dlg.interop.env.matlab_label": "MATLAB\uff1a",  # MATLAB：
    "dlg.interop.env.matlab_placeholder": "MATLAB \u53ef\u6267\u884c\u6587\u4ef6\u8def\u5f84",  # MATLAB 可执行文件路径
    "dlg.interop.env.pick_matlab_title": "\u9009\u62e9 MATLAB \u53ef\u6267\u884c\u6587\u4ef6",  # 选择 MATLAB 可执行文件
    "dlg.interop.env.matlab_filter": "\u53ef\u6267\u884c\u6587\u4ef6 (*.exe *.bin *.sh);;\u6240\u6709\u6587\u4ef6 (*)",  # 可执行文件 (*.exe *.bin *.sh);;所有文件 (*)
    "dlg.interop.env.startup_label": "EIDORS startup\uff1a",  # EIDORS startup：
    "dlg.interop.env.startup_placeholder": "startup.m \u8def\u5f84",  # startup.m 路径
    "dlg.interop.env.pick_startup_title": "\u9009\u62e9 EIDORS startup.m",  # 选择 EIDORS startup.m
    "dlg.interop.env.startup_filter": "MATLAB \u811a\u672c (*.m);;\u6240\u6709\u6587\u4ef6 (*)",  # MATLAB 脚本 (*.m);;所有文件 (*)
    "dlg.interop.env.manual_entry": "\u5f53\u524d\u624b\u52a8\u8f93\u5165",  # 当前手动输入
    "dlg.interop.env.saved_default_name": "\u5df2\u4fdd\u5b58\u7684 EIDORS \u73af\u5883",  # 已保存的 EIDORS 环境
    # Step 2 — Source
    "dlg.interop.source.title": "\u7b2c 2 \u6b65 \u00b7 \u9009\u62e9\u6765\u6e90",  # 第 2 步 · 选择来源
    "dlg.interop.source.label": "\u6765\u6e90\uff1a",  # 来源：
    "dlg.interop.source.placeholder": "\u9009\u62e9 EIDORS .m \u811a\u672c\u3001bridge \u76ee\u5f55\u3001legacy .mat \u6216 bridge JSON",  # 选择 EIDORS .m 脚本、bridge 目录、legacy .mat 或 bridge JSON
    "dlg.interop.source.pick_title": "\u9009\u62e9 EIDORS \u811a\u672c\u3001bridge \u6587\u4ef6\u6216 bridge \u76ee\u5f55",  # 选择 EIDORS 脚本、bridge 文件或 bridge 目录
    "dlg.interop.source.pick_filter": "\u652f\u6301\u7c7b\u578b (*.m *.mat *.json);;MATLAB \u811a\u672c (*.m);;MAT \u6587\u4ef6 (*.mat);;JSON (*.json);;\u6240\u6709\u6587\u4ef6 (*)",
    "dlg.interop.source.capture_label": "\u91c7\u96c6\u8f93\u51fa\uff1a",  # 采集输出：
    "dlg.interop.source.pick_capture_title": "\u9009\u62e9\u6865\u63a5\u91c7\u96c6\u8f93\u51fa\u76ee\u5f55",  # 选择桥接采集输出目录
    "dlg.interop.source.hint": "\u652f\u6301\u4e09\u79cd\u6765\u6e90\uff1a\u7528\u6237\u811a\u672c\u3001\u5df2\u6709 bridge \u5de5\u7a0b\u3001legacy \u51e0\u4f55 .mat\u3002",  # 支持三种来源：用户脚本、已有 bridge 工程、legacy 几何 .mat。
    # Step 3 — Capture & preview actions
    "dlg.interop.actions.title": "\u7b2c 3 \u6b65 \u00b7 \u91c7\u96c6\u4e0e\u9884\u89c8",  # 第 3 步 · 采集与预览
    "dlg.interop.actions.preview_button": "\u751f\u6210\u9884\u89c8",  # 生成预览
    "dlg.interop.actions.reload_button": "\u91cd\u8f7d\u4e0a\u6b21\u7ed3\u679c",  # 重载上次结果
    "dlg.interop.actions.no_preview_yet": "\u5c1a\u672a\u751f\u6210\u8fc1\u79fb\u9884\u89c8\u3002",  # 尚未生成迁移预览。
    # Step 4 — Preview & import
    "dlg.interop.preview.title": "\u7b2c 4 \u6b65 \u00b7 \u9884\u89c8\u4e0e\u5bfc\u5165",  # 第 4 步 · 预览与导入
    "dlg.interop.preview.waiting": "\u7b49\u5f85 bridge \u5305\u9884\u89c8\u3002",  # 等待 bridge 包预览。
    "dlg.interop.preview.source_col_header": "EIDORS \u6765\u6e90",  # EIDORS 来源
    "dlg.interop.preview.value_col_header": "\u503c",  # 值
    "dlg.interop.preview.mapping_col_header": "PyEIDORS \u6620\u5c04",  # PyEIDORS 映射
    "dlg.interop.preview.warnings_placeholder": "\u8b66\u544a\u4e0e\u672a\u89e3\u6790\u5b57\u6bb5\u4f1a\u663e\u793a\u5728\u6b64\u3002",  # 警告与未解析字段会显示在此。
    "dlg.interop.preview.missing_fallback": "\u9700\u7528\u6237\u8865\u5145\uff0c\u6216\u6539\u7528\u6865\u63a5\u6a21\u677f\u5305\u88c5\u811a\u672c",  # 需用户补充，或改用桥接模板包装脚本
    "dlg.interop.preview.overview": "EIDORS \u2192 PyEIDORS \u6620\u5c04\u9884\u89c8\uff1a{dim}\uff0c{n_elec} \u7535\u6781/\u73af\uff0c{pts} \u4e2a\u8fb9\u754c\u7535\u538b\u70b9\u3002",  # EIDORS → PyEIDORS 映射预览：{dim}，{n_elec} 电极/环，{pts} 个边界电压点。
    "dlg.interop.preview.counts": "\u5df2\u8bc6\u522b {recognized} \u9879  |  \u5df2\u63a8\u65ad {inferred} \u9879  |  \u5f85\u8865\u5145 {missing} \u9879",  # 已识别 {recognized} 项 | 已推断 {inferred} 项 | 待补充 {missing} 项
    "dlg.interop.preview.no_warnings": "\u672a\u53d1\u73b0\u9700\u8981\u4eba\u5de5\u786e\u8ba4\u7684\u9ad8\u98ce\u9669\u9879\u3002",  # 未发现需要人工确认的高风险项。
    "dlg.interop.preview.done": "\u9884\u89c8\u5b8c\u6210\uff1a{dim}  |  {n_elec} \u7535\u6781/\u73af  |  {pts} \u4e2a\u8fb9\u754c\u7535\u538b\u70b9\u3002",  # 预览完成：{dim} | {n_elec} 电极/环 | {pts} 个边界电压点。
    "dlg.interop.preview.smoke_placeholder": "\u5bfc\u5165\u540e\u7684\u9006\u95ee\u9898\u70df\u6d4b\u7ed3\u679c\u4f1a\u663e\u793a\u5728\u8fd9\u91cc\u3002",  # 导入后的逆问题烟测结果会显示在这里。
    # Import target combo
    "dlg.interop.import_target.hardware": "\u786c\u4ef6\u914d\u7f6e\u6a21\u677f",  # 硬件配置模板
    "dlg.interop.import_target.simulation": "\u4eff\u771f\u914d\u7f6e",  # 仿真配置
    "dlg.interop.import_target.dataset": "\u6570\u636e\u96c6\u914d\u7f6e",  # 数据集配置
    "dlg.interop.import_target.measurements": "\u4ec5\u8fb9\u754c\u7535\u538b\u6570\u636e",  # 仅边界电压数据
    "dlg.interop.import_target.geometry": "\u4ec5\u51e0\u4f55\u8d44\u4ea7",  # 仅几何资产
    "dlg.interop.auto_smoke_check": "\u5bfc\u5165\u540e\u81ea\u52a8\u505a\u4e00\u6b21\u9006\u95ee\u9898\u70df\u6d4b\u9a8c\u8bc1",  # 导入后自动做一次逆问题烟测验证
    "dlg.interop.import_button": "\u5bfc\u5165\u5230 PyEIDORS",  # 导入到 PyEIDORS
    "dlg.interop.smoke_button": "\u8fd0\u884c\u70df\u6d4b\u9a8c\u8bc1",  # 运行烟测验证
    # Export tab
    "dlg.interop.export.title": "\u5bfc\u51fa\u5230 EIDORS",  # 导出到 EIDORS
    "dlg.interop.export.source.simulation": "\u5f53\u524d\u4eff\u771f\u914d\u7f6e",  # 当前仿真配置
    "dlg.interop.export.source.hardware": "\u5f53\u524d\u786c\u4ef6\u5e03\u5c40\u914d\u7f6e",  # 当前硬件布局配置
    "dlg.interop.export.source.recording": "\u5f53\u524d\u5f55\u5236/\u91cd\u6784\u7ed3\u679c",  # 当前录制/重构结果
    "dlg.interop.export.source_label": "\u6765\u6e90\uff1a",  # 来源：
    "dlg.interop.export.output_label": "\u8f93\u51fa\u76ee\u5f55\uff1a",  # 输出目录：
    "dlg.interop.export.pick_output_title": "\u9009\u62e9\u5bfc\u51fa Bridge \u5de5\u7a0b\u76ee\u5f55",  # 选择导出 Bridge 工程目录
    "dlg.interop.export.hint": "\u5bfc\u51fa bridge \u5de5\u7a0b\u65f6\uff0c\u4f1a\u4f18\u5148\u5199\u5165\u5f53\u524d\u624b\u52a8\u6307\u5b9a\u7684 MATLAB / startup.m \u8def\u5f84\uff1b\u82e5\u672a\u6307\u5b9a\uff0c\u4e5f\u4ecd\u53ef\u53ea\u5bfc\u51fa\u6570\u636e\u4e0e\u914d\u7f6e\u3002",  # 导出 bridge 工程时，会优先写入当前手动指定的 MATLAB / startup.m 路径；若未指定，也仍可只导出数据与配置。
    "dlg.interop.export.include_label": "\u5305\u542b\u5185\u5bb9\uff1a",  # 包含内容：
    "dlg.interop.export.include_geometry": "\u51e0\u4f55",  # 几何
    "dlg.interop.export.include_data": "\u8fb9\u754c\u7535\u538b",  # 边界电压
    "dlg.interop.export.include_scripts": "\u53ef\u8fd0\u884c EIDORS \u811a\u672c",  # 可运行 EIDORS 脚本
    "dlg.interop.export.generate_button": "\u751f\u6210 Bridge \u5de5\u7a0b",  # 生成 Bridge 工程
    "dlg.interop.export.log_placeholder": "\u5bfc\u51fa\u8bf4\u660e\u3001\u751f\u6210\u8def\u5f84\u548c\u4efb\u4f55\u964d\u7ea7\u884c\u4e3a\u90fd\u4f1a\u5199\u5728\u8fd9\u91cc\u3002",  # 导出说明、生成路径和任何降级行为都会写在这里。
    "dlg.interop.export.success": "[OK] Bridge \u5de5\u7a0b\u5df2\u751f\u6210\uff1a{root}",  # [OK] Bridge 工程已生成：{root}
    "dlg.interop.export.source_tag": "      \u6765\u6e90\uff1a{source_kind}",  #       来源：{source_kind}
    # Profiles & Paths tab
    "dlg.interop.profiles.group_title": "\u5df2\u4fdd\u5b58\u7684\u73af\u5883\u753b\u50cf",  # 已保存的环境画像
    "dlg.interop.profiles.name_label": "\u540d\u79f0\uff1a",  # 名称：
    "dlg.interop.profiles.matlab_label": "MATLAB\uff1a",  # MATLAB：
    "dlg.interop.profiles.startup_label": "startup.m\uff1a",  # startup.m：
    "dlg.interop.profiles.script_label": "\u4e0a\u6b21\u811a\u672c\uff1a",  # 上次脚本：
    "dlg.interop.profiles.output_label": "\u4e0a\u6b21\u8f93\u51fa\uff1a",  # 上次输出：
    "dlg.interop.profiles.save_button": "\u4fdd\u5b58\u5f53\u524d\u73af\u5883",  # 保存当前环境
    "dlg.interop.profiles.remove_button": "\u5220\u9664\u9009\u4e2d\u9879",  # 删除选中项
    "dlg.interop.profiles.note": "\u8fd9\u91cc\u4fdd\u5b58\u7684\u662f EIDORS \u73af\u5883\u753b\u50cf\uff0c\u4e0d\u4f1a\u4fee\u6539\u7528\u6237\u539f\u59cb MATLAB \u5de5\u7a0b\u3002",  # 这里保存的是 EIDORS 环境画像，不会修改用户原始 MATLAB 工程。
    "dlg.interop.profiles.unnamed": "\u672a\u547d\u540d\u7684 EIDORS \u73af\u5883",  # 未命名的 EIDORS 环境
    "dlg.interop.profiles.custom_default": "\u81ea\u5b9a\u4e49 EIDORS \u73af\u5883",  # 自定义 EIDORS 环境
    "dlg.interop.profiles.manual_name": "\u624b\u52a8\u73af\u5883",  # 手动环境
    # Status-bar / message-box text
    "dlg.interop.msg.no_source": "\u8bf7\u5148\u9009\u62e9\u4e00\u4e2a EIDORS \u811a\u672c\u6216 bridge \u5305\u6765\u6e90\u3002",  # 请先选择一个 EIDORS 脚本或 bridge 包来源。
    "dlg.interop.msg.missing_before_script": "\u8fd0\u884c EIDORS \u811a\u672c\u524d\uff0c\u8bf7\u5148\u624b\u52a8\u6307\u5b9a\uff1a{parts}\u3002",  # 运行 EIDORS 脚本前，请先手动指定：{parts}。
    "dlg.interop.msg.missing_joiner": "\u3001",  # 、
    "dlg.interop.msg.preview_failed": "\u751f\u6210\u9884\u89c8\u5931\u8d25\uff1a{error}",  # 生成预览失败：{error}
    "dlg.interop.msg.no_bundle": "\u5f53\u524d\u8fd8\u6ca1\u6709\u5df2\u52a0\u8f7d\u7684 bridge \u5305\u3002",  # 当前还没有已加载的 bridge 包。
    "dlg.interop.msg.no_callback_import": "\u5f53\u524d\u7a97\u53e3\u672a\u63a5\u5165\u5bfc\u5165\u56de\u8c03\u3002",  # 当前窗口未接入导入回调。
    "dlg.interop.msg.no_callback_smoke": "\u5f53\u524d\u7a97\u53e3\u672a\u63a5\u5165\u70df\u6d4b\u56de\u8c03\u3002",  # 当前窗口未接入烟测回调。
    "dlg.interop.msg.no_callback_export": "\u5f53\u524d\u7a97\u53e3\u672a\u63a5\u5165\u5bfc\u51fa\u6570\u636e\u63d0\u4f9b\u5668\u3002",  # 当前窗口未接入导出数据提供器。
    "dlg.interop.msg.no_snapshot": "\u5f53\u524d\u6765\u6e90\u6682\u65f6\u6ca1\u6709\u53ef\u5bfc\u51fa\u7684\u4e0a\u4e0b\u6587\u3002",  # 当前来源暂时没有可导出的上下文。
    "dlg.interop.msg.import_failed": "\u5bfc\u5165\u5931\u8d25\uff1a{error}",  # 导入失败：{error}
    "dlg.interop.msg.smoke_failed": "\u70df\u6d4b\u5931\u8d25\uff1a{error}",  # 烟测失败：{error}
    "dlg.interop.msg.smoke_no_bundle": "\u5f53\u524d\u6ca1\u6709\u53ef\u7528\u4e8e\u70df\u6d4b\u7684 bridge \u5305\u3002",  # 当前没有可用于烟测的 bridge 包。
    "dlg.interop.msg.export_failed": "\u5bfc\u51fa\u5931\u8d25\uff1a{error}",  # 导出失败：{error}
    "dlg.interop.msg.bundle_no_preview": "\u5f53\u524d\u8fd8\u6ca1\u6709\u5df2\u52a0\u8f7d\u7684 bridge \u5305\u3002",  # 当前还没有已加载的 bridge 包。
    "dlg.interop.msg.profile_saved": "\u5df2\u4fdd\u5b58 profile\uff1a{name}",  # 已保存 profile：{name}
    "dlg.interop.msg.profile_removed": "\u5df2\u5220\u9664 profile\uff1a{name}",  # 已删除 profile：{name}
    # ==================================================================
    # Visual path picker (pick_visual_path)
    # ==================================================================
    "path_picker.sidebar.wsl_home": "WSL \u4e3b\u76ee\u5f55",  # WSL 主目录
    "path_picker.sidebar.wsl_root": "WSL \u6839\u76ee\u5f55",  # WSL 根目录
    "path_picker.sidebar.windows_home": "Windows \u7528\u6237\u76ee\u5f55",  # Windows 用户目录
    "path_picker.sidebar.linux_home": "Linux \u4e3b\u76ee\u5f55",  # Linux 主目录
    "path_picker.sidebar.linux_root": "Linux \u6839\u76ee\u5f55",  # Linux 根目录
    "path_picker.label.look_in": "\u4f4d\u7f6e\uff1a",  # 位置：
    "path_picker.label.file_name": "\u540d\u79f0\uff1a",  # 名称：
    "path_picker.label.file_type": "\u7c7b\u578b\uff1a",  # 类型：
    "path_picker.label.accept": "\u9009\u62e9",  # 选择
    "path_picker.label.reject": "\u53d6\u6d88",  # 取消
    "path_picker.button.choose_current_folder": "\u9009\u62e9\u5f53\u524d\u6587\u4ef6\u5939",  # 选择当前文件夹
}
