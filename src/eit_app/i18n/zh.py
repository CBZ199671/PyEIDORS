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
    "app.title": "EIT \u5de5\u4f5c\u7ad9",                                      # EIT 工作站

    # ------------------------------------------------------------------
    # Tab labels
    # ------------------------------------------------------------------
    "tab.hardware": "\u5b9e\u6d4b",                                             # 实测
    "tab.simulation": "\u4eff\u771f",                                           # 仿真
    "tab.dataset": "\u6570\u636e\u96c6",                                        # 数据集
    "tab.database": "\u6570\u636e\u5e93",                                       # 数据库

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------
    "menu.file": "\u6587\u4ef6(&F)",                                            # 文件(&F)
    "menu.file.settings": "\u8bbe\u7f6e(&S)\u2026",                              # 设置(&S)…
    "menu.file.exit": "\u9000\u51fa(&X)",                                       # 退出(&X)

    # ------------------------------------------------------------------
    # Tools menu
    # ------------------------------------------------------------------
    "menu.tools": "\u5de5\u5177(&T)",                                           # 工具(&T)
    "menu.tools.interop_hub": "EIDORS \u4e92\u64cd\u4f5c(&I)\u2026",             # EIDORS 互操作(&I)…

    # ------------------------------------------------------------------
    # Language menu
    # ------------------------------------------------------------------
    "menu.language": "\u8bed\u8a00(&L)",                                         # 语言(&L)
    "menu.language.zh": "\u4e2d\u6587",                                          # 中文
    "menu.language.en": "English",
    "menu.language.tooltip": "\u5728\u4e2d\u6587\u548c\u82f1\u6587\u4e4b\u95f4\u5207\u6362",  # 在中文和英文之间切换

    # ==================================================================
    # Hardware tab — Step labels on the left QToolBox
    # ==================================================================
    "hw.step.link": "\u6b65\u9aa4\u4e00 \u00b7 \u8fde\u63a5",                    # 步骤一 · 连接
    "hw.step.setup": "\u6b65\u9aa4\u4e8c \u00b7 \u8bbe\u7f6e",                   # 步骤二 · 设置
    "hw.step.acquire": "\u6b65\u9aa4\u4e09 \u00b7 \u91c7\u96c6",                 # 步骤三 · 采集

    # ==================================================================
    # Hardware tab — Step 1 Connection panel
    # ==================================================================
    "hw.connection.title": "1. \u8fde\u63a5\u4e0e\u9a8c\u8bc1",                   # 1. 连接与验证
    "hw.connection.flow_hint": "\u8bf7\u5148\u9009\u62e9\u4f20\u8f93\u65b9\u5f0f\u5e76\u9a8c\u8bc1\u8bbe\u5907\u8fde\u63a5\u3002",  # 请先选择传输方式并验证设备连接。
    "hw.connection.transport_label": "\u4f20\u8f93\u65b9\u5f0f\uff1a",            # 传输方式：
    "hw.connection.transport.serial": "\u4e32\u53e3",                             # 串口
    "hw.connection.transport.relay_4g": "4G \u4e2d\u7ee7",                        # 4G 中继
    "hw.connection.port_label": "\u7aef\u53e3\uff1a",                             # 端口：
    "hw.connection.scan_button": "\u626b\u63cf",                                  # 扫描
    "hw.connection.scan_button_tooltip": "\u5237\u65b0\u4e32\u53e3\u5217\u8868",   # 刷新串口列表
    "hw.connection.baud_label": "\u6ce2\u7279\u7387\uff1a",                       # 波特率：
    "hw.connection.host_label": "\u670d\u52a1\u5668\u5730\u5740\uff1a",          # 服务器地址：
    "hw.connection.port_spin_label": "\u670d\u52a1\u5668\u7aef\u53e3\uff1a",      # 服务器端口：
    "hw.connection.board_id_label": "\u677f\u5361 ID\uff1a",                      # 板卡 ID：
    "hw.connection.user_id_label": "\u7528\u6237 ID\uff1a",                       # 用户 ID：
    "hw.connection.connect_button": "\u8fde\u63a5",                               # 连接
    "hw.connection.connect_button_tooltip": "\u5efa\u7acb\u8fde\u63a5\u5e76\u9a8c\u8bc1\u8bbe\u5907",  # 建立连接并验证设备
    "hw.connection.disconnect_button": "\u65ad\u5f00",                            # 断开
    "hw.connection.port_hint.no_ports": "\u672a\u68c0\u6d4b\u5230\u53ef\u7528\u4e32\u53e3\u3002\u542f\u52a8\u5668\u4f1a\u81ea\u52a8\u68c0\u67e5\u672c\u5730 Linux \u4e32\u53e3\u548c Windows COM \u6865\u63a5\uff1b\u8bf7\u786e\u8ba4 USB \u7ebf\u3001\u9a71\u52a8\u548c\u8bbe\u5907\u4f9b\u7535\u540e\u518d\u70b9\u626b\u63cf\u3002",  # 未检测到可用串口。启动器会自动检查本地 Linux 串口和 Windows COM 桥接；请确认 USB 线、驱动和设备供电后再点扫描。
    "hw.connection.port_hint.still_no_ports": "\u4ecd\u672a\u68c0\u6d4b\u5230\u53ef\u7528\u4e32\u53e3\uff0c\u6682\u4e0d\u53d1\u8d77\u8fde\u63a5\u3002\u8bf7\u68c0\u67e5 USB \u8fde\u63a5\u3001\u9a71\u52a8\u548c\u8bbe\u5907\u7535\u6e90\u3002",  # 仍未检测到可用串口，暂不发起连接。请检查 USB 连接、驱动和设备电源。
    "hw.connection.port_hint.single_port_bridge": "\u5df2\u81ea\u52a8\u9009\u4e2d\u552f\u4e00\u4e32\u53e3\uff1a{port}\u3002\u8fde\u63a5\u65f6\u4f1a\u81ea\u52a8\u4f7f\u7528 Windows \u4e3b\u673a\u4e32\u53e3\u6865\u63a5\u3002",  # 已自动选中唯一串口：{port}。连接时会自动使用 Windows 主机串口桥接。
    "hw.connection.port_hint.single_port": "\u5df2\u81ea\u52a8\u9009\u4e2d\u552f\u4e00\u4e32\u53e3\uff1a{port}\u3002",  # 已自动选中唯一串口：{port}。
    "hw.connection.port_hint.multi_port_bridge": "\u68c0\u6d4b\u5230 {count} \u4e2a\u4e32\u53e3\uff0c\u5f53\u524d\u9009\u62e9 {port}\u3002\u8fde\u63a5\u65f6\u4f1a\u81ea\u52a8\u4f7f\u7528 Windows \u4e3b\u673a\u4e32\u53e3\u6865\u63a5\u3002",  # 检测到 {count} 个串口，当前选择 {port}。连接时会自动使用 Windows 主机串口桥接。
    "hw.connection.port_hint.multi_port": "\u68c0\u6d4b\u5230 {count} \u4e2a\u4e32\u53e3\uff0c\u8bf7\u786e\u8ba4\u5e76\u9009\u62e9\u786c\u4ef6\u5bf9\u5e94\u7aef\u53e3\u3002",  # 检测到 {count} 个串口，请确认并选择硬件对应端口。
    "hw.connection.relay_hint.dynamic": "4G \u4e2d\u7ee7\u5c06\u8fde\u63a5\u5230 {host}:{port}\uff1b\u70b9\u51fb\u8fde\u63a5\u524d\u4f1a\u5148\u505a\u670d\u52a1\u5668\u53ef\u8fbe\u6027\u68c0\u67e5\u3002",  # 4G 中继将连接到 {host}:{port}；点击连接前会先做服务器可达性检查。

    # ==================================================================
    # Hardware tab — Step 2 Control panel
    # ==================================================================
    "hw.control.title": "2. \u53c2\u6570\u8bbe\u7f6e\u4e0e\u8bca\u65ad",          # 2. 参数设置与诊断
    "hw.control.power_header": "\u6d4b\u91cf\u7535\u6e90",                        # 测量电源
    "hw.control.power_on_button": "\u5f00\u542f\u7535\u6e90",                     # 开启电源
    "hw.control.power_off_button": "\u5173\u95ed\u7535\u6e90",                    # 关闭电源
    "hw.control.power_hint": "\u7535\u6e90\u5f00\u5173\u76f4\u63a5\u63a7\u5236\u677f\u5361\u4f9b\u7535\uff1b\u5355\u70b9\u6d4b\u8bd5\u4ec5\u7528\u4e8e\u529f\u80fd\u9a8c\u8bc1\u3002",  # 电源开关直接控制板卡供电；单点测试仅用于功能验证。
    "hw.control.layout_header": "\u786c\u4ef6\u5e03\u5c40",                        # 硬件布局
    "hw.control.rotate_meas_check": "\u6d4b\u91cf\u968f\u6fc0\u52b1\u65cb\u8f6c",  # 测量随激励旋转
    "hw.control.use_meas_current_check": "\u6d4b\u91cf\u6fc0\u52b1\u76f8\u5173\u7535\u6781",  # 测量激励相关电极
    "hw.control.setup_header": "\u6d4b\u91cf\u53c2\u6570",                         # 测量参数
    "hw.control.frequency_label": "\u9891\u7387",                                  # 频率
    "hw.control.freq_apply_button": "\u5e94\u7528",                                # 应用
    "hw.control.stim_amp_label": "\u6fc0\u52b1\u7535\u6d41",                       # 激励电流
    "hw.control.stim_apply_button": "\u5e94\u7528",                                # 应用
    "hw.control.voltage_gain_label": "\u7535\u538b\u589e\u76ca",                   # 电压增益
    "hw.control.vamp_apply_button": "\u5e94\u7528",                                # 应用
    "hw.control.diag_header": "\u8bca\u65ad",                                      # 诊断
    "hw.control.spt_button": "\u5355\u70b9\u6d4b\u8bd5",                           # 单点测试
    "hw.control.impedance_button": "\u963b\u6297\u6d4b\u91cf",                     # 阻抗测量
    "hw.control.layout_grid.mode": "\u6a21\u5f0f",                                 # 模式
    "hw.control.layout_grid.elec_ring": "\u6bcf\u73af\u7535\u6781\u6570",          # 每环电极数
    "hw.control.layout_grid.rings": "\u73af\u6570",                                # 环数
    "hw.control.layout_grid.stim_pattern": "\u6fc0\u52b1\u6a21\u5f0f",              # 激励模式
    "hw.control.layout_grid.meas_pattern": "\u6d4b\u91cf\u6a21\u5f0f",              # 测量模式
    "hw.control.layout_grid.extra_neighbors": "\u989d\u5916\u6392\u9664\u90bb\u5c45\u6570",  # 额外排除邻居数
    "hw.control.cem_grid.radius": "\u534a\u5f84",                                   # 半径
    "hw.control.cem_grid.elec_length": "\u7535\u6781\u957f\u5ea6",                  # 电极长度
    "hw.control.cem_grid.contact_z": "\u63a5\u89e6\u963b\u6297",                    # 接触阻抗

    # ==================================================================
    # Hardware tab — Step 3 Acquisition panel
    # ==================================================================
    "hw.acquisition.title": "3. \u91c7\u96c6\u4e0e\u5f55\u5236",                    # 3. 采集与录制
    "hw.acquisition.flow_hint": "\u8bf7\u8bbe\u7f6e\u4fdd\u5b58\u8def\u5f84\u548c\u91c7\u96c6\u8ba1\u5212\uff0c\u7136\u540e\u542f\u52a8\u91c7\u96c6\u3002",  # 请设置保存路径和采集计划，然后启动采集。
    "hw.acquisition.record_header": "\u5f55\u5236\u8bbe\u7f6e",                     # 录制设置
    "hw.acquisition.save_to_label": "\u4fdd\u5b58\u5230\uff1a",                     # 保存到：
    "hw.acquisition.dir_placeholder": "\u8f93\u51fa\u76ee\u5f55\u2026",              # 输出目录…
    "hw.acquisition.browse_button": "\u6d4f\u89c8\u2026",                            # 浏览…
    "hw.acquisition.record_check": "\u5f55\u5236\u5230\u672c\u5730",                 # 录制到本地
    "hw.acquisition.plan_header": "\u91c7\u96c6\u8ba1\u5212",                        # 采集计划
    "hw.acquisition.timed_interval_check": "\u5b9a\u65f6\u95f4\u9694",               # 定时间隔
    "hw.acquisition.interval_label": "\u95f4\u9694\uff1a",                           # 间隔：
    "hw.acquisition.count_label": "\u91c7\u96c6\u6b21\u6570\uff1a",                  # 采集次数：
    "hw.acquisition.count_continuous": "\u8fde\u7eed",                                # 连续
    "hw.acquisition.freq_step_check": "\u8de8\u9891\u626b\u9891\u91c7\u96c6",         # 跨频扫频采集
    "hw.acquisition.start_freq_label": "\u8d77\u59cb\u9891\u7387\uff1a",              # 起始频率：
    "hw.acquisition.end_freq_label": "\u7ed3\u675f\u9891\u7387\uff1a",                # 结束频率：
    "hw.acquisition.plan_hint": "\u91c7\u96c6\u6b21\u6570 = 0 \u8868\u793a\u65e0\u9650\u8fde\u7eed\u91c7\u96c6\uff1b\u8bbe\u4e3a\u5927\u4e8e 0 \u65f6\uff0c\u5c06\u6267\u884c\u6709\u9650\u6b21\u91c7\u96c6\u5e76\u5728\u5b8c\u6210\u540e\u81ea\u52a8\u505c\u6b62\u3002",  # 采集次数 = 0 表示无限连续采集；设为大于 0 时，将执行有限次采集并在完成后自动停止。
    "hw.acquisition.action_header": "\u91c7\u96c6\u64cd\u4f5c",                       # 采集操作
    "hw.acquisition.start_button": "\u5f00\u59cb",                                     # 开始
    "hw.acquisition.start_button_tooltip": "\u6309\u5f53\u524d\u8ba1\u5212\u5f00\u59cb\u91c7\u96c6",  # 按当前计划开始采集
    "hw.acquisition.single_frame_button": "\u5355\u5e27\u91c7\u96c6",                  # 单帧采集
    "hw.acquisition.single_frame_button_tooltip": "\u91c7\u96c6\u4e00\u5e27\u6570\u636e",  # 采集一帧数据
    "hw.acquisition.stop_button": "\u505c\u6b62",                                       # 停止
    "hw.acquisition.stop_button_tooltip": "\u505c\u6b62\u5f53\u524d\u91c7\u96c6",        # 停止当前采集
    "hw.acquisition.frames_acquired_label": "\u5df2\u91c7\u96c6\u5e27\u6570\uff1a",      # 已采集帧数：
    "hw.acquisition.file_dialog_title": "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55",          # 选择输出目录

    # ==================================================================
    # Hardware tab — Session summary footer
    # ==================================================================
    "hw.summary.field.identity": "\u8eab\u4efd\uff1a",                                   # 身份：
    "hw.summary.field.transport": "\u4f20\u8f93\uff1a",                                   # 传输：
    "hw.summary.field.layout": "\u5e03\u5c40\uff1a",                                      # 布局：
    "hw.summary.field.drive": "\u6fc0\u52b1\uff1a",                                       # 激励：
    "hw.summary.field.record": "\u5f55\u5236\u8def\u5f84\uff1a",                          # 录制路径：
    "hw.summary.field.plan": "\u8ba1\u5212\uff1a",                                        # 计划：
    "hw.summary.indicator.link": "\u8fde\u63a5",                                          # 连接
    "hw.summary.indicator.power": "\u7535\u6e90",                                         # 电源
    "hw.summary.indicator.record": "\u5f55\u5236",                                        # 录制
    "hw.summary.indicator.acq": "\u91c7\u96c6",                                            # 采集
    # Banner — per-state title / detail / action
    "hw.summary.banner.link_down.title": "\u672a\u8fde\u63a5",                                                     # 未连接
    "hw.summary.banner.link_down.detail": "\u6682\u65e0\u5df2\u9a8c\u8bc1\u7684\u8bbe\u5907\u8fde\u63a5\u3002",    # 暂无已验证的设备连接。
    "hw.summary.banner.link_down.action": "\u9009\u62e9\u4f20\u8f93\u65b9\u5f0f\uff0c\u7136\u540e\u70b9\u51fb\u8fde\u63a5\u5e76\u9a8c\u8bc1\u3002",  # 选择传输方式，然后点击连接并验证。
    "hw.summary.banner.fault.title": "\u94fe\u8def\u6545\u969c",                                                    # 链路故障
    "hw.summary.banner.fault.detail": "\u94fe\u8def\u5904\u4e8e\u9519\u8bef\u72b6\u6001\uff0c\u9700\u8981\u64cd\u4f5c\u5458\u5904\u7406\u3002",  # 链路处于错误状态，需要操作员处理。
    "hw.summary.banner.fault.action": "\u65ad\u5f00\u8fde\u63a5\uff0c\u68c0\u67e5\u4f20\u8f93\u8bbe\u7f6e\u540e\u91cd\u65b0\u9a8c\u8bc1\u3002",  # 断开连接，检查传输设置后重新验证。
    "hw.summary.banner.verifying.title": "\u6b63\u5728\u9a8c\u8bc1\u94fe\u8def",                                     # 正在验证链路
    "hw.summary.banner.verifying.detail": "\u5de5\u4f5c\u7ad9\u6b63\u5728\u63a2\u6d4b\u8bbe\u5907\u5e76\u8bfb\u53d6\u534f\u8bae\u80fd\u529b\u3002",  # 工作站正在探测设备并读取协议能力。
    "hw.summary.banner.verifying.action": "\u8bf7\u7b49\u5f85\u94fe\u8def\u9a8c\u8bc1\u5b8c\u6210\u3002",           # 请等待链路验证完成。
    "hw.summary.banner.acquiring.title": "\u6b63\u5728\u91c7\u96c6",                                                 # 正在采集
    "hw.summary.banner.acquiring.detail": "\u6b63\u5728\u4ece\u5f53\u524d\u4f20\u8f93\u94fe\u8def\u91c7\u96c6\u5e27\u3002",  # 正在从当前传输链路采集帧。
    "hw.summary.banner.acquiring.action": "\u67e5\u770b\u5b9e\u65f6\u56fe\u6216\u5b8c\u6210\u540e\u505c\u6b62\u91c7\u96c6\u3002",  # 查看实时图或完成后停止采集。
    "hw.summary.banner.acquiring_recording.title": "\u6b63\u5728\u91c7\u96c6 + \u5f55\u5236",                        # 正在采集 + 录制
    "hw.summary.banner.acquiring_recording.detail": "\u5e27\u6b63\u5728\u91c7\u96c6\u5e76\u5199\u5165\u5f53\u524d\u4f1a\u8bdd\u3002",  # 帧正在采集并写入当前会话。
    "hw.summary.banner.acquiring_recording.action": "\u67e5\u770b\u5165\u6765\u5e27\u6216\u5b8c\u6210\u540e\u505c\u6b62\u91c7\u96c6\u3002",  # 查看入来帧或完成后停止采集。
    "hw.summary.banner.ready_simulator.title": "\u5373\u53ef\u91c7\u96c6",                                            # 即可采集
    "hw.summary.banner.ready_simulator.detail": "\u6a21\u62df\u5668\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u53ef\u7acb\u5373\u5f00\u59cb\u751f\u6210\u5e27\u3002",  # 模拟器链路已验证，可立即开始生成帧。
    "hw.summary.banner.ready_simulator.action": "\u5f00\u59cb\u8fde\u7eed\u91c7\u96c6\u6216\u5355\u5e27\u91c7\u96c6\u3002",  # 开始连续采集或单帧采集。
    "hw.summary.banner.ready_record_armed.title": "\u5c31\u7eea + \u5f85\u5f55\u5236",                                # 就绪 + 待录制
    "hw.summary.banner.ready_record_armed.detail": "\u8bbe\u5907\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u6d4b\u91cf\u7535\u6e90\u5df2\u5f00\uff0c\u4e0b\u4e00\u6b21\u91c7\u96c6\u5c06\u4fdd\u5b58\u3002",  # 设备链路已验证，测量电源已开，下一次采集将保存。
    "hw.summary.banner.ready_record_armed.action": "\u5f00\u59cb\u91c7\u96c6\u5e76\u5f55\u5236\u4e0b\u4e00\u4f1a\u8bdd\u3002",  # 开始采集并录制下一会话。
    "hw.summary.banner.ready.title": "\u5373\u53ef\u91c7\u96c6",                                                      # 即可采集
    "hw.summary.banner.ready.detail": "\u8bbe\u5907\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u6d4b\u91cf\u7535\u6e90\u5df2\u5f00\u3002",  # 设备链路已验证，测量电源已开。
    "hw.summary.banner.ready.action": "\u5f00\u59cb\u8fde\u7eed\u91c7\u96c6\u6216\u5355\u5e27\u91c7\u96c6\u3002",     # 开始连续采集或单帧采集。
    "hw.summary.banner.link_verified_armed.title": "\u94fe\u8def\u5df2\u9a8c\u8bc1",                                  # 链路已验证
    "hw.summary.banner.link_verified_armed.detail": "\u94fe\u8def\u5df2\u9a8c\u8bc1\u4e14\u5f55\u5236\u5df2\u5f85\u547d\uff0c\u4f46\u6d4b\u91cf\u7535\u6e90\u5c1a\u672a\u786e\u8ba4\u5f00\u542f\u3002",  # 链路已验证且录制已待命，但测量电源尚未确认开启。
    "hw.summary.banner.link_verified_armed.action": "\u786e\u8ba4\u786c\u4ef6\u5c31\u7eea\u540e\u5f00\u542f\u6d4b\u91cf\u7535\u6e90\uff0c\u7136\u540e\u5f00\u59cb\u91c7\u96c6\u3002",  # 确认硬件就绪后开启测量电源，然后开始采集。
    "hw.summary.banner.link_verified.title": "\u94fe\u8def\u5df2\u9a8c\u8bc1",                                         # 链路已验证
    "hw.summary.banner.link_verified.detail": "\u8bbe\u5907\u94fe\u8def\u5df2\u9a8c\u8bc1\uff0c\u7b49\u5f85\u6d4b\u91cf\u7535\u6e90\u6216\u4e0b\u4e00\u6b65\u8bbe\u7f6e\u3002",  # 设备链路已验证，等待测量电源或下一步设置。
    "hw.summary.banner.link_verified.action": "\u786e\u8ba4\u786c\u4ef6\u5c31\u7eea\u540e\u5f00\u542f\u6d4b\u91cf\u7535\u6e90\uff0c\u7136\u540e\u5f00\u59cb\u91c7\u96c6\u3002",  # 确认硬件就绪后开启测量电源，然后开始采集。

    # Indicator short chips
    "hw.summary.chip.link.down": "\u65ad\u5f00",      # 断开
    "hw.summary.chip.link.check": "\u9a8c\u8bc1\u4e2d",  # 验证中
    "hw.summary.chip.link.ok": "\u5df2\u8fde",         # 已连
    "hw.summary.chip.link.fault": "\u6545\u969c",      # 故障
    "hw.summary.chip.link.unk": "\u672a\u77e5",        # 未知
    "hw.summary.chip.power.unk": "\u672a\u77e5",       # 未知
    "hw.summary.chip.power.off": "\u5173",             # 关
    "hw.summary.chip.power.on": "\u5f00",              # 开
    "hw.summary.chip.record.off": "\u5173",            # 关
    "hw.summary.chip.record.arm": "\u5f85\u5f55",      # 待录
    "hw.summary.chip.record.rec": "\u5f55\u5236",      # 录制
    "hw.summary.chip.acq.idle": "\u7a7a\u95f2",        # 空闲
    "hw.summary.chip.acq.run": "\u8fde\u7eed",         # 连续
    "hw.summary.chip.acq.sch": "\u5b9a\u65f6",         # 定时
    "hw.summary.chip.acq.fin": "\u6709\u9650",         # 有限
    "hw.summary.chip.acq.step": "\u626b\u9891",        # 扫频
    "hw.summary.chip.acq.1fr": "\u5355\u5e27",         # 单帧

    # Legacy state keys kept for the default first-run indicator states.
    "hw.summary.state.down": "\u65ad\u5f00",                                               # 断开
    "hw.summary.state.unknown": "\u672a\u77e5",                                            # 未知
    "hw.summary.state.off": "\u5173",                                                      # 关
    "hw.summary.state.idle": "\u7a7a\u95f2",                                               # 空闲

    # ==================================================================
    # Hardware tab — Right-side Frame browser
    # ==================================================================
    "hw.frame_browser.title": "\u5df2\u5f55\u5236\u7684\u5e27",                            # 已录制的帧
    "hw.frame_browser.hint": "\u6bcf\u6b21\u91c7\u96c6\u7684\u9996\u5e27\u81ea\u52a8\u4f5c\u4e3a\u53c2\u8003\u3002\u70b9\u51fb\u4efb\u4e00\u5e27\u5e76\u6309\u201c\u8bbe\u4e3a\u53c2\u8003\u201d\u53ef\u8986\u76d6\uff1b\u6700\u65b0\u91c7\u96c6\u7684\u5e27\u59cb\u7ec8\u4f5c\u4e3a\u76ee\u6807\u3002",  # 每次采集的首帧自动作为参考。点击任一帧并按"设为参考"可覆盖；最新采集的帧始终作为目标。
    "hw.frame_browser.count_label": "\u5df2\u5f55\u5236\u5e27\u6570\uff1a{count}",          # 已录制帧数：{count}
    "hw.frame_browser.column.index": "\u5e8f\u53f7",                                        # 序号
    "hw.frame_browser.column.timestamp": "\u65f6\u95f4\u6233",                              # 时间戳
    "hw.frame_browser.column.file": "\u6587\u4ef6",                                         # 文件
    "hw.frame_browser.set_ref_button": "\u8bbe\u4e3a\u53c2\u8003",                          # 设为参考
    "hw.frame_browser.clear_button": "\u6e05\u7a7a\u5217\u8868",                            # 清空列表

    # ==================================================================
    # Hardware tab — Live measurement plot
    # ==================================================================
    "hw.live_plot.title": "\u5b9e\u65f6\u6d4b\u91cf\u901a\u9053",                            # 实时测量通道
    "hw.live_plot.y_label": "\u7535\u538b (V)",                                             # 电压 (V)
    "hw.live_plot.x_label_dynamic": "\u6d4b\u91cf\u5e8f\u53f7 (1-{count})",                  # 测量序号 (1-{count})
    "hw.live_plot.curve.real": "\u5b9e\u90e8",                                               # 实部
    "hw.live_plot.curve.imag": "\u865a\u90e8",                                               # 虚部
    "hw.live_plot.empty_overlay": "\u8fd8\u6ca1\u6709\u5b9e\u65f6\u5e27\u3002\n\u542f\u52a8\u91c7\u96c6\u540e\u5c06\u663e\u793a\u5b9e\u90e8\u4e0e\u865a\u90e8\u3002",  # 还没有实时帧。\n启动采集后将显示实部与虚部。

    # ==================================================================
    # Hardware tab — Reconstruction widget
    # ==================================================================
    "hw.reconstruction.title": "\u91cd\u6784\u56fe\u50cf",                                    # 重构图像
    "hw.reconstruction.empty_overlay": "\u6682\u65e0\u91cd\u6784",                            # 暂无重构
    "hw.reconstruction.error.expect_2d_triangles": "\u5feb\u901f\u91cd\u6784\u89c6\u56fe\u76ee\u524d\u4ec5\u652f\u6301 2D \u4e09\u89d2\u7f51\u683c",  # 快速重构视图目前仅支持 2D 三角网格

    # ==================================================================
    # Hardware tab — Boundary voltage fit plot
    # ==================================================================
    "hw.boundary.title": "\u8fb9\u754c\u7535\u538b\u62df\u5408",                              # 边界电压拟合
    "hw.boundary.y_label": "\u7535\u538b (V)",                                                # 电压 (V)
    "hw.boundary.x_label_dynamic": "\u8fb9\u754c\u7535\u538b\u5e8f\u53f7 (1-{count})",         # 边界电压序号 (1-{count})
    "hw.boundary.primary.measured": "\u5b9e\u6d4b",                                            # 实测
    "hw.boundary.primary.ground_truth": "\u771f\u503c",                                        # 真值
    "hw.boundary.secondary": "\u91cd\u6784\u62df\u5408",                                       # 重构拟合
    "hw.boundary.empty.hardware": "\u91cd\u6784\u66f4\u65b0\u540e\u5c06\u663e\u793a\u5b9e\u6d4b\u4e0e\u62df\u5408\u7684\u8fb9\u754c\u7535\u538b\u66f2\u7ebf\u3002",  # 重构更新后将显示实测与拟合的边界电压曲线。
    "hw.boundary.empty.simulation": "\u6b63\u95ee\u9898\u6216\u9006\u95ee\u9898\u66f4\u65b0\u540e\u5c06\u663e\u793a\u771f\u503c\u4e0e\u62df\u5408\u7684\u8fb9\u754c\u7535\u538b\u66f2\u7ebf\u3002",  # 正问题或逆问题更新后将显示真值与拟合的边界电压曲线。

    # ==================================================================
    # Shared plot-legend overlay (used across Hardware and Simulation)
    # ==================================================================
    "plot_legend.drag_tooltip": "\u53ef\u62d6\u62fd\u8c03\u6574\u56fe\u4f8b\u4f4d\u7f6e",   # 可拖拽调整图例位置
}
