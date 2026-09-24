# 保持原服务在线的设备日执行器交接

`device-day-companion` 是持续运行的正式设备日执行入口，使用同一阶段队列、
原位预处理实现、资源配额、回执、五目录和发布流程。它不占用原宿主的
`Worker.lock`，不启动第二套 NAS 视频扫描，不启动完整的 `runtime_services`。
原分析宿主和网页进程继续运行；`WorkerStatus.json` 中原宿主的版本不会冒充
新执行器版本。新执行器的真实 PID、构建身份、心跳及管线状态记录在
`state/DeviceDayCompanionStatus.json`。

## 交接边界

旧版 `334d158` 不认识原位输入身份。仅靠阶段队列租约或每分片锁，不能防止
旧版重新入队、重建日索引或处理新回执。因此交接必须使旧版设备日调度线程及
它拥有的执行线程自然结束。不得在两个版本之间伪造完成状态、修改来源身份或
无限续租已经完成的任务。

1. 固定新代码 SHA，校验解释器、配置、模型身份与存储根；准备独立的旧宿主配置，
   唯一调度差异为 `device_day.enabled: false`。保留共享网页/新执行器配置的
   `enabled: true`、相机绑定和用户现有参数。
2. 为旧分析宿主安装**下次启动才使用**的配置入口，并核对 systemd 实际
   `ExecStart` 使用独立禁用配置。`daemon-reload` 不代表重启服务。
3. 在旧调度器仍读取的共享配置中短暂关闭设备日调度。旧 `_loop` 退出后会在
   executor 的自然收尾中等待已有任务；它不设置任务取消事件。不发送停止、
   重启或强杀指令。用非暂停栈读取确认它已经进入退出收尾后即可恢复共享配置。
   这个短暂窗口内网页进程保持在线，但 NAS 自动处理状态字段与手工批次路由
   会按关闭配置解释，应记录窗口长度与实际影响。
4. 确认交接前记录的旧 dispatcher 原生线程 ID 已消失。dispatcher 退出其
   `ExitStack` 后，所拥有的阶段、恢复、索引、总览、发布等执行器已经收尾。
   同时核对原任务租约及回执，保存真实交接证据。不能只看“配置已关闭”。
5. 启动并持久化新 companion，验证真实领取、模型执行、先预处理后归档以及
   最终索引。没有运行证据时只可声明部署或实现，不能声明吞吐已提高。

启动接口：

```bash
visioncortex device-day-companion --config /绝对路径/生产配置.yaml \
  --legacy-config /绝对路径/旧宿主禁用配置.yaml \
  --handoff-receipt /绝对路径/VerifiedHandoff.json
```

3090 Ti 的两份执行配置可分别继承同目录的
`rtx3090ti-ubuntu-production.yaml`：`rtx3090ti-device-day-companion.yaml`
启用新执行器并声明候选代码对应的语音适配器摘要；
`rtx3090ti-legacy-host.yaml` 只关闭旧宿主设备日调度。共享配置继续保留旧宿主
与网页兼容的摘要。这样后续相机绑定更新仍由同一共享配置传递，不能把新摘要
直接覆盖到仍运行旧代码的宿主配置。

新 companion systemd 服务的关键参数示例（实际固定提交路径与回执由部署生成）：

```ini
[Service]
WorkingDirectory=/绝对路径/不可变源码目录
Environment=PYTHONPATH=/绝对路径/不可变源码目录/src
ExecStart=/绝对路径/已校验环境/bin/python -m visioncortex.cli device-day-companion --config /绝对路径/configs/rtx3090ti-device-day-companion.yaml --legacy-config /绝对路径/configs/rtx3090ti-legacy-host.yaml --handoff-receipt /绝对路径/VerifiedHandoff.json
Restart=on-failure
RestartSec=15
KillMode=mixed
TimeoutStopSec=infinity
```

`KillMode=mixed` 使初始停止信号只到主进程；不得用默认 control-group 信号同时
终止正在解码、转写或裁片的子进程。旧宿主的下次 `ExecStart` 使用旧宿主配置，
本次不重启它；新 companion 使用自己独立的 systemd unit。

交接回执要求：

```json
{
  "schema_version": "visioncortex-device-day-handoff/1",
  "config_path": "/绝对路径/生产配置.yaml",
  "legacy_config_path": "/绝对路径/旧宿主禁用配置.yaml",
  "runtime_root": "/绝对路径/Runtime",
  "legacy_pid": 1234,
  "legacy_start_ticks": 123456,
  "legacy_dispatch_thread_ids": [1235],
  "legacy_dispatch_drained": true,
  "legacy_restart_configuration_verified": true
}
```

PID、启动时钟与线程 ID 必须来自现场；这些数字只是格式示例。两个 `true`
必须由完成现场核查后写入，不能用默认值替代取证。实现检查配置路径、根目录、
禁用状态及 `/proc` 中旧线程是否已经消失；下次启动配置的证据由部署记录负责。

原 NAS 宿主即使之后使用禁用配置重新启动，仍会生成本地
`state/nas-recording-monitor.json`。companion 将成功的本地监控快照转入原
持久清单；重试快照不被冒充新鲜的输入可用性证据。旧宿主仍存活时继续负责
原有文件/照片索引；它退出后，companion 接管禁用设备日宿主不会启动的这两类
辅助索引。通用资源、任务事件与进度服务仍由宿主负责。NAS 宿主本身离线时，
已有队列可继续处理，但新输入发现必须如实显示尚未更新。

## 退出与回退

companion 的 SIGTERM/SIGINT 只停止新领取，等待已有任务及发布自然完成，
不调用设置任务取消事件的 `DeviceDayService.stop()`。部署 unit 应使用足够的
自然收尾时间（例如 `TimeoutStopSec=infinity`），不得沿用 180 秒后强杀策略。
恢复旧版本执行器前，必须先让 companion 收尾，并核对新原位回执的兼容性；
不能直接把旧宿主配置改为启用，让旧版消费新格式结果。
