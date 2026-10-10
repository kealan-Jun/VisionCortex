# Ubuntu RTX 3090 Ti 部署

[文档导航](../../docs/README.md) · [本地启动](../../docs/guides/local-development.zh-CN.md) · [交付验收](../../docs/VisionCortex-交付验收单模板.md)

首次克隆先检查本地开发环境；生产推理需要匹配目标 GPU 的独立运行环境、
已认证模型和私有站点配置。硬件配置不是某台已验收主机的证据。

## 环境与配置

使用项目支持的 Python 版本以及所选部署依赖锁。安装脚本、解释器和配置都应来自
同一个已核对的源码版本。安装前核对磁盘容量和依赖来源，不迁移或覆盖活跃环境。
本目录脚本面向已配置的站点；先检查其参数和私有配置要求，不执行带其他站点路径的命令。

`05-Install-Local-Service.sh` 使用当前克隆和 `.venv`，默认本地配置为
`configs/development-local.yaml`，仅监听 `127.0.0.1:8002`；端口可通过
`VISIONCORTEX_WEB_PORT` 显式设置。桌面入口对应当前服务，登录自动打开需显式设置
`VISIONCORTEX_DESKTOP_AUTOSTART=1`。

公开本地配置关闭 NAS 和自动采集。生产配置由部署方提供输入索引、存储根、
设备绑定、服务地址和凭据位置；这些站点信息不进入源码或客户通用文档。
生产脚本必须显式提供 `VISIONCORTEX_CONFIG` 或 `VISIONCORTEX_SITE_CONFIG`，
拒绝直接启动未完成站点配置的模板。`06-Run-LAN-Server.sh` 的模式通过
`VISIONCORTEX_DEPLOYMENT_MODE=local|production` 指定。服务文件须经同目录
`render_service.py` 按当前克隆和解释器渲染后验证，不能直接复制带占位符的模板。
缺少路径或挂载身份验证时停止，不能建立本地替代目录。

## 模型与引擎

源码不包含模型二进制。闭集模型由注册表固定类别与内容身份；实际模型来源由部署方提供，
校验通过后才准备运行。公共资产须满足其许可证、固定版本及摘要要求。
TensorRT 引擎由目标 GPU、驱动和运行栈生成，保存在本地；不得直接复用另一台机器的引擎。

请求 batch 受引擎实际容量约束，提高配置值不会扩大引擎。
容量试验使用独立本地输入和输出，比较同一输入的帧身份、动作选择、实际 batch 与显存回执。
实验候选不会自动晋升为生产模型或吞吐结论。

## 服务与存储

源码、环境、当前模型、SQLite 队列/租约及活动缓存使用本地磁盘。
原始数据和已完成归档依站点合同保存；缓存有明确容量和保留策略。
核对输入索引与归档身份后才启动生产服务，不复制原片或改写既有归档路径。

Web 与 worker 按配置的角色运行；认证、监听地址及跨机访问由站点部署配置决定。
切换版本前核对所属服务和活动任务并排空。源码同步、安装、激活和服务重启分别执行，
运行中不能因更新源码而中断其他任务。

### 拆分服务中的归档恢复

使用独立阶段消费者时，额外部署 `visioncortex.device_day_recovery_worker`；Web、
元数据发现和阶段消费者均不能替代归档恢复。保持原单体处理调度关闭，避免重复执行。
恢复消费者每次核验一条封存快照的完整归档字节，凭回执身份、内容摘要和稳定文件身份
恢复留存完成及输入可用性，不修改采集源，也不调用模型或重置其他阶段。

私有配置的 `device_day.recovery.storage_checks` 必须列出实际挂载身份，每项包含
`mount_path`、`filesystem`、`source`、`marker_path`、`marker_id`，以及可选
`marker_id_field`（默认 `volume_id`）。挂载须与 mountinfo 精确一致，身份标记须在
该挂载内；每次核验前重新检查，失败时停止本次恢复。运行队列与核验检查点保持本地。
`batch_size` 默认 128、上限 512；未改变的任务检查点默认冷却 900 秒，可用
`cooldown_seconds` 设置。重启后沿用冷却检查点，内容信任仍必须由当前进程重新校验。

按已验证的解释器、固定源码和私有配置渲染服务，再执行所属站点的排空和激活流程：

```bash
"$VISIONCORTEX_PYTHON" deployment/rtx3090ti-ubuntu/render_service.py render-recovery \
  "$VISIONCORTEX_PROJECT_ROOT" "$VISIONCORTEX_PYTHON" "$VISIONCORTEX_CONFIG" \
  deployment/rtx3090ti-ubuntu/visioncortex-recovery.service "$RECOVERY_UNIT_PATH"
```

本地运行根的 `device-day/RetentionRecovery/Service.json` 包含进程身份、源码身份、
持续心跳、已完成 tick 数量、最近结果及恢复数量。验收同时检查心跳和实际 tick
结果；进程在线不能证明归档核验成功。旧厂商 gate 仅在激活、适用于当前厂商且到期时
复用原探测逻辑；现代 circuit 开启后仍需恢复其兼容的激活旧标记。没有适用旧标记时，
现代 scoped circuit 继续由请求 transport 执行单请求半开恢复。

## 验收与回退

拆分阶段的历史调度由 `device_day.backfill` 明确配置。`mode: idle` 保留空闲补跑；
持续有新输入的站点可选择 `mode: fair`，并在 `stages` 列出
`retention`、`vision`、`stt`、`understanding`、`report`。每阶段必须部署对应 worker。
`fair_interval_seconds` 默认 60；`quantum_seconds` 在可恢复单位完成后让出，
不会打断已发送的付费请求或接受半份内容。历史按最老输入和阶段轮转处理，实际执行
仍验证输入与前序回执，并加入有容量上限和等待老化的共享资源队列。
`fair_turn_timeout_seconds` 默认 15，避免缺失消费者一直占住历史轮次。
monitor、用户暂停、处理时段和内容验证失败仍会阻止不安全执行。
历史和实时消费者并行轮询，通过共享资源容量与独占任务租约协调；历史执行不会关闭实时准入。

无独立录音的分片同样进入 STT 阶段，实际探测视频内嵌音轨后才记录 `no_audio`；
未发布录音或关联错误继续等待。录音晚到只重入受影响阶段，旧回执保留在历史中。

达到失败预算的任务须先核验具体修复，再通过 `visioncortex.device_day_retry`
的 `plan` / `apply` 授予最多三次追加尝试。证据绑定来源、队列修订、原失败摘要
和稳定修复身份；凭据修复额外核对既有私有验证记录。累计 attempts 和原失败保留，
同一修复不能反复续预算。计划文件、修复证据及授权日志属于站点私有运行数据。
不能以清空队列或无限重试代替故障恢复。

依次核对解释器及依赖、CUDA/TensorRT、模型/引擎身份、输入指纹、存储容量、
服务健康与任务回执。健康响应或合成媒体通过仅证明相应接口/结构。
真实模型执行、真实视频质量、浏览器交付、恢复及稳定生产分别记录证据。

回退保留队列与归档，不用旧空库覆盖任务数据，不删除现场输入或输出。
使用[交付验收单](../../docs/VisionCortex-交付验收单模板.md)记录实际目标机结果。
