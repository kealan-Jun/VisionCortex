# VisionCortex 统一运行层

当前配置与运行契约；现场部署记录由部署方私有保存。

## 已落盘的代码

| 项目 | 实现与行为 |
| --- | --- |
| 共用阶段 | `stage_execution.py` 定义依赖和资源边界，调用原阶段函数。离线 `pipeline.py` 的录音异步提交，视觉独立推进，在需要录音的语义步骤前汇合。 |
| 共用资源 | `runtime_control.py` 用本地 SQLite 租约协调 API、CLI 和 NAS 的视觉、存储、CPU、云端及 STT 资源。支持等待超时、取消、优先级老化；相机仍动态发现。 |
| Web/worker 分离 | `runtime_process.py` 约束同一运行根只有一个后台拥有者。`web` 提供接口，`worker` 消费离线与自动采集队列，`combined` 保留原启动方式。`run_queue.py` 事务合并最新状态，避免旧内存快照覆盖进度；worker 读取持久化重试状态，关闭时归还已取消任务并保留原封存输入。 |
| 增量更新 | `observed_inventory.py` 更新变化记录，兼容旧 JSON。`receipt_projection.py` 首次载入一天回执，后续更新变化分片。`timeline_invalidation.py` 持久化变化日期和范围，无变化不重建所有历史日期。同设备日的 comment/protocol 在一次索引刷新中只读一次。 |
| 缓存依赖 | `stage_dependencies.py` 和 `cache-impact` 输出阶段变更计划。执行逻辑变化生成新身份，不自动接受未知旧缓存或重写历史来源。 |
| 生命周期 | `shared_inference.py` 限定模型池、请求等待和关闭行为；调用者取消不取消其他相机请求。`owned_subprocess.py` 管理本地 ASR 子进程超时和取消。`sqlite_store.py` 明确事务及连接关闭。 |
| 发布恢复 | `publication_journal.py` 在处理前记录待发布项，按代际确认。恢复前排除仍在运行的阶段，避免提前确认；不可用项延期，不阻塞其他日期。索引锁竞争保留待发布状态，后台只补发布，不为补索引重新推理。 |
| 共用读取 | `media_time.py` 提供正反时间映射；`artifact_reader.py` 区分缺失、权限和 I/O 故障。旧离线目录内合法链接继续兼容，NAS 归档读取遵守冻结契约。 |
| 厂商隔离 | `provider_control.py` 按厂商、连接、凭据引用和服务隔离熔断。欠费/凭据问题按账户，限流/网络问题按服务隔离；恢复由一条实际请求探测，其他阶段独立运行。 |
| 可见状态 | `/api/runtime` 和运行状态页面展示 worker 心跳、代码身份、离线/NAS 资源请求及等待/运行状态。`/health/live` 不扫描 NAS、不启动模型。 |
| 配置与升级 | `runtime_options.py` 校验角色和资源配置；`deployment/runtime/` 提供内容校验、隔离安装、原子切换和回退脚本。API 与 worker 使用同一构建身份。 |

## 固定边界

设备日五目录没有改变：`MetaVideo`、`ProcessedClips`、
`MultimodalUnderstanding`、`LaboratoryDailyReport`、`Comment`。
没有新增 `AlignedExperiment...` 等对外目录规则。
新数据库、租约、发布日志和失效通知均放在配置的本地运行根。

离线入口、归档阅读器、第一/第三人称分析和对齐算法保留。
时间线保留跨分片连续性；变化触发相关日期的共享分析，复用原有源级/日期级缓存。
没有把跨视角分析强行切成彼此无关的小任务，也没有宣称所有阶段已实现范围级重算。

关键帧仍只用于五类物理动作，无活动区间用 `scene_frames`。
录音及 STT 保留实际设备/日期。缺失来源不阻塞其他有效输入；权限和 I/O
故障不会当成“没有数据”；存储不可见与输入为空分别记录。
维护状态继续生效，没有扩大删除授权。

## 配置与入口

`runtime.role` 为 `combined`、`web`、`worker`，环境变量
`VISIONCORTEX_RUNTIME_ROLE` 可覆盖角色。缺省 `combined` 兼容原入口。
worker 可通过 `visioncortex worker --config <配置>` 独立启动。

`runtime.resource_limits` 支持 `vision/storage/cpu/cloud/stt`，各值须为
1–256 的整数。缺省空字典兼容原配置。所有需要共享额度的入口必须使用
相同 `storage.local_runtime_root` 和资源上限；运行中配置冲突拒绝准入。

3090 Ti 生产配置写入视觉 12、存储 2、CPU 6、云端 4、STT 2 的上限。
这些是资源预算，不是固定相机数，也不是 TensorRT batch 大小；原有引擎
批量、供帧和 CPU/GPU 调度仍生效。`runtime.admission_timeout_seconds`
默认 300 秒。租约持续续期，进程死亡后过期释放。

`runtime.local_only: true` 禁止 NAS 采集和同步，检查输入、缓存、暂存和
归档根落在本地运行根下。此检查不挂载存储，不验证操作系统实际挂载关系；
部署时仍需保证运行根确实为本地磁盘。

`runtime.provider_circuit_enabled` 默认关闭以兼容既有配置，3090 Ti
生产配置开启。只保存凭据引用，不保存密钥。录音可使用
`speech_recognition.connection`，未配置时沿用既有连接解析。

自动预处理持续入队；自动多模态和日报仍在 20:30–次日 08:00 窗口运行。
无新增输入的日期不生成空报告。用户主动提交的离线任务按需执行。

## 版本部署和回退

脚本位于 `deployment/runtime/`，支持 `--help`。执行前核对固定源码、站点配置、依赖及适用授权。

1. 固定已审查提交并通过适用发布门后，在构建环境生成不可变
   `Releases/<版本>/`，包含 `Wheelhouse/`、带哈希的 `requirements.lock`、
   `Acceptance.json` 及配置。只允许一份 VisionCortex wheel；普通版本锁
   不能冒充 `--require-hashes` 所需安装锁。
2. `python Release.py seal <版本目录> --commit <40位提交SHA> --version <版本>`
   校验接受回执与 SHA 一致，生成 `BuildManifest.json`。内容校验不代替
   完整发布门、目标机运行证据或正式发布权限。
3. `python Install.py <发布根> <版本>` 从本地 wheelhouse 安装到
   `Environments/<摘要>/`，禁止联网解析依赖；`pip check` 成功才写安装回执。
   失败环境保留诊断，不覆盖其他版本，也没有启动资格。
4. `python Release.py activate <发布根> <版本>` 校验包和已安装环境，
   原子切换 `Current`，记录 `Previous`。安装和切换共用排他锁。
5. `runtime.env` 配置 `VISIONCORTEX_RELEASE_ROOT`、
   `VISIONCORTEX_DEPLOY_TOOLS`、`VISIONCORTEX_CONFIG`，按环境安装
   `visioncortex-web.service`、`visioncortex-worker.service`。
   HTTP 模板只监听本机 8001，对外访问沿用项目认证/代理配置。
6. 维护结束后，停止旧 `combined` 拥有者，再启动匹配版本的 Web/worker。
   核对 `/health/live` 和 `/api/runtime` 的构建身份、心跳，再恢复队列消费。
   切换脚本本身不重启服务、不迁移归档。
7. `python Release.py rollback <发布根>` 切回上一程序/依赖环境，再重启
   对应角色。不会回滚或删除 NAS 数据；后续不兼容数据库迁移需独立恢复方案。

Linux 模板通过控制组收尾进程。Python 无法安全强杀卡住的原生 CUDA 线程，
超时后隔离模型实例，拒绝无限创建替代实例，需进程级重启恢复。
Windows 的子进程取消目前保证直接持有进程；完整子树收尾依赖部署宿主，
不能将 Linux 控制组行为当成 Windows 证据。
