# VisionCortex 统一运行层：实现与交接

本次为代码开发。离线上传、CLI 分析和 NAS 自动采集继续使用同一个系统；
入口和触发时间可以不同，检测、粗扫精扫、对齐、物理动作、语义理解和
历史产出读取仍复用现有实现。

基线分支：`codex/rtx3050-device-delivery-20260904`。
基线提交：`7db0dcd052891e0e189c5e931a27558256fea032`。
初稿包含当时尚未提交或部署的工作区修改；基线 SHA 不代表修改后的版本。
当前代码同步状态以 Git 提交与两仓分支为准，部署状态以运行服务记录为准。

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
故障不会当成“没有数据”。本轮未访问 NAS、调用模型、替换原视频或重启服务。
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

脚本位于 `deployment/runtime/`，支持 `--help`。本轮没有执行实际安装或切换。

1. 两个仓库同步当前提交并通过适用发布门后，在构建环境生成不可变
   `Releases/<版本>/`，包含 `Wheelhouse/`、带哈希的 `requirements.lock`、
   `Acceptance.json` 及配置。只允许一份 VisionCortex wheel；普通版本锁
   不能冒充 `--require-hashes` 所需安装锁。
2. `python Release.py seal <版本目录> --commit <40位提交SHA> --version <版本>`
   校验接受回执与 SHA 一致，生成 `BuildManifest.json`。内容校验不代替
   对应提交 CI 通过、完整发布门或正式发布权限。
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

## 检查与交接状态

本轮检查使用临时目录、假回调和本地源码，覆盖共享额度、取消、超时、
SQLite 回滚、清单迁移、发布代际、厂商恢复竞态、离线录音并行、历史读取和
维护模式。最终 28 个模块共 **586 项通过，0 失败、0 错误、0 跳过**，
检查耗时 29.475 秒（仅为本地检查耗时，不是视频处理性能）。结果文件：
`/tmp/VisionCortexUnifiedRuntimeFinalTests.xml`。

同时通过 `ruff check src tests deployment/runtime`、
`python -m compileall -q src tests deployment/runtime`、
`node --check src/visioncortex/web/app.js` 和 `git diff --check`。
`worker`、`Release.py`、`Install.py` 帮助入口可执行；配置中的录音适配器摘要
与当前源码一致。这些结果不表示服务已经加载新版本。

结论为 `PARTIAL_EVIDENCE`：代码及确定性检查已完成；线上加载和稳定发布
为 `NOT_PROVEN`，尚缺当前工作区正式构建/CI、版本接受及部署交接。
用户正在维修 NAS，因此本轮没有越过维护边界执行上述步骤。

恢复时须使用新代码身份和阶段变更计划，不沿用旧 `SourceTimeDeployment`
八文件快照。`visioncortex cache-impact --before <旧SourceManifest.json>`
只读源码摘要，列出变化阶段，不访问视频，不自动接受旧缓存。
历史结果保留原执行来源，不使用统一白名单放行未知 CV 修改。

## 2026-09-15 NAS 修复后的恢复核对

用户已确认 NAS 修复，解除此前禁止访问 NAS 的维护核对限制。
共享最初可读写：只用自建临时目录完成写入、fsync、原子重命名和读取，
检查后清理了这些自建临时文件，没有删除或替换采集原片。
随后核对输入时 CIFS 进入 `cifs_wait_for_server_reconnect`；当前挂载服务器
`169.254.10.62:445` 拒绝连接。后台维护开关仍为 1，未重启或切换服务。

恢复准备记录位于
`/srv/sentinel-data/VisionCortex3090Ti/Runtime/NasResume20260915/`：
`QueueBackup/` 是本地 SQLite 一致性备份，`ResumeStatus.json` 为当前恢复状态，
`CheckpointPreparation.json` 记录逐回执的文件可用性检查。
历史“已归档”回执中，781 条存在引用文件缺失或大小不符；这只证明本次
挂载视图与回执不一致，不能据此断言文件已被删除或判断删除责任。
输入可用性核对脚本在连接中断后已停止，未执行其队列调整；前后各阶段
队列状态计数一致。需要连接恢复后重新核对，不能采用未完成的输入调查。

代码增加可选 `completed_stage_receipts` 恢复清单，环境覆盖为
`VISIONCORTEX_COMPLETED_STAGE_RECEIPTS` 和
`VISIONCORTEX_COMPLETED_STAGE_RECEIPTS_SHA256`，两者必须同时提供。
清单只接受逐条指定的阶段、输入/配置/代码期望键、原回执摘要、原队列
版本和分片 ID。保留历史完成队列的原版本，不把旧结果改称新模型产出；
实际用于下游时仍校验素材哈希。历史可读性与当前文件可用性分别记录，
不能以清单代替原片完整性验证。未启用时沿用原严格失效规则和旧版引擎
衔接入口。归档五目录没有改变。

最新恢复相关检查：4 个模块 128 项通过、0 失败；结果文件
`/tmp/VisionCortexNasResumeTests.xml`。Ruff、compileall、JavaScript 语法和
diff 空白检查通过。执行环境为 Python 3.12.13、TensorRT cu12 10.16.1.11，
本轮未调用 YOLO 或云模型。

准备了只读源码快照 `RuntimeSource/`，其内容摘要为
`501ccc92239534db0b339913d6c18559f238af67a650f85888da2bf8cd3ec92c`；
它来自上述基线 SHA 加未提交修改，是开发运行验证快照，尚未加载，
不是通过对应提交 CI 和正式发布门的构建。不得将其基线 SHA 作为
整份快照已经被 CI 接受的证明。当前结论仍为 `PARTIAL_EVIDENCE`；
恢复运行尚缺稳定的 NAS 连接、完整输入核对与实际运行回执。


## 2026-09-15 双仓策略变更

用户要求两个仓库直接同步同一个提交，取消开发仓与稳定仓的职责划分。
本文件上文记录的历史检查和未部署状态不因此改变。代码推送以两端目标 SHA
一致为完成条件；安装包封存和服务上线仍有独立的运行证据要求。
`Acceptance.json` 新格式使用 `commit_sha`，旧 `development_sha` 只作兼容读取，
不再要求某一指定仓库先行接受。两字段同时提供时必须一致。

用户同日进一步明确：常规双仓同步只做代码管理，不执行 CI 自检。
自动 push/PR 触发已取消，工作流保留手动入口；本次此前的检查结果仅作为
历史记录，不再为代码同步补跑或等待 CI。
