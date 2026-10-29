# 统一实验链路的集成接口

[文档导航](../README.md) · [算法与业务逻辑](lab-vision-algorithms.zh-CN.md) · [留存契约](unified-retention.zh-CN.md) · [当前边界](lab-vision-status.zh-CN.md)

VisionCortex 同时承载视频、FR识别及实验时间线业务能力。一次安装、同一仓库；`visioncortex.lab` 是所属业务模块，不是另建的客户项目。客户名称只属于本地部署与私有配置，设备清单、原片、材料批次及数据库实例不随源码发布。

## 启动与职责

从完整源码检出目录进行 editable 安装：

```bash
python -m pip install -e ".[dev,lab]"
visioncortex lab --help
visioncortex lab verify-components
visioncortex lab verify-rules
```

也可以使用 `python -m visioncortex.lab`；两者调用同一个CLI。当前兼容组件、契约和迁移依赖完整源码检出，不宣称仅安装一个不带资源的wheel就具备运行条件。旧视频CLI及Web入口保留，实验数据服务不新增人工上传产品流程。

复制 [环境变量示例](../../configs/lab/env.example) 到版本控制之外，配置专用PostgreSQL和服务令牌。按明确部署范围执行 `migrate`、`init`、`run` 或 `serve`；代码同步不会执行这些命令。`native` 为兼容默认，`timeline-v1` 显式启用时间线；模型、输入、存储和发布分别配置。缺少模型或挂载身份验证时停止对应阶段，不生成模拟成功。

服务由 [api.create_app](../../src/visioncortex/lab/api.py) 组装；除健康检查外使用Bearer令牌，并固定在实例的租户范围。集成方负责Agent界面、用户核对及调用编排。本文说明已有内部API与业务命令，不增加一套Agent专用工具协议。

## 现有模型与分析服务

沿用已有VC／FR权重和对应解释器，不要求另准备一套客户权重。权重文件、绝对路径、令牌及实际服务配置继续保留在私有部署中；源码提交不复制模型、不下载模型，也不自动启动推理。

[ModelRegistry](../../src/visioncortex/lab/processing.py) 的本地vision／ocr描述可以成对提供 `reuse_source=visioncortex|field_recognition` 和 `model_root`，引用现有本地权重目录；每个运行引用仍须登记并核对SHA256及字节数。未提供复用描述时保持原 `runtime_root/models` 行为。根目录、解释器及模型版本必须明确，NAS模型根或指纹不符会阻止加载。复用说明的是模型来源，不能替代识别质量验收。

真实本地ASR／OCR服务及VLM进程管理源码位于 [`tools/inference`](../../tools/inference)，部署入口见[本地分析服务](../../deployment/inference/lab-services.md)。这是现有分析服务的配置化移植，保留回环地址、鉴权、文件校验、容量边界与自有进程检查；无需在本模块另建一个Agent系统。视频完整片段与原生像素策略仍由任务合同决定，服务的图像请求上限不能冒充完整视频已被消费。

## 一条物理时间线上的业务事件

主动扫码是实验员的操作声明，与后台画面偶然看见二维码有不同语义。

| 输入 | 处理与归属 |
| --- | --- |
| 主动设备扫码 | 已登记且在事件时刻有效的设备身份解析成功后，从该时刻建立当前操作设备。切换只改变当前操作上下文，不结束之前实验、释放设备或换一批材料。 |
| 主动材料扫码 | 按扫码发生时刻解析材料单元、来源批次及标签绑定版本，关联明确的实验／目标制备批次；不产生实际消耗、扣库或未知量的零值。 |
| 后台二维码观察 | 保存像素、原始文本及时间证据；不调用主动扫码业务命令，不伪造实验员意图。 |
| 口述Comment | 保存原音、转写正文、字词／片段时间范围和出处。它与扫码、画面及读数共同组成实验记录；佩戴人不自动等于说话人，转写不自动变成事实或用料指令。 |
| OCR／视频观察 | 保存设备／面板／字段、时刻、候选状态及源画面。当前主要操作设备不能替代画面中每个面板的独立归属。 |

同一样品／目标批次可在不同设备上连续处理，也可并行占用多台设备。同一设备可关联多个样品。实验记录、当前操作上下文和设备持续运行状态是不同记录；不得用一次扫码切换把其他记录自动关闭。

主动扫码使用 `POST /internal/commands/timeline-operator-scan`，历史查询使用 `POST /internal/commands/timeline-operator-context`。事件必须显式标明 `intent=operator_scan`、幂等key、发生时刻、行为人／采集来源与依据；`scan_kind` 区分 `equipment`、`material`，可附明确 `run_id`／`batch_id`。准确参数以源码方法签名及鉴权后的OpenAPI为准。接收时间不能冒充未校准的原始发生时间。只知道设备时允许先记录设备上下文，缺少材料／实验信息保持待关联。

集成方应提供真实主动扫码事件，不能把每个视频帧中的解码结果包装成该命令。本接口依据被授权的主动声明建立关联，不证明所有视觉识别结果都正确。

材料扫码没有重复传入目标时，只能继承同一人／来源在该发生时刻之前已经明确的实验／批次上下文；解析结果、标签版本和继承依据固定在事件快照中。之后晚到的其他声明不能悄悄重挂这条材料记录。没有明确上下文时保持待关联。

## 口述完成与下一步计划

`POST /internal/commands/timeline-comment-transition` 对应 [CommentTransitions.apply](../../src/visioncortex/lab/comment_transition.py)，接收已有口述记录的结构化解释：

- `stage_complete`：该样品当前步骤完成，保留实验、样品和批次。
- `next_stage_plan`：下一步计划，尚未开始，也不切换当前设备；到达下一设备主动扫码后再更新操作上下文。
- `experiment_complete`：明确整项实验完成，才允许结束对应实验记录。

“已完成该样品在天平前的配制，接着去烘箱”对应前两类。不能仅凭句中出现“完成”就执行第三类。

请求固定 `run_id`、`sample_id`、`batch_id`、`source_activity_id`、原转写 `source_text_sha256`、`source_evidence_ids`、原文、发生时刻和person／camera scope。原音、转写、范围、版本、事件时间与目标必须相符；提供独立持久化的 `reviewed_note` 及其 `comment_interpretation_review`，固定同一来源和结构化解释。该解释记录可由授权的上游Agent／语义服务产出，不要求本模块另开一次用户确认界面。缺少依据、解释待核或目标冲突时进入 `needs_attention`；调用方给一个 `verified=true` 不能绕过校验。

本模块实现解释后的事务性状态转换，尚不独立完成任意中文口述的ASR/NLU解析。阶段事件和主动扫码进入已有活动、trace和日报路径，原音／扫码证据引用随记录保留；材料用量、设备释放和全实验结束不由阶段计划自动触发。

## 现有内部读写入口

| 接口 | 用途 |
| --- | --- |
| `GET /internal/status` | 任务、接入异常、留存、索引及规则版本状态；阶段成功分别记录。 |
| `POST /internal/commands/{name}` | 事务性业务写入，含登记、标签历史、实验、扫描上下文、确认和更正。每次写入使用独立幂等key，重复同key不能改成另一请求。 |
| `GET /internal/timeline/trace` | 查询实验活动、来源跨度、绑定及历史修订。查询参数以对应方法/OpenAPI为准。 |
| `GET /internal/timeline/readouts/{session_id}` | 带字段状态与来源证据的读数会话。 |
| `GET /internal/timeline/reports/{day}` | 同一数据库派生的日报及修订；日报归档不证明实验已物理结束。 |
| `GET /internal/timeline/person-days/{person_id}/{day}` | 同一日报快照中的人员事件投影，返回相同 revision／fingerprint；分别标明操作声明、明确证据关联、实验相关证据及历史佩戴来源。 |
| `GET /internal/timeline/metrics/{day}` | 阶段耗时与预算；缺失实测不能当零耗时或夜间通过。 |
| `GET /internal/materials/source-lots/{lot_id}/summary` | 来源批次的材料单元、流水及下游关系。 |
| `GET /internal/materials/process-batches/{batch_id}/summary` | 目标制备批次的明确用料和流量，单位及未知量保持独立。 |
| `GET /internal/evidence/{evidence_id}` | 证据的媒体版本、哈希、时间范围及可用性。 |
| `GET /internal/media/{media_id}` | 鉴权媒体读取，支持Range；URL可读不等于模型已消费或读数正确。 |

设备业务命令包括 `reserve-equipment-catalogue`、`activate-equipment`；预留与实物核对激活分开。二维码承载永久设备UUID，名称、型号、实验和材料关系保存在私有数据库中；补打不换身份。材料／可复用标签与设备永久码有独立类型和历史。

日报的 `document.person_timelines` 使用 `vc-lab-person-timelines/1`。每条事件沿用活动UUID，关联多个实验不重复计为多个事件；不同视角仍保留独立观察。`direct_operator_declaration` 核对实际扫码事件及正文指纹，`recorded_by` 只说明有效历史佩戴来源，`experiment_related_evidence` 只说明与该人员登记的实验有关；后两种均不证明此人操作或说话。未知时钟不借用当前佩戴关系，跨交接边界不选最新人员。未知人员和未定时事件单独保留，覆盖状态为 partial，不能展示为连续全天轨迹或定位轨迹。

时间线命令包括 `timeline-create-run`、`timeline-bind-qr`、`timeline-label-event`、`timeline-observe-context`、`timeline-context-window`、`timeline-register-temporal-ocr`、`timeline-register-video-qr`。全部命令的唯一装配入口见 [command_methods](../../src/visioncortex/lab/timeline_commands.py)。材料、读数及实验事实的确认／更正走现有Core事务，不能直接改识别原始观察。

## 数据库与NAS的分工

[PostgreSQL迁移](../../migrations/lab) 保存身份、实验／批次、佩戴／标签历史、主动扫码、候选、事实、账本、任务、outbox和发布回执。使用专用实例或明确隔离的数据库，不连接旧FR运行SQLite充当业务库，不修改已应用迁移。

NAS按[统一留存契约](unified-retention.zh-CN.md)保存原始证据和处理结果：`Comment/Audio`保存原音相关转写与时间映射，`Comment/OCR`保存FR兼容读数，视频片段、理解与日报进入既定目录。数据库保存关联与版本，NAS文件保存可复核内容。数据库事务完成、文件发布完成、文件回读通过是不同状态；集成方分别展示或检查，不能看到HTTP成功就宣称NAS完整留存。

私有存储清单明确提供挂载路径、确切share、标记文件与SHA及项目根；程序先核对实际挂载，再核对标记。NAS不可用时不在同名本地目录静默落盘。客户既有归档及原始命名不随这次仓库统一搬迁或重写。

## 交付边界

这次提交提供可整合的源代码、通用规则、迁移、测试和接口说明。实际模型消费、现场时钟、主动扫码入口、口述联动、材料归属、读数质量和NAS发布仍须用真实链路验收。不能把fixture通过、旧模型测量或源码上传当成这些门槛已经通过。

## 从已留存扫码证据接入

鉴权命令 `timeline-consume-scan` 接收 `key`、已完成照片 QR `job_id`、明确 `intent`（`operator_scan` 或 `background_qr`）、`scan_kind`（`equipment` 或 `material`）、`actor`、`reason`，以及已有的可选 `person_id`、`run_id`、`batch_id`。设备二维码沿用登记的永久 UUID，不要求改印为旧演示 JSON。相机与事件时间取原收据，不能由调用者覆盖；未校准时钟不会获得当前人的身份。

自动采集的原 `CaptureReceipt.json` 可携带同样结构的 `scan_intent`（不含 `key`、`job_id`）；QR worker 完成后，既有 timeline tick 自动消费。采集方必须提供真实主动扫码意图，被动视频帧不填此字段。当前 live Receiver 默认路径仍不是这种主动事件的证明；配置和数据路由须单独验收。

可选的 [`scan_source_scanner.py`](../../src/visioncortex/lab/scan_source_scanner.py) 为原 FR `LiveScanner` 提供实例级端口：在原匹配器过滤二维码前，保存人工扫码会话选中的原像素及原始解码诊断，再由原接入队列、QR worker 独立解码，进入同一 Core 的 `ScanIntake`。它支持已登记的永久 UUID；无需改印为旧演示 JSON。原 FR 组件字节和默认启动器不变，后台自动会话不声明主动操作，预览线程不能串用另一帧。

该来源需私有 `VC_LAB_SCAN_SOURCE_MANIFEST` 明确启用，固定相机、来源实例、设备或材料类型、归档输入根和本地容量。无清单时不启动来源。原图先完整写入，`CaptureReceipt.json` 最后提交；中断恢复及同会话去重不生成第二次操作。相机自报时间但没有可核对的时钟不确定度时，仍保留未知事件时间；接收日期只用于文件分区，不作实验发生时刻。来源桥接的软件验证与实机扫码、实物激活、人员/时间依据和 NAS 发布分别验收。

原 FR HTTP 来源入口为 `visioncortex.lab.scan_source_scanner:app_factory`，需用私有 `VC_LAB_SCAN_LAUNCHER_MANIFEST`（`vc-lab-scan-launcher/1`）固定已有 FR 静态资源及首页 SHA、独立本地运行目录、容量、`source_enabled` 与允许的 `field_config`。它复用原路由，不在统一仓库复制一套界面。关闭来源时 `field_config` 必须为空，并且不继承环境中旧的接收器、模型或归档开关；启用来源时必须显式给 Receiver、相机及发布清单。本地资源缺失时明确拒绝，不默认指向某个现场目录。

自动消费保持新任务优先。未指定目标的材料扫码会先完成同一精确 scope 内已到达且事件时间更早的目标声明，受同一次预算约束；依赖尚未完成则等待。失败项有持久退避和次数上限，不能用一个坏记录永远挡住后续来源，也不能在晚到目标出现后偷偷改挂旧材料。

`/internal/status` 的 `timeline.scan_intake` 给出设备激活计数、已存扫码计数、来源配置与发布开关，`field_scan_acceptance` 保持独立于 API 健康检查。ASR、价值候选、未决记录与原生日报自动刷新能力见[状态说明](lab-vision-status.zh-CN.md)。
