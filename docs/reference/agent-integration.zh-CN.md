# 统一实验链路的集成接口

[文档导航](../README.md) · [算法与业务逻辑](lab-vision-algorithms.zh-CN.md) · [留存契约](unified-retention.zh-CN.md) · [当前边界](lab-vision-status.zh-CN.md)

VisionCortex 同时承载视频、FR识别及实验时间线业务能力。一次安装、同一仓库；`visioncortex.lab` 是所属业务模块，不是另建的客户项目。客户名称只属于本地部署与私有配置，设备清单、原片、材料批次及数据库实例不随源码发布。

## 启动与职责

识别处理服务不接收、索取或等待实验员身份。人员绑定完全由硬件与 Agent 团队处理；本工程只按采集来源、会话和时间关联实验设备、面板、样品及材料批次。下文人员查询和字段仅为历史接口兼容，不是本次接入前置条件；无需为识别补建人员绑定。

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

默认主动扫码是操作声明。另有显式启用的相机自然看码策略：
`automatic_equipment_qr` 从已完成、独立解码且有原像素和来源时间的照片／视频建立设备上下文；
可选 `material=true` 同时接入材料标签。同一帧可有一个设备码和一个材料码，多个同类身份保留歧义。
未启用该策略的后台观察保持原行为。采集设备不是某台仪器的永久身份，也不要求接入人员身份。

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

## 通用口述用料与称量接入

设备读数和口述都先作为记录留存；温度、转速、时间和设置值不自动变成材料用量。
`spoken_material_usage` 私有策略提供 `enabled`、`revision`、`max_activities_per_tick`（1–20）及
`post_actual_declarations`。口述消费使用原音版本、原转写和句段偏移，按同一采集来源的历史二维码上下文
关联材料、来源批次与目标批次。实际取用、计划、累计、剩余、更正分别保存；“取了2g，还剩98g”中的
剩余量只核账，不再次扣量，也不覆盖账本。口述冲突、近似量、未知时钟或无目标时照常留存，待核对。
由派生样品取用时可保留来源批次未知，不能把目标批次补成来源批次。

称量是其中一条专用路径。`weighing` 策略提供 `enabled`、`revision`、`max_activities_per_tick`（1–20）。
必须有本面板的设备归属证据、可读质量字段、明确单位及零点／基准，同一称量的后续照片更新累计用量；
相同读数不再扣量，下降／去皮／冲突不自动退款。缺少这些条件不影响普通OCR数字记录。
命令 `timeline-start-weighing`、`timeline-observe-weighing`、`timeline-close-weighing`、`timeline-weighing`
提供现有鉴权入口；更正使用 `expected_sequence`，保留原操作和反向账目。

录音与照片有同一次用料疑点时停止第二次自动扣量，不能仅凭数量相同或时间接近合并。
已明确为同一次操作，可在 `timeline-post-spoken-usage` 或 `timeline-observe-weighing` 提供
`existing_operation_id`；校验材料、目标、单位和数量后仅连接已有操作。其后的更正走原Core账本。
口述记账证明“按声明入账”，不等同于人工核实实际使用；所有展示保留这一依据。
口述和OCR共用 `material_use_context` 的来源范围、历史标签、目标批次、同笔用料锁及跨通道重复检查，
已有操作的连接统一由Core校验。关闭称量记录不清除这次操作的防重复依据；新零点可明确下一次称量。
两种记录的中文摘要共用时间、记录类型、数量、单位、设备、材料、来源批次、目标批次及账本操作字段。
外部更正通过同一账本修订链反映到两种视图，原口述和原OCR结果保持原样。

视频理解的同一先验构建器读取这些已留存观察：视频／音频原文、OCR字段、二维码及材料数量语义、
已确认事实版本分别标明，按采集来源、物理范围和任务开始时已知的数据截取。该快照随真实模型请求保存；
计划、剩余量和候选不改写成新发生的动作。没有可靠时钟或明确源映射的跨通道证据不强行合并，
省略原因及预算不足仍显式保留；日报从同一数据库快照读取，不另造一套上下文。

已有原生ASR可通过 `NativeCommentIntake` 复用已完成的 `Comment/Audio/Index.json` 和原转写。
索引晚到时也可按采集元数据中的精确音频范围读取已有时间目录，仍须核验录制身份、原音摘要与转写原文。
原系统把 Comment 证据存入独立缓存时，私有配置设置 `VC_LAB_NATIVE_COMMENT_BACKEND_ROOT` 为原有缓存根；
沿用 `device-day-comment-evidence` 命名空间和原逻辑引用，不复制原文。已配置的缓存缺失或摘要不符时停止导入；
NAS 缓存仍须通过既有挂载与标记校验。未配置时保留原归档读取方式。
验证原音摘要、录音身份、原文摘要和句段范围，不调用新模型、不改写原音、不复制一套音频目录。
原生入口因归档根不同而生成不同录制编号时，只接受显式原采集索引中的同一路径、相机、会话、
源时间元数据和签名映射，并继续核验实际原音摘要。个别原ASR句段越出音频边界时保留原文和原偏移为待核对，
不拉长音频、不裁成伪对齐，也不把该句用于精确阶段结束判断。
它保留实际旧模型名称，不把历史结果标成新ASR。
用料中文摘要及机器记录进入原 `Comment/Bindings.json`，同一快照供日报消费；
`Comment/Audio`、`Comment/OCR` 和原片路径保持既有契约。上述软件检查与真实扫码、ASR正确率、
面板识别、材料标签发行及生产加载分别核验。

## 从已留存扫码证据接入

鉴权命令 `timeline-consume-scan` 接收 `key`、已完成照片 QR `job_id`、明确 `intent`（`operator_scan` 或 `background_qr`）、`scan_kind`（`equipment` 或 `material`）、`actor`、`reason`，以及已有的可选 `person_id`、`run_id`、`batch_id`。设备二维码沿用登记的永久 UUID，不要求改印为旧演示 JSON。相机与事件时间取原收据，不能由调用者覆盖；未校准时钟不会获得当前人的身份。

自动采集的原 `CaptureReceipt.json` 可携带同样结构的 `scan_intent`（不含 `key`、`job_id`）；QR worker 完成后，既有 timeline tick 自动消费。采集方必须提供真实主动扫码意图，被动视频帧不填此字段。当前 live Receiver 默认路径仍不是这种主动事件的证明；配置和数据路由须单独验收。

可选的 [`scan_source_scanner.py`](../../src/visioncortex/lab/scan_source_scanner.py) 为原 FR `LiveScanner` 提供实例级端口：在原匹配器过滤二维码前，保存人工扫码会话选中的原像素及原始解码诊断，再由原接入队列、QR worker 独立解码，进入同一 Core 的 `ScanIntake`。它支持已登记的永久 UUID；无需改印为旧演示 JSON。原 FR 组件字节和默认启动器不变，后台自动会话不声明主动操作，预览线程不能串用另一帧。

该来源需私有 `VC_LAB_SCAN_SOURCE_MANIFEST` 明确启用，固定相机、来源实例、设备或材料类型、归档输入根和本地容量。无清单时不启动来源。原图先完整写入，`CaptureReceipt.json` 最后提交；中断恢复及同会话去重不生成第二次操作。相机自报时间但没有可核对的时钟不确定度时，仍保留未知事件时间；接收日期只用于文件分区，不作实验发生时刻。来源桥接的软件验证与实机扫码、实物激活、人员/时间依据和 NAS 发布分别验收。

原 FR HTTP 来源入口为 `visioncortex.lab.scan_source_scanner:app_factory`，需用私有 `VC_LAB_SCAN_LAUNCHER_MANIFEST`（`vc-lab-scan-launcher/1`）固定已有 FR 静态资源及首页 SHA、独立本地运行目录、容量、`source_enabled` 与允许的 `field_config`。它复用原路由，不在统一仓库复制一套界面。关闭来源时 `field_config` 必须为空，并且不继承环境中旧的接收器、模型或归档开关；启用来源时必须显式给 Receiver、相机及发布清单。本地资源缺失时明确拒绝，不默认指向某个现场目录。

自动消费保持新任务优先。未指定目标的材料扫码会先完成同一精确 scope 内已到达且事件时间更早的目标声明，受同一次预算约束；依赖尚未完成则等待。失败项有持久退避和次数上限，不能用一个坏记录永远挡住后续来源，也不能在晚到目标出现后偷偷改挂旧材料。

`/internal/status` 的 `timeline.scan_intake` 给出设备激活计数、已存扫码计数、来源配置与发布开关，`field_scan_acceptance` 保持独立于 API 健康检查。ASR、价值候选、未决记录与原生日报自动刷新能力见[状态说明](lab-vision-status.zh-CN.md)。
