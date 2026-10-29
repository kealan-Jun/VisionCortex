# 视频 录音与读数的统一留存

VisionCortex 使用一个工程和一个配置的归档根，视频处理与 FieldRecognition 读数按内容进入同一采集设备日目录，不再按两个项目建立并行结果根。客户名称可用于本地部署目录；Git 项目名保持 VisionCortex，实际目录、身份与凭据在仓库外绑定。

部署沿用既有 VC 归档根和 `YYYY-MM-DD_<camera_key>` 路径，采集发送方与 Agent 读取方继续使用原配置。升级发生在相同四类结果内容及其兼容引用中，不另建一个客户命名的正式归档根。已有原件、历史结果和读取入口原位保留；源码更新本身不切换正在运行的发布服务。

原件由采集生产者拥有。`MetaData` 与已有 `MetaVideo` 的原名、目录、文件字节、时间文件及封存回执必须保留。新规则使用独立版本，不通过搬移原件、改旧回执或修改冻结算法来制造统一格式。

## 目录合同

```text
<archive-root>/
├── .System/
│   ├── StorageVolume.json
│   ├── ArchiveContract.json
│   ├── Receipts/
│   └── Versions/
└── YYYY-MM-DD_<camera_key>/
    ├── MetaData/                    原件与采集回执
    ├── MetaVideo/                   已有生产者原目录，存在时原位保留
    ├── ProcessedClips/              已留存片段、索引与选择证据
    ├── Comment/
    │   ├── Audio/                   转写、时间索引及执行证据
    │   ├── Bindings.json            当前二维码、设备阶段及批次关联视图
    │   └── OCR/
    │       ├── VoicePhotoReadings/   照片/连拍事件的读数与证据
    │       └── VideoReadings/        视频触发事件的读数与证据
    ├── MultimodalUnderstanding/     视频理解结果
    └── LaboratoryDailyReport/       日报与历史版本
```

四个结果目录是 `ProcessedClips`、`Comment`、`MultimodalUnderstanding`、`LaboratoryDailyReport`。录音结果固定在 `Comment/Audio`，读数结果固定在 `Comment/OCR`；不能再把 `VoicePhotoReadings` 或 `VideoReadings` 新建为并行顶层结果根。

日常查看二维码关联直接打开 `Comment/Bindings.json`。其中保留原码观察、已成立的操作声明、未决原因、操作阶段、实验/目标批次以及读数/转写入口；看到码但身份或时间不足时仍为候选。它是数据库版本化关系的最新文件视图，不是第二套账本。发布器核对不可变副本与数据库版本高水位，原子更新这个入口；历史版本保留在 `LaboratoryDailyReport/Timeline/Associations/Bindings`，日常查看无需进入历史目录。已有同名外部文件或损坏/回退的视图拒绝覆盖。

Timeline 的 OCR 正文进入 `Comment/OCR/VoicePhotoReadings/<事件>/Readings.json` 或 `VideoReadings/<事件>/Readings.json`，保留 `Evidence.json` 和必要的 `UnboundReadings.json`。转写正文进入 `Comment/Audio/<事件>/Transcript.json` 与 `Transcript.txt`。这些来源固定的候选产物不覆盖原生产者事件，不在缺少面板证据时制造逐仪器 `Result.json`，也不把没有对齐的转写伪装成逐词对齐。

规范口述是 Comment 的业务证据。`Comment/Audio` 不只是附件目录：转写正文、词/段时间索引及原音引用与扫码历史、视觉动作、OCR 共同支撑实验记录。原始录音仍在生产者原件目录，语音产物通过精确偏移回指，不再重复保存一份被误认为新采集的原音。

每条口述保留原音/转写范围、说话人证据、来源与版本。佩戴者不自动等于说话人，未知时钟或相近时间不自动证明同一实验。口述与视觉冲突时双方证据和未知状态都留存，等待有依据的关联或同事集成的复核流程；目录统一不等于跨模态业务归属已验收。

目录名使用既有设备日校验器、配置时区日界和完整采集相机键。不得截短、散列或用仪器编号替换相机键。同一相机的视频、音频、照片共用该日目录；多路相机保留各自来源，通过数据库的实验、人员和时间关系关联，不重复复制整片到每台仪器或每种材料名下。

拍摄时钟未知时，明确保留未知状态与接收日期放置依据。文件夹日期、文件修改时间和接收时间都不能被自动升级为经过校准的拍摄时间。

## 原件接入与完成条件

原生录像按既有格式整体检查 RGB、配对时间 CSV、会话、关闭/完成信息和可处理状态。音频保留独立原件及实际关联；没有音频输入或没有实际转写时，不能生成成功转写正文。生产者声明部分完整的数据继续标为 partial，不因封存完成就变成完整采集。

照片入口保留原拍摄与连拍身份。显式 `PhotoReceipt` 或已启用的目录连拍模式决定成员关系、期望张数和完成状态；第三张晚到可修订同一组，不把另一时刻照片拼入。祖先 `.uploading`、文件不稳定、回执成员不符或字节变化都阻塞接入。裸图模式必须有明确相机范围，不能扫描任意目录后自动建业务记录。

新照片的推荐位置是 `YYYY-MM-DD_<camera_key>/MetaData/Photos/HH-MM-SS/<原文件名>`，按已有回执协议完成提交；旧的 `MetaData/<时间>/` 等已登记原件仍兼容读取，不搬移。录像、录音与原时间文件继续按既有封存包发送，不为了分类浏览拆散原包。处理服务不提供一个供采集端猜测的通用 HTTP 上传地址。

跨兼容路径的同一照片只有字节及其它不可变元数据一致，才补充来源位置并去重。不同相机即使图片字节相同也仍是不同来源。既有 `MetaVideo`、时间文件和封存包不为浏览美观拆散成新的强制发送协议。

实时 Receiver 普通帧在内存中使用；仅被选为证据的帧写入 `MetaData/VideoEvidence` 和耐久 `CaptureReceipt`。该回执可恢复数据库提交前的中断。实时证据帧不代表已保存完整实时录像，也不意味着二维码、仪器身份或读数已正确。

## 固定六字段与详细证据

逐仪器 `Result.json`、`Result02.json` 等沿用既有稳定分配方式，不随重处理任意换号。正式顶层仍只有六个既定字段：

| 字段 | 含义与边界 |
| --- | --- |
| `wearer_id` | 事件时间有效的佩戴者；未知时间不能借用当前人员 |
| `device_model` | 已明确的仪器类别/型号字段；未知为 `null` |
| `device_no` | 已登记并有证据对应的仪器标识；不是采集相机键 |
| `qr_hash` | 原二维码关联字段；解码失败不能从字典补值 |
| `photo_time` | 原拍摄时间及其既有表示；未知不以处理时间替代 |
| `values` | 原读数字段结构，未知值为 `null`，不伪零 |

新增设备类型、字段含义或布局能力须显式版本化，不能往六字段中塞入新的不兼容业务格式。实际值、设定值、通道、单位依据和缺失小数等详情在读数与证据层保留。

`Readings.json`、`Evidence.json` 保留原文候选、期望/已到成员、partial/complete、原媒体引用、原时钟依据、帧 PTS/媒体偏移、区域/变换、模型与规则版本、执行状态和确认关系。未绑定或无法读出的观察保留候选/未绑定证据，不制造正式仪器结果。

同一次采集可产生视频理解、ASR、OCR、材料候选等多种结果，均引用同一原件版本及精确范围。正文和范围要能回查，不只留证据 ID。没有可靠世界时间时仍可保存媒体相对时间，但不据此臆造实验或人员归属。

主动扫码声明与后台被动码观察保留不同事件类型。实验员主动扫码设备应立即建立当前设备操作上下文，不以视觉动作再佐证为前提；被动入镜只保留观察。声明、操作者、设备、事件时间及已有实验/目标批次关系进入同一时间线和不可变历史，不改变 Result 六字段。

主动材料扫码按事件时间解析来源批次，并关联明确的当前实验/目标制备批次。一个目标批次允许多材料、多设备；显式目标优先，否则可继承同一精确人员/相机 scope 内严格早于该材料扫码的已声明目标，固定当时快照和来源扫码引用。没有明确上下文时待关联，不跨 scope 猜默认目标，晚到声明不悄悄重挂旧材料。此关联不是消耗确认，切换当前设备也不结束旧实验。主操作上下文不能替代每块面板的空间归属证据。

## 片段与来源映射

`ProcessedClips` 保留原筛选片段、索引及选择依据；没有完整选择回执就不把整段原视频复制成“精选片段”。新内容边界不修改历史选择或原筛选参数。

名义选择区间、事件内容边界、上下文 padding、导出容器范围和播放长度分开。离散帧映射保留原片/片段版本、原始 PTS、精确时间和逐帧像素对应。未映射尾隙保持 unknown，不用取整或容差放宽补成已覆盖。supplied 映射与本地实际像素核验凭据分别标记。

完整片段策略必须按其注册输入单位执行。显式允许的研究抽样另记转换与未观察范围；不能覆盖原片或被包装成全帧模型消费证明。

## 不可变历史与幂等发布

业务观察、关联修订、确认与数量操作由本地 PostgreSQL 管理，文件是可追溯的产物和证据。发布计划先提交数据库，再写文件，失败沿同一计划重放；任务完成、文件已写、回读可用和业务已确认分别表示。

`.System/Receipts` 与 `.System/Versions` 保存不可变回执和版本。日报及索引的最新视图可以刷新，历史版本不能因更正而删除。晚到证据、标签换绑、实验归属修订和正文恢复到旧内容，都应产生真正的新修订；不能复用旧内容哈希后让最新查询仍指向过时版本。

技术记录归档不证明实验物理完成，不自动释放仪器占用或样品。材料消耗必须经过原业务账本授权入口，不能由 OCR、ASR、Agent 描述或一次二维码出现直接扣量。

口述完成须区分范围：“某样品在天平前配制已完成、现在去烘箱”留为 `stage_complete` 与 `next_stage_plan`，保留样品、批次和实验运行，不归档整个实验。下一设备计划不改变当前设备，实际到下一设备主动扫码后才切换操作上下文。参数/设置说明也不触发归档；只有明确整体全部完成的 `experiment_complete`，经授权整体结束操作才进入记录归档。

结构化口述转换须保存原 speech 正文指纹、原音 evidence/version/范围、源时钟、精确 scope 和持久 `reviewed_note` 解释记录，并与其派生的阶段/计划/整体状态相互回查。该上游记录可由获授权的 Agent 服务产出，不增加用户手工确认步骤。`timeline-comment-transition` 薄接口已实现，选定 fixture 专项已验证；实际 ASR/NLU 语义自动入口和实机 NAS 链仍未验收，见 [Agent 集成](agent-integration.zh-CN.md)。

录音 `Index.json`、`TimeIndex.json`、转写和执行回执同步引用 `Comment/Audio`；OCR 的事件组与逐仪器结果同步引用 `Comment/OCR`。历史冻结发布计划可继续按原路径重放，不自动改写旧档。

历史留存说明中的四个结果目录、逐片段 `Understanding.json` 与相机 `LaboratoryDailyReport.json` 继续由原生发布器维护。统一数据库的跨相机／人员日报与相机日报是不同范围，版本化 `Timeline` 索引不能冒充已经生成的原生步骤理解、全天模型综合或真实筛选视频。没有相应模型结果或视频时保留 pending；关联信息的发布状态与原生主入口的刷新状态分别核对。

原生主入口只消费已有数据库提交回执所固定的关联侧车。目录里存在 JSON 不代表出版事务成功；直接调用纯文档构建器不会自行纳入未提交侧车。刷新期间出现更高但未提交的版本时保持 pending，保留原主文件，等其正式提交后再刷新。日报快照固定自己的原媒体 ID 集合，新到数据进入下一版，不能改变失败恢复时的旧版不可变内容。

## 存储与私有部署配置

归档根、共享卷绑定、标记指纹及凭据均由私有部署配置提供，仓库不包含现场地址、客户目录、实物清单或业务原件。正式访问先验证正确挂载和卷/项目标记，并持有目录锚点执行原子读写与回读。挂载丢失、错误或合同不符时停止重试，不在未挂载目录下建本地替代树。

代码、环境、当前权重、活动数据库、队列和锁位于本地 SSD。工作媒体缓存有明确大小与保留期，删除副本前复核原件一致且没有运行中读者；原件缺失或变化时不删。原件、已完成输出和不可变回执不直接套用缓存 TTL。专用研究策略的小缓存预算不改变正式原件保留约定。

统一留存合同已有可重入准备/升级和历史保留机制，准备空根不激活采集或模型。有设备日数据的历史目录不做隐式重排。格式升级、路径绑定和识别规则都需显式新版本，旧语义与冻结组件保持可恢复。

既有合同如果仅有顶层 `project`、`schema_version`、`supersedes` 名称差异，可通过本地私有 `VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST` 明确声明 `vc-lab-archive-compatibility/1` 绑定。每条绑定固定绝对归档根、既有文件SHA、新安装合同SHA与原字符串；规范化后的其余内容及类型必须完全相同。此入口不允许目录、字段、归属或存储语义变化，也不改写既有NAS合同；未提供绑定时仍按原严格字节合同核验。

已有 VC 根若没有 `ArchiveContract.json`，使用同一环境变量的独立私有模式 `vc-lab-existing-native-root/1`：明确绑定唯一归档根、原 `DataRetentionGuide.docx` 的 SHA、新安装目录合同 SHA 和固定原目录约束。描述文件必须在本地、属于当前用户且权限为 0600；实际访问仍核对全部挂载、卷锚点和持有的目录句柄。只读绑定不写入或替换旧指南、不初始化旧根，也不证明历史结果正确。已有但错误的合同不能回退到指南模式，指南变化、符号链接或挂载变化均拒绝。该模式只授权同一路径下未来的兼容内容；生产启用单独执行。

## 代码与合同导航

| 责任 | 入口 |
| --- | --- |
| 设备日命名与合同 | [`archive_layout.py`](../../src/visioncortex/lab/archive_layout.py)、[`contracts/lab`](../../contracts/lab) |
| 输入、照片路径与原件 | [`ingest.py`](../../src/visioncortex/lab/ingest.py)、[`input_layout.py`](../../src/visioncortex/lab/input_layout.py)、[`media.py`](../../src/visioncortex/lab/media.py) |
| 原生结果与路径绑定 | [`native_comment_layout.py`](../../src/visioncortex/lab/native_comment_layout.py)、[`delivery.py`](../../src/visioncortex/lab/delivery.py)、[`components/lab`](../../components/lab) |
| 读数原格式 | [`src/field_recognition`](../../src/field_recognition)、[`retention.py`](../../src/visioncortex/lab/retention.py) |
| 片段映射 | [`retained_clips.py`](../../src/visioncortex/lab/retained_clips.py)、[`retained_frame_map.py`](../../src/visioncortex/lab/retained_frame_map.py) |
| 主动扫码声明 | [`operator_scan.py`](../../src/visioncortex/lab/operator_scan.py) |
| 原音与结构化口述转换 | [`comment_transition.py`](../../src/visioncortex/lab/comment_transition.py) |
| 日报、发布与实际计时 | [`timeline_report.py`](../../src/visioncortex/lab/timeline_report.py)、[`timeline_publication.py`](../../src/visioncortex/lab/timeline_publication.py)、[`timeline_metrics.py`](../../src/visioncortex/lab/timeline_metrics.py) |
| 挂载守卫与缓存 | [`storage.py`](../../src/visioncortex/lab/storage.py)、[`cache.py`](../../src/visioncortex/lab/cache.py) |

这些是统一工程内的模块导航，不要求额外部署另一套项目或建立新的 Agent 工具协议。
