# VisionCortex 人的全流程实验归档契约 v1

本规范依据 2026-09-10 用户确认及随后明确补充固化：所有文件夹用英文；已关闭的不完整分片按实际时长处理；录音必须保留并执行 STT；无活动抽帧是场景采样帧，不能标为五大类物理动作关键帧。它是 NAS 自动采集归档的唯一设备日目录契约，版本为 `visioncortex-device-day/1`。后续实现、页面和报告必须遵守本规范；未经用户明确变更，不得更改目录名称、层级、活动分类名称、时间单位、引用语义或恢复规则。报告正文、活动说明及原始文件名可保留中文。

运行常量及校验：`src/visioncortex/device_day_contract.py`；机器可读索引契约：`docs/contracts/device-day-v1.schema.json`。历史六目录归档保留历史含义，不静默重写，不作为新采集自动归档模板。

## 固定目录

NAS 根目录使用生产配置的 `storage.archive_root`，当前解析到 `/mnt/realityloop-nas/VisionCortexExperimentArchive`。设备名称来自采集目录及明确配置的人称绑定，不使用“设备01”等占位名。2026-09-10 可见的当前六台设备如下，每台每天独立一个目录：

```text
VisionCortexExperimentArchive/
├── 2026-09-10_lubancat-52d2ef0c_cam01/
├── 2026-09-10_lubancat-e8cc0cb3_cam01/
├── 2026-09-10_orangepi5pro-ab748372_cam01/
├── 2026-09-10_orangepi5pro-b439137c_cam02/
├── 2026-09-10_orangepi5pro-fe0f7222_cam01/
└── 2026-09-10_rk3588-ubuntu_cam01/
```

每个设备日目录严格包含下面五个一级目录。分片、片段和帧在其所属目录内增加，不能在设备日根目录引入第六个产出目录。

采集 CSV、标定、时间戳、就绪说明、录音元数据、日志及其他采集文件全部纳入 `MetaVideo`，深度视频排除且在 `CaptureFiles.json` 中记录。原始文件内容保持不变，清单保存原名与归档名的映射、来源、大小及 SHA-256。尚在写入的附属文件等稳定后补归档，不阻塞已就绪视频的 YOLO；不跟随符号链接，并显式记录原因。总索引通过各分片的 `capture_manifest` 引用该清单。采集端删除仍然关闭。

```text
<日期>_<采集设备名>/
├── MetaVideo/
│   ├── <开始时间>_<结束时间>.mp4          # 原视频直接可见，保留原始字节和实际采集边界
│   ├── Audio/<开始时间>_<结束时间>.opus   # 同设备同日原录音
│   └── Metadata/<开始时间>_<结束时间>/
│       ├── Video/                      # Frames.csv、Meta.json、RecordingReady.json、Calibration.json
│       ├── Audio/                      # AudioTiming.csv、AudioMeta.json、AudioReady.json
│       ├── Capture/                    # Ffmpeg.log 等其他采集数据，保持相对层级并转为 PascalCase
│       └── CaptureFiles.json           # 全部采集文件的原名、来源、归档引用、校验值和排除记录
├── ProcessedClips/
│   ├── Index.json                        # 全部有/无活动片段、帧、理解文本及素材引用
│   └── Clips/
│       ├── <开始时间>_<结束时间>_ExperimentActivity_<唯一后缀>/
│       │   ├── ExperimentActivity.mp4
│       │   ├── ExperimentActivity.json             # 本片段索引、原片引用、关键帧路径/时间/校验值
│       │   ├── KeyFrames/<五类动作>/<event_id>/<视频时间毫秒>.jpg
│       │   └── SceneFrames/<视频时间毫秒>.jpg
│       └── <开始时间>_<结束时间>_NoExperimentActivity_<唯一后缀>/
│           ├── NoExperimentActivity.json             # 引用meta video中的原片及明确起止位置
│           └── SceneFrames/<视频时间毫秒>.jpg # 场景采样，不是动作关键帧
├── MultimodalUnderstanding/
│   ├── Understanding.json
│   ├── UnderstandingReport.html
│   └── ClipUnderstanding/<segment_id>/<处理版本>/<片内开始时间>_<片内结束时间>/
│       ├── Input.json                      # 帧索引、comment、protocol与来源快照
│       ├── Result.json                      # 实际模型结果、请求回执和Token用量
│       └── SceneFrames/<视频时间毫秒>.jpg
├── LaboratoryDailyReport/
│   ├── LaboratoryDailyReport.html                    # 单一日报的可阅读形式
│   └── LaboratoryDailyReport.json                    # 同一份日报的结构化形式
└── Comment/
    ├── Comment.jsonl                      # 该设备该时间的人的备注，按需产生
    ├── Protocol.json                      # 用户提供的protocol快照/版本，按需产生
    ├── Stt/<recording_id>/<处理版本>/      # 自动识别，不冒充人的手写comment
    │   ├── Comments.json                  # 带时间、原录音引用与STT来源的机器comment
    │   └── <片内开始时间>_<片内结束时间>/
    │       ├── Transcript.json            # 句段和词级时间、识别置信信息
    │       ├── Transcript.txt
    │       ├── Transcript.srt
    │       ├── Transcript.vtt
    │       ├── Audio.m4a                  # 派生试听片段，原始录音仍在meta video
    │       ├── Response.json              # Qwen 原始响应、请求 ID 与服务端用量；无密钥和 Base64
    │       └── Result.json                # 指向后台原始执行回执及本目录的转写文件
    └── History/                              # protocol变更的旧版本
```

`MetaVideo` 是原视频实体的唯一最终留存位置。用户最新要求（2026-09-11）：**每个分片**归档校验通过、对应预处理结果与索引完整落盘后，将采集端该视频原路径原子替换为指向 `MetaVideo` 原片的相对软链接，释放重复实体；不等待其他分片、多模态或日报。有活动/无活动均适用，音频及 CSV 不在本次清理范围。严禁先 unlink 再创建链接。

新实现为 `capture_link_cleanup.py`，由预处理完成回调提交独立清理任务，并在后台发布轮次恢复未完成的请求。当前 NAS 原生链接探测失败，采集读取器兼容性也尚未验收，因此 `capture_video_link_cleanup.enabled/native_links_verified/capture_readers_verified` 均保持 false；**实际采集原片尚未清理，不能宣称已经只剩一份**。旧 `delete_capture_sources` 是不保留链接的删除开关，继续禁止。下文历史记录中“删除关闭”描述当时状态，以本段最新授权与准入条件为准。

无活动片段直接引用归档原片的时间区间，避免为完全无活动的 15 分钟再生成相同视频副本；两类片段均保留单独 JSON 和可打开的场景采样帧。只有围绕五大类物理动作、经过现有关键事件筛选的画面进入 keyframes；无活动区间的 key_frames 必须为空。关键帧 `path` 是设备日根目录下的相对路径；页面将它解析为文件链接。JSON 本身不执行打开操作。

## 字段与溯源

| 对象 | 固定数据含义 |
| --- | --- |
| `recording_id` | 原生采集记录的稳定标识；采集源位置、完成状态、签名保留在 retention 回执中 |
| `segment_id` | 采集标识、处理版本、起止时间、活动分类的确定性标识；重试不随机增加同版本片段 |
| `activity` / `activity_label` | `active` / `有实验活动` 或 `inactive` / `无实验活动`；错误和采样不全不能编码成 inactive |
| `start_ms` / `end_ms` | 相对于所引用原视频的媒体时间，毫秒，左闭右开；不是片段文件内部的重新计时 |
| `start_us` / `end_us` / `capture_us` | Unix 时间微秒，显示时区固定 `Asia/Shanghai`；优先使用采集 CSV 插值，估计值必须明确标记 |
| `source_ref` / `clock_ref` / `video` | 设备日相对路径 `path`、文件大小 `size_bytes`、内容校验 `sha256`；无活动 `video` 可为空，原片引用不能为空 |
| `key_frames[]` | 仅动作关键帧：`frame_kind=action_keyframe`、`event_id`、五类 `action_type`、被筛选事件校验值，以及帧路径、时间、源引用。必须来自现有 `select_key_events`，不能用定时抽帧充数；无活动必须为空 |
| `scene_frames[]` | 场景采样帧：`frame_kind=scene_sample`、帧路径、时间、源引用与 `understanding_text`；用于无活动、无关区间及步骤理解的画面上下文，不计入动作关键帧 |
| `sampling_coverage` | 粗扫实际采样数、预期采样数、覆盖率、最大间隔、FPS；覆盖不足失败，不宣称无活动 |
| `steps[]` | 时间、观察描述、输入帧 ID、comment ID、observed/inferred/uncertain；拒绝虚构素材 ID 和越界时间 |
| `comment` | `comment_id`、绝对时间区间、text、author、human_comment 来源；不把人的口述变成视觉确认 |
| 录音和 STT | 总索引 `recordings[].audio` 保留原录音及元数据引用；`transcription` 保留状态、机器comment与执行回执。`machine_transcribed_speech`、`human_reviewed: false`、时间对齐依据必须显式保存，不能将机器转写标为人审真值 |
| `evidence_status` | 单设备筛选和模型理解保持 `PARTIAL_EVIDENCE`；`physical_action_confirmed` 为 false |
| 阶段回执 | schema_version、key、stage、status、recording_id、时间/耗时、产出引用；失败保留类型和原因 |

五类动作固定指手部与物体接触、物体移动、液体移动、容器状态变化、设备面板操作。现有扩展的移液器源到目标操作保留在事件回执，不改写为已看见液体移动。动作关键帧、场景采样帧、原视频、截取视频和模型输入都保留真实素材引用。CSV 插值不等于已验证原生 PTS 或跨相机精确对齐；单设备记录不宣称恢复了多视角物理动作真值。

同一天的多视角投影由 `device_day_multiview.py` 将已发布候选、精扫索引和原片时间戳送入离线共用的 `build_alignments`、动作审计、边界规范化、正式实验筛选、`build_experiment_groups` 和 `select_key_events`。分片边界不作为实验边界；每个来源仍保留设备、日期、原分片、局部时间及校验值。总索引附带 `multiview_analysis`、`aligned_experiments`，不改变五目录及原有 `segments` 含义。只有对齐与实验分组通过门控才调用离线 `archive.materialize_experiment_clips`，在对应设备的 `ProcessedClips/Clips/<开始时间>_<结束时间>_ExperimentActivity_<唯一后缀>/` 保存 `ExperimentActivity.mp4`、`ExperimentActivity.json`；第一人称目录同时保存 `AlignedFirstThird.mp4`。导出前按封存 CSV 保留停录及缺片间隔，逐分片复核共享时间与采集时间；超限来源单独隔离。含未核验第三人称区间或跨第三人称机位路由的分组仅保留关联及原片查阅入口，不把占位画面或混合机位视频发布成某台设备的活动视频。未经边界审核的动作关联不能称为完整实验。这些关联派生产物不重复计入原始分片或单设备活动区间数量。共享对齐和分组不等于人审物理动作真值。

多视角投影在独立后台工作线程更新，复用已完成检测，不调用新的 YOLO 或云模型。新分片到来时复用未改变的媒体探测和实验导出；缺失原片、损坏时间戳按来源记录，其他可读分片继续。播放优先读切分片段，片段不可读时回退到其自身 MetaVideo 的对应时间窗口；单路缓冲或失败不得阻塞其他可用视角。

## 解耦与持续处理

1. **retention**：确认采集已关闭并发布、文件稳定，校验复制原片和附属元数据，发布留存回执。`capture_complete` 只描述采集质量，不是是否允许分析的开关；已关闭的 incomplete/partial 分片同样留存并进入后续阶段。
2. **vision**：每个停止写入的采集分片独立入队，**以实际视频时长处理，不要求满15分钟**。短片、末尾不足15分钟及采集程序标记不完整但已关闭稳定的片段也处理，原始质量问题随产物保留；仍在写入的文件等待稳定，无法解码的文件保留具体失败原因。采集文件按实际起止时间形成处理批次，不再人为按900秒切出尾部小批次；精扫仍仅覆盖候选活动区间。设备按有界工位并发。先粗扫，覆盖合格且没有候选活动则跳过精扫；有候选时只对候选区间精扫。粗扫阳性仅扩大精扫范围，候选和拒绝原因保留在审计回执，不能直接命名“有实验活动”。接入现有精扫、移动画面核验、动作审计、连续状态、可观测性和边界规范化规则，只有通过设备级活动边界判定的区间命名为有活动。跨视角正式实验门禁单独保留，不因按设备归档而绕过。
3. **stt**：与 YOLO 并行消费原片留存回执。独立录音或原视频内嵌音轨必须进入已配置的 STT。生产节点按用户要求复用已绑定的 Aliyun API，调用 Qwen-Audio-3.0-ASR-Flash；按实际录音时长分有界窗口，保存句段、词级时间、原音频引用、模型身份和执行回执，生成第五目录的机器comment。无音频保留 no_audio；模型识别结果为空保留 no_transcript，不能据此证明静音；只有明确的语音活动检测结果才可标 no_speech，不伪造模型调用或识别文本。STT失败不占用或阻止YOLO工位。
4. **understanding**：消费落盘视频片段与STT产出。有活动按有界窗口理解步骤；无活动用稀疏抽帧描述场景及其他可见行为。两者都结合对应时间的人的comment、STT机器comment和protocol，保留输入、输出、帧来源及用量。定时抽取的模型理解图片统一标为 scene_sample；不得因为模型引用了一张图片就把它提升为五类动作关键帧。语音内容不能独立确认视觉动作；时间缺少依据时显式保留估计/未验证状态。
5. **report**：消费总索引中的关键帧文字、STT和多模态理解，重组成一份按时间组织的实验室日报。HTML 和 JSON 是一份报告的两种表达；不是三份报告并列或文本机械相加。

五个执行阶段使用独立 SQLite WAL 持久化队列，保存在配置的本地 runtime 根目录；NAS 仍为五个产出目录。依赖关系为 retention → vision、retention → stt，vision + stt → understanding → report。文件级互斥、任务租约和原子文件发布支持重启续跑。完成且输入、代码、模型及产物校验一致的阶段复用；YOLO 扫描和多模态响应另以实际输入及模型身份校验复用，缓存复用与本轮新模型调用分别记录；录音晚到不会重新运行不变视频的YOLO，comment/protocol/STT变化使理解与日报重新产出。失败保留为 failed，付费模型请求不会在每轮轮询无穷重试。

自动处理消费现有 NAS 采集批次监控发布的持久化清单，不另起重复扫描。监控发现已关闭稳定的分片后唤醒处理队列，各设备可处理分片独立推进，不等待整个采集会话或全部相机完成。网页沿用 NAS 批次卡片，显示各设备原片留存、YOLO、STT、多模态、日报状态与耗时、队列积压和归档入口。生产自动发现从 `device_day.start_date` 开始；指定历史日期可通过 CLI 补跑。新处理发现不继承旧 UI “每相机只取最近32条”的截断，已入队记录不会因后续扫描不可见而消失。全局发现上限及读取错误必须在状态中呈现；达到上限不能视为扫描完整。

10 台设备 × 每小时 4 片 × 8–12 小时 = **每天 320–480 片**，入队速度约 **40 片/小时**。队列记录各阶段待处理/运行/失败数量、待处理采集跨度、已知的实际媒体时长、未知时长数量、最老等待时间与实测服务率。480 条持久化恢复测试只能验证队列契约，不能替代连续 8–12 小时真实稳定性测试。粗扫 FPS、精扫 FPS、实际推理吞吐、视频解码与素材导出耗时必须分开报告。

## 执行与查看

```bash
# 在已验证的3090 Ti解释器/模型环境中执行；现有已验证AI设置按环境开关加载。
VISIONCORTEX_WEB_AI_SETTINGS=1 PYTHONPATH=src \
  /srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python -m visioncortex process-device-days \
  --config configs/rtx3090ti-ubuntu-production.yaml --date 2026-09-01

# 独立重跑某阶段（先满足前序回执），失败重试需显式标记。
VISIONCORTEX_WEB_AI_SETTINGS=1 PYTHONPATH=src \
  /srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python -m visioncortex process-device-days \
  --config configs/rtx3090ti-ubuntu-production.yaml --date 2026-09-01 \
  --stage understanding --retry-failed
```

新服务启用 `device_day.enabled` 时自动发现并处理；NAS 批次按钮提交到设备日流程。网页入口 `/device-days`，数据入口 `/api/device-days`、`/api/device-days/{name}/index`；文件入口限定在五目录内。comment 与 protocol 分别通过该设备日的 `/comments` 和 `/protocol` 提交。旧实验上传、多视角历史归档阅读保留其既有接口，不自动迁移历史产物。

本机 local/no-NAS 配置明确关闭该流程。不得从生产继承 NAS 路径去“测试本地模式”。本规范不代表开发代码已正式发布；正式发布仍需双仓库策略规定的 CI、真实质量验收与一致 SHA 门槛。

NAS 根目录的 `README.html` 是阅读入口，五个设备日主目录不变。第二目录严格只有 `Index.json` 和 `Clips/`。详细运行回执、YOLO 审计、旧试跑产物保存在配置的 `storage.local_cache_root/device-day-receipts/<设备日>/<recording_id>/`，此持久化回执区不得按临时缓存淘汰。总索引 `recordings[].sources` 保留原片来源，`processing` 保留粗精扫批次、分类依据、速度与审计引用。审计引用带 `storage_root: local_cache_root` 并相对此配置根解析，不能混作归档相对路径。迁移回执保存旧新路径和校验值，原始采集文件不被清理。

## 日期隔离

一个设备日只消费该设备该日期的采集记录、录音、comment 与 protocol。历史补跑按日期分别生成输入清单、执行回执、质量说明和日报。9 月 1 日缺少录音时必须如实记录 no_audio；其他日期的录音不得补入该日期，也不得用其他日期的 STT 成功证明该日期流程完成。跨日期的总队列仅表示任务调度，不能替代单日验收统计。

## 固定报告模板

设备日报使用 `VC-DEVICE-DAY-REPORT-V1` 适配契约，栏目和顺序直接读取现有 `VC-LAB-DAILY-REPORT-V2`：今日概览、实验简报、关注事项与交接、耗时与模型用量。适配契约保留单设备观察与五类动作/场景帧区分，不冒充 V2 的正式双视角验收。原有 V1/V2 模板文件不回写。多模态理解报告使用 `VC-DEVICE-UNDERSTANDING-REPORT-V1`：理解范围、时序步骤与场景、comment/protocol/录音来源、不确定项、模型执行与素材溯源。

两类 JSON 均保存模板 ID、版本、SHA-256 与渲染器 SHA-256；日报同时保留基模板 ID 和 SHA-256。模板栏目由代码固定，模型只填结构化内容。模板呈现不新增模型调用和 Token。

网页沿用 NAS 采集批次监控，并提供 `/#/capture` 导航入口；设备日页面 `/device-days/<日期_设备名>` 对应同一五目录，支持视频区间、帧图片和原录音查阅。实时上传优先于显式历史补跑，同一优先级按 FIFO 处理。空间不足时暂停接收新的原片留存，保留排队任务和已归档结果，采集端删除仍关闭。

## 最终命名与短片段连续调度

用户最终命名要求：所有展示目录、子目录和文件使用英文，采用大驼峰 PascalCase，每个单词首字母大写，无空格、无排序序号；扩展名小写。日期、设备原始标识、时间范围以及作为数据主键的确定性标识原样保留。五个一级目录是 `MetaVideo`、`ProcessedClips`、`MultimodalUnderstanding`、`LaboratoryDailyReport`、`Comment`。子目录为 `Audio`、`Metadata`、`Clips`、`SceneFrames`、`KeyFrames`、`ClipUnderstanding`、`Stt`；关键帧的动作子目录由既有 action_type 去除下划线并将每个单词首字母大写。模型窗口使用实际片内起止时间命名，不使用序号。

0–5、5–7、7–10 分钟这样的连续短片独立入队，不拼接、凑批或补齐到15分钟。每个 GPU 处理名额独立领取一个分片；有空闲名额立即补入下一片，不等待其他分片结束。默认最多2个并行视觉分片，复用现有模型池；该上限是资源控制，不是已验证的最大硬件吞吐。多模态、STT、原片归档和报告有独立并发名额，旧手工流水线与自动视觉组共享 GPU 预约锁。

3090 Ti 生产监控每5秒进行下一轮扫描，关闭发布的文件稳定5秒后可被发现；一条采集记录检查完成即递交处理，不等待整批扫描或其他相机。复制前后再次检查身份并验证哈希。扫描、网络和复制耗时另计，不能宣称上传后固定5秒内完成YOLO。无空间时保留持久待办，采集端删除继续关闭。

### 已绑定 Qwen 专用语音识别（2026-09-10 用户补充）

生产配置固定 `speech_recognition.provider=aliyun_qwen`、`model=qwen-audio-3.0-asr-flash`；不静默回退到 Whisper 或旧 Qwen 模型。录音按实际时长划分最长 180 秒的识别窗口，原录音不切改；16 kHz 单声道派生音频通过 Base64 请求发送给已绑定的 Aliyun 服务，无需公开 NAS 文件。保存句段与词级原录音时间、派生字幕时间、模型名称、适配器 SHA-256、原始响应、请求 ID、用量及执行耗时。流式累计文本按供应商最终时间戳去重，未完成的句子或缺失时间戳不能发布。空识别结果保留来源，不能冒充没有人说话。模型别名的服务端权重无法在本地验证；真实转写准确性仍需人工复核。

模型选择及接口依据：[Aliyun 语音识别模型](https://help.aliyun.com/zh/model-studio/asr-model/)、[Qwen-Audio-3.0-ASR-Flash HTTP API](https://help.aliyun.com/zh/model-studio/fun-asr-flash-recorded-speech-recognition-http-api)。此变更仅更新语音识别实现与相应回执，保留既定五目录、日期/设备隔离及关闭采集端删除的规则。

### 设备内活动筛选与物理动作确证

手工实验入口与 NAS 设备日入口共用算法核心，区别仅限于输入组织、调度和产物发布。两者调用 `actions.fine_scan_windows` 规划精扫窗口，调用 `actions.build_view_activity_intervals` 生成各相机的活动区间，并复用 `candidate_index` 的精扫覆盖核验、轨迹衔接及本地索引位置解析。运行中的 SQLite 索引位于配置的本地 runtime 根目录，关闭数据库连接后才将快照归档；不得在 NAS 上运行 SQLite WAL。设备日入口保留既有第一人称运动召回与开放词汇粗扫门禁，命中候选的短片沿用既有整段精扫策略；未命中候选且覆盖合格的片段仍可快速通过。

活动连续性同样由共享核心处理：精扫活动证据作为起点，按照已有实验间隔及上下文扩展上限，关联具有共同物体或轨迹的精扫观察；有物体依据的粗扫运动仅用于衔接精扫观察，不能单独开启或延长活动边界。保存衔接依据及被引用的候选，剩余区间作为无活动结果留存。手工入口将同一活动区间写入审计产物，设备日入口据此导出片段；两者均不因此修改正式物理动作的接受状态。按设备独立运行不等于已经完成多相机联合对齐和确证，后者仍需独立的真实多视角验收。

单设备片段筛选不能把“未通过跨视角/语义物理动作确证”直接等同于“无实验活动”。保留原有粗扫、精扫、动作审计和五类动作关键帧规则；已有精扫接受事件，或达到既有强上下文置信度、持续时间、观察数且包含接近/释放变化的精扫接触证据，可以在相交粗扫范围及精扫边界内界定设备活动片段。只有粗扫检测、弱接触和静态邻近不作为这类片段依据。`device_activity_decisions` 保存候选与事件引用、起止时间、依据、置信度和 `physical_action_confirmed: false`，不修改原事件的 accepted 状态，也不因此增加动作关键帧。未通过确证的物理动作不作为正式动作结论；单设备活动召回仍需真实视频质量复核。

音频元数据出现 NAS 读取错误时，视频只依据已核验的 RGB 与配对 CSV/就绪说明继续预处理；音频保持 pending_publication，保存 read_errors，禁止假定无录音或使用其他设备/日期的录音。附属采集文件由独立发布任务补齐，不能占用完成原片留存后的 YOLO 唤醒路径。

### 动态相机通道与自动恢复（2026-09-11）

启用 `device_day.camera_lanes` 后，原片及采集资料归档、YOLO、录音 STT、多模态理解、日报分别按已发现且已绑定视角的实际相机建立通道，不固定相机数量。每个阶段同一相机同时领取一个分片，按采集时间推进；其他相机无需等待。阶段间通过持久化结果衔接，预处理不等待云端理解。GPU 的批大小及单通道解码并行度仍受资源配置限制，实际吞吐需运行回执验证。

NAS 监控按相机分别扫描当日新增数据和历史数据，历史文件读取阻塞不阻塞其他相机或该相机的当日发现。闭合短片按实际长度入队。相同处理版本失败自动尝试最多三次；百炼明确返回 Arrearage 时隔离云端理解和 STT，保留待办，原片归档和 YOLO 继续。系统每十五分钟至多进行一次连接恢复验证，验证成功后自动恢复云端调度，不需要人工重置整批任务。网页独立显示预处理完成与全流程完成。采集端删除保持关闭。

### 单设备动作关键帧的筛选范围

单设备关键帧与完整实验分段分别准入：已通过现有正式 CV 审核、属于五类物理动作、位于该设备活动时间范围且未被状态机标记为重复组成动作的事件，使用与完整实验入口相同的关键素材去重和覆盖预算算法筛选。完整实验段或实验组为空不再自动清空这类动作素材。选择窗口仅用于素材预算，不作为已确认实验段或跨视角实验组发布。待复核、被拒绝事件及普通场景采样不得因此提升为动作关键帧。保留原有 PARTIAL_EVIDENCE 和 physical_action_confirmed: false，CV 准入不替代动作语义确认。回执保存 key_selection_summary 与 key_selection_decisions，目录与 v1 索引语义不变。

### 采集边界与实验时间线

采集分片是调度与溯源单位，不是实验开始或结束的依据。单设备活动审核不调用要求第一、第三人称配对的实验组构建；正式多视角实验入口继续使用原有门禁。片段保留原片引用、局部时间及采集时间映射后的 `start_us` / `end_us`，供同一日期的连续时间线关联。相邻分片本身不能证明属于同一实验；实验员身份需要对应时段的真实绑定信息，不能从相机名称推定。跨分片完整实验恢复仍须单独验证，不能以单片审核通过代替。

### 吞吐与证据留存约束

检测明细及扫描回执在后端封存后，可以通过原有 `audit_artifacts` 引用共享，不为每次下游重试再复制同一份明细；被回执引用的扫描文件属于留存证据，不能作为可随意清理的临时缓存。历史复制式回执继续可读。进程内可在文件身份和预期摘要均未变化时共享成功的完整性校验，重启、文件身份或摘要变化后重新校验。单阶段任务在自身产出后更新日索引，不为已发布的前置阶段重复重建。

相机通道数量仍动态发现。原片大文件复制可通过 `device_day.retention_io_workers` 单独限制 I/O 并发，0 表示不额外限制；视觉、录音和理解通道不继承该复制上限。当前 3090 Ti 配置复制上限为 2，以减少复制与视频解码之间的 NAS 竞争。采集端删除继续关闭。


### 调度版本与归档供给（2026-09-11）

调度轮询、相机轮转和索引呈现不再通过整个 runner 文件摘要使已完成产物失效；执行函数采用跨 Python 版本归一化的 AST 指纹。完全相同的当前执行实现映射回迁移前代码摘要，保持现有五阶段任务键不变。实际执行逻辑、输入、模型、阶段配置及原有依赖文件变化仍使相应键改变。新增执行函数必须纳入身份函数的依赖覆盖，不能用调度修改掩盖产物语义变更。

只对已审查的归档旧实现允许精确键兼容：`36019d3ed2d5d5faec1a88604e4ec1835c915753c91bf3c47afff41c1cb91497` 的原片复制与归档函数与当前基线语义一致，且旧键必须重新匹配当前源签名、归档契约、采集/音频代码及配置。复用时仍逐项校验归档材料，保留旧回执本身；该兼容不适用于 YOLO 或其他模型结果。未知旧版本、变化的输入和损坏的材料不得复用。

同一调度优先级下，相机以持久化的最近领取时间轮转，同相机内部按采集时间依次处理，仍禁止同阶段同相机同时领取多个分片。大文件 I/O 上限仅限制同时复制数量，不再让较早日期相机耗尽所有归档机会。原片归档完成即解除对应 YOLO 前置条件，不等待其他相机或云端阶段。
