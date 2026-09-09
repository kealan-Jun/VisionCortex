# VisionCortex

`VisionCortex` 是仓库、安装包、命令行、API、网页、模型与数据注册表、报告和证据产物的唯一项目名称。新增能力和产物必须继续使用该名称及 `visioncortex-` 机器标识前缀。

## RTX 4050 客户更新

已有离线包的客户统一从 [4050 更新分支](https://github.com/RealityLoopAI/VisionCortex/tree/codex/rtx4050-optimization-20260908)
拉取源码，再运行代码目录的 `deployment\rtx4050-windows\Update-From-Source.cmd`，选择原解压目录。
更新后仍从原目录启动 `VisionCortex.exe`，无需反复下载模型和大压缩包。
[详细更新步骤](docs/RTX4050-GIT-UPDATES.md) · [分支与历史归档](docs/REPOSITORY-BRANCHES.md)。
该分支用于产品集成和客户验收，真实视频质量及稳定发布门禁仍需各自的执行回执。

## 交付与使用文档

按角色选择入口，不需要从头通读所有文档：

| 角色/目的 | 首先阅读 |
| --- | --- |
| 第一次提交视频分析 | [普通用户 10 分钟操作指南](docs/VisionCortex-普通用户10分钟操作指南.md) |
| 安装、运维、升级和故障处理 | [端到端视频分析交付手册](docs/VisionCortex-端到端视频分析交付手册.md) |
| 正式交付签字 | [交付验收单模板](docs/VisionCortex-交付验收单模板.md) |
| 理解真实运行证据及其边界 | [历史真实六路验收示例](docs/VisionCortex-历史真实六路验收示例.md) |

页面截图来自本地合成验收，只用于识别按钮和页面，不是生产质量证明。正式交付必须
在生产配置下完成真实视频、真实模型、NAS 归档和人工抽查，并保存对应回执。

## 用户快速启动

第一次使用时，不需要先判断自己的显卡或操作系统。启动入口会先检查环境，
只给出中文结果，不会自动安装软件、访问 NAS 或运行模型。

Windows 用户双击仓库根目录的 `Start-VisionCortex.bat`。Linux 用户运行：

```bash
./start-visioncortex.sh
```

macOS 用户可以双击 `Start-VisionCortex.command`，也可以执行上面的 Linux 命令。
默认启动 `configs/development-local.yaml` 本地开发配置，并打开
`http://127.0.0.1:8000/#/home`。该入口只证明本地网页可以启动；真实模型推理、
真实视频效果和正式发布能力仍须在对应 NVIDIA GPU 节点单独验收。

只检查、不启动：

```powershell
# Windows
.\Start-VisionCortex.ps1 -CheckOnly

# Linux / macOS
./start-visioncortex.sh --check-only

# 需要机器可读的完整环境结果
python tools/doctor.py --json
```

RTX 4060 真实六路运行节点必须先阅读
[`docs/RTX4060-Codex-真实六路全链路执行任务书.md`](docs/RTX4060-Codex-真实六路全链路执行任务书.md)，
并在运行结束后填写
[`docs/RTX4060-真实六路运行回传模板.md`](docs/RTX4060-真实六路运行回传模板.md)。
运行节点只同步冻结提交并执行真实全链路，不修改代码、不运行开发测试。

> 冻结基线：六路、多视角、3 小时湿实验视频的时间对齐、有界实验筛选、六类关键素材和细粒度步骤理解流水线。RTX 4060 部署、固定 NAS 基准、缓存目录与开发协作方式见 [RTX4060-交接与运行说明.md](RTX4060-交接与运行说明.md)。

## 本机选择 AI 厂商与输入密钥

Ubuntu 3090 Ti 本机分析服务的左侧菜单提供 **AI 服务设置**：
打开 `http://127.0.0.1:8001/#/ai-settings`，选择服务并输入该服务的 Key，点击 **读取可用模型**，
选择视觉模型后点击 **验证并启用**。预设包含火山引擎、阿里云百炼、智谱、硅基流动、
Google Gemini 和 OpenRouter；其他厂商或自建服务可填写 HTTPS Base URL，使用
Chat Completions 图像接口接入。预设目录位于 `configs/mllm-providers.json`，新增兼容厂商无需修改后端厂商白名单。

读取模型只请求所选接口的 `/models`，不遍历厂商、不跟随重定向，也不保存或启用配置。
厂商明确标注不支持图像的模型会被排除；未标注能力的模型保留为待验证项。
列表缺失、未完整返回或需要私有部署时，可以直接填写视觉模型 / 部署 ID。
Key 不能可靠识别全部厂商；没有兼容接口的服务仍需要专用协议适配。
新增预设依据官方接口说明接入（[Gemini](https://ai.google.dev/gemini-api/docs/openai)、
[硅基流动](https://docs.siliconflow.cn/docs/api/models-get)、
[OpenRouter](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties)），
真实账号调用与实验质量须分别验证，不能由目录读取成功推出。
验证会联网发送两张随机测试图并产生少量 API 费用；多图识别和 JSON 响应通过后才保存。

每家厂商可分别保存密钥；再次选择同厂商、同接口地址时可留空复用。保存成功后无需
重启，新提交任务使用所选配置；已有任务保留提交时的厂商、模型和凭据版本。
右侧显示当前模型、验证时间、请求编号与 Token 用量。连接通过不等于实验质量通过。
适配器代码更新后，旧连接验证会失效；须使用原厂商、模型和已保存密钥重新验证后再复跑。

阿里云质量模式使用流式响应接收长思考结果，完整结束标记、最终答案格式和截断检查
全部通过后才接受结果；流式思考文本不保存为实验结论。尾部 Token 回执会计入用量，
未返回用量的失败请求仍是未知用量。此修复不改变图片、提示词、模型、思考开关或输出
预算，因此已通过严格校验的语义缓存可继续复用。其他兼容厂商仍按各自已验证协议调用。

连接验证使用 180 秒流式总预算及 30 秒无数据等待上限，仍只有一次付费尝试。
实验分析默认流式总预算为 600 秒，超过预算后在下一块数据到达时停止；没有新数据时，
由原有 `timeout_seconds` 中止。因此最坏等待还包括一次无数据等待，并非无限等待。
8 MiB 接收大小限制同样会拒绝不完整结果。网络、厂商拥塞、响应格式与质量检查仍可能
导致分析未完成，连接验证成功不能替代真实全流程验收。

本机设置接口只允许本机回环地址与管理员身份访问，并校验请求来源。安装模板通过
`VISIONCORTEX_WEB_AI_SETTINGS=1` 启用；其他部署默认不读取这份本机设置。
密钥保存在当前用户 `~/.config/VisionCortex/ai-services`，目录权限为 700、文件为 600，
不进入 Git、任务配置、NAS 或报告。凭据历史版本供已提交任务恢复使用，不要手动删除。

## Ubuntu 本机独立体验实例

已配置模型、Python 环境和 AI 服务的 Ubuntu 工作站，可用 `tools/local_experience.py`
建立一个独立实例，模拟首次配置和提交真实视频的用户流程：

```bash
python tools/local_experience.py prepare --root /本地磁盘/LocalExperience --source /本地原视频目录
python tools/local_experience.py check --root /本地磁盘/LocalExperience
"/本地磁盘/LocalExperience/打开 VisionCortex.sh"
```

必须使用已验证的工作站 Python 环境。该实例保存源码快照、独立配置、队列和运行结果，
复用本机依赖、模型及原视频引用，不复制模型和原片，不包含密钥，不可直接搬到其他电脑。
预检会核对依赖、模型文件哈希、输入元数据、GPU 身份和剩余空间；它不证明真实分析质量。
首次界面连接验证仍须成功后才启用 AI 配置。

同一目录再次 `prepare` 会拒绝；再次启动复用所属服务。修复在原实例内完成，并更新
`Acceptance` 中的文件身份和实测回执。`stop --root /本地磁盘/LocalExperience` 只停止
该实例且拒绝中断活动任务。真实视频完整运行、浏览器结果可用和正式发布资格须分别验收。

分析已执行完但最终质量检查未通过时，任务以 `partial`（分析结束 · 证据不足）结束，
释放执行队列并保存阶段 HTML 报告、片段、素材及具体缺口。它不会发布正式档案或生成
已验收的完整结论，也不要求人工批准候选。任务页、实验记录和全局素材库均可查阅阶段
成果；原任务支持重新分析，复用经过验证的输入和缓存。相同证据不足不会自动无限复跑
或反复调用付费模型；语义请求自身继续使用有界恢复策略。实际执行异常仍记录为失败。

实验候选片段之间的间隙不等于实验结束。启用视觉模型后，导出前按第一人称录像分别
复核相邻切点。明确的操作延续合并为同一实验单元；有样品/器具承接的不同实验单元
保留在同一连续实验链中。换实验台或第三人称机位不再强制分开，但必须有移动/携带的
画面依据；同一实验员本身不足以合并。原始 CV 原子片段、动作成员与模型单元分开保存。对可能承接但证据不足的切点，最多增加一次高密度第一人称复核，阈值不降低；尾部复核推进切点后，允许核对一次变化后的相邻边界。独立请求按模型并发配置执行。

系统继续读取检测终点之后的原视频。默认切点复核预算为间隙 120 秒、两侧各 12 秒、
20 张图；尾部按 90 秒窗口继续、每组最多 8 窗口。预算耗尽、采样不足或模型失败均保留
“边界待核实”，不会解释为实验完成。已观察到录像末尾仍操作时标记“录像结束，实验待续”，
保留可确认承接的尾部视频。后续录像可通过同一输入的分段索引一并分析；尚未实现跨任务
自动拼接。单步结束、实验单元结束与整个连续链结束不能互相替代。

尾部复核 v4 将单步、实验单元和整个连续实验链的完成分开记录。确认整个链结束时，
必须引用结束之后至少两张画面（含本次最后一张），覆盖默认至少 12 秒的后续核对，
且未观察到相关操作继续。只有录像最后一帧或单步完成的依据不足以关闭实验链；
这一检查也不保证尚未录制的未来不会继续。尾部承接需操作者与实际操作对象的前后证据，
跨台还需携带对象的画面引用。

一次不确定结果不会立即截断检查。默认可增加最多两次有界前探，仍计入每组 8 窗口
总预算，保留原始切点与上一检查终点；只有后续承接依据通过后才延长输出视频。
独立新任务、模型调用失败或预算耗尽均停止检查，保留未确认状态与停止原因。
参数为 `mllm.boundary_review.maximum_uncertain_tail_windows` 和
`completion_followup_seconds`（不能低于 12 秒）。旧尾部缓存不会用于新规则的完成确认。

连续链保留单元时间范围及机位时间表。第三人称随对应单元切换；未确认机位的过渡和尾部
显示“未确认对应机位”，第一人称连续保留，不用旧实验台冒充对照画面。复核图片、编号、
模型结果及 Token 保存在 `JSON-Config-Files/experiment_boundary_review.json` 和
`Boundary-Review-Frames/`。这些是模型边界证据，不是动作人工真值或完整步骤召回证明。
边界未解决、或延长视频后步骤尚未重新核验的产出保留为阶段结果，不能通过完整实验验收；不会自动更新旧安装包和旧任务。

## 3090 Ti 局域网服务器

正式使用时，Windows、macOS、Linux 用户都不需要安装本项目，也不需要配置
CUDA、TensorRT 或模型。管理员只在 Ubuntu 3090 Ti 主机完成一次安装，然后运行：

```bash
./deployment/rtx3090ti-ubuntu/07-Install-LAN-Server.sh
```

安装器会要求管理员在终端中设置一次网页登录密码，不回显密码，也不会把密码写入
Git。完成后会显示类似 `http://192.168.x.x:8000/#/home` 的局域网地址。其他用户
只需打开该地址，以用户名 `visioncortex` 和管理员设置的密码登录，即可选择 NAS
批次、提交任务、查看进度和结果。多人同时提交时，3090 Ti 每次只执行一个正式
GPU 任务，其余任务保留为“排队等待”，避免互相争抢显存。排队内容与页面任务
状态先写入服务器本地 `Runtime/state/web_run_queue.sqlite3`；Web 服务或服务器重启
后会自动恢复等待任务，不需要用户重新提交。

浏览器上传不再使用一次性大请求，也不按路数、时长或固定总 GB 数拒绝任务。页面先
把本次全部文件的真实字节数交给服务器；服务器结合 NAS 当前剩余空间、其他上传
尚未消耗的预留量、预计处理产出和安全余量，决定是否接收。通过后以 16 MiB 小块
续传，网络中断会从服务器确认的字节位置继续；重新选择同一批文件后也会核对断点
内容，避免把不同文件拼在一起。3090 Ti 生产配置只在 NAS 正式档案保存一份原片，
不在服务器本地再复制一份。连续 7 天没有继续的未完成会话会在服务启动或下一次
容量检查时过期并清理；已经进入任务队列的会话不受该规则影响。

该服务使用 3090 Ti 正式生产配置，启动前检查 GPU、NAS、固定 Python 环境和本地
空间；检查不通过时拒绝启动。它只接受回环地址和常见私有局域网地址，并要求每次
浏览器会话登录，不是公网发布方案。服务状态检查：

```bash
./deployment/rtx3090ti-ubuntu/08-Server-Status.sh
```

4060 和 3060 可以继续作为开发或备用机器，但普通用户不再需要在这些机器上拉取
仓库。真实模型、真实视频质量和正式归档能力仍须在 3090 Ti 主机按下面的生产门禁
验收，网页能打开本身不代表完整推理已经通过。

归档查阅不会在每次打开页面时重新遍历并读取全部 JSON。服务会把正式归档中已有的
`evidence_index.sqlite` 同步为 3090 Ti 本地运行目录下的可重建读取索引；归档 JSON
和逐归档 SQLite 仍是权威数据，读取索引损坏或删除后会自动重建，不复制原视频。
归档目录、实验片段和关键事件均分页加载，中文动作词会扩展到现有动作本体的中英文
同义词；素材 URL 绑定当前 `release_id`，旧版本链接会返回冲突提示而不会继续展示
浏览器缓存。关键帧延迟加载、关键片段不预加载正文，服务默认最多并发传输 8 个归档
文件；可用 `VISIONCORTEX_WEB_MAX_CONCURRENT_ARCHIVE_STREAMS` 调整，超出时返回
`429` 并提示浏览器稍后重试，避免多人播放把 NAS 链路拖死。

安装器创建的单一 `visioncortex` 管理账号继续兼容。多人使用时，可改用权限为 `600`
的 `VISIONCORTEX_WEB_USERS_FILE`，每个人使用独立账号和 `viewer`、`operator` 或
`admin` 角色；`viewer` 只能查阅，后两者可提交任务。密码文件只保存 PBKDF2 哈希，
可在服务器终端生成：

```bash
visioncortex hash-web-password
```

用户文件格式如下，不能写入明文密码，也不能提交到 Git：

```json
{
  "schema_version": "visioncortex-web-users/1",
  "users": [
    {"username": "researcher-01", "role": "viewer", "password_hash": "<PBKDF2_HASH>"},
    {"username": "operator-01", "role": "operator", "password_hash": "<PBKDF2_HASH>"}
  ]
}
```

局域网 API 的账号、角色、来源 IP、方法、路径、状态码和耗时会按日写入服务器本地
`Runtime/state/web_access_audit-YYYY-MM-DD.jsonl`，不记录密码、Authorization 或
查询参数。若通过 Nginx/Caddy 等受控反向代理提供 TLS，可设置
`VISIONCORTEX_WEB_REQUIRE_HTTPS=true` 强制拒绝非 HTTPS 请求；当前安装器显示的
`http://192.168.x.x` 仅适用于受信任、隔离的内网，不能作为公网入口。

## Ubuntu RTX 3090 Ti 本地结构

```text
/home/x1/Projects/VisionCortex
  └── 源代码、模型注册表、配置与部署脚本

/home/x1/VisionCortex-project        # 指向上述源码目录的稳定入口

/srv/sentinel-data/VisionCortex3090Ti
  ├── .venv/                         # 固定 Python/CUDA 依赖
  ├── Models/ClosedSetYOLO/<role>/   # 可训练源权重，不进入 Git
  ├── Engines/                       # TensorRT 与公共模型权重
  └── Runtime/
      ├── ThirdParty/                # 隔离的第三方可变配置
      ├── Model-Quality/              # 评估、认证就绪度与验收收据
      └── NoNasWeb/                   # NAS 断开时的 Web/缓存/产出
```

Git 仓库只保存代码、模型版本/下载地址和 SHA-256，不保存 `.pt`、
`.engine`、`.safetensors`、原视频、运行产出或密钥。两套项目训练的闭集
权重须按 `configs/models/closed-set-yolo.json` 注册到本地
`Models/ClosedSetYOLO/<role>/best.pt`；3090 Ti 安装器会验固定哈希、重建 TensorRT
引擎，并自动下载和验签 YOLO-World、CLIP、Grounding DINO、SAM2.1 与
LabPics 液体/填充语义分割公共资产。

当前最终模型链不是单一 YOLO：两套 21 类 TensorRT 模型负责全时间轴候选；
ByteTrack 风格的两阶段关联保持对象轨迹；YOLO-World 与 Grounding DINO 仅在
已接受关键帧和有界时序补救中补齐小物体；豆包对双视角时序证据做动作与步骤
裁决；SAM2.1 对裁决后的参与对象框在短关键片段中双向传播，收紧最终展示框并
留下连续性收据；LabPics PSPNet 只在已接受的最终关键帧补充 vessel、filled、
liquid/solid phase 像素观察。开放词汇框、SAM2 掩码和 LabPics 掩码都不能单独
确认动作，正式生产仍受事件/参与对象框质量认证的 fail-closed 门禁约束。

RTX 3090 Ti 配置还启用有界选择性复核：液体、容器状态和移液关键帧固定进入
本地二次复核；其他动作仅在参与对象缺失、低置信、多实例或跨视角冲突时调用
开放词汇模型。清晰的闭集结果不重复推理。每次运行最多复核 120 个事件、每个
事件 2 个视角，并写入 `final_key_material_annotation.json` 的决策、模型耗时、
预算和 `source_copy_bytes=0` 回执。紧急回退可设置
`VISIONCORTEX_SELECTIVE_KEY_MATERIAL_VERIFICATION=false`，恢复原有行为。

### NAS 不可用时的本地验收

以下入口全部只使用 `/srv/sentinel-data/VisionCortex3090Ti/Runtime`，不会创建、
轮询或写入 NAS 路径：

```bash
# 六视角媒体、六类关键素材、步骤理解、日报、PDF、JSON/JSONL/SQLite
visioncortex run-local-acceptance \
  --output /srv/sentinel-data/VisionCortex3090Ti/Runtime/LocalAcceptance \
  --config configs/rtx3090ti-ubuntu-local.yaml

# 六路 CUDA 解码 + 双 TensorRT 角色引擎硬件压测
visioncortex benchmark-local-hardware --output <local-output> \
  --media <h264-1> --media <h264-2> --media <h264-3> \
  --media <h264-4> --media <h264-5> --media <h264-6>

# 依次测量每角色 1/2/3 个 TensorRT 压测上下文；只给出容量结论，不自动改生产并发
visioncortex tune-local-hardware --output <new-local-output> \
  --duration-seconds 20 --config configs/rtx3090ti-ubuntu-local.yaml \
  --media <h264-1> --media <h264-2> --media <h264-3> \
  --media <h264-4> --media <h264-5> --media <h264-6>

# 真实执行全部本地生产 CV 模型；使用公开人工标注样本，不访问 NAS/豆包
visioncortex accept-local-models \
  --dataset /srv/sentinel-data/VisionCortex3090Ti/Runtime/PublicDatasets/LabPicsChemistry/extracted \
  --output /srv/sentinel-data/VisionCortex3090Ti/Runtime/Model-Quality/<new-run> \
  --config configs/rtx3090ti-ubuntu-local.yaml

# 只读统计正式认证仍缺多少真值；不扫描生产归档
visioncortex model-certification-readiness \
  --config configs/rtx3090ti-ubuntu-production.yaml \
  --output /srv/sentinel-data/VisionCortex3090Ti/Runtime/Model-Quality/readiness.json

# 安装仅绑定 127.0.0.1、重启自恢复且不继承 NAS 路径的离线验收 Web
deployment/rtx3090ti-ubuntu/05-Install-Local-Service.sh
```

本地六视角验收素材是明确标记的合成媒体，只证明结构、媒体、索引、报告与硬件
链路，不能作为真实实验质量结论。Web 使用
`configs/rtx3090ti-ubuntu-local.yaml`，关闭豆包与 collection ingest，并把归档、
输入、缓存、运行和 staging 全部固定到 `Runtime/NoNasWeb`。

液体语义公共数据由 `configs/public-data-sources.json` 管理；只有许可证明确、
HTTPS 且 SHA-256 固定的条目允许自动获取。`prepare-public-dataset` 会先验证完整
ZIP/RAR，再拒绝路径穿越、重复成员、链接和特殊文件，并在有界解包、文件数与
字节数复核全部通过后原子发布目录；失败的 partial 只作审计，绝不冒充完成数据。
模型共识只能通过
`build-consensus-labels` 生成 `pseudo_labels_not_ground_truth`，不能冒充人工真值。

双闭集源权重可用 `validate-closed-set-models` 对注册哈希和 21 类本体做双重
校验。框真值完成后，`evaluate-yolo-boxes` 计算独立 P/R/AP；
`build-yolo-training-dataset` 用软链接或硬链接构建零拷贝训练集；
`train-yolo-model` 只接受 reviewed ground truth，并把新权重标记为“未认证候选”。
公开模型、模型共识或训练日志都不能解除生产认证门禁。

Waseda Chemical Apparatus 公共人工框数据已固定 URL、大小与 SHA-256。它可用于
补充 hand、pipette 和六类实验器具，但不能直接替换项目的 21 类生产本体：

```bash
visioncortex prepare-public-dataset --dataset-id WasedaChemicalApparatus \
  --destination <local-public-dataset-root>
visioncortex build-public-yolo-training-view --source <extracted-root> \
  --dataset-receipt <dataset-receipt.json> --output <new-zero-copy-view>
visioncortex train-yolo-model --dataset <new-zero-copy-view> --base-model <best.pt> \
  --output <new-candidate> --epochs 60 --max-hours 1.5 --patience 15
visioncortex evaluate-yolo-model-on-human-truth --dataset <new-zero-copy-view> \
  --model <new-candidate>/weights/best.pt --split test --output <new-evaluation>
```

从已训练候选继续微调时，应显式固定优化器与学习率，避免 `optimizer=auto`
重新进入高学习率 warmup；这些参数会写入训练回执：

```bash
visioncortex train-yolo-model --dataset <mapped-21-class-union> \
  --audit-receipt <integrity-audit.json> \
  --base-model <previous-best.pt> --output <fine-tuned-candidate> \
  --optimizer AdamW --learning-rate 0.001 \
  --final-learning-rate-fraction 0.05 --cosine-schedule --warmup-epochs 1
```

ChemEq25 公开人工框数据也已固定 Figshare v3 的 RAR 工件 URL、大小、MD5 与
SHA-256；157 条多边形标注会在验证后确定性转为其最小外接框。两个来源必须先按
`configs/models/public-yolo-ontology-map.json` 显式映射进生产 21 类本体；未映射
类别逐类写明为 `null`，不能靠名称猜测。训练集可对稀缺的 hand/pipette 做有界
过采样，但 val/test 永不重复：

```bash
visioncortex prepare-public-dataset --dataset-id ChemEq25 \
  --destination <local-public-dataset-root>
visioncortex build-public-yolo-training-view --source <chemeq-extracted-root> \
  --dataset-receipt <chemeq-dataset-receipt.json> --output <chemeq-zero-copy-view>
visioncortex build-mapped-public-yolo-union \
  --source WasedaChemicalApparatus=<waseda-zero-copy-view> \
  --source ChemEq25=<chemeq-zero-copy-view> --output <mapped-21-class-union>
visioncortex audit-yolo-dataset-integrity --dataset <mapped-21-class-union> \
  --output <new-integrity-audit> --focus-classes hand,pipette
```

映射合并会在任何过采样之前按源图 SHA-256 去重；同内容跨 split 时按
`test > val > train` 只保留一个权威 split，同 split 的重复框做确定性合并。训练、
阈值校准和 TensorRT 候选实测都必须引用通过的完整性审计，发现跨 split 内容泄漏
立即 fail-closed。阈值只允许从验证集冻结；精度、召回双门槛通过后优先选择召回率
最高的候选生成阈值，随后才可读取测试集：

```bash
visioncortex calibrate-yolo-confidence --dataset <leakage-free-union> \
  --model <candidate>/weights/best.pt --audit-receipt <integrity-audit.json> \
  --output <new-val-calibration> --target-classes hand,pipette
visioncortex evaluate-yolo-calibrated --dataset <leakage-free-union> \
  --model <candidate>/weights/best.pt \
  --calibration-receipt <new-val-calibration>/threshold-calibration.json \
  --output <new-test-evaluation> \
  --test-exposure-status first_use_independent
visioncortex calibrate-yolo-world-prompts --dataset <leakage-free-union> \
  --model <yolo-world-v2.pt> --audit-receipt <integrity-audit.json> \
  --prompt-map configs/models/yolo-world-hand-pipette-prompts.json \
  --output <new-yolo-world-val-calibration>
visioncortex measure-yolo-candidate-tensorrt --dataset <leakage-free-union> \
  --model <candidate>/weights/best.pt --audit-receipt <integrity-audit.json> \
  --output <new-candidate-tensorrt-benchmark> \
  --image-size 960 --export-batch 4 --benchmark-image-limit 256
```

同一测试 split 一旦参与过候选取舍，后续模型必须显式传
`--test-exposure-status repeat_comparative_benchmark`，回执会禁止声称它仍是独立
测试；此时真正的生产泛化结论只能由尚未参与开发的内部六视角 A/B 给出。
TensorRT 候选基准固定空间尺寸与导出 batch，并显式按完整静态 batch 分块；回执
只声明该固定形状的端到端吞吐，不外推到未实测 batch 或动态分辨率。

训练与测试均在 epoch 边界强制墙钟上限，记录逐类 P/R/F1/AP、GPU/显存/功耗
遥测，且固定 `production_certified=false`。Web 的“关键素材模型质量账本”展示每个
事件实际执行的闭集 YOLO、YOLO-World、Grounding DINO、SAM2/LabPics、框置信度、
二次核验耗时与保留的不确定性；没有人工真值时不会把模型置信度冒充准确率。
已完成的公共模型候选及其按曝光状态标记的留出集指标登记在
`configs/models/public-apparatus-candidates.json`；注册不等于上线，只有本体兼容且
通过内部真实六视角认证的候选才允许写入生产配置。自动判定命令只生成不可变
决策回执，不会改生产配置；即便公开留出集全部达标，缺少经复核的真实六视角
A/B 回执也必须 fail-closed：

```bash
visioncortex evaluate-yolo-candidate-promotion \
  --evaluation-receipt <evaluation>/visioncortex-evaluation-receipt.json \
  --output <promotion>/promotion-decision.json
```

固定基准不会重复创建实验目录：先启动 Web，再运行 `deployment\rtx4060\04-重跑固定六路基准.ps1`，由常驻 Web 服务异步执行并复用 `Y:\VisionCortexExperimentArchive\CustomFlow_standard_correct_12_ABCFA_0001--exp_20260810_144014_e918b762`。不要在有超时限制的命令包装器里前台运行 `run-fixed-benchmark`。其他用户上传任务仍按实际上传路数动态创建独立档案。

面向化学湿实验长视频的多视角证据流水线。系统把第一人称与第三人称视频先对齐，再以两套 21 类 YOLO 模型和 ByteTrack 生成候选，经过跨视角审计后，提取实验片段、六类物理动作关键帧/关键片段/时间戳，并调用豆包多模态模型生成步骤级理解。

## 产出

每次运行生成以下结构：

```text
<output>/<experiment-id>/
  Experiment-Clips/
    EXP-0001_<start>-<end>/
      EXP-0001_aligned_multiview.mp4  # 主交付：严格等长、同一全局时间轴
      first_person.mp4
      third_person.mp4
      EXP-0001.json
  JSON-Config-Files/
    run_manifest.json
    time_alignment.json
    aligned_timestamps.csv
    alignment_quality_gate.json       # 正式证据门禁、逐路可用区间与隔离分片
    evidence_package.json
    physical_change_log.json
    evidence_package_eval.json
    quality_acceptance.json            # 自动门禁与证据等级；结构通过不等于准确率已测量
    run_metrics.json
    run_provenance.json                # 代码、配置、模型认证与关键产物哈希绑定
    schema_contract_manifest.json      # 跨文件契约与事件引用一致性
    delivery_metrics.json              # Web/NAS 请求到发布门禁的交付耗时，不改写运行指标
    evidence_index.sqlite              # 可重建的事件检索索引（JSON 仍是权威数据）
    evidence_index_manifest.json       # 数量、FTS 能力与文件 SHA-256
    artifact_registry.jsonl            # 事件 → 素材/sidecar/大小/SHA-256
    evidence_registry.jsonl            # 证据 → JSON Pointer → 原视频物理分片/帧
    physical_change_registry.jsonl     # 明确前后状态变化；不推断 unknown 区间
  Key-Materials/
    Key-Clips/<event-id>/first_person.mp4, third_person.mp4
    Key-Frames/<event-id>/first_person.jpg, third_person.jpg
    Key-Materials-Model-Understanding.json
    Key-Material-Timestamps.csv
    Key-Material-Timestamps.xlsx
    Screening-Notes.txt
  Lab-Daily-Reports/<YYYY-MM-DD>/
    Lab-Daily-Report-<date>.json, .md, .html
    Daily-Report-Eval.json
    Automatic-Acceptance.json
  Professional-PDFs/
    VisionCortex-Professional-Evidence-Report-<date>.pdf
  .VisionCortex-Current-Release.json    # 全部门禁通过后才原子切换的当前正式版本
```

日报采用固定的 `VC-LAB-DAILY-REPORT-V2`：模型只产出结构化实验理解，确定性渲染器填充固定栏目，日报阶段不新增模型调用或 Token；不确定证据由算法自动隔离，不设置人工审批兜底。

六类动作是 `hand_object_contact`（手与明确物体接触）、`object_movement`、`liquid_movement`、`container_state_change`、`device_panel_operation` 和 `pipette_transfer_operation`。每个记录保留候选、接受/拒绝理由、视角支持、对齐置信度和不确定性，YOLO 框不会被直接当成最终证据。

## 安装与安全配置

```powershell
Set-Location -LiteralPath 'C:\Users\Xx7\Documents\ChatGPT\New project'
# 轻量开发/确定性测试：不会下载 PyTorch、CUDA、YOLO 或 Transformer。
python -m pip install -e ".[dev]"

# 需要真实模型推理时显式安装模型与 TensorRT 运行栈。
python -m pip install -e ".[models,tensorrt]"
$env:ARK_API_KEY = '<在本机安全设置，不要写入 yaml 或 git>'
```

`models` 包含固定版本的 Transformer、YOLO 与 CLIP 运行依赖；SAM2 仍通过
`sam2` 可选项或部署脚本的固定 revision 独立安装。生产部署脚本优先使用各硬件
目录内的锁定依赖，不会因为轻量开发安装而改变现有模型或 TensorRT 引擎。

聊天中出现过的 API Key 应在火山方舟控制台轮换。代码只读取 `ARK_API_KEY`，不会保存或打印密钥。

## 输入清单

复制 `examples/manifest.example.yaml`，为每一路填写视角、视频和 CSV。CSV 支持常见列名：

- 帧号：`frame_index` / `frame` / `frame_id`
- 时间戳：`timestamp_ms` / `timestamp_s` / `timestamp` / `pts_time`
- 可选共同时间：`global_timestamp_ms` / `wallclock_ms`

时间戳为 ISO-8601 时也可解析。对齐阶段对每个物理分片读取有上限的首、中、尾
分布采样，验证 `clock_sync_valid`、单调性、跳时和漂移，再保存逐分片变换与误差。
系统优先使用质量最好的时钟作为内部基准，同时保留配置中的第一人称偏好；生产
任务即使已有高置信度共同时间，也执行开头、中间、结尾的轻量视觉锚点审计。
若 CSV 没有共同时间列，系统仍保留原有视频起点粗对齐能力并明确标记为
`local_timeline_assumption`；3090 Ti 正式配置要求该结果获得可靠视觉支持后才能
进入正式证据。`aligned_timestamps.csv` 使用完整基准时间轴并为每路记录
`<view>_available`，单路短录或坏分片只会被隔离，不再截断其他视角。
`alignment_runtime.json` 分别记录共享时钟采样、逐路视觉审计、分片拟合耗时和
视觉特征缓存规模，便于在真实长视频上核算新增质量检查的速度成本。
进入 GPU 队列前的 `prequeue_input_preflight.json` 同时记录媒体探测、时钟检查和
总耗时；NAS 模式只做有界元数据/时钟读取，不复制或完整哈希原始长视频。

## 运行

```powershell
visioncortex validate-models --config .\configs\default.yaml
visioncortex prepare-engine --config .\configs\default.yaml
visioncortex run --manifest .\examples\manifest.example.yaml --config .\configs\default.yaml
```

没有真实视频时可完整验证目录、JSON 契约和评估器：

```powershell
visioncortex dry-run --output .\outputs\dry-run
pytest -q
```

上传服务：

```powershell
visioncortex serve --host 127.0.0.1 --port 8000
```

浏览器先以 `POST /api/upload-sessions` 创建动态空间预留，再对
`PATCH /api/upload-sessions/{session_id}/files/{file_id}` 发送可恢复小分块，最后调用
`POST /api/upload-sessions/{session_id}/finalize` 完整校验并进入后台队列；任务状态仍
从 `GET /api/runs/{run_id}` 查询。旧的 `POST /api/runs` 一次性 multipart 接口暂时保留
用于兼容旧客户端，新页面不再使用它。API 只绑定本机，除非显式改为 `0.0.0.0`。

同一浏览器机位既可上传一个连续视频，也可按顺序上传多个原始分片；两种输入都会
转换为同一份 `RunManifest`，不会创建另一条分析链。新上传默认以 2 个文件并发、每个
文件内部顺序分块传输，每块都必须通过 SHA-256；小文件保留完整 SHA-256，大文件用
持久化有序分块哈希树封存，提交时不再从 NAS 全量重读。上传完成后、进入 GPU 队列前，
服务会校验设备角色、媒体可读性、分片顺序和 CSV 时钟覆盖，并在
`JSON-Config-Files/Input-Manifests/input_seal.json` 写入与 NAS 零复制入口一致的输入
封条。未提交的会话可用 `DELETE /api/upload-sessions/{session_id}` 取消并释放预留空间。

已完成档案不会依赖浏览器加载整份大 JSON 才能查找关键素材：`GET /api/key-events` 支持跨档案或指定档案的全文、动作类型、实验组、双视角和时间范围筛选，并通过与筛选条件绑定的 `cursor` 分页；`GET /api/key-events/{event_uid}` 返回事件及带 SHA-256 的素材引用；`GET /api/evidence/{evidence_uid}` 可一跳回到 `evidence_package.json` 的 JSON Pointer 和原视频物理分片；`GET /api/physical-changes` 查询明确观测到的对象前后状态变化，不会替 unknown 区间补状态。稳定事件 UID 格式为 `{archive_id}:{parent_event_uid}:{event_id}`；其中稳定实验组 UID 不会因前面插入其他实验而改变，原有 `GROUP-xxxx` 继续作为页面顺序编号。SQLite/JSONL 都是权威归档 JSON 的派生产物，可随时重建，不会取代原 JSON。

开发仓与稳定发布仓采用单向晋升，具体规则见 [双仓发布策略](docs/DUAL-REPOSITORY-RELEASE-POLICY.md)；旧 RealityLoopAI 仓库的全量只读审计与复用结论见 [旧仓库审计报告](docs/REALITYLOOP-LEGACY-REPOSITORY-AUDIT-20260817.md)。

已验证的容量基线是至少 **6 路 × 每路 3 小时**，它不是上传上限。上传路数、单路
时长和总字节数不设固定小上限；能否接收由当时真实 NAS 剩余空间和活动任务预留量
决定，不足时在传输前给出缺口。分析阶段以 600 秒为可恢复分块，最多 6 路并行
解码，但同一角色的帧合并成有界 GPU batch；检测结果逐行落盘，重启后跳过已完成
分块，因此内存/显存占用不随视频时长增长。**7 路 × 8 小时的真实 NAS 上传与完整
推理尚未实测，当前状态为 `NOT_PROVEN`；代码与确定性测试只证明其不会被固定路数、
时长或总容量阈值拦截。**

6 路输入不等于 6 路输出。系统逐路做“有效实验视角门控”：只有持续手—物体操作、容器/设备状态变化或物料转移等证据达到阈值，且边界内动作密度合格的视角才会生成实验 MP4。空镜、纯走动/穿戴、等待、长静止、无实际实验操作的视角会在筛选备注中记录拒绝原因，不进入后续关键素材提取。

## 性能策略

- 优先使用已存在的 `.engine`；否则使用 `.pt` + CUDA FP16。
- `prepare-engine` 以当前图像尺寸和 batch 上限导出 TensorRT engine。
- FFmpeg 优先 NVDEC 解码和 NVENC 裁切；不可用时自动回退 CPU 解码/编码。
- 六路解码生产者并发工作；第一/第三人称两套模型同时常驻 GPU，分别消费有界队列并做角色内 batch。显存超过预算时各扫描器自动减小 batch，避免 6GB 显存 OOM。
- 全量粗筛默认 `0.125 FPS / 512px`，任一路候选会按全局时间扩散到所有输入视角，再以 `8 FPS / 960px` 精扫、判定每路有效性并收紧边界；最终关键帧和裁切始终从原始视频精确取时间戳。

`run_metrics.json` 记录总墙钟耗时、各阶段起止/耗时，以及每个关键素材调用的输入 token、输出 token、总 token、缓存命中 token、延迟和重试次数；随后汇总关键素材阶段与整次运行。Token 只采用服务端 `usage`，缺失时保留 `null`，不做伪精确估算。

实验详情的 **性能与报告** 先显示各阶段墙钟耗时，再展示硬件与 Token。
扫描器的等待供帧、推理和跟踪落盘是并行工作累计时间，不能相加当作整次耗时；
供帧等待包含读取、解码与调度，不能直接视为纯解码耗时。新模型请求逐次保存起止时间、
失败耗时与实际重试退避，以计算实际请求并发。旧请求缺少相应字段时显示未记录。

**恢复与补全** 提供操作步骤整理、报告更新和完整复跑。恢复方案先核对当前任务版本、
保存的模型验证和已有材料，说明缓存校验、新增调用与历史产出的去向。操作整理使用
已审核事件与保存画面，不重新解码原视频；报告更新不调用模型。质量门未通过时仍只
输出阶段报告。完整复跑会把旧派生产出移入同一任务的 `Retry-Attempts` 历史目录，
原输入不复制；实际缓存命中必须以执行回执确认。阶段任务完成不表示原实验正式归档。

默认 `mllm.operation_review.enabled: true`，在事件审核后以有界请求整理具体操作。
每个步骤必须引用当前已审核事件，不遗漏事件、不跨较大证据空隙合并，功能性动作须由
该步骤所引用的事件支持。未通过整理校验时保留原操作记录和调用回执。该阶段会增加
模型调用，其缓存与 Token 单独记账；可通过上述配置关闭。停止任务也可通过
`POST /api/runs/{run_id}/refresh/operations?revision=<当前分组文件SHA-256>` 重新整理。
新版本保存为 `JSON-Config-Files/operation_review.json`，绑定原证据版本，Web 和阶段报告
使用同一修订；原始事件、视频、实验边界和质量状态保持原记录。

步骤整理区分当前操作、整段审阅上下文和后续判断。当前结果保存为 `observed_result`，
原始长描述保留在 `source_operation_records`；Web 与日报 JSON 均保留这两层记录。
当前步骤不能借用其他步骤已确认的功能性动作，也不能把下一步的事件编号加入当前依据。
“已观察后续”需引用本实验组中独立、靠后且审核通过的事件；无法定位的原后续判断仍保留，
显示为待核对。`time_scope` 说明事件区间用于定位证据，并非完整操作起止时间。
这些约束不证明操作无遗漏；完整步骤召回与起止定位仍需真实视频人工对照验收。

页面提供没有步骤记录的较长时间段供回看核对。这些区间可能包含等待、遮挡或未识别
操作，不能将其数量或时长当作步骤召回率。录像到尾、准备动作结束或步骤请求完成均不
证明整场实验结束；没有明确结束复核证据时保持待续或未确认。完整操作召回与跨台连续性
的真实质量仍须按原视频进行人工验收（`PARTIAL_EVIDENCE`）。

运行期间，Web“任务进度”页直接读取归档账本，而不是展示估算值：`source_progress.json` 给出每路解码后端、分块进度与状态，`resource_telemetry_live.json` 给出 GPU/NVDEC、CPU、内存、网络与进程 I/O，`run_metrics_live.json` 给出已执行/已复用模型调用及输入/输出 Token。尚未开始的任务由本地 SQLite 队列在服务重启后继续领取；执行中断的任务会用同一运行身份重新进入执行器，检测分块和已完成的豆包结果按持久账本续用，而不是从零开始。

任务页按“原视频留存 → 预检/对齐 → 有界实验发现 → 实验片段 → 关键素材 → 证据验收 → 日报/PDF”七个用户环节展示。完成状态以 `JSON-Config-Files/Stage-Receipts/*.json` 的原子阶段回执为准，并明确显示每步归档目录、阶段耗时、当前/下一步、逐视角进度和数据更新时间；遥测超过 20 秒没有更新时会显式提示状态可能延迟，而不会误判任务已经停止。

固定六路基准在归档前还会生成 `JSON-Config-Files/quality_acceptance.json`。评估基线只用于验收、不参与推理，分别检查五段实验的 Precision/Recall、起止边界误差、连续/独立关系、六类关键动作覆盖、关键帧/关键片段与模型结果完整性、跨视角支持或显式不确定性。任何正式固定归档缺少自动质量验收、日报/PDF或证据包验收，均拒绝覆盖 NAS 正式目录；目录提升后再做 SHA-256 清单核验。

已有档案可以离线复验，不解码视频，也不调用模型：

```powershell
visioncortex validate-archive-quality `
  --archive <实验档案目录> `
  --baseline configs/acceptance/six-view-three-hour-reviewed-baseline.json `
  --write
```

预处理 SLA 定义为视频探测、时间对齐、1 FPS 全量粗筛、8 FPS 候选窗精扫和边界审计，默认目标为 1200 秒（20 分钟）。`run_metrics.json.preprocessing_sla` 保存目标、实测值和是否达标；裁片编码与豆包调用不计入预处理时间。

系统无法凭空保证“精确”：液体本身不在 21 类标签中，因此液体移动由容器/移液器目标、ROI 光流和跨视角一致性共同提出候选，再交给多模态模型确认；低置信或仅单视角可见的事件会明确标为不确定，而不会伪装成确定结果。

## 实验录音、字幕与转写搜索

录音转写作为同一次分析的 `speech` 阶段运行，位于多视角对齐之后，与视频共用
任务队列和实验归档。用户继续选择 NAS 采集批次，或从浏览器上传视频。
NAS 输入仅根据所选视频的 `meta.json` / `recording_ready.json` 音频引用配对，
核对设备、采集会话、时间窗和文件回执；没有音频输入或采集不完整时明确记录原因。
浏览器上传支持自动读取视频内音轨，也可在每个视频分片下附带独立录音。
独立录音的开始偏移以秒填写：晚于视频为正、同时开始为 0、未知留空。
未知偏移仍可转写和播放，但不会生成与视频对齐的时间。

实验详情的 **录音与转写** 页支持试听、播放字幕、文字搜索和点击文字跳转。
试听文件是 16 kHz 单声道 AAC 派生音频，原始录音仍由源路径与 SHA-256 追溯；
SRT/VTT 使用各试听分块的时间，JSON 同时保留原始录音时间、对齐依据及未复核标记。
产物清单位于 `JSON-Config-Files/speech.json`，录音、文字、字幕和执行回执位于
`Key-Materials/Experiment-Audio/<机位与分片>/<分块>/`，跟随现有归档发布清单迁移。
`transcript.json` 对应原始 worker 回执，`aligned-transcript.json` 增加实验时间映射，
两者分别记录哈希。缺失模型、源文件变化或转写失败会使该分析停止并保留阶段产出。

事件理解、实验步骤理解与最终步骤整理读取同一时间窗内的转写，模型缓存绑定
转写哈希和实际输入上下文。NAS 时钟通过实际 RGB 帧时间映射到视频时间轴；
无已知偏移或处于视频时钟缺口的语句不进入模型理解。上下文默认最多 80 句、
8000 字，截断和未对齐数量保留在模型回执中。
录音说明保存在 `speech_interpretation`，引用保存为来源片段 ID；虚构引用或
越出步骤时间窗的引用会被拒绝。实验页面支持从理解结果或步骤引用跳转回听，
日报 JSON、Markdown、HTML 和 PDF 保留录音说明及引用，不新增报告模型调用。
若视觉分析未划定实验片段，理解阶段仍会用转写对应时刻的双视角抽样画面生成
实验级录音说明，保存在 `JSON-Config-Files/speech_understanding.json`，可在录音页
查看和回听；调用用量进入同一任务账本。该说明不会新增实验分组或动作事件，
也不会绕过无有效实验素材时的质量门。

默认 `speech_recognition.enabled: false`，本地配置不会为转写自动访问继承的 NAS。
管理员启用时须配置语音环境的 `python_executable`、本机 `model_directory` 和
`configs/models/speech-recognition.json` 中冻结的模型。3090 Ti 配置使用本机已验证的
隔离语音环境；其他节点需单独配置，不能直接套用该机器路径。可选安装依赖为
`.[speech]`，也可在隔离 Python 环境安装其中列出的依赖；需另行具备 FFmpeg/ffprobe。
执行时完全离线、CPU INT8 推理，不自动下载模型或上传录音。长录音按最多 30 分钟
有界解码、逐语音区间转写，支持校验后的分段恢复；预算由 `max_audio_seconds` 控制。

证据边界：确定性检查证明输入、字幕和归档契约；真实 ASR 调用需要执行回执。
设备共用时钟仅提供配对与估计对齐依据，实际音画同步、实验术语准确率和设备播报
区分仍需人工样本验收（`PARTIAL_EVIDENCE` / `NOT_PROVEN`）。转写文字不参与确认
物理动作，也不作为人工标注或真实实验准确率证明。历史实验没有转写产物时显示
“尚无录音转写产出”，不会在浏览时自动补跑模型。

录音优化与联动：实验详情分别显示录音转写、模型理解、视觉证据和正式报告状态。
理解请求采用 `visioncortex-speech-wire/2`，用原文词典和短引用减少重复传输；逐句时间、
原文与完整来源引用仍保留在归档中，不做自动纠错或删句。视觉缓存排除纯理解、转写和
报告模块及配置；共享编排、schema、视觉算法、模型和视频输入变化仍保守失效。
ASR 在配置的缓存根目录使用独立 `speech-v1` 缓存，复用前验证源、模型、运行库与所有
产物；原始执行回执不变，本轮是否解码/调用模型记录在 `speech_execution.json`。

停止且未正式发布的任务支持 `POST /api/runs/{run_id}/refresh/understanding` 和
`.../refresh/reports`，通过原持久队列执行。录音理解刷新复用已保存且通过哈希检查的
抽样画面，仅适用于没有划定实验片段的录音说明；含实验片段的完整理解仍通过“复跑并补全”
完成，并校验复用视觉、ASR 和语义缓存。报告刷新不调用模型，质量门未通过时只更新阶段报告。

新分析在录音阶段保存 `speech_timeline.json`，旧暂存实验可用
`POST /api/runs/{run_id}/refresh/timeline` 补建。录音页可按字幕、理解引用、步骤或
统一时间条跳转并切换视角。时间轴从本实验封存输入生成最长边受限、15 fps 的浏览器
预览（`Video-Previews`），保留原时长和源身份回执；不更改或复制原视频作为便捷来源。
首次生成预览需要解码视频，后续时间轴刷新校验复用；理解和报告刷新均不解码源视频。
未知音频偏移、倒退时钟和超过两秒的录音时钟缺口不会被插值成可靠同步；采集时间映射
属于 `PARTIAL_EVIDENCE`，实际音画误差仍需复核，不能作为动作已发生的证明。

## 录音质量、检索与片段重算

NAS 采集批次和浏览器上传共用采集预检：`capture_quality.enabled` 默认开启，默认每段视频检查 4 帧、每路录音检查 4 个不超过 3 秒的短窗，最多检查 24 个来源。任务页和录音页显示偏暗、缺音轨、静音、削波及实际抽样覆盖率；未抽样部分不计为已验证。`JSON-Config-Files/capture_quality.json` 记录样本指标，录音理解同时接收这些局限。指标不代替人工转写或真实实验准确率验收。

录音页默认折叠同一路录音的重复文字，展开后保留每次出现的原文、时间和回听位置。声源提示仅依据播放记录邻近或操作术语，不能确认说话人。字符倒排索引 `speech_search.json` 由转写生成，并由独立回执绑定完整性；缓存命中仍校验原始转写。可关闭术语扩展以按文字查询；跨实验检索每页检查 10 个实验，每个实验最多展示 100 条，结果属于“录音提及”，不确认视觉动作。

未正式发布且已停止的任务可在录音页选择单个理解片段重算：有实验分组时使用保留的分组画面，无分组时使用已有录音理解窗口。只刷新所选录音解释及与既有步骤重叠的录音引用；视觉动作、时间边界和原质量门保持不变。旧理解和报告索引保存在 `JSON-Config-Files/Stage-Refreshes`，报告在理解变化后失效并重新生成；质量门未通过时仅更新阶段报告。旧实验若没有保留分组理解画面，须完整重试一次生成依赖，不能凭空补出证据。

“人工基线评测”可下载引用模板并提交人工校对 JSON，要求匹配转写 SHA-256、显式人工校对声明和校对人。返回所提交样本的字符错误率及覆盖条数，不修改原转写；没有基线时准确率为 `NOT_PROVEN`。此功能不包含转写校正或音画偏移编辑。

## 实验结果阅读与阶段报告

实验详情按使用任务组织：**视频与操作步骤** 保留同步双视角、第一人称、第三人称三个播放器；
**日报与报告** 集中展示日报、PDF 和可追溯 JSON；**分析记录** 集中展示性能、Token、
结果检查和素材筛选记录。实验页只显示简短状态，片段分析依据从独立对话框查看。
“处理已结束”只描述任务状态；实验结束、步骤覆盖和完整质量验收分别记录，不互相替代。

未通过完整质量门的暂存结果也会生成带品牌 Logo、封面、已保存证据图及步骤说明的阶段 PDF，
与正式报告共用品牌封面。阶段产出为 `Partial-Results/Stage-Evidence-Report.pdf`、
`Stage-Lab-Daily-Report.html` 和 `Analysis-Result.json`；原审计 HTML 仍保留。
这些文件明确标注 `PARTIAL_EVIDENCE`，不会创建正式日报或发布清单，也不会改写原质量结论。
报告刷新复用已有控制记录及有界的派生关键帧，不调用模型、不解码原视频。
JSON 记录结果版本、步骤事件引用、输入清单路径、控制文件及代表图片 SHA-256；
不把这种校验描述为重新读取并核验了全部原视频。网页仅提供与报告回执哈希匹配的阶段 PDF/HTML。

全局素材库和日报库包含可查看的暂存结果，保留阶段/候选身份。
暂存库摘要使用 `GET /api/staging-runs/{run_id}/archive?section=library-materials`
或 `section=library-reports`，不为列表加载完整证据包、硬件采样账本或执行步骤一致性重检。
完整证据仍通过实验详情和逐事件接口获取。列表优化不改变质量门、素材数量或原始文件。

实验边界新复核使用 `visioncortex-experiment-boundary-review/3`：合并需要跨候选边界的
操作对象、前后状态及对应帧引用；跨实验台还需有效的移动/交接画面。同一路第一人称本身
不能证明操作者身份或同一实验。一个输入清单内的物理分片按实际对齐时间审查接缝两侧，
不把 15 分钟切片当作实验结束。跨任务自动拼接尚未实现；缺少后续录像时保留“实验待续”。
这些实现与确定性测试不等于新实验的真实质量验收；实际模型调用和连续性准确率需独立回执。
