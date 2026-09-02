# LabVision Evidence

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

> 冻结基线：六路、多视角、3 小时湿实验视频的时间对齐、有界实验筛选、五类关键素材和细粒度步骤理解流水线。RTX 4060 部署、固定 NAS 基准、缓存目录与开发协作方式见 [RTX4060-交接与运行说明.md](RTX4060-交接与运行说明.md)。

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
GPU 任务，其余任务保留为“排队等待”，避免互相争抢显存。

该服务使用 3090 Ti 正式生产配置，启动前检查 GPU、NAS、固定 Python 环境和本地
空间；检查不通过时拒绝启动。它只接受回环地址和常见私有局域网地址，并要求每次
浏览器会话登录，不是公网发布方案。服务状态检查：

```bash
./deployment/rtx3090ti-ubuntu/08-Server-Status.sh
```

4060 和 3060 可以继续作为开发或备用机器，但普通用户不再需要在这些机器上拉取
仓库。真实模型、真实视频质量和正式归档能力仍须在 3090 Ti 主机按下面的生产门禁
验收，网页能打开本身不代表完整推理已经通过。

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

### NAS 不可用时的本地验收

以下入口全部只使用 `/srv/sentinel-data/VisionCortex3090Ti/Runtime`，不会创建、
轮询或写入 NAS 路径：

```bash
# 六视角媒体、六类关键素材、步骤理解、日报、PDF、JSON/JSONL/SQLite
labvision run-local-acceptance \
  --output /srv/sentinel-data/VisionCortex3090Ti/Runtime/LocalAcceptance \
  --config configs/rtx3090ti-ubuntu-local.yaml

# 六路 CUDA 解码 + 双 TensorRT 角色引擎硬件压测
labvision benchmark-local-hardware --output <local-output> \
  --media <h264-1> --media <h264-2> --media <h264-3> \
  --media <h264-4> --media <h264-5> --media <h264-6>

# 真实执行全部本地生产 CV 模型；使用公开人工标注样本，不访问 NAS/豆包
labvision accept-local-models \
  --dataset /srv/sentinel-data/VisionCortex3090Ti/Runtime/PublicDatasets/LabPicsChemistry/extracted \
  --output /srv/sentinel-data/VisionCortex3090Ti/Runtime/Model-Quality/<new-run> \
  --config configs/rtx3090ti-ubuntu-production.yaml

# 只读统计正式认证仍缺多少真值；不扫描生产归档
labvision model-certification-readiness \
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
ZIP，再拒绝路径穿越/软链接并有界解包。模型共识只能通过
`build-consensus-labels` 生成 `pseudo_labels_not_ground_truth`，不能冒充人工真值。

双闭集源权重可用 `validate-closed-set-models` 对注册哈希和 21 类本体做双重
校验。框真值完成后，`evaluate-yolo-boxes` 计算独立 P/R/AP；
`build-yolo-training-dataset` 用软链接或硬链接构建零拷贝训练集；
`train-yolo-model` 只接受 reviewed ground truth，并把新权重标记为“未认证候选”。
公开模型、模型共识或训练日志都不能解除生产认证门禁。

固定基准不会重复创建实验目录：先启动 Web，再运行 `deployment\rtx4060\04-重跑固定六路基准.ps1`，由常驻 Web 服务异步执行并复用 `Y:\VisionCortexExperimentArchive\CustomFlow_standard_correct_12_ABCFA_0001--exp_20260810_144014_e918b762`。不要在有超时限制的命令包装器里前台运行 `run-fixed-benchmark`。其他用户上传任务仍按实际上传路数动态创建独立档案。

面向化学湿实验长视频的多视角证据流水线。系统把第一人称与第三人称视频先对齐，再以两套 21 类 YOLO 模型和 ByteTrack 生成候选，经过跨视角审计后，提取实验片段、五类物理动作关键帧/关键片段/时间戳，并调用豆包多模态模型生成步骤级理解。

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
    evidence_package.json
    physical_change_log.json
    evidence_package_eval.json
    run_metrics.json
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
    Human-Review.json
  Professional-PDFs/
    Lab-Daily-Report-<date>.pdf
```

日报采用固定的 `VC-LAB-DAILY-REPORT-V1`：模型只产出结构化实验理解，确定性渲染器填充固定栏目，日报阶段不新增模型调用或 Token。模板栏目、版本策略和本地开发方式见 [实验室日报固定模板 V1](docs/daily-report-template-v1.md)。

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

时间戳为 ISO-8601 时也可解析。若 CSV 没有共同时间列，系统以视频起点为粗对齐，并用跨视角运动变化序列进行视觉锚点互相关校准；如果已有共同时间，先做最近邻鲁棒仿射拟合，再用视觉锚点修正残余偏移。

## 运行

```powershell
labvision validate-models --config .\configs\default.yaml
labvision prepare-engine --config .\configs\default.yaml
labvision run --manifest .\examples\manifest.example.yaml --config .\configs\default.yaml
```

没有真实视频时可完整验证目录、JSON 契约和评估器：

```powershell
labvision dry-run --output .\outputs\dry-run
pytest -q
```

上传服务：

```powershell
labvision serve --host 127.0.0.1 --port 8000
```

`POST /api/runs` 以 `videos[] + view_specs_json` 接收任意多路视频及 CSV，后台运行后从 `GET /api/runs/{run_id}` 查询状态。API 只绑定本机，除非显式改为 `0.0.0.0`。

已完成档案不会依赖浏览器加载整份大 JSON 才能查找关键素材：`GET /api/key-events` 支持跨档案或指定档案的全文、动作类型、实验组、双视角和时间范围筛选，并通过与筛选条件绑定的 `cursor` 分页；`GET /api/key-events/{event_uid}` 返回事件及带 SHA-256 的素材引用；`GET /api/evidence/{evidence_uid}` 可一跳回到 `evidence_package.json` 的 JSON Pointer 和原视频物理分片；`GET /api/physical-changes` 查询明确观测到的对象前后状态变化，不会替 unknown 区间补状态。稳定事件 UID 格式为 `{archive_id}:{parent_event_id}:{event_id}`。SQLite/JSONL 都是权威归档 JSON 的派生产物，可随时重建，不会取代原 JSON。

开发仓与稳定发布仓采用单向晋升，具体规则见 [双仓发布策略](docs/DUAL-REPOSITORY-RELEASE-POLICY.md)；旧 RealityLoopAI 仓库的全量只读审计与复用结论见 [旧仓库审计报告](docs/REALITYLOOP-LEGACY-REPOSITORY-AUDIT-20260817.md)。

容量目标按至少 **6 路 × 每路 3 小时** 设计：600 秒一个可恢复分块，最多 6 路并行解码，但同一角色的帧合并成有界 GPU batch；检测结果逐行落盘，重启后跳过已完成分块，因此内存/显存占用不随视频时长增长。磁盘预检会在开始前估算原片、临时候选和交付片段所需空间，不足时拒绝启动并给出缺口。

6 路输入不等于 6 路输出。系统逐路做“有效实验视角门控”：只有持续手—物体操作、容器/设备状态变化或物料转移等证据达到阈值，且边界内动作密度合格的视角才会生成实验 MP4。空镜、纯走动/穿戴、等待、长静止、无实际实验操作的视角会在筛选备注中记录拒绝原因，不进入后续关键素材提取。

## 性能策略

- 优先使用已存在的 `.engine`；否则使用 `.pt` + CUDA FP16。
- `prepare-engine` 以当前图像尺寸和 batch 上限导出 TensorRT engine。
- FFmpeg 优先 NVDEC 解码和 NVENC 裁切；不可用时自动回退 CPU 解码/编码。
- 六路解码生产者并发工作；第一/第三人称两套模型同时常驻 GPU，分别消费有界队列并做角色内 batch。显存超过预算时各扫描器自动减小 batch，避免 6GB 显存 OOM。
- 全量粗筛默认 `0.125 FPS / 512px`，任一路候选会按全局时间扩散到所有输入视角，再以 `8 FPS / 960px` 精扫、判定每路有效性并收紧边界；最终关键帧和裁切始终从原始视频精确取时间戳。

`run_metrics.json` 记录总墙钟耗时、各阶段起止/耗时，以及每个关键素材调用的输入 token、输出 token、总 token、缓存命中 token、延迟和重试次数；随后汇总关键素材阶段与整次运行。Token 只采用服务端 `usage`，缺失时保留 `null`，不做伪精确估算。

运行期间，Web“任务进度”页直接读取归档账本，而不是展示估算值：`source_progress.json` 给出每路解码后端、分块进度与状态，`resource_telemetry_live.json` 给出 GPU/NVDEC、CPU、内存、网络与进程 I/O，`run_metrics_live.json` 给出已执行/已复用模型调用及输入/输出 Token。服务重启后，未完成任务会标记为 `interrupted`，检测分块和已完成的豆包结果按同一运行身份续用。

任务页按“原视频留存 → 预检/对齐 → 有界实验发现 → 实验片段 → 关键素材 → 证据验收 → 日报/PDF”七个用户环节展示。完成状态以 `JSON-Config-Files/Stage-Receipts/*.json` 的原子阶段回执为准，并明确显示每步归档目录、阶段耗时、当前/下一步、逐视角进度和数据更新时间；遥测超过 20 秒没有更新时会显式提示状态可能延迟，而不会误判任务已经停止。

固定六路基准在归档前还会生成 `JSON-Config-Files/quality_acceptance.json`。评估基线只用于验收、不参与推理，分别检查五段实验的 Precision/Recall、起止边界误差、连续/独立关系、五类关键动作覆盖、关键帧/关键片段与模型结果完整性、跨视角支持或显式不确定性。任何正式固定归档缺少自动质量验收、日报/PDF或证据包验收，均拒绝覆盖 NAS 正式目录；目录提升后再做 SHA-256 清单核验。

已有档案可以离线复验，不解码视频，也不调用模型：

```powershell
labvision validate-archive-quality `
  --archive <实验档案目录> `
  --baseline configs/acceptance/six-view-three-hour-reviewed-baseline.json `
  --write
```

预处理 SLA 定义为视频探测、时间对齐、1 FPS 全量粗筛、8 FPS 候选窗精扫和边界审计，默认目标为 1200 秒（20 分钟）。`run_metrics.json.preprocessing_sla` 保存目标、实测值和是否达标；裁片编码与豆包调用不计入预处理时间。

系统无法凭空保证“精确”：液体本身不在 21 类标签中，因此液体移动由容器/移液器目标、ROI 光流和跨视角一致性共同提出候选，再交给多模态模型确认；低置信或仅单视角可见的事件会明确标为不确定，而不会伪装成确定结果。
