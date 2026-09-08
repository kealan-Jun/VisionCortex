# VisionCortex 工作交接 — 2026-09-07

## 1. 接手结论与用户目标

这是一份本机继续开发/实测的交接，不是生产验收通过证明。交接时只核对了本地 Git、服务状态、GPU 进程和已有回执；没有启动模型、调用豆包、扫描 NAS、重跑测试或发布版本。

用户最终要求：充分发挥本机 RTX 3090 Ti 的有效硬件性能，模拟普通用户真实端到端流程，从网页上传视频或选择 NAS 采集批次开始，自动完成分析、证据生成、关键素材、报告、可溯源归档及网页查阅。无需人工审批；个别素材失败要自动隔离，不能使整批无条件崩溃，也不能把失败或不确定素材当成已证实动作。

必须清楚报告：输入几路、每路多长、合计视频时长、共同实验时间轴、冷/热缓存、预处理时间、完整流水线时间、用户提交至正式归档可查阅时间、GPU/CPU/解码/编码/磁盘指标、真实模型调用及服务端 Token。不能把预处理 20 分钟说成全链路 20 分钟，也不能用短视频线性换算声称六路长视频已达标。

**最终目标仍为 `NOT_PROVEN`。最近实际豆包请求返回欠费错误；当前合并版本还没有完成真实网页提交至正式归档的全链路验收。**

## 2. Git 状态：保住本机未推送工作

- 工作目录：`/home/x1/Projects/VisionCortex`。
- 当前分支：`codex/rtx3050-device-delivery-20260904`。
- 当前 HEAD：`7db0dcd052891e0e189c5e931a27558256fea032`。
- 合并前本机父提交：`7c5f3030a95a74912dd11c2eb405fc9c66b56415`。
- 已合入 4060 分支：`codex/input-alignment-identity`，SHA `b2a3c7df47a3d66e2554fbb6be33c9826584ed95`。
- 最新这一轮远端增量实际是两个提交：`6a7f823`（发布/证据包加固）和 `b2a3c7d`（归档检索/交付加固），不是十个新增提交。“十阶段”是用户对累计改动的描述。
- 开发远端：`development = https://github.com/kealan-Jun/VisionCortex.git`。
- 稳定远端：`origin = https://github.com/RealityLoopAI/VisionCortex.git`。**不要因为它叫 origin 就向它提交开发改动。**
- 本地跟踪的 upstream：`development/codex/rtx3050-device-delivery-20260904`，SHA `4a33a253b338dda5e71ec9f840c9d799680029bf`；本地领先 30 个提交，尚未推送本次合并。
- 本地缓存的 `development/main`：`f8801cbfc2486fb73f78d66948b1adb50129c975`。9 月 5 日核对的远端默认分支是 main；9 月 7 日未 fetch/ls-remote，不能把缓存当作最新远端状态。本地没有 `development/HEAD` 符号引用。

用户未提交工作必须保留：

- `README.md` 原有新增 14 行。
- `docs/VisionCortex-交付验收单模板.md`。
- `docs/VisionCortex-历史真实六路验收示例.md`。
- `docs/VisionCortex-普通用户10分钟操作指南.md`。
- `docs/VisionCortex-端到端视频分析交付手册.md`。
- `docs/assets/`。

临时 README 保护 stash 仍在：`c746e34648e2d04d404677f20cf2991d22964b1f`，9 月 7 日为 `stash@{0}`。已成功 apply，恢复后的 patch-id 与 stash 一致；**不要再次 apply**。本次交接不清理它。不要 reset/checkout 覆盖用户文件，也不要 `git add .` 或 `git add -A`。

## 3. 已完成的实现与验证边界

### 合并后的功能

- 上游：证据包 fail-closed 发布、代码/配置/模型/产物哈希溯源、跨文件契约、原子正式版本指针及恢复；自动验收和日报 V2；本地可重建 SQLite 检索目录、分页、中文检索、release_id 绑定媒体 URL、传输并发限制；可选多人角色和访问审计。
- 保留本机：3090 Ti 性能参数、解码器复用、关键素材编码重叠、Ark 失败熔断和事件隔离、未完成任务的初步产物展示、NAS 相机发现和时间匹配、桌面自启与大屏/窄屏适配。
- 合并补齐：分页摘要默认值、真实总数、报告加载、档案状态更新、release 缓存失效、素材/报告库独立分页状态、缺失焦点事件不误跳到其他素材、视频不预加载正文。
- 合并相对于本机前一 HEAD，没有改动 `configs/rtx3090ti-ubuntu-production.yaml`、`src/visioncortex/archive.py`、`src/visioncortex/mllm.py` 的已有本机优化。

### 9 月 5 日验证记录（不是 9 月 7 日重跑）

- `PROVEN`：本地确定性检查 775 项收集，771 passed、4 skipped；`node --check src/visioncortex/web/app.js`、`ruff check src tests`、`.venv/bin/python -m compileall -q src tests`、`git diff --check` 均通过。
- 相关新增回归：`tests/test_web_archive_pagination.py`、`tests/test_evidence_indexing.py`。
- `PARTIAL_EVIDENCE`：浏览器实际验证了档案、六个素材、筛选、报告入口及 3840×2160 / 390×844 视口无横向溢出，没有捕获到 JavaScript 错误。使用的是 `VC-LOCAL-SIX-VIEW-ACCEPTANCE` **合成档案**，只证明界面/结构，不是真实实验质量；不声称所有浏览器全部适配完成。
- `NOT_PROVEN`：合并 SHA 的真实模型全链路、真实动作准确率、真实 NAS 正式发布、六路长视频性能、跨平台 CI 与稳定发布就绪。

旧合成档案动作键仍可出现 `liquid_transfer`，与新本体 `liquid_movement` 有历史别名差异。不要把选中旧键成功说成新键完全兼容；旧档案也不是生产准确性基准。

## 4. 本机运行与凭据边界

- GPU：RTX 3090 Ti，`nvidia-smi` 报告显存总量 24564 MiB。9 月 7 日交接快照没有 CUDA 计算进程；桌面仍使用部分 GPU。这不代表整个产品目标完成，也不授权启动其他训练。
- 生产解释器：`/srv/sentinel-data/VisionCortex3090Ti/.venv/bin/python`。
- 9 月 5 日实测依赖：Python 3.12.13、Torch 2.6.0、Ultralytics 8.4.28、TensorRT cu12 10.16.1.11、httpx 0.28.1。TensorRT 发行包名为 `tensorrt-cu12` / `tensorrt-cu12-bindings`，不能因查不到名为 `tensorrt` 的 distribution metadata 就误判未安装。
- 生产配置：`configs/rtx3090ti-ubuntu-production.yaml`；本地无 NAS 配置：`configs/rtx3090ti-ubuntu-local.yaml`。
- 模型/引擎：`/srv/sentinel-data/VisionCortex3090Ti/Models/ClosedSetYOLO/` 与 `/srv/sentinel-data/VisionCortex3090Ti/Engines/`。真实运行前重新核对身份/哈希/运行环境，不能只看文件存在。
- `visioncortex-local.service`：9 月 7 日 active；既有入口 `http://127.0.0.1:8000/#/home`，独立无 NAS 合成验收/备用服务，MLLM 关闭。
- `visioncortex-analysis.service`：9 月 7 日 active；既有入口 `http://127.0.0.1:8001/#/home`，NAS 生产配置及监控服务，MLLM 开启。9 月 5 日两项服务均已重启加载合并版本，9 月 7 日没有重启。
- 9 月 5 日生产队列无 queued/running；9 月 7 日没有重新读取队列。接手先检查队列及 GPU 所有者，不要在工作中重启/抢占。
- NAS 正式归档配置根：`/home/x1/桌面/nas/VisionCortexExperimentArchive`；正式 staging 是其中 `.VisionCortex-Run-Staging`；持久缓存：`/home/x1/桌面/nas/VisionCortexExperimentCache`。
- 本地 Runtime：`/srv/sentinel-data/VisionCortex3090Ti/Runtime`，NoNasWeb 必须保持所有存储根不继承 NAS。

Ark 凭据只经 `src/visioncortex/credentials.py::ensure_ark_api_key` 从安全环境/批准的本机文件加载。文件路径 `/home/x1/.config/VisionCortex/ark_api_key`，9 月 5 日验证 owner、普通文件、0600 和格式；不要打印文件、进程环境或请求 Authorization，不要把密钥写入回执/文档/测试。

**最近一次实际 Ark 检查：2026-09-05，生产解释器和生产模型配置，纯文本连通性请求，无实验图像；HTTP 403，错误码 `AccountOverdueError`，耗时 0.169 s，usage 为 null。9 月 7 日未重试，不能宣称账户现在仍欠费或已经恢复。**

接手先做一次安全、有界的账户连通性预检。若仍欠费，明确要求用户恢复方舟账户，不要购买/充值，不要反复启动注定不能完成语义证据的长任务。可独立推进不依赖 Ark 的本地瓶颈检查，但必须标为局部性能证据。

## 5. 输入、拷贝与历史真实视频回执

本地运行证据根：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/RealVideoValidation-20260904`。以下均为已有回执；交接只读取本地文件，没有复查远端媒体。

### U 盘数据已留存的证据

- 原入口 `/media/x1/USB/video`；本地副本 `/srv/sentinel-data/VisionCortex3090Ti/Runtime/USB-Imports/2026-09-04/video`。
- `nas-index/usb-nas-copy-receipt.json` 记录：149 个文件、11342242051 字节，源/目标数量及大小相同；递归 checksum dry-run 差异为空，`status=verified`、`usb_safe_to_remove=true`。这是 9 月 4 日复制完成的历史证据，不是本次又复制了一遍。
- NAS 已命名位置记录为 `/mnt/realityloop-nas/VisionCortexValidationSources/2026-09-04_USB_Import/`，三组：`01-Raw-Four-Camera-RealityLoop-20260616`、`02-Raw-CustomFlow-ACEBDF-0002`、`90-Reference-Existing-Derived-Experiments-20260803`。
- 第 90 组是既有衍生产物参考，不能当作原视频或独立真值。
- 旧路径与生产配置路径可能来自既有不同挂载/解析入口。接手用现有配置和 resolver 检查，不要自行改写冻结路径或为省事复制原片。

### NAS 索引批量验证

- 本地汇总：`nas-index/nas-validation-consolidated.json`。
- 历史源索引：`/mnt/realityloop-nas/experiment_videos_1h/index.csv`；派生索引在本地 `nas-index/experiment_videos_1h-derived-index.csv`。
- 回执的 45 个实验记录：30 个清单预检可运行，15 个缺少准确配对来源；运行后 8 个流程完成，22 个输入/时间对齐门禁失败，合计 37 个未完成。
- 这些运行使用 `2cc70b8` / `cb01d47`，不是当前合并 SHA；批量验证关闭了 Ark，不能称为 45 个真实端到端通过。
- 回执 `timeline_seconds=4676.565` 是该索引汇总口径，不是“六路每路 1 小时”。需查看各清单实际时长后再报告。
- 典型失败：CSV 不足两行、第一/第三人称没有相同时段覆盖。不能拿其他时间相机视频凑齐视角绕过门禁。

### NAS 实时相机监控

生产配置按顶层 `*_cam*` 自动发现目录，30 s 轮询、120 s 稳定等待、每相机近期 32 个 recording，有界元数据扫描。按 recorder 的 session 身份及跨相机时钟重叠配批次，不是目录中任取最近视频。

9 月 5 日健康信息显示 9 个相机目录、monitor=watching、连续失败 0；其中 `lubancat-52d2ef0c_cam01`、`orangepi5pro-fe0f7222_cam01` 缺角色配置，必须显式阻塞，不猜第一/第三人称。9 月 7 日没有重扫相机目录；不要把 9 当永久固定数量。仍需模拟符合采集协议的写入，验证稳定后自动出现批次与正确时间配对。

## 6. 性能历史口径：不要误报

可核对的四路真实输入清单：`input-manifests/usb-customflow-acebdf-0002.yaml`，实验 `usb_customflow_acebdf_0002`，1 路第一人称 + 3 路第三人称，4.728 GB 原视频。

`runs-opt-reader-1bd7f0e-cold/usb_customflow_acebdf_0002/JSON-Config-Files/input_volume_report.json` 记录各路 899.234、899.965、899.773、899.953 s；合计 3598.925 s，约 59 分 59 秒视频量，但共同实验时间轴仅约 15 分钟。

同目录 `run_metrics.json`：

- 流水线内部总时长 **553.576401 s（约 9 分 14 秒）**。
- 已记录的预处理口径 **184.500462 s**。
- 精扫 163.041479 s，实验裁片 86.079566 s，关键素材 247.815280 s。
- MLLM 阶段虽标 completed，但调用数为 0、Token 为 null，**不证明实际完成豆包推理**。
- 历史对话曾报约 557.15 s；交接核对到的权威内部指标是 553.576401 s，不能混用未核对的外层耗时，更不能把它说成用户上传至正式归档全耗时。
- 这是 `1bd7f0e` 历史局部证据，不是当前 `7db0dcd` 真实全链路成绩；目录名 cold 也不能替代对缓存回执的检查。

用户记忆中的“六路每路 3.3 小时，20 分钟”在本任务尚未复现。必须找回准确原始清单和对应回执，核对 3.3 h 是每路还是合计、20 min 是预处理还是全链路；找不到就明确缺失，不能拿重复拼接短片替代。

保留现有实测调参依据：

- 引擎动态 profile 最大 batch 4，运行请求 batch 16 会被拆分；此前动态最大 batch 16 的构建尝试 OOM。盲目增大 batch 不等于有效提高吞吐，不要直接重建替换生产引擎。
- GPU resize 曾改变检测输入分布并使边界偏移超标，因此生产仍 CPU scale；不能为利用率数字降低质量。
- 实验组 aligned 编码重叠 A/B 无收益，当前 `overlap_aligned_experiment_clips=false`；关键素材 aligned 重叠保留。
- 下一步以真实阶段遥测确定 CPU resize、解码、细扫、帧读取、NVENC/NAS I/O、模型排队等瓶颈；固定输入和质量约束做有界 A/B，记录资源利用率和实际耗时，不能用单张 nvidia-smi 截图证明“已打满”。

## 7. 新对话推进顺序

1. 先读仓库 `AGENTS.md`、`README.md`、`docs/DUAL-REPOSITORY-RELEASE-POLICY.md` 及本交接。检查本机分支、工作树、队列和 GPU 所有者；不要回到 main 丢掉未推送合并，不要无故新建从旧默认分支开始的工作树。
2. 安全验证 Ark 可用性；核对 SHA、真实解释器、配置、模型/引擎哈希、输入清单、存储根、容量和独立输出目的地。真实运行与回执绑定这些身份。
3. 用现有原片走真实网页上传入口及 NAS 批次入口，验证续传、输入封条、排队、进度、自动处理、单事件隔离、报告、原子归档、哈希复核及网页检索/播放/下载；不要只跑 CLI 然后称为“模拟用户全流程”。浏览器工作要读可用 browser 技能。
4. 输入配对、时间对齐、自动证据门禁不可绕过。单事件可隔离，整批缺合法跨视角时间覆盖则必须报告输入失败。自动可信产出与模型准确率是否有独立真值测量是两个结论；不要恢复人工审批兜底，也不要删除溯源/质量门禁。
5. 在可信输入和输出约束下做 3090 Ti 性能 A/B；冷运行/复用运行分别计量，不挪用缓存成绩。先四路已知输入，再真实更长/更多路输入；不能外推六路长视频达标。
6. 补齐 NAS 不同时间窗口不混批、所有已注册相机发现、写入稳定成批、服务/任务恢复等场景。发现源文件或角色缺失，只修复有依据的配置，需用户确认的采集身份不要猜。
7. 结束时逐项给出 `PROVEN / PARTIAL_EVIDENCE / NOT_PROVEN`，附真实路径、代码 SHA、模型调用/Token、路数/时长/耗时、失败/隔离数量及缺失门禁。稳定发布按双仓策略单独执行，当前没有晋升授权或通过证据。

历史 GPU 协调：另一训练任务 `01a064e9-fe5b-7d73-bd02-9b277c35ee06` 曾明确暂停 GPU 训练、只做 CPU 数据治理，要求本任务全部完成并释放 GPU 后告知。当前只是交接且目标未完成，不要发“全部完成可启动训练”；真正准备使用/释放 GPU 时先重新核对协调状态。

## 8. 本次交接交付范围

仅新增本文件，保留所有既有修改，没有 Git 提交/推送，没有新建任务，没有运行或终止训练。用户可以在本项目新对话中发送：

> 请读取 `/home/x1/Projects/VisionCortex/docs/WORK-HANDOFF-20260907.md` 并接手当前工作。保留本机分支与未提交修改，先检查 Ark 和运行环境，再继续 3090 Ti 性能优化及真实网页/NAS 端到端自动可信证据与溯源归档验收。不要把历史测试、短视频或合成档案当作当前全链路达标证据。


## 9. 持续目标续接入口（2026-09-08，goal25）

以上为原始交接记录，后续状态以[工作进度](WORK-PROGRESS-20260907.md)、[模型迭代记录](MODEL-ITERATION-RESULTS-20260907.md)及运行根`Project-Detector-Pilot-20260907/goal25-work-receipt.json`为准。用户已授权ChatGPT逐件初标、第二遍复核、分人称训练、增强和持续迭代，并已设置active goal；当前尚未达标，不能重复新建goal或声称全部完成。方舟未恢复期间继续独立工作，不反复发起欠费预检或覆盖已完成产出。

最新已完整训练47轮（FP23/TP24），旧中断另列。v24的1280训练仅改善部分小枪头，完整整图/连续视频门禁未过，生产权重及当前分支/未提交修改保持。数据212图3079框，86完整、45仅区域外、24待复核、6保留测试草稿、51排除；F035已续接到版本5但仍holdout。优先读取最新冻结回执和状态，继续未完成标注、训练来源覆盖及真实系统验证；不得将旧候选浏览器播放或历史性能当作新模型验收。


### goal26 接续更新

用户将后续提供第三人称原始数据集，当前不得把整理视频抽帧称为已收到该原始包。F082/F083实际双遍区域外复核至版本6，保留原val；当前212图/3079框、86完整、47区域外、22待复核、6测试草稿、51排除。最新源快照在工作台`review-goal26/snapshot-after.json`，统计`counts-goal26.json`，本阶段无新训练，47完整训练保持。公开来源只完成元数据初筛，待实图、来源及全类别检查后才能采用。接续以运行根`goal26-work-receipt.json`及`goal26-public-source-research.json`为本阶段证据，保留全部未知和保留测试分组；继续独立工作，Goal active/PROGRESS，生产不晋升。


### goal27 接续更新：第三人称原始包已经收到

用户本次提供/home/x1/桌面/third_images.rar，实际169原图、无标签，已解包校验并导入现有工作台T000–T168。原始源根在RealVideoValidation-20260907/Motion-Delivery-20260907/Original-Datasets/Third-Images，包含导入/图片清单/摘要与实际初看回执。不要再让用户提供同一包，也不要将导入等同标注完成。T024双遍32已知/3未知、版本2、third_person/train，仅区域外完成；另168新图仍draft/unknown/unassigned。优先续接其他清晰的新TP来源与密集实例，核对同源跨视角分组，不按帧随机切分。

最新381图/3106框，87完整、51区域外、18待复核、174草稿、51排除，370源图/11裁剪。旧F086/F087/F117/F118已续至5/4/5/4，F117完整，其余未知保留。普通导出643阻断。读取工作台review-goal27/snapshot-final.json、counts-goal27.json及运行根goal27-work-receipt.json；47完整训练不变，本阶段无新训练/模型调用或生产替换。当前架构建议继续分人称检测、关联后互补证据；统一模型待固定对照，不得称已有最优证明。Goal active/PROGRESS，保留分支HEAD和全部并行未提交修改，继续局部工作及后续完整质量门禁。
