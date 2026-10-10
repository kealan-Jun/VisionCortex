# VisionCortex RTX 3050 离线部署与使用交付手册

> 适用机器：Ubuntu 20.04.6 LTS、Linux 5.15、NVIDIA GeForce RTX 3050 6GB<br>
> 建议整机配置：15 GiB 内存、2 GiB Swap、NVIDIA 驱动 570 或更高<br>
> 安装位置：`/opt/visioncortex-rtx3050`<br>
> 交付形式：exFAT 或 ext4 U 盘中的完整离线包<br>
> 文档版本：2026-10-10

本手册是 RTX 3050 交付的唯一主入口，覆盖 U 盘准备、离线安装、启动、任务提交、
结果验收和故障判断。它不适用于 RTX 3090 Ti 服务器，也不要求目标机重新下载
Python、CUDA Python 包、模型权重或 FFmpeg。

## 1. 先看结论

离线包默认完成环境与资产安装，可用于本机页面查看。默认不准备或调用模型，
不接入 NAS，也不要求外部厂商密钥。生产运行需要部署者先准备明确的私有站点配置。

只有新安装时显式选择 `--prepare-models --config PRIVATE_SITE`，才会在目标 RTX 3050
构建两套 TensorRT 引擎，并从 `16 → 8 → 4 → 2 → 1` 中测量稳定静态 batch。
安装回执与模型准备回执分别记录；已有安装不能通过重跑安装器覆盖。

当前使用边界如下：

| 使用方式 | 当前边界 |
| --- | --- |
| 本机页面和本地隔离模式 | 安装通过后按本地入口查看；不访问 NAS、不调用模型，不代表生产质量 |
| 已配置的正式分析 | 必须具备私有配置、实际输入与存储、已验证模型/引擎和适用认证；外部阶段使用所选厂商凭据，缺失门槛拒绝运行 |

目标 3050 上尚未产生的实机回执，不能由打包机结果代替。只有显式模型准备后，
目标机写出对应的通过执行回执，才能证明该机的 TensorRT 引擎真实执行成功。
只有完成一次冷启动真实六路运行，才能证明该机的端到端耗时、GPU/内存峰值、Token
和真实视频质量。

## 2. 交付物与硬件门槛

U 盘中应有一个完整目录，目录名类似：

```text
VisionCortexRTX3050-Ubuntu20-Offline-<版本>
├── VisionCortex-RTX3050-交付手册.md
├── Verify-Package.sh
├── Install-VisionCortex.sh
├── PACKAGE-MANIFEST.json
├── SHA256SUMS
├── app/
├── models/
├── research-candidates/  # 仅显式选择研究资产时存在
└── vendor/
```

离线包包含固定 Python 3.12 运行时、完整离线 wheelhouse、两套生产 21 类闭集 YOLO
源权重、YOLO-World、CLIP、Grounding DINO、SAM2.1、LabPics 和支持 CUDA/NVENC 的
便携 FFmpeg。研究候选默认不打包；显式提供 `--research-registry` 时只收录摘要校验
通过的权重与中性资产信息，不附内部训练或运行回执，也不会替换生产模型。

包内不包含以下内容：

- 其他显卡生成的 TensorRT `.engine`；3050 必须在本机重建。
- 外部厂商 API Key、网页登录密码或其他密钥。
- NAS 原视频、缓存、归档和运行产出。
- 尚不存在的真实视频质量认证结论。

安装器会逐项检查以下门槛：

| 项目 | 最低要求 |
| --- | --- |
| 操作系统 | Ubuntu 20.04 x86_64 |
| 内核 | 5.15.0 |
| GPU | NVIDIA GeForce RTX 3050 |
| 显存 | 5900 MiB |
| NVIDIA 驱动 | 570.0 |
| CUDA 计算能力 | 8.6 |
| 系统内存 | 14 GiB |
| Swap | 1.5 GiB |
| 安装前系统盘可用空间 | 18 GiB |
| 安装后运行可用空间 | 8 GiB |

安装前实测可用空间并保留运行余量。wheelhouse 在安装期间直接从 U 盘读取，
避免额外复制大资源；原始视频和已完成数据按私有站点存储合同保存。

## 3. 准备 U 盘

U 盘必须格式化为 **exFAT 或 ext4，不能使用 FAT32**。离线包中的 TensorRT wheel
单文件约 4.30 GB，超过 FAT32 的单文件上限。建议使用 32 GB 或更大的 U 盘。

把整个离线包目录原样复制到 U 盘。不要只复制安装脚本，不要拆分 wheelhouse，也
不要重命名、删除或修改包内文件。正式构建器已经移除不需要且会在 exFAT 合并的
Python 私有 terminfo 别名，并检查整包不存在仅大小写不同的路径。复制完成后，在
U 盘目录打开终端。

下面用实际 U 盘挂载路径替换尖括号部分：

```bash
cd "<实际离线包目录>"
bash ./Verify-Package.sh
```

只有看到下面一行才可以安装：

```text
package_verification=passed
```

如果校验失败，停止安装，重新复制完整目录。不要修改 `SHA256SUMS` 来绕过校验。

## 4. 第一次离线安装

在已通过校验的 U 盘目录执行：

```bash
bash ./Install-VisionCortex.sh
```

默认安装目录为 `/opt/visioncortex-rtx3050`；也可通过 `--install-root` 指定有写权限的
本地 SSD 目录。解释器、当前模型、活动 SQLite 和运行状态留在本地磁盘。

默认安装执行包完整性、基础依赖和资产校验，不准备 GPU 引擎，不询问厂商密钥，
不接入 NAS。安装回执记录 `model_preparation`，不能把 `status=passed` 单独当作
目标 GPU 已准备或真实视频质量已验收。

真实模型准备由部署者另行显式选择，并提供经过核对的私有站点配置：

```bash
bash ./Install-VisionCortex.sh --prepare-models --config "/absolute/private/site.yaml"
```

该路径是新安装的选项，不是对已有活跃安装的原地覆盖命令。模型准备包含固定资产
校验、目标 GPU 引擎构建、batch `16、8、4、2、1` 的有界测量和重复推理冒烟。
先确认没有使用该 GPU 的任务；保存原安装、输入和回执，按受控升级流程处理已有目录。

安装完成的末尾会显示：

```text
VisionCortex RTX 3050 offline installation completed.
```

安装后运行不依赖 U 盘。实际安装路径以下记为 `INSTALL_ROOT`；使用默认路径时：

```bash
export INSTALL_ROOT="/opt/visioncortex-rtx3050"
```

## 5. 安装后验收

先查看 `$INSTALL_ROOT/Runtime/install-receipt.txt`，核对包身份、安装路径和
`model_preparation`。默认安装的模型准备为跳过状态，这属于预期行为。

只有显式准备模型后，才核对安装回执所指向的 engine smoke 文件、两套引擎和
`.engine.build.json`：摘要、角色、batch、重复推理、吞吐与显存边界须对应当前目标机。
黑帧冒烟仅证明实际执行及资源边界；真实实验准确率使用独立真值和交付验收单记录。

## 6. 私有站点配置与凭据

生产配置由部署者保存在 Git 之外。明确声明 `project.run_purpose=analysis` 或
`production`、`project.site_configuration_required=false`、`runtime.local_only=false`，提供该站点的输入索引、
存储身份、已验证模型/引擎、设备绑定、服务设置和认证回执。

厂商按用户选择启用；本地页面不需要外部密钥。真实外部阶段仅通过所选厂商的环境
变量或批准的未跟踪密钥存储配置凭据。不要把原值写进 YAML、命令参数、Git、U 盘、
截图、聊天或日志。旧部署的 Ark 专用存储仅在仍选择该厂商时适用。

## 7. 启动与停止

### 7.1 本地页面

```bash
"$INSTALL_ROOT/Start-VisionCortex.sh" --local
```

默认只监听当前机器 `http://127.0.0.1:8003/#/home`；可显式提供 `--port`。
本地输入、缓存、staging 和产出位于本地运行根，NAS、自动采集和外部模型关闭。
页面能打开只证明本地界面与服务入口，真实分析须另行配置和准备。
源码克隆也可直接使用 `deployment/rtx3050-ubuntu20/Start-VisionCortex.sh --local`，
它按实际克隆及 `.venv` 解析路径，不依赖某台开发机的目录。

### 7.2 私有站点已准备：生产模式

先核对真实挂载身份、索引、归档/缓存根、模型与认证回执；缺失时停止，
不创建空目录冒充 NAS，不替换现有站点地址或输入。

```bash
"$INSTALL_ROOT/Start-VisionCortex.sh" --production --config "/absolute/private/site.yaml"
```

配置必须保留独立真值认证门禁，认证身份与当前资产和目标相符。通用包不内置
“自动通过”的现场质量结论。启动仍只绑定本机，跨机访问由部署方另行配置认证和监听。

前台进程需保持终端开启。停止前排空活动任务并保存回执，不强杀模型子进程。
重启后先重新核对配置、存储和资产，再启动同一已验收版本。

### 7.3 可选用户服务

```bash
VISIONCORTEX_INSTALL_ROOT="$INSTALL_ROOT" \
  bash "$INSTALL_ROOT/app/deployment/rtx3050-ubuntu20/Install-Autostart.sh" --local
```

默认安装独立 `visioncortex-rtx3050.service`。生产模式须显式提供
`--production --config "/absolute/private/site.yaml"`，不会覆盖分析或其他本地服务。
单位文件是模板，由当前克隆和解释器渲染并验证后安装；不能直接复制模板。
登录自动打开桌面入口需显式选择 `VISIONCORTEX_DESKTOP_AUTOSTART=1`。

## 8. 提交一次正式实验

生产页面启动后，优先使用 NAS 索引中的现有实验：

1. 打开“新建实验”。
2. 在 NAS 批次中选择已封口且确实发生实验的记录。
3. 核对第一人称、第三人称、路数、时间范围和实验 ID。
4. 输入新的英文安全归档名；不要覆盖已有正式归档。
5. 确认页面显示自动完整流水线，以及索引输入 `source_copy_bytes=0`。
6. 提交后记录任务 ID、源实验 ID、归档名和开始时间。

索引输入直接读取 NAS 原始物理分片，不复制原视频。缓存只保存可校验的派生中间
结果，用于同一任务断点恢复；正式冷运行不会把历史缓存冒充新的完整运行。

长视频不是只取前 5 分钟。正式索引任务对完整物理时间轴做实验发现，再对候选窗口
精扫并生成实验片段。只有真实动作证据达到门槛的视角才进入后续关键素材；拒绝原因
会保留在筛选说明中。

## 9. RTX 3050 性能策略

batch 并非固定为 1，也不保证固定为 16。显式模型准备时，安装器在目标机上按 `16、8、4、2、1`
从大到小实测，保留约 22% 显存给硬件解码和桌面合成，选出首个稳定候选。最终短
batch 会补齐后送入静态引擎，再裁回真实结果，不丢源帧。

已准备的 6GB 分析配置采用以下资源策略；实际执行与质量须通过回执确认：

- 六路输入仍全部处理；默认四路 CUDA 解码、两路 CPU 解码。
- 第一人称和第三人称全时间轴扫描器顺序驻留，避免双引擎同时挤爆显存。
- YOLO-World、Grounding DINO、SAM2 和 LabPics 只对有界关键素材运行，并在事件后释放。
- Grounding DINO 和 LabPics 放在 CPU，SAM2 状态和视频帧可卸载到内存。
- 解码、GPU 推理、CPU 后处理、NAS I/O 和已配置的外部理解阶段的瓶颈不同，不会每一秒都显示
  100% GPU；“打满硬件”应以整条流水线吞吐和无 OOM 为准，而不是强行锁定 batch。

最终只能以目标机真实冷运行回执回答性能：`run_metrics.json` 看总耗时和阶段耗时，
资源遥测看 GPU/显存、CPU 和内存峰值，所选厂商调用账本看输入/输出/总 Token。模型准备时的
引擎 images/s 不能外推为六路三小时端到端速度。

## 10. 去哪里看产出

生产模式完成并正式提升后，一个实验对应一个文件夹；归档根以配置与回执为准。
以下为已配置的传统 NAS 实验档案示例：

```text
<NAS挂载点>/VisionCortexExperimentArchive/<归档名称>/
```

本地页面默认归档根位于实际源码根下的 `outputs/development-runtime/archives/`，
可在运行状态中核对；首次启动通常为空。自定义配置以 `storage.archive_root` 为准，
不能把本地页面打开或空目录视为真实分析归档。

正式归档至少应包含：

```text
<归档名称>/
├── Experiment-Clips/                 # 筛选后的实验片段及理解 JSON
├── Key-Materials/
│   ├── Key-Clips/                    # 关键素材视频
│   ├── Key-Frames/                   # 只展示参与对象的关键帧/框
│   ├── Key-Materials-Model-Understanding.json
│   └── Key-Material-Timestamps.xlsx
├── Lab-Daily-Reports/                # 日报 JSON、Markdown、HTML、复核文件
├── Professional-PDFs/                # 专业 PDF
└── JSON-Config-Files/
    ├── pipeline_status.json
    ├── run_metrics.json
    ├── evidence_package.json
    ├── evidence_package_eval.json
    ├── final_key_material_annotation.json
    ├── evidence_index.sqlite
    ├── artifact_registry.jsonl
    ├── evidence_registry.jsonl
    └── Stage-Receipts/
```

不要只交付 PDF 或只有 JSON 的目录。关键素材事件应同时保留可查看的图像/视频、模型
理解、时间戳、参与对象框、权威 JSON 和可追溯到原始物理分片的引用。某候选若被
拒绝，也必须保留拒绝理由和不确定性，不能静默删除以制造高置信结果。

## 11. 一次正式运行的验收标准

任务只有同时满足以下条件才可标为 `PROVEN`：

- 页面最终状态为 `completed`，不是仅 HTTP 200 或页面可打开。
- `pipeline_status.json` 的阶段为 `completed`。
- `evidence_package_eval.json` 和 `daily_report_manifest.json` 均通过。
- 实验片段、关键帧、关键片段、理解 JSON、日报和 PDF 都存在且能打开。
- `run_metrics.json` 有真实总耗时、阶段耗时、GPU/显存和内存峰值。
- 正式外部模型调用有所选厂商返回的输入、输出和总 Token；缺失时保留 `null`，不能估算。
- NAS 索引输入的 `source_copy_bytes=0`。
- `artifact_registry.jsonl`、`evidence_registry.jsonl` 和 `evidence_index.sqlite` 可索引。
- 人工抽查确认时间对齐、实验边界、动作、参与对象框和文字理解合理。
- 正式目录提升与 SHA-256 校验完成，staging 没有冒充正式归档。

合成素材、单元测试、引擎黑帧、开放词汇模型置信度或模型共识都不能代替真实实验
准确率。没有独立真值时，结论只能写 `PARTIAL_EVIDENCE` 或 `NOT_PROVEN`。

## 12. 常见问题

| 现象 | 原因与处理 |
| --- | --- |
| `package_verification` 失败 | 包不完整或复制损坏；重新复制整个目录，禁止改校验表 |
| 安装前提示空间不足 | 系统盘可用空间低于 18 GiB；先按审批清理，不能把门槛改小 |
| 安装目标已存在 | 安装器拒绝覆盖；保存旧回执后走受控升级，禁止直接暴力删除 |
| GPU/驱动检查失败 | 机器不是约定的 RTX 3050 6GB，或驱动低于 570；停止安装 |
| batch 最终为 1、2、4 或 8 | 这是实机显存门禁的正常回退；以 build/smoke 回执为准 |
| 出现 CUDA OOM | 不要强制 batch 16；保存日志和两份 build 回执后排查其他显存进程 |
| 本地页面能开但不能正式运行 | 本地模式关闭 NAS 和外部模型；切换生产前补齐全部生产门禁 |
| 提示所选厂商密钥缺失/权限错误 | 按第 6 节提供站点私有凭据，并限制读权限 |
| 提示 NAS index/archive/cache 不可用 | 先恢复正确挂载；禁止创建同名空目录冒充 NAS |
| 提示模型认证缺失或过期 | 需要独立真值重新认证；禁止关闭门禁或伪造回执 |
| 其他电脑连不上 `8003` | 3050 启动器只绑定本机 `127.0.0.1`，不是 LAN 交付 |
| 重启后页面打不开 | 核对同一安装与配置；本地模式不需要 NAS，生产模式先核对明确配置的存储和资产，再按第 7 节启动 |

## 13. 明确禁止事项

- 不使用 FAT32 U 盘，不拆包，不跳过 SHA-256 校验。
- 不复制 3090 Ti、4060 或其他 GPU 的 TensorRT engine 到 3050。
- 安装期间不拔 U 盘；正式任务期间不卸载正在使用的存储、切换代码或修改模型配置。
- 不把密钥、模型、原视频、缓存、归档或运行产出提交到 Git。
- 不把开放词汇框、SAM2/LabPics 掩膜、伪标签或模型置信度冒充人工真值。
- 不为追求事件数量降低质量门槛，不隐藏被拒候选和不确定性。
- 不把本机 HTTP 服务端口直接暴露到公网。

## 14. 交付登记

交机时填写并随设备保存：

```text
离线包目录名：
PACKAGE-MANIFEST.json SHA-256：
SHA256SUMS SHA-256：
源代码提交 SHA：
目标机 GPU：
目标机驱动：
安装完成时间：
自动选择 batch（第一人称 / 第三人称）：
引擎冒烟回执：通过 / 未通过
本地页面验收：通过 / 未通过
NAS 挂载验收：通过 / 未通过 / 未执行
模型认证回执 SHA-256：
真实六路冷运行归档：
真实运行总耗时：
GPU / 显存 / 内存峰值：
所选厂商输入 / 输出 / 总 Token：
最终结论：PROVEN / PARTIAL_EVIDENCE / NOT_PROVEN
未完成门禁：
交付人 / 接收人 / 日期：
```

交付时应把本手册和整个离线包一起给用户。普通用户日常只需要第 7、8、10 节；
管理员负责安装、密钥、NAS、模型认证、性能回执和升级。
