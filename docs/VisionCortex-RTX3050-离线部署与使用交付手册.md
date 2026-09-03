# VisionCortex RTX 3050 离线部署与使用交付手册

> 适用机器：Ubuntu 20.04.6 LTS、Linux 5.15、NVIDIA GeForce RTX 3050 6GB<br>
> 建议整机配置：15 GiB 内存、2 GiB Swap、NVIDIA 驱动 570 或更高<br>
> 安装位置：`/opt/visioncortex-rtx3050`<br>
> 交付形式：exFAT 或 ext4 U 盘中的完整离线包<br>
> 文档版本：2026-09-03

本手册是 RTX 3050 交付的唯一主入口，覆盖 U 盘准备、离线安装、启动、任务提交、
结果验收和故障判断。它不适用于 RTX 3090 Ti 服务器，也不要求目标机重新下载
Python、CUDA Python 包、模型权重或 FFmpeg。

## 1. 先看结论

离线包可以完成一次性环境安装，并在目标 RTX 3050 上自动构建两套 TensorRT 引擎、
从 `16 → 8 → 4 → 2 → 1` 中选择实际可稳定运行的最大静态 batch。安装完成后不需要
再次配置 Python 环境或模型，重启电脑后只需重新运行启动脚本。

但“开箱即用”分为两个边界：

| 使用方式 | 当前边界 |
| --- | --- |
| 本机页面和本地隔离模式 | 安装通过后可用；不访问 NAS，不调用豆包，不代表生产质量 |
| NAS 正式全链路 | 必须同时具备 NAS、Ark 密钥和有效模型认证回执；缺一项就拒绝运行 |

目标 3050 上尚未产生的实机回执，不能由打包机结果代替。只有安装器在目标机写出
通过回执，才能证明该机的 TensorRT 引擎真实执行成功；只有完成一次冷启动真实六路
运行，才能证明该机的端到端耗时、GPU/内存峰值、Token 和真实视频质量。

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
├── research-candidates/
└── vendor/
```

离线包包含固定 Python 3.12 运行时、106 个离线 wheel、两套生产 21 类闭集 YOLO
源权重、YOLO-World、CLIP、Grounding DINO、SAM2.1、LabPics 和支持 CUDA/NVENC 的
便携 FFmpeg。研究候选只随包保存供审计，不会自动替换生产模型。

包内不包含以下内容：

- 其他显卡生成的 TensorRT `.engine`；3050 必须在本机重建。
- Ark API Key、网页登录密码或其他密钥。
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

目标机当前约 22 GiB 可用，只够按本方案安装。不要把 wheelhouse 或 U 盘内容再复制
到系统盘；它们在安装期间直接从 U 盘读取。安装完成后也不要在系统盘保存原视频。

## 3. 准备 U 盘

U 盘必须格式化为 **exFAT 或 ext4，不能使用 FAT32**。离线包中的 TensorRT wheel
单文件约 4.30 GB，超过 FAT32 的单文件上限。建议使用 32 GB 或更大的 U 盘。

把整个离线包目录原样复制到 U 盘。不要只复制安装脚本，不要拆分 wheelhouse，也
不要重命名、删除或修改包内文件。复制完成后，在 U 盘目录打开终端。

下面用实际 U 盘挂载路径替换尖括号部分：

```bash
cd "/media/$USER/<U盘卷标>/VisionCortexRTX3050-Ubuntu20-Offline-<版本>"
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

系统可能要求输入当前 Ubuntu 管理员密码，这是 `sudo` 创建 `/opt` 安装目录所需。
安装器还可能无回显地询问 Ark API Key：

- 近期要运行 NAS 正式全链路：安全输入一次。
- 只先检查本地页面：直接回车跳过，之后再按第 6 节设置。

安装过程会真实执行以下操作：

1. 再次校验整个离线包。
2. 检查系统、GPU、驱动、内存、Swap、磁盘、CUDA 解码和 NVENC。
3. 在 `/opt/visioncortex-rtx3050` 创建隔离环境并离线安装依赖。
4. 校验生产模型哈希。
5. 在目标 RTX 3050 上分别构建第一人称和第三人称 TensorRT 引擎。
6. 对 batch `16、8、4、2、1` 从大到小做有界实测，选择首个满足显存预留的候选。
7. 用两套引擎做重复推理冒烟测试并写出审计回执。
8. 清理仅用于其他 GPU 架构的 TensorRT builder 资源，保留 SM86 和 PTX。

首次建引擎可能需要较长时间，终端没有回到提示符前不要关机、拔 U 盘或按
`Ctrl+C`。安装器不会覆盖已有 `/opt/visioncortex-rtx3050`；若该目录已存在，它会
主动停止。不要直接删除旧目录，应先由管理员保存回执并执行受控升级。

成功时末尾应包含：

```text
preflight=passed
VisionCortex RTX 3050 offline installation completed.
```

安装成功后可拔出 U 盘；日常运行不再依赖 U 盘。

## 5. 安装后验收

先检查三个回执：

```bash
cat /opt/visioncortex-rtx3050/Runtime/install-receipt.txt
python3 -m json.tool /opt/visioncortex-rtx3050/Runtime/rtx3050-engine-smoke.json
python3 -m json.tool /opt/visioncortex-rtx3050/Engines/first_person.engine.build.json
python3 -m json.tool /opt/visioncortex-rtx3050/Engines/third_person.engine.build.json
```

验收要点：

- `install-receipt.txt` 中 `status=passed`。
- `rtx3050-engine-smoke.json` 中 `status` 为 `passed`，两个角色都有记录。
- 两份 build 回执都有最终 `batch`、吞吐、显存测量和被接受的 autotune 记录。
- 两个 `.engine` 均位于 `/opt/visioncortex-rtx3050/Engines/`。

这里的黑帧冒烟测试证明 TensorRT 真实执行和显存边界，不证明实验视频准确率。

## 6. 安全设置 Ark 密钥

安装时已经输入过密钥可跳过本节。否则在目标机终端执行：

```bash
install -d -m 700 "$HOME/.config/VisionCortex"
IFS= read -r -s -p 'Ark API key: ' VISIONCORTEX_ARK_INPUT; printf '\n'
umask 077
printf '%s\n' "$VISIONCORTEX_ARK_INPUT" > "$HOME/.config/VisionCortex/ark_api_key"
chmod 600 "$HOME/.config/VisionCortex/ark_api_key"
unset VISIONCORTEX_ARK_INPUT
```

密钥只保存在当前用户的本地文件中，权限必须是 `600`。不要把密钥写进命令参数、
YAML、Git、U 盘、截图、聊天或日志。

## 7. 启动与停止

### 7.1 NAS 未就绪：本地隔离模式

```bash
/opt/visioncortex-rtx3050/Start-VisionCortex.sh --local
```

在同一台电脑打开：

```text
http://127.0.0.1:8000/#/home
```

本地模式的输入、缓存、staging 和产出全部位于：

```text
/opt/visioncortex-rtx3050/Runtime/NoNasWeb
```

本地模式不会读取或写入 NAS，且豆包关闭。它适合检查页面和本机隔离功能，不能作为
真实全链路交付结果。

### 7.2 NAS 已就绪：生产模式

系统接受以下任一既有 NAS 挂载点：

```text
$HOME/桌面/nas
/mnt/visioncortex-nas
```

挂载点内必须已有：

```text
experiment_record_index.csv
VisionCortexExperimentArchive/
VisionCortexExperimentArchive/.VisionCortex-Run-Staging/
VisionCortexExperimentCache/
```

然后执行：

```bash
/opt/visioncortex-rtx3050/Start-VisionCortex.sh --production
```

生产模式还要求有效模型认证回执位于：

```text
/opt/visioncortex-rtx3050/Runtime/Model-Quality/production_model_certification.json
```

该回执必须由独立真值评估生成，并与当前模型哈希、真值、框评估和认证目标一致。
通用离线包不会伪造或内置一个“自动通过”的认证回执。回执缺失、过期或指标不达标
时，正式任务会 fail-closed，这是质量保护，不是安装故障。

启动脚本当前绑定 `127.0.0.1`，因此页面只供这台 3050 本机浏览器访问，不是 3090 Ti
版本的局域网服务器。如果需要其他电脑访问，必须另行交付身份认证、私网绑定和服务
自启动方案，不能直接把端口暴露到公网。

脚本以前台方式运行，终端必须保持打开。在该终端按 `Ctrl+C` 可正常停止。电脑重启
后不需要重新安装，但需要确认 NAS 已重新挂载，再重新执行对应启动命令。当前离线包
没有安装 systemd 自启动服务。

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

batch 并非固定为 1，也不保证固定为 16。安装器在目标机上按 `16、8、4、2、1`
从大到小实测，保留约 22% 显存给硬件解码和桌面合成，选出首个稳定候选。最终短
batch 会补齐后送入静态引擎，再裁回真实结果，不丢源帧。

6GB 显存下采用以下策略保证完整链路：

- 六路输入仍全部处理；默认四路 CUDA 解码、两路 CPU 解码。
- 第一人称和第三人称全时间轴扫描器顺序驻留，避免双引擎同时挤爆显存。
- YOLO-World、Grounding DINO、SAM2 和 LabPics 只对有界关键素材运行，并在事件后释放。
- Grounding DINO 和 LabPics 放在 CPU，SAM2 状态和视频帧可卸载到内存。
- 解码、GPU 推理、CPU 后处理、NAS I/O 和豆包阶段的瓶颈不同，不会每一秒都显示
  100% GPU；“打满硬件”应以整条流水线吞吐和无 OOM 为准，而不是强行锁定 batch。

最终只能以目标机真实冷运行回执回答性能：`run_metrics.json` 看总耗时和阶段耗时，
资源遥测看 GPU/显存、CPU 和内存峰值，豆包账本看输入/输出/总 Token。安装时的
引擎 images/s 不能外推为六路三小时端到端速度。

## 10. 去哪里看产出

生产模式完成并正式提升后，一个实验对应一个文件夹：

```text
<NAS挂载点>/VisionCortexExperimentArchive/<归档名称>/
```

本地隔离模式的对应位置是：

```text
/opt/visioncortex-rtx3050/Runtime/NoNasWeb/Archives/<归档名称>/
```

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
- 正式豆包调用有服务端返回的输入、输出和总 Token；缺失时保留 `null`，不能估算。
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
| 本地页面能开但不能正式运行 | 本地模式关闭 NAS 和豆包；切换生产前补齐全部生产门禁 |
| 提示 Ark 密钥缺失/权限错误 | 按第 6 节设置，文件必须属于当前用户且权限为 `600` |
| 提示 NAS index/archive/cache 不可用 | 先恢复正确挂载；禁止创建同名空目录冒充 NAS |
| 提示模型认证缺失或过期 | 需要独立真值重新认证；禁止关闭门禁或伪造回执 |
| 其他电脑连不上 `8000` | 3050 启动器只绑定本机 `127.0.0.1`，不是 LAN 交付 |
| 重启后页面打不开 | 环境仍在；先挂载 NAS，再重新执行第 7 节启动命令 |

## 13. 明确禁止事项

- 不使用 FAT32 U 盘，不拆包，不跳过 SHA-256 校验。
- 不复制 3090 Ti、4060 或其他 GPU 的 TensorRT engine 到 3050。
- 不在正式任务期间拔 U 盘、卸载 NAS、切换代码或修改模型配置。
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
豆包输入 / 输出 / 总 Token：
最终结论：PROVEN / PARTIAL_EVIDENCE / NOT_PROVEN
未完成门禁：
交付人 / 接收人 / 日期：
```

交付时应把本手册和整个离线包一起给用户。普通用户日常只需要第 7、8、10 节；
管理员负责安装、密钥、NAS、模型认证、性能回执和升级。
