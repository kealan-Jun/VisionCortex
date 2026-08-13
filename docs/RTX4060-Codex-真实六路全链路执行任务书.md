# RTX 4060 Codex：真实六路三小时全链路执行任务书

## 任务性质

这是运行任务，不是开发任务。你是 RTX 4060 实验运行节点，只负责同步已冻结代码、执行真实实验、保存正式产出和填写运行回传文档。

严禁：

- 修改任何源码、配置、脚本、权重或模板；
- 运行单元测试、开发测试、dry-run 或人工构造样例；
- 自行调整 FPS、batch、解码槽、阈值、模型、Token 或归档参数；
- 使用旧版的 82–88 GB 本地连续视频副本作为输入；
- 删除、reset、clean、覆盖或移动现场已有文件；
- 在文档、日志或终端输出中打印 `ARK_API_KEY`。

唯一允许新增的人工文件是本次运行回传文档，保存到正式 NAS 归档的 `JSON-Config-Files`。

## 1. 先同步 GitHub 冻结版本

仓库：`https://github.com/kealan-Jun/VisionCortex.git`

目标分支：`codex/daily-reports`

先在现有仓库执行只读检查：

```powershell
Set-Location -LiteralPath 'D:\VisionCortex'
git status --short --branch
git remote -v
```

若工作树干净：

```powershell
git fetch origin
git switch codex/daily-reports
git pull --ff-only origin codex/daily-reports
$RunCommit = (git rev-parse HEAD).Trim()
git status --short --branch
Write-Host "RUN_COMMIT=$RunCommit"
```

若工作树不干净，不得 reset、stash、clean 或覆盖。改用新的干净运行目录：

```powershell
$Runner = 'D:\VisionCortex-FrozenRunner'
if (Test-Path -LiteralPath $Runner) {
    throw "Frozen runner already exists; do not overwrite it. Record this blocker."
}
git clone --branch codex/daily-reports --single-branch https://github.com/kealan-Jun/VisionCortex.git $Runner
Set-Location -LiteralPath $Runner
$RunCommit = (git rev-parse HEAD).Trim()
git status --short --branch
Write-Host "RUN_COMMIT=$RunCommit"
```

同步后 `git status --porcelain` 必须为空。记录最终绝对仓库路径、分支和完整 commit SHA。

## 2. 固定输入、缓存与输出

- 索引：`Y:\experiment_record_index.csv`
- 实验 ID：`exp_20260810_144014_e918b762`
- 原始输入：索引字段 `rgb_file`、`frames_file` 指向的六路 15 分钟 NAS 分片
- 固定正式归档：`Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13`
- 隔离 staging：`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-Staging`
- 历史派生产物：`Y:\VisionCortexExperimentArchive\.VisionCortex-Run-History`
- 本地缓存：配置中的 `local_cache_root`，仅允许保存 TensorRT/YOLO 检测账本和断点

输入规则：

- 必须显示 6 个视角，每个视角 15 个视频分片及对应时钟 CSV；
- 必须使用 `nas_segmented_virtual_timeline`；
- `copied_source_bytes` 必须为 `0`；
- `continuous_source_copies_created` 必须为 `0`；
- 不得创建每路三小时的本地 `video.mp4`；
- 最终有界实验片段仍必须是连续可播放 MP4；跨分片时只拼接命中的短区间。

## 3. 运行前只读现场记录

不要运行测试。只记录：

```powershell
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber,CsTotalPhysicalMemory
nvidia-smi
Get-SmbConnection | Select-Object ServerName,ShareName,Dialect,NumOpens,Encrypted
Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object Name,InterfaceDescription,LinkSpeed
Get-PSDrive -Name D,Y | Select-Object Name,Used,Free,Root
```

检查 `ARK_API_KEY` 只能输出是否存在：

```powershell
$key = [Environment]::GetEnvironmentVariable('ARK_API_KEY','User')
Write-Host ('ARK_API_KEY_CONFIGURED=' + (-not [string]::IsNullOrWhiteSpace($key)))
$key = $null
```

若旧版 `run-fixed-benchmark` 仍在运行，不得并行启动新版。只终止命令行明确属于旧固定基准的进程，并在报告中记录 PID、命令行和停止时间；不得终止其他 Python/FFmpeg 进程。

## 4. 执行真实六路三小时全链路

不要启动 dry-run，不要运行 pytest。使用仓库原有正式脚本：

```powershell
Set-Location -LiteralPath '<第1步确认的干净仓库绝对路径>'
.\deployment\rtx4060\04-重跑固定六路基准.ps1
```

运行期间不要改参数。持续观察正式 NAS staging 中的：

- `JSON-Config-Files\pipeline_status.json`
- `JSON-Config-Files\resource_telemetry.json`
- `JSON-Config-Files\scan_runtime_coarse.json`
- `JSON-Config-Files\scan_runtime_fine.json`
- `JSON-Config-Files\run_metrics.json`

必须让任务自然完成或自然失败。不得因为 GPU 利用率短时不高而擅自改并发。

## 5. 必须核验的结果

成功运行至少应有：

- 五个真实有界实验组，或对数量差异给出明确证据；
- 每个实验组：第一人称连续 MP4、第三人称连续 MP4、第一+第三并排连续 MP4，以及三份对应 JSON；
- 关键素材：五大类事件、对齐关键帧、连续关键片段、统一事件 JSON；
- 实验级步骤理解：当前步骤、下一步骤、跨视角一致性与不确定性；
- `Lab-Daily-Reports` 下的日报 JSON/Markdown/HTML；
- `Professional-PDFs` 下的日报 PDF；
- `evidence_package_eval.json` 和日报校验均通过；
- `run_metrics.json` 有总耗时、阶段耗时、实验理解 Token、关键素材 Token 和总 Token；
- 性能账本能证明粗扫六路同时活跃，解码后端为 4 路 CUDA/NVDEC + 2 路 CPU；
- TensorRT 两个角色均实际加载 `.engine`。

## 6. 回传文档

复制 `docs\RTX4060-真实六路运行回传模板.md`，填写为：

`Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13\JSON-Config-Files\RTX4060-Full-Run-Handoff-<YYYYMMDD-HHMMSS>.md`

禁止在报告中写密钥。所有结论必须引用 NAS 中的 JSON、文件路径、时间戳或命令输出。即使运行失败也必须生成报告，准确写明失败阶段、错误、已完成产物和未完成项。
