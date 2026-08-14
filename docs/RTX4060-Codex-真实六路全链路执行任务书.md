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

运行记录只追加到 NAS 单一协作文档，不再为每次心跳或回传新建人工文档：

`Y:\VisionCortexExperimentArchive\.VisionCortex-Collaboration\Six-View-Three-Hour-Experiment-2026-08-13.md`

## 1. 先同步 GitHub 冻结版本

仓库：`https://github.com/kealan-Jun/VisionCortex.git`

目标分支：`codex/post-dev008-adaptive-throughput`

在 4060 Codex 当前已经打开的 VisionCortex 仓库中执行。不得假定仓库位于 `D:`，也不得另建仓库副本：

```powershell
$ProjectRoot = & git rev-parse --show-toplevel 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ProjectRoot)) {
    throw '当前目录不在 Git 仓库中；请报告实际仓库位置，禁止自行猜测路径或重复克隆。'
}
$ProjectRoot = $ProjectRoot.Trim()
Set-Location -LiteralPath $ProjectRoot
$OriginUrl = & git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($OriginUrl)) {
    throw '当前仓库没有可用的 origin 远端。'
}
$OriginUrl = $OriginUrl.Trim()
if ($OriginUrl -notmatch 'github\.com[/:]kealan-Jun/VisionCortex(?:\.git)?$') {
    throw "当前仓库不是 kealan-Jun/VisionCortex：$OriginUrl"
}
git status --short --branch
git remote -v
$Dirty = @(git status --porcelain)
if ($Dirty.Count -gt 0) {
    $Dirty
    throw '工作树不干净；停止同步并报告，不得处理、覆盖或复制用户改动。'
}
```

若 `git status --porcelain` 不为空，立即停止并把状态写入回传文档；不得 reset、stash、clean、覆盖、另建副本或处理用户改动。工作树干净时只同步 GitHub：

```powershell
git fetch origin
git switch codex/post-dev008-adaptive-throughput
git pull --ff-only origin codex/post-dev008-adaptive-throughput
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
- 本轮检测 `reused` work units 必须为 `0`，不得把旧 checkpoint 当冷启动成绩；
- 不得创建每路三小时的本地 `video.mp4`；
- 最终有界实验片段仍必须是连续可播放 MP4；跨分片时只拼接命中的短区间。

## 3. 运行前只读现场记录

不要运行测试。只记录：

```powershell
Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber,CsTotalPhysicalMemory
nvidia-smi
Get-SmbConnection | Select-Object ServerName,ShareName,Dialect,NumOpens,Encrypted
Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object Name,InterfaceDescription,LinkSpeed
$ProjectDrive = ([System.IO.Path]::GetPathRoot($ProjectRoot)).TrimEnd('\').TrimEnd(':')
Get-PSDrive -Name @($ProjectDrive,'Y') | Select-Object Name,Used,Free,Root
```

检查 `ARK_API_KEY` 只能输出是否存在：

```powershell
$key = [Environment]::GetEnvironmentVariable('ARK_API_KEY','User')
Write-Host ('ARK_API_KEY_CONFIGURED=' + (-not [string]::IsNullOrWhiteSpace($key)))
$key = $null
```

若旧版 `run-fixed-benchmark` 仍在运行，不得并行启动新版。只终止命令行明确属于旧固定基准的进程，并在报告中记录 PID、命令行和停止时间；不得终止其他 Python/FFmpeg 进程。

## 4. 执行真实六路三小时全链路

不要启动 dry-run，不要运行 pytest。先确认 Web 服务已作为隐藏后台进程启动，再用正式脚本提交一次：

```powershell
Set-Location -LiteralPath $ProjectRoot
.\deployment\rtx4060\03-停止Web.ps1
.\deployment\rtx4060\02-启动Web.ps1 -NoBrowser
.\deployment\rtx4060\04-重跑固定六路基准.ps1
```

只有在第 3 节确认没有旧固定基准进程后，才允许执行 `03` 停止旧 Web。新 Web 健康检查必须报告 `fixed_benchmark.submission_protocol_version=1`，否则禁止提交。

`04` 脚本必须在 30 秒内返回 `SUBMITTED_RUN_ID`、`STATUS_URL`、`NAS_STAGING` 和 `SUBMISSION_RECEIPT`。它只是短生命周期客户端，不拥有流水线进程；真实任务由此前已启动的 Web 服务持有。若 Web 不健康、协议版本错误、返回信息不完整或脚本超时，视为未成功提交并立即报告，禁止退回前台 CLI、禁止自行重试。提交成功后不得关闭或重启 Web 服务，也不得再次 POST。

持续监控只允许读取返回的 `STATUS_URL` 与对应 NAS staging。不要让任何带超时的命令持续等待流水线，也不要把客户端命令退出码误当作流水线最终退出码。最终状态以状态 API 和 `JSON-Config-Files\pipeline_status.json` 为准。

运行期间不要改参数。持续观察正式 NAS staging 中的：

- `JSON-Config-Files\pipeline_status.json`
- `JSON-Config-Files\run_submission.json`
- `JSON-Config-Files\resource_telemetry.json`
- `JSON-Config-Files\scan_runtime_motion_probe.json`
- `JSON-Config-Files\scan_runtime_coarse.json`
- `JSON-Config-Files\scan_runtime_fine.json`
- `JSON-Config-Files\motion_probe_windows.json`
- `JSON-Config-Files\run_metrics.json`

必须让任务自然完成或自然失败。运动探针只对哨兵视角做低成本稀疏跳读，此阶段 GPU Compute 低属于设计预期；候选粗扫与有界精扫必须显示第一/第三角色推理通道并发、六路来源活跃和实际 TensorRT batch 填充率。不得因为单个阶段 GPU 利用率短时不高而擅自改并发。

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
- `scan_runtime_motion_probe/coarse/fine.json` 均明确报告 `cold_start=true`、`computed` 与 `reused`；
- `input_volume_report.json` 明确报告唯一 MP4 数量、82.13 GiB 左右的实际输入体积及不存在重复路径。

## 6. 单一协作日志回传

收到任务后先在单一协作文档追加 `ACK`；运行阶段变化时和每 5 分钟追加 `HEARTBEAT`；结束时追加 `COMPLETED` 或 `INCIDENT`。不得创建新的 handoff 文档。

每条记录至少包含：`run_id`、阶段与阶段耗时、六路 work-unit 进度、computed/reused、GPU Compute、显存、NVDEC、NVENC、温度、功耗、CPU、内存、NAS 收发吞吐、队列深度、batch 填充率和对应 NAS 证据路径。取不到就写 `unavailable`，不得猜测。

禁止写密钥。所有结论必须引用 NAS 中的 JSON、文件路径、时间戳或命令输出。即使运行失败也必须追加记录，准确写明失败阶段、错误、已完成产物和未完成项。
