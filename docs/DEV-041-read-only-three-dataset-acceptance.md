# DEV-041 Read-Only Three-Dataset Acceptance

This command turns the durable JSON ledgers from the six-view, A, and Z runs
into one quality, performance, Token, hardware, network, output, and promotion
matrix. It never opens source MP4, clock CSV, extracted image, key clip, PDF, or
model API payloads.

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python -m visioncortex.cli accept-three-datasets `
  --archive-root 'Y:\VisionCortexExperimentArchive' `
  --spec 'configs\acceptance\dev041-three-dataset.json' `
  --output 'C:\VisionCortex-Acceptance\DEV-041-three-dataset-acceptance.json'
```

The command writes JSON plus a sibling Markdown report outside the NAS archive.
It exits `0` only when all three latest runs pass every applicable gate. It exits
`1` for an incomplete, failed, unpromoted, or quality/performance-regressed run.

Candidate discovery compares the formal archive with every staging run for the
same English archive name and selects the newest durable `pipeline_status`
timestamp. Therefore an old successful formal archive cannot hide a newer
failed staging run. When the final execution handoff contains exact run paths,
pin them explicitly:

```powershell
python -m visioncortex.cli accept-three-datasets `
  --archive-root 'Y:\VisionCortexExperimentArchive' `
  --candidate 'six_view=Y:\...\collection-six' `
  --candidate 'a=Y:\...\collection-a' `
  --candidate 'z=Y:\...\collection-z' `
  --output 'C:\VisionCortex-Acceptance\DEV-041-pinned.json'
```

The six-view dataset receives the reviewed five-group, G2/G4, 80-event, closed
formal-anchor, and 1,200-second gates. A and Z never inherit those priors; they
must produce their own natural bounded groups and event distribution. All three
still require aligned first-/third-person evidence, object-aware semantic
filenames, current/next-step understanding, searchable references, continuous
state receipts, branded reports, exact timing/Token ledgers, and verified atomic
promotion.


## 2026-09-16 主线适配

命令已迁入统一 `visioncortex` 包，同时读取 `Processing` 和历史 staging 目录。
选择最新尝试优先依据回执时间；缺少某个时间字段不会让文件修改时间盖过已有回执。
`passed` 只表示所选归档回执通过配置检查。`production_release_ready` 保留兼容字段但
不由只读脚本设为 true；`evidence_status` 区分部分证据与未证明，当前部署、真实视频
质量和浏览器交付须分别验收。合成测试不证明生产已就绪。
