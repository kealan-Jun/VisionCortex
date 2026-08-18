# VisionCortex 工程化加固与模块边界

## 结论先行

本轮采用“冻结算法基线，先抽离新能力”的方式工程化，不对已经通过真实六路验收的 CV 边界、连续性、跨视角和关键事件选择逻辑做大规模搬迁。新增能力分别落在独立模块：

- `schema_contracts.py`：正式归档 JSON 契约和关键素材引用校验；
- `replay_acceptance.py`：只读账本回放，不重新读取视频；
- `annotation_workspace.py`：YOLO badcase 人工审核、决策账本和真值导出；
- Web 仅增加批次处理账本和 YOLO 标注入口；
- `pipeline.py` 只增加契约清单写入调用，不改变候选、边界或事件选择。

这比一次性拆分四千行主流水线更安全：真实运行能力先被冻结，模块迁移必须在字符化测试和真实数据验收均通过后逐步进行。

## 当前冻结能力

以下能力在本轮不得因重构而改变：

1. 输入路数动态解析，至少包含第一人称和第三人称，不写死 2+4 或 1+5；
2. NAS 15 分钟物理分片组成零复制虚拟连续时间线；
3. CSV 最近邻与视觉锚点对齐；
4. 三层漏斗：候选、审计、证据包；
5. 正式实验片段必须具备对齐的第一/第三人称媒体；
6. 独立实验与连续实验分组；
7. 五类关键动作、统一事件 JSON、关键帧/片段/时间戳引用；
8. 关键素材按实验及五大类归档，对象语义进入名称与 JSON；
9. 模型理解记录当前步骤、下一步骤、事实、推断、不确定项和 Token；
10. 每阶段产出和回执持续归档，完成后原子提升正式目录；
11. 已验收六路基线：5 组实验、至少 80 个关键事件、G2 独立单原子、G4 连续双原子、隔离事件零泄漏。

## 大文件现状与拆分顺序

| 文件 | 当前约行数 | 职责问题 | 安全拆分顺序 |
|---|---:|---|---|
| `pipeline.py` | 4,136 | 编排、性能、质量门、断点续跑混合 | 先抽阶段执行器接口，再按 preflight/alignment/candidate/audit/package 拆分 |
| `api.py` | 1,993 | 档案、批次、运行、搜索、标注路由集中 | 先抽只读 router，再抽写入 router；保持 URL 契约不变 |
| `archive.py` | 1,945 | 媒体物化、事件 JSON、证据包与评估混合 | 先抽序列化器，再抽媒体物化器 |
| `video_io.py` | 1,600 | 探测、稀疏解码、持久解码、编码混合 | 以 FFmpeg session 为边界拆分，保留帧账本测试 |
| `grouping.py` | 1,503 | 片段规范化、连续性图、关键事件选择混合 | 先抽纯连续性判定，再抽选择器 |
| `detection.py` | 1,371 | 模型加载、批处理、跟踪和扫描混合 | 先抽后端适配器；TensorRT 上下文不复制 |
| `actions.py` | 1,354 | 五类候选、粗筛、精筛与液体规则混合 | 按候选类型抽纯函数，保持审计输出字段一致 |
| `indexing.py` | 1,167 | JSONL、SQLite、哈希和查询 | 已有清晰产物边界，最后拆分 |

## 目标模块图

```text
index / upload
  -> ingest manifest
  -> alignment
  -> motion/coarse/fine candidates
  -> cross-view audit
  -> bounded experiment grouping
  -> key-material selection
  -> MLLM understanding
  -> media materialization
  -> evidence package + index + reports
  -> staged verified promotion
```

每条箭头必须由版本化 JSON 或显式 Python 数据模型连接。禁止下游通过猜测文件名、目录排序或中文名称来恢复上游语义。

## JSON 与索引契约

- 统一关键素材事件必须保留 `event_id`、时间范围、对象、观察、跨视角关联、决策、分数、关键帧、关键片段、证据 ID 和来源追踪；
- 素材引用必须是归档内相对路径，契约检查会拒绝越界路径和缺失文件；
- `evidence_index.sqlite` 与 JSONL registry 是可重建派生索引，JSON 仍是权威事实源；
- `schema_contract_manifest.json` 记录每个契约的存在性、版本、字段和失败原因；
- 旧归档缺少后加的可选索引时只产生兼容性警告，不伪装成新能力已经存在。

## 无视频回放验收

命令：

```powershell
$env:PYTHONPATH = (Resolve-Path 'src').Path
python tools/replay_archive_acceptance.py `
  --archive 'Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13' `
  --baseline 'configs\evaluation\six-view-archive-regression-baseline.json' `
  --output 'D:\VisionCortexLocal\Runtime\Archive-Replay\Six-View-Archive-Replay-2026-08-18.json'
```

该回放只读取正式 JSON 账本，视频和时钟 CSV 打开数固定为 0。它不能替代冷启动真实性能测试，但可以在重构后快速证明正式业务语义没有回退。

## YOLO 标注工作台边界

- 定位：仅供内部模型开发与数据审核，默认关闭，不出现在实验用户主导航；
- 输入：badcase 审计生成的 `YOLO-Annotation-Queue.json`；
- 自动行为：发现队列、排序、筛选、呈现图片和既有模型证据；
- 人工行为：确认漏检/误检/类别错误，填写正确类别和真实框；
- 输出：原子写入的决策账本，以及仅包含人工给框样本的真值导出；
- 明确禁止：系统根据模型框自动伪造 ground truth，或把未审核 badcase 计入召回率。

## 分阶段拆分门

每次只迁移一个纯职责，并同时满足：

1. 全量单元/回归测试通过；
2. Python 编译和前端语法检查通过；
3. 六路正式归档 JSON 回放通过；
4. dry-run 不依赖视频、FFmpeg、GPU 或模型 API；
5. 4060 冷启动真实运行通过质量门后，才允许进入稳定仓；
6. 不把开发分支直接当作 4060 冻结 SHA。

## Real Issue List

| 状态 | 现象 | 证据 | 下一步 |
|---|---|---|---|
| 已修复 | 既有正式归档导致 DEV-039 CLI 拒绝 | DEV-039 INCIDENT；修复提交 `c648868` | DEV-040 验证原子替换与历史保留 |
| 已缓解 | JSON 可索引但旧归档缺少后加索引清单 | 正式六路回放产生兼容性 warning | 新归档自动写 schema manifest 与 evidence index |
| 已实现 | badcase 有队列但缺少可操作审核页 | 本地队列 72 条，P0/P1/P2=32/36/4 | 在 Web 完成人工审核并积累训练真值 |
| 需数据 | 21 类 YOLO 不直接表达液体/液面状态 | 两套模型类别元数据均无液体状态类 | 先用容器 ROI + 状态专家；再按 badcase 决定增标 |
| 需真实验收 | 预处理速度与质量尚未在同一冷启动运行同时证明 | DEV-031 914.92 秒但质量失败；旧正式质量通过但 2391.82 秒 | 继续按 one-shot 冻结任务验收，不合并两套证据 |
| 计划内 | 主文件仍过大 | 上表行数统计 | 按字符化测试逐模块迁移，禁止大爆炸重构 |

## 双仓发布规则

- `kealan-Jun/VisionCortex`：开发、评审、真实实验候选；
- `RealityLoopAI/VisionCortex`：仅接收已通过真实数据和发布门的成熟稳定能力；
- 本轮工程化分支在 4060 真实验收前不得同步覆盖稳定仓。
