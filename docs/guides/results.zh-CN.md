# 产物与结果阅读

[文档导航](../README.md) · [项目首页](../../README.md)

当前操作参考；执行前核对所选配置、固定源码 SHA 和适用运行回执。

## 产出

以下是离线多视角实验入口的既有布局。NAS 自动采集使用设备日固定五目录，
见[设备日归档契约](../DEVICE-DAY-ARCHIVE-CONTRACT.zh-CN.md)；不要将两种布局混用。

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

旧离线实验入口的六类事件是 `hand_object_contact`（手与明确物体接触）、`object_movement`、`liquid_movement`、`container_state_change`、`device_panel_operation` 和 `pipette_transfer_operation`。每个记录保留候选、接受/拒绝理由、视角支持、对齐置信度和不确定性，YOLO 框不会被直接当成最终证据。


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
