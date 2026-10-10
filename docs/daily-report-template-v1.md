# 实验室日报固定模板 V1

`VC-LAB-DAILY-REPORT-V1` 是流水线的固定日报契约。模型只负责实验片段与关键素材的结构化理解；日报不再次调用模型，也不允许模型修改栏目、顺序或版式。渲染器仅把已经验收的证据包字段填入模板，因此日报阶段新增调用数与 Token 恒为 0。

## 固定栏目

1. 当日总览：输入视角、有界实验、候选/接受/拒绝事件、关键事件、状态变化、证据检查、总耗时和总 Token。
2. 全天实验矩阵：实验名称、边界、连续性、时长、步骤、关键事件、状态变化和参与视角。
3. 跨视角对齐审计：每一路的角色、对齐状态、置信度、CSV 最近邻匹配率与 RMSE、视觉校准量。
4. 逐实验步骤与证据：模型名称/状态/置信度/Token、细粒度当前步骤和下一步骤、物理变化、关键事件、跨视角支持和素材路径。
5. 六类关键动作汇总。
6. 质量、不确定性与矛盾。
7. 总耗时、阶段耗时、总 Token、阶段输入/输出 Token 和调用数。
8. 自动验收结论与可选备注区。

模板源文件是 `src/visioncortex/templates/VC-LAB-DAILY-REPORT-V1.json`。每份日报保存模板 ID、版本与 SHA-256；评估器会核对固定栏目和模板哈希。需要调整栏目时应新增 V2，历史 V1 不回写、不覆盖。

## 产物

```text
Lab-Daily-Reports/<YYYY-MM-DD>/
  Lab-Daily-Report-<date>.json
  Lab-Daily-Report-<date>.md
  Lab-Daily-Report-<date>.html
  Daily-Report-Eval.json
  Human-Review.json
Professional-PDFs/
  Lab-Daily-Report-<date>.pdf
JSON-Config-Files/
  daily_report_manifest.json
```

日报 JSON 是事实源，PDF 用于日常阅读和签署，Web 端读取同一 JSON。只有底层证据包评估通过、日报评估通过且 PDF 存在，固定归档才能正式提升。

## 生成与验收

在已验证的环境中使用当前归档与配置生成日报，不读取其他日期的音频或证据：

```bash
: "${VISIONCORTEX_SITE_CONFIG:?Set the selected config path}"
: "${VISIONCORTEX_REPORT_ARCHIVE:?Set the verified task archive}"
visioncortex generate-daily-report --archive "$VISIONCORTEX_REPORT_ARCHIVE" \
  --config "$VISIONCORTEX_SITE_CONFIG"
```

使用当前回执验收来源、时间、模型实际输出和报告；缺失项明确保留，不用旧任务数据填补。
