# YOLO 独立框真值与人工标注合同

## 结论

`Predictions.jsonl`、旧事件 sidecar 和跨视角支持均不是独立真值。只有按本合同双人复核的框级 GT 才允许计算并对外报告逐类别 Precision、Recall、F1、AP50 和 AP50-95。

## 首轮优先标注

1. P0 小目标：`bottle_cap`、`tube_cap`、`magnetic_stir_bar`、`spatula`。
2. P0 手部口径：把 `hand` 定义为手实体，把是否戴手套作为属性；若暂不改类表，则两类必须互斥且写清边界。
3. P0 冲突类：`beaker/container`、`sample_bottle/reagent_bottle`、`pipette/tube_rack/spearhead`。
4. P1 困难条件：遮挡、运动模糊、画面边缘、暗光、第一人称头动和密集同类目标。
5. P1 硬负样本：纸张反光、空盒、瓶身标签、设备旋钮、线缆及手部附近的非目标小物体。

## 标注规则

- 框覆盖可见目标主体，不凭遮挡补画不可见区域。
- 每张图保存 `event_id + role + frame_index`，与预测账本形成稳定复合键。
- 记录 `visibility`、`occlusion`、`split` 和 `review_status`。
- 相邻帧、同一实验批次、同一操作员只能进入同一数据集分区，禁止随机拆分造成泄漏。
- 自动候选由标注员判定为真阳性、假阳性、假阴性或标签口径问题；不能把所有候选默认当错例。
- 训练集可单人标注抽检，验证集和测试集必须双人复核；争议样本由第三人裁决。

## 独立评测命令

```powershell
$env:PYTHONPATH='src'
python tools/evaluate_yolo_ground_truth.py `
  --predictions 'Predictions.jsonl' `
  --ground-truth 'yolo-box-ground-truth.json' `
  --output 'yolo-ground-truth-evaluation.json'
```

没有独立 GT 时状态必须写为 `not_evaluated_no_ground_truth`，不得复用训练日志或旧 sidecar 指标冒充本次正式评测。
