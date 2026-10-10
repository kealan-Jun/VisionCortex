"""Run a local, separately archived first/third-person detector pilot."""

import argparse
import json
from pathlib import Path

from visioncortex.project_annotation_training import run_experiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True, type=Path)
    p.add_argument("--receipt-sha256", required=True)
    p.add_argument("--role", required=True, choices=["first_person", "third_person"])
    p.add_argument("--base-model", required=True, type=Path)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--image-size", type=int, default=960)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--nominal-batch", type=int, default=64,
                   help="梯度累积的基准批量；设为实际 batch 可避免跨批累积")
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--warmup-bias-lr", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=20,
                   help="验证早停耐心；0 禁用早停，仍受 epochs 和时间上限约束")
    p.add_argument("--freeze-layers", type=int, default=10,
                   help="冻结前 N 层，范围 0–10；0 允许主干参与训练，默认保持 10")
    p.add_argument("--trace-branch-loss", action="store_true",
                   help="记录两个分支的实际训练损失/权重；仅用于区域外监督")
    p.add_argument("--branch-loss-policy", choices=["native", "one2many"], default="native",
                   help="默认原生目标；one2many 仅优化一对多分支，须启用分支记录并禁用早停")
    p.add_argument("--supervision", choices=["complete", "outside_ignore"], default="complete",
                   help="区域外训练仅接受独立部分监督导出及显式忽略区域")
    p.add_argument("--sampling-policy", choices=["image_uniform", "source_balanced"],
                   default="image_uniform", help="来源平衡仅用于无增强部分监督实验，验证集不变")
    p.add_argument(
        "--training-scope", choices=["detector", "class_outputs"], default="detector",
        help="class_outputs 仅训练分类输出，固定特征、定位和归一化统计量",
    )
    p.add_argument(
        "--migrate-legacy-tip-box",
        action="store_true",
        help="按已核查的旧整盒枪头语义初始化枪头盒类别，单支枪头重新初始化",
    )
    p.add_argument(
        "--preserve-class-head",
        action="store_true",
        help="保留已核对类别对应的旧分类输出参数",
    )
    a = p.parse_args()
    result = run_experiment(
        a.export,
        a.receipt_sha256,
        a.role,
        a.base_model,
        a.model_sha256,
        a.output,
        epochs=a.epochs,
        image_size=a.image_size,
        batch=a.batch,
        max_seconds=a.max_seconds,
        preserve_class_head=a.preserve_class_head,
        learning_rate=a.learning_rate,
        migrate_legacy_tip_box=a.migrate_legacy_tip_box,
        training_scope=a.training_scope,
        supervision=a.supervision,
        sampling_policy=a.sampling_policy,
        nominal_batch=a.nominal_batch,
        warmup_epochs=a.warmup_epochs,
        warmup_bias_lr=a.warmup_bias_lr,
        patience=a.patience,
        freeze_layers=a.freeze_layers,
        trace_branch_loss=a.trace_branch_loss,
        branch_loss_policy=a.branch_loss_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
