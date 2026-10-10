# 硬件、数据与本地验收

[文档导航](../README.md) · [项目首页](../../README.md)

当前操作参考；执行前核对所选配置、固定源码 SHA 和适用运行回执。

## 本地与站点存储

源码使用实际安装目录；运行环境、模型、引擎与活动数据库保留在配置的本地存储。
站点部署私有保存输入索引、设备绑定、NAS 根及运行证据，不把维护者的绝对路径当作通用默认值。
推荐按职责组织：

```text
<本地部署根>/
  ├── .venv/
  ├── Models/ClosedSetYOLO/<role>/
  ├── Engines/
  └── Runtime/
      ├── ThirdParty/
      ├── Model-Quality/
      └── NoNasWeb/
```

Git 仓库只保存代码、模型版本/下载地址和 SHA-256，不保存 `.pt`、
`.engine`、`.safetensors`、原视频、运行产出或密钥。两套项目训练的闭集
权重须按 `configs/models/closed-set-yolo.json` 注册到本地
`Models/ClosedSetYOLO/<role>/best.pt`；3090 Ti 安装器会验固定哈希、重建 TensorRT
引擎，并自动下载和验签 YOLO-World、CLIP、Grounding DINO、SAM2.1 与
LabPics 液体/填充语义分割公共资产。

当前最终模型链不是单一 YOLO：两套 21 类 TensorRT 模型负责全时间轴候选；
ByteTrack 风格的两阶段关联保持对象轨迹；YOLO-World 与 Grounding DINO 仅在
已接受关键帧和有界时序补救中补齐小物体；豆包对双视角时序证据做动作与步骤
裁决；SAM2.1 对裁决后的参与对象框在短关键片段中双向传播，收紧最终展示框并
留下连续性收据；LabPics PSPNet 只在已接受的最终关键帧补充 vessel、filled、
liquid/solid phase 像素观察。开放词汇框、SAM2 掩码和 LabPics 掩码都不能单独
确认动作，正式生产仍受事件/参与对象框质量认证的 fail-closed 门禁约束。

RTX 3090 Ti 配置还启用有界选择性复核：液体、容器状态和移液关键帧固定进入
本地二次复核；其他动作仅在参与对象缺失、低置信、多实例或跨视角冲突时调用
开放词汇模型。清晰的闭集结果不重复推理。每次运行最多复核 120 个事件、每个
事件 2 个视角，并写入 `final_key_material_annotation.json` 的决策、模型耗时、
预算和 `source_copy_bytes=0` 回执。紧急回退可设置
`VISIONCORTEX_SELECTIVE_KEY_MATERIAL_VERIFICATION=false`，恢复原有行为。

### NAS 不可用时的本地验收

以下命令在已配置模型的本地环境使用独立输出；选定本地 profile 后核对其有效存储根，不能继承 NAS 路径。

```bash
# 六视角媒体、六类关键素材、步骤理解、日报、PDF、JSON/JSONL/SQLite
visioncortex run-local-acceptance \
  --output ./LocalAcceptance/Run \
  --config configs/rtx3090ti-ubuntu-local.yaml

# 六路 CUDA 解码 + 双 TensorRT 角色引擎硬件压测
visioncortex benchmark-local-hardware --output <local-output> \
  --media <h264-1> --media <h264-2> --media <h264-3> \
  --media <h264-4> --media <h264-5> --media <h264-6>

# 依次测量每角色 1/2/3 个 TensorRT 压测上下文；只给出容量结论，不自动改生产并发
visioncortex tune-local-hardware --output <new-local-output> \
  --duration-seconds 20 --config configs/rtx3090ti-ubuntu-local.yaml \
  --media <h264-1> --media <h264-2> --media <h264-3> \
  --media <h264-4> --media <h264-5> --media <h264-6>

# 真实执行全部本地生产 CV 模型；使用公开人工标注样本，不访问 NAS/豆包
visioncortex accept-local-models \
  --dataset ./LocalAcceptance/Dataset \
  --output ./LocalAcceptance/Run \
  --config configs/rtx3090ti-ubuntu-local.yaml

# 只读统计正式认证仍缺多少真值；不扫描生产归档
visioncortex model-certification-readiness \
  --config configs/rtx3090ti-ubuntu-production.yaml \
  --output ./LocalAcceptance/Run

# 安装仅绑定 127.0.0.1、重启自恢复且不继承 NAS 路径的离线验收 Web
deployment/rtx3090ti-ubuntu/05-Install-Local-Service.sh
```

本地六视角验收素材是明确标记的合成媒体，只证明结构、媒体、索引、报告与硬件
链路，不能作为真实实验质量结论。上述本地服务默认使用
`configs/development-local.yaml` 和端口 `8002`，关闭模型、多模态与自动采集，
归档、输入、缓存、运行和 staging 位于 `outputs/development-runtime`。
真实模型验收须明确选择已配置的分析 profile，并核对资产及实际执行回执。

液体语义公共数据由 `configs/public-data-sources.json` 管理；只有许可证明确、
HTTPS 且 SHA-256 固定的条目允许自动获取。`prepare-public-dataset` 会先验证完整
ZIP/RAR，再拒绝路径穿越、重复成员、链接和特殊文件，并在有界解包、文件数与
字节数复核全部通过后原子发布目录；失败的 partial 只作审计，绝不冒充完成数据。
模型共识只能通过
`build-consensus-labels` 生成 `pseudo_labels_not_ground_truth`，不能冒充人工真值。

双闭集源权重可用 `validate-closed-set-models` 对注册哈希和 21 类本体做双重
校验。框真值完成后，`evaluate-yolo-boxes` 计算独立 P/R/AP；
`build-yolo-training-dataset` 用软链接或硬链接构建零拷贝训练集；
`train-yolo-model` 只接受 reviewed ground truth，并把新权重标记为“未认证候选”。
公开模型、模型共识或训练日志都不能解除生产认证门禁。

Waseda Chemical Apparatus 公共人工框数据已固定 URL、大小与 SHA-256。它可用于
补充 hand、pipette 和六类实验器具，但不能直接替换项目的 21 类生产本体：

```bash
visioncortex prepare-public-dataset --dataset-id WasedaChemicalApparatus \
  --destination <local-public-dataset-root>
visioncortex build-public-yolo-training-view --source <extracted-root> \
  --dataset-receipt <dataset-receipt.json> --output <new-zero-copy-view>
visioncortex train-yolo-model --dataset <new-zero-copy-view> --base-model <best.pt> \
  --output <new-candidate> --epochs 60 --max-hours 1.5 --patience 15
visioncortex evaluate-yolo-model-on-human-truth --dataset <new-zero-copy-view> \
  --model <new-candidate>/weights/best.pt --split test --output <new-evaluation>
```

从已训练候选继续微调时，应显式固定优化器与学习率，避免 `optimizer=auto`
重新进入高学习率 warmup；这些参数会写入训练回执：

```bash
visioncortex train-yolo-model --dataset <mapped-21-class-union> \
  --audit-receipt <integrity-audit.json> \
  --base-model <previous-best.pt> --output <fine-tuned-candidate> \
  --optimizer AdamW --learning-rate 0.001 \
  --final-learning-rate-fraction 0.05 --cosine-schedule --warmup-epochs 1
```

ChemEq25 公开人工框数据也已固定 Figshare v3 的 RAR 工件 URL、大小、MD5 与
SHA-256；157 条多边形标注会在验证后确定性转为其最小外接框。两个来源必须先按
`configs/models/public-yolo-ontology-map.json` 显式映射进生产 21 类本体；未映射
类别逐类写明为 `null`，不能靠名称猜测。训练集可对稀缺的 hand/pipette 做有界
过采样，但 val/test 永不重复：

```bash
visioncortex prepare-public-dataset --dataset-id ChemEq25 \
  --destination <local-public-dataset-root>
visioncortex build-public-yolo-training-view --source <chemeq-extracted-root> \
  --dataset-receipt <chemeq-dataset-receipt.json> --output <chemeq-zero-copy-view>
visioncortex build-mapped-public-yolo-union \
  --source WasedaChemicalApparatus=<waseda-zero-copy-view> \
  --source ChemEq25=<chemeq-zero-copy-view> --output <mapped-21-class-union>
visioncortex audit-yolo-dataset-integrity --dataset <mapped-21-class-union> \
  --output <new-integrity-audit> --focus-classes hand,pipette
```

映射合并会在任何过采样之前按源图 SHA-256 去重；同内容跨 split 时按
`test > val > train` 只保留一个权威 split，同 split 的重复框做确定性合并。训练、
阈值校准和 TensorRT 候选实测都必须引用通过的完整性审计，发现跨 split 内容泄漏
立即 fail-closed。阈值只允许从验证集冻结；精度、召回双门槛通过后优先选择召回率
最高的候选生成阈值，随后才可读取测试集：

```bash
visioncortex calibrate-yolo-confidence --dataset <leakage-free-union> \
  --model <candidate>/weights/best.pt --audit-receipt <integrity-audit.json> \
  --output <new-val-calibration> --target-classes hand,pipette
visioncortex evaluate-yolo-calibrated --dataset <leakage-free-union> \
  --model <candidate>/weights/best.pt \
  --calibration-receipt <new-val-calibration>/threshold-calibration.json \
  --output <new-test-evaluation> \
  --test-exposure-status first_use_independent
visioncortex calibrate-yolo-world-prompts --dataset <leakage-free-union> \
  --model <yolo-world-v2.pt> --audit-receipt <integrity-audit.json> \
  --prompt-map configs/models/yolo-world-hand-pipette-prompts.json \
  --output <new-yolo-world-val-calibration>
visioncortex measure-yolo-candidate-tensorrt --dataset <leakage-free-union> \
  --model <candidate>/weights/best.pt --audit-receipt <integrity-audit.json> \
  --output <new-candidate-tensorrt-benchmark> \
  --image-size 960 --export-batch 4 --benchmark-image-limit 256
```

同一测试 split 一旦参与过候选取舍，后续模型必须显式传
`--test-exposure-status repeat_comparative_benchmark`，回执会禁止声称它仍是独立
测试；此时真正的生产泛化结论只能由尚未参与开发的内部六视角 A/B 给出。
TensorRT 候选基准固定空间尺寸与导出 batch，并显式按完整静态 batch 分块；回执
只声明该固定形状的端到端吞吐，不外推到未实测 batch 或动态分辨率。

训练与测试均在 epoch 边界强制墙钟上限，记录逐类 P/R/F1/AP、GPU/显存/功耗
遥测，且固定 `production_certified=false`。Web 的“关键素材模型质量账本”展示每个
事件实际执行的闭集 YOLO、YOLO-World、Grounding DINO、SAM2/LabPics、框置信度、
二次核验耗时与保留的不确定性；没有人工真值时不会把模型置信度冒充准确率。
已完成的公共模型候选及其按曝光状态标记的留出集指标登记在
`configs/models/public-apparatus-candidates.json`；注册不等于上线，只有本体兼容且
通过内部真实六视角认证的候选才允许写入生产配置。自动判定命令只生成不可变
决策回执，不会改生产配置；即便公开留出集全部达标，缺少经复核的真实六视角
A/B 回执也必须 fail-closed：

```bash
visioncortex evaluate-yolo-candidate-promotion \
  --evaluation-receipt <evaluation>/visioncortex-evaluation-receipt.json \
  --output <promotion>/promotion-decision.json
```

用户任务按实际提交的输入创建档案，异步提交成功后保存任务与回执身份。站点专用基准由私有任务配置提供，不作为通用安装流程。

面向化学湿实验长视频的多视角证据流水线。系统把第一人称与第三人称视频先对齐，再以两套 21 类 YOLO 模型和 ByteTrack 生成候选，经过跨视角审计后，提取实验片段、六类物理动作关键帧/关键片段/时间戳，并调用豆包多模态模型生成步骤级理解。
