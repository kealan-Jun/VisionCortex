# 项目标注的本地检测器训练与复跑

此入口消费独立 AnnotationWorkbench 的带哈希导出，数据保持 `project_annotations`、`independent_ground_truth=false`。它不改变原有独立人工真值训练、稳定发布和正式归档门禁，也不自动替换生产权重。

先通过工作台的全项目或明确训练批次检查，冻结图片/标签/分区，审查近重复与类别覆盖，再启动实际模型。第一、第三人称分别运行。每个实验必须指定新目录、导出回执SHA和原模型SHA；源文件、导出目录中的图片/标签清单、标签版本及分区会再次核对。

`--freeze-layers`可在0–10之间明确设置冻结前N层，默认10保持原训练范围；0允许主干参与训练。仅分类输出的探针使用自身固定范围，不接受另设冻结数量。回执的`parameter_training_scope`在框架和项目限制均应用后记录实际参数名、可训练状态与数量，不能只看传入的freeze值。此含义依据[Ultralytics训练参数](https://docs.ultralytics.com/modes/train/)及本机8.4.28源码核对；冻结策略同时影响相应归一化层的训练状态，单次对照不能把差异只归因于权重或BN。新增选项不代表推荐解冻。

goal23已完成相同v14数据、80轮与学习率.001的冻结10层/解冻对照，各320次实际更新；解冻候选在3张整图的已知匹配由78/106退至68/106。两张验证裁剪没有手套、单支枪头、枪头盒和搅拌子的正例，不能用于证明这些类已达标。新候选未晋升；详细回执见迭代记录。

goal24已完成显式部分监督v2离线增强对照，原16输入/80轮对16父图加64派生/16轮，各320更新。固定整图已知匹配78/106→84/106，但验证裁剪误检1→3，深色衣料出现手套误报，单支枪头和搅拌子仍漏检。只证明局部收益，完整质量未达标。

```bash
python tools/train_project_annotations.py \
  --export /absolute/workbench-export \
  --receipt-sha256 REAL_RECEIPT_SHA256 \
  --role first_person \
  --base-model /absolute/first_person/best.pt \
  --model-sha256 REAL_MODEL_SHA256 \
  --output /absolute/new-experiment \
  --epochs 80 --image-size 960 --batch 4 --max-seconds 3600 \
  --preserve-class-head --learning-rate 0.0001
```

上面的SHA是必须替换的说明占位。使用已核对的GPU解释器执行，不自动安装依赖或下载模型。训练目录独立保存Ultralytics设置，禁用自动安装与联网；关闭在线增强，离线增强来自工作台的可追溯导出。当前冻结前十个骨干层参数，使用FP32、AdamW、固定随机种子及验证早停；有效配置保存在 `experiment.json` 与训练器的 `candidate/args.yaml`。

小样本实验可显式设置 `--nominal-batch`、`--warmup-epochs`、
`--warmup-bias-lr` 和 `--patience`，默认仍为 64、3、0.1、20，保留旧实验行为。
`nominal-batch` 是梯度累积基准，并非每批实际读取的图片数；设成与 `--batch` 相同可避免跨批累积。
本机框架在启用预热时至少预热 100 个批次，少量图片的 3 个预热 epoch 因此可能实际跨越很多 epoch。
`--warmup-epochs 0` 关闭预热；`--patience 0` 关闭验证早停，仍受 epoch 和时间上限约束。
这些是对照实验参数，没有根据验证分数自动调整，也不自动改变原有配置。

每次新训练在 `optimization-trace.jsonl` 记录实际 optimizer step、所在 epoch、梯度累积数和各参数组学习率；
`experiment.json.optimization_usage` 保存真实更新次数和文件摘要。
累计看图次数、训练批数、epoch 数不能代替参数更新次数。新回执的失败后恢复也必须通过该记录重验；
旧实验没有这份记录时不能倒填为当时已实测。该记录只证明训练确实更新，不证明检测效果。

新训练的 JSON 回执先写临时文件、执行文件 `fsync`、原子替换；POSIX 平台还同步父目录。权重保存和训练结束时的优化器剥离使用框架原有逻辑，但先写入独立暂存目录；成功落盘后才替换 `last.pt`、`best.pt` 或周期权重。写到一半的进程退出不会先截断已发布的旧权重，失败暂存文件保留。

每次保存权重后，`training-evidence/epoch-NNNN/` 独立保留优化轨迹、采样计划/轨迹（适用时）、参数与 CSV 快照及损失消费计数。`finalized/` 记录框架最终验证和权重处理后的状态，`latest.json` 绑定具体回执。完成前逐一检查全部快照、连续 epoch、实际更新数、最终原始轨迹和权重摘要；部分监督还核对采样 epoch 数。失败回执恢复不能跳过这些新记录。新增证据模块的导入时源码也进入运行归档。

这些文件提供中断后的可验证产出，不意味着整组权重与 JSON 是一个文件系统事务。若在多个文件发布之间断电，记录不一致必须继续失败；不把最新 JSON 的存在当作完成。历史 epoch 回执中的权重摘要记录当时身份，实际权重仍按原有 best/last/周期策略保留，不宣称每一轮权重永久存在。Windows 只声明文件同步和原子替换，不声明 POSIX 目录同步。捕获到 `KeyboardInterrupt` / `SystemExit` 时状态为 `interrupted`；强制杀进程或主机断电可能留下 `training` 状态，须先确认进程已停止，再检查保留证据。当前不支持把部分监督训练从中间 epoch 自动续训；修复后使用新目录重新运行，保留中断目录。损坏的旧轨迹不倒填为完整训练。

流程实际执行旧模型预测、候选训练、候选预测和内部评测。两次预测使用相同原图、960输入、0.001预测底限、请求IoU参数0.7、1000个最大目标；指标在0.25置信度及IoU0.50–0.95计算，并保留逐项错例和按来源组结果。当前YOLO模型使用end-to-end直接预测，Ultralytics在此模式跳过NMS；设置IoU参数不证明执行了去重，旧回执里的 `nms_iou` 只表示传入参数。具体值随入口参数改变并记录，不与不同阈值的训练日志直接混比。旧第一人称类别10的 `tube-cap` 显式对应稳定ID10 `tube_cap`；其他类名不一致即失败。

扩充类别时，默认框架会跳过形状变化的分类输出层。`--preserve-class-head` 仅在已核对的旧/新类别映射下拷贝旧类别行，新类别行保留初始化值；普通分支和one-to-one分支、模型及EMA同步处理。回执逐项列出转移参数，不把其他形状不合的参数强行拼接。

`--training-scope class_outputs` 用于单独验证分类适配：只更新已核对的最终
1×1 分类输出卷积，固定特征层、定位分支及全部归一化统计量。训练前和训练后
校验其余状态摘要；每批恢复 BN 的 eval 状态，但检测器整体仍使用训练输出。
每轮验证和保存前同步 EMA 的固定部分，分类输出继续保留 EMA。
这与默认 `detector` 模式分开记录，不自动启用或替换生产权重。
本机 8.4.28 框架本身已经冻结默认 freeze=10 范围内的 BN 统计量；
不能把此探针说成修复此前主干 BN 未冻结的问题。

第五轮的 `--migrate-legacy-tip-box` 是明确的初始化实验：经真实预测与项目标注核对，旧类11的高分误检实际覆盖整盒枪头，因此把旧类11参数初始化到新类22（枪头盒），类11（单支枪头）恢复新初始化。其他旧类别仍按原ID保留。它不修改标签或宣称单支枪头质量通过，仍须固定验证图比较及补充细粒度样本。

`--supervision outside_ignore` 是单独的部分监督实验，默认 `complete` 路径拒绝这种数据。
它只接受工作台 `export-partial-training` 的独立回执：已知框保持项目标注，未解决区域逐图保留；只有训练图可处于 `reviewed_partial/all_visible_outside_ignore`，验证/测试仍完整复核且无 ignore。
适配器使用与已知框相同的 Instances/LetterBox 变换，忽略区内 anchor 的负分类项不参与损失，已知正分类项及原有定位/DFL损失保留；普通和 one-to-one 两分支均应用。
当前只验证 Ultralytics 8.4.28，禁止在线增强、多尺度、类别筛选和图编译。默认部分监督v1仍禁止离线增强；显式v2见下文。缺失每图区域元数据或尺寸变化直接失败。
`experiment.json` 的 `ignore_loss_usage` 记录每分支调用、忽略 anchor 访问和保留正项数；这些是重复训练访问次数，不是新样本数量。
部分监督复跑还须保留实际损失调用记录，不能只凭预测文件补成完成。

工作台显式离线增强导出采用 `annotation-workbench-partial-training-export/2`，
`supervision.augmentation=offline_affine_ignore_regions/1`。训练入口仍须显式 `--supervision outside_ignore`。
模型启动前逐个核对派生图的父图版本、原图和原标注哈希、训练来源组、角色、逐件实物与未知区域、
完整保留画面的缩放填边矩阵、变换后标签哈希与实际YOLO标签。未登记、掉框、丢未知区域、错位区域、
验证图作为训练父图或不兼容旧v1回执均被拒绝。加载时继续通过同一 LetterBox 变换已知框和未知区域，
两个分支仍保留已知正项，仅抑制未知区内负分类项。不能用普通训练器忽略区域文件。
导入时源码归档包含 `project_augmentation.py`。这些检查证明消费契约，不能单独证明增强视觉质量或模型增益。
这是解决部分器材不清晰时利用整图已知区域的训练方式，不能替代完整标注、独立验证或系统效果验收。

部分监督入口可显式使用`--sampling-policy source_balanced`做来源分布对照。
默认`image_uniform`保留框架逐图打乱方式；平衡模式令每张训练图的抽样权重为其来源组图片数的倒数，
各来源组总抽样权重相等。每epoch仍抽取原训练图总数，允许重复，因此单个epoch可能未访问某张图；
期望权重相等不等于每轮实际访问数相等。独立随机生成器使用记录的seed，不修改标签、原始分组或验证/测试图。
当前仅支持已验证的部分监督v1或显式离线增强v2、单GPU、新训练；不支持DDP、矩形批次或训练断点恢复。
增强条目使用独立文件ID但保留父图来源组；采样中的 `unique_images` 包含这些派生文件，不代表新增独立样本。

两种方式均保存`candidate/source-sampling-plan.json`和`source-sampling-trace.jsonl`，
后者在图片真正进入训练预处理时记录，未消费的预取图片不计数。
`experiment.json.sampling_usage`包含文件摘要、每来源/每图/每epoch访问次数以及独特图片数。
这些是实际重复访问记录，不是独立新数据，也不是效果提升证明。
训练结束和失败回执恢复时重新检查来源成员、完整epoch及摘要；缺失或更改记录不能补称平衡训练完成。

主要文件：`experiment.json`、`baseline-predictions.json`、`candidate-predictions.json`、`baseline-evaluation.json`、`candidate-evaluation.json`，以及 `candidate/weights/best.pt`、训练日志/CSV。数据、模型、代码文件和解释器身份分别保留；新运行还归档导入时的训练/评测源代码，第四轮补存内容已核对启动哈希。前几轮只有代码哈希，未宣称这些未提交源码版本已全部可重建。类别不足20个验证正实例、验证裁剪、旧模型接触原始数据等限制必须另报；总体指标改善不允许掩盖逐类回退。所有候选默认 `production_ready=false`。

如果训练和预测已经完成、只有最终回执检查失败，先修复实际错误，再调用 `visioncortex.project_annotation_training.recover_evaluated_experiment(Path('/absolute/run'))`。它重新校验冻结导出和权重、从原预测重新计算指标并核对原报告，保存原失败回执，成功后补全结果，`retrained=false`。文件缺失或内容不一致会继续报错；不能把尚未完成训练的目录伪装成完成。其他失败保留原目录，修复后建立新实验，不覆盖失败证据。

训练和局部评测完成是实际模型调用证据；完整视频、困难背景、机位对应、关键素材质量、推理引擎差异、浏览器播放和正式语义归档仍各有独立验收步骤。

### 候选 PyTorch 分支的系统对照

产品扫描入口可显式配置`models.prediction_branch_by_role`，按角色选择`one2one`或`one2many`，例如：

```yaml
models:
  prediction_branch_by_role:
    first_person: one2many
    third_person: one2many
  duplicate_suppression_iou_by_role:
    first_person: 0.7
    third_person: 0.7
performance:
  tensor_rt: false
```

这是需合并到已验证本地候选配置的片段，不是完整运行配置。仍须提供各角色已核对的候选PT路径、23类映射、输入/输出根及其他性能参数；不得直接套用到生产。显式分支要求PT及同时含两分支的检测头，推理后核对后端实际分支。引擎不能在运行时切换，需另外导出和验收。未显式配置的角色保持原行为。最终坐标去重发生在跟踪前，保留原始框、移除/保留关系及IoU，不把重复关系当作物理实例确认。

显式候选的断点绑定权重路径/SHA、分支、confidence/IoU、图尺寸、最大框数、精度和预期类别数。身份改变时拒绝复用旧目录，在模型加载和旧账本截断前报错；尚无完整块的断点也保护。新配置使用新目录，旧结果保留。该身份保护不等于完整输入/环境/PTS验收，也没有给未配置的旧生产断点倒填身份。

goal21已在本地两段约95秒真实派生视频上运行两分支，各1520源帧，核对原像素和去重审计，并验证同配置复跑无源图推理。实看仍有多枪共框、手持枪漏检及盒体部件重复，因此仅为候选实际调用与局部质量证据，不能按框数或耗时晋升。

后续goal22在三张密集斜置枪的整图上发现，最终0.7去重后已知一对一匹配减少。上面的配置仅用于重现实验，不是候选或生产推荐；原始框应继续留证，不能把框数减少当作质量通过。原生分支还存在枪类退步及重要小目标/手套缺失，分支选择和去重效果分别验收。
