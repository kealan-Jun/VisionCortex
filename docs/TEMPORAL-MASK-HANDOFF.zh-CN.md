# 有界视频掩膜交接 v1

`python -m visioncortex.temporal_mask_handoff --request REQUEST.json --config CONFIG.json --output NEW_DIRECTORY`
复用 `temporal_segmentation._load_predictor` 的官方 SAM2、已校验检查点和 CUDA/BF16 路径。
调用者提供已冻结的采样帧及起始框；本模块不发现视频、不改生产配置、不重启服务。
这是开发交接入口，尚未作为全天自动分析的新输出发布。

请求 `visioncortex-temporal-mask-request/1` 必须包含原始来源记录、已知视角角色、
train/val 分区、父结果摘要、画面尺寸和有序窗口。每窗口 1–50 帧、1–8 个起点提示框；
既有三目标请求保持兼容。
保留源帧序号、PTS/time base、片段时间与 RGB 摘要；第一帧提示只向未来传播。
不同相机身份与角色独立保留，不从设备名称推导关系。

输入使用官方加载器支持的 JPEG95，不缩放。`frames/0000/00000.jpg` 等衍生文件同时记录
文件 SHA256 和 PIL 解码 RGB SHA256，**不声称 JPEG 与原帧像素相同**。
额外、缺失、损坏、乱序、尺寸不符或未绑定的输入拒绝进入推理。

输出 `visioncortex-temporal-mask-result/1` 包含每帧、每提示实例的原尺寸二值 PNG
（0/1）、校验值、像素数、含孔洞轮廓、起点身份和 prompted/memory_propagated 状态。
空掩膜保留，confidence=null，质量状态始终是 unreviewed_model_proposals。
窗口间重置身份，不证明跨窗口/跨相机同一物体、物理接触或实验动作。
可用 `temporal_prompts.select_temporal_prompts` 对 IoU≥0.85 的互相重合框进行有界分组，
禁止依靠 A–B–C 链式重叠合并不相交的 A/C；再优先覆盖不同类别。代表框保留模型首选
标签，其他类别、框和分数随 proposal_group 全部回传；冲突始终为 ambiguous_model_proposals。
该策略减少重复几何提示，不解决真实类别，也不把嵌套物体声明成同一实物。
缺帧、身份变更、非有限值和错误尺寸使运行失败，不产生成功回执。

`visioncortex-temporal-mask-receipt/1` 绑定请求、全部输出、实现文件、模型配置和检查点。
原有关键片段连续性审计保持原接口；其小范围面积/包围框证据与这里的逐像素输出分开。
消费者应校验回执、帧身份、PNG 尺寸/像素数、种子和父结果，不把生成成功等同质量提升。
耗时包含本阶段加载、JPEG 解码、传播和 PNG 保存，不包含抽帧或其他视觉模块。

验证：输入身份/角色/分区/额外文件拒绝，孔洞与空掩膜保存、缺帧失败均有定向测试。
2026-09-24 在 3090 Ti 既有官方 SAM2 运行环境执行第一/第三人称完整片段交接；
原始结果在 LabPrism NAS 的 `evaluations/upstream-temporal-20260924`，不是标注真值。
真实视频只提供 PARTIAL_EVIDENCE；没有独立掩膜准确率或全天稳定运行的结论。
