# 新视频观察的时序分割交接

`visioncortex.temporal_mask_handoff` 保留 request/1 的 train/val 开发回放；新增
request/2，要求显式 `data_use.purpose`。`development` 仍限 train/val；
`production_observation` 要求接收方原视频观察回执、null split、未授权训练且
不是真值。角色未知、封存测试、伪装训练用途均拒绝。结果回传相同用途，接收方核对。

执行仍复用现有官方 SAM2、固定权重摘要、每窗最多 50 帧和 8 个种子。来源、帧
序号、原生 PTS、像素/文件摘要、JPEG 派生和掩膜回执规则不变。窗口内身份只用于
模型传播；不推断跨窗口/跨相机物理身份、物理接触或实验步骤。

本修改是可调用推理接口，不改设备日五目录、自动分析队列或生产模型配置。
LabPrism 用它衔接新视频的 TensorRT 检测与分割；Web 任务全链路调用仍须另外验收。
23 项时序交接/种子选择检查通过，包含新观察用途及旧版拒绝行为、真实 PNG
回执与缺帧失败。确定性检查不代表真实分割精度。
