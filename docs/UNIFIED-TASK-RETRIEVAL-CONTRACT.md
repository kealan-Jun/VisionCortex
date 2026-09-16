# 统一任务、检索及回传契约

离线视频分析和 NAS 自动采集共用任务交付及检索层。设备日目录继续是
`MetaVideo`、`ProcessedClips`、`MultimodalUnderstanding`、
`LaboratoryDailyReport`、`Comment`；本次不新增 NAS 目录、不修改素材筛选算法。
运行数据库、进度快照、索引及问答回执保存在配置的本地 runtime 根目录。

## 提交、防重与恢复

分析入口接受 `Idempotency-Key`（1–200 位字母数字、下划线、点、冒号、短横线）。
同一登录身份、接口、键、请求内容只启动一个提交；重复请求返回原响应。
同键不同内容返回 409。JSON 字段顺序和 multipart boundary 不影响内容身份。
未提供键的创建请求按内容、当前构建及模型配置防重；问答还包含证据索引版本。
需要有意重跑同一输入时提供新键。相同 HTTP 请求重传应始终复用原键。

`retry` / `refresh` 无键请求保持原有队列版本 CAS 行为，不永久缓存用户的一次重试。
调用方可为一次明确的重试操作提供键。待提交状态不会因超时而擅自再启动任务；
若服务在落任务与绑定回执之间崩溃，保留待核对状态，不声称严格 exactly-once。

NAS 队列继续按实际相机、源版本和阶段管理。`input_status` 是独立的输入可用性：
`ready`、`missing`、`waiting`、`unavailable`。缺失、未写完、读取异常分别展示，
不计入可运行积压，也不能写成“无实验活动”。后台独立小批量检查与既有归档恢复
流程处理输入重现；成功保留的 MetaVideo 原片仍可使用，不要求采集端再次出现。

## 进度与事件

工作进程每 5 秒独立发布本地 `device-day/ProgressSnapshot.json`。
HTTP 读取最近成功快照；刷新慢或失败时保留旧快照并返回年龄、过期标记和错误类别。
首次尚无快照时 `/api/device-day-progress` 返回 202 `progress_initializing`，
不把取进度超时当成分析服务故障。模型/存储真实故障仍保留其错误状态。

阶段、已处理帧数和阶段耗时通过本地 SQLite 跨进程可见。它们是运行观察，不能
代替产物验收。任务入队、认领及终态与 outbox 同事务提交；独立汇聚器每 2 秒补取。
本版本以前的状态仍由原进度接口展示，不伪造此前事件时间。

- `GET /api/task-events?after=0&limit=100`：按全局单调序号增量读取。
- `GET /api/tasks/{recording_id或run_id}/history`：查询任务历史。
- NAS 提交返回 `recording_ids` 和批次 `status_url`，批次历史汇总其分片事件。
- `GET /api/task-events?consumer=MyAgent`：读取该登录身份下消费者尚未确认的事件。
- `POST /api/task-events/consumers/MyAgent/ack`，JSON `{"cursor":123}`：
  确认已交付游标，超出已交付范围返回 409。断线和重启后仍可补取。

语义为 **at-least-once + ACK**，消费者按 `event_id` 去重。确认全部本地处理完成后
再 ACK，不能把刚收到 HTTP 200 当成业务处理成功。

可选 `runtime.result_callbacks` 每项包含 `name`、`url`（HTTPS）、`secret_env`。
默认空列表，不自动向外部系统发送。只有部署管理员配置的接收方会被调用；密钥
从环境变量读取。请求头携带 `X-VisionCortex-Timestamp` 和
`X-VisionCortex-Signature`，签名为 HMAC-SHA256(secret, timestamp + '.' + 原始请求体)。
接收方校验签名、时间窗口并按事件 ID 去重，成功回复
`{"acknowledged_cursor":收到的next_cursor}`。失败指数退避，最多间隔一小时；
重试通知不会重新运行视频或模型。

## 统一检索与 RAG

应用左侧“证据检索与追踪”同时访问旧离线证据索引和当前设备日索引、理解、日报、
comment 及当前索引引用的真实 STT。后台跟随任务完成事件更新，每五分钟巡检补漏；
HTTP 检索不遍历 NAS、不加载 GPU、不调用模型。

- `GET /api/knowledge/search?q=移液器&day=2026-09-09&camera=...`：支持 kind、limit、offset。
- `POST /api/knowledge/ask`：`{"question":"问题","day":null,"camera":null}`。
- `GET /api/knowledge/answers/{id}`：保存的问题、证据版本、逐条引用、模型用量及构建身份。
- `GET /api/knowledge/evidence/{id}?revision=...`：来源、时间、素材引用和证据状态。

这是基于归档文本的词法检索增强回答，未声称部署 embedding 或向量召回。
只有主动点击问答才调用现有配置的模型，并使用共用的并发限额及厂商熔断。
单次问答最多一次传输尝试，无证据时不调用。逐条结论必须引用召回集合中的 ID；
未知引用或源版本变化会拒绝接受答案。陈旧/不可读资料不作为当前证据。
引用格式通过不等于模型结论正确；回答保持 `PARTIAL_EVIDENCE`，保留不确定性，
不能把单视角候选动作或伪标签升级为人工真值。

## 采集写入到预处理时延

接收端完成上传并关闭文件后，提交：

```http
POST /api/receiver/upload-completed
Content-Type: application/json

{"recording_id":"已发现的分片ID","source_signature":"该版本签名","completed_at":1789530000.125}
```

`completed_at` 是 Unix 秒，来自接收端真实完成时刻。需要使用正常写入权限，不能
把管理员凭据印入设备代码。分片版本尚未发现或不匹配返回 409，发现后重发同一回执；
不允许替换已有时间。接收端和工作站需保持时钟一致，负时延返回未测量。

API 记录上传完成→发现、上传完成→预处理启动、上传完成→预处理落盘；与历史补跑、
启动盘点、实时发现分开。原有发现→可处理→归档→YOLO→落盘计时继续保留。
接收端尚未发回执的分片不填造上传完成时间；接口实现不能当成接收端已接通的证据。

## 发布边界

API 与工作进程共用构建身份：提交 SHA、源码摘要、工作树状态；正式包继续使用
BuildManifest。源码运行会明确标注 development_checkout，不据此宣布正式发布验收。
现有夜间多模态窗口和动态相机调度不变。本次不启用采集端删除或软链接替换。

确定性检查覆盖并发提交、不同 multipart 封包重传、事务回滚、过期所有者、ACK、
回调签名、双布局索引、引用失效、无证据不调用模型、时延回执版本绑定和断进度刷新。
这些检查与真实模型调用、真实视频质量及外部 receiver/callback 的实际接通分别记录。
