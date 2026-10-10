# 输入与运行

[文档导航](../README.md) · [项目首页](../../README.md)

当前操作参考；执行前核对所选配置、固定源码 SHA 和适用运行回执。

## 输入清单

复制 `examples/manifest.example.yaml`，为每一路填写视角、视频和 CSV。CSV 支持常见列名：

- 帧号：`frame_index` / `frame` / `frame_id`
- 时间戳：`timestamp_ms` / `timestamp_s` / `timestamp` / `pts_time`
- 可选共同时间：`global_timestamp_ms` / `wallclock_ms`

时间戳为 ISO-8601 时也可解析。对齐阶段对每个物理分片读取有上限的首、中、尾
分布采样，验证 `clock_sync_valid`、单调性、跳时和漂移，再保存逐分片变换与误差。
系统优先使用质量最好的时钟作为内部基准，同时保留配置中的第一人称偏好；生产
任务即使已有高置信度共同时间，也执行开头、中间、结尾的轻量视觉锚点审计。
若 CSV 没有共同时间列，系统仍保留原有视频起点粗对齐能力并明确标记为
`local_timeline_assumption`；3090 Ti 正式配置要求该结果获得可靠视觉支持后才能
进入正式证据。`aligned_timestamps.csv` 使用完整基准时间轴并为每路记录
`<view>_available`，单路短录或坏分片只会被隔离，不再截断其他视角。
`alignment_runtime.json` 分别记录共享时钟采样、逐路视觉审计、分片拟合耗时和
视觉特征缓存规模，便于在真实长视频上核算新增质量检查的速度成本。
进入 GPU 队列前的 `prequeue_input_preflight.json` 同时记录媒体探测、时钟检查和
总耗时；NAS 模式只做有界元数据/时钟读取，不复制或完整哈希原始长视频。


## 运行

首次克隆的 `default.yaml` 和 `development-local.yaml` 是本地演示配置，不启动真实模型。
实际分析先准备[私有站点配置](../../configs/README.md#私有站点配置)、已核验的模型环境和自己的输入清单。

```powershell
if (-not $env:VISIONCORTEX_SITE_CONFIG -or -not $env:VISIONCORTEX_INPUT_MANIFEST) {
  throw "请设置已配置的私有站点文件和实际输入清单路径"
}
visioncortex validate-models --config "$env:VISIONCORTEX_SITE_CONFIG"
visioncortex run --manifest "$env:VISIONCORTEX_INPUT_MANIFEST" --config "$env:VISIONCORTEX_SITE_CONFIG"
```

仅在所选 GPU 环境需要导出 TensorRT 引擎时，显式执行：

```powershell
visioncortex prepare-engine --config "$env:VISIONCORTEX_SITE_CONFIG"
```

没有真实视频时可完整验证目录、JSON 契约和评估器：

```powershell
visioncortex dry-run --output .\outputs\dry-run
```

开发测试需要在同一环境安装 `".[dev]"`，随后执行 `python -m pytest`。

首次启动的本地演示服务：

```powershell
visioncortex serve --host 127.0.0.1 --port 8000
```

页面打开不表示模型、厂商和输入已经准备好；提交真实任务前，按所选部署入口提供已验证的分析配置。

浏览器先以 `POST /api/upload-sessions` 创建动态空间预留，再对
`PATCH /api/upload-sessions/{session_id}/files/{file_id}` 发送可恢复小分块，最后调用
`POST /api/upload-sessions/{session_id}/finalize` 完整校验并进入后台队列；任务状态仍
从 `GET /api/runs/{run_id}` 查询。旧的 `POST /api/runs` 一次性 multipart 接口暂时保留
用于兼容旧客户端，新页面不再使用它。API 只绑定本机，除非显式改为 `0.0.0.0`。

同一浏览器机位既可上传一个连续视频，也可按顺序上传多个原始分片；两种输入都会
转换为同一份 `RunManifest`，不会创建另一条分析链。新上传默认以 2 个文件并发、每个
文件内部顺序分块传输，每块都必须通过 SHA-256；小文件保留完整 SHA-256，大文件用
持久化有序分块哈希树封存，提交时不再从 NAS 全量重读。上传完成后、进入 GPU 队列前，
服务会校验设备角色、媒体可读性、分片顺序和 CSV 时钟覆盖，并在
`JSON-Config-Files/Input-Manifests/input_seal.json` 写入与 NAS 零复制入口一致的输入
封条。未提交的会话可用 `DELETE /api/upload-sessions/{session_id}` 取消并释放预留空间。

已完成档案不会依赖浏览器加载整份大 JSON 才能查找关键素材：`GET /api/key-events` 支持跨档案或指定档案的全文、动作类型、实验组、双视角和时间范围筛选，并通过与筛选条件绑定的 `cursor` 分页；`GET /api/key-events/{event_uid}` 返回事件及带 SHA-256 的素材引用；`GET /api/evidence/{evidence_uid}` 可一跳回到 `evidence_package.json` 的 JSON Pointer 和原视频物理分片；`GET /api/physical-changes` 查询明确观测到的对象前后状态变化，不会替 unknown 区间补状态。稳定事件 UID 格式为 `{archive_id}:{parent_event_uid}:{event_id}`；其中稳定实验组 UID 不会因前面插入其他实验而改变，原有 `GROUP-xxxx` 继续作为页面顺序编号。SQLite/JSONL 都是权威归档 JSON 的派生产物，可随时重建，不会取代原 JSON。


上传路数、单路时长和总字节数不设固定小上限；接收能力由所选存储的实际剩余空间
和活动任务预留量决定，不足时在传输前给出缺口。恢复分块、解码并发和角色内 GPU
batch 按有效配置执行，检测结果逐行落盘，已有阶段通过校验后才可恢复。
这些实现和确定性测试不能证明任意规模输入的真实吞吐、显存或稳定性；
容量与完整处理能力须由当前源码、配置和真实输入的运行回执独立验收。

6 路输入不等于 6 路输出。系统逐路做“有效实验视角门控”：只有持续手—物体操作、容器/设备状态变化或物料转移等证据达到阈值，且边界内动作密度合格的视角才会生成实验 MP4。空镜、纯走动/穿戴、等待、长静止、无实际实验操作的视角会在筛选备注中记录拒绝原因，不进入后续关键素材提取。
