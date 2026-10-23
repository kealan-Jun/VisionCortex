# LabVideo 采集目录接入

按用户 2026-09-22 明确的目录含义，日期后的 `1`、`3` 分别表示第一人称、第三人称。
这是输入命名的映射；检测、录音识别、多模态理解与日报沿用现有处理流程。

| 输入目录 | 处理机位名 | 视角 | 统一归档位置 |
| --- | --- | --- | --- |
| `lab_video/<日期>-1/<时分秒>/` | `lab-video-1` | first_person | `VisionCortexExperimentArchive/<日期>_lab-video-1/` |
| `lab_video/<日期>-3/<时分秒>/` | `lab-video-3` | third_person | `VisionCortexExperimentArchive/<日期>_lab-video-3/` |

两个设备日目录内部均保持：

```text
MetaVideo/
ProcessedClips/
MultimodalUnderstanding/
LaboratoryDailyReport/
Comment/
```

沿用原来的网页入口、NAS 总览、任务队列、持久缓存与模型池。
不在 `lab_video` 采集目录内创建另一套归档，也不需要另一个网页服务。
原设备编号保存在 `recorder_camera_key` 和原生元数据中；视频时间、录音及 CSV 不改写。
同一来源的第一／第三人称继续进入现有跨视角分析；不同实验室不会仅因会话号或时间相同而配对。

历史归档的完整原片时间仍取自对应的原生元数据。读取物理设备号与
`lab-video-1/3` 处理机位不同时，必须有同一原片的视频、元数据来源描述共同指向
相同的日期／视角／采集目录，且元数据字节校验值与原留存记录一致。
输出 JSON 保留 `recorder_identity_binding` 说明两种身份的关系；这不更改归档机位名，
也不允许将普通设备日中的其他设备元数据用于命名。

运行配置 `configs/rtx3090ti-ubuntu-production.yaml` 中：

- `collection_ingest.additional_camera_directories` 将 `lab_video` 加入现有监控。
- `directory_camera_bindings` 显式将日期／视角目录映射到处理机位。
- `camera_role_map` 指定模型视角，`camera_group_map` 指定同一实验室的配对范围。

新的日期目录自动发现。未登记的视角目录等待绑定，不继承其他实验室的同名设备身份。
重新检查输入及断点恢复应用相同映射。已关闭 partial 分片按实际时长处理；
仍在写入的分片等待完成标记。录音与真实 STT 结果保留在自己的设备日目录。

当前付费视频理解按用户要求暂停；归档、视频预处理、STT 和已配置的发布流程继续执行。
视频预处理每机位允许两个独立分片并发，共用原模型池和全局资源上限；
同一分片仍通过队列租约及文件锁防重，设备日索引继续原子发布。
目录命名不影响这些阶段的算法或输入契约，阶段暂停属于全局运行设置。
采集视频软链接替换仍关闭。

新采集使用独立的 `visioncortex-live-discovery.service`，其 `metadata_only`
范围不包含旧的 `lab_video` 目录。显式启用的旧来源通过
`python -m visioncortex.device_day_legacy_ingest --config <已验证的运行配置>` 接入，
不扩大新采集监控的范围，也不启动模型。配置中的 `device_day.legacy_ingest`
必须声明 `enabled: true`、`directories: [lab_video]`，以及实际留存执行器的
`retention_source_root`、`retention_source_sha`、`retention_source_digest`。
留存执行器固定为生产原有不可变源码；仅通过有界留存端口处理明确绑定的旧来源。
端口先校验来源目录、视角、NAS 身份和源文件，再以既有校验复制算法补齐
该设备日的新留存文件，原文件不移动、不删除、不改写。全局原片只读设置不变。
留存完成后才写入既有观察清单。该次用户要求补处理的精确来源通过原生
`current_recovery` 队列入口恢复 CV 准入，避免全局实时历史选择器一直等待空闲；
来源时间、排序、租约、输入验证和重试预算仍由原有 worker 校验和执行。
此入口不准入付费视频理解。其余 STT、日报和发布也由既有服务继续处理。
已完成输入通过来源版本检查点复用；失败只延后对应原片。扫描按最新时间优先，
默认每完成一段就重新扫描新日期和新增分片，空闲时每 60 秒检查；
`max_imports_per_turn` 可显式设为 1–8，默认 1，避免整批历史留存阻挡新输入。
状态保存在本地
`device-day/LegacyIntake/Service.json`，不会覆盖实时监控的调度状态。

旧来源最多检查三层目录，兼容 `<日期>-3/<日期>/<时分秒>` 的额外日期层；
该布局仍须配置独立的 `directory_camera_bindings`。用户已确认
`lab_video/2026-10-20/<时分秒>` 为第三人称，使用精确日期规则绑定
`lab-video-3`；其他没有视角的日期不自动继承这条绑定。
查阅入口仍是一段原片对应一个完整起止时间目录，多个筛选区间放在目录内部，
模型理解目录与之对应。目录身份与完整时间必须同时通过来源视频、原始元数据
及其留存校验值验证；显式绑定不能绕过字节校验。公开结果不用 HTML。

接入后在原设备日列表查看 `lab-video-1`、`lab-video-3`，
处理回执仍在 `device-day` 的既有阶段队列。
监控间隔不是处理完成时延；模型调用、真实视频质量与完整链路分别以实际回执验收。
