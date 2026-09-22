# LabVideo 采集目录接入

`Z:\lab_video` 在当前 3090 Ti 主机对应 `/home/x1/桌面/nas/lab_video`。
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

运行配置 `configs/rtx3090ti-ubuntu-production.yaml` 中：

- `collection_ingest.additional_camera_directories` 将 `lab_video` 加入现有监控。
- `directory_camera_bindings` 显式将日期／视角目录映射到处理机位。
- `camera_role_map` 指定模型视角，`camera_group_map` 指定同一实验室的配对范围。

新的日期目录自动发现。未登记的视角目录等待绑定，不继承其他实验室的同名设备身份。
重新检查输入及断点恢复应用相同映射。已关闭 partial 分片按实际时长处理；
仍在写入的分片等待完成标记。录音与真实 STT 结果保留在自己的设备日目录。

当前阶段开关为：归档、视频预处理、STT 自动执行；`understanding`、`report` 暂停。
目录命名不影响这些阶段的算法或输入契约，阶段暂停属于全局运行设置。
采集视频软链接替换仍关闭。

后台服务仍为 `visioncortex-analysis.service`。接入后在原页面的 NAS 监控及设备日
列表中查看 `lab-video-1`、`lab-video-3`。发现状态记录在原本地运行根的
`state/nas-recording-monitor.json`，处理回执仍在 `device-day` 的既有阶段队列。
监控间隔不是处理完成时延；模型调用、真实视频质量与完整链路分别以实际回执验收。
