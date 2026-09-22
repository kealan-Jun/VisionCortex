# LabVideo 实验室自动处理

`Z:\lab_video` 在当前 3090 Ti 主机对应 `/home/x1/桌面/nas/lab_video`。
该目录使用 `<日期>-<机位>/<时分秒>/rgb.mp4` 和原生采集元数据。
2026-09-22 接入配置沿用元数据中的设备身份与已登记视角：

| 设备 | 视角 |
| --- | --- |
| lubancat-4df661d7_cam01 | first_person |
| orangepi5pro-f022c4_cam01 | third_person |

目录名中的 `1`、`3` 不用于推断视角。新设备没有显式绑定时等待配置；
新日期目录由 `20??-??-??-*` 自动发现。已关闭的 partial 分片按实际时长处理；
仍在写入的分片等待完成标记，录音按各自设备、日期留存和识别。

## 隔离与资源

入口为 `configs/rtx3090ti-ubuntu-production.yaml` 的
`device_day.additional_labs`。LabVideo 与原实验室共享后台进程、模型池及
本地资源额度，保留相同的模型与处理参数。分别使用：

- 输入：`Z:\lab_video`
- 归档：`Z:\lab_video\VisionCortexExperimentArchive`
- 持久缓存：`Z:\lab_video\VisionCortexExperimentCache`
- 本地队列：`/srv/sentinel-data/VisionCortex3090Ti/Runtime/LabVideo`

归档内继续使用 `<日期>_<原设备编号>` 及固定五目录。
即使设备编号或会话号重名，两实验室也不会共用任务、回执、语音或跨视角分组。
不将原实验室的历史兼容回执导入新实验室。
`runtime.resource_root` 只指定共享资源额度数据库，任务和检索数据库仍在各自运行根。
未配置该字段时沿用原来的本地运行根。本地无 NAS 配置及 3050 配置不启用此实验室。

新实验室沿用当前阶段开关：归档、视频预处理、STT 自动执行；
`understanding`、`report` 暂停。采集视频软链接替换仍关闭。
监控间隔不代表处理完成延迟，也不保证排队中的历史录像立即处理完成。

## 服务与查看

`visioncortex-analysis.service` 同时管理两个实验室，修改实验室配置后需重启该 worker。
部署前检查输入元数据、存储可用空间、解释器、模型身份及当前提交。
新增网页服务不启动模型或第二套监控：

```bash
install -m 644 deployment/rtx3090ti-ubuntu/visioncortex-lab-video-web.service \
  ~/.config/systemd/user/visioncortex-lab-video-web.service
systemctl --user daemon-reload
systemctl --user enable --now visioncortex-lab-video-web.service
```

主机浏览器入口：`http://127.0.0.1:8003/#/device-days`。
网页通过 `VISIONCORTEX_LAB=LabVideo` 选择独立配置，未知实验室拒绝启动，
不能静默回退到原实验室。网页默认只监听本机；Windows 用户可直接查看上述 NAS 归档。

本地 `state/nas-recording-monitor.json` 保存发现状态；
`device-day/service.json`、阶段 SQLite 队列和 `ProgressSnapshot.json` 保存处理状态。
阶段完成回执及实际发布文件才是完成证据。确定性测试、模型实际调用、真实视频质量、
浏览器操作和稳定发布资格分别验收；自动发现或 HTTP 成功不代表完整实验分析通过。
