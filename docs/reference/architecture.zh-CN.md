# 目录与模块职责

[文档导航](../README.md) · [发布与部署](../RELEASE-POLICY.md)

本项目保持单一 Python 包、既有进程和持久队列。领域服务通过显式参数组合；旧导入与脚本入口提供兼容，算法在所属模块维护。

| 路径 | 责任与边界 |
| --- | --- |
| `src/visioncortex/analysis/` | 逐帧观察、候选、运动召回、跨视角审查、分段、最终事件同步和扫描规划/执行 |
| `src/visioncortex/media/` | 实验片段、关键素材、参与对象定位；模型/写入端口由调用方传入 |
| `src/visioncortex/evidence/` | 轻量布局、序列化、归档缓存、结构产物、语义复核/修订和最终封包 |
| `src/visioncortex/application/` | RuntimeHost、任务提交尾段、上传适配、读模型、采集编排和素材修复 |
| `src/visioncortex/web_api/` | 按上传、任务、档案、采集、设置、标注、健康、事件及 Receiver 划分 HTTP 路由 |
| `src/visioncortex/web/` | 浏览器模块；`app.js` 装配，uploads/archives/materials/tasks 各持有领域行为，stores 持有唯一状态 |
| `src/visioncortex/training/` | 公共数据/已审核数据、训练执行、留出评测及项目 cohort、类别头、溯源、实验和恢复 |
| `src/visioncortex/device_day*.py` | 冻结设备日契约与阶段适配；旧归档读取和原位发布边界继续保留 |
| `tools/{packaging,diagnostics,evaluation,training,desktop}/` | 工具按使用职责组织；根目录同名脚本保留入口 |
| `configs/` | 中性硬件 base、生产/本地配置和模型/厂商目录；profile 文件名兼容现有部署 |
| `tests/` | 领域行为与兼容验证，见[测试职责](../../tests/README.md) |
| `docs/{guides,operations,reference,history}/` | 使用指南、运维、当前参考及历史证据索引 |
| `deployment/`、`release/` | 各平台交付和正式发布门禁，独立于代码同步 |

保留的兼容入口包括 `actions.py`、`archive.py`、`api.py`、`yolo_training.py`、`project_annotation_training.py` 和根目录工具脚本。新代码使用所属领域模块；兼容入口只装配命名端口和转发公开符号。2027-01-31 复核这些入口的实际调用方与迁移覆盖；复核日期不会自动删除接口。

共同时间窗口合并由 `analysis/windows.py` 维护，类别规范化由 `action_semantics.py` 维护。离线流水线与设备日多视角适配共用 `analysis/experiments.py` 的审查后实验组合；各自的时钟、track namespace、FrameWindows、完整时间线策略与回执仍由来源适配负责。

`source_identity.py` 按包相对路径递归记录 Python 文件。CV 缓存和设备日扫描身份使用 schema 3，阶段恢复请求使用 schema 2；新执行键拒用旧执行缓存，重新计算时将旧回执原样保留在历史中。构建身份包含 Python/Web 资源；执行、CV 和训练来源有各自依赖范围。未知子包变化保守重算，子目录同名文件不能继承顶层排除规则。

以下实现继续独立：

- DurableRunQueue 管理整次任务的租约和重试；DeviceDayQueue 管理分片修订、阶段可用性与机位公平性。
- Web 身份/LAN 角色与机器 Bearer/scope/预留准备有不同边界。
- 流式候选与历史累计候选、设备日与离线归档、正式/阶段/观察报告保留自身策略与质量门。
- `training/ignore.py` 是显式可选模型实现，兼容 adapter 延迟导入；正常服务与确定性测试不提前加载模型栈。
- 包构建共享安全路径、哈希及固定 Git 导出，但候选包、便携包和正式发布各自检查证据。

组织方式参考了 [FastAPI 的 APIRouter 分域](https://fastapi.tiangolo.com/tutorial/bigger-applications/)、[Immich 的服务职责](https://docs.immich.app/developer/architecture/) 和 [Home Assistant 的组件边界](https://developers.home-assistant.io/docs/architecture_components/)。本项目采用适合现有运行方式的模块边界，后续拆分依据真实依赖和测试覆盖。
