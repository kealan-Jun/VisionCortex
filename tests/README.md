# 测试职责

在已有开发环境中，从仓库根目录运行 `python -m pytest tests/<领域>`。
使用 `repo_paths.ROOT` 定位仓库资源，`default_config` 由顶层 `conftest.py` 提供。

| 目录或测试族 | 验证范围 |
| --- | --- |
| `architecture/` | 递归源码身份、缓存隔离、轻量依赖和兼容入口 |
| `test_actions*`、`test_motion*`、`test_group*` | 动作观察、候选、审查与分组 |
| `test_archive*`、`test_key*`、`test_material*` | 素材、回执、归档与发布 |
| `test_device_day*`、`test_inplace*` | 冻结设备日契约、独立阶段和原位发布 |
| `test_api*`、`test_run*`、`test_machine*` | HTTP 适配、应用服务、任务与机器接入 |
| `web/`、`test_web*` | 独立前端模块接口和浏览行为字符化 |
| `training/`、`evaluation/` | 训练资格、数据来源、人工真值及评测边界 |
| `packaging/`、`desktop/`、`diagnostics/`、`test_rtx*` | 固定源码包、桌面连接、部署入口与配置兼容 |

新测试按领域放入子目录。现有扁平测试仍有共享 fixture 导入，随相关改动逐组迁移，保持测试 ID 和共享 fixture 的可追溯性。旧前端字符化使用 `repo_paths.web_source()` 查找跨文件函数；新增测试直接实例化模块的公共 factory 和显式依赖。

默认回归使用自有临时文件、假模型端口和确定性数据，不访问生产 NAS 或付费模型。明确依赖可选模型库的测试在库缺失时跳过，不能用跳过记录推定模型运行成功。通过仅证明该测试范围；真实视频效果、浏览器交付与发布资格各自需要适用证据。
