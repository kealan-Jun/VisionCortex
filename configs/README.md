# 配置目录

[项目首页](../README.md) · [加载实现](../src/visioncortex/config.py)

公开配置不包含站点设备、凭据引用或生产存储信息。`default.yaml` 与
`development-local.yaml` 提供首次安装的本地页面和合成契约示例：使用 CPU，关闭
多模态调用、语音识别、NAS 同步及自动采集；输入、缓存、暂存和归档均位于本地
运行根中。演示模式经过环境覆盖后仍检查这些边界，配置冲突会拒绝启动。

| 类型 | 配置 | 用途 |
| --- | --- | --- |
| 首次使用 | `default.yaml`、`development-local.yaml` | 本地页面和合成数据，不是实际模型分析 |
| 硬件参数 | `*-base.yaml` | 显式选择模型与性能参数；不提供权重或站点身份 |
| 本地分析 | `*-local.yaml` | 配置本地工作目录，实际分析前验证依赖、模型和输入 |
| 部署模板 | `*-production.yaml`、兼容 profile | 未配置模板，默认关闭后台生产功能；须明确提供私有站点配置 |

文件名中的 `production` 表示配置用途，不表示该文件已经完成部署。模型路径采用
相对路径；仓库不携带模型权重、实际采集清单、真实评估记录或现场设备注册表。
设备注册表默认为空；[示例注册表](../examples/device-registry.example.json)仅说明结构。

## 私有站点配置

加载顺序为：基线 `default.yaml` → 明确指定的 profile 及其 `extends` → 显式的
`VISIONCORTEX_SITE_CONFIG` → 支持的环境覆盖。`extends` 仅允许引用同目录文件，
路径越界或循环会拒绝加载。站点文件不存在时直接失败，不创建替代文件。

```bash
export VISIONCORTEX_SITE_CONFIG="$HOME/.config/visioncortex/site.yaml"
export VISIONCORTEX_DEVICE_REGISTRY="$HOME/.config/visioncortex/devices.json"
visioncortex serve --config configs/rtx3090ti-ubuntu-production.yaml
```

站点文件应定义实际存储根、模型位置、厂商设置和设备规则。生产用途明确设置
`project.run_purpose: production`、`project.site_configuration_required: false`；
需要 NAS 生产者时同时设置 `runtime.local_only: false` 并按实际契约核验挂载。
这不会自动证明部署、模型质量或链接清理门禁。密钥只通过环境变量或批准的未跟踪
私密存储提供，不能写入 YAML、设备注册表或评估夹具。

## 保留既有部署

已有部署继续使用其固定提交、原配置、注册表和回执；清理公开模板不迁移现场
路径，也不重启服务。迁移到新版本时，先在部署方私有位置保存并校验原始配置
全集、同目录继承文件、设备注册表和评估数据，保留原身份及可恢复副本。

如需加载原始完整配置而非新的公开基线，可以明确指定私有基线：

```bash
export VISIONCORTEX_DEFAULT_CONFIG="$HOME/.config/visioncortex/frozen/default.yaml"
export VISIONCORTEX_DEVICE_REGISTRY="$HOME/.config/visioncortex/frozen/devices.json"
visioncortex serve --config "$HOME/.config/visioncortex/frozen/profile.yaml"
```

使用此方式不会把公开基线偷偷合入冻结配置；仍应用明确的站点覆盖和环境覆盖。
不要同时遗留无关的 `VISIONCORTEX_SITE_CONFIG`。启动前比较完整有效配置及模型、
源代码和归档身份，并取得适用的真实运行证据。历史记录不作为新版本运行成功的证明。
