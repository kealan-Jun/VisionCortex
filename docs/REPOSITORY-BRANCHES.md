# 仓库分支与历史归档

2026-09-09 按用户要求统一客户更新仓库，并整理历次遗留分支。

## 当前日常入口（2026-09-16 更新）

两个仓库均以 `main` 为日常代码入口，使用相同提交：

- `https://github.com/kealan-Jun/VisionCortex.git`
- `https://github.com/RealityLoopAI/VisionCortex.git`

历史分支的独有能力逐项整合到同一套 `visioncortex` 实现。离线分析与 NAS 自动
处理继续共享算法和运行层，不要求切换到某个硬件专属分支。同步代码不等于
部署或真实全流程验收；仅保留手动 CI，不因代码同步触发检查。
详见 [双仓同步规则](DUAL-REPOSITORY-RELEASE-POLICY.md) 和
[历史能力整合记录](HISTORICAL-BRANCH-INTEGRATION-20260916.md)。

以下内容是 2026-09-09 的历史记录；分支整理、删除及测试结果仅描述当时操作。
本次没有删除旧分支、标签或改写历史。

## 历史入口整理

清理前核对开放 PR 为空。已完整合并的分支删除入口后，其全部提交仍可由现存分支到达。
仍有独有历史的旧分支先保存同 SHA 的归档标签，验证远端标签后才删除旧分支入口。
不强推、不删除提交历史、不把尚未合并的旧代码直接覆盖当前产品。

| 旧分支 | 原 SHA | 保留位置 |
| --- | --- | --- |
| `codex/local-optimizations-20260908` | `49a9ad123c5c98dde08836db4d7a36dbf8b18d6d` | 已包含于现有主线或 4050 分支历史 |
| `codex/local-recovery-sam2-20260825` | `3f4c425fd5e18b59dab1692b8197a3adc404b9ff` | 已包含于现有主线或 4050 分支历史 |
| `codex/main-hardening-20260826` | `1f2ecf180e65e68f2516630ba4895ec4bee08197` | 已包含于现有主线或 4050 分支历史 |
| `codex/reconstruct-stable-0e388f6-capability-port` | `447634ae16d1b804f76e18371d45acf2ec920ff8` | `archive/20260909/reconstruct-stable` |
| `codex/release-readme` | `e8fe8bb82100003e39f357748889c317c1d2cfc5` | `archive/20260909/release-readme` |
| `codex/selective-key-material-verification-20260826` | `c70dd3111e2ba421bac3915b25eb9d7ca71a7c34` | `archive/20260909/selective-key-material-verification` |
| `codex/stable-product-readme` | `ae244be42bf5bcf33b61ff4ad254ab68f72c1806` | `archive/20260909/stable-product-readme` |

要查看或继续某个归档版本：

```powershell
git fetch origin --tags
git show archive/20260909/release-readme
git switch -c codex/resume-release-readme archive/20260909/release-readme
```

当时统一客户更新地址是用户指定的集成分支操作。当前两个仓库的 `main` 均按
[双仓发布政策](DUAL-REPOSITORY-RELEASE-POLICY.md)完成适用门禁；本轮真实全链路质量和
4050 原机验收未因此自动通过。

## 本轮集成证据

- 纳入 4050 分步硬件诊断、独立超时监督、非阻塞退出管道、Git 源码更新和完整性缓存。
- 固定共享工作区相对 `49a9ad1` 的 67 个后续变更文件；产品任务 19 路径回执中，
  18 个文件保持原字节，README 仅另加本轮客户更新入口。训练记录截至 goal64。
- 汇总页面/报告、素材库、连续实验边界、逐步骤复核、源帧身份、性能诊断和训练支持工具。
  `result_review.py` 纳入的是既有完整性实现，已撤回的压缩上下文试验不在本轮代码中。
- 修复集成回归发现的训练入口重型依赖提前导入：可选实现改为按需加载，训练计算 AST
  除相对导入层级外不变，并将新实现文件加入训练源码身份记录。
- 本地完整确定性回归：1,575 passed、17 skipped，覆盖率 74.49%；可选训练依赖环境
  的 CPU 损失/采样检查另有 40 passed，桌面 Node 测试 33 passed。
  模拟 Key 夹具改用无真实凭据外形的固定测试值，相关 29 项测试通过；未放宽凭据检查。
- Ruff、Python 编译、Web JavaScript 和部署脚本语法、差异空白检查通过。
- 集成后的跨平台 CI 发现 FFmpeg 6+ 不再提供旧版 showinfo 包位置，随后补齐
  原生输入 PTS 统计与 ffprobe 帧记录的唯一映射；保留旧版 FFmpeg 的追溯路径。
  时间戳、包位置和像素摘要必须同时通过，缺失或冲突时保持未验证。
  使用 [FFmpeg 原生帧统计接口](https://www.ffmpeg.org/ffmpeg.html#Advanced-options)，
  并验证非零起点、变帧率、B 帧、跳转和不连续窗口的采样像素保持一致。
  后续回归改用编码前原生输入记录，以兼容 FFmpeg 6；FFmpeg 9 复用项目已有的
  帧率参数探测。额外验证了实时抽帧时进度回车与 Windows 换行不会吞掉来源记录。

上述为集成实现和确定性证据。跨平台结果以对应提交的 GitHub Actions 为准；
局部真实调用或局部图像指标不能替代完整多视角实验验收。4050 原机启动耗时、
完整语义质量及正式归档/稳定发布就绪仍为 **PARTIAL_EVIDENCE / NOT_PROVEN**。
本轮未创建新离线包，未上传模型、原视频、Runtime、标注数据库或凭据。
