# VisionCortex 双仓同步与发布规则

2026-09-15 按用户明确要求生效，取代此前“开发仓 → 稳定发布仓”的单向晋升规则。
保留本文件路径，兼容已有文档和打包脚本的引用。

## 同一代码，两个平级仓库

- `https://github.com/kealan-Jun/VisionCortex.git`
- `https://github.com/RealityLoopAI/VisionCortex.git`

两个仓库都可以承载代码、分支、评审和 CI，不设置开发专用或稳定专用权限。
现有本地别名 `development`、`origin` 只是历史配置，不代表仓库等级。
离线视频分析和 NAS 自动处理继续使用同一套代码。

同步完成的标准：

```text
本地目标提交 SHA
  == kealan-Jun/VisionCortex main SHA
  == RealityLoopAI/VisionCortex main SHA
```

需要共享的工作分支也推送同一个提交。已有其他分支、标签和提交不删除。
正式发布标签如需同步，两端的 peeled commit 必须相同，不移动已发布标签。

## 同步步骤

1. 核对两个远端 URL、默认分支、工作分支、HEAD、upstream 和工作树。
2. 获取两个远端的新提交，检查祖先关系。发生分叉时先保留并合入双方变更；
   不用强推、`reset --hard`、`git clean` 或 `git push --mirror` 消除分歧。
3. 检查本次代码、配置、文档和测试；只按明确路径暂存。不得提交密钥、模型、
   引擎、原视频、录音、NAS 数据、运行数据库、缓存或失败暂存目录。
4. 完成代码管理检查（范围、凭据、提交历史），形成一个本地提交。对两个仓库
   推送同一个 SHA，中间不改写提交；常规同步不启动或等待 CI 自检。
5. 再次查询远端，确认目标分支的完整 SHA 相同。Git 对不同服务器的推送不是
   跨仓事务；若只有一端成功，明确报告不同步状态，修复原因后补推另一端，
   不能假称两端已同步，也不能强制回退已成功的一端。

当前远端别名对应上述 URL 时，可使用以下命令；执行前仍需完成检查和冲突处理：

```bash
git fetch development
git fetch origin
git push development HEAD:refs/heads/main
git push origin HEAD:refs/heads/main
git ls-remote development refs/heads/main
git ls-remote origin refs/heads/main
```

按用户同日补充要求，两端 GitHub Actions 仅保留手动入口；推送和 PR 不自动
执行 CI，常规代码同步也不手动触发或等待 CI。已有检查结果保留真实历史，
不把失败改写为通过。“已同步”只表示代码一致，不代表服务已经升级。

## 正式构建与服务部署

代码入仓不触发 NAS 扫描、模型运行、数据迁移、服务重启或原片删除。
安装包和服务使用固定提交，升级前校验依赖锁、配置、输入/存储根及模型身份。

正式交付仍需要：完整确定性检查、凭据/运行数据检查、适用的真实视频质量回执、
归档完整性回执和已知限制。性能数据区分预处理和全链路；模型调用与 Token
使用情况按真实回执记录。这些要求适用于两个仓库，不再要求先在某一仓库晋升。

`deployment/runtime/Release.py` 的 `Acceptance.json` 使用中性的
`commit_sha` 与 `release_ready`。为兼容已有安装包，仍可读取旧字段
`development_sha`，但它不再表示一个独占开发仓；同时出现两个 SHA 字段时
必须一致。封存、安装、切换和回退继续校验同一个不可变提交与内容摘要。

## 数据与执行节点边界

凭据只通过环境变量或未跟踪的私密存储提供，检查输出只报告文件路径和凭据
类别，不输出值。发现历史泄露不能通过仅修改当前文件宣称已清理。

RTX 4060 仍为冻结执行节点，可从任一仓库获取任务指定的同一个提交；除用户
另有授权，只运行生产视频并回传证据，不修改、提交、测试或调参。
