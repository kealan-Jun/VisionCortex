# RTX 4050 通过 Git 更新源码

已有完整离线包只需保留一份。运行环境、视觉模型和目标机生成的引擎继续使用原目录；
后续拉取代码，通过更新入口将变更应用到原包，无需反复传输多 GB 压缩包。

## 首次连接代码仓库

Windows 需安装 Git，并具备私有仓库 `RealityLoopAI/VisionCortex` 的读取权限。
使用 Git 的账户登录方式，不把访问令牌写进命令、远端 URL 或聊天。

在 PowerShell 中执行，代码目录与已解压的应用目录必须分开放置：

```powershell
git clone --depth 1 --branch codex/rtx4050-optimization-20260908 --single-branch https://github.com/RealityLoopAI/VisionCortex.git D:\VisionCortexSource
cd D:\VisionCortexSource
```

等待当前分析结束并关闭 VisionCortex，双击代码目录中的
`deployment\rtx4050-windows\Update-From-Source.cmd`，选择原来的应用目录。
也可在 PowerShell 执行：

```powershell
powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File .\deployment\rtx4050-windows\Update-From-Source.ps1 -PackageRoot D:\VisionCortex4050
```

成功后仍从原应用目录双击 `VisionCortex.exe`。更新不安装 Python、模型或引擎。
源码或配置变更可能使引擎身份失效，后续启动仍按现有规则检查是否需要重新准备。

## 后续更新

```powershell
cd D:\VisionCortexSource
git pull --ff-only
```

随后再次运行 `Update-From-Source.cmd`。程序只接受已提交、无受版本控制文件修改的
代码快照；同一提交重复执行会核对源码后直接返回。未跟踪的本地文件不会进入更新。

更新器自身和 `apply-update.py` 也必须存在于选定提交，工作区内容须与提交一致。
执行阶段使用从 Git blob 提取并暂存的 `apply-update.py`，避免读到未跟踪或随后改变的
工作区脚本。干净 Windows 检出中的 CRLF 换行可以正常校验，实际补丁仍使用提交字节。

更新会验证旧文件与补丁文件，备份变更内容到原应用的 `Runtime/Updates`，支持新增
与删除源码文件，最后替换完整性清单。失败会恢复已替换的文件；下次启动仍执行原有
全包校验。若所选提交改变了运行环境资产或核心/视觉依赖，更新会拒绝继续，须单独
准备兼容运行环境。它不会把缺失依赖伪装为已安装。

## 初始化停滞的诊断

硬件检查在独立进程中执行，界面会依次显示系统与内存、NVIDIA 驱动、PyTorch 导入、
CUDA 设备查询、设备属性、TensorRT 导入、CUDA 分配、FP16 运算、同步和释放。
每一步有独立等待上限，完成后立即进入下一步，没有固定等待 15 分钟的行为。
`nvidia-smi` 的调用另有 20 秒超时。

失败时保留以下文件及窗口报错，无需发送 Key 或原视频：

- `Runtime/Logs/hardware-preflight.json`：当前步骤、已完成步骤耗时、失败或超时位置。
- `Runtime/Logs/hardware-preflight.log`：本次子进程输出；长时间停顿时包含 Python 栈。
- `Runtime/Logs/desktop.log`：桌面启动过程。
- 若存在，`Runtime/target-preflight.json`：硬件检查成功回执。

前两份诊断文件在下一次硬件预检时更新，回传前先保留当前失败记录。检查失败不会
生成新的成功回执；旧的 `target-preflight.json` 不能当作本次成功证明。

截至 `6ebb97d` 的证据证明诊断、超时终止和源码更新的确定性合同；Windows/RTX 4050 原机执行
仍为 **NOT_PROVEN**。用户提供的三次等待态采样属于 **PARTIAL_EVIDENCE**，尚不足以
认定具体库或驱动缺陷已定位或解决。真实视频质量和稳定发布资格另行验收。

2026-09-09 已在本机复现另一处启动阻塞：后台线程持续等待标准输入，会让 NumPy
导入停顿。修复以 `PeekNamedPipe` 检查已有字节，再作有界读取，保留关闭指令、父进程
监控及所属进程树清理。独立硬件预检通过后，主进程也必须使用这一修复才能继续启动。
原机修复快照已完成硬件、两个引擎、四个模型及桌面工作页面验证；证据绑定的源码快照、
实测耗时、回归方法及分支整体运行仍缺少的验证，见
[RTX 4050 启动修复复盘](RTX4050-STARTUP-REPAIR-20260909.md)。

## 本轮开发验证（2026-09-08）

4050 启动、连接、增量更新和厂商接口专项测试共 **99 passed**；Ruff、Python 编译
和 diff 检查通过。使用已加载依赖的本地 Python，并显式绑定本分支 `src`。
完整回归为 **1435 passed、4 skipped、2 failed**：原训练模块顶层重型依赖导入，
以及原片段生成并发时序断言。本分支没有修改这两处实现或放宽测试。

基础同步提交的 CI 另暴露 Windows 默认编码读取中文配置、4050 测试依赖执行机剩余
空间的问题；本分支已明确测试读取编码、固定模拟可用空间，生产空间检查保持有效。
跨平台 CI 与目标机原生执行仍需分别确认，以上确定性检查不替代它们。
