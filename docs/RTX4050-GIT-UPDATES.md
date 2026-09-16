# RTX 4050 通过 Git 更新源码

已有完整离线包只需保留一份。运行环境、视觉模型和目标机生成的引擎继续使用原目录；
后续拉取代码，通过更新入口将变更应用到原包，无需反复传输多 GB 压缩包。

## 首次连接代码仓库

Windows 需安装 Git，并具备两个平级仓库之一的读取权限。自 2026-09-16 起，
统一从 `main` 获取源码；不再使用硬件专属长期分支。以下命令以 RealityLoopAI
仓库为例，也可使用 kealan-Jun 仓库中相同 SHA；代码同步不代表目标机验收。
使用 Git 的账户登录方式，不把访问令牌写进命令、远端 URL 或聊天。

在 PowerShell 中执行，代码目录与已解压的应用目录必须分开放置：

```powershell
git clone --depth 1 --branch main --single-branch https://github.com/RealityLoopAI/VisionCortex.git D:\VisionCortexSource
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

两边此前分叉的 4050 启动修复已合并，保留原提交历史。已有代码目录只需将
远端选择下列平级仓库之一，切换到main 分支；`D:\VisionCortexSource` 替换为自己的代码目录：

```powershell
cd D:\VisionCortexSource
git remote set-url origin https://github.com/RealityLoopAI/VisionCortex.git
git fetch origin
git switch main
git branch --set-upstream-to=origin/main
git pull --ff-only
```

完成一次设置后，以后在代码目录执行 `git pull --ff-only` 即可。如提示有本地修改，
先保留修改再处理，不使用强制重置。Git 只更新代码目录，必须继续执行安装目录更新入口。

随后再次运行 `Update-From-Source.cmd`。程序只接受已提交、无受版本控制文件修改的
代码快照；同一提交重复执行会核对源码后直接返回。未跟踪的本地文件不会进入更新。

更新会验证旧文件与补丁文件，备份变更内容到原应用的 `Runtime/Updates`，支持新增
与删除源码文件，最后替换完整性清单。失败会恢复已替换的文件；下次启动核对新清单，
重新哈希变化的文件，复用满足下述条件的未变化文件。若所选提交改变了运行环境资产或核心/视觉依赖，更新会拒绝继续，须单独
准备兼容运行环境。它不会把缺失依赖伪装为已安装。

## 首次校验与再次启动

原 R6 包清单有 32,381 个文件，共 10,659,197,771 字节。旧版每次启动都重新读取
整包计算 SHA-256，首次 AI 连接验证之后启动应用还会再次计算。

桌面启动和 AI 连接验证现在共享 `Runtime/Cache/PackageIntegrity` 中的一份小型
校验缓存。首次仍完整检查；后续核对文件身份与修改记录，仅对新增或变化文件重新
计算哈希。普通退出重开、再次验证连接不会主动清除缓存；源码更新也可继续复用
哈希预期与文件身份均未变化的大模型和运行库。

Windows 使用 NTFS/ReFS 的卷与文件 ID、大小、创建时间、写入时间和 ChangeTime，
不将 Python 的 Windows `st_ctime` 创建时间误当修改记录。刚写入的文件在时间戳
精度窗口内重新校验；FAT/exFAT、网络盘或无法取得可靠修改记录时回退到完整读取。
缓存用本机随机密钥认证，Windows 密钥通过当前用户 DPAPI 保护；缓存损坏、换账户
或移动程序目录时先重新验证；校验器本身更新也会重新建立缓存。缓存只是上次内容校验的复用，不声称每次启动都重新
完成了全量哈希，也不替代磁盘介质诊断。

`Runtime/Logs/package-verification.json` 记录本次实际读取字节、复用字节、文件数、
耗时和清单 SHA。缓存和日志都覆盖固定文件，不生成新的离线包或多份运行环境。
需要主动完整复检时，在安装目录运行以下命令；它刷新同一缓存，不启动 GPU 或调用 AI：

```powershell
.\python\python.exe -I -B .\tools\rtx4050_portable.py --verify-package-only
```

本机复用原 R6 解压目录实测（Ubuntu/ext4/NVMe，清单 SHA
`9f0ddf6c77811a083a9bb3e2d67c0d9cb038e4708eefcaac3f34f99458b85c72`）：首次校验
10.36 秒、读取全部 10.66 GB；紧接的重复校验 1.52 秒、内容重读 0 字节，所有文件
均复用记录。这里只测文件校验，不能推定 Windows 机械盘耗时、整机启动耗时或推理质量。

开发分支同时保留了客户侧启动修复所需的非阻塞命令管道处理，以及更新执行器绑定
Git 提交、从已暂存的提交内容执行和 CRLF 检出兼容性，避免升级后重新引入旧启动问题。
Windows 原生管道与缓存回归纳入跨平台 CI；4050 原机再次启动耗时仍需客户实测。

## Windows 缓存路径过长（WinError 206）

客户回传的 s04 失败发生在创建 SAM2 帧目录时，属于 Windows 路径长度问题。
本轮采用 `local_cache_root/s2/<完整 64 位指纹>`，去掉重复的实验、事件与机位长名称；
Windows 工作缓存也去掉实验名称层级。输入、视角和模型的身份仍由完整指纹绑定。
新路径在创建前按 UTF-16 长度检查，并给出可操作的存储设置提示。

待当前任务结束后，按上文拉取源码并更新原安装目录，再从原安装位置复测。
若运行目录本身过长，按提示在「应用 → 保存位置设置」选择更短的本地目录。
不需要为本次源码更新重新传输完整离线包。Windows 原机修复效果仍需回传确认。

本轮同时纳入阶段产出清单、已完成视频的查看入口、辅助活动分类及实验边界修复。
录音未启用或缺失时明确标为跳过；可选录音失败时保留失败状态并继续视频分析，
不将不完整转写作为模型证据。中文或损坏的录音索引不会再次阻断这条恢复路径。

## 初始化停滞的诊断

硬件检查在独立进程中执行，界面会依次显示系统与内存、NVIDIA 驱动、PyTorch 导入、
CUDA 设备查询、设备属性、TensorRT 导入、CUDA 分配、FP16 运算、同步和释放。
每一步有独立等待上限，完成后立即进入下一步，没有固定等待 15 分钟的行为。
`nvidia-smi` 的调用另有 20 秒超时。

桌面进程另有独立硬件检查看门狗：硬件入口或同一步骤最多等待 4 分钟，整个硬件
检查最多 20 分钟。新步骤会获得自己的等待预算；重复步骤消息、普通日志不会续期。
正常完成立即进入下一阶段。即使 Python 监督进程本身没有继续输出，桌面也会显示
超时错误并请求退出，必要时只终止本次应用创建的进程树；不会自动重试。
这层保护只覆盖硬件检查，不改变引擎构建、AI 服务验证或视频分析的超时策略。

`Runtime/Logs` 位于包含 `VisionCortex.exe` 的安装目录，和 Git 代码目录不同。
可直接点击窗口的「应用 → 打开诊断目录」。通过聊天附件回传诊断文件即可，
不要把运行日志提交到代码仓库。若要确认更新是否实际应用，再提供安装目录的
`BUNDLE-METADATA.json`；其中的 `source_git_commit` 才是已应用的 Git 源码版本。
旧 R6 包可能没有该字段；仅看到文件夹名 R6 不能判断是否已更新。

失败时保留以下文件及窗口报错，无需发送 Key 或原视频：

- `Runtime/Logs/hardware-preflight.json`：当前步骤、已完成步骤耗时、失败或超时位置。
- `Runtime/Logs/hardware-preflight.log`：本次子进程输出；长时间停顿时包含 Python 栈。
- `Runtime/Logs/desktop.log`：桌面启动过程。
- 若存在，`Runtime/target-preflight.json`：硬件检查成功回执。

前两份诊断文件在下一次硬件预检时更新，回传前先保留当前失败记录。检查失败不会
生成新的成功回执；旧的 `target-preflight.json` 不能当作本次成功证明。

当前改动证明诊断、超时终止和源码更新的确定性合同；Windows/RTX 4050 原机执行
仍为 **NOT_PROVEN**。用户提供的三次等待态采样属于 **PARTIAL_EVIDENCE**，尚不足以
认定具体库或驱动缺陷已定位或解决。真实视频质量和稳定发布资格另行验收。

2026-09-09 回传的 293 字节 `desktop.log` 仅包含全包校验开始、硬件检查入口，
没有任何硬件子步骤。它能证明程序已进入硬件检查，不能确认安装目录是否应用了
分步检查修复，也不能确定具体阻塞库。需要已安装版本及硬件日志补齐证据。

本次桌面监督改动的本地确定性检查：33 项 Node 桌面测试、44 项 Python 4050 硬件、
部署与源码更新测试通过，JavaScript 语法及 diff 检查通过。新增回归模拟了监督
进程无输出、重复进度、总预算耗尽和超时后的迟到成功消息；跨平台 CI 增加同组
桌面测试。此结果不代表 Windows 原机上的 CUDA 初始化已成功。

## 本轮开发验证（2026-09-08）

4050 启动、连接、增量更新和厂商接口专项测试共 **99 passed**；Ruff、Python 编译
和 diff 检查通过。使用已加载依赖的本地 Python，并显式绑定本分支 `src`。
完整回归为 **1435 passed、4 skipped、2 failed**：原训练模块顶层重型依赖导入，
以及原片段生成并发时序断言。本分支没有修改这两处实现或放宽测试。

基础同步提交的 CI 另暴露 Windows 默认编码读取中文配置、4050 测试依赖执行机剩余
空间的问题；本分支已明确测试读取编码、固定模拟可用空间，生产空间检查保持有效。
跨平台 CI 与目标机原生执行仍需分别确认，以上确定性检查不替代它们。
