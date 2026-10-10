# Windows 桌面应用的源码更新

[文档导航](README.md) · [源码版本选择](SOURCE-UPDATES.md)

使用项目提供的源码地址建立独立代码目录，保留原应用安装目录与运行环境。
不把访问令牌写入命令、远端 URL 或聊天。更新前等待当前分析结束并关闭所属应用。

```powershell
# 在已配置上游且没有本地修改的代码目录执行。
git status --short --branch
git fetch origin
git pull --ff-only
```

随后使用代码目录的 `deployment\rtx4050-windows\Update-From-Source.cmd`，选择原应用目录。
更新器接受已提交源码，验证原包和变更内容，保存可回滚备份；不要强制重置用户文件。
再次从原目录启动 `VisionCortex.exe`。源码更新不安装模型或引擎；启动仍核验对应身份。
实际应用的源码版本取自安装目录 `BUNDLE-METADATA.json` 的 `source_git_commit`，不能由文件夹名称推断。

## 首次校验与再次启动

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

## Windows 缓存路径过长（WinError 206）

Windows 缓存路径过长时，使用界面的保存位置设置选择较短的本地目录。
SAM2 工作目录通过完整内容指纹绑定输入、视角和模型，路径长度在创建前检查；
缩短路径不能绕过身份或完整性检查。失败任务的原始诊断保留，不自动覆盖成功回执。

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

诊断文件和更新回执只证明其对应执行。目标机初始化耗时、真实模型与视频质量
需在当前版本实际运行后记录，不把其他平台测试结果当作本机通过证据。
