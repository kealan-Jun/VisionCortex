# RTX 4050 启动修复复盘（2026-09-09）

本次解决了桌面长时间停在“正在准备应用／正在检查本机运行环境”的阻塞，并实际打开
“实验工作台”。成功依赖于定位并修复阻塞读取，以及分阶段验证；增加诊断和更新入口
本身不能保证启动成功。

## 做对了什么

### 1. 先确认文件身份，再复现具体阻塞

原离线包的 158 个源码文件均符合原始哈希回执。随后保持 Python、NumPy 和输入条件
一致，只改变监控线程，使用有超时的子进程复现：

| 对照条件 | 本机观察 |
| --- | --- |
| 无桌面监控线程 | NumPy 1.26.4 导入约 1 秒完成 |
| 仅父进程句柄监控或仅 Job Object | 导入完成 |
| 后台线程阻塞读取标准输入 | NumPy 导入超时 |
| 仅换成阻塞式 `ReadFile` | 仍超时 |
| `PeekNamedPipe` 检查已有字节，再读取这些字节 | 导入完成，关闭指令和管道断开可正常退出 |

Python 栈将停顿定位到 NumPy 原生扩展导入和等待标准输入的监控线程。现象与
[NumPy #24290](https://github.com/numpy/numpy/issues/24290) 的 Windows 复现一致。
这些证据定位了本机触发条件，并不声称已经证明 DLL 内部每一把锁的实现原因。

### 2. 硬件检查放到独立进程，并由外部监督超时

`tools/rtx4050_hardware.py` 将驱动、PyTorch、CUDA 和 TensorRT 各步骤分别记录，
父进程负责期限和终止。导入卡住时，不依赖被卡住的线程自己返回错误。

这使本机硬件检查能够在约 10 秒完成，也揭示了下一处停顿：主进程加载 AI 配置时仍
会导入 NumPy/OpenCV。因此硬件预检通过之后，仍须修复桌面生命周期中的阻塞读取。

### 3. 改掉阻塞读取，保留进程生命周期约束

`tools/windows_desktop_lifecycle.py` 每次只读取管道中已有的字节，无输入时短暂等待。
仍支持分段到达的 `stop` 指令、管道 EOF、父进程退出及 Job Object 后台进程清理；
未结束的异常长指令按失败退出，避免持续积累内存。没有靠取消监控让导入通过。

### 4. 用经过校验的增量更新应用修复

更新读取已提交源码，校验旧文件和新文件，先备份，再替换，最后更新完整性清单；
错误时恢复已替换文件。本机更新后 160 个源码文件与新回执一致。

代码复核进一步发现执行脚本也必须绑定提交。本次补上对两个更新执行脚本的提交存在性
和内容一致性检查，并从暂存的 Git blob 加载执行器；回归覆盖未跟踪执行器、暂存内容
和 Windows CRLF 检出。源码更新不等于安装模型或依赖，也不等于获得稳定发布资格。

### 5. 从测试一直验证到实际桌面页面

本机修复快照完成了 30 项针对性测试、Ruff、Python 编译和 diff 检查。随后通过正常
桌面入口执行完整文件检查、硬件检查、引擎构建和推理测试、模型自检，最后读取桌面
实际显示的“实验工作台”“服务运行正常”“档案存储可用”“智能理解连接已验证”。

没有把单次 HTTP 200 或文件生成当作整个桌面启动已经成功。

## 原机实测与证据边界

以下实测绑定本机重建的修复源码快照
`7dcc6aa84bc78d1aad898938b017ea8c57f88ccc`，对应包清单 SHA-256 为
`f2c4b4e1c456b1fb2bb0fcfc253486f9475133cfdcfdc72e72e6dad2eb7754a1`。
它不是 GitHub 分支的完整源码树，不能把这些运行结果直接归给整个 GitHub 新提交。

执行环境：Windows x64、RTX 4050 Laptop 6 GB、NVIDIA 驱动 556.12、嵌入式
Python 3.12.10、NumPy 1.26.4、PyTorch 2.6.0+cu124、TensorRT 10.4.0。

| 阶段 | 实测耗时 | 证据含义 |
| --- | ---: | --- |
| 完整性校验（约 10.7 GB） | 约 4 分钟 | 本次包文件符合清单 |
| 硬件检查 | 10.297 秒 | 本次 CUDA FP16 运算执行成功 |
| 第一视角引擎构建及测试 | 478.3 秒 | batch 16，6 轮推理，候选无失败 |
| 第三视角引擎构建及测试 | 459.1 秒 | batch 16，6 轮推理，候选无失败 |
| YOLO-World 自检 | 21.794 秒 | 合成输入上的模型调用 |
| Grounding DINO 自检 | 45.658 秒 | 合成输入上的模型调用 |
| LabPics 自检 | 20.905 秒 | 合成输入上的模型调用 |
| SAM2 自检 | 16.275 秒 | 合成输入上的模型调用 |

上述耗时是该机器首次执行的观察值，不是其他硬件或每次启动的时间保证。引擎缓存只有
在硬件、源码清单、配置和模型身份仍符合规则时才可复用。

- **PROVEN**：该本机修复快照的硬件检查、两个引擎执行、四个模型合成自检与桌面可见启动。
- **PARTIAL_EVIDENCE**：GitHub `6ebb97d` 基线加本补丁的整树运行。本次移入的两个运行
  脚本与原机验证版本一致，但该分支其他源码差异未整包部署到原机重跑。
- **NOT_PROVEN**：真实实验视频准确性、生产六路全链路质量和稳定发布就绪度。

原始日志和机器回执保留在原机 `Runtime/Logs`、`Runtime/Engines`、`Runtime/Updates`。
本文件为人工整理的复盘，不包含原始视频、模型、引擎、原始日志或凭据。

## 回归与提交范围

Windows 原生回归会启动有管道输入的独立 Python 进程，读取本检出目录的生命周期模块，
在输入空闲时导入 NumPy，再检查退出行为。默认使用运行 pytest 的 Python；可指定离线
包解释器复现目标依赖组合。非 Windows 主机明确跳过这组原生测试。

在项目支持的 Python 3.11/3.12 测试环境中，绑定本检出的源码后执行：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$env:VISIONCORTEX_TEST_PYTHON = 'D:\VisionCortex4050\python\python.exe'
python -m pytest tests/test_rtx4050_hardware.py tests/test_rtx4050_source_update.py tests/test_windows_desktop_lifecycle.py tests/test_rtx4050_repair_integration.py tests/test_rtx4050_windows_deployment.py tests/test_rtx4050_connection_update.py
ruff check src tests tools/windows_desktop_lifecycle.py tools/update_rtx4050_source.py
python -m compileall -q src tests tools/windows_desktop_lifecycle.py tools/update_rtx4050_source.py
git diff --check
```

本次发布检出已用原包 Python 3.12.10、临时独立目录中的 pytest 8.4.2，显式绑定本
检出的 `src`，执行上述六个测试文件：**55 passed**。Ruff 和 Python 编译检查通过。
原分支加新增测试时先得到 5 项预期失败，合入修复后通过；这包括原生 NumPy 导入
超时与更新执行器未绑定提交的回归。GitHub 完整跨平台 CI 和整包重新部署仍是独立门禁。

此次按用户明确指定提交到 `RealityLoopAI/VisionCortex` 的
`codex/rtx4050-optimization-20260908` 分支，基线为
`6ebb97d47ec7d9d5bcfde2e4969ab305ccda3c19`。只补充生命周期修复、更新执行器校验、
回归测试和本文档，保留该分支已有工作。此次分支提交不是稳定 `main` 或发布标签晋级；
[双仓发布政策](DUAL-REPOSITORY-RELEASE-POLICY.md) 的正式发布门禁仍独立适用。
