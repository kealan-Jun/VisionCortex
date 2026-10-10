# 本地启动与开发环境

[文档导航](../README.md) · [项目首页](../../README.md)

当前操作参考；执行前核对所选配置、固定源码 SHA 和适用运行回执。

## 首次克隆与安装

本页从一个完整的新克隆开始，不依赖项目作者的目录、设备或 Python 环境。
准备 Git 和 Python 3.11/3.12；将 `REPOSITORY_URL` 替换为你访问的本仓库克隆地址。

Linux / macOS：

```bash
git clone "REPOSITORY_URL" VisionCortex
cd VisionCortex
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell：

```powershell
git clone "REPOSITORY_URL" VisionCortex
cd VisionCortex
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Python 3.12 对应 `python3.12` 或 `py -3.12`。Windows 不需要先激活环境，
启动入口可直接找到仓库内的 `.venv`；这也避免激活脚本被 PowerShell 执行策略阻止。
环境可放在其他本地磁盘：先激活它，或给启动器指定 `--python` / `-PythonExecutable`。
不要把 Python 环境放在网络共享中。

基础安装不会安装可选模型栈或下载权重，不需要 API 密钥。
要运行开发检查，再用同一个解释器安装开发依赖，其中包括 pytest、pytest-cov 与 Ruff：

```bash
# Linux / macOS，已激活环境
python -m pip install -e ".[dev]"
```

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`.env.example` 是部署变量说明，不会自动加载；
本地页面无需复制或填写它。

## 检查并启动页面

Linux / macOS：

```bash
./start-visioncortex.sh --check-only
./start-visioncortex.sh
```

Windows PowerShell：

```powershell
.\Start-VisionCortex.bat -CheckOnly
.\Start-VisionCortex.bat
```

Windows 可双击 `Start-VisionCortex.bat`，macOS 可双击 `Start-VisionCortex.command`。
默认页面是 `http://127.0.0.1:8000/#/home`，只接受本机访问。
端口被占用时，使用 `--port 8010` 或 Windows 的 `-Port 8010`。

解释器选择顺序是：显式参数、已激活的虚拟环境/Conda 环境、仓库内 `.venv`、
系统可用的 Python。预检会检查基础依赖的实际导入以及 Web/CLI 入口是否来自本仓库，
不会自动安装软件、探测 GPU、访问 NAS、启动模型或发送模型 API 请求。

默认配置 `configs/development-local.yaml` 的输入、队列、缓存和产出均为本地目录。
NAS 自动采集和模型理解默认关闭；`examples/development-index.csv` 只有列头，
所以采集批次为空。第一次打开网页不需要创建假视频或借用其他实验数据。

机器可读预检：

```bash
python tools/doctor.py --json
```

只有需要核对部署硬件时才显式使用 `--check-gpu`；它查询 GPU 身份与可选依赖，
不运行推理。NAS 检查是另一个显式操作，须先完成所属部署的挂载与身份核验。
预检通过仅证明所检查的依赖和导入条件；实际网页启动、模型调用、真实视频质量和
正式发布资格仍分别记录证据，不能用一个 `ready` 标记代替。

## 首次使用故障排查

| 现象 | 处理 |
| :--- | :--- |
| 未找到 Python / 版本不支持 | 安装 Python 3.11 或 3.12，用该版本重新创建独立环境。 |
| 创建环境时提示缺少 `ensurepip` / `venv` | 安装与所选 Python 版本对应的系统 venv 组件，再重新创建环境。 |
| 基础依赖缺失 | 按预检显示的当前解释器执行安装，不切换到另一环境。 |
| 显示 `DifferentCheckout` | 当前环境安装了另一份源码；在本仓库用同一解释器执行 `python -m pip install -e .`。 |
| OpenCV 无法导入，提示 `libGL.so.1` 等系统库 | Linux 安装所需系统库，例如 Ubuntu 的 `libgl1`、`libglib2.0-0`；随后重新运行预检。 |
| 未发现 FFmpeg | 仍可浏览页面；处理真实视频前安装 FFmpeg，确认 `ffmpeg -version` 和 `ffprobe -version` 可执行。 |
| 没有 GPU 或未安装模型栈 | 本地页面可用；模型推理须按所选硬件的部署说明单独配置。 |
| 端口已有其他服务 | 换用空闲端口；不要停止或覆盖他人的服务。 |
| 采集批次为空 | 新克隆附带的是空索引；本地首次使用不会连接已有采集设备。 |

## Ubuntu 本机独立体验实例

已配置模型、Python 环境和 AI 服务的 Ubuntu 工作站，可用 `tools/local_experience.py`
建立一个独立实例，模拟首次配置和提交真实视频的用户流程：

```bash
python tools/local_experience.py prepare --root /本地磁盘/LocalExperience --source /本地原视频目录
python tools/local_experience.py check --root /本地磁盘/LocalExperience
"/本地磁盘/LocalExperience/打开 VisionCortex.sh"
```

必须使用已验证的工作站 Python 环境。该实例保存源码快照、独立配置、队列和运行结果，
复用本机依赖、模型及原视频引用，不复制模型和原片，不包含密钥，不可直接搬到其他电脑。
预检会核对依赖、模型文件哈希、输入元数据、GPU 身份和剩余空间；它不证明真实分析质量。
首次界面连接验证仍须成功后才启用 AI 配置。

同一目录再次 `prepare` 会拒绝；再次启动复用所属服务。修复在原实例内完成，并更新
`Acceptance` 中的文件身份和实测回执。`stop --root /本地磁盘/LocalExperience` 只停止
该实例且拒绝中断活动任务。真实视频完整运行、浏览器结果可用和正式发布资格须分别验收。

分析已执行完但最终质量检查未通过时，任务以 `partial`（分析结束 · 证据不足）结束，
释放执行队列并保存阶段 HTML 报告、片段、素材及具体缺口。它不会发布正式档案或生成
已验收的完整结论，也不要求人工批准候选。任务页、实验记录和全局素材库均可查阅阶段
成果；原任务支持重新分析，复用经过验证的输入和缓存。相同证据不足不会自动无限复跑
或反复调用付费模型；语义请求自身继续使用有界恢复策略。实际执行异常仍记录为失败。

实验候选片段之间的间隙不等于实验结束。启用视觉模型后，导出前按第一人称录像分别
复核相邻切点。明确的操作延续合并为同一实验单元；有样品/器具承接的不同实验单元
保留在同一连续实验链中。换实验台或第三人称机位不再强制分开，但必须有移动/携带的
画面依据；同一实验员本身不足以合并。原始 CV 原子片段、动作成员与模型单元分开保存。对可能承接但证据不足的切点，最多增加一次高密度第一人称复核，阈值不降低；尾部复核推进切点后，允许核对一次变化后的相邻边界。独立请求按模型并发配置执行。

系统继续读取检测终点之后的原视频。默认切点复核预算为间隙 120 秒、两侧各 12 秒、
20 张图；尾部按 90 秒窗口继续、每组最多 8 窗口。预算耗尽、采样不足或模型失败均保留
“边界待核实”，不会解释为实验完成。已观察到录像末尾仍操作时标记“录像结束，实验待续”，
保留可确认承接的尾部视频。后续录像可通过同一输入的分段索引一并分析；尚未实现跨任务
自动拼接。单步结束、实验单元结束与整个连续链结束不能互相替代。

连续链保留单元时间范围及机位时间表。第三人称随对应单元切换；未确认机位的过渡和尾部
显示“未确认对应机位”，第一人称连续保留，不用旧实验台冒充对照画面。复核图片、编号、
模型结果及 Token 保存在 `JSON-Config-Files/experiment_boundary_review.json` 和
`Boundary-Review-Frames/`。这些是模型边界证据，不是动作人工真值或完整步骤召回证明。
边界未解决、或延长视频后步骤尚未重新核验的产出保留为阶段结果，不能通过完整实验验收；不会自动更新旧安装包和旧任务。


## 安装与安全配置

```powershell
# 在项目根目录执行
# 轻量开发/确定性测试：不会下载 PyTorch、CUDA、YOLO 或 Transformer。
python -m pip install -e ".[dev]"

# 需要真实模型推理时显式安装模型与 TensorRT 运行栈。
python -m pip install -e ".[models,tensorrt]"
$env:ARK_API_KEY = '<在本机安全设置，不要写入 yaml 或 git>'
```

`models` 包含固定版本的 Transformer、YOLO 与 CLIP 运行依赖；SAM2 仍通过
`sam2` 可选项或部署脚本的固定 revision 独立安装。生产部署脚本优先使用各硬件
目录内的锁定依赖，不会因为轻量开发安装而改变现有模型或 TensorRT 引擎。

密钥通过环境变量或本机 AI 服务设置提供，不写入 YAML、Git、日志或报告；本机密钥存储与版本恢复规则见[AI 厂商与设置](../reference/providers.zh-CN.md)。

本地体验实例使用本地存储，关闭自动设备日与目录采集生产者。指定的原视频目录用于只读验收；在页面中显式导入待分析视频。实例配置可重新加载，重新准备会拒绝覆盖既有身份和数据。
