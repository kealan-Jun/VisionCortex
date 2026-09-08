# VisionCortex · RTX 4050 桌面应用离线包

适用：Windows 10/11 64 位、RTX 4050 Laptop 6GB、64GB 内存；针对驱动
556.12 固定 Python 3.12.10、PyTorch 2.6.0 CUDA 12.4、TensorRT 10.4.0。
包内已放入桌面运行环境、Python、全部 Python 依赖、FFmpeg 和全部本地生产模型权重。
无需安装 Python、Git、CUDA Toolkit、nvcc 或另行下载模型。

## 使用

1. 把 ZIP **完整解压**到本地 SSD 的短目录，例如 `D:\VisionCortex4050`。
   不要在压缩包预览窗口直接启动。建议路径不超过 60 个字符，避免 Windows
   长路径限制。解压后还需至少 15 GiB 可用空间用于首次引擎准备；视频与产出另计。
2. 双击 **`VisionCortex.exe`**。应用在自己的窗口中打开，无需安装浏览器组件，
   无需输入命令，也无需分别启动前端或后端。
3. 首次先选择**数据保存目录**。应用检测当前 Windows 用户已连接的网络盘并优先推荐；
   没有网络盘时推荐包内 `Runtime`。可点击「选择文件夹」改为其他本地盘，或填写
   `\\NAS名称\共享名\VisionCortexData`。「检查目录并保存」验证创建、写入、重命名、
   读取和剩余空间，成功才保存。自动检测不遍历共享文件，也不扫描局域网所有地址。
   同页可选择**已有原视频目录**（本机或 NAS），与产出目录分开放置。选择后可直接
   索引视频与配套时间戳，不重复上传；留空则使用文件上传。
4. 选择 AI 服务厂商与使用偏好，确认推荐的视觉模型，填写 API 密钥，点击
   「验证连接并打开应用」。密钥通过 Windows
   保护机制加密保存在当前用户 `%APPDATA%\VisionCortex\Secrets`，不进入
   压缩包目录。支持火山引擎豆包、阿里云百炼、智谱及其他兼容服务，费用由所选厂商账号承担。
5. 应用先完整校验本地离线包，并显示已读取的字节数与文件数；这一步尚未调用 AI。
   本地文件校验与云端请求分别计时，不再共用一个 4 分钟倒计时。
   本地校验通过后，用正式分析使用的调用代码发送两张随机测试图，验证多图理解及 JSON
   响应。只有真实验证通过才保存并启用，失败会说明认证、权限、模型或网络问题。
   更换厂商、接口地址、模型或偏好后必须重新验证；验证会产生少量 API 费用。
6. 应用自动检查文件与硬件、准备加速引擎和模型，然后进入现有分析工作空间。
   首次准备可能需要数分钟或更久，进度在窗口内显示；后续复用校验通过的引擎。
   若系统缺少 Microsoft VC++ 运行组件，会自动调用包内微软安装器；完成 Windows
   安装提示即可，整个准备过程无需联网下载。若安装器要求重启，界面会说明。
7. 如果设置了原视频目录，在新建实验中选择采集批次，或展开「按文件选择实验视角」，
   勾选视频、指定第一/第三人称、确认采集完成并保存批次，再提交分析。支持同名 CSV、
   `<视频名>_frames.csv` 和采集程序的配套时间戳格式。其他文件可添加机位后上传。
   机位数量按本次上传决定，不固定为六路；可逐个添加拍摄视角。
   当前跨视角分析至少需要一路第一人称和一路第三人称，单路或全部同类视角会在上传前提示。
   多机位按显存与 CPU 预算调度；并发数不是可上传机位数的上限。停止使用时直接关闭应用窗口。

重复双击会回到已打开的窗口。应用自行选择空闲本地端口；关闭时终止自己启动的
分析进程树。分析过程中关闭会中断当前工作，请等任务完成后退出。
启动失败时在窗口点击「重新尝试」或「打开诊断目录」；更换厂商、模型或密钥用「应用 → AI 服务设置」。

用「应用 → 保存位置设置」修改数据目录，「应用 → 打开产出文件夹」查看实际文件。
有分析任务或未完成上传时会阻止修改设置，避免中断工作。修改目录不搬迁或删除已有
文件；切回原目录可继续查看原档案。已验证的 AI 配置可直接继续使用，无需再次付费验证。
已保存的 NAS 不可用时会提示处理，用户可以恢复连接或明确改选本地，不会悄悄改写其他目录。

通过索引提交时，原视频留在用户选择的目录，档案记录源引用，分析期间请保持其可访问。
通过上传提交时，选择本地保存则原片、任务数据库、缓存和归档都在所选数据目录内；选择 NAS 时，
原片和归档在 NAS，任务数据库与缓存仍在包内本地 `Runtime`，避免网络盘上的数据库锁问题。
桌面日志、加速引擎和启动配置仍在包内 `Runtime`。目录检测只要求基础空余空间，
上传前会根据本次视频字节数、已有预留量和处理余量另做容量预检。

每个实验自动创建以下六类文件夹：

```text
所选数据目录/Archives/
  Processing/<实验归档名>/<任务编号>/  # 处理中或未通过最终门禁的产出
  <实验归档名>/                       # 原片与通过发布门禁的结果
    Original-Experiment-Videos/       # 原片、时间戳或源引用
    Experiment-Clips/                # 实验片段
    Key-Materials/                   # 关键帧与关键片段
    Lab-Daily-Reports/                # 实验日报
    Professional-PDFs/                # PDF 报告
    JSON-Config-Files/                # 结构化结果、索引、配置与证据回执
```

各阶段完成后可查看阶段产出；只有最终门禁通过才发布完整结果。分析执行结束但证据
不足时，任务以「部分结果」正常收尾，显示具体缺口，并保留视频、步骤、阶段 HTML
报告和 JSON 导出。全局素材库分别同步正式、阶段和候选数量，不自动将阶段素材转为
正式证据，也不会因同一个质量缺口无限重试。各厂商密钥分别加密保存。
应用和全部本地模型已随包提供；AI 连接验证和分析中的云端调用需要网络。
建议插上电源并使用高性能电源模式，减少其他程序对显存的占用。

## 模型怎么选

默认「质量优先」会为所选厂商填入适合多图与结构化分析的视觉候选，同时展示
推荐理由、官方说明及目录核验日期（2026-09-07）。可切换「速度与成本优先」，
或从同厂商的多个候选中选择，也可填写控制台中的自定义视觉模型/部署 ID。
不会把纯文本、图像生成或 OCR 专用模型列为实验步骤理解的默认候选。

| 厂商 | 质量优先候选 | 另一类选择 |
| --- | --- | --- |
| 火山引擎 | Seed 2.1 Pro（260628） | Seed 2.0 Lite |
| 阿里云 | Qwen3.8 Max（0902 快照） | Qwen3.7 Plus、Qwen3.7 Flash、Qwen3 VL Plus |
| 智谱 | GLM-5.3-Flash | GLM-4.6V、GLM-4.6V-FlashX、GLM-5V-Turbo |

推荐来自官方能力资料和本项目的多图、时序对照、JSON 合同需求，是选型起点，
不是已经测得的实验准确率排名。账号权限和地域必须由真实请求验证；失败时
不会自动降级为另一模型。模型别名可能由厂商更新，每次正式调用都记录所选
模型、服务返回模型、厂商、接口类型、请求编号和服务报告的 Token 用量。
语义缓存同时绑定实际接口、模型、思考参数、输出额度与图像传输设置，避免切换
模型或使用偏好后复用不同请求策略的答案。

应用按模型适配思考模式和图像格式。质量优先为支持的模型启用思考并保留
至少 8192 的输出额度；速度与成本优先通常关闭思考。GLM-5.3-Flash 官方仅
支持开启思考，因此不会给它发送禁用思考参数。推荐目录随离线包提供，不会
在安装时下载；超过 30 天会提示核对最新可用型号，不会自动更换任务模型。

上次连接验证的时间、模型、请求编号和 Token 用量显示在设置窗口，公共回执
位于 `Runtime/Desktop/connection-verification.json`。这只能证明当时的连接
和合成多图检查通过；实验动作、步骤完整性和跨视角判断仍需真实样本验收。
密钥不写入 YAML、回执或压缩包，正式分析读取同一份已验证配置。

- [火山引擎官方 SDK 的视觉与 Responses 示例](https://github.com/volcengine/ark-runtime-python)。
- [阿里云视觉模型选型与能力表](https://www.alibabacloud.com/help/en/model-studio/vision-model)。
- [智谱 GLM-5.3-Flash 参数与能力](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)，[GLM-4.6V 系列](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v)。

## 性能与证据边界

6GB 显存由一个闭集角色引擎使用，第一/第三人称阶段轮换；六路解码有界并发，
FP16 batch 按首次实测收缩，GPU 总显存预算 86%，引擎准备要求至少 22% 余量。
SAM2 视频和状态可卸载至内存，Grounding DINO 最终复核与 LabPics 在 CPU 执行，
GPU 阶段之间释放辅助模型。64GB 内存用于有界缓存与 CPU 模型，不会人为占满。
笔记本功耗、散热、CPU 型号与视频编码都会影响吞吐，不能承诺全程 100% GPU
利用率或固定完成时间。任务页显示实际资源遥测。桌面界面关闭自身 GPU 加速，为模型保留显存。

首次按 `[16, 8, 4, 2, 1]` 的 batch 候选构建并实测两个角色引擎，接受满足显存
余量的第一个候选。之后对 YOLO-World/CLIP、Grounding DINO、LabPics 与 SAM2
依次执行合成输入自检，记录可执行性回执；这些自检不能证明真实实验质量。

`PROVEN` 只能用于对应回执已验证的事实。离线包完整性和确定性测试不能证明
Windows 双击启动、4050 真实模型全链路、六路真实实验质量或稳定发布就绪。
交付时这些目标机结论为 `NOT_PROVEN`，需要在此机器用真实输入完成运行并保存
模型调用、阶段回执、Token 用量、质量门禁与归档哈希后更新。未通过证据门禁的
结果仍保留不确定性，模型掩码和共识不等于动作真值。

`BUNDLE-METADATA.json` 和 `receipts/source-snapshot.json` 记录基础提交、分支及
实际源码哈希。若 `working_tree_snapshot=true`，这是含未提交工作的开发候选，
不代表开发仓已接受或稳定仓已发布的版本。不要把旧机器的 `.engine` 复制进来。

## 故障回传

保留应用窗口报错、`Runtime/Logs/desktop.log`、`Runtime/Logs/connection-verification.jsonl`、`Runtime/target-preflight.json`、
`Runtime/Engines/*/*.build.json`、`Runtime/Logs/web.log` 及该任务的
`JSON-Config-Files` 回执。不要把密钥或原视频当作普通日志发送。
包内文件损坏时重新解压到新目录，不覆盖正在运行的旧目录。
连接验证日志仅记录阶段、进度、耗时与脱敏的失败提示；可区分本地校验、组件加载及云端等待。
本地校验最多等待 30 分钟，云端请求单独等待最多 4 分钟；持续进度不会无限延长等待。

阿里云质量模式接收流式响应，连接验证的流式总预算为 180 秒、无数据等待上限为
30 秒；桌面进程另有 4 分钟外层保护。正式分析使用独立的请求预算，完整结束标记、
JSON 格式与截断检查通过后才接受结果。网络或厂商响应异常仍可能导致超时，连接
通过不能替代真实实验验收。

本轮完整源码同步记录在 `receipts/delivery-refresh.json`。已合入连接修复的完整包
无需再安装旧 R6 连接补丁；旧补丁绑定旧清单，会拒绝应用到更新后的包。文件名保留
原名便于替换交付，实际修订以该回执、`source-snapshot.json` 和 ZIP 的 SHA-256 为准。
Windows 包未包含可选语音识别环境及语音模型，因此默认关闭录音转写，并清除继承的
Ubuntu 语音路径；视频证据分析仍使用完整流水线。启用转写须另行准备目标机语音环境。

连接修复包必须绑定原包 `SHA256SUMS.json` 的哈希；关闭应用后，在独立解压目录双击
`Apply-Update.cmd` 并选择原应用目录。版本、原文件或补丁文件不匹配会在替换前拒绝更新。
修改前备份文件到 `Runtime/Updates`，清单最后替换，写入失败会尝试恢复原文件；
下次启动仍执行完整文件校验。补丁不包含模型、不更改原包其他源码，不代表完整升级。
构建命令：

```bash
python tools/build_rtx4050_offline_package.py \
  --connection-update-from <原桌面包目录> \
  --output <仓库外修复包目录> --allow-working-tree
```

首次交付验收应从清空应用配置的新用户环境开始，记录解压、双击启动、保存位置选择、
AI 验证、引擎准备、网页提交真实实验、任务完成、素材/报告查阅及退出重开结果。
验收绑定最终压缩包哈希及源码快照；本地模拟、历史机器运行与相近版本不能代替此记录。
Windows 运行组件、PowerShell 更新入口、4050 引擎与阿里云请求的目标机验收缺失时，
“开箱即用”仍为 `NOT_PROVEN`。

## 版本依据

- [PyTorch 官方历史版本](https://docs.pytorch.org/get-started/previous-versions/)：2.6.0 的 Windows CUDA 12.4 构建。
- [CUDA 12.4 Update 1](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/)：Windows 配套驱动 551.78；所提供 556.12 高于此版本。
- [TensorRT 支持矩阵](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1040/pdf/TensorRT-Support-Matrix-Guide.pdf)：10.4 平台与运行环境；引擎在目标机生成。
- [Electron 安全规范](https://www.electronjs.org/docs/latest/tutorial/security)：应用窗口使用沙箱、上下文隔离与受限 IPC；内置 44.2.0 x64 运行环境。
- [Python 3.12.10](https://www.python.org/downloads/release/python-31210/)：官方嵌入式 Windows Python。
- [FFmpeg 7.0.2 构建](https://github.com/GyanD/codexffmpeg/releases/tag/7.0.2)：保留构建说明和 GPL 许可于 `vendor/ffmpeg`，避免新 NVENC API 超出现有驱动能力。

依赖许可随各自 `*.dist-info` 保留；本地模型身份由固定 SHA-256 与包清单校验。

## 开发端重建

构建只在开发主机进行。使用 `assets-lock.json` 的 HTTPS 地址准备并校验资产；
SAM2 使用固定 revision 且 `SAM2_BUILD_CUDA=0` 构建纯 Python wheel，CLIP
使用固定 revision。构建 TensorRT 包装 wheel 时必须设置
`NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1`，避免其安装钩子修改构建环境。
将这些 wheel、iopath 与 antlr 的纯 Python wheel 放入 `<build>/wheelhouse`。

```bash
uv pip compile deployment/rtx4050-windows/requirements-offline.txt \
  --python-platform windows --python-version 3.12 --only-binary :all: \
  --find-links <build>/wheelhouse --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match --generate-hashes --no-annotate \
  --output-file <build>/windows-resolved.txt
uv pip install --target <build>/site-packages \
  --python-platform windows --python-version 3.12 --only-binary :all: \
  --find-links <build>/wheelhouse --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match --require-hashes --link-mode copy \
  --requirement <build>/windows-resolved.txt
python tools/build_rtx4050_offline_package.py \
  --build-root <build> --output <outside-repository>/VisionCortex-RTX4050
```

脏工作树需显式 `--allow-working-tree`，输出按文件哈希标记为未提交候选。
构建器拒绝非 Windows 原生库、缺失模型、资产哈希错误与目录大小写冲突，
并生成文件清单、ZIP CRC 检查和压缩包 SHA-256。

重用已核验的基础包源码与运行环境时，可添加 `--base-package <原包解压目录>`；
构建器先验证基础包，再更新桌面入口及启动管理文件。新的回执保留基础包清单哈希。

若同时更新完整应用源码，添加 `--refresh-source`。此选项只重用基础包中已核验的
模型和 Windows 运行环境，重新记录本次应用源码与配置的逐文件哈希。
