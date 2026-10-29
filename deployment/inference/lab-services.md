# 本地实验分析服务

[集成接口](../../docs/reference/agent-integration.zh-CN.md) · [实现边界](../../docs/reference/lab-vision-status.zh-CN.md)

`tools/inference/lab_model_service.py` 提供已有ASR／OCR分析服务的本地HTTP入口，
`tools/inference/lab_model_services.py` 管理明确配置的ASR、OCR和vLLM进程。
这两份源码与统一业务工程一起维护，导入或读取帮助不加载模型，也不启动服务。
仓库不携带权重、私有配置、令牌或运行回执。

优先复用VC／FR已有模型目录与独立解释器，不重新下载或另训一套模型。
ASR环境按显式后端需要现有`qwen_asr`／PyTorch或`faster_whisper`／CTranslate2及音频依赖，OCR环境需要现有PaddleOCR及图像依赖，
VLM环境使用已有vLLM。轻量业务工程安装不自动安装这三套推理环境。
服务代码可以运行不表示当前环境／权重已完成质量或容量验收。

## 私有配置与启动

启动必须显式提供已有、当前用户拥有的本地规范目录`--runtime-root`和私有`--config`。
配置JSON的格式是`vc-lab-model-services/1`，包含：

| 字段 | 内容 |
| --- | --- |
| `token_file` | 已有本地私有令牌文件的绝对路径；不自动生成令牌，禁止组／其他用户访问 |
| `checkpoints` | 名称到`{manifest, manifest_sha256?}`的映射；manifest为本地绝对路径，可额外固定清单SHA |
| `services` | 已启用的`asr`／`ocr`／`vlm`描述；明确选择解释器、检查点与运行参数 |

每个检查点清单包含`model_root`、`revision`与`files`，其中`files`按相对路径记录
`{bytes, sha256}`。`model_root`可以指向现有VC／FR本地权重目录；目录必须规范、非NAS、
没有符号链接，逐文件验证完整清单，不静默切换到其他模型或联网下载。
使用这套服务清单和业务层`vc-lab-models/1`的本地权重描述是两种入口，各自检查各自的运行引用。

服务字段：

- 公共字段：`python`、`port`。解释器未显式提供时使用该运行根下对应`envs/<kind>/bin/python`。
- ASR 默认 `engine=qwen3-asr`：`checkpoint`与`aligner_checkpoint`，使用实际已具备的识别／对齐检查点。
- ASR 可显式选择 `engine=faster-whisper`，并提供 `checkpoint`、`device`、`cpu_threads`、`compute_type`，可选 `language`。例如既有 small 权重可使用 CPU、2线程、int8；不自动换后端或下载模型。该路径返回模型原生词／段相对时间，`alignment_revision=null`，不能标成独立强制对齐结果。
- OCR：`detection_checkpoint`、`recognition_checkpoint`、`detection_model_name`、
  `recognition_model_name`和`device`。本次升级显式选用 PP-OCRv6；旧 PaddleOCR 后端仍兼容历史 PP-OCRv5 配置，不自动修改运行中的服务。
- PP-OCRv6 ONNX：选择 `engine=ppocr-onnx`、`device=cpu`、明确的检测/识别模型名与对应检查点，
  并设置 `input_mode=full_image`（检测后识别）或 `field_roi`（仅对上游已给定字段区域识别）、`cpu_threads` 与有界 `max_recognitions`。
  当前处理器支持 medium 检查点；模型 YAML 名称、处理器、字典及 ONNX 输入/输出形状须匹配。
  检测为零不回退为伪造全图检测框；字段模式不声称自动定位了仪器/面板。超过识别预算直接拒绝本次结果，
  检测候选检查范围保留在响应中。此实现是明确记录预处理的本地 ONNX 路径，不宣称完整 PaddleOCR SDK 或 native/ONNX 数值等价。
- VLM：`checkpoint`、`served_model_name`、`gpu_memory_utilization`、`max_model_len`和`cuda_visible_devices`。

在完整源码检出中执行，以下路径变量由部署方提供：

```bash
python tools/inference/lab_model_services.py \
  --runtime-root "$LOCAL_SERVICE_ROOT" --config "$PRIVATE_SERVICE_CONFIG" status ocr
python tools/inference/lab_model_services.py \
  --runtime-root "$LOCAL_SERVICE_ROOT" --config "$PRIVATE_SERVICE_CONFIG" start ocr
```

同一入口支持`stop`以及一个服务的前台`run`。没有明确启动命令不执行模型加载。
状态与停止按PID、UID、进程启动身份及进程组核对，只处理这份部署拥有的进程；
遇到已有监听端口不会替换其他服务。

## 请求、证据与资源边界

ASR／OCR监听回环地址，提供`GET /health`及Bearer鉴权的`POST /v1/infer`。
请求包含`media_base64`，可带`source_kind`。ASR接受最长180秒原音并返回正文、词／段范围、
对齐版本；OCR返回原文、置信度与框，业务层继续负责面板／字段归属、单位与时序恢复。
faster-whisper 路径有界解码为16kHz单声道，保留原音字节摘要；输入超过180秒拒绝，服务本身不裁剪原音。
结果保留源字节SHA、模型及服务源码版本、耗时、实际调用／缓存命中，不能把缓存命中计作新模型执行。
ASR／OCR 的 health 与结果另提供 `execution_identity`，固定实际校验的完整权重清单及有效后端、设备、精度和线程参数。业务层 `vc-lab-local-providers/1` 可显式固定该64位摘要，健康检查和每次响应都核对；摘要变化应更新提供者配置身份。旧清单仍兼容，但回执明确其未固定服务配置。
未知读数和物理动作不能由服务返回直接确认，质量标记仍是`NOT_PROVEN`。

业务层准备 Opus 等原音时，容器长度与实际音频样本范围分别核对。已证明的编码预跳过或尾部非样本范围可保留在覆盖回执中，不能通过任意放宽时长容差吞掉内部缺口。转写时间从实际样本起点映回原音。模型返回的零时长词保留原文与点时刻；需要区间的事件使用已返回的有效段，缺少有效对齐则只保留实际输入样本范围和 `alignment_unavailable`，不人为扩展词长。

传输、媒体、像素、单结果和回执容量有显式上限；忙碌返回429，容量耗尽停止接纳。
回执保存在本地运行根，由上层按既有发布合同转入NAS；本HTTP服务不自行写客户NAS或数据库。
保留现有离线模式和推理环境，启动前按目标机器核对资源，不停止其他人的模型进程。

VLM使用本机vLLM及其鉴权协议，保留原服务的有界图像输入设置。
图像请求与完整视频片段策略分别记录；不能把该入口的图像窗口当作原视频全帧消费证明。
真实模型调用、现场效果、夜间多路成本及端到端NAS回读仍需使用明确输入另行验收。
