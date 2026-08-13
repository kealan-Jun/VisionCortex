# LabVision Evidence

> 冻结基线：六路、多视角、3 小时湿实验视频的时间对齐、有界实验筛选、五类关键素材和细粒度步骤理解流水线。RTX 4060 部署、固定 NAS 基准、缓存目录与开发协作方式见 [RTX4060-交接与运行说明.md](RTX4060-交接与运行说明.md)。

固定基准不会重复创建实验目录：`labvision run-fixed-benchmark --config configs/rtx4060-laptop-production.yaml` 复用 `Y:\VisionCortexExperimentArchive\Six-View-Three-Hour-Experiment-2026-08-13`。其他用户上传任务仍按实际上传路数动态创建独立档案。

面向化学湿实验长视频的多视角证据流水线。系统把第一人称与第三人称视频先对齐，再以两套 21 类 YOLO 模型和 ByteTrack 生成候选，经过跨视角审计后，提取实验片段、五类物理动作关键帧/关键片段/时间戳，并调用豆包多模态模型生成步骤级理解。

## 产出

每次运行生成以下结构：

```text
<output>/<experiment-id>/
  Experiment-Clips/
    EXP-0001_<start>-<end>/
      EXP-0001_aligned_multiview.mp4  # 主交付：严格等长、同一全局时间轴
      first_person.mp4
      third_person.mp4
      EXP-0001.json
  JSON-Config-Files/
    run_manifest.json
    time_alignment.json
    aligned_timestamps.csv
    evidence_package.json
    physical_change_log.json
    evidence_package_eval.json
    run_metrics.json
  Key-Materials/
    Key-Clips/<event-id>/first_person.mp4, third_person.mp4
    Key-Frames/<event-id>/first_person.jpg, third_person.jpg
    Key-Materials-Model-Understanding.json
    Key-Material-Timestamps.csv
    Key-Material-Timestamps.xlsx
    Screening-Notes.txt
  Lab-Daily-Reports/<YYYY-MM-DD>/
    Lab-Daily-Report-<date>.json, .md, .html
    Daily-Report-Eval.json
    Human-Review.json
  Professional-PDFs/
    Lab-Daily-Report-<date>.pdf
```

日报采用固定的 `VC-LAB-DAILY-REPORT-V1`：模型只产出结构化实验理解，确定性渲染器填充固定栏目，日报阶段不新增模型调用或 Token。模板栏目、版本策略和本地开发方式见 [实验室日报固定模板 V1](docs/daily-report-template-v1.md)。

五类动作是 `hand_object_contact`（手与明确物体接触）、`object_movement`、`liquid_movement`、`container_state_change` 和 `device_panel_operation`。每个记录保留候选、接受/拒绝理由、视角支持、对齐置信度和不确定性，YOLO 框不会被直接当成最终证据。

## 安装与安全配置

```powershell
Set-Location -LiteralPath 'C:\Users\Xx7\Documents\ChatGPT\New project'
python -m pip install -e ".[dev]"
$env:ARK_API_KEY = '<在本机安全设置，不要写入 yaml 或 git>'
```

聊天中出现过的 API Key 应在火山方舟控制台轮换。代码只读取 `ARK_API_KEY`，不会保存或打印密钥。

## 输入清单

复制 `examples/manifest.example.yaml`，为每一路填写视角、视频和 CSV。CSV 支持常见列名：

- 帧号：`frame_index` / `frame` / `frame_id`
- 时间戳：`timestamp_ms` / `timestamp_s` / `timestamp` / `pts_time`
- 可选共同时间：`global_timestamp_ms` / `wallclock_ms`

时间戳为 ISO-8601 时也可解析。若 CSV 没有共同时间列，系统以视频起点为粗对齐，并用跨视角运动变化序列进行视觉锚点互相关校准；如果已有共同时间，先做最近邻鲁棒仿射拟合，再用视觉锚点修正残余偏移。

## 运行

```powershell
labvision validate-models --config .\configs\default.yaml
labvision prepare-engine --config .\configs\default.yaml
labvision run --manifest .\examples\manifest.example.yaml --config .\configs\default.yaml
```

没有真实视频时可完整验证目录、JSON 契约和评估器：

```powershell
labvision dry-run --output .\outputs\dry-run
pytest -q
```

上传服务：

```powershell
labvision serve --host 127.0.0.1 --port 8000
```

`POST /api/runs` 以 `videos[] + view_specs_json` 接收任意多路视频及 CSV，后台运行后从 `GET /api/runs/{run_id}` 查询状态。API 只绑定本机，除非显式改为 `0.0.0.0`。

容量目标按至少 **6 路 × 每路 3 小时** 设计：600 秒一个可恢复分块，最多 6 路并行解码，但同一角色的帧合并成有界 GPU batch；检测结果逐行落盘，重启后跳过已完成分块，因此内存/显存占用不随视频时长增长。磁盘预检会在开始前估算原片、临时候选和交付片段所需空间，不足时拒绝启动并给出缺口。

6 路输入不等于 6 路输出。系统逐路做“有效实验视角门控”：只有持续手—物体操作、容器/设备状态变化或物料转移等证据达到阈值，且边界内动作密度合格的视角才会生成实验 MP4。空镜、纯走动/穿戴、等待、长静止、无实际实验操作的视角会在筛选备注中记录拒绝原因，不进入后续关键素材提取。

## 性能策略

- 优先使用已存在的 `.engine`；否则使用 `.pt` + CUDA FP16。
- `prepare-engine` 以当前图像尺寸和 batch 上限导出 TensorRT engine。
- FFmpeg 优先 NVDEC 解码和 NVENC 裁切；不可用时自动回退 CPU 解码/编码。
- 六路解码生产者并发工作；第一/第三人称两套模型同时常驻 GPU，分别消费有界队列并做角色内 batch。显存超过预算时各扫描器自动减小 batch，避免 6GB 显存 OOM。
- 全量粗筛默认 `0.125 FPS / 512px`，任一路候选会按全局时间扩散到所有输入视角，再以 `8 FPS / 960px` 精扫、判定每路有效性并收紧边界；最终关键帧和裁切始终从原始视频精确取时间戳。

`run_metrics.json` 记录总墙钟耗时、各阶段起止/耗时，以及每个关键素材调用的输入 token、输出 token、总 token、缓存命中 token、延迟和重试次数；随后汇总关键素材阶段与整次运行。Token 只采用服务端 `usage`，缺失时保留 `null`，不做伪精确估算。

预处理 SLA 定义为视频探测、时间对齐、1 FPS 全量粗筛、8 FPS 候选窗精扫和边界审计，默认目标为 1200 秒（20 分钟）。`run_metrics.json.preprocessing_sla` 保存目标、实测值和是否达标；裁片编码与豆包调用不计入预处理时间。

系统无法凭空保证“精确”：液体本身不在 21 类标签中，因此液体移动由容器/移液器目标、ROI 光流和跨视角一致性共同提出候选，再交给多模态模型确认；低置信或仅单视角可见的事件会明确标为不确定，而不会伪装成确定结果。
