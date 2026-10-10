# VisionCortex 独立机器接口与 GPU worker

受认证的机器接口复用持久任务队列、输入封存和正式发布流程，支持 Web 与 GPU worker 分离部署。
服务地址、调用方身份、来源范围和存储根由私有站点配置提供。

## 启动入口

API：
~~~sh
VISIONCORTEX_CONFIG=/path/to/pipeline.yaml \
VISIONCORTEX_MACHINE_CONFIG=/path/to/machine.json \
python -m uvicorn visioncortex.machine_api:create_app --factory --host 127.0.0.1 --port 8001
~~~

GPU worker：先在私有站点环境中设置目标 `GPU_UUID` 和本地锁目录 `GPU_LOCK_DIR`。
~~~sh
python deployment/inference/gpu_guard.py \
  --config /path/to/pipeline.yaml \
  --gpu "$GPU_UUID" --lock-dir "$GPU_LOCK_DIR" -- \
  python -m visioncortex.machine_worker --config /path/to/pipeline.yaml
~~~

machine_worker 复用已有 worker 命令，但要求持有 GPU 生命周期锁，且拒绝开启
device_day.enabled / collection_ingest.enabled。新服务只消费明确提交的任务；
不能继承某现场配置后自动扫描整个 NAS。原自动采集/设备日服务仍按原方式管理。
启动器传入 GPU UUID、已持锁的文件描述符及锁文件路径，worker 校验锁文件身份并保持
排他锁后才启动。历史启动器只有文件描述符和 UUID 时，Linux /proc 用于核对同一锁文件。

## 配置和存储

公开 `pipeline.example.yaml` 是未准备的安全模板，只用于接口和队列检查。
实际 worker 使用私有 `pipeline.yaml`，显式设置 `project.site_configuration_required=false`
和 `project.run_purpose=analysis` 或 `production`，并提供已经验证的资产、路径及所选厂商配置。
未配置模板会在 GPU 锁和设备查询前拒绝执行。

从已验证模型环境生成专用 pipeline.yaml。沿用正确模型 registry、权重、
TensorRT engine 及推理参数，逐项设置以下实际路径和开关，不复制旧主机用户名/盘符：

- storage.local_runtime_root=/runtime，队列 SQLite 必须在本地持久盘，不能放 SMB。
- 本地无 NAS 配置使用 runtime.local_only=true，storage.archive_root=/runtime/archive，
  input/cache/staging/output 根全部位于 /runtime 内。授权输入按作用域从 /inputs/video 只读引用。
- device_day.enabled=false、collection_ingest.enabled=false。
- runtime.role=web 用于配置/API；machine_worker 启动时切换为 worker。
- 本地验收使用本地输出且关闭 NAS 同步；正式 NAS 发布须另行审核授权的配置，显式设置
  runtime.local_only=false 及已获准的独立结果根（例如挂载后的 /outputs）。
- 未配置外部模型凭证时保持 mllm.enabled=false，不伪造完整分析成功。
  真正全流程验收需要实际模型、数据和有效的外部阶段配置。

复制 config.example.json 为 machine.json，用随机机器凭证的 SHA-256 替换占位值。
每个 client 的 scopes 列出其可读相机根；default_scope 用于兼容当前 Gateway
只发送 Authorization 的请求。Gateway 仍负责最终用户/设备/资产授权；
若不同调用方不应共享任务，使用不同机器凭证或显式 X-Execution-Scope。
配置目录、凭证和数据库权限限制给服务账号，不把原始凭证写到 Git。

输入按相机根只读挂载。这里拒绝输入符号链接；录制系统使用原片软链接的目录
需由部署者提供直接指向已授权原件的只读映射，不能为跑通放开整个主机/NAS。
参数只指定固定文件路径，实际合法性和输入封存仍由原流水线校验。
文件不存在或 NAS 断连不会当成空输入。源文件不可由本服务修改。

## Compose

CPU API 使用 Dockerfile.api，可从源码独立构建，不需要先准备 GPU 镜像。
Worker 的 Dockerfile 使用 VISION_RUNTIME_IMAGE 指定的已验证 VisionCortex 镜像，
复用其模型依赖；不会借 OCR 镜像混装 Paddle/TensorRT。首次部署可在目标 Linux x86_64 上
运行 `docker build -f deployment/inference/Dockerfile.runtime -t visioncortex-runtime:deployment .`，
按原 Ubuntu 依赖锁构建运行环境。此入口不含权重/engine，不自动运行模型，目标机仍需验证
驱动/实际模型兼容并固定镜像 digest。可选语音依赖需按已有 speech 配置另外安装。

两种镜像都显式指定容器内默认配置文件，避免安装 Python 包后寻找不存在的 site-packages/configs。
`pipeline.example.yaml` 提供受本地运行根约束的容器路径基线，自动扫描关闭，可用于 API/队列检查；
不能直接当作真实生产推理配置，必须合入已验证的模型/阶段设置。

从 deploy.env.example 复制未跟踪的私有部署环境文件：
VISION_RUNTIME_IMAGE、VISION_CONFIG_DIR、VISION_INPUT_ROOT、VISION_RUNTIME_ROOT、
VISION_OUTPUT_ROOT、VISION_MODEL_ROOT、GPU_LOCK_DIR、GPU_UUID。
所有宿主目录预先创建并确认权限及 NAS 实际挂载。
VISION_CONFIG_DIR 中必须有 pipeline.yaml 和 machine.json。
Compose 拒绝自动创建缺失的挂载路径。两进程共享 Vision 本地 runtime，不能改成各自独立空卷。

~~~sh
docker compose --env-file /path/to/private/deploy.env -f deployment/inference/compose.yaml config --quiet
docker compose --env-file /path/to/private/deploy.env -f deployment/inference/compose.yaml up -d api
docker compose --env-file /path/to/private/deploy.env -f deployment/inference/compose.yaml --profile gpu up -d worker
~~~

API 只绑定 127.0.0.1:8001，跨机使用可信 TLS 反代。GPU worker 不对外开端口。
运行镜像必须支持 nvidia-smi/utility 和配置的驱动。服务使用 UID/GID 10001:10001，与 OCR 相同；配置/模型需可读，runtime/outputs/锁目录需可写。模型、输入只读；
本地样例在 runtime 内写归档；独立 /outputs 挂载仅供另行审核的生产输出配置使用。
API 容器只安装本项目 CPU 依赖，不加载模型。
不同服务不能使用同一个 Python 环境。

## 与当前 AgentDesk 兼容

保留路径：
POST /api/runs/from-paths
GET /api/runs/{id}
GET /api/archives/{name}
GET /api/key-events?archive=…
GET /api/key-events/{uid}?archive=…
GET /api/evidence/{uid}?archive=…
GET /api/health

所有上述接口需 Bearer；/health/live 仅返回进程活性。
没有开放原管理、设备日、任意文件或全局归档列表 API。
事件/证据必须指定归档，并检查该归档属于此 client/scope 已完成的任务。
其他调用方的 run_id/归档即使猜中也拒绝。根权限缩小后旧结果访问也会重新检查。

提交支持稳定 Idempotency-Key；缺省使用 experiment_id，兼容原 Gateway
将 collection_id 和平台 job UUID 放入 experiment_id 的实现。
同键同内容返回原 12 位 run_id；不同内容/服务配置版本返回 409。
GET /api/run-submissions/{key} 可找回丢失回执。返回 queue_persistence=sqlite、
state=queued；准备期间 phase=preparing_input。API 先持久保存 run/archive 身份，
后台封存输入并转入原生队列。中断后重用原 run/archive，不重新命名第二份。
原生入队成功但机器回执尚未更新时，以原生持久任务为准恢复。

恢复准备期间检查接纳时的配置/构建摘要；若配置已改变，任务明确失败
CONFIGURATION_CHANGED，不用新模型偷偷解释旧请求。已进入原队列的任务遵守
原队列的封存配置和恢复规则。输入无效等失败仍可查询，网络重试不重新启动。

原生状态与正式发布仍由已有流水线决定。新接口不把 dry-run/partial
改成 completed，不绕过质量门禁，不另造分析报告或第二套 RAG。
当前 AgentDesk 的受保护 Web 结果策略继续在业务机生效。

接入原 Gateway 时，只合并 analysis 的 baseUrl、authorizationEnv、
pathFlavor=posix、storageRoots；保留 cameras/readers/managers。
analysis.authorizationEnv 的值是完整 Bearer 认证值。
部署/地址变更前检查旧 job 的 provider fingerprint，不改数据库伪造兼容。

## 单 GPU 与回滚

可由部署方的独立 GPU 调度器切换需要共享显卡的任务。调度器需遵守共同的锁和排空协议，
其来源、部署配置和运行证据由站点单独维护。
worker 响应 `VISIONCORTEX_DRAIN_FILE`：处理完当前任务后不再领取新任务并退出。
兼容历史 `REALITYLOOP_DRAIN_FILE`；同时设置时以 `VISIONCORTEX_DRAIN_FILE` 为准。
GPU 锁协议优先使用 `VISIONCORTEX_GPU_UUID`、`VISIONCORTEX_GPU_GUARD_FD`、
`VISIONCORTEX_GPU_LOCK_PATH`，兼容对应 `REALITYLOOP_*`；启动器同时导出两组字段，
保留共享 OCR 调度器的现有宿主 `GPU_LOCK_DIR` 及 drain 文件路径约定。
到管理员设定的收尾上限仍未退出时，调度器请求停止，复用已有队列/断点恢复；强制结束
可能需要重算，不承诺无损暂停任意推理指令。API 和提交队列全天在线。
Compose 已提供按 GPU UUID 区分的 drain 路径；共享锁目录仍须是本地持久目录。
计划以部署方验证过的 GPU 调度配置为准，初始关闭。
下述手动切换方式仅在调度器未接管时使用。

OCR 与 Vision 使用完全相同的 gpu_guard.py 协议和宿主 GPU_LOCK_DIR；
同一个 UUID 只有一个模型进程运行期。先排空/停止 OCR worker，再启动 Vision；
Vision 停止并释放 GPU 后恢复 OCR。两个 API 可一直工作，OCR 请求在分析期间排队。
双 GPU 分别指定真实 UUID。不要把 SQLite/Redis 租约当成显存释放证明。

已有不受该启动器管理的 GPU 进程会导致启动拒绝，不能擅自终止它。
Docker init/进程组收尾应保持启用；不强杀 supervisor 后留下 CUDA 子进程。

回滚时停止新提交和本次 worker，保存 runtime/结果卷与一致性 SQLite 备份
（backup API 或停写后备份，保留 WAL），切回兼容程序和配置，再启动。
不能 down -v、删除 NAS 数据、用旧空库覆盖新任务。
源码更新不会升级已部署服务。GPU、模型、存储、视频质量、浏览器联调及正式发布资格
按当前固定源码和部署身份分别验收，使用[交付验收单](../../docs/VisionCortex-交付验收单模板.md)记录实际结果。
