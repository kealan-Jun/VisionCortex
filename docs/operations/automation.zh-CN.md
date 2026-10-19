# 自动采集与生产服务

[文档导航](../README.md) · [项目首页](../../README.md)

当前操作参考；执行前核对所选配置、固定源码 SHA 和适用运行回执。

## 已预配主机的 NAS 自动处理服务

系统、GPU、网络、Receiver、NAS、模型与 Python 环境提前准备好后，交付工程师使用
已审核的客户配置制作服务包。客户解压后执行安装入口：

```bash
tar -xzf VisionCortexAutomation-Candidate.tar.gz
bash VisionCortexAutomation-Candidate/install.sh --setup
```

服务包预先写入该客户的解释器、配置、服务名和端口，安装时先核对全部内容哈希，再安装
常驻后台服务。此入口复用已配齐的环境；候选包不代表正式交付验收通过。工程师打包方式
见[后台服务安装说明](../../deployment/rtx3090ti-ubuntu/README.md#web-lifecycle)。

在已准备好的源码目录内，也可直接安装：

```bash
./deployment/rtx3090ti-ubuntu/09-Install-Analysis-Service.sh
```

客户机器的解释器和配置位置由交付工程师通过 `VISIONCORTEX_PYTHON`、
`VISIONCORTEX_CONFIG` 预先指定，详见[后台服务安装说明](../../deployment/rtx3090ti-ubuntu/README.md#web-lifecycle)。
服务自动发现 Receiver 已发布的 NAS 分片、校验留存原片、推进处理队列并发布设备日
五目录归档，无需打开网页或手工提交。阶段暂停和处理时间窗沿用预配配置。

安装器通过 `/health/automation` 核对所属进程、后台心跳、NAS 监控和处理调度的启动
状态，HTTP 在线不能替代这些检查。此检查只确认后台启动；真实分片归档、目标机重启
和 NAS 中断恢复须另行验收，不能由启动成功推定。

### 处理停滞与恢复

`/health/automation` 的 `processing_health` 与设备日页面分别显示心跳和处理进度。
进度按本地队列区分可执行元数据、输入缺失标记、当前清单未观察到、来源版本不一致、
前置阶段未完成、重试退避和请求结果待核对；这些分类不会删除任务或接受未经验证的媒体。
`stages.<stage>.workload` 按阶段任务统计，不能将跨阶段数量相加当作独立分片数：

- `processing_backlog` 是本系统待处理任务，不含运行中、输入等待、按计划等待及范围外任务。
- `waiting_for_input` 单列输入缺失、输入尚未就绪，以及同来源版本的前序输入等待；不计处理积压。
- `system_attention` 是处理积压中待排查的子集，包含存储访问故障、清单未观察到、来源版本不一致、
  请求结果待核对和失败预算耗尽等；这些问题仍须本系统排查。
- `running`、`scheduled_wait`、`completed`、`outside_processing_scope` 分别保留实际状态。

缺失标记只表示当前采集路径未找到输入，不能据此确认原片丢失或将责任归给输入方；
`input_loss_evidence` 为 `NOT_PROVEN`。实际原片和归档原件的核验仍由输入恢复流程负责。
已完成任务不会因采集路径缺失而降为待处理。旧消费者接口的 `pending` / `blocked` 是原始未完成
状态汇总，不能用作处理积压指标；消费者表仅展示本地可运行数量和运行中任务。
`workload_complete=false` 表示本地状态读取不全，不能把缺失统计视为零任务。

默认有可处理积压且 1800 秒没有新增阶段完成时报告 `no_completed_progress`；可用
`device_day.progress_stall_seconds` 调整，最小 60 秒。等待和长任务均保留实际状态，告警
不代表原片丢失，也不代表模型质量已经验证。

模型提交后的超时或连接断开保留 `submitted_unknown`，同一请求继续禁止自动重复提交。
未知请求身份与响应缓存容量分别核算：没有待接收响应的未决记录不占用空的缓存预留，
实际缓存和仍在执行的请求继续受 `mllm.local_receipt_max_bytes`、
`mllm.local_receipt_total_bytes` 约束。账单和结果核对仍需对应厂商回执，不能靠清空记录恢复。

阶段等待与发布失败采用持久化退避，最长 900 秒和 3600 秒；新执行修订或新的发布令牌
立即重新检查。相同来源代际的已验证回执不反复复制到本地映射数据库。

历史回执可能把同一个完整活动审计重复展开到多个片段。站点可显式设置
`VISIONCORTEX_RECEIPT_PROJECTION_SHARED_AUDITS=1`，流式读取并按完整摘要及内容相等性
恢复共享对象，也支持完整相等的候选证据。原始回执、全部字段、执行键和历史证据保持原样。
唯一对象与单个临时共享对象分别受内存上限约束；该上限默认沿用原配置，可通过
`VISIONCORTEX_RECEIPT_PROJECTION_RESIDENT_BYTES` 显式设置，最多 1 GiB。编码流最多 1 GiB，
每轮临时缓存最多 2 GiB，超限保留旧发布版本。
使用前核对站点磁盘余量；此选项不会放宽输入、来源、产物或模型结果的验证。

独立的可读文件发布可使用标准入口：

```bash
python -m visioncortex.device_day_publication_worker --config <站点配置> --readable
```

该入口复用文件索引发布服务，保留设备日轮转状态和实际构建身份，不启动模型。
标准发布重放在收到停止请求后完成当前分片，剩余条目保留在持久日志中。
一次性格式迁移核验结束后应停用临时迁移常驻扫描，保留游标和修订记录；
日常发现、阶段准入和恢复继续由对应的正式消费者负责。

### 存储切换后的输入封存核验

文件系统重挂载或已验证的存储迁移可能改变设备号、inode，既有封存仍会拒绝直接使用。
显式恢复端口 `input_availability.revalidate_input_seals` 在生产消费者相同的存储视图中，
核对挂载与卷身份、原采集清单、设备/时间/角色、全部源文件的稳定完整哈希后，才可更新
封存中的文件系统身份。文件大小、修改时间、内容哈希和执行输入绑定保持原样；旧封存
逐字节保存在历史目录，核验回执保存在本地 `device-day/InputRevalidation`。

该操作不启动模型、不改采集文件、不清空失败或请求缓存，也不自动授予重试。
内容不同、来源版本变化、归档目标冲突或仍有运行阶段时，须继续隔离排查；
恢复执行仅使用经核验的限定任务和有限重试次数，并保留原失败记录。


## 3090 Ti 局域网服务器

正式使用时，Windows、macOS、Linux 用户都不需要安装本项目，也不需要配置
CUDA、TensorRT 或模型。管理员只在 Ubuntu 3090 Ti 主机完成一次安装，然后运行：

```bash
./deployment/rtx3090ti-ubuntu/07-Install-LAN-Server.sh
```

安装器会要求管理员在终端中设置一次网页登录密码，不回显密码，也不会把密码写入
Git。完成后会显示类似 `http://192.168.x.x:8000/#/home` 的局域网地址。其他用户
只需打开该地址，以用户名 `visioncortex` 和管理员设置的密码登录，即可选择 NAS
批次、提交任务、查看进度和结果。多人同时提交时，3090 Ti 每次只执行一个正式
GPU 任务，其余任务保留为“排队等待”，避免互相争抢显存。排队内容与页面任务
状态先写入服务器本地 `Runtime/state/web_run_queue.sqlite3`；Web 服务或服务器重启
后会自动恢复等待任务，不需要用户重新提交。

浏览器上传不再使用一次性大请求，也不按路数、时长或固定总 GB 数拒绝任务。页面先
把本次全部文件的真实字节数交给服务器；服务器结合 NAS 当前剩余空间、其他上传
尚未消耗的预留量、预计处理产出和安全余量，决定是否接收。通过后以 16 MiB 小块
续传，网络中断会从服务器确认的字节位置继续；重新选择同一批文件后也会核对断点
内容，避免把不同文件拼在一起。3090 Ti 生产配置只在 NAS 正式档案保存一份原片，
不在服务器本地再复制一份。连续 7 天没有继续的未完成会话会在服务启动或下一次
容量检查时过期并清理；已经进入任务队列的会话不受该规则影响。

该服务使用 3090 Ti 正式生产配置，启动前检查 GPU、NAS、固定 Python 环境和本地
空间；检查不通过时拒绝启动。它只接受回环地址和常见私有局域网地址，并要求每次
浏览器会话登录，不是公网发布方案。服务状态检查：

```bash
./deployment/rtx3090ti-ubuntu/08-Server-Status.sh
```

普通用户通过浏览器访问管理员提供的服务地址。RTX 4060 是冻结执行节点，
按运行任务指定的不可变 SHA 执行并回传证据。真实模型、视频质量和正式归档
能力仍须按对应主机的门禁验收；网页能打开不代表完整推理已经通过。

归档查阅不会在每次打开页面时重新遍历并读取全部 JSON。服务会把正式归档中已有的
`evidence_index.sqlite` 同步为 3090 Ti 本地运行目录下的可重建读取索引；归档 JSON
和逐归档 SQLite 仍是权威数据，读取索引损坏或删除后会自动重建，不复制原视频。
归档目录、实验片段和关键事件均分页加载，中文动作词会扩展到现有动作本体的中英文
同义词；素材 URL 绑定当前 `release_id`，旧版本链接会返回冲突提示而不会继续展示
浏览器缓存。关键帧延迟加载、关键片段不预加载正文，服务默认最多并发传输 8 个归档
文件；可用 `VISIONCORTEX_WEB_MAX_CONCURRENT_ARCHIVE_STREAMS` 调整，超出时返回
`429` 并提示浏览器稍后重试，避免多人播放把 NAS 链路拖死。

安装器创建的单一 `visioncortex` 管理账号继续兼容。多人使用时，可改用权限为 `600`
的 `VISIONCORTEX_WEB_USERS_FILE`，每个人使用独立账号和 `viewer`、`operator` 或
`admin` 角色；`viewer` 只能查阅，后两者可提交任务。密码文件只保存 PBKDF2 哈希，
可在服务器终端生成：

```bash
visioncortex hash-web-password
```

用户文件格式如下，不能写入明文密码，也不能提交到 Git：

```json
{
  "schema_version": "visioncortex-web-users/1",
  "users": [
    {"username": "researcher-01", "role": "viewer", "password_hash": "<PBKDF2_HASH>"},
    {"username": "operator-01", "role": "operator", "password_hash": "<PBKDF2_HASH>"}
  ]
}
```

局域网 API 的账号、角色、来源 IP、方法、路径、状态码和耗时会按日写入服务器本地
`Runtime/state/web_access_audit-YYYY-MM-DD.jsonl`，不记录密码、Authorization 或
查询参数。若通过 Nginx/Caddy 等受控反向代理提供 TLS，可设置
`VISIONCORTEX_WEB_REQUIRE_HTTPS=true` 强制拒绝非 HTTPS 请求；当前安装器显示的
`http://192.168.x.x` 仅适用于受信任、隔离的内网，不能作为公网入口。
