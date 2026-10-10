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
