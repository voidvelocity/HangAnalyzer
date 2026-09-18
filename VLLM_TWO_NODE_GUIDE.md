# 两机 vLLM-Ascend 卡死现场接入与定位指南

本指南交给在真实 Ascend 950PR / CANN 9.2.0 环境工作的 agent。目标是保留请求进入、长序列调度、PP、split KV、HCCL、超时和退出全过程的 Host 证据，以及**已注册的实际 Stream** 的 Device 完成水位。现新增统一 msPTI ring，可在适用的 worker 启动方式下通过 `LD_PRELOAD=libmspti.so` 与 `sitecustomize` 观察 CANN Runtime/HCCL 调用和已完成设备算子；它不会自动获得 Scheduler、请求 ID、PP/KV 语义，也不是 CANN 内部任务队列。完整边界、命令与验证见 [统一 msPTI 采集器指南](MSPTI_FLIGHT_RECORDER.md)。

现场交付物应包含：接入代码 diff、软件版本和启动参数、global Rank/节点/PID/Device/PP stage/通信组/Stream 映射、请求 ID 与时间线、两机原始 `.flight` 和最早的活体 `stalled-*` 快照、每个相关通信组的可读报告，以及结论和证据边界。除已获授权的故障复现外，不终止无关服务。不要记录原始 prompt、KV 内容或凭据。

## 1. 固定版本和拓扑

在两台机器上保存：vLLM、vLLM-Ascend、PyTorch、torch_npu、CANN、驱动版本及包路径/commit、完整服务启动参数，以及 global Rank → 节点、PID、Device、PP stage、HCCL communicator 成员的映射。另存 compute、PP、KV/transfer、HCCL Stream 的完整 handle 和本工具使用的 32 位 stream ID；一个 worker 内 ID 必须唯一。明确 API 进程到 worker 的 request ID 传递方式，跨进程 hash 应使用稳定的 64 位算法，不能用 Python 内置 `hash()`。每次实验使用唯一目录，例如节点 0 的 `/data/hang/incident-001/node0/` 与节点 1 的 `node1/`。容器内目录应挂载到宿主机，避免容器退出丢失现场。

两台机器用相同源码编译：

```bash
cd /path/to/HangAnalyzer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCANN_HOME=/path/to/CANN-9.2.0
cmake --build build -j
ctest --test-dir build --output-on-failure
```

确认生成 `build/libflightrecorder.so`、`build/libflightcheckpoint_cann.so` 与 `build/libhangmspti.so`，worker 能加载 CANN 运行库。两机都要在目标版本重新编译；A2/CANN 9.1.0 的 `.so` 不应直接复用。在可用测试卡上分别运行 [msPTI 单卡 smoke](tests/a2_mspti_ring_smoke.py)与 [checkpoint 真机 smoke](tests/a2_checkpoint_smoke.py)。前者必须从进程启动时预加载目标版本的 `libmspti.so`；不预加载时 Runtime Domain 会报错 6。后者验证目标 CANN 的 Event query、后台线程与 generation，提交 30 次 1024×1024 矩阵乘。选择可承担负载的卡和新输出目录：

```bash
python3 tests/a2_checkpoint_smoke.py \
  --recorder-library "$PWD/build/libflightrecorder.so" \
  --checkpoint-library "$PWD/build/libflightcheckpoint_cann.so" \
  --directory /data/hang/preflight-checkpoint-node0 \
  --device <available-local-device-id>

LD_PRELOAD=/path/to/CANN-9.2.0/lib64/libmspti.so PYTHONPATH="$PWD" \
python3 tests/a2_mspti_ring_smoke.py \
  --library "$PWD/build/libhangmspti.so" \
  --output /data/hang/preflight-mspti-node0.msflight \
  --device <available-local-device-id>
```

若目标版本的编译或 Event query 验证失败，保存错误，先只启用 Host recorder；不要把未验证的 Device confirmation 用于结论。

### 1.1 统一 Runtime 与 Device ring

正式启动服务前，在两机每个**最终 worker** 都启用统一采集。spawn/exec 式 worker 可按 [统一采集器指南](MSPTI_FLIGHT_RECORDER.md)配置 `LD_PRELOAD`、`PYTHONPATH`、`HANG_MSPTI_*` 与 `/home/enable_prof`；fork 式 worker 需在 fork 完成后显式调用 `hang_mspti.start()`，不能继承父进程的 msPTI 订阅。先让控制文件为 `0` 拉起服务，再设为 `1` 并发短请求，确认每个 PID 的 `.msflight` 出现 `runtime_enter`、`kernel_done`，且无 `.error`。真实故障请求前重新设为 `1`。运行中每 0.5 秒 flush 已完成 Activity；若 msPTI 自身卡在 flush，mmap 中已经发布的 Callback 仍可由外部读取。

故障后，每个 worker 直接查看：

```bash
PYTHONPATH=/path/to/HangAnalyzer python3 -m flightrecorder.mspti_ring \
  /data/hang/incident-001/node0/mspti/mspti-<PID>.msflight --tail 100
```

两机文件按节点、PID 保留。`.msflight` 里的 Callback 是 CANN API 层，`.flight` 里的显式埋点是 vLLM 语义层，Event checkpoint 是注册 Stream 的完成确认。分析时按这三个层次合并证据，不把 Callback exit 或缺失 Activity 直接解释成 Device 已完成/未执行。该自动路径在 A2/CANN 9.1.0 已验证，在 950PR/CANN 9.2.0 必须做本节的预检后才能用于结论。

## 2. 最小接入点

每个 NPU worker 在启动/fork 完成、选择 NPU Device 后创建自己的 Recorder，使用 **global Rank**。不要在父进程初始化后把 Recorder 或 Event manager 传入子进程；每个 worker 应写独立 `.flight`：

```python
from flightrecorder import Recorder

flight = Recorder(
    directory="/data/hang/incident-001/node0/flight",  # 节点 1 改为 node1
    rank=global_rank,
    device=local_device,
    library="/path/to/HangAnalyzer/build/libflightrecorder.so",
)
```

在部署版本源码中找真实调用位置，不依赖本指南中的固定函数名：

```bash
rg -n 'execute_model|scheduler|kv_connector|kv_transfer|send|recv|all_reduce|all_to_all|wait_event|record_event|synchronize' \
  /path/to/vllm /path/to/vllm_ascend
```

至少在下表的真实调用前后记录 Begin/End：

| 阶段 | 事件 | 关联字段 |
|---|---|---|
| 请求接收/结束 | `REQUEST_BEGIN/END` | 同一 request ID 的稳定 64 位哈希，`arg0=prompt_tokens` |
| 调度与模型执行 | `SCHEDULER_*`、`MODEL_*` | scheduler step ID |
| Pipeline send/recv | `PP_SEND_*`、`PP_RECV_*` | transfer ID、peer global Rank、字节数、Stream |
| KV connector 传输/等待 | `KV_BEGIN/END` | transfer ID、layer/block/request 映射、Stream |
| HCCL API | `HCCL_BEGIN/END` | 共同 communicator ID、组内共同 operation ID、collective 类型 |
| Stream Event | `EVENT_RECORD/WAIT` | 进程内逻辑 Event ID 与本地 Stream ID |
| Graph replay | `GRAPH_BEGIN/END` | graph ID、batch/shape 等非敏感参数 |
| 同步 API | `STREAM_SYNC_*`、`DEVICE_SYNC_*` | 调用前 BEGIN，成功返回后 END |

Begin/End 必须使用相同 `correlation` 和 `stream`。跨 Rank 的 communicator/operation ID 必须在同一通信组内一致；本地递增计数或指针值不能直接用于跨 Rank 对齐。建立并保存 correlation、collective 类型和 Stream ID 字典。不要写原始 prompt。`HCCL_END` 仅表示 Host API 返回。只有原本已有的同步/查询成功后才可手工写 `DEVICE_CONFIRMED`；不要为了诊断额外插入设备同步，以免改变卡死条件。长 prefill 可在 scheduler step、layer group、PP/KV 边界添加低频阶段事件，避免正常运行期间长时间没有 recorder 事件。

若 API Server 与 worker 是不同进程，需把 request hash 传到 worker，或在 API 日志和 worker 埋点中保留同一 request ID；否则报告不能从 HTTP 收到请求的时刻直接关联到 Rank。

### 2.1 在实际 Stream 增加 Device checkpoint

Event checkpoint 是当前工具已实现的能力，详见 [CHECKPOINTS.md](CHECKPOINTS.md)。它使用每 Stream 固定 Event 池、非阻塞 `aclrtQueryEventStatus` 和不依赖 Python GIL 的 C++ 后台 poller。先取得真实 Stream handle，在流量开始前注册；一个 worker 内的 32 位 stream ID 必须唯一，截断指针后先检查冲突，并保留完整 handle 映射。Stream handle 在 manager 生命周期内必须有效。

```python
from flightrecorder import CannCheckpointManager

checkpoints = CannCheckpointManager(
    "/path/to/HangAnalyzer/build/libflightcheckpoint_cann.so",
    slots_per_stream=4,
)
compute_handle = int(compute_stream.npu_stream)
compute_id = compute_handle & 0xffffffff
checkpoints.register_stream(compute_id, compute_handle)
# 对可以取得真实 handle 的 PP、KV/transfer、HCCL Stream 分别注册。
checkpoints.start_poller(interval_us=1000, budget=32)
```

异步任务全部提交到对应 Stream 后，记录阶段结束 Host 事件，用返回 seq 在**同一 Stream 尾部**记录 Event。对不同阶段设置可查字典的 checkpoint ID：

```python
submitted_seq = flight.record("MODEL_END", stream=compute_id, correlation=step_id)
generation = checkpoints.submit(
    compute_id, submitted_seq, checkpoint_id=model_checkpoint_id)
```

Event 完成确认会携带 `(stream_id, checkpoint_id, submitted_seq, generation)`；同一物理 Event 复用后 generation 递增。池满返回 `EAGAIN` 并记录 `POOL_FULL`，该 checkpoint 没有成功提交。池容量应覆盖两次轮询之间的最大 pending 数。业务线程可能卡在 HCCL 或同步 API，后台 poller 仍能查询；它不会执行 Stream/Device synchronize。正常关闭时先 `checkpoints.close()`，再 `flight.close()`；pending Event 不能强制销毁或靠额外同步清理。

**关键证据边界：**只对 Event 所在的实际 Stream 作完成推断。PyTorch 默认计算 Stream 的 checkpoint 已完成，不能证明框架内部 HCCL 或 KV Stream 已完成，除非已核实跨 Stream 依赖。A2 四卡实测出现默认 Stream checkpoint 7102 已完成、设备同步仍卡住；[报告](results/a2-fourcard-checkpoint-20260916/report/report.txt)给出了 `COVERAGE WARNING`。若无法取得内部 Stream handle，应明确标出覆盖缺口，不要写成“设备/HCCL 已完成”。

## 3. 两机各启动 watchdog

在发请求前，节点 0 执行：

```bash
python3 -m flightrecorder.watchdog \
  --directory /data/hang/incident-001/node0/flight \
  --output /data/hang/incident-001/node0/dumps \
  --timeout 30 --interval 0.2 --npu-smi \
  --log /path/to/vllm.log --log /path/to/hccl.log
```

节点 1 把 `node0` 改成 `node1`。若 worker PID 已知，可重复传 `--pid PID`；动态产生 PID 时可以省略，实验结束手动停止 watchdog。容器需有读取 worker `/proc` 和执行 `npu-smi` 的权限；权限不足时快照中会有 `.error`。`--timeout` 必须高于正常最长无埋点间隔；单条长序列先校准阈值或增加阶段性埋点。每个 Event generation 的 NOT_READY 最多记录一次，后台 poller 不会持续刷新停滞计时。watchdog 默认不会杀服务。

## 4. 校准并复现

先发短请求，确认两机预期的每个 worker 都生成 `.flight`，Request/Scheduler/Model 的 Begin/End 成对，Rank 与通信组映射正确。检查 `checkpoint_mappings` 中 Stream/generation/confirmation 与真实调用吻合。再发受控的正常长序列，确认 watchdog 不会误判。最后用相同服务启动参数发一条故障长序列，保存请求发送时间、request ID/hash、输入 token 数、客户端响应/错误、服务退出时间；请求体如果敏感，只保留安全的哈希与长度。例如：

```bash
curl --max-time 180 -sS -D response.headers \
  -H 'Content-Type: application/json' --data-binary @long_request.json \
  http://<node0-api-host>:<port>/v1/completions > response.json
```

优先保留最早的 `stalled-*` 活体快照（含线程现场）；服务退出时的 `exit-*` 是补充。`stalled` 仅表示超过配置阈值没有新的埋点，不能单独证明死锁。复制两机完整 incident 目录、拓扑表和日志字典。跨节点时间戳只有在时钟校准后才可直接排序；优先用每 Rank 的 seq 和显式 correlation 建立因果关系，不要混合不同 incident 的快照。

## 5. 合并与分析

将两机快照放到同一父目录：

```text
/data/hang/incident-001/merged/node0/dumps/...
/data/hang/incident-001/merged/node1/dumps/...
```

对一个**确定的 communicator**及其成员集合分析，以下数字仅为示例：

```bash
python3 -m flightrecorder.analyzer /data/hang/incident-001/merged \
  --communicator-id 700 --expected-ranks 0,1,2,3,4,5,6,7 \
  --output /data/hang/incident-001/report-comm700
```

不同 PP stage、TP/HCCL group 应分别用真实 `communicator-id` 与成员集合分析。将 world ranks 误当成每个通信组的成员会产生假 `missing_expected`；CLI 因此要求 `--expected-ranks` 同时指定 `--communicator-id`。

## 6. 人工判读

```bash
cat /data/hang/incident-001/report-comm700/report.txt
less -S /data/hang/incident-001/report-comm700/events.txt
cat /data/hang/incident-001/report-comm700/snapshots.txt
```

先看 `HOST PROGRESS` 的 `last_host` 与未闭合 scope；`last_event` 包含后台 probe，可能晚于业务线程卡点。再看 `DEVICE CHECKPOINT PROGRESS` 的逐 Stream `submitted_seq`、`completed_seq`、`first unconfirmed` 和未确认区间。`report.json` 的 `checkpoint_mappings` 可逐项核对 `(stream, checkpoint_id, submitted_seq, generation)` 与 confirmation。随后看 `COLLECTIVES` 中同一 operation 谁进入、谁缺席、类型是否一致，最后看 `BLOCKING EVIDENCE` 与最早 `stalled-*` 下的 `metadata.json`、`pid*/tid*.stack`、`.wchan`、日志尾部和 `npu-smi.txt`。对照 [已验证的四卡 checkpoint 报告](results/a2-fourcard-checkpoint-20260916/report/report.txt)：Rank 6 停在 `KV_BEGIN`，其他 Rank 停在 `DEVICE_SYNC_BEGIN`，已观察默认 Stream 全部确认仍不能证明内部 HCCL 完成。

`NOT_READY` 只证明查询当时该 Event 未完成，不能区分正在执行、依赖等待、HCCL 等待或设备错误。每条 Stream 的完成水位只覆盖该 Stream 上 checkpoint 之前的任务；全局 Host seq 不能当作跨 Stream 的总执行顺序。`HCCL_END` 只表示 Host API 返回。`COVERAGE WARNING` 说明已观察 Stream 完成而设备同步仍未返回，应调查最后 checkpoint 后的工作和未注册的内部 Stream。`ring_wrap_or_loss`、缺 Rank、ID 未对齐、快照时刻不同或 Event query 错误，都要在结论中标为证据限制。

## 设备算子边界

四卡示例的 `--device-profile` 会在故障前关闭 profiler 窗口，保存真实设备 kernel CSV；该选项**不是 vLLM 服务开关**。真实 vLLM 需按部署版本接入分段 torch_npu profiler，且窗口必须在卡死前完成落盘。官方 vLLM-Ascend 也提供 `--profiler-config` 与 `/start_profile`、`/stop_profile`，但卡死后无法执行 stop 时不能保证完整输出。[官方 profiling 指南](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)

本工具目前没有 CANN 内部逐 Task completion、HCCL 内部等待图或 vLLM Scheduler/PP/KV 自动语义接入。A2/CANN 9.1.0 已验证统一 Runtime/HCCL Callback + Kernel/HCCL Activity ring、0/1 开关、Event 池、非阻塞查询、后台 poller、generation、Analyzer 和四卡真实 HCCL 缺 Rank；950PR/CANN 9.2.0 必须完成第 1 节编译预检和第 4 节校准后，才把对应 Device 证据用于该现场。
