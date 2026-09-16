# Per-Stream Device Checkpoints

该能力为 Host 异步下发增加稀疏的 Device 完成水位，不会调用 Stream/Device synchronize。

## 构建 CANN 适配器

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCANN_HOME=/usr/local/Ascend/cann-9.1.0
cmake --build build -j
```

生成基础库 `libflightrecorder.so` 和可选的 `libflightcheckpoint_cann.so`。CANN 版本变化后应重新构建和真机验证。

## Python 接入

在 worker 初始化且已选择 NPU Device 后，注册真正需要观察的 Stream。注册会预创建固定 Event 池，必须在流量开始前完成：

```python
from flightrecorder import Recorder, CannCheckpointManager

flight = Recorder(FLIGHT_DIR, global_rank, local_device,
                  "/path/build/libflightrecorder.so")
checkpoints = CannCheckpointManager(
    "/path/build/libflightcheckpoint_cann.so",
    slots_per_stream=4,
)

compute_id = int(compute_stream.npu_stream) & 0xffffffff
kv_id = int(kv_stream.npu_stream) & 0xffffffff
checkpoints.register_stream(compute_id, int(compute_stream.npu_stream))
checkpoints.register_stream(kv_id, int(kv_stream.npu_stream))
```

在一个语义阶段的异步任务全部提交到 Stream 后，先记录该阶段的 Host 事件，使用其返回 seq 创建 checkpoint：

```python
submitted_seq = flight.record(
    "MODEL_END", stream=compute_id, correlation=scheduler_step)
generation = checkpoints.submit(
    compute_id, submitted_seq, checkpoint_id=model_checkpoint_id)
```

`submit` 在同一 Stream 尾部执行 `aclrtRecordEvent`，然后写入：

```text
CHECKPOINT(stream, checkpoint_id, submitted_seq, generation)
```

在原有 scheduler tick 或 worker loop 中调用非阻塞轮询：

```python
newly_completed = checkpoints.poll(budget=32)
```

它只调用 `aclrtQueryEventStatus`，不等待。Event COMPLETE 时记录完全匹配的：

```text
DEVICE_CONFIRMED(stream, checkpoint_id, submitted_seq, generation)
```

建议在同一个提交线程的后续调度轮次调用 `poll`。不要创建高频 Python 后台线程；CANN 文档提示跨线程 Record/Query 需要特别处理顺序。本实现保证 Record 返回后才发布 pending 槽，但目标 CANN 版本仍须测试。

如果业务线程可能阻塞在 HCCL 或同步 API，可启动 C++ poller；它不需要 Python GIL：

```python
checkpoints.start_poller(interval_us=1000, budget=32)
```

poller 对每个 generation 首次查到 NOT_READY 时写一条带 `FL_CHECKPOINT_NOT_READY` 的事件；报告会显示 `explicit NOT_READY observations`。进程正常退出前调用 `checkpoints.close()` 会先停止 poller。A2/CANN 9.1.0 已验证跨线程 query，但 CANN 版本或运行模式变化后仍应重新测试。

## Event 池与 generation

- 每个 Stream 独立拥有 `slots_per_stream` 个预创建 Event。
- Event 未完成时绝不复用；池满时 `submit` 返回 `EAGAIN`，并记录 `FL_CHECKPOINT_POOL_FULL`。
- Event 完成后槽位才回到池；复用同一物理 Event 时 generation 单调递增。
- 分析器以 `(stream, checkpoint_id, submitted_seq, generation)` 精确配对。旧 generation 的 confirmation 被列为 `stale/unmatched`，不会推进完成水位。
- `close()` 遇到 pending Event 会报错且保留 manager，不会隐式同步或销毁仍在使用的 Event。进程异常退出时由 OS/CANN 清理资源。

## checkpoint 粒度

优先放在 scheduler step、PP send/recv、KV transfer、HCCL 后置依赖以及长 prefill 的 layer group 边界。不要每个 kernel 都放 checkpoint。一个 Stream 的池必须能覆盖两次 poll 之间最多的 pending checkpoint 数；持续出现 pool-full 说明池太小或 poll 太少。

对于框架内部的 HCCL Stream，只有拿到真实 Stream handle 才能直接放后置 Event。在默认计算 Stream 上放 Event，不能自动证明内部通信 Stream 已完成，除非框架已经建立了明确的跨 Stream 依赖。

## 报告

`report.txt` 的 `DEVICE CHECKPOINT PROGRESS` 会按 Rank/Stream 输出：

```text
submitted_seq=160 completed_seq=100
confirmed through submitted_seq=100 checkpoint=5001 generation=1
first unconfirmed submitted_seq=120 checkpoint=5002 generation=2
unconfirmed submitted_seq interval=120..160
```

这表示设备已确认越过 seq 100；真正停点在完成水位之后、Host 已提交上界以内。NOT_READY 本身不能区分仍在执行、依赖等待或任务失败，需结合 HCCL 跨 Rank、错误日志和设备健康状态。

`report.json` 还为每个 Stream 保存 `checkpoint_mappings`，逐项列出 `submitted_seq`、`checkpoint_id`、`generation`、确认状态和 confirmation recorder sequence，便于程序精确重建 `submitted_seq → checkpoint → completed_seq`。

报告同时保留 `last_event`（含后台 probe 的原始尾部）和 `last_host_event`（业务线程最后事件）。后台 poller 写入 NOT_READY 或 DEVICE_CONFIRMED 后，不会覆盖 `DEVICE_SYNC_BEGIN` 等 Host 卡点。如果所有已观察 Stream 均确认完成，但 `DEVICE_SYNC_BEGIN` 仍未闭合，报告会输出 `COVERAGE WARNING`：未完成工作位于最后一个 checkpoint 之后，或位于未埋点的内部 Stream。

## A2 验证

```bash
python3 tests/a2_checkpoint_smoke.py \
  --recorder-library "$PWD/build/libflightrecorder.so" \
  --checkpoint-library "$PWD/build/libflightcheckpoint_cann.so" \
  --directory /tmp/a2-checkpoint-smoke --device 7
```

测试在真实 A2 Stream 上排队矩阵乘，首次非阻塞 poll 得到 pending 并保存报告；完成后复用单槽 Event，generation 从 1 变为 2；另一个 transfer Stream 使用独立池。合成后端还验证池满、pending destroy 和多 Stream 行为。

2026-09-16 的 A2/CANN 9.1.0 实测结果已保存：

- [pending-report/report.txt](results/a2-checkpoint-20260916/pending-report/report.txt)：计算 Stream 明确观测到 NOT_READY，`first unconfirmed submitted_seq=1`。
- [final-report/report.txt](results/a2-checkpoint-20260916/final-report/report.txt)：同一物理 Event 复用后 generation 从 1 增至 2，计算与传输 Stream 均独立确认完成。
- [四卡 checkpoint 报告](results/a2-fourcard-checkpoint-20260916/report/report.txt)：真实 HCCL 缺 Rank 时，4/5/7 卡停在 `DEVICE_SYNC_BEGIN`。默认 Stream 的 checkpoint 已完成，Analyzer 因此正确报告内部 Stream 覆盖缺口。
