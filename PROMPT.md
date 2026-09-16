# Ascend Hang Flight Recorder

## 实施校正（2026-09-16，优先于下文概念性设计）

本仓库第一版是**显式埋点的 Host flight recorder**，面向两台 Ascend 950PR 机器上的 vLLM / vLLM-Ascend、PP 和 KV 传输。每台节点在请求发出前启动 watchdog；在 API 入口、返回处记录事件，进程停滞和退出时分别快照。两节点快照复制到同一目录后离线分析。进程已 SIGKILL 时，`/proc` 栈不可再读取，因此停滞时的首次快照非常关键。

**证据边界：**Host `*_END` 只代表调用返回；异步 ACL/HCCL 下发成功不代表 NPU 执行完成。只有上层在确知同步/查询成功后主动写入 `DEVICE_CONFIRMED`，才可报告 Device completion。`EVENT_RECORD/WAIT` 反映 Host 的依赖设置，不能凭它们单独证明真实 device deadlock。每个 communicator 的 operation ID 要由调用方在各 Rank 使用共同逻辑 ID；本地递增计数不自动具有跨 Rank 可比性。跨 Rank 时钟不作绝对时间排序。

**必要埋点：**API 请求接收/结束（可只写 request hash，不能写原文）、scheduler iteration、model execute、PP send/recv、KV transfer begin/end、HCCL begin/end、Event record/wait、Graph replay、同步 API 入口/返回。`correlation_id` 用同一请求/调度轮次/传输或 collective 的明确逻辑 ID；`arg0/arg1` 记录 communicator、collective 类型、token 数、layer 等约定值，在项目集成层形成字典并留在现场。长 prefill 的超时阈值要高于正常最大无埋点时长，或增加阶段 checkpoint。

**当前实现与待接入：**本仓库提供 C++ mmap writer、Python 显式埋点封装、独立 watchdog 和离线 analyzer。尚未提供 CANN ABI 级 hook，也没有直接的 NPU task completion、HCCL 内部状态或 vLLM 自动接入；这些需要在目标环境针对确切 vLLM-Ascend/PyTorch/CANN 9.2.0 构建和验证。不要仅凭 `LD_PRELOAD` 启动就声称采集到全部设备执行。下文的 device_seq、算子级任务、自动根因图和 `<1%` 等为后续目标，须在真机测量验证。

**设备算子补充采集：**A2/CANN 9.1.0 上的四卡复现增加可选 `--device-profile`。每 Rank 用 torch_npu profiler 采集并关闭故障前一个完整调度阶段；原始 profile、`kernel_details.csv` 和人类可读的 `device_operators.txt` 可在后续卡死和进程终止后离线分析。它证明该采集窗口内哪些 kernel 实际执行过，不覆盖窗口关闭后的故障阶段，也不改变上述 Device completion 的证据边界。

## 1. 项目目标

开发一个面向 Ascend NPU 多机多卡推理服务的轻量级 Hang Profiling / Flight Recorder 工具。

主要解决 vLLM、vLLM-Ascend、MindIE 等推理服务出现偶现卡死时，传统 profiling 无法正常 `stop/finalize`，导致无法确定：

* 进程最后执行/下发到了哪个算子；
* Host 侧最后下发了什么任务；
* 哪个 NPU Stream 停止推进；
* Stream 上最后完成和未完成的 Task 是什么；
* `EventRecord/EventWait` 形成了怎样的跨 Stream 依赖；
* 哪个 HCCL collective 没有完成；
* 不同 Rank 是否进入了相同的 collective；
* 是 Host 没有继续下发，还是 Device 已经停止执行；
* 是否存在 Compute Stream 与 Communication Stream 的循环等待；
* ACL Graph Replay 卡在什么位置；
* Hang 发生前几秒完整的执行现场是什么。

工具定位：

> **不是完整性能 Profiler，而是面向 Hang/Deadlock 的低开销黑匣子 Flight Recorder。**

核心原则：

> Always On、Ring Buffer、External Watchdog、Crash Survivable、Cross-Rank Correlation。

---

# 2. 典型问题

目标场景：

```text
Machine 0                       Machine 1

Rank 0       Rank 1             Rank 8       Rank 9
  │            │                  │            │
Compute       HCCL               Compute       HCCL
  │            │                  │            │
Stream 5     Stream 17          Stream 3     Stream 21
  │            │                  │            │
Graph          │                 Graph          │
  │            │                  │            │
Record E81     │                 Record E93     │
  │            │                  │            │
  └─────────► Wait               │             │
               │                 │             │
            AllToAll #813        │         AllToAll #813
               │                 │             │
              ???                │            ???
               │                 │
          服务整体 Hang
```

需要回答的不是：

> “HCCL 看起来可能卡住了。”

而应该输出：

```text
Hang detected at 14:32:18.531

Rank 3
Host last submitted:
    seq=192831
    GraphReplay graph=decode_bs8
    stream=12

Device last completed:
    seq=192817
    MatMul
    stream=12

Outstanding:

    stream 12:
        EventWait(event=914)

    stream 21:
        HCCL AllToAll
        collective_seq=8812
        BEGIN
        no END

Event dependency:

    stream12
        |
        Wait E914
        ^
        |
    stream21
        |
        HCCL AllToAll #8812
        |
        no completion

Cross-rank:

    rank0 #8812 BEGIN
    rank1 #8812 BEGIN
    rank2 #8812 BEGIN
    rank3 #8812 BEGIN
    rank4 #8812 MISSING
    rank5 #8812 BEGIN
    ...

Suspect:
    rank4 did not submit collective #8812.

Rank4 host:
    last submitted task:
        GraphReplay #192827

    host progress stopped before HCCL #8812.
```

工具的价值就在于把问题从：

```text
“多机多卡偶现卡死”
```

缩小到：

```text
Rank4
Scheduler iteration 1932
Graph decode_bs8
Stream12
EventRecord E914 前后
HCCL collective #8812 未提交
```

---

# 3. 总体架构

采用两层架构：

```text
                 ┌───────────────────────────┐
                 │       Hang Analyzer       │
                 │                           │
                 │ Cross-Rank Correlation    │
                 │ Stream Dependency Graph   │
                 │ Collective Matching       │
                 │ Hang Report               │
                 └─────────────▲─────────────┘
                               │
                         dump files
                               │
───────────────────────────────┼──────────────────────────

                 每台机器一个独立进程

                 ┌───────────────────────────┐
                 │        Watchdog           │
                 │                           │
                 │ heartbeat                 │
                 │ progress detector         │
                 │ snapshot                  │
                 │ stack collection          │
                 └─────────────▲─────────────┘
                               │
                     mmap / shared memory
                               │
───────────────────────────────┼──────────────────────────

                 推理进程 / Worker

                 ┌───────────────────────────┐
                 │      Flight Recorder      │
                 │                           │
                 │ Lock-Free Ring Buffer     │
                 └─────────────▲─────────────┘
                               │
          ┌────────────────────┼─────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
     Host Task            Runtime Task             HCCL
     Scheduler            Stream/Event          Collective

          │                    │                     │
          └────────────────────┼─────────────────────┘
                               │
                              NPU
```

分成三个核心组件：

```text
libflightrecorder.so
        +
hang-watchdog
        +
hang-analyzer
```

---

# 4. 核心设计原则

## 4.1 不依赖正常退出

禁止把数据保存依赖于：

```python
profiler.stop()
```

因为 Hang 时 stop 本身可能执行不到。

Flight Recorder 数据持续存在于：

```text
/tmp/npu-flight-recorder/
    rank0.bin
    rank1.bin
    ...
```

文件使用 `mmap(MAP_SHARED)`。

即使：

```text
Python deadlock
HCCL hang
NPU runtime hang
SIGKILL
```

已经记录的数据仍然存在。

---

## 4.2 Always-On Ring Buffer

不记录整个服务生命周期，只保存：

> 最近 N 秒 / N 个事件。

例如：

```text
Ring Buffer = 64 MB / rank

Event = 64 Bytes

≈ 1,000,000 events
```

如果平均 10 万 event/s：

```text
可以保存约 10 秒现场
```

旧数据自动覆盖。

因此可以常驻生产/测试环境。

---

## 4.3 Hot Path 禁止锁

业务线程写 recorder：

```cpp
record_event(...)
```

必须满足：

```text
No mutex
No malloc
No JSON
No filesystem syscall
No Python callback
No synchronize
```

只做：

```text
atomic fetch_add
+
memory write
```

目标：

```text
几十 ns ~ 亚微秒级
```

---

# 5. Event 数据结构

第一版使用固定 64 Byte Event：

```cpp
struct alignas(64) FlightEvent {
    uint64_t timestamp_ns;

    uint64_t seq;

    uint32_t pid;
    uint32_t tid;

    uint16_t device_id;
    uint16_t rank_id;

    uint32_t stream_id;

    uint16_t event_type;
    uint16_t flags;

    uint64_t correlation_id;

    uint64_t arg0;
    uint64_t arg1;
};
```

其中：

```text
timestamp_ns
    Host monotonic clock

seq
    当前 rank 全局递增 sequence

stream_id
    Runtime stream

correlation_id
    op / graph / hccl / event ID

arg0 / arg1
    不同 Event 自定义
```

---

# 6. Event 类型

第一版不要追求全量 profiling。

只记录能定位 Hang 的事件。

## Host

```text
SCHEDULER_STEP_BEGIN
SCHEDULER_STEP_END

MODEL_EXECUTE_BEGIN
MODEL_EXECUTE_END

HOST_OP_BEGIN
HOST_OP_END
```

## ACL / Runtime

```text
GRAPH_REPLAY_BEGIN
GRAPH_REPLAY_END

KERNEL_LAUNCH

EVENT_RECORD
EVENT_WAIT

STREAM_SYNC_BEGIN
STREAM_SYNC_END

DEVICE_SYNC_BEGIN
DEVICE_SYNC_END
```

## HCCL

```text
HCCL_BEGIN
HCCL_END
```

HCCL 进一步记录：

```text
AllReduce
AllGather
ReduceScatter
AllToAll
AllToAllV
Broadcast
Send
Recv
...
```

---

# 7. 三种 Sequence

这是整个工具最重要的数据模型之一。

不要只有一个 seq。

至少维护：

```text
host_seq
device_seq
collective_seq
```

## host_seq

每次 Host 下发任务：

```text
host_seq++
```

例如：

```text
10081 KernelLaunch
10082 KernelLaunch
10083 EventRecord
10084 HCCL AllToAll
10085 EventWait
```

代表：

> Host 已经走到哪里。

---

## device_seq

代表：

> Device 已经确认执行到哪里。

因此：

```text
submitted_seq = 10085
completed_seq = 10082
```

意味着：

```text
10083
10084
10085
```

仍 outstanding。

这是判断：

> Host Hang

还是：

> Device Hang

非常重要的信息。

第一版如果无法低成本获取每个 Device Task 的 completion，可以只在少量关键 checkpoint 更新 `device_seq`，不要为了精确 completion 引入同步。

Device checkpoint 必须满足以下约束，避免诊断代码改变被诊断系统的时序：

1. 每个被观察的 Stream 使用独立的、初始化阶段预创建的固定 Event 池；运行期不创建 Event。
2. checkpoint 在该 Stream 尾部 Record Event，只用非阻塞 Event status query 轮询，禁止用 Stream/Device synchronize 推进诊断水位。
3. 每次提交记录 `submitted_seq -> checkpoint_id -> completed_seq`。完成确认必须精确匹配 `(stream_id, checkpoint_id, submitted_seq, event_generation)`。
4. 物理 Event 只有确认完成后才能复用；每次复用递增 generation。旧 generation 的迟到结果只能报告为 stale/unmatched，不能推进新 checkpoint。
5. Analyzer 按 Stream 给出 `confirmed through`、`first unconfirmed` 和 `unconfirmed submitted_seq interval`，并区分 Host 提交、明确 Device 完成及尚未确认三种状态。
6. 业务线程可能卡在 HCCL 或同步 API，因此轮询器必须能在不依赖 Python GIL 的后台线程继续工作。每个 generation 的 NOT_READY 只需记录一次，避免探针写入掩盖 watchdog 的停滞判断。
7. 只允许对实际持有 handle 的 Stream 作完成推断。默认计算 Stream checkpoint 已完成，不能证明框架内部 HCCL/KV Stream 已完成；若 Device sync 仍未返回，报告应明确提示 checkpoint 覆盖缺口。

---

## collective_seq

每个 communicator 维护：

```cpp
collective_seq++;
```

例如：

```text
#8810 AllReduce
#8811 AllToAll
#8812 AllToAll
#8813 ReduceScatter
```

跨 Rank 对齐后，可以快速发现 collective mismatch。

---

# 8. HCCL Flight Recorder

每次 HCCL 调用记录：

```text
timestamp
rank
communicator_id
collective_seq
collective_type
stream
count
dtype
BEGIN / END
```

例如：

```text
Rank0:

8810 AllReduce     BEGIN END
8811 AllToAll      BEGIN END
8812 AllToAll      BEGIN ...


Rank1:

8810 AllReduce     BEGIN END
8811 AllToAll      BEGIN END
8812 AllToAll      BEGIN ...


Rank2:

8810 AllReduce     BEGIN END
8811 AllToAll      BEGIN END


Rank3:

8810 AllReduce     BEGIN END
8811 AllToAll      BEGIN END
8812 AllToAll      BEGIN ...
```

Analyzer 自动输出：

```text
Collective #8812 mismatch

Expected ranks:
0 1 2 3

Entered:
0 1 3

Missing:
2
```

然后自动跳到 Rank2 的最后几十个事件。

---

# 9. Stream / Event Dependency Recorder

这是定位计算流/HCCL流死锁的核心。

例如实际执行：

```text
Compute Stream 12

Kernel A
Kernel B
EventRecord E81
                \
                 \
Communication Stream 21

                 EventWait E81
                 HCCL AllToAll
                 EventRecord E82
                         \
                          \
Compute Stream 12

                 EventWait E82
                 Kernel C
```

记录：

```text
stream12 EVENT_RECORD E81

stream21 EVENT_WAIT   E81
stream21 HCCL_BEGIN   #8812
stream21 EVENT_RECORD E82

stream12 EVENT_WAIT   E82
```

Analyzer 构建 Dependency Graph：

```text
stream12
    |
 Record E81
    |
    v
stream21
    |
 HCCL #8812
    |
 Record E82
    |
    v
stream12
```

如果 HCCL 不完成：

```text
stream12 waits E82
        ^
        |
stream21 HCCL #8812
        X
```

可以直接输出：

```text
Blocked stream:
    stream12

Waiting:
    Event E82

Producer:
    stream21

Producer blocked at:
    HCCL AllToAll #8812
```

如果形成真正循环：

```text
Stream A → Event X → Stream B
   ↑                   |
   └──── Event Y ──────┘
```

Analyzer 使用 DFS / SCC 自动检测 cycle。

---

# 10. Graph Recorder

ACL Graph 模式不能只记录：

```text
GraphReplay
```

至少记录：

```text
graph_id
graph_name/hash
scheduler_step
batch information
stream
BEGIN
END
```

例如：

```text
GRAPH_REPLAY_BEGIN

graph_id      = 17
graph_name    = decode_bs8
scheduler     = 9182
stream        = 12
batch_size    = 8
num_tokens    = 8
```

这样最后报告可以关联到：

```text
Request
  ↓
Scheduler Step
  ↓
ModelRunner
  ↓
Graph
  ↓
Stream
  ↓
HCCL
```

而不是只知道：

```text
aclmdlExecute(...)
```

卡住。

---

# 11. Host Hang 检测

必须区分三种状态。

### Case A：Host 不再推进

```text
host_seq
10000
10000
10000
10000
```

同时 Device 也不动。

重点检查：

```text
Python/C++ stack
mutex
condition_variable
future
runtime API
HCCL API
```

---

### Case B：Host 继续提交，Device 不推进

```text
host_seq:

10000
10100
10200
10300

device_seq:

9981
9981
9981
9981
```

说明问题更偏向：

```text
Device execution
Stream dependency
HCCL
Runtime queue
```

---

### Case C：部分 Rank 停止

```text
rank0 host_seq 10291
rank1 host_seq 10293
rank2 host_seq  9811  ←
rank3 host_seq 10292
```

立即把 Rank2 标为 first stalled rank。

---

# 12. Heartbeat

共享状态：

```cpp
struct RankState {
    atomic<uint64_t> last_host_progress_ns;
    atomic<uint64_t> last_device_progress_ns;

    atomic<uint64_t> host_seq;
    atomic<uint64_t> device_seq;

    atomic<uint64_t> scheduler_step;
    atomic<uint64_t> collective_seq;
};
```

watchdog 每：

```text
200 ms
```

读取一次。

例如：

```text
host no progress > 5 sec
```

触发：

```text
SUSPECTED_HANG
```

再观察 1~2 秒确认。

阈值必须可配置，不能把长 Prefill/长算子误判为 Hang。

---

# 13. Watchdog

每台机器启动：

```bash
hang-watchdog \
    --directory /tmp/npu-flight-recorder \
    --hang-timeout 5 \
    --output /data/hang-dumps
```

watchdog 是独立进程。

不要 Python thread。

Hang 时：

```text
detect hang
     |
     +---- snapshot ring buffer
     |
     +---- /proc/<pid>/task/*/stack
     |
     +---- process status
     |
     +---- thread status
     |
     +---- fd information
     |
     +---- maps
     |
     +---- vLLM log tail
     |
     +---- HCCL log tail
```

第一版不要自动 kill 服务。

默认：

```text
capture-only
```

---

# 14. Hang Snapshot

产生：

```text
hang-20260916-143218/
│
├── metadata.json
│
├── node0/
│   ├── rank0.flight
│   ├── rank1.flight
│   ├── rank0.stack
│   ├── rank1.stack
│   ├── process.txt
│   └── hccl.log
│
├── node1/
│   └── ...
│
└── report/
```

`.flight` 使用原始 binary。

Hang 时只：

```text
memcpy/write
```

不要在线做复杂解析。

---

# 15. Offline Analyzer

命令：

```bash
hang-analyzer hang-20260916-143218/
```

生成：

```text
report.txt
report.json
timeline.json
dependency.dot
```

其中 `report.txt`：

```text
==================================================
Ascend Hang Report
==================================================

Hang:
2026-09-16 14:32:18

World size:
32

First stalled rank:
Rank 17


HOST PROGRESS
--------------------------------------------------

rank    host_seq    device_seq    collective
0       193821      193817        8812
...
17      193790      193789        8811
...


COLLECTIVE ANALYSIS
--------------------------------------------------

Suspected collective:

    #8812
    HCCL AllToAll

Entered:
    0-16, 18-31

Missing:
    Rank17


RANK 17 LAST EVENTS
--------------------------------------------------

+0.000000 GraphReplay BEGIN
+0.000124 Kernel MatMul
+0.000139 EventRecord E813
+0.000142 GraphReplay END

No HCCL #8812 submission observed.


OTHER RANKS
--------------------------------------------------

Rank16:

    HCCL AllToAll #8812 BEGIN
    no END

Rank18:

    HCCL AllToAll #8812 BEGIN
    no END


LIKELY BLOCKING CHAIN
--------------------------------------------------

Rank0..16/18..31

HCCL AllToAll #8812
        |
        | waiting for
        v
Rank17
        |
        X
No #8812 submission


RANK17 HOST STACK
--------------------------------------------------

ModelRunner.execute_model
...
aclrtSynchronizeStream(...)
```

最后一部分不要武断输出“根因”，而是输出：

```text
Evidence
Suspected blocking point
Missing events
Outstanding tasks
```

避免诊断工具把相关性错误解释成因果。

---

# 16. Timeline

提供一个简单 Chrome Trace 格式导出：

```bash
hang-analyzer dump/ --chrome-trace trace.json
```

可以看到：

```text
Rank17 Host
─────────────────────────────────────────

GraphReplay
██████

Rank17 Stream12
─────────────────────────────────────────

MatMul ███
Record E81
               Wait E82 ────────────────>

Rank17 Stream21
─────────────────────────────────────────

Wait E81
        AllToAll ███████████████████████→
```

第一版不需要自己开发 GUI。

---

# 17. 如何 Hook

建议分阶段。

## Level 1：显式埋点

首先集成 vLLM-Ascend：

```python
flight.scheduler_begin(step)

execute_model()

flight.scheduler_end(step)
```

Graph：

```cpp
FLIGHT_RECORD(GRAPH_BEGIN, graph_id);

aclGraphLaunch(...);

FLIGHT_RECORD(GRAPH_END, graph_id);
```

HCCL：

```cpp
FLIGHT_HCCL_BEGIN(...);

HcclAlltoAll(...);

FLIGHT_HCCL_END(...);
```

这是 MVP。

---

## Level 2：LD_PRELOAD Hook

为了减少侵入：

```bash
LD_PRELOAD=libnpu_flight_hook.so
```

Hook 关键 Runtime/HCCL API。

概念：

```cpp
aclError aclrtRecordEvent(...) {

    record(EVENT_RECORD, ...);

    return real_aclrtRecordEvent(...);
}
```

以及：

```text
aclrtRecordEvent
aclrtStreamWaitEvent

aclrtSynchronizeStream
aclrtSynchronizeDevice

关键 Graph API

HCCL collective API
```

具体 hook API 名称必须根据目标 CANN/HCCL 版本确认，不能硬编码假设 ABI 永远稳定。

---

# 18. Operator Name

不能只看到：

```text
KernelLaunch 19281
```

最好建立：

```text
correlation_id → operator
```

例如：

```text
19281

layer = 37
module = self_attn
operator = npu_fused_infer_attention_score
```

但 operator string 不要每次复制。

使用 dictionary：

```text
op_id = 371

371 →
"layer37.attention.npu_fused_infer_attention_score"
```

Ring Buffer 只存：

```text
op_id=371
```

降低 overhead。

---

# 19. 多机时间问题

不能简单依赖不同机器：

```text
CLOCK_MONOTONIC
```

绝对值一致。

每个节点记录：

```text
local monotonic
wall clock
node_id
```

跨机主要依赖：

```text
rank
scheduler_step
collective_seq
correlation_id
```

进行逻辑对齐。

时间戳用于辅助分析。

如果环境有 PTP/NTP，可以额外利用，但正确性不能依赖机器时钟严格同步。

---

# 20. 性能目标

第一版目标：

```text
常驻性能损耗 < 1%

单 Event 写入 < 500 ns

业务 Hot Path:
    0 malloc
    0 mutex
    0 filesystem syscall
    0 device synchronize

Memory:
    <= 64 MB / rank

Hang snapshot:
    < 1 second

支持：
    32 / 64 / 128+ ranks
```

其中 `<500 ns` 是工程目标，需要 benchmark 验证，而不是设计保证。

---

# 21. 不做什么

MVP 明确不做：

```text
完整 msprof 替代
AI Core PMU counter
Tensor dump
Tensor shape 全量记录
Memory profiling
完整 Python profiling
完整算子性能统计
在线 GUI
```

否则项目很容易变成另一个大型 profiler。

核心问题始终是：

> **Where did execution stop?**

---

# 22. MVP

第一阶段只实现：

```text
libflightrecorder

    mmap ring buffer
    atomic event writer

    HCCL BEGIN/END
    EventRecord
    EventWait
    Graph BEGIN/END
    Scheduler BEGIN/END

hang-watchdog

    heartbeat
    hang detection
    snapshot

hang-analyzer

    event parser
    cross-rank collective matching
    stream/event dependency
    last-event report
```

做到这里就已经有很高价值。

---

# 23. 第二阶段

加入：

```text
LD_PRELOAD
ACL Runtime hook

Host/Device progress separation

Chrome Trace export

Python/C++ stack collection

Graph metadata

Operator dictionary
```

---

# 24. 第三阶段

加入自动 Deadlock Analyzer。

建立图：

```text
Task
Stream
Event
Collective
Rank
```

依赖关系：

```text
Stream
  waits-for
Event

Event
  produced-by
Stream

HCCL
  waits-for
Rank

Task
  ordered-before
Task
```

得到 Wait-For Graph：

```text
Rank0:HCCL8812
        |
        v
Rank17:HCCL8812 submission
        |
        v
Stream12:E82
        |
        v
Stream21:HCCL8811
        |
        ...
```

使用：

```text
Tarjan SCC
```

寻找循环依赖。

输出：

```text
Dependency cycle detected:

Stream12
   ↓ waits E81
Stream21
   ↓ HCCL #8812
Rank17
   ↓ waits E93
Stream12
```

这会成为项目真正有差异化价值的能力。

---

# 25. 推荐代码结构

```text
ascend-hang-recorder/
│
├── CMakeLists.txt
├── README.md
│
├── include/
│   ├── flight_event.h
│   ├── flight_recorder.h
│   └── rank_state.h
│
├── recorder/
│   ├── ring_buffer.cpp
│   ├── recorder.cpp
│   └── metadata.cpp
│
├── hooks/
│   ├── acl_hook.cpp
│   ├── hccl_hook.cpp
│   └── preload.cpp
│
├── watchdog/
│   ├── main.cpp
│   ├── heartbeat.cpp
│   ├── snapshot.cpp
│   └── procfs.cpp
│
├── analyzer/
│   ├── parser.py
│   ├── collective.py
│   ├── stream_graph.py
│   ├── hang_detector.py
│   └── chrome_trace.py
│
├── python/
│   └── flight_recorder/
│       └── __init__.py
│
├── tests/
│   ├── ring_buffer_test.cpp
│   ├── collective_test.py
│   ├── deadlock_test.py
│   └── crash_test.py
│
└── examples/
    ├── hccl_mismatch/
    ├── stream_deadlock/
    └── host_hang/
```

C++ 负责 Hot Path，Python 负责离线 Analyzer，是比较合适的边界。

---

# 26. Agent 开发验收用例

必须人为制造至少四种 Hang。

### Case 1：Rank 缺失

```text
rank0 HCCL #100
rank1 HCCL #100
rank2 skip
rank3 HCCL #100
```

必须识别：

```text
Missing Rank2
Collective #100
```

### Case 2：Collective 顺序不一致

```text
rank0:
AllReduce → AllToAll

rank1:
AllToAll → AllReduce
```

识别 collective mismatch。

### Case 3：Stream/Event 循环依赖

```text
Stream A waits E2
Stream B waits E1

E1 produced by A after wait
E2 produced by B after wait
```

识别 dependency cycle。

### Case 4：Host Hang

人为：

```cpp
while (true) {}
```

或 mutex deadlock。

报告必须显示：

```text
Host progress stopped
Device progress ...
last host task
thread stack
```

只有这四个 case 都能自动给出有效报告，MVP 才算完成。

---

# 27. 最终用户体验

启动服务只增加：

```bash
LD_PRELOAD=/opt/flight/libnpu_flight.so \
FLIGHT_DIR=/tmp/npu-flight \
vllm serve ...
```

旁边启动：

```bash
flight-watchdog \
    --pid <vllm-pid> \
    --timeout 5s \
    --output /data/hang
```

出现 Hang 后：

```bash
flight-analyze /data/hang/20260916-143218
```

直接得到：

```text
HANG SUMMARY

First abnormal rank:
    rank17

Host:
    stopped at scheduler step 9182

Last submitted:
    Graph decode_bs8
    stream12

Last completed:
    MatMul
    stream12

Outstanding:
    EventWait E914
    HCCL AllToAll #8812

Collective #8812:
    31/32 ranks entered

Missing:
    rank17

rank17 last operation:
    GraphReplay decode_bs8

Stream dependency:
    stream12 waits E914
    E914 producer = stream21
    stream21 blocked in HCCL #8811

See:
    dependency.dot
    trace.json
```

这就是项目最核心的验收标准：

> **服务已经完全卡死、不需要调用任何 `stop_profile()`，仍然能够回答“哪个 Rank、哪个 Host Task、哪个 Graph、哪个 Stream、哪个 Event、哪个 HCCL collective、哪个算子附近停止推进”。**
