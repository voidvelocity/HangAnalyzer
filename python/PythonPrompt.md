可以。你给的 26.1.0 Python API 对你的场景其实很合适，因为它已经把底层 Activity Buffer 封装成了 `KernelMonitor`、`CommunicationMonitor`、`MstxMonitor`。官方 `python_monitor` 样例的定位就是用 Monitor 获取计算和通信算子的耗时。([GitCode][1])

对你的 `npu-flight-recorder`，第一版甚至可以完全用 Python 写。

## 1. Python API 能拿到什么

26.1 的 Python Monitor 大致是：

```text
KernelMonitor
    ↓ callback(KernelData)
Kernel:
    start
    end
    device_id
    stream_id
    correlation_id
    type
    name

CommunicationMonitor
    ↓ callback(CommunicationData)
Communication:
    start
    end
    device_id
    stream_id
    correlation_id
    name
    comm_name
    alg_type
    count
    data_type
```

其中 Kernel 的 `start/end` 是 NPU 执行时间，而不是 Python 调用时间，因此：

```text
duration = end - start
```

就能知道一个 Kernel 在 NPU 上真正执行了多久。

通信数据也带 `device_id / stream_id / start / end / correlation_id`，这正是我们前面 flight recorder 需要的核心字段。([GitCode][2])

---

## 2. 最小版本可以这么写

按 26.1 Python Monitor API 的生命周期：

```python
from mspti import KernelMonitor, CommunicationMonitor

kernel_monitor = KernelMonitor()
comm_monitor = CommunicationMonitor()


def on_kernel(data):
    print(
        f"KERNEL "
        f"device={data.device_id} "
        f"stream={data.stream_id} "
        f"name={data.name} "
        f"type={data.type} "
        f"start={data.start} "
        f"end={data.end} "
        f"duration={(data.end - data.start) / 1e6:.3f}ms "
        f"corr={data.correlation_id}"
    )


def on_comm(data):
    print(
        f"COMM "
        f"device={data.device_id} "
        f"stream={data.stream_id} "
        f"name={data.name} "
        f"comm={data.comm_name} "
        f"start={data.start} "
        f"end={data.end} "
        f"duration={(data.end - data.start) / 1e6:.3f}ms "
        f"corr={data.correlation_id}"
    )


kernel_monitor.start(on_kernel)
comm_monitor.start(on_comm)

# -----------------------
# 原来的业务代码
# model(...)
# -----------------------

kernel_monitor.stop()
comm_monitor.stop()
```

Monitor 的统一生命周期还包括：

```python
start()
stop()
flush_all()
set_buffer_size()
```

`stop()` 会停止采集并进行 flush；也可以主动 `flush_all()`。([GitCode][2])

实际 import 路径最好直接以你机器上 `${ASCEND_HOME_PATH}/tools/mspti/samples/python_monitor` 的 26.1 样例为准，因为安装包的 Python 暴露路径可能和仓库源码布局不同。

---

## 3. 但不要真的 `print`

上面只是演示。

对于你的 HCCL hang 场景，callback 里面 `print()` 是我最不建议做的事情。

因为：

```text
NPU
 │
 ├ Kernel
 ├ Kernel
 ├ Kernel
 ├ Kernel
 ├ ...
 │
msPTI
 │
 ▼
C extension
 │
 ▼
Python callback
 │
 ├ print
 ├ format string
 ├ filesystem
 └ lock
```

你本来就是在调查时序/死锁问题，这样反而可能明显扰动现场。

官方设计也明确建议 Python Monitor callback **只做轻量操作，并使用消费者线程处理数据**。([GitCode][2])

所以改成：

```python
from collections import deque

MAX_RECORDS = 1_000_000

records = deque(maxlen=MAX_RECORDS)


def on_kernel(x):
    records.append((
        0,                  # KERNEL
        x.device_id,
        x.stream_id,
        x.start,
        x.end,
        x.correlation_id,
        x.name,
    ))


def on_comm(x):
    records.append((
        1,                  # COMM
        x.device_id,
        x.stream_id,
        x.start,
        x.end,
        x.correlation_id,
        x.name,
    ))
```

这就已经是最简单的：

> **Python NPU Flight Recorder**

---

## 4. 我更推荐 Queue + 后台消费者

如果运行十几分钟甚至几个小时，callback 只负责搬数据：

```text
              msPTI
                │
                ▼
         Python callback
                │
         只复制必要字段
                │
                ▼
         SimpleQueue
                │
                ▼
       background thread
                │
                ▼
         Ring Buffer
      保留最近 30~60 秒
```

例如：

```python
import queue
import threading
from collections import deque

q = queue.SimpleQueue()
history = deque(maxlen=2_000_000)


def on_kernel(x):
    q.put((
        "K",
        x.device_id,
        x.stream_id,
        x.start,
        x.end,
        x.correlation_id,
        x.name,
    ))


def on_comm(x):
    q.put((
        "C",
        x.device_id,
        x.stream_id,
        x.start,
        x.end,
        x.correlation_id,
        x.name,
    ))


def consumer():
    while True:
        record = q.get()
        history.append(record)


threading.Thread(
    target=consumer,
    daemon=True,
).start()
```

这样 callback 本身非常轻。

不过最终要做长期生产级工具，我还是会把 ring buffer 放回 C/C++ 层；Python 版非常适合先验证你的问题到底能不能被 msPTI 捕获。

---

## 5. 一个很重要的问题：它能不能看到“挂住的 HCCL”？

这里需要特别注意。

Activity 数据的工作方式本质上是：

```text
NPU 执行
   │
   ▼
Activity Buffer
   │
buffer 满 / flush
   ▼
Python callback
```

官方底层 Activity API 也是 `RequestFunc → Buffer → CompleteFunc → 消费 record` 的异步模式。([GitCode][3])

所以假设：

```text
12:01:01 MatMul END
12:01:01 AllToAll START
               │
               │
               │ HANG
               ▼
             ?????
```

**不要假设 Python `CommunicationMonitor` 一定会立刻给你一个 `end=0` 的 AllToAll record。**

Activity 更擅长告诉你：

> 已经完成并进入 Activity Buffer 的任务是什么。

因此挂死现场可能是：

```text
Stream 7:

MatMul      DONE
RMSNorm     DONE
GMM         DONE

然后没有任何新 Activity
----------------------- HANG
```

而不是：

```text
AllToAllV start=xxx end=???
```

这两者对 hang debugger 的意义完全不同。

---

## 6. 所以一定要周期性 `flush_all()`

这对你的场景尤其重要。

比如单独起一个线程：

```python
import time

def periodic_flush():
    while True:
        time.sleep(0.5)

        kernel_monitor.flush_all()
        comm_monitor.flush_all()
```

官方 Python Monitor 暴露 `flush_all()`，底层对应的就是主动刷新 Activity Buffer。([GitCode][2])

这样：

```text
Activity Buffer
       ↓
每 500ms flush
       ↓
Python
       ↓
ring buffer
```

即使：

```text
12:31:45
HCCL hang
```

你至少已经把：

```text
12:31:44.5
之前发生的东西
```

拿回来了。

对 flight recorder，我建议一开始测试：

```text
flush interval = 100 ms
                 500 ms
                 1 s
```

然后测一下 throughput/TPOT 扰动，再决定生产值。

---

## 7. 但只用 Python Monitor 有一个核心缺口

假设最终得到：

```text
12:00:00.001 D0 S7 MatMul
12:00:00.003 D0 S7 RMSNorm
12:00:00.005 D0 S7 MoeGMM

12:00:00.006 D0 S9 Attention

------------------------------
之后什么都没有
------------------------------
12:00:10 HANG
```

现在有两个可能：

```text
情况 A

Host:
    HCCL AllToAllV 已经下发
           ↓
Device:
    HCCL 卡住

情况 B

Host:
    根本没有下发 HCCL
           ↓
Host / Event / Graph 调度卡住
```

**仅靠 KernelMonitor + CommunicationMonitor 不足以区分这两个情况。**

这就是为什么我前面一直强调：

```text
Host Submit
      +
Device Activity
```

两条线必须同时有。

---

## 8. 所以你的第一版建议分两阶段

### Phase 1：纯 Python，今天就可以验证

```text
KernelMonitor
+
CommunicationMonitor
+
500ms flush
+
deque ring buffer
```

记录：

```text
type
device
stream
start
end
duration
correlation_id
name
```

最终按照：

```python
(device_id, stream_id)
```

分组。

你会得到：

```text
Device 0 / Stream 7

time              duration      op
-----------------------------------------------
12:00:01.001      0.121 ms      RMSNorm
12:00:01.002      1.821 ms      MatMul
12:00:01.004      3.217 ms      GroupedMatmul
12:00:01.008      0.043 ms      SwiGLU
...

Device 0 / Stream 9

12:00:01.002      0.823 ms      AllGather
12:00:01.004      1.132 ms      AllToAllV
...
```

这已经能够回答你的第一个问题：

> **NPU Util 100% 那十几秒，device 上到底执行了什么？**

---

### Phase 2：再增加 Host Runtime 侧

如果 Phase 1 发现：

```text
最后一个 Device Activity
之后什么都没有
```

再补：

```text
Runtime API
+
correlationId
```

形成：

```text
HOST                              DEVICE

LaunchKernel corr=100
       │
       └────────────────────────► GMM
                                  S7
                                  3.2ms

HCCL AllToAllV corr=101
       │
       └────────────────────────► AllToAllV
                                  S9
                                  ???

WaitEvent(E37)
       │
       X
```

这时候才能真正回答：

> **HCCL 自己挂了，还是 HCCL 根本没有被正确调度执行。**

而 msPTI 的底层 Activity API 本身就支持 `API → Kernel` 的 `correlationId` 关联，官方也专门提供 `mspti_correlation` 样例。([GitCode][4])

---

## 9. 我建议你现在直接做这个 100 行左右的 PoC

不要马上开发完整 flight recorder。

先在你实际的 vLLM/MindIE 服务进程里插：

```text
KernelMonitor
CommunicationMonitor

          ↓

callback
          ↓

deque(maxlen=N)

          ↑
500ms flush thread
```

然后在正常运行 30 秒后 dump：

```text
device
stream
kernel/HCCL
start
end
duration
correlation_id
```

**先验证三件事：**

1. ACL Graph 模式下能否看到 Graph 内部 Kernel；
2. HCCL 能否通过 `CommunicationMonitor` 正常看到；
3. 你出现 NPU Util 100% 的那十几秒，是否有一个/一批异常长 Kernel 或 Communication activity。

如果这三点都能拿到，Python 版已经足够让我们继续做 `npu-flight-recorder v0.1`；如果第 1 或第 2 点拿不到，再下沉到 C Activity/Callback API，不值得现在就先写 C++。

另外，官方明确说 **msPTI 不能与其他性能数据采集工具同时使用，否则可能丢数据**，所以做这个 PoC 时不要同时开 msprof。([GitCode][5])

[你提供的 msPTI 26.1 Python API 文档](https://gitcode.com/Ascend/mspti/blob/26.1.0/docs/zh/api_reference/python_api/README.md?utm_source=chatgpt.com)

[1]: https://gitcode.com/Ascend/mspti/tree/cb0c9916c36f7183f6d4440937583cef66b411bf/docs/zh?utm_source=chatgpt.com "mspti/docs/zh · Ascend/MindStudio-Profiler-Tools-Interface - AtomGit"
[2]: https://gitcode.com/cai-weiwei1989/MSPTI1/blob/master/docs/zh/development_guide/MindStudio-Profilier-Tools-Interface%E7%89%B9%E6%80%A7%E5%88%86%E6%9E%90%E4%B8%8E%E8%AE%BE%E8%AE%A1%E8%AF%B4%E6%98%8E%E4%B9%A6.md?utm_source=chatgpt.com "MSPTI1/docs/zh/development_guide/MindStudio-Profilier-Tools-Interface特性分析与设计说明书.md-代码预览-MSPTI1:基于 Ascend 设备的 Profiling API 集合项目 - AtomGit"
[3]: https://gitcode.com/Ascend/mspti/blob/cb0c9916c36f7183f6d4440937583cef66b411bf/docs/zh/c_api/README.md?utm_source=chatgpt.com "mspti/docs/zh/c_api/README.md-代码预览-MindStudio-Profiler-Tools-Interface:基于Ascend设备的Profiling API项目 - AtomGit"
[4]: https://gitcode.com/Ascend/mspti/blob/master/docs/en/README.md?utm_source=chatgpt.com "mspti/docs/en/README.md-代码预览-MindStudio-Profiler-Tools-Interface:基于Ascend设备的Profiling API项目 - AtomGit"
[5]: https://gitcode.com/Ascend/mspti/blob/cb0c9916c36f7183f6d4440937583cef66b411bf/docs/zh/README.md?utm_source=chatgpt.com "mspti/docs/zh/README.md-代码预览-MindStudio-Profiler-Tools-Interface:基于Ascend设备的Profiling API项目 - AtomGit | GitCode"

