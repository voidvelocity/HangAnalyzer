# msPTI 26.1 Python 实时算子采集

入口：[mspti_activity_capture.py](mspti_activity_capture.py)。采集端只调用 msPTI 的 **Python** `KernelMonitor`、`CommunicationMonitor`、`set_buffer_size`、`flush_all` 和 `stop`；不调用 C API，不需要编译本项目。每个 NPU worker 创建一个实例。回调只复制必要字段并入有界队列，独立线程写 JSONL；默认每 0.5 秒主动 flush，并将新数据 `fsync` 到磁盘。

## 接入业务 worker

在 worker 选择 NPU 之后、执行模型之前启动；路径在每个 Rank/PID 上必须唯一。`MsptiCapture` 必须进入**目标进程**，独立的旁路 Python 进程无法看见已有 vLLM worker 的 activity。

```python
import os
from pathlib import Path
from mspti_activity_capture import MsptiCapture

capture = MsptiCapture(
    Path("/data/hang/mspti") / f"rank{global_rank}-pid{os.getpid()}.jsonl",
    interval_s=0.5,
    buffer_mb=64,
)
capture.start()
try:
    run_worker()
finally:
    capture.stop()
```

业务线程卡在调用中时，独立 Python flush/writer 线程可继续保存**已进入 msPTI Activity Buffer** 的记录；如果 C 扩展长期持有 GIL，或进程被立即 SIGKILL，不能保证下一次刷新。`flush_tick` 是 Host 侧刷新标记，不代表 Device 完成。对于仍未完成的 HCCL，不能假设会收到一条 `end=0` 的通信 Activity。

输出每行是一条 JSON：

- `kernel`：`device_id`、`stream_id`、`correlation_id`、`name`、`kernel_type`、`start_ns`、`end_ns`、`duration_ns`。
- `communication`：同样的时间/Stream 字段，加 `comm_name`、`alg_type`、`count`、`data_type`。
- `flush_tick`：本机 Host 时间、该次 flush 是否成功、队列丢弃数。

`*.summary.json` 汇总记录数、丢弃、回调/flush 错误。队列满时丢弃最新记录并累计 `dropped`；若不为 0，采集结果不完整。JSONL 每次 fsync 前可能有极短尾部未落盘；出现卡死时应同时保留本仓库 Host flight recorder 的 PP/KV/HCCL 提交证据。

## A2 双卡 smoke

在已安装 msPTI 26.1.0、PyTorch 与 torch_npu 的容器内，从项目根目录运行。以下示例将两个 local rank 分配到物理 4、5 卡，每轮执行 MatMul、Add 和真实 HCCL AllReduce：

```bash
python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29673 \
  python/mspti_activity_capture.py --demo \
  --output /tmp/mspti-python-demo --devices 4,5 \
  --iterations 8 --interval 0.5

python3 -m unittest discover -s python -v
```

`--demo` 用于验证，不会自动注入已有服务。查看结果：

```bash
cat /tmp/mspti-python-demo/*.summary.json
grep '"kind":"communication"' /tmp/mspti-python-demo/*.jsonl | head
grep '"kind":"kernel"' /tmp/mspti-python-demo/*.jsonl | head
grep '"kind":"flush_tick"' /tmp/mspti-python-demo/*.jsonl | head
```

2026-09-18 在 `173.125.1.2` 的 `gl_main_a2`、A2 物理 4/5 卡、CANN 9.1.0 + msPTI 26.1.0 验证。每 Rank 8 轮得到 18 条计算 kernel、8 条 `hcom_allReduce_` 通信记录、15 条 `flush_tick`；`dropped=0`、`callback_errors=0`、`flush_errors=0`。活体读取同一次运行的 JSONL，总行数在相邻 0.5 秒采样中从 0、2、4、16、24、26、34 增长；tick 间隔中位数 0.500 秒。样例输出见 [验证目录](results/a2-20260918)。这些数据证明已完成算子能周期落盘，不证明挂住的通信算子会产生未完成记录。

## AICore / MTE / Vector 利用率

msPTI 26.1.0 的这两个 Python Monitor 返回算子起止时间及通信元数据，**没有** AICore/MTE/Vector 管线利用率计数字段。本脚本将 `pipe_utilization` 显式写为 `null`。不能用 MatMul 耗时或 `npu-smi` 的总 AICore 百分比推算 MTE/Vector 利用率。若现场必须获得这些指标，需要另一套支持 PipeUtilization 的性能采集配置；该模式应单独验证与 msPTI 并发是否冲突。

参考：[msPTI 26.1 Python API](https://gitcode.com/Ascend/mspti/blob/26.1.0/docs/zh/api_reference/python_api/README.md)、[官方 Python Monitor 样例索引](https://gitcode.com/Ascend/mspti/blob/master/samples/README.md)。
