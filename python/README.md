# msPTI 26.1 Python 实时算子采集

入口：[mspti_activity_capture.py](mspti_activity_capture.py)。采集端只调用 msPTI 的 **Python** `KernelMonitor`、`CommunicationMonitor`、`set_buffer_size`、`flush_all` 和 `stop`；不调用 C API，不需要编译本项目。每个 NPU worker 创建一个实例。回调只复制必要字段并入有界队列，独立线程写 JSONL；采集启用时默认每 0.5 秒主动 flush，并将新数据 `fsync` 到磁盘。

## 启停文件

默认每隔 `interval_s` 读取 `/home/enable_prof`：**文件缺失或内容为 `0` 时不启动/停止 Monitor；内容为 `1` 时启动 Monitor**。其它内容按停用处理并写入 `last_error`。因此服务拉起前可先创建内容为 `0` 的文件，待服务就绪、准备发请求时写 `1`，诊断结束写回 `0`。确保该路径在每个 worker 的**容器内部**可读；多机部署时各节点分别设置，或用相同挂载。读取不是即时的，状态变化最多等待一个轮询周期，外加 msPTI start/stop 调用时间。

```bash
printf '0\n' > /home/enable_prof  # 服务启动前
printf '1\n' > /home/enable_prof  # 准备发送推理请求
printf '0\n' > /home/enable_prof  # 停止采集，触发最后一次 flush
```

为避免写文件瞬间被读到空内容，可先写临时文件再 `mv` 到同目录。启动业务前文件不存在同样表示关闭。每次实际切换写一条 `capture_state`；停用时 Monitor 不注册回调，也不周期调用 `flush_all()`。`flush_tick` 在关闭时仍每隔 `interval_s` 记录一次，`enabled=false` 且 `flush_ok=null`，用于证明控制线程存活。若框架已有其它 profiler 占用采集资源，应先在目标环境验证兼容性。

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
    enable_file="/home/enable_prof",  # 默认值，可省略
)
capture.start()
try:
    run_worker()
finally:
    capture.stop()
```

业务线程卡在调用中时，独立 Python flush/writer 线程可继续保存**已进入 msPTI Activity Buffer** 的记录；如果 C 扩展长期持有 GIL，或进程被立即 SIGKILL，不能保证下一次刷新。`flush_tick` 是 Host 侧刷新标记，不代表 Device 完成。对于仍未完成的 HCCL，不能假设会收到一条 `end=0` 的通信 Activity。

输出每行是一条 JSON：

- `kernel`：`device_id`、`stream_id`、`correlation_id`、`name`、`kernel_type`、`start_ns`、`end_ns`、`timestamp_valid`、`duration_ns`。
- `communication`：同样的时间/Stream 字段，加 `comm_name`、`alg_type`、`count`、`data_type`。
- `capture_state`：Monitor 实际启用/停用的 Host 时间。
- `flush_tick`：本机 Host 时间、Monitor 是否启用、该次 flush 是否成功、队列丢弃数；停用时不调用 msPTI flush。

`*.summary.json` 汇总记录数、丢弃、回调/flush 错误。队列满时丢弃最新记录并累计 `dropped`；若不为 0，采集结果不完整。JSONL 每次 fsync 前可能有极短尾部未落盘；出现卡死时应同时保留本仓库 Host flight recorder 的 PP/KV/HCCL 提交证据。

## A2 双卡 smoke

在已安装 msPTI 26.1.0、PyTorch 与 torch_npu 的容器内，从项目根目录运行。以下示例将两个 local rank 分配到物理 4、5 卡，每轮执行 MatMul、Add 和真实 HCCL AllReduce：

```bash
printf '1\n' > /tmp/mspti-demo-enable
python3 -m torch.distributed.run --nproc_per_node=2 --master_port=29673 \
  python/mspti_activity_capture.py --demo \
  --output /tmp/mspti-python-demo --devices 4,5 \
  --iterations 8 --interval 0.5 --enable-file /tmp/mspti-demo-enable

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

动态开关另做双卡验证，输出见 [开关验证目录](results/a2-toggle-20260918)。两个 Rank 的 `capture_state` 均为 `[true, false, true, false]`，其中最后一次 `false` 来自正常退出；停用窗口的 `flush_tick` 为 `enabled=false, flush_ok=null`。Rank 0 采到 30 条计算和 15 条通信记录，Rank 1 采到 32 条计算和 16 条通信记录；两者 `dropped=0`、`callback_errors=0`、`flush_errors=0`。本地模拟测试还验证了缺失文件默认关闭和重新启用。

使用**默认路径 `/home/enable_prof`** 的最终复验见 [默认开关验证目录](results/a2-default-control-20260918)。两个 Rank 的状态序列同样为 `[true, false, true, false]`；Rank 0 采到 28 条计算、14 条通信，Rank 1 采到 30 条计算、15 条通信，所有算子时间戳有效，丢弃和错误均为 0。测试前该文件不存在，测试后已恢复为不存在。

## 记录代表什么

`kernel` 和 `communication` 记录中的 `start_ns/end_ns` 是**NPU 上的执行时间**，不是 Host 下发时间，也不是 Stream 队列等待时间。正常情况下，一条 `timestamp_valid=true` 的完整 Activity 对应已经执行并上报的算子；若时间戳缺失或倒序，脚本设置 `timestamp_valid=false`、`duration_ns=null`，不能用它证明完成。周期 `flush_all()` 只是把缓冲区中**已有**的记录交给 Python，不会查询 Stream 队列里还有哪些任务，也不能强制让尚未完成的 HCCL 产生一条记录。因此如果最后一条 MatMul 后出现空白，不能直接断言下一条 HCCL 是尚未下发、正在排队，还是执行中卡住。需要与本仓库 Host flight recorder 的提交事件和真实 Stream Event checkpoint 配合。参考 [昇腾 Stream/异步执行说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/programug/Ascendcopdevg/docs/guide/%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97/%E7%BC%96%E8%AF%91%E4%B8%8E%E8%BF%90%E8%A1%8C/%E5%BC%82%E6%AD%A5%E6%89%A7%E8%A1%8C.md)与 [msPTI Kernel Activity 字段定义](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msPTI/docs/en/api_reference/c_api/context/msptiActivityKernel.md)。

## AICore / MTE / Vector 利用率

msPTI 26.1.0 的这两个 Python Monitor 返回算子起止时间及通信元数据，**没有** AICore/MTE/Vector 管线利用率计数字段。本脚本将 `pipe_utilization` 显式写为 `null`。不能用 MatMul 耗时或 `npu-smi` 的总 AICore 百分比推算 MTE/Vector 利用率。若现场必须获得这些指标，需要另一套支持 PipeUtilization 的性能采集配置；该模式应单独验证与 msPTI 并发是否冲突。

参考：[msPTI 26.1 Python API](https://gitcode.com/Ascend/mspti/blob/26.1.0/docs/zh/api_reference/python_api/README.md)、[官方 Python Monitor 样例索引](https://gitcode.com/Ascend/mspti/blob/master/samples/README.md)。
