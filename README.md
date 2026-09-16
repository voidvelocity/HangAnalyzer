# Ascend Hang Flight Recorder（显式埋点 MVP）

用于在 vLLM-Ascend 服务卡住、profiler 无法正常停止时保留最近的 **Host** 事件和进程现场。支持两台机器分别采集、离线合并。当前没有 CANN/HCCL ABI hook，也不会把异步 API 返回解释成 NPU 已完成。

Ascend A2 / CANN 9.1.0 的真机环境、测试项与结果见 [VALIDATION.md](VALIDATION.md)。

将工具接入真实两机 vLLM-Ascend 服务的步骤见 [VLLM_TWO_NODE_GUIDE.md](VLLM_TWO_NODE_GUIDE.md)。

Per-Stream CANN Event checkpoint 的构建、接入、轮询和证据边界见 [CHECKPOINTS.md](CHECKPOINTS.md)。

4–7 号卡的真实多 Stream / HCCL 缺 Rank 卡死复现、重现命令和人类可读报告说明见 [FOUR_CARD_VALIDATION.md](FOUR_CARD_VALIDATION.md)。

需要保存设备上实际执行的 kernel 时，可运行该复现脚本的 `--device-profile` 模式；它对已完成的阶段分段落盘，并生成 `device_operators.txt` 与每 Rank 的 `kernel_details.csv`。这类采集有明显额外开销，适合复现环境。

## 构建与启动

目标环境为 Linux。需要 C++17、CMake、Python 3.9+；构建本身不依赖 CANN。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
mkdir -p /data/flight /data/hang
python3 -m flightrecorder.watchdog --directory /data/flight --output /data/hang \
  --timeout 30 --log /path/to/vllm.log --log /path/to/hccl.log --npu-smi
```

每台机器都在请求前启动 watchdog，并给进程配置各自的 `FLIGHT_DIR`。若已有 worker PID，增加可重复的 `--pid PID`，这样该进程退出后 watchdog 会结束。`--timeout` 要大于正常长 prefill 中的最长无事件间隔。

在对应 worker 初始化后、发请求前：

```python
from flightrecorder import Recorder
flight = Recorder("/data/flight", rank=global_rank, device=local_device,
                  library="/path/to/build/libflightrecorder.so")
flight.record("REQUEST_BEGIN", correlation=request_hash, arg0=prompt_tokens)
flight.record("SCHEDULER_BEGIN", correlation=scheduler_step)
flight.record("MODEL_BEGIN", correlation=scheduler_step)
flight.record("PP_SEND_BEGIN", stream=stream_id, correlation=transfer_id,
              arg0=peer_rank, arg1=payload_bytes)
# 实际 PP/KV/HCCL 调用
flight.record("PP_SEND_END", stream=stream_id, correlation=transfer_id,
              arg0=peer_rank, arg1=payload_bytes)
flight.record("MODEL_END", correlation=scheduler_step)
```

将埋点放在 **真实调用两侧**，尤其是 scheduler、PP send/recv、KV connector 的提交与等待、HCCL、EventRecord/Wait、stream/device sync。每个 Begin/End 使用同一 `correlation` 和 `stream`。`HCCL_BEGIN/END` 的 `arg0` 必须是跨 Rank 一致的 communicator ID、`correlation` 必须是跨 Rank 一致的组内操作 ID、`arg1` 为调用方统一约定的 collective 类型编号。不同 PP stage 的通信组请分开分析。不要对原始指针值直接当作跨 Rank 统一 ID。`DEVICE_CONFIRMED` 仅在确证设备完成之后记录。

首版 Python `ctypes` 封装适合低频的阶段性埋点；高频路径应直接链接 C API，测量开销后启用。每 Rank 默认约 64 MiB。文件用 `O_EXCL` 创建，重启时 PID 新文件不会覆盖旧现场；启动新诊断前应清理或更换目录。

## 分析

将两台机器同一次故障的 snapshot 目录复制到分析机的同一父目录，文件名中带 Rank/PID。以对应 communicator 的全局 Rank 集合分析：

```bash
python3 -m flightrecorder.analyzer /data/merged-incident \
  --communicator-id 700 --expected-ranks 0,1,2,3 \
  --output /data/merged-incident/report
```

输出 `report.txt`（阻塞摘要）、`events.txt`（逐 Rank 全部保留事件）、`snapshots.txt`（快照路径）、`report.json` 和 `trace.json`。`trace.json` 合并同一 PID 各快照中仍保留的全部事件。原始 `.flight` 文件保留完整 ring。报告中 `missing_expected` 表示在该快照窗口未看到 Begin；若 ring 已覆盖、组 ID 错误或该 Rank 未埋点，结论不能成立。`/proc/*/stack` 需要系统权限，权限不足时 watchdog 会保存 `.error`。

`--npu-smi` 在快照时执行一次 `npu-smi info`，保存设备健康/利用率现场；它不能提供逐 Task completion。HCCL/CANN 设备日志可用重复的 `--log` 附加，确保选取当前环境实际日志路径。

## 已知限制

- 采集的是 Host 事件，不能直接显示 NPU 上最后完成的算子或 HCCL 内部 task。需要 CANN 9.2.0 目标环境的独立 device 侧证据。
- 进程收到 SIGKILL 后无法再取 live stack；提前触发的 stalled snapshot 可保留当时 `/proc` 现场。
- Python `ctypes` 每次调用有额外开销，不代表 C++ writer 的热路径性能。
- 版本 1 的 watchdog 以“无 recorder 事件”判停滞；长时间正常执行需通过 checkpoint 和阈值避免误报。

参考： [CANN 9.2.0 beta2 Event 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclpythondevg_01_0093.html)说明 RecordEvent 是异步下发；[vLLM-Ascend KVPP 设计](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/kvpp.html)说明 PP stage 内的 KV broadcast 和独立 transfer stream。实际接口以部署版本为准。
