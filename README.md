# Ascend Hang Analyzer

用于在 vLLM-Ascend 服务卡住、profiler 无法正常停止时保留现场。推荐的统一 msPTI 采集器在每个 worker 内把 **CANN Runtime/HCCL API 进入与返回**、**已上报的 Kernel/HCCL Activity** 写到同一个 mmap ring；独立的显式埋点与 per-stream Event checkpoint 可补充请求、PP/KV 语义与设备完成水位。两机分别采集后离线对齐。

先读 [统一 msPTI 采集器指南](MSPTI_FLIGHT_RECORDER.md)。它包含构建、`LD_PRELOAD`、`/home/enable_prof`、0.5 秒刷新、人类可读报告和 A2 真机测试命令。**Runtime callback 只能说明 CANN API 被调用，不能证明 torch_npu 上层队列或 Device 执行；Activity 缺失也不能证明算子没有运行。**

要调查通信是否占用 AICPU，使用 [AICPU 卡死观察指南](AICPU_HANG_GUIDE.md)：独立采样设备总体 AICPU 利用率、保留已完成 Kernel 的类型，并在正常复现中用 msprof 对照 Task Type。

Ascend A2 / CANN 9.1.0 的真机环境、测试项与结果见 [VALIDATION.md](VALIDATION.md)。

将工具接入真实两机 vLLM-Ascend 服务的步骤见 [VLLM_TWO_NODE_GUIDE.md](VLLM_TWO_NODE_GUIDE.md)。

Per-Stream CANN Event checkpoint 的构建、接入、轮询和证据边界见 [CHECKPOINTS.md](CHECKPOINTS.md)。

4–7 号卡的真实多 Stream / HCCL 缺 Rank 卡死复现、重现命令和人类可读报告说明见 [FOUR_CARD_VALIDATION.md](FOUR_CARD_VALIDATION.md)。

需要保存设备上实际执行的 kernel 时，优先使用统一 msPTI ring；旧四卡复现脚本的 `--device-profile` 模式仍可用于对照，它在故障前结束 profile 窗口并生成 `device_operators.txt` 与 `kernel_details.csv`。

## 构建与启动

目标环境为 Linux。需要 C++17、CMake、Python 3.9+。不指定 `CANN_HOME` 时仅构建原有显式埋点和 checkpoint 核心；指定后还构建 `libhangmspti.so`。

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

## 原有显式埋点路径的限制

- `.flight` 采集的是显式 Host 事件；统一 `.msflight` 采集器则能显示 msPTI 已上报的设备算子。两者为不同格式和证据来源，当前需要人工对照。
- 进程收到 SIGKILL 后无法再取 live stack；提前触发的 stalled snapshot 可保留当时 `/proc` 现场。
- Python `ctypes` 每次调用有额外开销，不代表 C++ writer 的热路径性能。
- 版本 1 的 watchdog 以“无 recorder 事件”判停滞；长时间正常执行需通过 checkpoint 和阈值避免误报。

参考： [CANN 9.2.0 beta2 Event 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclpythondevg_01_0093.html)说明 RecordEvent 是异步下发；[vLLM-Ascend KVPP 设计](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/kvpp.html)说明 PP stage 内的 KV broadcast 和独立 transfer stream。实际接口以部署版本为准。
