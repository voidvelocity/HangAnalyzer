# A2 四卡多 Stream / HCCL 卡死复现

## 统一 msPTI ring 复验（2026-09-18）

在相同的 4、5、6、7 卡受控缺 Rank 场景中，为四个 worker 用 `LD_PRELOAD=libmspti.so` 与 `sitecustomize` 启动 `libhangmspti.so`，保持 0.5 秒 Activity flush。控制器捕获 stalled 快照后，仅停止自己启动的 worker。执行命令：

```bash
PYTHONPATH=. python examples/four_card_hccl_hang.py \
  --library /tmp/hang-mspti-build/libflightrecorder.so \
  --checkpoint-library /tmp/hang-mspti-build/libflightcheckpoint_cann.so \
  --mspti-library /tmp/hang-mspti-build/libhangmspti.so \
  --mspti-preload /usr/local/Ascend/cann-9.1.0/lib64/libmspti.so \
  --output /tmp/hang-fourcard-mspti-249457 --hang-timeout 5
```

运行结果 `PASS`。四个 PID 的 `.msflight` 都在卡死/终止后可读，`overwritten=0`、`dropped_buffers=0`，且有 Runtime enter/exit、Kernel Activity 和 HCCL Activity。Rank 4/5/7 各观察到 **2 次 HCCL enter/exit**，Rank 6 只观察到 **1 次**；四卡均只交付了第一轮的 HCCL 完成 Activity。第二轮未交付 HCCL 完成记录，需与显式 `.flight` 的缺 Rank 和开放 `DEVICE_SYNC_BEGIN` 一起判读，不能只凭缺失 Activity 断言设备任务未运行。人类可读原始报告见 [rank4](results/a2-fourcard-mspti-20260918/report/mspti/rank4.txt)、[rank5](results/a2-fourcard-mspti-20260918/report/mspti/rank5.txt)、[rank6](results/a2-fourcard-mspti-20260918/report/mspti/rank6.txt)、[rank7](results/a2-fourcard-mspti-20260918/report/mspti/rank7.txt) 和 [跨 Rank 显式事件报告](results/a2-fourcard-mspti-20260918/report/report.txt)。原始 33 MiB/PID 的 ring 保留在测试容器 `/tmp/hang-fourcard-mspti-249457/mspti/`，未纳入仓库。

## 场景与结果

在 `173.125.1.2` 的 `gl_main_a2` 容器中，分别用一个进程绑定 Ascend A2 的 4、5、6、7 号卡。每个进程创建计算 Stream 和传输 Stream，运行矩阵乘、跨 Stream Event 等待、KV 数据副本任务，再进行设备同步。第一轮四卡 HCCL AllReduce 正常完成。

第二轮让 **6 号卡故意停在模拟的 Host KV transfer gate**，不调用 HCCL AllReduce；4、5、7 号卡调用真实的 `torch.distributed.all_reduce`。这三个 Rank 的 HCCL Host API 均返回，然后卡在 `torch.npu.synchronize()`，第二轮没有 `DEVICE_SYNC_END` 或 `DEVICE_CONFIRMED`。这是受控缺 Rank 场景；模拟的是 KV 前置阶段阻断，未执行真实 vLLM 的 split KV 传输。

watchdog 在进程存活时取得四个 Rank 的 `.flight`、`/proc` 线程信息与 `npu-smi`，随后控制器只结束本脚本启动的四个子进程，并取得退出快照。最终断言和报告生成均通过。实验结束后 4–7 号卡在 `npu-smi info` 中为 `OK`。

最重要的现场结论：

```text
comm=700 op=1: entered=[4,5,6,7], Host END=[4,5,6,7]
comm=700 op=2: entered=[4,5,7],   missing=[6]
rank6: last=KV_BEGIN corr=3006
rank4/5/7: last=DEVICE_SYNC_BEGIN corr=2
last device confirmation on all ranks: corr=1
```

正常的计算 Stream → 传输 Stream → 计算 Stream 依赖没有被误报为环。最初版本曾把 stream 级往返当作死锁；现改为按 Event 操作顺序建依赖图，并增加正反两个回归用例。

## 重现方法

在目标机器的容器中执行，输出目录需是未使用的新路径：

```bash
cd /tmp/HangAnalyzer-validation
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 examples/four_card_hccl_hang.py \
  --library "$PWD/build/libflightrecorder.so" \
  --output /tmp/hang-a2-fourcard-new \
  --hang-timeout 5
```

控制器默认等待初始化 90 秒、故障阶段 90 秒；所有子进程均由控制器持有并在快照后结束。若失败，仍可查看 `logs/rank*.log` 和 `logs/watchdog.log`。该实验会真实占用 4–7 卡并发起 HCCL collective，应在可用于实验的环境中运行。

手工对任何一次快照重新分析：

```bash
python3 -m flightrecorder.analyzer /tmp/hang-a2-fourcard-new/dumps \
  --communicator-id 700 --expected-ranks 4,5,6,7 \
  --output /tmp/hang-a2-fourcard-new/recheck
```

## 人类可读结果怎么看

本次通过的实验结果已复制到仓库 [results/a2-fourcard-20260916/report/report.txt](results/a2-fourcard-20260916/report/report.txt)。

1. `cat report/report.txt`：先看 `HOST PROGRESS` 中每 Rank 最后的 Host 事件和未闭合 scope，再看 `COLLECTIVES` 的 `missing_expected`，最后看 `BLOCKING EVIDENCE`。`HCCL_END` 仅代表 Host API 返回。
2. `less -S report/events.txt`：按 Rank 查看 `seq`、该 Rank 内的相对毫秒时间、Stream、事件名、correlation 和参数。对应文件见 [events.txt](results/a2-fourcard-20260916/report/events.txt)。不同进程的相对毫秒不用于跨 Rank 绝对排序。
3. `cat report/snapshots.txt`：找最早的 `stalled-*` 活体快照。对应目录下 `metadata.json` 记录四 Rank 的 PID、seq 和采集错误；`pid<PID>/tid<TID>.stack`、`.wchan` 可查看线程现场；`npu-smi.txt` 可查看设备健康状态。本次 metadata 的 `errors=[]`，`npu_smi_returncode=0`。
4. `report/report.json` 便于程序查询；`report/trace.json` 是 Chrome Trace 格式，可在支持该格式的 trace viewer 中打开。

本次完整远端现场保留在容器 `/tmp/hang-a2-fourcard-20260916-b/`。仓库中的 `results/` 只保存可读报告和结构化报告，未复制原始 `.flight` 与全部 `/proc` 快照。长期保留时请先将该远端目录归档到正式存储。

## 保存真正的设备算子

需要设备 kernel 名称与执行耗时，可在复现命令中添加 `--device-profile`，并把 `--hang-timeout` 调大到 10 秒。脚本用每个 Rank 的 `torch_npu.profiler` 只采集**第一轮已完成的 scheduler step**，在第二轮故意卡死前结束并落盘。控制器在 worker 退出后离线解析 profile，输出：

- `report/device_operators.txt`：按 Rank 列出的设备 kernel 名称、设备开始时间和执行耗时。
- `report/device_kernel_csv/rank4.csv` 至 `rank7.csv`：原始 `kernel_details.csv` 的便于分享的副本。
- `report/device_profile_status.json`：profile 路径、每 Rank 的 kernel 数及解析错误。
- `profiles/rank*/..._ascend_pt/`：完整 torch_npu profiler 原始目录和分析结果。

四卡带 profiler 的重现实验已通过，目录在容器 `/tmp/hang-a2-fourcard-profile-20260916/`；每卡解析出 **7 个实际执行的设备 kernel**，包括 `aclnnMatmul_MatMulCommon_MatMulV2` 和 `hcom_allReduce__612_0_1`，四卡 `analysis_errors=[]`。仓库内可直接查看 [设备算子列表](results/a2-fourcard-profile-20260916/report/device_operators.txt)。

```bash
python3 examples/four_card_hccl_hang.py \
  --library "$PWD/build/libflightrecorder.so" \
  --output /tmp/hang-a2-fourcard-with-device-profile \
  --hang-timeout 10 --device-profile

less -S /tmp/hang-a2-fourcard-with-device-profile/report/device_operators.txt
less -S /tmp/hang-a2-fourcard-with-device-profile/report/device_kernel_csv/rank4.csv
```

**边界：**这些 kernel 来自故障前已经关闭的 profile 窗口，能证明第一轮确实在 NPU 上执行了哪些算子。第二轮缺 Rank HCCL 所提交的设备 task 无法由这个已关闭的窗口直接列出；该阶段仍依据 Flight Recorder 的 Host 事件、缺 Rank 对齐与未返回的设备同步判断。若把常规 profiler 一直开到卡死，原始数据是否完整落盘取决于其缓冲与退出处理，不能作为可靠保证。[vLLM-Ascend 的 profiling 指南](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/performance_and_debug/service_profiling_guide.html)也将 `kernel_details.csv` 列为 torch_npu profiler 的设备 kernel 输出。

## Per-Stream checkpoint 四卡复验

启用 CANN Event checkpoint 的命令：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCANN_HOME=/usr/local/Ascend/cann-9.1.0
cmake --build build -j
python3 examples/four_card_hccl_hang.py \
  --library "$PWD/build/libflightrecorder.so" \
  --checkpoint-library "$PWD/build/libflightcheckpoint_cann.so" \
  --output /tmp/hang-a2-fourcard-checkpoint \
  --hang-timeout 5
```

2026-09-16 真机复验通过，结果见 [checkpoint 报告](results/a2-fourcard-checkpoint-20260916/report/report.txt)。Rank 6 最后停在 `KV_BEGIN`，Rank 4/5/7 最后停在 `DEVICE_SYNC_BEGIN`。4/5/7 上默认 Stream 的 checkpoint 7102 均已明确完成，但设备同步未返回，因此报告给出 `COVERAGE WARNING`。这证明 HCCL 未完成工作不受该默认 Stream Event 覆盖；接入 vLLM 时必须取得实际 HCCL/KV Stream handle，或把设备级同步作为更宽的完成边界。

## 证据边界

本实验验证真实 A2 多 Stream、NPU task 和 HCCL 异步执行下，工具可区分 Host 下发、已观察 Stream 的 Event 完成与设备级同步完成，并定位缺席的 Rank。它不等同于 vLLM-Ascend 的实际 PP + split KV 故障，也不证明 CANN 9.2.0 / 950PR 上的行为。工具尚未读取 CANN 内部 Task 完成队列或 HCCL 内部等待图。
