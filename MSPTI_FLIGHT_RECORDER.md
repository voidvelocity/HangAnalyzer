# 统一 msPTI Flight Recorder

## 设计与证据边界

引文提出的方向基本合理：msPTI 26.1 的 Python Monitor 主要暴露计算/通信 Activity；C Callback API 提供 Runtime/HCCL domain 的 API enter/exit。一个 C++ 采集库统一写 mmap ring，故障时不依赖 Python 回调或 profiler 正常 stop，更适合保留最近的 Host 观察点。代码见 `src/hang_mspti.cpp`，格式与报告见 `flightrecorder/mspti_ring.py`。

但不能把 Runtime Callback 命名为完整的 **Host submitted**。它只覆盖进入 CANN Runtime/HCCL 的调用。torch_npu TaskQueue、vLLM Scheduler/PP/KV 阶段可能在更上游；API exit 只代表调用返回。Kernel/HCCL Activity 是完成后交给采集器的记录，周期 flush 不能迫使正在执行或卡住的任务产生记录。尤其 HCCL Activity 在 msPTI 26.1 的结构里没有 correlation ID，只能按 Rank、时间窗口、通信组和 Stream 辅助分析，不能与某次 HCCL callback 精确一一配对。

| 记录 | 可以证明 | 不能证明 |
|---|---|---|
| `runtime_enter/exit`、`hccl_enter/exit` | 对应 CANN API 已进入/返回 | 上游任务入队、对应设备任务完成 |
| `runtime_api_done` | msPTI 已交付该 Host Runtime Activity；非零 correlation 可辅助关联 | 设备执行完成 |
| `kernel_done`、`hccl_done` | msPTI 已交付已完成的设备活动记录、设备/Stream/算子名 | 尚未上报的 Task 不存在；整个 Stream 完成 |
| 原有 `.flight` + Event checkpoint | 自定义 vLLM 阶段、在**注册的真实 Stream** 上的完成水位 | 未注册 HCCL/KV Stream 的完成 |

A2/CANN 9.1.0 + msPTI 26.1.0 实测发现 Callback `correlationId` 在 Runtime enter/exit 两侧可能不同或为 0。分析器因此用 **线程 ID + callback ID** 配对，并单独统计 Runtime API Activity 与 Kernel Activity 的非零 correlation 匹配。不要用 Callback ID 跨 Rank 对齐请求。`observed_ns` 是 Host 读取/回调时的 monotonic 时间；Activity 的 `start_ns/end_ns` 属于 msPTI 时钟域，报告不把两者直接相减。若 ring 覆盖旧数据或 buffer 申请失败，报告显示 `overwritten`、`dropped_buffers`。

当前 ring 不采集 AICore/MTE/Vector 利用率，也不读取 CANN 内部 Stream 排队深度或 HCCL 等待图。这些指标需要另行验证对应性能计数器与 msPTI 并用是否可行。

## 构建与部署

每个节点在**该节点的 CANN 版本**上构建：

```bash
cd /path/to/HangAnalyzer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCANN_HOME=/usr/local/Ascend/cann
cmake --build build -j
test -f build/libhangmspti.so
```

msPTI Runtime Callback **要求在进程启动前** `LD_PRELOAD=.../libmspti.so`。A2 真机上不设置时 `msptiEnableDomain` 返回 6（`MSPTI_ERROR_WITHOUT_LD_PRELOAD`）；只在启动后 `ctypes.CDLL` 加载采集库不够。每个最终 NPU worker 还须调用 `hang_mspti_start`，或在 **spawn/exec 式 worker** 中通过 Python `sitecustomize` 引导。不要从已初始化 msPTI 的父进程 `fork` 后沿用其订阅、线程与 mmap；fork 式 worker 应在子进程完成 fork 后显式启动。当前没有已验证的 vLLM 通用自动注入，因此先在实际 worker 的启动方式上做短请求预检。

可在 spawn/exec 式服务启动环境设置：

```bash
export LD_PRELOAD=/usr/local/Ascend/cann/lib64/libmspti.so
export PYTHONPATH=/path/to/HangAnalyzer/python/bootstrap:/path/to/HangAnalyzer:$PYTHONPATH
export HANG_MSPTI_LIBRARY=/path/to/HangAnalyzer/build/libhangmspti.so
export HANG_MSPTI_DIR=/data/hang/incident-001/node0/mspti
export HANG_MSPTI_ENABLE_FILE=/home/enable_prof
export HANG_MSPTI_INTERVAL_S=0.5
printf '0\n' > /home/enable_prof
# 在上述环境内启动服务；确认所有 NPU worker 继承 LD_PRELOAD 与 PYTHONPATH。
```

`sitecustomize.py` 在 Python 启动时加载配置线程；文件内容 `0` 时不订阅采集，改为 `1` 后最多约一个间隔激活。已激活后改回 `0` 会 flush 并关闭 domain/activity，ring 保留；再次设 `1` 可恢复。文件不存在或内容不是 `1` 均视为禁用。每 worker 产生 `mspti-<PID>.msflight`；初始化失败时旁边产生 `.error`。**若 vLLM 使用 fork worker，须改用子进程初始化 hook，不能依赖这个 sitecustomize 自动路径。**这需要少量接入代码，但不修改 CANN、torch_npu 或 vLLM 的源码仓库也可通过部署方插件/启动 hook 完成，具体取决于部署版本。

在真实 worker 中显式启动的示例（必须在最终子进程中执行）：

```python
from hang_mspti import start
start('/data/hang/incident-001/node0/mspti',
      '/path/to/HangAnalyzer/build/libhangmspti.so',
      interval_s=0.5, control_file='/home/enable_prof')
```

启动后先发短请求，确认每个 worker 都有 ring、无 `.error`，且报告中有 `runtime_enter` 和 `kernel_done`。再把 `/home/enable_prof` 设为 `0`，重新确认 count 不增长；发正式故障请求前设为 `1`。容器中的 ring 目录应挂载到宿主机，并给足磁盘：默认 262144 条 × 128 字节，约 32 MiB/worker。Activity buffer 是固定的 8 × 4 MiB 池，池耗尽时增加 `dropped_buffers`，不能把缺失记录当作未执行。进程被 SIGKILL 后 mmap 文件仍可读取已发布的槽位；机器掉电前未同步到稳定存储的页面不保证保留。

## 人类可读结果

```bash
PYTHONPATH=/path/to/HangAnalyzer python3 -m flightrecorder.mspti_ring \
  /data/hang/incident-001/node0/mspti/mspti-12345.msflight --tail 80
```

报告给出各类事件数量、未闭合 Callback、每 Stream 最后一条已交付的完成记录、Runtime API Activity 与 Kernel 的 correlation 匹配数，以及最近 Host 观察时间线。先看 `status`（含启停和错误）、`overwritten/dropped_buffers`；再看有无 Runtime/HCCL enter 没有 exit；最后看每 Stream 最近的 `kernel_done/hccl_done`。若 Host API 已返回但没有后续设备 Activity，只能报告**未确认区间**，不能直接断言任务卡在该算子。需要原有 Event checkpoint 的已注册 Stream 完成水位与服务线程栈来收窄位置。

## A2 已执行的验证

在 `173.125.1.2` 的 `gl_main_a2`、A2/CANN 9.1.0、msPTI 26.1.0 上：

```bash
LD_PRELOAD=/usr/local/Ascend/cann-9.1.0/lib64/libmspti.so PYTHONPATH=. \
  python tests/a2_mspti_ring_smoke.py \
  --library build/libhangmspti.so --output /tmp/smoke.msflight --device 7

# 4、5 号卡并发 HCCL（容器内该脚本的构建路径可按实际修改）
bash tests/run_a2_mspti_hccl.sh

LD_PRELOAD=/usr/local/Ascend/cann-9.1.0/lib64/libmspti.so PYTHONPATH=.:python \
  python tests/a2_mspti_toggle_smoke.py \
  --library build/libhangmspti.so --directory /tmp/toggle-smoke
```

单卡测试捕获到 Runtime enter/exit、MatMul Kernel、Runtime API Activity；两卡测试的两个 Rank 均捕获到 HCCL enter/exit 与 HCCL Activity；开关测试确认 `0→1→0` 与算子记录。`sitecustomize` 独立进程自动引导测试也通过。进一步在 4、5、6、7 卡受控缺 Rank 卡死场景中，四个 worker 的 ring 在终止后都可读，HCCL Callback 次数准确反映缺席的 Rank，见 [四卡验证](FOUR_CARD_VALIDATION.md)。没有 `LD_PRELOAD` 的负例返回错误码 6。这些实测不代表 950PR/CANN 9.2.0 的 ABI 或性能已通过；需在目标环境逐项预检，并评估 msPTI 与其他 profiler 的并用限制。

官方参考：[msPTI 样例索引](https://gitcode.com/Ascend/mspti/tree/master/samples)、[Callback API](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msPTI/docs/en/api_reference/c_api/context/msptiCallbackData.md)、[Kernel Activity](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msPTI/docs/en/api_reference/c_api/context/msptiActivityKernel.md)。
