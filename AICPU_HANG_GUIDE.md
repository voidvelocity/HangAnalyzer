# 卡死时观察 AICPU 与通信任务

## 当前能看到什么

`libhangmspti.so` 的 v2 ring 现在保存 msPTI Kernel Activity 的 `type` 和 HCCL Activity 的通信组名。报告按 `KERNEL_AICORE`、`KERNEL_AIVEC` 等类型汇总已交付的完成记录；若目标版本上报 `KERNEL_AICPU`，也会保留。但 Activity 必须先完成并从设备上报，**正在等待、尚未结束的 AICPU/HCCL 任务不会因为 0.5 秒 flush 自动出现**。HCCL Activity 的 `HcclAllReduce` 名称不能证明这次通信占用了 AICPU。

独立的 [AICPU 采样器](flightrecorder/aicpu_sampler.py)每轮在指定设备并行运行 `npu-smi info -t usages -i <device>`，记录 `Aicpu/Aicore/Aivector/Ctrlcpu Usage Rate(%)`、命令起止墙钟时间和错误。它不订阅 msPTI，可在业务进程卡住期间继续采样；数据是**设备总体占用率**，没有 PID、算子名、通信组或等待原因。若 `npu-smi` 单次查询耗时超过 0.5 秒，实际采样间隔也会超过 0.5 秒，以 JSONL 时间戳为准。样本按轮次 `fsync`。

```bash
# 在每台机器、业务进程之外启动；按当地物理卡号修改。
PYTHONPATH=/path/to/HangAnalyzer python3 -m flightrecorder.aicpu_sampler capture \
  --devices 4,5,6,7 --interval 0.5 \
  --output /data/hang/incident-001/node0/aicpu_usage.jsonl

# 故障后查看可读汇总；JSONL 保留每个样本的时间与百分比。
PYTHONPATH=/path/to/HangAnalyzer python3 -m flightrecorder.aicpu_sampler summarize \
  /data/hang/incident-001/node0/aicpu_usage.jsonl

PYTHONPATH=/path/to/HangAnalyzer python3 -m flightrecorder.mspti_ring \
  /data/hang/incident-001/node0/mspti/mspti-<PID>.msflight --tail 100
```

## 如何判断“通信占用了 AICPU”

1. 用短请求、正常长请求、故障请求分别采样，比较故障前后的 AICPU 曲线；高占用只能说明该**设备** AICPU 繁忙，不能直接归因于 HCCL。
2. 对照 `.msflight` 的 HCCL enter/exit、最后完成的 HCCL Activity、Kernel type，以及 `.flight` 的 PP/KV/同步阶段。若 HCCL API 已返回但第二轮 HCCL Activity 缺失，只能写“设备完成未确认”。
3. 在**独立的正常复现**中运行 msprof 基线，获得 `task_time_*.csv` 的 `kernel_type`、`op_summary_*.csv` 的 `Task Type`，以及通信任务明细。A2 容器已验证命令：

   ```bash
   msprof --output=/tmp/aicpu-baseline --task-time=on --aicpu=on --hccl=on \
     python your_single_worker.py
   ```

   两 Rank 分别启动时需要相同的 HCCL rendezvous 配置。不要在同一 worker 同时启动本工具的 msPTI 采集与 msprof；二者的 profiler 状态与 buffer 可能冲突。msprof 的可读 CSV/DB 通常依赖采集正常停止和导出，因此它用于验证该通信模式的 **AICPU/SDMA/其他 Task 类型基线**，不能保证保存卡死中未完成的任务。官方 [msProf 快速入门](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msProf/docs/en/quick_start/msprof_quick_start.md)说明 `op_summary` 的 `Task Type` 可区分 AICore/AICPU；[DB 格式说明](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/msProf/docs/en/user_guide/profile_data_file_references_db.md)给出通信小任务与 `TASK` 表的关联。

950PR 的实际通信加速模式必须现场核实，不能把所有 HCCL 算子预设为 AICPU 执行；[CANN 的 950PR 发布说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/releasenote/release-notes.md)也列有 CCU 通信加速。若 AICPU 占用率在卡死时很高，而已完成 Activity 没有对应 AICPU 记录，当前公开的 msPTI Activity/`npu-smi` 组合仍**无法列出该时刻正在执行的 AICPU 函数或判断它在等待谁**；需要目标版本提供的 AICPU/Task trace、设备日志或厂商诊断接口。

## A2 受控卡死实测

在 `173.125.1.2` 的 `gl_main_a2` 容器复现 4、5、6、7 卡缺 Rank HCCL 等待，每卡取得 26 个利用率样本，AICPU 均为 0%。本机原始 JSONL 与可读报告保存在 `results/a2-fourcard-aicpu-20260918/`；这些快照不能排除采样间隙中的短暂 AICPU 活动。四份 msPTI 可读报告记录到 `KERNEL_AICORE`、`KERNEL_AIVEC` 与第一轮已完成的 `HcclAllReduce`，第二轮未交付完成 Activity。独立的正常两卡 HCCL msprof 基线 CSV 保存在本机 `results/a2-msprof-aicpu-20260918/`；rank 0 的任务类型计数为 `EVENT_RECORD=30`、`EVENT_WAIT=20`、`AI_VECTOR_CORE=20`、`SDMA=10`，没有 AICPU 类型。原始运行数据未推送到远程仓库。该 A2 结果不能外推至 950PR/CANN 9.2.0。
