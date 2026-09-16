# Ascend A2 / CANN 9.1.0 验证记录

验证时间：2026-09-16。远端机器 `173.125.1.2`，容器 `gl_main_a2`，临时目录 `/tmp/HangAnalyzer-validation`。

## 环境

- AArch64 Linux 5.10，8 张 Ascend 910B4（A2），`npu-smi` 健康状态为 OK。
- CANN 9.1.0，驱动 25.2.2。
- Python 3.12.13，PyTorch 2.10.0，torch_npu 2.10.0.post4。
- GCC 13.3.0，CMake 4.4.2。

## 已验证能力

1. Linux/AArch64 编译 `libflightrecorder.so` 成功。
2. C++ 4 线程并发写入 200,000 个事件，所有返回 sequence 唯一连续，文件中 200,000 个槽位逐一校验。两次完整运行测得约 **327.7–343.7 ns/event**；这是该容器本次微基准结果，不作为所有负载下的保证。
3. 10 个离线分析/解析测试覆盖 C++ ABI 对齐、collective 缺 Rank、collective 类型不一致、Event 依赖环候选、Host open scope、checkpoint generation、未确认区间和内部 Stream 覆盖缺口；另有 watchdog 与跨 Rank 两个集成测试，最终共 12 个 Python 测试通过。
4. watchdog 集成测试在子进程仍存活但 0.5 秒无进展时产生 stalled snapshot，保存 flight、`/proc` 信息和 `npu-smi`；子进程退出后产生 exit snapshot；合并快照后正确报告最后停在 `PP_RECV_BEGIN`，并识别 3 个未闭合 Host scope。
5. 真机 `torch_npu` 测试在 `npu:7` 下发 256×256 矩阵乘，先记录异步 `MODEL_END`，调用 `torch.npu.synchronize(7)` 成功后记录 `DEVICE_CONFIRMED`。离线报告得到 5 个事件并显示最后一次明确的 device confirmation。
6. 多进程 C++ writer 到分析器的跨 Rank 测试使用 Rank 0、1、3 进入 communicator 77 / operation 9001，正确报告 Rank 2 缺失、collective type mismatch，以及 Rank 3 最后停在 `PP_RECV_BEGIN`。
7. C++ checkpoint 合成后端验证每 Stream 固定池、池耗尽 `EAGAIN`、非阻塞 pending/complete、后台 poller、pending destroy 和物理 Event generation 复用。
8. A2/CANN 9.1.0 真机验证计算与传输 Stream 独立 Event 池。计算 checkpoint 首次 query 为 NOT_READY；后台 C++ poller在业务线程之外确认完成；单槽 Event 复用 generation 从 1 增至 2。
9. 四卡真实 HCCL 缺 Rank复验启用 checkpoint 后通过。默认 Stream checkpoint 7102 已完成而 `DEVICE_SYNC` 仍卡住，Analyzer 输出覆盖缺口警告，没有把默认 Stream 完成误判为内部 HCCL 完成。

## 执行命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -V

FLIGHT_TEST_LIBRARY=$PWD/build/libflightrecorder.so \
FLIGHT_TEST_NPU_SMI=1 \
python3 -m unittest discover -s tests -v

python3 tests/a2_torch_npu_smoke.py \
  --library $PWD/build/libflightrecorder.so \
  --directory /tmp/a2-flight-smoke --device 7
```

checkpoint CANN 适配器与真机测试命令见 [CHECKPOINTS.md](CHECKPOINTS.md)，四卡报告见 [results/a2-fourcard-checkpoint-20260916/report/report.txt](results/a2-fourcard-checkpoint-20260916/report/report.txt)。

## 验证中发现并修复

- Python 解析器最初未处理 C++ `flight_event` 在 `correlation_id` 前的 4 字节 ABI padding。真机 smoke test 通过 `arg0` 精确值检查发现该问题。格式已修正为 `<QQIIHHIHH4xQQQ>`，C++ 增加 `sizeof/offsetof` 编译期断言，Python 增加固定值回归测试。
- watchdog 最初在 snapshot 目录建立后才逐个写文件，外部读取者可能看到不完整目录。现在先写隐藏 partial 目录，完成 `metadata.json` 后原子 rename。

## 结论边界

A2 测试证明 recorder、watchdog、离线分析器和显式同步后的 device confirmation 可在 CANN 9.1.0 环境工作。它没有验证 CANN 9.2.0/950PR ABI，也没有证明仅凭 Host 埋点能获知每个异步 NPU task 的完成状态。vLLM、PP、split KV cache 的实际调用点仍需按部署版本接入；跨 Rank collective ID 仍必须由集成层明确提供。

后续补充的四卡真实 HCCL 卡死实验与可读报告见 [FOUR_CARD_VALIDATION.md](FOUR_CARD_VALIDATION.md)。
