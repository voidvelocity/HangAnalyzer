"""Bounded four-card A2 reproduction: real streams/tasks and one missing HCCL rank.

Uses physical devices 4,5,6,7. Only this script's child PIDs are stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flightrecorder.analyzer import analyze, write_outputs
from flightrecorder.device_ops import summarize
from flightrecorder.format import read_events

DEVICES = (4, 5, 6, 7)
COMM = 700
COLLECTIVE_TYPE_ALLREDUCE = 1


def worker(args: argparse.Namespace) -> None:
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist
    from flightrecorder import Recorder, CannCheckpointManager

    rank = args.device
    local_rank = DEVICES.index(rank)
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", init_method=f"tcp://127.0.0.1:{args.port}",
                            rank=local_rank, world_size=4)
    compute = torch.npu.Stream(device=rank)
    transfer = torch.npu.Stream(device=rank)
    compute_id = int(compute.npu_stream) & 0xFFFFFFFF
    transfer_id = int(transfer.npu_stream) & 0xFFFFFFFF
    recorder = Recorder(str(args.output / "flight"), rank=rank, device=rank,
                        library=args.library, capacity=4096)
    checkpoint_manager = None
    default_stream = torch.npu.current_stream(rank)
    default_id = int(default_stream.npu_stream) & 0xFFFFFFFF
    if args.checkpoint_library:
        checkpoint_manager = CannCheckpointManager(args.checkpoint_library, slots_per_stream=4)
        checkpoint_manager.register_stream(compute_id, int(compute.npu_stream))
        checkpoint_manager.register_stream(transfer_id, int(transfer.npu_stream))
        checkpoint_manager.register_stream(default_id, int(default_stream.npu_stream))
        checkpoint_manager.start_poller(interval_us=1000, budget=32)
    (args.output / f"rank{rank}.ready").write_text(str(os.getpid()))
    while not (args.output / "START").exists():
        time.sleep(0.05)

    profiler = None
    if args.device_profile:
        profile_dir = args.output / "profiles" / f"rank{rank}"
        handler = torch_npu.profiler.tensorboard_trace_handler(
            str(profile_dir), worker_name=f"rank{rank}", analyse_flag=False)
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=handler, record_shapes=False,
            profile_memory=False, with_stack=False)
        profiler.start()

    recorder.record("REQUEST_BEGIN", correlation=101, arg0=131072)
    recorder.record("SCHEDULER_BEGIN", correlation=1)
    recorder.record("MODEL_BEGIN", stream=compute_id, correlation=1)
    with torch.npu.stream(compute):
        x = torch.randn((128, 128), device=f"npu:{rank}")
        y = torch.randn((128, 128), device=f"npu:{rank}")
        z = x @ y
        ready = torch.npu.Event()
        ready.record(compute)
    recorder.record("EVENT_RECORD", stream=compute_id, correlation=1000 + rank)
    recorder.record("KV_BEGIN", stream=transfer_id, correlation=2000 + rank,
                    arg0=rank, arg1=z.numel())
    with torch.npu.stream(transfer):
        transfer.wait_event(ready)
        recorder.record("EVENT_WAIT", stream=transfer_id, correlation=1000 + rank)
        kv = z.clone()  # transfer-stream payload stand-in; actual NPU copy/task
        transferred = torch.npu.Event()
        transferred.record(transfer)
    recorder.record("EVENT_RECORD", stream=transfer_id, correlation=2000 + rank)
    kv_seq = recorder.record("KV_END", stream=transfer_id, correlation=2000 + rank,
                             arg0=rank, arg1=kv.numel())
    if checkpoint_manager:
        checkpoint_manager.submit(transfer_id, kv_seq, checkpoint_id=6001)
    with torch.npu.stream(compute):
        compute.wait_event(transferred)
        recorder.record("EVENT_WAIT", stream=compute_id, correlation=2000 + rank)
        result = kv @ y
    model_seq = recorder.record("MODEL_END", stream=compute_id, correlation=1)
    if checkpoint_manager:
        checkpoint_manager.submit(compute_id, model_seq, checkpoint_id=5001)
    recorder.record("DEVICE_SYNC_BEGIN", correlation=1)
    torch.npu.synchronize(rank)
    recorder.record("DEVICE_SYNC_END", correlation=1)
    recorder.record("DEVICE_CONFIRMED", correlation=1, arg0=result.numel())

    # First real HCCL collective succeeds on all four cards.
    tensor = torch.ones(8, device=f"npu:{rank}") * (local_rank + 1)
    recorder.record("HCCL_BEGIN", stream=default_id, correlation=1,
                    arg0=COMM, arg1=COLLECTIVE_TYPE_ALLREDUCE)
    dist.all_reduce(tensor)
    hccl1_seq = recorder.record("HCCL_END", stream=default_id, correlation=1,
                                arg0=COMM, arg1=COLLECTIVE_TYPE_ALLREDUCE)
    if checkpoint_manager:
        checkpoint_manager.submit(default_id, hccl1_seq, checkpoint_id=7101)
    torch.npu.synchronize(rank)
    assert torch.allclose(tensor.cpu(), torch.full((8,), 10.0))
    recorder.record("DEVICE_CONFIRMED", correlation=1, arg0=10)
    if checkpoint_manager:
        deadline = time.monotonic() + 10
        while True:
            _, current = read_events(recorder.path)
            confirmed_ids = {e["correlation_id"] for e in current
                             if e["type"] == "DEVICE_CONFIRMED"}
            if {5001, 6001, 7101}.issubset(confirmed_ids):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("completed first-step checkpoints were not observed")
            time.sleep(0.001)
    recorder.record("SCHEDULER_END", correlation=1)
    if profiler is not None:
        profiler.stop()  # completed window; raw profile survives the later hang/kill
        recorder.record("CHECKPOINT", correlation=1, arg0=1)
    recorder.record("SCHEDULER_BEGIN", correlation=2)

    if rank == args.missing_rank:
        # Deliberate host-side KV gate: this rank never enters collective #2.
        recorder.record("KV_BEGIN", stream=transfer_id, correlation=3000 + rank,
                        arg0=rank, arg1=4096)
        while True:
            time.sleep(1)
    recorder.record("HCCL_BEGIN", stream=default_id, correlation=2,
                    arg0=COMM, arg1=COLLECTIVE_TYPE_ALLREDUCE)
    dist.all_reduce(tensor)  # Host API may return after asynchronous task submission.
    hccl2_seq = recorder.record("HCCL_END", stream=default_id, correlation=2,
                                arg0=COMM, arg1=COLLECTIVE_TYPE_ALLREDUCE)
    if checkpoint_manager:
        checkpoint_manager.submit(default_id, hccl2_seq, checkpoint_id=7102)
    recorder.record("DEVICE_SYNC_BEGIN", correlation=2)
    torch.npu.synchronize(rank)  # expected to wait for the missing rank
    recorder.record("DEVICE_SYNC_END", correlation=2)
    recorder.record("DEVICE_CONFIRMED", correlation=2)


def reserve_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_until(predicate, deadline: float, label: str) -> None:
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError(label)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"choose a fresh output directory: {output}")
    (output / "flight").mkdir(parents=True)
    (output / "dumps").mkdir()
    (output / "logs").mkdir()
    if args.mspti_library:
        (output / "mspti").mkdir()
        (output / "enable_prof").write_text("1")
    port = reserve_port()
    children: dict[int, subprocess.Popen] = {}
    log_handles = []
    watchdog = None
    try:
        for rank in DEVICES:
            log = (output / "logs" / f"rank{rank}.log").open("w")
            log_handles.append(log)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--device", str(rank), "--missing-rank", str(args.missing_rank),
                       "--library", args.library, "--output", str(output), "--port", str(port)]
            if args.device_profile:
                command.append("--device-profile")
            if args.checkpoint_library:
                command += ["--checkpoint-library", args.checkpoint_library]
            env = os.environ.copy()
            if args.mspti_library:
                env["LD_PRELOAD"] = args.mspti_preload + (
                    " " + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(ROOT / "python" / "bootstrap"), str(ROOT), env.get("PYTHONPATH", "")])
                env["HANG_MSPTI_LIBRARY"] = args.mspti_library
                env["HANG_MSPTI_DIR"] = str(output / "mspti")
                env["HANG_MSPTI_ENABLE_FILE"] = str(output / "enable_prof")
            children[rank] = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                              stderr=subprocess.STDOUT)
        wait_until(lambda: all((output / f"rank{r}.ready").exists() for r in DEVICES),
                   time.monotonic() + args.startup_timeout, "workers did not initialize")
        watch_log = (output / "logs" / "watchdog.log").open("w")
        log_handles.append(watch_log)
        command = [sys.executable, "-m", "flightrecorder.watchdog", "--directory",
                   str(output / "flight"), "--output", str(output / "dumps"),
                   "--timeout", str(args.hang_timeout), "--interval", "0.2", "--npu-smi"]
        for child in children.values():
            command += ["--pid", str(child.pid)]
        watchdog = subprocess.Popen(command, cwd=ROOT, stdout=watch_log,
                                    stderr=subprocess.STDOUT)
        (output / "START").write_text("1")

        def reached_hang() -> bool:
            for rank in DEVICES:
                paths = list((output / "flight").glob(f"rank{rank}-pid*.flight"))
                if not paths:
                    return False
                _, events = read_events(paths[0])
                names = [e["type"] for e in events if e["correlation_id"] == 2]
                if rank == args.missing_rank:
                    if "SCHEDULER_BEGIN" not in names:
                        return False
                elif "HCCL_BEGIN" not in names:
                    return False
            return True

        wait_until(reached_hang, time.monotonic() + args.run_timeout,
                   "collective #2 was not reached")
        wait_until(lambda: any((output / "dumps").glob("*-stalled-*")),
                   time.monotonic() + args.hang_timeout + 30,
                   "watchdog did not create a stalled snapshot")
        print("stalled snapshot captured; stopping only test workers", flush=True)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.kill()
        for child in children.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if watchdog is not None:
            try:
                watchdog.wait(timeout=15)
            except subprocess.TimeoutExpired:
                watchdog.kill(); watchdog.wait()
        for handle in log_handles:
            handle.close()

    paths = sorted((output / "dumps").rglob("*.flight"))
    if not paths:
        raise RuntimeError("no watchdog snapshot")
    report = analyze(paths, set(DEVICES))
    write_outputs(report, output / "report", paths)
    if args.mspti_library:
        from flightrecorder.mspti_ring import read_ring, report as mspti_report
        mspti_reports = output / "report" / "mspti"
        mspti_reports.mkdir()
        for rank, child in children.items():
            ring = output / "mspti" / f"mspti-{child.pid}.msflight"
            if not ring.exists():
                raise RuntimeError(f"missing msPTI ring for rank {rank}: {ring}")
            _, mspti_events = read_ring(ring)
            if not any(e["kind"] == "runtime_enter" for e in mspti_events):
                raise RuntimeError(f"no Runtime callback on rank {rank}")
            (mspti_reports / f"rank{rank}.txt").write_text(mspti_report(ring, tail=100))
    target = [c for c in report["collectives"] if c["operation_id"] == 2 and
              c["communicator_id"] == COMM]
    assert target and target[0]["missing_expected"] == [args.missing_rank], target
    assert target[0]["entered"] == [r for r in DEVICES if r != args.missing_rank]
    # HCCL_END is only the Host API return; a successful device sync for #2 must be absent.
    assert target[0]["end_observed"] == [r for r in DEVICES if r != args.missing_rank]
    for rank in DEVICES:
        if rank != args.missing_rank:
            assert report["ranks"][rank]["last_host_event"]["type"] == "DEVICE_SYNC_BEGIN"
    assert report["ranks"][args.missing_rank]["last_host_event"]["type"] == "KV_BEGIN"
    if args.checkpoint_library:
        for rank in DEVICES:
            if rank == args.missing_rank:
                continue
            streams = report["ranks"][rank]["device_checkpoint_progress"]
            hccl_progress = [p for p in streams.values() if
                (p["first_unconfirmed"] and p["first_unconfirmed"]["correlation_id"] == 7102) or
                (p["confirmed_through"] and p["confirmed_through"]["correlation_id"] == 7102)]
            assert hccl_progress, (rank, streams)
            # The default Stream checkpoint can complete while an internal HCCL
            # Stream remains stuck.  The open DEVICE_SYNC scope is the broader
            # completion boundary; the analyzer reports this as a coverage gap.
            if hccl_progress[0]["first_unconfirmed"] is None:
                assert report["ranks"][rank]["checkpoint_coverage_warning"]
    (output / "scenario.json").write_text(json.dumps({"devices": DEVICES,
        "missing_rank": args.missing_rank, "communicator_id": COMM,
        "successful_operation": 1, "hanging_operation": 2,
        "note": "rank 6 is held at a synthetic host KV gate; other ranks enter a real HCCL all_reduce"}, indent=2))
    if args.device_profile:
        import torch_npu
        profile_roots = sorted((output / "profiles").glob("rank*/*_ascend_pt"))
        analysis_errors = []
        for root in profile_roots:
            try:
                torch_npu.profiler.profiler.analyse(str(root))
            except Exception as exc:
                analysis_errors.append(f"{root}: {exc}")
        counts = summarize(output / "profiles", output / "report" / "device_operators.txt")
        csv_output = output / "report" / "device_kernel_csv"
        csv_output.mkdir(parents=True, exist_ok=True)
        for rank in DEVICES:
            files = sorted((output / "profiles" / f"rank{rank}").rglob("kernel_details.csv"))
            if files:
                shutil.copyfile(files[0], csv_output / f"rank{rank}.csv")
        (output / "report" / "device_profile_status.json").write_text(
            json.dumps({"profile_roots": [str(p) for p in profile_roots],
                        "kernel_counts": counts, "analysis_errors": analysis_errors}, indent=2))
        if not counts or any(counts.get(rank, 0) == 0 for rank in DEVICES):
            raise RuntimeError("device profiler produced no kernels on at least one rank")
        with (output / "report" / "report.txt").open("a") as summary:
            summary.write("\nDEVICE KERNELS (completed pre-hang window)\n")
            for rank in DEVICES:
                summary.write(f"rank {rank}: {counts.get(rank, 0)} kernels\n")
            summary.write("See device_operators.txt and device_kernel_csv/. "
                          "The hanging collective is outside this completed profile window.\n")
    print(f"PASS: {output / 'report' / 'report.txt'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--device", type=int, choices=DEVICES)
    p.add_argument("--missing-rank", type=int, choices=DEVICES, default=6)
    p.add_argument("--library", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--hang-timeout", type=float, default=5)
    p.add_argument("--startup-timeout", type=float, default=90)
    p.add_argument("--run-timeout", type=float, default=90)
    p.add_argument("--device-profile", action="store_true",
                   help="Profile and persist the completed first scheduler step on each NPU")
    p.add_argument("--checkpoint-library",
                   help="libflightcheckpoint_cann.so; enables per-stream nonblocking checkpoints")
    p.add_argument("--mspti-library", help="libhangmspti.so; record callbacks and activities during hang")
    p.add_argument("--mspti-preload", help="absolute path to libmspti.so; required with --mspti-library")
    args = p.parse_args()
    if bool(args.mspti_library) != bool(args.mspti_preload):
        p.error("--mspti-library and --mspti-preload must be used together")
    if args.worker:
        worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
