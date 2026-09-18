"""Behavioral test without requiring NPU hardware or msPTI installation."""
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mspti_activity_capture import MsptiCapture, _record


class FakeData:
    device_id = 4
    stream_id = 7
    correlation_id = 11
    name = "MatMul"
    type = "AI_CORE"
    start = 100
    end = 250


class FakeComm(FakeData):
    name = "AllReduce"
    comm_name = "test_group"
    alg_type = "RING"
    count = 1024
    data_type = 3


class FakeMonitor:
    instances = []

    def __init__(self):
        self.callback = None
        self.flushes = 0
        self.starts = 0
        self.instances.append(self)

    def set_buffer_size(self, size):
        return 0 if size > 0 else 1

    def start(self, callback):
        self.callback = callback
        self.starts += 1
        return 0

    def flush_all(self):
        self.flushes += 1
        for monitor in self.instances:
            if monitor.callback:
                monitor.callback(FakeData() if isinstance(monitor, FakeKernel) else FakeComm())
        return 0

    def stop(self):
        self.callback = None
        return 0


class FakeKernel(FakeMonitor):
    pass


class FakeCommunication(FakeMonitor):
    pass


class CaptureTest(unittest.TestCase):
    def test_missing_device_timestamps_are_not_claimed_as_completion(self):
        data = FakeData()
        data.start = 0
        data.end = 0
        record = _record("kernel", data)
        self.assertFalse(record["timestamp_valid"])
        self.assertIsNone(record["duration_ns"])

    @staticmethod
    def wait_for(predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError("timed out waiting for monitor state")
            time.sleep(0.005)

    def test_periodic_flush_persists_both_activity_types(self):
        FakeMonitor.instances.clear()
        module = types.ModuleType("mspti")
        module.KernelMonitor = FakeKernel
        module.CommunicationMonitor = FakeCommunication
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"mspti": module}):
            path = Path(directory) / "activity.jsonl"
            capture = MsptiCapture(path, interval_s=0.05, enable_file=None)
            capture.start()
            time.sleep(0.13)
            result = capture.stop()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            kinds = {row["kind"] for row in rows}
            self.assertEqual(kinds, {"kernel", "communication", "flush_tick", "capture_state"})
            self.assertGreaterEqual(result["counts"]["flush_tick"], 2)
            self.assertEqual(result["dropped"], 0)
            self.assertEqual(result["flush_errors"], 0)
            self.assertEqual(next(row for row in rows if row["kind"] == "kernel")["duration_ns"], 150)
            self.assertTrue(path.with_suffix(".summary.json").exists())

    def test_file_disables_then_enables_and_disables_monitor(self):
        FakeMonitor.instances.clear()
        module = types.ModuleType("mspti")
        module.KernelMonitor = FakeKernel
        module.CommunicationMonitor = FakeCommunication
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"mspti": module}):
            enable = Path(directory) / "enable_prof"
            output = Path(directory) / "activity.jsonl"
            capture = MsptiCapture(output, interval_s=0.05, enable_file=enable)
            capture.start()
            time.sleep(0.11)
            self.assertFalse(capture.monitor_active)
            self.assertTrue(all(m.starts == 0 for m in FakeMonitor.instances))
            enable.write_text("1\n")
            self.wait_for(lambda: capture.monitor_active)
            self.wait_for(lambda: capture.written["kernel"] > 0)
            enable.write_text("0\n")
            self.wait_for(lambda: not capture.monitor_active)
            before = capture.written["kernel"]
            time.sleep(0.11)
            self.assertEqual(capture.written["kernel"], before)
            enable.write_text("1\n")
            self.wait_for(lambda: capture.monitor_active)
            capture.stop()
            self.assertEqual(FakeMonitor.instances[0].starts, 2)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            states = [r["enabled"] for r in rows if r["kind"] == "capture_state"]
            self.assertEqual(states, [True, False, True, False])
            self.assertEqual(capture.summary()["transitions"], 4)


if __name__ == "__main__":
    unittest.main()
