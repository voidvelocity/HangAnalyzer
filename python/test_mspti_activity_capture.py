"""Behavioral test without requiring NPU hardware or msPTI installation."""
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mspti_activity_capture import MsptiCapture


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
        self.instances.append(self)

    def set_buffer_size(self, size):
        return 0 if size > 0 else 1

    def start(self, callback):
        self.callback = callback
        return 0

    def flush_all(self):
        self.flushes += 1
        for monitor in self.instances:
            if monitor.callback:
                monitor.callback(FakeData() if isinstance(monitor, FakeKernel) else FakeComm())
        return 0

    def stop(self):
        return 0


class FakeKernel(FakeMonitor):
    pass


class FakeCommunication(FakeMonitor):
    pass


class CaptureTest(unittest.TestCase):
    def test_periodic_flush_persists_both_activity_types(self):
        FakeMonitor.instances.clear()
        module = types.ModuleType("mspti")
        module.KernelMonitor = FakeKernel
        module.CommunicationMonitor = FakeCommunication
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"mspti": module}):
            path = Path(directory) / "activity.jsonl"
            capture = MsptiCapture(path, interval_s=0.05)
            capture.start()
            time.sleep(0.13)
            result = capture.stop()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            kinds = {row["kind"] for row in rows}
            self.assertEqual(kinds, {"kernel", "communication", "flush_tick"})
            self.assertGreaterEqual(result["counts"]["flush_tick"], 2)
            self.assertEqual(result["dropped"], 0)
            self.assertEqual(result["flush_errors"], 0)
            self.assertEqual(next(row for row in rows if row["kind"] == "kernel")["duration_ns"], 150)
            self.assertTrue(path.with_suffix(".summary.json").exists())


if __name__ == "__main__":
    unittest.main()
