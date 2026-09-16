import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from flightrecorder.analyzer import analyze


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc")
class WatchdogIntegrationTest(unittest.TestCase):
    def test_live_stall_then_process_exit(self):
        library = os.environ.get("FLIGHT_TEST_LIBRARY")
        if not library:
            self.skipTest("set FLIGHT_TEST_LIBRARY to libflightrecorder.so")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); flight_dir = root / "flight"; dumps = root / "dumps"
            flight_dir.mkdir(); dumps.mkdir()
            child_code = """
import sys, time
from flightrecorder import Recorder
r=Recorder(sys.argv[1], rank=0, device=0, library=sys.argv[2], capacity=128)
r.record('REQUEST_BEGIN', correlation=42, arg0=131072)
r.record('SCHEDULER_BEGIN', correlation=7)
r.record('PP_RECV_BEGIN', stream=3, correlation=99, arg0=1, arg1=4096)
print(r.path, flush=True)
time.sleep(30)
"""
            child = subprocess.Popen([sys.executable, "-c", child_code, str(flight_dir), library],
                                     stdout=subprocess.PIPE, text=True)
            watchdog = None
            try:
                self.assertTrue(child.stdout.readline().strip())
                cmd = [sys.executable, "-m", "flightrecorder.watchdog", "--directory", str(flight_dir),
                       "--output", str(dumps), "--timeout", "0.5", "--interval", "0.1", "--pid", str(child.pid)]
                if os.environ.get("FLIGHT_TEST_NPU_SMI") == "1": cmd.append("--npu-smi")
                watchdog = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertEqual(watchdog.stdout.readline().strip(), "watchdog ready")
                deadline = time.time() + 8
                stalled = []
                while time.time() < deadline:
                    stalled = [p for p in dumps.glob("*-stalled-*") if not p.name.startswith(".")]
                    if stalled: break
                    time.sleep(0.1)
                self.assertTrue(stalled, "watchdog did not capture live stall")
                self.assertIsNone(child.poll(), "child should still be alive at stalled snapshot")
                meta = json.loads((stalled[0] / "metadata.json").read_text())
                self.assertEqual(meta["reason"], f"stalled-pid{child.pid}")
                self.assertTrue((stalled[0] / f"pid{child.pid}" / "status").exists())
                if os.environ.get("FLIGHT_TEST_NPU_SMI") == "1":
                    self.assertTrue((stalled[0] / "npu-smi.txt").exists())
                child.terminate(); child.wait(timeout=5)
                stdout, stderr = watchdog.communicate(timeout=8)
                self.assertEqual(watchdog.returncode, 0, stderr)
                self.assertTrue(list(dumps.glob("*-exit-*")), stdout)
                files = list(dumps.rglob("*.flight"))
                rank = analyze(files, {0})["ranks"][0]
                self.assertEqual(rank["last_event"]["type"], "PP_RECV_BEGIN")
                self.assertEqual(len(rank["outstanding_host_scopes"]), 3)
            finally:
                if child.poll() is None:
                    child.kill(); child.wait()
                if watchdog is not None and watchdog.poll() is None:
                    watchdog.kill(); watchdog.wait()
                if child.stdout:
                    child.stdout.close()
                if watchdog is not None:
                    if watchdog.stdout: watchdog.stdout.close()
                    if watchdog.stderr: watchdog.stderr.close()


if __name__ == "__main__":
    unittest.main()
