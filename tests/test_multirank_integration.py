import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from flightrecorder.analyzer import analyze


class MultiRankIntegrationTest(unittest.TestCase):
    def test_cpp_writer_to_cross_rank_analysis(self):
        library = os.environ.get("FLIGHT_TEST_LIBRARY")
        if not library:
            self.skipTest("set FLIGHT_TEST_LIBRARY to the built recorder library")
        code = """
import sys
from flightrecorder import Recorder
rank=int(sys.argv[1]); r=Recorder(sys.argv[2], rank, rank, sys.argv[3], 128)
r.record('REQUEST_BEGIN', correlation=55, arg0=131072)
r.record('HCCL_BEGIN', stream=8, correlation=9001, arg0=77, arg1=(2 if rank == 1 else 1))
if rank == 3: r.record('PP_RECV_BEGIN', stream=9, correlation=8001, arg0=2, arg1=4096)
r.close()
"""
        with tempfile.TemporaryDirectory() as d:
            for rank in (0, 1, 3):
                subprocess.run([sys.executable, "-c", code, str(rank), d, library], check=True)
            result = analyze(sorted(Path(d).glob("*.flight")), {0, 1, 2, 3})
            collective = result["collectives"][0]
            self.assertEqual(collective["entered"], [0, 1, 3])
            self.assertEqual(collective["missing_expected"], [2])
            self.assertTrue(collective["type_mismatch"])
            self.assertEqual(result["ranks"][3]["last_event"]["type"], "PP_RECV_BEGIN")


if __name__ == "__main__":
    unittest.main()
