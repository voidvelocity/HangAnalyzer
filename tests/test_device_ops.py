import tempfile
import unittest
from pathlib import Path
from flightrecorder.device_ops import summarize


class DeviceOpsTest(unittest.TestCase):
    def test_human_readable_kernel_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            csv_path = root / "rank7" / "sample_ascend_pt" / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
            csv_path.parent.mkdir(parents=True)
            csv_path.write_text("Device_id,Name,Start Time(us),Duration(us)\n"
                                "7,aclnnMatmul_MatMulCommon_MatMulV2,100.5,10.12\n")
            output = root / "report" / "device_operators.txt"
            self.assertEqual(summarize(root, output), {7: 1})
            self.assertIn("aclnnMatmul_MatMulCommon_MatMulV2", output.read_text())


if __name__ == "__main__":
    unittest.main()
