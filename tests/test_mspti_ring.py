import struct
import tempfile
import unittest
from pathlib import Path

from flightrecorder.mspti_ring import EVENT, EVENT_V1, MAGIC, read_ring, report


class RingTests(unittest.TestCase):
    def test_wrap_and_incomplete_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.msflight"
            header = struct.pack("<QIIQIIQQQQ", MAGIC, 2, EVENT.size, 3, 42, 0,
                                 5, 100, 200, 0).ljust(128, b"\0")
            data = bytearray(header + b"\0" * (3 * EVENT.size))

            def put(seq, kind, name, *, published=True):
                slot = (seq - 1) % 3
                EVENT.pack_into(data, 128 + slot * EVENT.size,
                                seq if published else 0, 200 + seq, 0, 0, seq,
                                42, 7, 0xFFFFFFFF, 0xFFFFFFFF, kind, 0, 10,
                                name.encode().ljust(64, b"\0"), b"AI_CORE".ljust(64, b"\0"))

            put(3, 1, "aclrtLaunch")
            put(4, 2, "aclrtLaunch")
            put(5, 5, "MatMul", published=False)
            path.write_bytes(data)
            hdr, events = read_ring(path)
            self.assertEqual([e["seq"] for e in events], [3, 4])
            self.assertEqual(hdr["overwritten"], 2)
            self.assertIn("callback entries without exits in retained window: 0", report(path))

    def test_reads_previous_ring_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.msflight"
            header = struct.pack("<QIIQIIQQQQ", MAGIC, 1, EVENT_V1.size, 1, 42, 0,
                                 1, 100, 200, 0).ljust(128, b"\0")
            event = EVENT_V1.pack(1, 201, 0, 0, 1, 42, 7, 1, 2, 5, 0, 0,
                                  b"MatMul".ljust(64, b"\0"))
            path.write_bytes(header + event)
            _, events = read_ring(path)
            self.assertEqual(events[0]["name"], "MatMul")
            self.assertEqual(events[0]["detail"], "")


if __name__ == "__main__":
    unittest.main()
