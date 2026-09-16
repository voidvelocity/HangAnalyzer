import tempfile
import unittest
from pathlib import Path
from flightrecorder.format import EVENT, HEADER, MAGIC, TYPES
from flightrecorder.analyzer import analyze, write_outputs


def flight(path, rank, rows, capacity=16):
    raw = bytearray(128 + capacity * 64)
    HEADER.pack_into(raw, 0, MAGIC, 1, 64, capacity, 1000 + rank, rank, rank,
                     len(rows), 1000000000 + len(rows), 0, 0)
    for seq, row in enumerate(rows, 1):
        name, stream, corr, a0, a1 = row[:5]
        flags = row[5] if len(row) > 5 else 0
        EVENT.pack_into(raw, 128 + ((seq - 1) % capacity) * 64,
                        1000000000 + seq, seq, 1000 + rank, 2000 + rank,
                        rank, rank, stream, TYPES[name], flags, corr, a0, a1)
    path.write_bytes(raw)
    return path


class AnalyzerTest(unittest.TestCase):
    def test_binary_layout_matches_cpp_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            p = flight(Path(d) / "r.flight", 3, [("CHECKPOINT", 9, 0x1122334455667788,
                                                   0x8877665544332211, 0x0102030405060708)])
            from flightrecorder.format import read_events
            _, events = read_events(p)
            self.assertEqual(events[0]["correlation_id"], 0x1122334455667788)
            self.assertEqual(events[0]["arg0"], 0x8877665544332211)
            self.assertEqual(events[0]["arg1"], 0x0102030405060708)

    def test_missing_rank_and_unfinished_collective(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = [flight(root / f"r{r}.flight", r, [("HCCL_BEGIN", 4, 100, 9, 2)])
                     for r in (0, 1, 3)]
            result = analyze(files, {0, 1, 2, 3})
            self.assertEqual(result["collectives"][0]["missing_expected"], [2])
            self.assertEqual(result["collectives"][0]["end_observed"], [])
            write_outputs(result, root / "report", files)
            self.assertIn("no HCCL_BEGIN observed for ranks [2]",
                          (root / "report" / "report.txt").read_text())
            self.assertIn("HCCL_BEGIN", (root / "report" / "events.txt").read_text())

    def test_collective_type_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = [flight(root / "a.flight", 0, [("HCCL_BEGIN", 1, 100, 9, 1)]),
                     flight(root / "b.flight", 1, [("HCCL_BEGIN", 1, 100, 9, 2)])]
            self.assertTrue(analyze(files)["collectives"][0]["type_mismatch"])

    def test_filter_communicator_before_expected_rank_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = [flight(root / "r0.flight", 0, [("HCCL_BEGIN", 0, 1, 10, 1)]),
                     flight(root / "r1.flight", 1, [("HCCL_BEGIN", 0, 1, 10, 1)]),
                     flight(root / "r2.flight", 2, [("HCCL_BEGIN", 0, 1, 20, 1)])]
            result = analyze(files, {0, 1}, communicator_id=10)
            self.assertEqual(len(result["collectives"]), 1)
            self.assertEqual(result["collectives"][0]["missing_expected"], [])

    def test_event_dependency_cycle_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            p = flight(Path(d) / "r.flight", 0,
                       [("EVENT_WAIT", 1, 22, 0, 0), ("EVENT_RECORD", 1, 11, 0, 0),
                        ("EVENT_WAIT", 2, 11, 0, 0), ("EVENT_RECORD", 2, 22, 0, 0)])
            self.assertTrue(analyze([p])["stream_cycles_candidate"])

    def test_normal_bidirectional_handoff_is_not_a_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            p = flight(Path(d) / "r.flight", 0,
                       [("EVENT_RECORD", 1, 11, 0, 0), ("EVENT_WAIT", 2, 11, 0, 0),
                        ("EVENT_RECORD", 2, 22, 0, 0), ("EVENT_WAIT", 1, 22, 0, 0)])
            self.assertEqual(analyze([p])["stream_cycles_candidate"], [])

    def test_host_open_scope_without_device_completion(self):
        with tempfile.TemporaryDirectory() as d:
            p = flight(Path(d) / "r.flight", 0,
                       [("REQUEST_BEGIN", 0, 42, 10000, 0),
                        ("SCHEDULER_BEGIN", 0, 7, 0, 0),
                        ("PP_RECV_BEGIN", 8, 11, 1, 4096)])
            r = analyze([p])["ranks"][0]
            self.assertEqual(r["last_event"]["type"], "PP_RECV_BEGIN")
            self.assertEqual(r["last_host_event"]["type"], "PP_RECV_BEGIN")
            self.assertIsNone(r["last_device_confirmation"])
            self.assertEqual(len(r["outstanding_host_scopes"]), 3)

    def test_checkpoint_generation_and_unconfirmed_interval(self):
        with tempfile.TemporaryDirectory() as d:
            p = flight(Path(d) / "r.flight", 0, [
                ("CHECKPOINT", 12, 700, 100, 1, 1),
                ("DEVICE_CONFIRMED", 12, 700, 100, 1),
                ("CHECKPOINT", 12, 701, 120, 2, 1),
                ("CHECKPOINT", 12, 701, 120, 2, 8),
                ("DEVICE_CONFIRMED", 12, 701, 120, 1),  # stale generation
                ("CHECKPOINT", 12, 702, 140, 0, 2),    # pool full warning
            ])
            rank = analyze([p])["ranks"][0]
            progress = rank["device_checkpoint_progress"][12]
            self.assertEqual(progress["confirmed_through"]["arg0"], 100)
            self.assertEqual(progress["first_unconfirmed"]["arg0"], 120)
            self.assertEqual(progress["unconfirmed_count"], 1)
            self.assertEqual(progress["not_ready_observed"], 1)
            self.assertEqual(progress["submitted_seq"], 120)
            self.assertEqual(progress["completed_seq"], 100)
            self.assertEqual([m["status"] for m in progress["checkpoint_mappings"]],
                             ["confirmed_exact", "unconfirmed"])
            self.assertEqual(len(rank["stale_or_unmatched_confirmations"]), 1)
            self.assertEqual(len(rank["checkpoint_warnings"]), 1)
            self.assertEqual(rank["last_event"]["flags"], 2)
            self.assertEqual(rank["last_host_event"]["correlation_id"], 701)

    def test_open_device_sync_after_confirmed_stream_is_coverage_warning(self):
        with tempfile.TemporaryDirectory() as d:
            p = flight(Path(d) / "r.flight", 0, [
                ("CHECKPOINT", 12, 700, 100, 1, 1),
                ("DEVICE_CONFIRMED", 12, 700, 100, 1),
                ("DEVICE_SYNC_BEGIN", 0, 2, 0, 0),
            ])
            rank = analyze([p])["ranks"][0]
            self.assertTrue(rank["checkpoint_coverage_warning"])
            self.assertEqual(rank["last_host_event"]["type"], "DEVICE_SYNC_BEGIN")


if __name__ == "__main__":
    unittest.main()
