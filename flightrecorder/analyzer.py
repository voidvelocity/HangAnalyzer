"""Offline evidence report; accepts one snapshot or a merged two-node directory."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from .format import read_events

CHECKPOINT_SUBMITTED = 1
CHECKPOINT_POOL_FULL = 2
CHECKPOINT_QUERY_ERROR = 4
CHECKPOINT_NOT_READY = 8

PAIRS = {"REQUEST_BEGIN": "REQUEST_END", "SCHEDULER_BEGIN": "SCHEDULER_END",
         "MODEL_BEGIN": "MODEL_END", "GRAPH_BEGIN": "GRAPH_END", "KV_BEGIN": "KV_END",
         "PP_SEND_BEGIN": "PP_SEND_END", "PP_RECV_BEGIN": "PP_RECV_END",
         "HCCL_BEGIN": "HCCL_END", "STREAM_SYNC_BEGIN": "STREAM_SYNC_END",
         "DEVICE_SYNC_BEGIN": "DEVICE_SYNC_END"}


def find_cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    cycles, visiting, visited, stack = [], set(), set(), []
    def visit(node: str) -> None:
        visiting.add(node); stack.append(node)
        for dest in edges.get(node, ()):
            if dest in visiting:
                cycle = stack[stack.index(dest):] + [dest]
                if cycle not in cycles:
                    cycles.append(cycle)
            elif dest not in visited:
                visit(dest)
        stack.pop(); visiting.remove(node); visited.add(node)
    for n in list(edges):
        if n not in visited:
            visit(n)
    return cycles


def analyze(paths: list[Path], expected_ranks: set[int] | None = None,
            communicator_id: int | None = None) -> dict:
    ranks: dict[int, dict] = {}
    errors = []
    for path in paths:
        try:
            h, ev = read_events(path)
            # One latest file per rank; users should merge snapshots from the same instant.
            old = ranks.get(h["rank"])
            if old is None or h["write_seq"] > old["header"]["write_seq"]:
                ranks[h["rank"]] = {"header": h, "events": ev}
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    expected = expected_ranks if expected_ranks is not None else set(ranks)
    groups: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)
    report_ranks = {}
    event_producers: dict[tuple[int, int], dict] = {}
    waits = []
    edges: dict[str, set[str]] = defaultdict(set)
    for rank, data in sorted(ranks.items()):
        ev = data["events"]
        pending: dict[tuple[str, int, int], dict] = {}
        confirmed = []
        checkpoint_submitted = []
        checkpoint_warnings = []
        not_ready_observations = []
        for e in ev:
            typ, corr = e["type"], e["correlation_id"]
            if typ in PAIRS:
                pending[(typ, corr, e["stream"])] = e
            elif typ.endswith("_END"):
                begin = typ.replace("_END", "_BEGIN")
                pending.pop((begin, corr, e["stream"]), None)
            if typ == "HCCL_BEGIN":
                # arg0 is explicit communicator id, correlation_id is explicit group-local operation id.
                groups[(e["arg0"], corr)][rank] = e
            if typ == "EVENT_RECORD":
                event_producers[(rank, corr)] = e
            elif typ == "EVENT_WAIT":
                waits.append(e)
            elif typ == "DEVICE_CONFIRMED":
                confirmed.append(e)
            if typ == "CHECKPOINT":
                if e["flags"] & CHECKPOINT_SUBMITTED:
                    checkpoint_submitted.append(e)
                if e["flags"] & (CHECKPOINT_POOL_FULL | CHECKPOINT_QUERY_ERROR):
                    checkpoint_warnings.append(e)
                if e["flags"] & CHECKPOINT_NOT_READY:
                    not_ready_observations.append(e)
        exact_confirmations = {(e["stream"], e["correlation_id"], e["arg0"], e["arg1"]): e
                               for e in confirmed if e["arg1"] != 0}
        submitted_keys = {(e["stream"], e["correlation_id"], e["arg0"], e["arg1"])
                          for e in checkpoint_submitted}
        progress = {}
        by_stream = defaultdict(list)
        for checkpoint in checkpoint_submitted:
            by_stream[checkpoint["stream"]].append(checkpoint)
        for stream_id, checkpoints in by_stream.items():
            checkpoints.sort(key=lambda e: e["seq"])
            confirmed_indices = [i for i, e in enumerate(checkpoints) if
                (e["stream"], e["correlation_id"], e["arg0"], e["arg1"]) in exact_confirmations]
            last_index = max(confirmed_indices) if confirmed_indices else -1
            after = checkpoints[last_index + 1:]
            mappings = []
            for i, checkpoint in enumerate(checkpoints):
                key = (checkpoint["stream"], checkpoint["correlation_id"],
                       checkpoint["arg0"], checkpoint["arg1"])
                confirmation = exact_confirmations.get(key)
                status = ("confirmed_exact" if confirmation else
                          "confirmed_by_later_checkpoint" if i <= last_index else "unconfirmed")
                mappings.append({
                    "submitted_seq": checkpoint["arg0"],
                    "checkpoint_id": checkpoint["correlation_id"],
                    "generation": checkpoint["arg1"],
                    "status": status,
                    "confirmation_record_seq": confirmation["seq"] if confirmation else None,
                })
            progress[stream_id] = {
                "submitted_count": len(checkpoints),
                "explicitly_confirmed_count": len(confirmed_indices),
                "submitted_seq": checkpoints[-1]["arg0"],
                "completed_seq": checkpoints[last_index]["arg0"] if last_index >= 0 else 0,
                "confirmed_through": checkpoints[last_index] if last_index >= 0 else None,
                "first_unconfirmed": after[0] if after else None,
                "last_unconfirmed": after[-1] if after else None,
                "unconfirmed_count": len(after),
                "not_ready_observed": sum(1 for e in after if any(
                    n["stream"] == e["stream"] and n["correlation_id"] == e["correlation_id"] and
                    n["arg0"] == e["arg0"] and n["arg1"] == e["arg1"]
                    for n in not_ready_observations)),
                "checkpoint_mappings": mappings,
            }
        stale_confirmations = [e for e in confirmed if e["arg1"] != 0 and
            (e["stream"], e["correlation_id"], e["arg0"], e["arg1"]) not in submitted_keys]
        # Poller observations can be newer than the blocked business thread.  Keep
        # both views: last_event is the raw recorder tail, while last_host_event
        # excludes events emitted by checkpoint status polling.
        host_events = [e for e in ev if not (
            e["type"] == "DEVICE_CONFIRMED" or
            (e["type"] == "CHECKPOINT" and not (e["flags"] & CHECKPOINT_SUBMITTED))
        )]
        open_device_sync = any(e["type"] == "DEVICE_SYNC_BEGIN" for e in pending.values())
        checkpoint_coverage_warning = bool(
            open_device_sync and progress and
            all(p["unconfirmed_count"] == 0 for p in progress.values()))
        report_ranks[rank] = {
            "pid": data["header"]["pid"], "last_sequence": ev[-1]["seq"] if ev else 0,
            "last_event": ev[-1] if ev else None,
            "last_host_event": host_events[-1] if host_events else None,
            "last_device_confirmation": confirmed[-1] if confirmed else None,
            "outstanding_host_scopes": list(pending.values()), "last_events": ev[-20:],
            "ring_wrap_or_loss": data["header"]["write_seq"] > len(ev),
            "device_checkpoint_progress": progress,
            "checkpoint_warnings": checkpoint_warnings,
            "stale_or_unmatched_confirmations": stale_confirmations,
            "checkpoint_coverage_warning": checkpoint_coverage_warning,
        }
    dependencies = []
    # A stream can legitimately wait on B, then B wait on A after the first record.
    # Build operation-level edges: a record depends on earlier waits in its own stream;
    # a wait depends on the matching record. Collapsing all operations to a stream node
    # would incorrectly report this normal A -> B -> A handoff as a deadlock.
    for rank, data in ranks.items():
        prior_waits: dict[int, list[dict]] = defaultdict(list)
        for e in data["events"]:
            if e["type"] == "EVENT_WAIT":
                prior_waits[e["stream"]].append(e)
            elif e["type"] == "EVENT_RECORD":
                record_node = f"r{rank}:s{e['stream']}:record:{e['seq']}"
                for previous in prior_waits[e["stream"]]:
                    wait_node = f"r{rank}:s{previous['stream']}:wait:{previous['seq']}"
                    edges[record_node].add(wait_node)
    for wait in waits:
        rank, event_id = wait["rank"], wait["correlation_id"]
        producer = event_producers.get((rank, event_id))
        item = {"rank": rank, "event_id": event_id, "waiting_stream": wait["stream"],
                "producer_stream": producer["stream"] if producer else None,
                "producer_observed": producer is not None}
        dependencies.append(item)
        if producer:
            a = f"r{rank}:s{wait['stream']}:wait:{wait['seq']}"
            b = f"r{rank}:s{producer['stream']}:record:{producer['seq']}"
            edges[a].add(b)
    collective = []
    for (comm, seq), entered in sorted(groups.items()):
        if communicator_id is not None and comm != communicator_id:
            continue
        # Only compare ranks in a known communicator. If not supplied, world ranks are a hypothesis.
        missing = sorted(expected - entered.keys()) if expected_ranks is not None else []
        collective.append({"communicator_id": comm, "operation_id": seq,
                           "entered": sorted(entered), "missing_expected": missing,
                           "types": sorted({e["arg1"] for e in entered.values()}),
                           "type_mismatch": len({e["arg1"] for e in entered.values()}) > 1,
                           "end_observed": sorted(r for r, d in ranks.items() if any(
                               e["type"] == "HCCL_END" and e["arg0"] == comm and
                               e["correlation_id"] == seq for e in d["events"]))})
    return {"ranks": report_ranks, "collectives": collective, "dependencies": dependencies,
            "stream_cycles_candidate": find_cycles(edges), "errors": errors,
            "limitations": ["All events are host observations unless DEVICE_CONFIRMED is explicitly emitted after a verified device completion.",
                            "A stream dependency cycle is a candidate only; EventRecord placement and device completion require validation.",
                            "A missing collective requires explicit expected ranks and a shared communicator/operation id."]}


def write_outputs(report: dict, output: Path, paths: list[Path] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    lines = ["Ascend Hang Flight Report", "", "HOST PROGRESS"]
    for rank, d in sorted(report["ranks"].items()):
        last = d["last_host_event"]
        lines.append(f"rank {rank}: seq={d['last_sequence']} pid={d['pid']} last_host={last['type'] if last else 'none'} "
                     f"corr={last['correlation_id'] if last else '-'}")
        for e in d["outstanding_host_scopes"]:
            lines.append(f"  open {e['type']} corr={e['correlation_id']} stream={e['stream']} seq={e['seq']}")
        confirmed = d["last_device_confirmation"]
        lines.append(f"  last device confirmation: seq={confirmed['seq']} corr={confirmed['correlation_id']}"
                     if confirmed else "  last device confirmation: none observed")
    lines += ["", "COLLECTIVES (end means Host API returned, not NPU completion)"]
    for c in report["collectives"]:
        lines.append(f"comm={c['communicator_id']} op={c['operation_id']} entered={c['entered']} "
                     f"end={c['end_observed']} missing_expected={c['missing_expected']} "
                     f"types={c['types']} type_mismatch={c['type_mismatch']}")
    lines += ["", "DEVICE CHECKPOINT PROGRESS"]
    for rank, d in sorted(report["ranks"].items()):
        if not d["device_checkpoint_progress"]:
            lines.append(f"rank {rank}: no per-stream device checkpoints")
        for stream_id, p in sorted(d["device_checkpoint_progress"].items()):
            done, first, last = p["confirmed_through"], p["first_unconfirmed"], p["last_unconfirmed"]
            lines.append(f"rank {rank} stream {stream_id}: submitted={p['submitted_count']} "
                         f"explicit_confirmations={p['explicitly_confirmed_count']} "
                         f"unconfirmed={p['unconfirmed_count']}")
            lines.append(f"  submitted_seq={p['submitted_seq']} completed_seq={p['completed_seq']}")
            if done:
                lines.append(f"  confirmed through submitted_seq={done['arg0']} "
                             f"checkpoint={done['correlation_id']} generation={done['arg1']}")
            else:
                lines.append("  confirmed through: none observed")
            if first:
                lines.append(f"  first unconfirmed submitted_seq={first['arg0']} "
                             f"checkpoint={first['correlation_id']} generation={first['arg1']}")
                lines.append(f"  unconfirmed submitted_seq interval={first['arg0']}..{last['arg0']}")
                lines.append(f"  explicit NOT_READY observations={p['not_ready_observed']}")
        if d["checkpoint_warnings"]:
            lines.append(f"rank {rank}: checkpoint warnings={len(d['checkpoint_warnings'])}")
        if d["stale_or_unmatched_confirmations"]:
            lines.append(f"rank {rank}: ignored stale/unmatched confirmations="
                         f"{len(d['stale_or_unmatched_confirmations'])}")
        if d["checkpoint_coverage_warning"]:
            lines.append(f"rank {rank}: COVERAGE WARNING: DEVICE_SYNC remains open although all "
                         "observed Stream checkpoints are confirmed; blocked work is after the last "
                         "checkpoint or on an uninstrumented/internal Stream")
    lines += ["", "EVENT DEPENDENCIES"]
    for d in report["dependencies"]:
        lines.append(f"rank {d['rank']} stream {d['waiting_stream']} waits event {d['event_id']} "
                     f"producer_stream={d['producer_stream']}")
    for cycle in report["stream_cycles_candidate"]:
        lines.append("candidate cycle: " + " -> ".join(cycle))
    lines += ["", "BLOCKING EVIDENCE"]
    for c in report["collectives"]:
        if not c["missing_expected"]:
            continue
        lines.append(f"comm={c['communicator_id']} op={c['operation_id']}: "
                     f"no HCCL_BEGIN observed for ranks {c['missing_expected']}")
        for rank in c["missing_expected"]:
            d = report["ranks"].get(rank)
            last = d["last_host_event"] if d else None
            if last:
                lines.append(f"  rank {rank} last host event: {last['type']} corr={last['correlation_id']} "
                             f"stream={last['stream']} seq={last['seq']}")
        for rank in c["entered"]:
            d = report["ranks"].get(rank)
            last = d["last_host_event"] if d else None
            if last:
                lines.append(f"  rank {rank} last host event: {last['type']} corr={last['correlation_id']} "
                             f"stream={last['stream']} seq={last['seq']}")
        lines.append("  HCCL_END, if present, only proves Host API return; check DEVICE_SYNC_END/DEVICE_CONFIRMED.")
    lines += ["", "LIMITATIONS", *report["limitations"]]
    (output / "report.txt").write_text("\n".join(lines) + "\n")
    trace = []
    full: dict[int, list[dict]] = {}
    if paths:
        for path in paths:
            try:
                h, events = read_events(path)
                if h["rank"] in report["ranks"] and h["pid"] == report["ranks"][h["rank"]]["pid"]:
                    old = {e["seq"]: e for e in full.get(h["rank"], [])}
                    old.update({e["seq"]: e for e in events})
                    full[h["rank"]] = sorted(old.values(), key=lambda e: e["seq"])
            except (OSError, ValueError):
                continue
    for rank, d in report["ranks"].items():
        for e in full.get(rank, d["last_events"]):
            trace.append({"name": e["type"], "ph": "i", "s": "t", "ts": e["timestamp_ns"] / 1000,
                          "pid": rank, "tid": e["tid"], "args": {"stream": e["stream"],
                          "correlation_id": e["correlation_id"], "seq": e["seq"]}})
    (output / "trace.json").write_text(json.dumps({"traceEvents": trace}, indent=2))
    event_lines = ["Flight events by rank (time is milliseconds since that rank's first retained event)",
                   "seq       +ms         stream       type                  corr          arg0          arg1"]
    for rank, d in sorted(report["ranks"].items()):
        events = full.get(rank, d["last_events"])
        event_lines += ["", f"RANK {rank} pid={d['pid']} events={len(events)}"]
        first = events[0]["timestamp_ns"] if events else 0
        for e in events:
            event_lines.append(f"{e['seq']:<9} {(e['timestamp_ns']-first)/1e6:>10.3f} "
                               f"{e['stream']:>12} {e['type']:<21} "
                               f"{e['correlation_id']:>12} {e['arg0']:>12} {e['arg1']:>12}")
    (output / "events.txt").write_text("\n".join(event_lines) + "\n")
    if paths:
        snapshots = []
        for parent in sorted({p.parent for p in paths}):
            meta = parent / "metadata.json"
            if meta.exists():
                try:
                    item = json.loads(meta.read_text())
                    snapshots.append(f"{item.get('captured_utc', '?')} {item.get('reason', '?')} {parent}")
                except (OSError, ValueError):
                    continue
        (output / "snapshots.txt").write_text("\n".join(snapshots) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("--output", type=Path, default=Path("report"))
    p.add_argument("--expected-ranks", help="comma-separated global ranks in the communicator")
    p.add_argument("--communicator-id", type=int,
                   help="restrict collective matching to this communicator; use with --expected-ranks")
    args = p.parse_args()
    expected = set(map(int, args.expected_ranks.split(","))) if args.expected_ranks else None
    files = sorted(args.snapshot.rglob("*.flight"))
    if not files:
        p.error("no .flight files found")
    if expected is not None and args.communicator_id is None:
        p.error("--expected-ranks requires --communicator-id to avoid false missing-rank reports across groups")
    report = analyze(files, expected, args.communicator_id)
    write_outputs(report, args.output, files)
    print(args.output / "report.txt")


if __name__ == "__main__":
    main()
