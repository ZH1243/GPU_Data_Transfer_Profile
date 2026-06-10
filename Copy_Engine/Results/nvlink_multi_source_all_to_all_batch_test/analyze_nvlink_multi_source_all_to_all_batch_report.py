#!/usr/bin/env python3
"""Analyze one multi-source all-to-all batch Nsight Systems report.

The benchmark submits multiple cudaMemcpyBatchAsync calls per iteration: one
batch call for each destination GPU, with one Memcpy PtoP entry per logical
source buffer. This analyzer reuses the all-to-all copy-engine SQLite queries
and adjusts warmup trimming for that per-destination batch structure.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ALL_TO_ALL_RESULTS_DIR = SCRIPT_DIR.parent / "nvlink_all_to_all_copy_engine_test"
sys.path.insert(0, str(ALL_TO_ALL_RESULTS_DIR))

import analyze_nvlink_all_to_all_copy_engine_report as base  # noqa: E402


DEFAULT_REPORT = SCRIPT_DIR / "total_128k_batch_3x8k+104k.nsys-rep"
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_TIMED_ITERATIONS = 100
DEFAULT_SOURCE_BUFFERS_PER_BATCH = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source-side Memcpy PtoP statistics and NVLink RX/TX metrics "
            "for nvlink_multi_source_all_to_all_batch_test.py reports."
        )
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=DEFAULT_REPORT,
        help="Input .nsys-rep or .sqlite file.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("auto", "separate", "batch"),
        default="auto",
        help="Copy mode to analyze. Default: infer from filename/runtime API names.",
    )
    parser.add_argument(
        "--gpu-bus-id",
        default=base.DEFAULT_GPU_BUS_ID,
        help=f"PCI bus ID of the collected/source GPU. Default: {base.DEFAULT_GPU_BUS_ID}",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="For .nsys-rep input, regenerate the sibling .sqlite with nsys export.",
    )
    parser.add_argument(
        "--nvlink-peak-gb-s",
        type=float,
        default=None,
        help=(
            "Optional peak unidirectional NVLink bandwidth in decimal GB/s. "
            "When provided, throughput percentages are also converted to GB/s."
        ),
    )
    parser.add_argument(
        "--skip-warmup-iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
        help=(
            "Ignore this many earliest benchmark iterations. For batch mode, "
            "one iteration contains --batch-calls-per-iteration CUDA batch calls. "
            f"Default: {DEFAULT_WARMUP_ITERATIONS}."
        ),
    )
    parser.add_argument(
        "--timed-iterations",
        type=int,
        default=DEFAULT_TIMED_ITERATIONS,
        help=(
            "Timed benchmark iterations. Used only when falling back to CUDA "
            f"runtime API timing. Default: {DEFAULT_TIMED_ITERATIONS}."
        ),
    )
    parser.add_argument(
        "--processes-per-report",
        type=int,
        default=None,
        help=(
            "Number of torchrun worker processes captured in the report. Default: "
            "infer from CUDA runtime API events."
        ),
    )
    parser.add_argument(
        "--batch-calls-per-iteration",
        type=int,
        default=None,
        help=(
            "Number of cudaMemcpyBatchAsync calls per iteration. Default: infer "
            "from the number of destination GPUs observed in Memcpy PtoP events."
        ),
    )
    parser.add_argument(
        "--ptop-events-per-batch",
        "--entries-per-batch",
        dest="ptop_events_per_batch",
        type=int,
        default=None,
        help=(
            "Memcpy PtoP activities per cudaMemcpyBatchAsync call. Default: "
            "infer from CUDA runtime correlation groups. The legacy spelling "
            "--entries-per-batch is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def parse_report_shape(path: Path) -> tuple[int, int] | None:
    match = re.search(
        r"total_(\d+(?:\.\d+)?[kKmMgG])_batch_(\d+)x8k\+\d+(?:\.\d+)?[kKmMgG]",
        path.name,
    )
    if match is not None:
        return size_bytes(match.group(1)), int(match.group(2)) + 1

    match = re.search(
        r"total_(\d+(?:\.\d+)?[kKmMgG])_batch_uniform_buffer_sizes",
        path.name,
    )
    if match is not None:
        return size_bytes(match.group(1)), DEFAULT_SOURCE_BUFFERS_PER_BATCH

    return None


def size_bytes(label: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", label.lower())
    if match is None:
        raise ValueError(f"Cannot parse size label: {label}")
    value = float(match.group(1))
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return int(value * multiplier)


def grouped_by_api(events: list[base.CopyEvent]) -> list[list[base.CopyEvent]]:
    grouped: dict[tuple[int | None, int | None], list[base.CopyEvent]] = defaultdict(list)
    for event in events:
        if event.correlation_id is not None:
            grouped[(event.global_pid, event.correlation_id)].append(event)
    return sorted(grouped.values(), key=lambda group: min(event.start_ns for event in group))


def infer_batch_calls_per_iteration(events: list[base.CopyEvent]) -> int:
    destinations = {event.dst_device_id for event in events if event.dst_device_id is not None}
    if destinations:
        return len(destinations)
    return 1


def infer_entries_per_batch(events: list[base.CopyEvent]) -> int:
    group_sizes = [len(group) for group in grouped_by_api(events)]
    if not group_sizes:
        return 1
    counts: dict[int, int] = defaultdict(int)
    for size in group_sizes:
        counts[size] += 1
    return max(counts, key=lambda size: (counts[size], size))


def infer_process_count(events: list[base.CopyEvent]) -> int:
    processes = {
        (event.global_pid, event.pid)
        for event in events
        if event.global_pid is not None or event.pid is not None
    }
    return max(1, len(processes))


def infer_batch_calls_from_runtime(
    events: list[base.CopyEvent],
    skip_warmup_iterations: int,
    timed_iterations: int,
    processes_per_report: int,
) -> int:
    total_iterations = skip_warmup_iterations + timed_iterations
    if total_iterations <= 0:
        raise ValueError("--timed-iterations plus --skip-warmup-iterations must be positive")
    denominator = max(1, processes_per_report * total_iterations)
    return max(1, round(len(events) / denominator))


def load_runtime_copy_events(
    conn: sqlite3.Connection,
    copy_mode: str,
    bytes_per_event: int,
) -> list[base.CopyEvent]:
    api_pattern = (
        "cudaMemcpyBatchAsync%"
        if copy_mode == "batch"
        else "cudaMemcpyPeerAsync%"
    )
    rows = conn.execute(
        """
        SELECT
            r.start,
            r.end,
            r.correlationId,
            p.globalPid,
            p.pid,
            p.name AS processName,
            s.value AS apiName
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
        JOIN StringIds AS s
            ON s.id = r.nameId
        LEFT JOIN PROCESSES AS p
            ON p.globalPid = r.globalTid - p.pid
        WHERE s.value LIKE ?
        ORDER BY r.start, r.end
        """,
        (api_pattern,),
    ).fetchall()

    events: list[base.CopyEvent] = []
    for row in rows:
        if row["end"] <= row["start"]:
            continue
        events.append(
            base.CopyEvent(
                start_ns=int(row["start"]),
                end_ns=int(row["end"]),
                bytes=bytes_per_event,
                dst_device_id=None,
                correlation_id=row["correlationId"],
                global_pid=row["globalPid"],
                pid=row["pid"],
                process_name=row["processName"],
                api_start_ns=int(row["start"]),
                api_end_ns=int(row["end"]),
                api_name=row["apiName"],
            )
        )
    return events


def summarize_gaps_by_process(events: list[base.CopyEvent]) -> dict[str, float | int | None]:
    grouped: dict[tuple[int | None, int | None], list[base.CopyEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.global_pid, event.pid)].append(event)

    gaps_ns: list[int] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda event: (event.start_ns, event.end_ns))
        gaps_ns.extend(
            ordered[index].start_ns - ordered[index - 1].end_ns
            for index in range(1, len(ordered))
        )

    return {
        "count": len(gaps_ns),
        "average_ns": base.average([float(value) for value in gaps_ns]),
        "average_us": (
            base.average([float(value) for value in gaps_ns]) / 1_000.0
            if gaps_ns
            else None
        ),
        "min_ns": min(gaps_ns) if gaps_ns else None,
        "max_ns": max(gaps_ns) if gaps_ns else None,
    }


def empty_wait_summary() -> dict[str, float | int | None]:
    return {
        "count": 0,
        "average_ns": None,
        "average_us": None,
        "min_ns": None,
        "max_ns": None,
    }


def summarize_api_groups_for_fallback(
    events: list[base.CopyEvent],
    entries_per_batch: int,
) -> dict[str, Any]:
    api_group_count = len(grouped_by_api(events))
    return {
        "api_group_count": api_group_count,
        "average_memcpy_ptop_per_api_group": float(entries_per_batch) if api_group_count else None,
        "min_memcpy_ptop_per_api_group": entries_per_batch if api_group_count else None,
        "max_memcpy_ptop_per_api_group": entries_per_batch if api_group_count else None,
    }


def trim_warmup_events(
    events: list[base.CopyEvent],
    copy_mode: str,
    skip_warmup_iterations: int,
    batch_calls_per_iteration: int,
    entries_per_batch: int,
    processes_per_report: int,
) -> tuple[list[base.CopyEvent], int, int, int]:
    if skip_warmup_iterations < 0:
        raise ValueError("--skip-warmup-iterations must be non-negative")
    if batch_calls_per_iteration <= 0:
        raise ValueError("--batch-calls-per-iteration must be positive")
    if entries_per_batch <= 0:
        raise ValueError("--entries-per-batch must be positive")
    if processes_per_report <= 0:
        raise ValueError("--processes-per-report must be positive")
    if not events or skip_warmup_iterations == 0:
        return events, 0, 0, 0

    if copy_mode == "batch":
        groups = grouped_by_api(events)
        groups_to_skip = min(
            skip_warmup_iterations * batch_calls_per_iteration * processes_per_report,
            len(groups),
        )
        skip_keys = {
            (group[0].global_pid, group[0].correlation_id)
            for group in groups[:groups_to_skip]
        }
        trimmed = [
            event
            for event in events
            if (event.global_pid, event.correlation_id) not in skip_keys
        ]
        skipped_iterations = groups_to_skip // (
            batch_calls_per_iteration * processes_per_report
        )
        return trimmed, len(events) - len(trimmed), skipped_iterations, groups_to_skip

    copies_per_iteration = batch_calls_per_iteration * entries_per_batch
    events_to_skip = min(
        skip_warmup_iterations * copies_per_iteration * processes_per_report,
        len(events),
    )
    skipped_iterations = events_to_skip // (copies_per_iteration * processes_per_report)
    return events[events_to_skip:], events_to_skip, skipped_iterations, 0


def build_summary(
    conn: sqlite3.Connection,
    sqlite_path: Path,
    bus_id: str,
    peak_gb_s: float | None,
    skip_warmup_iterations: int,
    copy_mode: str = "auto",
    batch_calls_per_iteration: int | None = None,
    entries_per_batch: int | None = None,
    timed_iterations: int = DEFAULT_TIMED_ITERATIONS,
    processes_per_report: int | None = None,
) -> dict[str, Any]:
    gpu = base.find_gpu(conn, bus_id)
    mode = base.detect_copy_mode(conn, sqlite_path, copy_mode)
    ptop_copy_kind = base.find_ptop_copy_kind(conn)
    all_events = base.load_source_copy_events(conn, int(gpu["cuDevice"]), ptop_copy_kind)
    event_source = "memcpy_ptop_source"
    report_shape = parse_report_shape(sqlite_path)

    if not all_events:
        if report_shape is None:
            raise RuntimeError(
                "No source-side Memcpy PtoP events were found, and the bytes per "
                "cudaMemcpyBatchAsync call cannot be inferred from the report name."
            )
        bytes_per_api_call, entries_from_name = report_shape
        all_events = load_runtime_copy_events(conn, mode, bytes_per_api_call)
        event_source = "cuda_runtime_api_fallback"
    else:
        entries_from_name = report_shape[1] if report_shape else None

    inferred_processes = infer_process_count(all_events)
    actual_processes = processes_per_report or inferred_processes
    if event_source == "cuda_runtime_api_fallback":
        inferred_batch_calls = infer_batch_calls_from_runtime(
            all_events,
            skip_warmup_iterations,
            timed_iterations,
            actual_processes,
        )
        inferred_entries = entries_from_name or 1
    else:
        inferred_batch_calls = infer_batch_calls_per_iteration(all_events)
        inferred_entries = infer_entries_per_batch(all_events)
    actual_batch_calls = batch_calls_per_iteration or inferred_batch_calls
    actual_entries = entries_per_batch or inferred_entries
    copies_per_iteration = actual_batch_calls * actual_entries

    events, skipped_events, skipped_iterations, skipped_api_groups = trim_warmup_events(
        all_events,
        mode,
        skip_warmup_iterations,
        actual_batch_calls,
        actual_entries,
        actual_processes,
    )

    throughputs_gb_s = [event.throughput_gb_s for event in events]
    throughputs_gib_s = [event.throughput_gib_s for event in events]
    first_start_ns = min((event.start_ns for event in events), default=None)
    last_end_ns = max((event.end_ns for event in events), default=None)
    paired_event_count = sum(1 for event in events if event.api_end_ns is not None)
    total_bytes = sum(event.bytes for event in events)
    api_groups = (
        summarize_api_groups_for_fallback(events, actual_entries)
        if event_source == "cuda_runtime_api_fallback"
        else base.summarize_api_groups(events)
    )
    gap_summary = (
        summarize_gaps_by_process(events)
        if event_source == "cuda_runtime_api_fallback"
        else base.summarize_gaps(events)
    )
    wait_summary = (
        empty_wait_summary()
        if event_source == "cuda_runtime_api_fallback"
        else base.summarize_waits(events)
    )

    process_counts: dict[str, int] = defaultdict(int)
    destination_counts: dict[str, int] = defaultdict(int)
    api_name_counts: dict[str, int] = defaultdict(int)
    for event in events:
        process = (
            f"{event.process_name or 'unknown'} pid={event.pid}"
            if event.pid is not None
            else "unknown"
        )
        process_counts[process] += 1
        destination_counts[str(event.dst_device_id)] += 1
        api_name_counts[event.api_name or "unpaired"] += 1

    return {
        "sqlite_path": str(sqlite_path),
        "copy_mode": mode,
        "event_source": event_source,
        "gpu": {
            "bus_id": gpu["busLocation"],
            "name": gpu["name"],
            "target_info_id": gpu["id"],
            "cuda_device_id": gpu["cuDevice"],
            "uuid": gpu["uuid"],
        },
        "copy_process_counts": dict(sorted(process_counts.items())),
        "destination_counts": dict(sorted(destination_counts.items())),
        "api_name_counts_by_memcpy_event": dict(sorted(api_name_counts.items())),
        "copy_window": {
            "first_memcpy_start_ns": first_start_ns,
            "last_memcpy_end_ns": last_end_ns,
            "duration_ns": (
                last_end_ns - first_start_ns
                if first_start_ns is not None and last_end_ns is not None
                else None
            ),
        },
        "counts": {
            "all_source_memcpy_ptop_count_before_warmup_skip": len(all_events),
            "inferred_batch_calls_per_iteration": inferred_batch_calls,
            "inferred_entries_per_batch": inferred_entries,
            "inferred_ptop_events_per_batch": inferred_entries,
            "inferred_processes_per_report": inferred_processes,
            "source_buffers_per_batch_from_filename": (
                report_shape[1] if report_shape is not None else None
            ),
            "batch_calls_per_iteration_used_for_warmup_skip": actual_batch_calls,
            "entries_per_batch_used_for_warmup_skip": actual_entries,
            "ptop_events_per_batch_used_for_warmup_skip": actual_entries,
            "processes_per_report_used_for_warmup_skip": actual_processes,
            "copies_per_iteration_used_for_warmup_skip": copies_per_iteration,
            "requested_skipped_warmup_iterations": skip_warmup_iterations,
            "skipped_warmup_iterations": skipped_iterations,
            "skipped_warmup_cuda_api_groups": skipped_api_groups,
            "skipped_warmup_source_memcpy_ptop_count": skipped_events,
            "source_memcpy_ptop_count": len(events),
            "paired_cuda_api_memcpy_ptop_count": paired_event_count,
            "unpaired_source_memcpy_ptop_count": len(events) - paired_event_count,
        },
        "api_groups": api_groups,
        "throughput": {
            "average_event_gb_s": base.average(throughputs_gb_s),
            "average_event_gib_s": base.average(throughputs_gib_s),
            "min_event_gib_s": min(throughputs_gib_s) if events else None,
            "max_event_gib_s": max(throughputs_gib_s) if events else None,
            "total_bytes": total_bytes,
        },
        "time_gap_between_consecutive_memcpy_ptop_source": gap_summary,
        "time_spent_in_memcpy_ptop_source": base.summarize_memcpy_durations(events),
        "waiting_time_memcpy_start_minus_api_end": wait_summary,
        "nvlink_user_data_metrics_over_copy_window": base.summarize_nvlink_metrics(
            conn, first_start_ns, last_end_ns, peak_gb_s, include_protocol=False
        ),
        "nvlink_user_plus_protocol_metrics_over_copy_window": base.summarize_nvlink_metrics(
            conn, first_start_ns, last_end_ns, peak_gb_s, include_protocol=True
        ),
    }


def print_text(summary: dict[str, Any]) -> None:
    gpu = summary["gpu"]
    counts = summary["counts"]
    api_groups = summary["api_groups"]
    throughput = summary["throughput"]
    gap = summary["time_gap_between_consecutive_memcpy_ptop_source"]
    memcpy_duration = summary["time_spent_in_memcpy_ptop_source"]
    wait = summary["waiting_time_memcpy_start_minus_api_end"]
    window = summary["copy_window"]
    nvlink_user = summary["nvlink_user_data_metrics_over_copy_window"]
    nvlink_all = summary["nvlink_user_plus_protocol_metrics_over_copy_window"]

    print(f"SQLite input: {summary['sqlite_path']}")
    print(f"Copy mode: {summary['copy_mode']}")
    print(f"Event source: {summary['event_source']}")
    print(
        "GPU: "
        f"{gpu['bus_id']} ({gpu['name']}, CUDA device {gpu['cuda_device_id']})"
    )
    print(f"Copy processes: {summary['copy_process_counts']}")
    print(f"Destination CUDA device counts: {summary['destination_counts']}")
    print(f"Runtime API names by Memcpy event: {summary['api_name_counts_by_memcpy_event']}")
    print(
        "Warmup skip: "
        f"{counts['skipped_warmup_iterations']} iterations, "
        f"{counts['skipped_warmup_cuda_api_groups']} CUDA API groups, "
        f"{counts['skipped_warmup_source_memcpy_ptop_count']} of "
        f"{counts['all_source_memcpy_ptop_count_before_warmup_skip']} "
        "source Memcpy PtoP events"
    )
    print(
        "Iteration shape: "
        f"{counts['batch_calls_per_iteration_used_for_warmup_skip']} batch calls/iter, "
        f"{counts['ptop_events_per_batch_used_for_warmup_skip']} PtoP events/batch, "
        f"{counts['source_buffers_per_batch_from_filename']} source buffers/batch, "
        f"{counts['processes_per_report_used_for_warmup_skip']} processes/report, "
        f"{counts['copies_per_iteration_used_for_warmup_skip']} copies/iter"
    )
    print()
    print("1. Counts")
    print(f"   Source Memcpy PtoP events: {counts['source_memcpy_ptop_count']}")
    print(f"   Paired CUDA API Memcpy PtoP events: {counts['paired_cuda_api_memcpy_ptop_count']}")
    print(f"   Unpaired source Memcpy PtoP events: {counts['unpaired_source_memcpy_ptop_count']}")
    print(
        "   API groups after warmup skip: "
        f"{api_groups['api_group_count']}, "
        f"avg Memcpy PtoP/API={base.fmt_float(api_groups['average_memcpy_ptop_per_api_group'])}"
    )
    print()
    print("2. Average Memcpy PtoP (source) throughput")
    print(
        "   "
        f"{base.fmt_float(throughput['average_event_gib_s'])} GiB/s "
        f"({base.fmt_float(throughput['average_event_gb_s'])} GB/s decimal)"
    )
    print()
    print("3. Average time gap between consecutive Memcpy PtoP (source)")
    print(f"   {base.fmt_float(gap['average_us'])} us over {gap['count']} gaps")
    print()
    print("4. Average time spent in each Memcpy PtoP (source)")
    print(
        "   "
        f"{base.fmt_float(memcpy_duration['average_us'])} us "
        f"over {memcpy_duration['count']} events"
    )
    print()
    print("5. Average waiting time: Memcpy PtoP start - CUDA API end")
    print(f"   {base.fmt_float(wait['average_us'])} us over {wait['count']} paired events")
    print()
    print("6. Average NVLink RX/TX metrics over the copy window")
    print(
        "   Window: "
        f"{base.fmt_ns_as_s_plus_ms(window['first_memcpy_start_ns'])} to "
        f"{base.fmt_ns_as_s_plus_ms(window['last_memcpy_end_ns'])} "
        f"({base.fmt_float(window['duration_ns'] / 1_000_000.0 if window['duration_ns'] else None)} ms)"
    )
    for label, data in (
        ("user data", nvlink_user),
        ("user + protocol data", nvlink_all),
    ):
        rx = data.get("rx", {})
        tx = data.get("tx", {})
        print(
            f"   {label}: "
            f"RX {base.fmt_float(rx.get('average_percent_of_peak'))}% of peak, "
            f"TX {base.fmt_float(tx.get('average_percent_of_peak'))}% of peak "
            f"(samples RX={rx.get('sample_count', 0)}, TX={tx.get('sample_count', 0)})"
        )
        if "average_gb_s" in rx or "average_gb_s" in tx:
            print(
                "      converted: "
                f"RX {base.fmt_float(rx.get('average_gb_s'))} GB/s, "
                f"TX {base.fmt_float(tx.get('average_gb_s'))} GB/s"
            )


def main() -> int:
    args = parse_args()
    try:
        sqlite_path = base.resolve_sqlite(args.report, args.force_export)
        with base.connect(sqlite_path) as conn:
            summary = build_summary(
                conn,
                sqlite_path=sqlite_path,
                bus_id=args.gpu_bus_id,
                peak_gb_s=args.nvlink_peak_gb_s,
                skip_warmup_iterations=args.skip_warmup_iterations,
                copy_mode=args.copy_mode,
                batch_calls_per_iteration=args.batch_calls_per_iteration,
                entries_per_batch=args.ptop_events_per_batch,
                timed_iterations=args.timed_iterations,
                processes_per_report=args.processes_per_report,
            )
    except (OSError, RuntimeError, sqlite3.Error, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
