#!/usr/bin/env python3
"""Analyze source-side NVLink all-to-all peer-copy activity from Nsight Systems.

The script accepts either an .nsys-rep file or an exported .sqlite file. It
handles both all-to-all modes produced by nvlink_all_to_all_copy_engine_test.py:

* separate: one cudaMemcpyPeerAsync per destination per iteration
* batch: one cudaMemcpyBatchAsync per iteration, mapping to multiple Memcpy PtoP
  source activities
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_REPORT = Path(__file__).with_name("a2a_separate_1m.nsys-rep")
DEFAULT_GPU_BUS_ID = "0000:03:00.0"
DEFAULT_SKIP_WARMUP_ITERATIONS = 10
BYTES_PER_GIB = 1024**3


@dataclass(frozen=True)
class CopyEvent:
    start_ns: int
    end_ns: int
    bytes: int
    dst_device_id: int | None
    correlation_id: int | None
    global_pid: int | None
    pid: int | None
    process_name: str | None
    api_start_ns: int | None
    api_end_ns: int | None
    api_name: str | None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def throughput_gb_s(self) -> float:
        # 1 byte/ns is exactly 1 decimal GB/s.
        return self.bytes / self.duration_ns

    @property
    def throughput_gib_s(self) -> float:
        return self.bytes * 1_000_000_000.0 / self.duration_ns / BYTES_PER_GIB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cudaMemcpyPeerAsync/cudaMemcpyBatchAsync to Memcpy PtoP "
            "source statistics and NVLink RX/TX metrics for one collected GPU."
        )
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=DEFAULT_REPORT,
        help="Input .nsys-rep or .sqlite file. Defaults to ./a2a_separate_1m.nsys-rep.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("auto", "separate", "batch"),
        default="auto",
        help="Copy mode to analyze. Default: infer from filename/runtime API names.",
    )
    parser.add_argument(
        "--gpu-bus-id",
        default=DEFAULT_GPU_BUS_ID,
        help=f"PCI bus ID of the collected/source GPU. Default: {DEFAULT_GPU_BUS_ID}",
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
        default=DEFAULT_SKIP_WARMUP_ITERATIONS,
        help=(
            "Ignore this many earliest all-to-all iterations before computing "
            f"statistics. Default: {DEFAULT_SKIP_WARMUP_ITERATIONS}."
        ),
    )
    parser.add_argument(
        "--copies-per-iteration",
        type=int,
        default=None,
        help=(
            "Number of source-side destination copies per all-to-all iteration. "
            "Default: infer from destination device count."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def resolve_sqlite(report: Path, force_export: bool) -> Path:
    report = report.expanduser().resolve()
    if report.suffix == ".sqlite":
        if not report.exists():
            raise FileNotFoundError(report)
        return report

    if report.suffix != ".nsys-rep":
        raise ValueError(f"Unsupported input suffix: {report.suffix}")

    sqlite_path = report.with_suffix(".sqlite")
    if sqlite_path.exists() and not force_export:
        return sqlite_path

    nsys = shutil.which("nsys")
    if nsys is None:
        if sqlite_path.exists():
            return sqlite_path
        raise RuntimeError(
            f"{sqlite_path} does not exist and `nsys` was not found to export {report}"
        )

    cmd = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(report),
    ]
    subprocess.run(cmd, check=True)
    return sqlite_path


def connect(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return row[0]


def find_ptop_copy_kind(conn: sqlite3.Connection) -> int:
    copy_kind = scalar(
        conn,
        """
        SELECT id
        FROM ENUM_CUDA_MEMCPY_OPER
        WHERE name = 'CUDA_MEMCPY_KIND_PTOP' OR label = 'Peer-to-Peer'
        LIMIT 1
        """,
    )
    if copy_kind is None:
        return 10
    return int(copy_kind)


def find_gpu(conn: sqlite3.Connection, bus_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT id, cuDevice, name, busLocation, uuid
        FROM TARGET_INFO_GPU
        WHERE lower(busLocation) = lower(?)
        """,
        (bus_id,),
    ).fetchone()
    if row is None:
        known = [
            dict(r)
            for r in conn.execute(
                "SELECT cuDevice, name, busLocation FROM TARGET_INFO_GPU ORDER BY cuDevice"
            )
        ]
        raise RuntimeError(f"GPU bus ID {bus_id!r} not found. Known GPUs: {known}")
    return row


def runtime_api_count(conn: sqlite3.Connection, pattern: str) -> int:
    return int(
        scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s
                ON s.id = r.nameId
            WHERE s.value LIKE ?
            """,
            (pattern,),
        )
        or 0
    )


def detect_copy_mode(conn: sqlite3.Connection, sqlite_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested

    lower_name = sqlite_path.name.lower()
    if "_batch_" in lower_name:
        return "batch"
    if "_separate_" in lower_name:
        return "separate"

    batch_count = runtime_api_count(conn, "cudaMemcpyBatchAsync%")
    peer_count = runtime_api_count(conn, "cudaMemcpyPeerAsync%")
    if batch_count and not peer_count:
        return "batch"
    if peer_count and not batch_count:
        return "separate"
    if batch_count:
        return "batch"
    return "separate"


def load_source_copy_events(
    conn: sqlite3.Connection, source_device_id: int, ptop_copy_kind: int
) -> list[CopyEvent]:
    rows = conn.execute(
        """
        WITH api AS (
            SELECT
                r.start,
                r.end,
                r.correlationId,
                p.globalPid,
                s.value AS apiName
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
            JOIN StringIds AS s
                ON s.id = r.nameId
            JOIN PROCESSES AS p
                ON p.globalPid = r.globalTid - p.pid
            WHERE s.value LIKE 'cudaMemcpyPeerAsync%'
                OR s.value LIKE 'cudaMemcpyBatchAsync%'
        )
        SELECT
            m.start AS memcpyStart,
            m.end AS memcpyEnd,
            m.bytes,
            m.dstDeviceId,
            m.correlationId,
            m.globalPid,
            p.pid,
            p.name AS processName,
            api.start AS apiStart,
            api.end AS apiEnd,
            api.apiName
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
        LEFT JOIN api
            ON api.correlationId = m.correlationId
            AND api.globalPid = m.globalPid
        LEFT JOIN PROCESSES AS p
            ON p.globalPid = m.globalPid
        WHERE m.copyKind = ?
            AND m.srcDeviceId = ?
        ORDER BY m.start, m.end
        """,
        (ptop_copy_kind, source_device_id),
    ).fetchall()

    events: list[CopyEvent] = []
    for row in rows:
        if row["memcpyEnd"] <= row["memcpyStart"]:
            continue
        events.append(
            CopyEvent(
                start_ns=int(row["memcpyStart"]),
                end_ns=int(row["memcpyEnd"]),
                bytes=int(row["bytes"]),
                dst_device_id=row["dstDeviceId"],
                correlation_id=row["correlationId"],
                global_pid=row["globalPid"],
                pid=row["pid"],
                process_name=row["processName"],
                api_start_ns=row["apiStart"],
                api_end_ns=row["apiEnd"],
                api_name=row["apiName"],
            )
        )
    return events


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def infer_copies_per_iteration(events: list[CopyEvent]) -> int:
    destinations = {event.dst_device_id for event in events if event.dst_device_id is not None}
    if destinations:
        return len(destinations)
    return 1


def trim_warmup_events(
    events: list[CopyEvent],
    copy_mode: str,
    skip_warmup_iterations: int,
    copies_per_iteration: int,
) -> tuple[list[CopyEvent], int, int]:
    if skip_warmup_iterations < 0:
        raise ValueError("--skip-warmup-iterations must be non-negative")
    if copies_per_iteration <= 0:
        raise ValueError("--copies-per-iteration must be positive")
    if not events or skip_warmup_iterations == 0:
        return events, 0, 0

    if copy_mode == "batch":
        grouped: dict[tuple[int | None, int | None], list[CopyEvent]] = defaultdict(list)
        ungrouped: list[CopyEvent] = []
        for event in events:
            if event.correlation_id is None:
                ungrouped.append(event)
            else:
                grouped[(event.global_pid, event.correlation_id)].append(event)
        groups = sorted(grouped.values(), key=lambda group: min(e.start_ns for e in group))
        skipped_groups = min(skip_warmup_iterations, len(groups))
        skip_keys = {
            (group[0].global_pid, group[0].correlation_id)
            for group in groups[:skipped_groups]
        }
        trimmed = [
            event
            for event in events
            if (event.global_pid, event.correlation_id) not in skip_keys
        ]
        return trimmed, len(events) - len(trimmed), skipped_groups

    events_to_skip = min(skip_warmup_iterations * copies_per_iteration, len(events))
    skipped_iterations = events_to_skip // copies_per_iteration
    return events[events_to_skip:], events_to_skip, skipped_iterations


def summarize_gaps(events: list[CopyEvent]) -> dict[str, float | int | None]:
    gaps_ns = [
        events[i].start_ns - events[i - 1].end_ns for i in range(1, len(events))
    ]
    return {
        "count": len(gaps_ns),
        "average_ns": average([float(v) for v in gaps_ns]),
        "average_us": (
            average([float(v) for v in gaps_ns]) / 1_000.0 if gaps_ns else None
        ),
        "min_ns": min(gaps_ns) if gaps_ns else None,
        "max_ns": max(gaps_ns) if gaps_ns else None,
    }


def summarize_waits(events: list[CopyEvent]) -> dict[str, float | int | None]:
    waits_ns = [
        event.start_ns - event.api_end_ns
        for event in events
        if event.api_end_ns is not None
    ]
    return {
        "count": len(waits_ns),
        "average_ns": average([float(v) for v in waits_ns]),
        "average_us": (
            average([float(v) for v in waits_ns]) / 1_000.0 if waits_ns else None
        ),
        "min_ns": min(waits_ns) if waits_ns else None,
        "max_ns": max(waits_ns) if waits_ns else None,
    }


def summarize_memcpy_durations(events: list[CopyEvent]) -> dict[str, float | int | None]:
    durations_ns = [event.duration_ns for event in events]
    return {
        "count": len(durations_ns),
        "average_ns": average([float(v) for v in durations_ns]),
        "average_us": (
            average([float(v) for v in durations_ns]) / 1_000.0
            if durations_ns
            else None
        ),
        "min_ns": min(durations_ns) if durations_ns else None,
        "max_ns": max(durations_ns) if durations_ns else None,
    }


def summarize_api_groups(events: list[CopyEvent]) -> dict[str, Any]:
    grouped: dict[tuple[int | None, int | None], list[CopyEvent]] = defaultdict(list)
    for event in events:
        if event.correlation_id is not None:
            grouped[(event.global_pid, event.correlation_id)].append(event)
    group_sizes = [len(group) for group in grouped.values()]
    return {
        "api_group_count": len(group_sizes),
        "average_memcpy_ptop_per_api_group": average([float(v) for v in group_sizes]),
        "min_memcpy_ptop_per_api_group": min(group_sizes) if group_sizes else None,
        "max_memcpy_ptop_per_api_group": max(group_sizes) if group_sizes else None,
    }


def summarize_nvlink_metrics(
    conn: sqlite3.Connection,
    first_start_ns: int | None,
    last_end_ns: int | None,
    peak_gb_s: float | None,
    include_protocol: bool,
) -> dict[str, dict[str, Any]]:
    if first_start_ns is None or last_end_ns is None:
        return {}

    user_filter = "" if include_protocol else "AND ti.metricName LIKE '%User Data%'"
    rows = conn.execute(
        f"""
        SELECT
            gm.timestamp,
            CASE
                WHEN ti.metricName LIKE 'NVLink RX%' THEN 'rx'
                WHEN ti.metricName LIKE 'NVLink TX%' THEN 'tx'
            END AS direction,
            SUM(gm.value) AS value,
            GROUP_CONCAT(DISTINCT ti.metricName) AS metricNames
        FROM GPU_METRICS AS gm
        JOIN TARGET_INFO_GPU_METRICS AS ti
            ON ti.typeId = gm.typeId
            AND ti.metricId = gm.metricId
        WHERE gm.timestamp BETWEEN ? AND ?
            AND (ti.metricName LIKE 'NVLink RX%' OR ti.metricName LIKE 'NVLink TX%')
            {user_filter}
        GROUP BY gm.timestamp, direction
        HAVING direction IS NOT NULL
        ORDER BY gm.timestamp, direction
        """,
        (first_start_ns, last_end_ns),
    ).fetchall()

    values: dict[str, list[float]] = defaultdict(list)
    metric_names: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        direction = row["direction"]
        values[direction].append(float(row["value"]))
        for name in (row["metricNames"] or "").split(","):
            if name:
                metric_names[direction].add(name)

    result: dict[str, dict[str, Any]] = {}
    for direction in ("rx", "tx"):
        direction_values = values.get(direction, [])
        avg_percent = average(direction_values)
        entry: dict[str, Any] = {
            "sample_count": len(direction_values),
            "average_percent_of_peak": avg_percent,
            "min_percent_of_peak": min(direction_values) if direction_values else None,
            "max_percent_of_peak": max(direction_values) if direction_values else None,
            "included_metrics": sorted(metric_names.get(direction, set())),
        }
        if avg_percent is not None and peak_gb_s is not None:
            entry["average_gb_s"] = avg_percent / 100.0 * peak_gb_s
        result[direction] = entry
    return result


def build_summary(
    conn: sqlite3.Connection,
    sqlite_path: Path,
    bus_id: str,
    peak_gb_s: float | None,
    skip_warmup_iterations: int,
    copy_mode: str = "auto",
    copies_per_iteration: int | None = None,
) -> dict[str, Any]:
    gpu = find_gpu(conn, bus_id)
    mode = detect_copy_mode(conn, sqlite_path, copy_mode)
    ptop_copy_kind = find_ptop_copy_kind(conn)
    all_events = load_source_copy_events(conn, int(gpu["cuDevice"]), ptop_copy_kind)
    inferred_copies_per_iteration = infer_copies_per_iteration(all_events)
    actual_copies_per_iteration = copies_per_iteration or inferred_copies_per_iteration
    events, skipped_events, skipped_iterations = trim_warmup_events(
        all_events,
        mode,
        skip_warmup_iterations,
        actual_copies_per_iteration,
    )

    throughputs_gb_s = [event.throughput_gb_s for event in events]
    throughputs_gib_s = [event.throughput_gib_s for event in events]
    first_start_ns = min((event.start_ns for event in events), default=None)
    last_end_ns = max((event.end_ns for event in events), default=None)
    paired_event_count = sum(1 for event in events if event.api_end_ns is not None)
    total_bytes = sum(event.bytes for event in events)

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
            "inferred_copies_per_iteration": inferred_copies_per_iteration,
            "copies_per_iteration_used_for_warmup_skip": actual_copies_per_iteration,
            "requested_skipped_warmup_iterations": skip_warmup_iterations,
            "skipped_warmup_iterations": skipped_iterations,
            "skipped_warmup_source_memcpy_ptop_count": skipped_events,
            "source_memcpy_ptop_count": len(events),
            "paired_cuda_api_memcpy_ptop_count": paired_event_count,
            "unpaired_source_memcpy_ptop_count": len(events) - paired_event_count,
        },
        "api_groups": summarize_api_groups(events),
        "throughput": {
            "average_event_gb_s": average(throughputs_gb_s),
            "average_event_gib_s": average(throughputs_gib_s),
            "min_event_gib_s": min(throughputs_gib_s) if events else None,
            "max_event_gib_s": max(throughputs_gib_s) if events else None,
            "total_bytes": total_bytes,
        },
        "time_gap_between_consecutive_memcpy_ptop_source": summarize_gaps(events),
        "time_spent_in_memcpy_ptop_source": summarize_memcpy_durations(events),
        "waiting_time_memcpy_start_minus_api_end": summarize_waits(events),
        "nvlink_user_data_metrics_over_copy_window": summarize_nvlink_metrics(
            conn, first_start_ns, last_end_ns, peak_gb_s, include_protocol=False
        ),
        "nvlink_user_plus_protocol_metrics_over_copy_window": summarize_nvlink_metrics(
            conn, first_start_ns, last_end_ns, peak_gb_s, include_protocol=True
        ),
    }


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_ns_as_s_plus_ms(value_ns: int | None) -> str:
    if value_ns is None:
        return "n/a"
    seconds = value_ns // 1_000_000_000
    remainder_ms = (value_ns % 1_000_000_000) / 1_000_000.0
    return f"{seconds}s + {remainder_ms:.3f}ms"


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
        f"{counts['skipped_warmup_source_memcpy_ptop_count']} of "
        f"{counts['all_source_memcpy_ptop_count_before_warmup_skip']} "
        "source Memcpy PtoP events"
    )
    print()
    print("1. Counts")
    print(f"   Source Memcpy PtoP events: {counts['source_memcpy_ptop_count']}")
    print(f"   Paired CUDA API Memcpy PtoP events: {counts['paired_cuda_api_memcpy_ptop_count']}")
    print(f"   Unpaired source Memcpy PtoP events: {counts['unpaired_source_memcpy_ptop_count']}")
    print(
        "   API groups after warmup skip: "
        f"{api_groups['api_group_count']}, "
        f"avg Memcpy PtoP/API={fmt_float(api_groups['average_memcpy_ptop_per_api_group'])}"
    )
    print()
    print("2. Average Memcpy PtoP (source) throughput")
    print(
        "   "
        f"{fmt_float(throughput['average_event_gib_s'])} GiB/s "
        f"({fmt_float(throughput['average_event_gb_s'])} GB/s decimal)"
    )
    print()
    print("3. Average time gap between consecutive Memcpy PtoP (source)")
    print(f"   {fmt_float(gap['average_us'])} us over {gap['count']} gaps")
    print()
    print("4. Average time spent in each Memcpy PtoP (source)")
    print(
        "   "
        f"{fmt_float(memcpy_duration['average_us'])} us "
        f"over {memcpy_duration['count']} events"
    )
    print()
    print("5. Average waiting time: Memcpy PtoP start - CUDA API end")
    print(f"   {fmt_float(wait['average_us'])} us over {wait['count']} paired events")
    print()
    print("6. Average NVLink RX/TX metrics over the copy window")
    print(
        "   Window: "
        f"{fmt_ns_as_s_plus_ms(window['first_memcpy_start_ns'])} to "
        f"{fmt_ns_as_s_plus_ms(window['last_memcpy_end_ns'])} "
        f"({fmt_float(window['duration_ns'] / 1_000_000.0 if window['duration_ns'] else None)} ms)"
    )
    for label, data in (
        ("user data", nvlink_user),
        ("user + protocol data", nvlink_all),
    ):
        rx = data.get("rx", {})
        tx = data.get("tx", {})
        print(
            f"   {label}: "
            f"RX {fmt_float(rx.get('average_percent_of_peak'))}% of peak, "
            f"TX {fmt_float(tx.get('average_percent_of_peak'))}% of peak "
            f"(samples RX={rx.get('sample_count', 0)}, TX={tx.get('sample_count', 0)})"
        )
        if "average_gb_s" in rx or "average_gb_s" in tx:
            print(
                "      converted: "
                f"RX {fmt_float(rx.get('average_gb_s'))} GB/s, "
                f"TX {fmt_float(tx.get('average_gb_s'))} GB/s"
            )


def main() -> int:
    args = parse_args()
    try:
        sqlite_path = resolve_sqlite(args.report, args.force_export)
        with connect(sqlite_path) as conn:
            summary = build_summary(
                conn,
                sqlite_path=sqlite_path,
                bus_id=args.gpu_bus_id,
                peak_gb_s=args.nvlink_peak_gb_s,
                skip_warmup_iterations=args.skip_warmup_iterations,
                copy_mode=args.copy_mode,
                copies_per_iteration=args.copies_per_iteration,
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
