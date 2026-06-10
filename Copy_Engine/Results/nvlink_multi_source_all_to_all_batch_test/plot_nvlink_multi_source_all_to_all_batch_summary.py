#!/usr/bin/env python3
"""Plot multi-source all-to-all batch summary graphs for available reports."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_nvlink_multi_source_all_to_all_batch_report as analyzer


SCRIPT_DIR = Path(__file__).resolve().parent
LAYOUTS = ("small-plus-large", "uniform")
LAYOUT_LABELS = {
    "small-plus-large": "3x8K + remainder",
    "uniform": "uniform",
}
LAYOUT_COLORS = {
    "small-plus-large": "#31688e",
    "uniform": "#d17c29",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multi-source all-to-all batch plots for Average Memcpy "
            "PtoP throughput, Memcpy PtoP gaps, and NVLink RX/TX metrics."
        )
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        default=default_reports(),
        help=(
            "Input .nsys-rep or .sqlite files. Defaults to all "
            "./total_*_batch_* reports, preferring .sqlite when present."
        ),
    )
    parser.add_argument(
        "--gpu-bus-id",
        default=analyzer.base.DEFAULT_GPU_BUS_ID,
        help=f"PCI bus ID of the collected/source GPU. Default: {analyzer.base.DEFAULT_GPU_BUS_ID}",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="For .nsys-rep inputs, regenerate sibling .sqlite files with nsys export.",
    )
    parser.add_argument(
        "--skip-warmup-iterations",
        type=int,
        default=analyzer.DEFAULT_WARMUP_ITERATIONS,
        help=(
            "Ignore this many earliest benchmark iterations in each report. "
            f"Default: {analyzer.DEFAULT_WARMUP_ITERATIONS}."
        ),
    )
    parser.add_argument(
        "--timed-iterations",
        type=int,
        default=analyzer.DEFAULT_TIMED_ITERATIONS,
        help=(
            "Timed benchmark iterations. Used only when falling back to CUDA "
            f"runtime API timing. Default: {analyzer.DEFAULT_TIMED_ITERATIONS}."
        ),
    )
    parser.add_argument(
        "--processes-per-report",
        type=int,
        default=None,
        help="Number of torchrun worker processes captured in each report. Default: infer.",
    )
    parser.add_argument(
        "--batch-calls-per-iteration",
        type=int,
        default=None,
        help="Number of cudaMemcpyBatchAsync calls per iteration. Default: infer.",
    )
    parser.add_argument(
        "--ptop-events-per-batch",
        "--entries-per-batch",
        dest="ptop_events_per_batch",
        type=int,
        default=None,
        help=(
            "Memcpy PtoP activities per cudaMemcpyBatchAsync call. Default: infer. "
            "The legacy spelling --entries-per-batch is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--nvlink-mode",
        choices=("user-data", "user-plus-protocol"),
        default="user-data",
        help="Which NVLink metric group to plot. Default: user-data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory for generated PNG graphs. Default: this script's folder.",
    )
    return parser.parse_args()


def parse_report_name(path: Path) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"total_(\d+(?:\.\d+)?[kKmMgG])_batch_(\d+)x8k\+(\d+(?:\.\d+)?[kKmMgG])"
        r"\.(?:sqlite|nsys-rep)",
        path.name,
    )
    if match is not None:
        num_8k = int(match.group(2))
        last_copy = match.group(3).lower()
        return (
            match.group(1).lower(),
            "small-plus-large",
            f"{num_8k}x8k+{last_copy}",
        )

    match = re.fullmatch(
        r"total_(\d+(?:\.\d+)?[kKmMgG])_batch_uniform_buffer_sizes"
        r"\.(?:sqlite|nsys-rep)",
        path.name,
    )
    if match is not None:
        return match.group(1).lower(), "uniform", "uniform"

    return None


def size_bytes(label: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", label.lower())
    if match is None:
        return float("inf")
    value = float(match.group(1))
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return value * multiplier


def default_reports() -> list[Path]:
    by_stem: dict[str, Path] = {}
    for path in sorted(SCRIPT_DIR.glob("total_*_batch_*")):
        if path.suffix not in (".sqlite", ".nsys-rep"):
            continue
        if parse_report_name(path) is None:
            continue
        current = by_stem.get(path.stem)
        if current is None or path.suffix == ".sqlite":
            by_stem[path.stem] = path
    return sorted(
        by_stem.values(),
        key=lambda path: (
            size_bytes(parse_report_name(path)[0]),  # type: ignore[index]
            path.name,
        ),
    )


def load_summary(
    path: Path,
    gpu_bus_id: str,
    force_export: bool,
    skip_warmup_iterations: int,
    timed_iterations: int,
    processes_per_report: int | None,
    batch_calls_per_iteration: int | None,
    entries_per_batch: int | None,
) -> dict[str, Any]:
    sqlite_path = analyzer.base.resolve_sqlite(path.expanduser().resolve(), force_export)
    with analyzer.base.connect(sqlite_path) as conn:
        return analyzer.build_summary(
            conn,
            sqlite_path=sqlite_path,
            bus_id=gpu_bus_id,
            peak_gb_s=None,
            skip_warmup_iterations=skip_warmup_iterations,
            copy_mode="batch",
            batch_calls_per_iteration=batch_calls_per_iteration,
                entries_per_batch=entries_per_batch,
            timed_iterations=timed_iterations,
            processes_per_report=processes_per_report,
        )


def require_number(value: Any, field_name: str) -> float:
    if value is None:
        raise RuntimeError(f"Missing value for {field_name}")
    return float(value)


def add_y_margin(ax: plt.Axes, values: list[float]) -> None:
    if not values:
        return
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0)
    ax.set_ylim(low - span * 0.12, high + span * 0.2)


def plot_size(labels: list[str]) -> tuple[float, float]:
    return (max(7.6, len(labels) * 0.78), 4.6)


def line_plot(
    labels: list[str],
    values_by_layout: dict[str, list[float | None]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=plot_size(labels), dpi=160)
    x_values = list(range(len(labels)))
    all_values: list[float] = []
    markers = {"small-plus-large": "o", "uniform": "s"}
    for layout in LAYOUTS:
        y_values = values_by_layout[layout]
        points = [
            (x_value, value)
            for x_value, value in zip(x_values, y_values)
            if value is not None
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        all_values.extend(float(value) for value in ys)
        ax.plot(
            xs,
            ys,
            marker=markers[layout],
            linewidth=2.2,
            label=LAYOUT_LABELS[layout],
            color=LAYOUT_COLORS[layout],
        )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Total bytes per destination")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    add_y_margin(ax, all_values)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def timing_plot(
    labels: list[str],
    gap_by_layout: dict[str, list[float | None]],
    duration_by_layout: dict[str, list[float | None]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=plot_size(labels), dpi=160)
    x_values = list(range(len(labels)))
    series = [
        ("3x8K + remainder gap", gap_by_layout["small-plus-large"], "#31688e", "o", "-"),
        ("3x8K + remainder duration", duration_by_layout["small-plus-large"], "#31688e", "s", "--"),
        ("uniform gap", gap_by_layout["uniform"], "#d17c29", "o", "-"),
        ("uniform duration", duration_by_layout["uniform"], "#d17c29", "s", "--"),
    ]
    all_values: list[float] = []
    for label, y_values, color, marker, linestyle in series:
        points = [
            (x_value, value)
            for x_value, value in zip(x_values, y_values)
            if value is not None
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        all_values.extend(float(value) for value in ys)
        ax.plot(
            xs,
            ys,
            marker=marker,
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
            color=color,
        )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Total bytes per destination")
    ax.set_ylabel("Average time (us)")
    ax.set_title("Average Memcpy PtoP gap and duration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    add_y_margin(ax, all_values)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def nvlink_plot(
    labels: list[str],
    rx_by_layout: dict[str, list[float | None]],
    tx_by_layout: dict[str, list[float | None]],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=plot_size(labels), dpi=160)
    x_values = list(range(len(labels)))
    series = [
        ("3x8K + remainder RX", rx_by_layout["small-plus-large"], "#31688e", "o", "-"),
        ("3x8K + remainder TX", tx_by_layout["small-plus-large"], "#31688e", "s", "--"),
        ("uniform RX", rx_by_layout["uniform"], "#d17c29", "o", "-"),
        ("uniform TX", tx_by_layout["uniform"], "#d17c29", "s", "--"),
    ]
    all_values: list[float] = []
    for label, y_values, color, marker, linestyle in series:
        points = [
            (x_value, value)
            for x_value, value in zip(x_values, y_values)
            if value is not None
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        all_values.extend(float(value) for value in ys)
        ax.plot(
            xs,
            ys,
            marker=marker,
            linewidth=2.0,
            linestyle=linestyle,
            label=label,
            color=color,
        )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Total bytes per destination")
    ax.set_ylabel("Average throughput (% of peak)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    add_y_margin(ax, all_values)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        rows: dict[str, dict[str, tuple[str, dict[str, Any]]]] = defaultdict(dict)
        for report_path in args.reports:
            name_info = parse_report_name(report_path)
            if name_info is None:
                raise RuntimeError(
                    f"Cannot infer total size from {report_path.name}; expected "
                    "total_<size>_batch_<n>x8k+<size> or "
                    "total_<size>_batch_uniform_buffer_sizes with .nsys-rep or .sqlite"
                )
            total_label, layout, layout_detail = name_info
            summary = load_summary(
                report_path,
                gpu_bus_id=args.gpu_bus_id,
                force_export=args.force_export,
                skip_warmup_iterations=args.skip_warmup_iterations,
                timed_iterations=args.timed_iterations,
                processes_per_report=args.processes_per_report,
                batch_calls_per_iteration=args.batch_calls_per_iteration,
                entries_per_batch=args.ptop_events_per_batch,
            )
            rows[total_label][layout] = (layout_detail, summary)

        labels = sorted(rows, key=size_bytes)
        throughput_gib_s = {layout: [] for layout in LAYOUTS}
        gap_us = {layout: [] for layout in LAYOUTS}
        memcpy_duration_us = {layout: [] for layout in LAYOUTS}
        nvlink_rx_percent = {layout: [] for layout in LAYOUTS}
        nvlink_tx_percent = {layout: [] for layout in LAYOUTS}

        nvlink_key = (
            "nvlink_user_data_metrics_over_copy_window"
            if args.nvlink_mode == "user-data"
            else "nvlink_user_plus_protocol_metrics_over_copy_window"
        )

        for label in labels:
            for layout in LAYOUTS:
                row = rows[label].get(layout)
                if row is None:
                    throughput_gib_s[layout].append(None)
                    gap_us[layout].append(None)
                    memcpy_duration_us[layout].append(None)
                    nvlink_rx_percent[layout].append(None)
                    nvlink_tx_percent[layout].append(None)
                    continue

                _, summary = row
                throughput_gib_s[layout].append(
                    require_number(summary["throughput"]["average_event_gib_s"], "throughput")
                )
                gap_us[layout].append(
                    require_number(
                        summary["time_gap_between_consecutive_memcpy_ptop_source"][
                            "average_us"
                        ],
                        "time gap",
                    )
                )
                memcpy_duration_us[layout].append(
                    require_number(
                        summary["time_spent_in_memcpy_ptop_source"]["average_us"],
                        "Memcpy PtoP duration",
                    )
                )
                nvlink_rx_percent[layout].append(
                    require_number(
                        summary[nvlink_key]["rx"]["average_percent_of_peak"],
                        "NVLink RX",
                    )
                )
                nvlink_tx_percent[layout].append(
                    require_number(
                        summary[nvlink_key]["tx"]["average_percent_of_peak"],
                        "NVLink TX",
                    )
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        throughput_path = args.output_dir / "average_memcpy_ptop_source_throughput.png"
        gap_path = args.output_dir / "average_memcpy_ptop_source_time_gap.png"
        nvlink_path = args.output_dir / "average_nvlink_rx_tx_metrics.png"

        line_plot(
            labels,
            throughput_gib_s,
            ylabel="Average throughput (GiB/s)",
            title="Average Memcpy PtoP (source) throughput",
            output_path=throughput_path,
        )
        timing_plot(labels, gap_us, memcpy_duration_us, gap_path)
        nvlink_title = (
            "Average NVLink RX/TX metrics over the copy window"
            if args.nvlink_mode == "user-data"
            else "Average NVLink RX/TX metrics over the copy window (user + protocol)"
        )
        nvlink_plot(labels, nvlink_rx_percent, nvlink_tx_percent, nvlink_title, nvlink_path)

    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Generated graphs:")
    print(f"  {throughput_path}")
    print(f"  {gap_path}")
    print(f"  {nvlink_path}")
    print()
    print("Values:")
    for label in labels:
        for layout in LAYOUTS:
            row = rows[label].get(layout)
            if row is None:
                continue
            layout_detail, summary = row
            label_index = labels.index(label)
            throughput = throughput_gib_s[layout][label_index]
            gap = gap_us[layout][label_index]
            duration = memcpy_duration_us[layout][label_index]
            rx = nvlink_rx_percent[layout][label_index]
            tx = nvlink_tx_percent[layout][label_index]
            counts = summary["counts"]
            api_groups = summary["api_groups"]
            print(
                f"  total={label} ({layout_detail}): "
                f"copies={counts['source_memcpy_ptop_count']}, "
                f"event_source={summary['event_source']}, "
                f"batch_calls/iter={counts['batch_calls_per_iteration_used_for_warmup_skip']}, "
                f"ptop_events/batch={counts['ptop_events_per_batch_used_for_warmup_skip']}, "
                f"source_buffers/batch={counts['source_buffers_per_batch_from_filename']}, "
                f"avg_ptop_events/api={api_groups['average_memcpy_ptop_per_api_group']:.3f}, "
                f"throughput={throughput:.3f} GiB/s, gap={gap:.3f} us, "
                f"memcpy_duration={duration:.3f} us, "
                f"NVLink RX={rx:.3f}%, TX={tx:.3f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
