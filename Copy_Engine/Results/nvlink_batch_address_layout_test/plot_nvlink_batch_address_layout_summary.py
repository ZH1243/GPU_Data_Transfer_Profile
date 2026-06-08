#!/usr/bin/env python3
"""Plot batch address-layout summary graphs across layout/mode combinations."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_nvlink_batch_address_layout_report as analyzer


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_RE = re.compile(
    r"(?P<copies>\d+)\*(?P<size>\d+(?:\.\d+)?[kKmMgG])_(?P<mode>[A-Za-z0-9-]+)\.sqlite$"
)
LAYOUT_LABELS = {
    "src_discontinuous": "src-discontinuous",
    "both_discontinuous": "both-discontinuous",
}
SERIES_STYLE = {
    ("src_discontinuous", "separate"): ("src-discontinuous / separate", "#31688e", "o"),
    ("src_discontinuous", "batch"): ("src-discontinuous / batch", "#35b779", "s"),
    ("both_discontinuous", "separate"): ("both-discontinuous / separate", "#d17c29", "^"),
    ("both_discontinuous", "batch"): ("both-discontinuous / batch", "#7c4fa3", "D"),
}
SERIES_ORDER = tuple(SERIES_STYLE)


@dataclass(frozen=True)
class ReportInfo:
    path: Path
    layout: str
    mode: str
    copies_per_iter: int
    size_label: str
    size_bytes: float


@dataclass(frozen=True)
class ReportSummary:
    info: ReportInfo
    summary: dict[str, Any]
    skipped_warmup_events: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate summary plots for the batch address-layout experiment. "
            "Each figure compares every available --layout/--copy-mode combination."
        )
    )
    parser.add_argument(
        "sqlite",
        nargs="*",
        type=Path,
        default=default_sqlites(),
        help=(
            "SQLite files to plot. Defaults to all ./*/*.sqlite files under this "
            "results directory."
        ),
    )
    parser.add_argument(
        "--gpu-bus-id",
        default=analyzer.DEFAULT_GPU_BUS_ID,
        help=f"PCI bus ID of the collected/source GPU. Default: {analyzer.DEFAULT_GPU_BUS_ID}",
    )
    parser.add_argument(
        "--skip-warmup-copies",
        type=int,
        default=None,
        help=(
            "Explicit number of earliest source-side Memcpy PtoP events to skip in "
            "each SQLite file. By default this is inferred from --warmup-iters and "
            "--timed-iters."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Warmup iterations used when collecting the reports. Default: 10.",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=100,
        help="Timed iterations used when collecting the reports. Default: 100.",
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


def default_sqlites() -> list[Path]:
    return sorted(SCRIPT_DIR.glob("*/*.sqlite"), key=sort_key_for_path)


def parse_report_path(path: Path) -> ReportInfo:
    match = REPORT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"Cannot parse {path}. Expected names like 4*256k_batch.sqlite."
        )

    layout = path.parent.name
    if layout not in LAYOUT_LABELS:
        raise ValueError(
            f"Cannot parse layout from {path}. Expected subfolders: "
            f"{', '.join(sorted(LAYOUT_LABELS))}."
        )

    size_label = match.group("size").lower()
    return ReportInfo(
        path=path,
        layout=layout,
        mode=match.group("mode"),
        copies_per_iter=int(match.group("copies")),
        size_label=size_label,
        size_bytes=message_size_bytes(size_label),
    )


def message_size_bytes(label: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", label.lower())
    if match is None:
        return float("inf")
    value = float(match.group(1))
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return value * multiplier


def sort_key_for_path(path: Path) -> tuple[str, int, float, str]:
    try:
        info = parse_report_path(path)
    except ValueError:
        return (path.parent.name, 0, float("inf"), path.name)
    return (info.layout, info.copies_per_iter, info.size_bytes, info.mode)


def infer_warmup_skip(
    conn: Any,
    bus_id: str,
    warmup_iters: int,
    timed_iters: int,
) -> int:
    if warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative")
    if timed_iters <= 0:
        raise ValueError("--timed-iters must be positive")

    gpu = analyzer.find_gpu(conn, bus_id)
    ptop_copy_kind = analyzer.find_ptop_copy_kind(conn)
    all_events = analyzer.load_source_copy_events(conn, int(gpu["cuDevice"]), ptop_copy_kind)
    total_iters = warmup_iters + timed_iters
    return round(len(all_events) * warmup_iters / total_iters)


def load_summary(
    info: ReportInfo,
    gpu_bus_id: str,
    skip_warmup_copies: int | None,
    warmup_iters: int,
    timed_iters: int,
) -> ReportSummary:
    path = info.path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with analyzer.connect(path) as conn:
        skipped_warmup_events = (
            skip_warmup_copies
            if skip_warmup_copies is not None
            else infer_warmup_skip(conn, gpu_bus_id, warmup_iters, timed_iters)
        )
        summary = analyzer.build_summary(
            conn,
            sqlite_path=path,
            bus_id=gpu_bus_id,
            peak_gb_s=None,
            skip_warmup_copies=skipped_warmup_events,
        )
    return ReportSummary(info=info, summary=summary, skipped_warmup_events=skipped_warmup_events)


def require_number(value: Any, field_name: str) -> float:
    if value is None:
        raise RuntimeError(f"Missing value for {field_name}")
    return float(value)


def value_or_nan(points: dict[tuple[str, str, str], ReportSummary], key: tuple[str, str, str], metric: str, nvlink_key: str) -> float:
    point = points.get(key)
    if point is None:
        return math.nan

    summary = point.summary
    if metric == "throughput":
        return require_number(summary["throughput"]["average_event_gib_s"], metric)
    if metric == "gap":
        return require_number(
            summary["time_gap_between_consecutive_memcpy_ptop_source"]["average_us"],
            metric,
        )
    if metric == "duration":
        return require_number(
            summary["time_spent_in_memcpy_ptop_source"]["average_us"],
            metric,
        )
    if metric == "nvlink_rx":
        return require_number(
            summary[nvlink_key]["rx"]["average_percent_of_peak"],
            metric,
        )
    if metric == "nvlink_tx":
        return require_number(
            summary[nvlink_key]["tx"]["average_percent_of_peak"],
            metric,
        )
    raise ValueError(f"Unknown metric: {metric}")


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def add_y_margin(ax: plt.Axes, values: list[float]) -> None:
    values = finite_values(values)
    if not values:
        return
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0)
    ax.set_ylim(max(0.0, low - span * 0.12), high + span * 0.18)


def plot_size(labels: list[str], rows: int = 1) -> tuple[float, float]:
    return (max(8.2, len(labels) * 0.9), 4.6 if rows == 1 else 6.6)


def plot_series(
    ax: plt.Axes,
    labels: list[str],
    points: dict[tuple[str, str, str], ReportSummary],
    metric: str,
    nvlink_key: str,
) -> list[float]:
    x_values = list(range(len(labels)))
    all_values: list[float] = []
    for layout, mode in SERIES_ORDER:
        series_label, color, marker = SERIES_STYLE[(layout, mode)]
        values = [
            value_or_nan(points, (layout, mode, size_label), metric, nvlink_key)
            for size_label in labels
        ]
        all_values.extend(values)
        ax.plot(
            x_values,
            values,
            marker=marker,
            linewidth=2.0,
            markersize=5,
            label=series_label,
            color=color,
        )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    add_y_margin(ax, all_values)
    return all_values


def throughput_plot(
    labels: list[str],
    points: dict[tuple[str, str, str], ReportSummary],
    copies_per_iter: int | None,
    output_path: Path,
    nvlink_key: str,
) -> None:
    fig, ax = plt.subplots(figsize=plot_size(labels), dpi=160)
    plot_series(ax, labels, points, "throughput", nvlink_key)
    ax.set_xlabel("Copy size")
    ax.set_ylabel("Average throughput (GiB/s)")
    title = "Average Memcpy PtoP (source) throughput"
    if copies_per_iter is not None:
        title += f" ({copies_per_iter} copies/iter)"
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def timing_plot(
    labels: list[str],
    points: dict[tuple[str, str, str], ReportSummary],
    output_path: Path,
    nvlink_key: str,
) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=plot_size(labels, rows=2),
        dpi=160,
        sharex=True,
    )
    plot_series(axes[0], labels, points, "gap", nvlink_key)
    axes[0].set_ylabel("Average gap (us)")
    axes[0].set_title("Average Memcpy PtoP gap and duration")
    axes[0].legend(ncol=2, fontsize=8)

    plot_series(axes[1], labels, points, "duration", nvlink_key)
    axes[1].set_xlabel("Copy size")
    axes[1].set_ylabel("Average duration (us)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def nvlink_plot(
    labels: list[str],
    points: dict[tuple[str, str, str], ReportSummary],
    output_path: Path,
    nvlink_key: str,
    include_protocol: bool,
) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=plot_size(labels, rows=2),
        dpi=160,
        sharex=True,
    )
    plot_series(axes[0], labels, points, "nvlink_rx", nvlink_key)
    axes[0].set_ylabel("RX (% of peak)")
    title = "Average NVLink RX/TX metrics over the copy window"
    if include_protocol:
        title += " (user + protocol)"
    axes[0].set_title(title)
    axes[0].legend(ncol=2, fontsize=8)

    plot_series(axes[1], labels, points, "nvlink_tx", nvlink_key)
    axes[1].set_xlabel("Copy size")
    axes[1].set_ylabel("TX (% of peak)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def print_values(
    labels: list[str],
    points: dict[tuple[str, str, str], ReportSummary],
    nvlink_key: str,
) -> None:
    print("Values:")
    for layout, mode in SERIES_ORDER:
        series_label = SERIES_STYLE[(layout, mode)][0]
        print(f"  {series_label}:")
        for size_label in labels:
            point = points.get((layout, mode, size_label))
            if point is None:
                continue
            summary = point.summary
            counts = summary["counts"]
            throughput = require_number(summary["throughput"]["average_event_gib_s"], "throughput")
            gap = require_number(
                summary["time_gap_between_consecutive_memcpy_ptop_source"]["average_us"],
                "time gap",
            )
            duration = require_number(
                summary["time_spent_in_memcpy_ptop_source"]["average_us"],
                "Memcpy PtoP duration",
            )
            rx = require_number(summary[nvlink_key]["rx"]["average_percent_of_peak"], "NVLink RX")
            tx = require_number(summary[nvlink_key]["tx"]["average_percent_of_peak"], "NVLink TX")
            print(
                f"    {size_label}: copies={counts['source_memcpy_ptop_count']}, "
                f"skipped={point.skipped_warmup_events}, "
                f"throughput={throughput:.3f} GiB/s, gap={gap:.3f} us, "
                f"memcpy_duration={duration:.3f} us, "
                f"NVLink RX={rx:.3f}%, TX={tx:.3f}%"
            )


def main() -> int:
    args = parse_args()
    try:
        infos = [parse_report_path(path) for path in args.sqlite]
        infos.sort(key=lambda info: (info.size_bytes, info.layout, info.mode))
        summaries = [
            load_summary(
                info,
                gpu_bus_id=args.gpu_bus_id,
                skip_warmup_copies=args.skip_warmup_copies,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
            )
            for info in infos
        ]

        points = {
            (item.info.layout, item.info.mode, item.info.size_label): item
            for item in summaries
        }
        labels = sorted(
            {item.info.size_label for item in summaries},
            key=message_size_bytes,
        )
        copies_per_iter_values = {item.info.copies_per_iter for item in summaries}
        copies_per_iter = (
            next(iter(copies_per_iter_values))
            if len(copies_per_iter_values) == 1
            else None
        )
        nvlink_key = (
            "nvlink_user_data_metrics_over_copy_window"
            if args.nvlink_mode == "user-data"
            else "nvlink_user_plus_protocol_metrics_over_copy_window"
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        throughput_path = args.output_dir / "average_memcpy_ptop_source_throughput.png"
        gap_path = args.output_dir / "average_memcpy_ptop_source_time_gap.png"
        nvlink_path = args.output_dir / "average_nvlink_rx_tx_metrics.png"

        throughput_plot(labels, points, copies_per_iter, throughput_path, nvlink_key)
        timing_plot(labels, points, gap_path, nvlink_key)
        nvlink_plot(
            labels,
            points,
            nvlink_path,
            nvlink_key,
            include_protocol=args.nvlink_mode == "user-plus-protocol",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("Generated graphs:")
    print(f"  {throughput_path}")
    print(f"  {gap_path}")
    print(f"  {nvlink_path}")
    print()
    print_values(labels, points, nvlink_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
