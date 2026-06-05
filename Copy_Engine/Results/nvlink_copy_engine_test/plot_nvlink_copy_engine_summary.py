#!/usr/bin/env python3
"""Plot copy-engine summary graphs for the available message-size reports."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_nvlink_copy_engine_report as analyzer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITES = [
    SCRIPT_DIR / "1*8k.sqlite",
    SCRIPT_DIR / "1*80k.sqlite",
    SCRIPT_DIR / "1*500k.sqlite",
    SCRIPT_DIR / "1*1m.sqlite",
    SCRIPT_DIR / "1*10m.sqlite",
    SCRIPT_DIR / "1*100m.sqlite",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate summary plots for Average Memcpy PtoP throughput, "
            "Memcpy PtoP gaps, and NVLink RX/TX metrics."
        )
    )
    parser.add_argument(
        "sqlite",
        nargs="*",
        type=Path,
        default=DEFAULT_SQLITES,
        help=(
            "SQLite files to plot. Defaults to ./1*8k.sqlite ./1*80k.sqlite "
            "./1*500k.sqlite ./1*1m.sqlite ./1*10m.sqlite ./1*100m.sqlite."
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
        default=analyzer.DEFAULT_SKIP_WARMUP_COPIES,
        help=(
            "Ignore this many earliest source-side Memcpy PtoP events in each "
            f"SQLite file. Default: {analyzer.DEFAULT_SKIP_WARMUP_COPIES}."
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


def message_size_label(path: Path) -> str:
    match = re.search(r"(\d+(?:\.\d+)?[kKmMgG])(?=\.sqlite$)", path.name)
    if match is None:
        return path.stem
    return match.group(1).lower()


def message_size_bytes(label: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", label.lower())
    if match is None:
        return float("inf")
    value = float(match.group(1))
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return value * multiplier


def load_summary(path: Path, gpu_bus_id: str, skip_warmup_copies: int) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with analyzer.connect(path) as conn:
        return analyzer.build_summary(
            conn,
            sqlite_path=path,
            bus_id=gpu_bus_id,
            peak_gb_s=None,
            skip_warmup_copies=skip_warmup_copies,
        )


def line_plot(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    x_values = list(range(len(labels)))
    ax.plot(x_values, values, marker="o", linewidth=2.2, color=color)
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Message size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    add_y_margin(ax, values)
    for x_value, value in zip(x_values, values):
        ax.annotate(
            f"{value:.3f}",
            (x_value, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def timing_plot(
    labels: list[str],
    gap_values: list[float],
    duration_values: list[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    x_values = list(range(len(labels)))
    ax.plot(
        x_values,
        gap_values,
        marker="o",
        linewidth=2.2,
        label="Time gap",
        color="#5f8f3f",
    )
    ax.plot(
        x_values,
        duration_values,
        marker="s",
        linewidth=2.2,
        label="Memcpy duration",
        color="#7c4fa3",
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Message size")
    ax.set_ylabel("Average time (us)")
    ax.set_title("Average Memcpy PtoP gap and duration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    add_y_margin(ax, gap_values + duration_values)
    for values, offset, color in (
        (gap_values, 9, "#5f8f3f"),
        (duration_values, -14, "#7c4fa3"),
    ):
        for x_value, value in zip(x_values, values):
            ax.annotate(
                f"{value:.3f}",
                (x_value, value),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=8,
                color=color,
            )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def add_y_margin(ax: plt.Axes, values: list[float]) -> None:
    if not values:
        return
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0)
    ax.set_ylim(low - span * 0.12, high + span * 0.18)


def nvlink_plot(
    labels: list[str],
    rx_values: list[float],
    tx_values: list[float],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    x_values = list(range(len(labels)))
    ax.plot(x_values, rx_values, marker="o", linewidth=2.2, label="RX", color="#2f6fbb")
    ax.plot(x_values, tx_values, marker="s", linewidth=2.2, label="TX", color="#d17c29")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Message size")
    ax.set_ylabel("Average throughput (% of peak)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    add_y_margin(ax, rx_values + tx_values)
    for values, offset, color in (
        (rx_values, 9, "#2f6fbb"),
        (tx_values, -14, "#d17c29"),
    ):
        for x_value, value in zip(x_values, values):
            ax.annotate(
                f"{value:.3f}",
                (x_value, value),
                textcoords="offset points",
                xytext=(0, offset),
                ha="center",
                fontsize=8,
                color=color,
            )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def require_number(value: Any, field_name: str) -> float:
    if value is None:
        raise RuntimeError(f"Missing value for {field_name}")
    return float(value)


def main() -> int:
    args = parse_args()
    try:
        rows = []
        for sqlite_path in args.sqlite:
            label = message_size_label(sqlite_path)
            summary = load_summary(
                sqlite_path,
                gpu_bus_id=args.gpu_bus_id,
                skip_warmup_copies=args.skip_warmup_copies,
            )
            rows.append((message_size_bytes(label), label, summary))

        rows.sort(key=lambda item: item[0])
        labels = [row[1] for row in rows]
        summaries = [row[2] for row in rows]

        throughput_gib_s = [
            require_number(s["throughput"]["average_event_gib_s"], "throughput")
            for s in summaries
        ]
        gap_us = [
            require_number(
                s["time_gap_between_consecutive_memcpy_ptop_source"]["average_us"],
                "time gap",
            )
            for s in summaries
        ]
        memcpy_duration_us = [
            require_number(
                s["time_spent_in_memcpy_ptop_source"]["average_us"],
                "Memcpy PtoP duration",
            )
            for s in summaries
        ]
        nvlink_key = (
            "nvlink_user_data_metrics_over_copy_window"
            if args.nvlink_mode == "user-data"
            else "nvlink_user_plus_protocol_metrics_over_copy_window"
        )
        nvlink_rx_percent = [
            require_number(s[nvlink_key]["rx"]["average_percent_of_peak"], "NVLink RX")
            for s in summaries
        ]
        nvlink_tx_percent = [
            require_number(s[nvlink_key]["tx"]["average_percent_of_peak"], "NVLink TX")
            for s in summaries
        ]

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
            color="#31688e",
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
    for label, summary, throughput, gap, duration, rx, tx in zip(
        labels,
        summaries,
        throughput_gib_s,
        gap_us,
        memcpy_duration_us,
        nvlink_rx_percent,
        nvlink_tx_percent,
    ):
        counts = summary["counts"]
        print(
            f"  {label}: copies={counts['source_memcpy_ptop_count']}, "
            f"throughput={throughput:.3f} GiB/s, gap={gap:.3f} us, "
            f"memcpy_duration={duration:.3f} us, "
            f"NVLink RX={rx:.3f}%, TX={tx:.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
