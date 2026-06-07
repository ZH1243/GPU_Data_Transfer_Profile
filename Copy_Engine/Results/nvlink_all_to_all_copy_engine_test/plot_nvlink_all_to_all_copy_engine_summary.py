#!/usr/bin/env python3
"""Plot all-to-all copy-engine summary graphs for batch and separate reports."""

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

import analyze_nvlink_all_to_all_copy_engine_report as analyzer


SCRIPT_DIR = Path(__file__).resolve().parent
MODES = ("batch", "separate")
MODE_LABELS = {
    "batch": "batch",
    "separate": "separate",
}
MODE_COLORS = {
    "batch": "#2f6fbb",
    "separate": "#d17c29",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all-to-all summary plots for Average Memcpy PtoP "
            "throughput, Memcpy PtoP gaps, and NVLink RX/TX metrics."
        )
    )
    parser.add_argument(
        "sqlite",
        nargs="*",
        type=Path,
        default=default_sqlites(),
        help=(
            "SQLite files to plot. Defaults to all ./a2a_batch_<size>.sqlite "
            "and ./a2a_separate_<size>.sqlite files."
        ),
    )
    parser.add_argument(
        "--gpu-bus-id",
        default=analyzer.DEFAULT_GPU_BUS_ID,
        help=f"PCI bus ID of the collected/source GPU. Default: {analyzer.DEFAULT_GPU_BUS_ID}",
    )
    parser.add_argument(
        "--skip-warmup-iterations",
        type=int,
        default=analyzer.DEFAULT_SKIP_WARMUP_ITERATIONS,
        help=(
            "Ignore this many earliest all-to-all iterations in each SQLite "
            f"file. Default: {analyzer.DEFAULT_SKIP_WARMUP_ITERATIONS}."
        ),
    )
    parser.add_argument(
        "--copies-per-iteration",
        type=int,
        default=None,
        help="Number of source-side destination copies per iteration. Default: infer.",
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


def parse_sqlite_name(path: Path) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"a2a_(batch|separate)_(\d+(?:\.\d+)?[kKmMgG])\.sqlite",
        path.name,
    )
    if match is None:
        return None
    return match.group(1).lower(), match.group(2).lower()


def message_size_bytes(label: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", label.lower())
    if match is None:
        return float("inf")
    value = float(match.group(1))
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return value * multiplier


def default_sqlites() -> list[Path]:
    discovered = [
        path for path in SCRIPT_DIR.glob("a2a_*.sqlite") if parse_sqlite_name(path)
    ]
    return sorted(
        discovered,
        key=lambda path: (
            message_size_bytes(parse_sqlite_name(path)[1]),  # type: ignore[index]
            parse_sqlite_name(path)[0],  # type: ignore[index]
        ),
    )


def load_summary(
    path: Path,
    gpu_bus_id: str,
    skip_warmup_iterations: int,
    copies_per_iteration: int | None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    name_info = parse_sqlite_name(path)
    requested_mode = name_info[0] if name_info else "auto"
    with analyzer.connect(path) as conn:
        return analyzer.build_summary(
            conn,
            sqlite_path=path,
            bus_id=gpu_bus_id,
            peak_gb_s=None,
            skip_warmup_iterations=skip_warmup_iterations,
            copy_mode=requested_mode,
            copies_per_iteration=copies_per_iteration,
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
    return (max(7.6, len(labels) * 0.72), 4.6)


def line_plot(
    labels: list[str],
    values_by_mode: dict[str, list[float | None]],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=plot_size(labels), dpi=160)
    x_values = list(range(len(labels)))
    all_values: list[float] = []
    markers = {"batch": "o", "separate": "s"}
    for mode in MODES:
        y_values = values_by_mode[mode]
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
            marker=markers[mode],
            linewidth=2.2,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Message size")
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
    gap_by_mode: dict[str, list[float | None]],
    duration_by_mode: dict[str, list[float | None]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=plot_size(labels), dpi=160)
    x_values = list(range(len(labels)))
    series = [
        ("batch gap", gap_by_mode["batch"], "#2f6fbb", "o", "-"),
        ("batch duration", duration_by_mode["batch"], "#2f6fbb", "s", "--"),
        ("separate gap", gap_by_mode["separate"], "#d17c29", "o", "-"),
        ("separate duration", duration_by_mode["separate"], "#d17c29", "s", "--"),
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
    ax.set_xlabel("Message size")
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
    rx_by_mode: dict[str, list[float | None]],
    tx_by_mode: dict[str, list[float | None]],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=plot_size(labels), dpi=160)
    x_values = list(range(len(labels)))
    series = [
        ("batch RX", rx_by_mode["batch"], "#2f6fbb", "o", "-"),
        ("batch TX", tx_by_mode["batch"], "#2f6fbb", "s", "--"),
        ("separate RX", rx_by_mode["separate"], "#d17c29", "o", "-"),
        ("separate TX", tx_by_mode["separate"], "#d17c29", "s", "--"),
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
    ax.set_xlabel("Message size")
    ax.set_ylabel("Average throughput (% of peak)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=2)
    add_y_margin(ax, all_values)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def append_value(
    values_by_mode: dict[str, list[float | None]],
    mode: str,
    value: float | None,
) -> None:
    for known_mode in MODES:
        values_by_mode[known_mode].append(value if known_mode == mode else None)


def main() -> int:
    args = parse_args()
    try:
        rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for sqlite_path in args.sqlite:
            name_info = parse_sqlite_name(sqlite_path)
            if name_info is None:
                raise RuntimeError(
                    f"Cannot infer mode/size from {sqlite_path.name}; expected "
                    "a2a_batch_<size>.sqlite or a2a_separate_<size>.sqlite"
                )
            mode, size_label = name_info
            rows[size_label][mode] = load_summary(
                sqlite_path,
                gpu_bus_id=args.gpu_bus_id,
                skip_warmup_iterations=args.skip_warmup_iterations,
                copies_per_iteration=args.copies_per_iteration,
            )

        labels = sorted(rows, key=message_size_bytes)
        throughput_gib_s = {mode: [] for mode in MODES}
        gap_us = {mode: [] for mode in MODES}
        memcpy_duration_us = {mode: [] for mode in MODES}
        nvlink_rx_percent = {mode: [] for mode in MODES}
        nvlink_tx_percent = {mode: [] for mode in MODES}

        nvlink_key = (
            "nvlink_user_data_metrics_over_copy_window"
            if args.nvlink_mode == "user-data"
            else "nvlink_user_plus_protocol_metrics_over_copy_window"
        )

        for label in labels:
            for mode in MODES:
                summary = rows[label].get(mode)
                if summary is None:
                    throughput_gib_s[mode].append(None)
                    gap_us[mode].append(None)
                    memcpy_duration_us[mode].append(None)
                    nvlink_rx_percent[mode].append(None)
                    nvlink_tx_percent[mode].append(None)
                    continue

                throughput_gib_s[mode].append(
                    require_number(summary["throughput"]["average_event_gib_s"], "throughput")
                )
                gap_us[mode].append(
                    require_number(
                        summary["time_gap_between_consecutive_memcpy_ptop_source"][
                            "average_us"
                        ],
                        "time gap",
                    )
                )
                memcpy_duration_us[mode].append(
                    require_number(
                        summary["time_spent_in_memcpy_ptop_source"]["average_us"],
                        "Memcpy PtoP duration",
                    )
                )
                nvlink_rx_percent[mode].append(
                    require_number(
                        summary[nvlink_key]["rx"]["average_percent_of_peak"],
                        "NVLink RX",
                    )
                )
                nvlink_tx_percent[mode].append(
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
        for mode in MODES:
            summary = rows[label].get(mode)
            if summary is None:
                print(f"  {label} {mode}: missing")
                continue
            idx = labels.index(label)
            counts = summary["counts"]
            api_groups = summary["api_groups"]
            print(
                f"  {label} {mode}: copies={counts['source_memcpy_ptop_count']}, "
                f"api_groups={api_groups['api_group_count']}, "
                f"memcpy/api={api_groups['average_memcpy_ptop_per_api_group']:.3f}, "
                f"throughput={throughput_gib_s[mode][idx]:.3f} GiB/s, "
                f"gap={gap_us[mode][idx]:.3f} us, "
                f"memcpy_duration={memcpy_duration_us[mode][idx]:.3f} us, "
                f"NVLink RX={nvlink_rx_percent[mode][idx]:.3f}%, "
                f"TX={nvlink_tx_percent[mode][idx]:.3f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
