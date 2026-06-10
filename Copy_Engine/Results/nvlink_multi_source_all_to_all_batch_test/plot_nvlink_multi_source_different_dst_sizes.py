#!/usr/bin/env python3
"""Plot NVLink utilization for destination-indexed size-layout reports."""

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

import analyze_nvlink_multi_source_all_to_all_batch_report as analyzer


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_RE = re.compile(
    r"total_(?P<total>\d+(?:\.\d+)?[kKmMgG])_different_dst_sizes_batch_"
    r"(?P<count>\d+)\*(?P<small>\d+(?:\.\d+)?[kKmMgG])\+"
    r"(?P<last>\d+(?:\.\d+)?[kKmMgG])\.(?:sqlite|nsys-rep)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an NVLink RX/TX utilization plot for "
            "destination-indexed multi-source all-to-all reports."
        )
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        default=default_reports(),
        help=(
            "Input .sqlite or .nsys-rep files. Defaults to all "
            "./total_*_different_dst_sizes_batch_* reports, preferring .sqlite."
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
        help=f"Warmup iterations to skip. Default: {analyzer.DEFAULT_WARMUP_ITERATIONS}.",
    )
    parser.add_argument(
        "--timed-iterations",
        type=int,
        default=analyzer.DEFAULT_TIMED_ITERATIONS,
        help=f"Timed benchmark iterations. Default: {analyzer.DEFAULT_TIMED_ITERATIONS}.",
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
        help="Memcpy PtoP activities per cudaMemcpyBatchAsync call. Default: infer.",
    )
    parser.add_argument(
        "--nvlink-mode",
        choices=("user-data", "user-plus-protocol"),
        default="user-data",
        help="Which NVLink metric group to plot. Default: user-data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "different_dst_sizes_nvlink_rx_tx_metrics.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def parse_report_name(path: Path) -> tuple[str, int, str, str] | None:
    match = REPORT_RE.fullmatch(path.name)
    if match is None:
        return None
    return (
        match.group("total").lower(),
        int(match.group("count")),
        match.group("small").lower(),
        match.group("last").lower(),
    )


def size_bytes(label: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", label.lower())
    if match is None:
        return float("inf")
    value = float(match.group(1))
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return value * multiplier


def default_reports() -> list[Path]:
    by_stem: dict[str, Path] = {}
    for path in sorted(SCRIPT_DIR.glob("total_*_different_dst_sizes_batch_*")):
        if path.suffix not in (".sqlite", ".nsys-rep"):
            continue
        if parse_report_name(path) is None:
            continue
        current = by_stem.get(path.stem)
        if current is None or path.suffix == ".sqlite":
            by_stem[path.stem] = path
    return sorted(
        by_stem.values(),
        key=lambda path: size_bytes(parse_report_name(path)[2]),  # type: ignore[index]
    )


def load_summary(
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sqlite_path = analyzer.base.resolve_sqlite(path.expanduser().resolve(), args.force_export)
    with analyzer.base.connect(sqlite_path) as conn:
        return analyzer.build_summary(
            conn,
            sqlite_path=sqlite_path,
            bus_id=args.gpu_bus_id,
            peak_gb_s=None,
            skip_warmup_iterations=args.skip_warmup_iterations,
            copy_mode="batch",
            batch_calls_per_iteration=args.batch_calls_per_iteration,
            entries_per_batch=args.ptop_events_per_batch,
            timed_iterations=args.timed_iterations,
            processes_per_report=args.processes_per_report,
        )


def require_number(value: Any, field_name: str) -> float:
    if value is None:
        raise RuntimeError(f"Missing value for {field_name}")
    return float(value)


def add_y_margin(ax: plt.Axes, values: list[float]) -> None:
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0)
    ax.set_ylim(max(0.0, low - span * 0.12), high + span * 0.2)


def plot_nvlink_utilization(
    labels: list[str],
    rx_values: list[float],
    tx_values: list[float],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(7.8, len(labels) * 1.05), 4.8), dpi=160)
    x_values = list(range(len(labels)))
    ax.plot(x_values, rx_values, marker="o", linewidth=2.2, label="RX", color="#31688e")
    ax.plot(
        x_values,
        tx_values,
        marker="s",
        linewidth=2.2,
        linestyle="--",
        label="TX",
        color="#d17c29",
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Destination-index sub-buffer layout")
    ax.set_ylabel("Average throughput (% of peak)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    add_y_margin(ax, rx_values + tx_values)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        rows: list[tuple[str, str, str, dict[str, Any]]] = []
        for report_path in args.reports:
            name_info = parse_report_name(report_path)
            if name_info is None:
                raise RuntimeError(
                    f"Cannot parse {report_path.name}; expected "
                    "total_<size>_different_dst_sizes_batch_<n>*<small>+<last>"
                )
            total_label, count, small, last = name_info
            layout_label = f"{count}x{small}+{last}"
            summary = load_summary(report_path, args)
            rows.append((small, total_label, layout_label, summary))

        rows.sort(key=lambda row: size_bytes(row[0]))
        nvlink_key = (
            "nvlink_user_data_metrics_over_copy_window"
            if args.nvlink_mode == "user-data"
            else "nvlink_user_plus_protocol_metrics_over_copy_window"
        )
        labels = [row[2] for row in rows]
        rx_values = [
            require_number(row[3][nvlink_key]["rx"]["average_percent_of_peak"], "NVLink RX")
            for row in rows
        ]
        tx_values = [
            require_number(row[3][nvlink_key]["tx"]["average_percent_of_peak"], "NVLink TX")
            for row in rows
        ]
        total_labels = sorted({row[1] for row in rows}, key=size_bytes)
        total_text = ", ".join(total_labels)
        title = (
            f"Destination-indexed NVLink RX/TX utilization, total={total_text}"
            if args.nvlink_mode == "user-data"
            else f"Destination-indexed NVLink RX/TX utilization, total={total_text} (user + protocol)"
        )
        plot_nvlink_utilization(labels, rx_values, tx_values, title, args.output)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated graph: {args.output}")
    print("Values:")
    for label, rx, tx in zip(labels, rx_values, tx_values):
        print(f"  {label}: NVLink RX={rx:.3f}%, TX={tx:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
