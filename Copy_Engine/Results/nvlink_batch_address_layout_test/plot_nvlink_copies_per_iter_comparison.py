#!/usr/bin/env python3
"""Compare NVLink utilization at equal total transfer size per iteration."""

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
NON_UNIFORM_REPORT_RE = re.compile(
    r"total_(?P<total>\d+(?:\.\d+)?[kKmMgG])_"
    r"(?P<mode>[A-Za-z0-9-]+)_"
    r"(?P<copies>\d+)x(?P<small>\d+(?:\.\d+)?[kKmMgG])\+"
    r"(?P<last>\d+(?:\.\d+)?[kKmMgG])\.sqlite$"
)
LAYOUT_ALIASES = {
    "src-discontinuous": "src_discontinuous",
    "src_discontinuous": "src_discontinuous",
    "both-discontinuous": "both_discontinuous",
    "both_discontinuous": "both_discontinuous",
}
COLORS = {
    "uniform-1": "#d17c29",
    "uniform-2": "#31688e",
    "uniform-4": "#35b779",
    "uniform-8": "#7c4fa3",
    "nonuniform-4": "#c44e52",
}
MARKERS = {
    "uniform-1": "^",
    "uniform-2": "o",
    "uniform-4": "s",
    "uniform-8": "D",
    "nonuniform-4": "X",
}
SERIES_ORDER = ("uniform-1", "uniform-2", "uniform-4", "uniform-8", "nonuniform-4")


@dataclass(frozen=True)
class ReportInfo:
    path: Path
    copies_per_iter: int
    size_label: str
    size_bytes: float
    total_size_bytes_value: float
    mode: str
    series_id: str
    series_label: str
    copy_size_description: str

    @property
    def total_size_bytes(self) -> float:
        return self.total_size_bytes_value


@dataclass(frozen=True)
class ReportPoint:
    info: ReportInfo
    rx_percent: float
    tx_percent: float
    skipped_warmup_events: int
    measured_copy_events: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one NVLink bandwidth-utilization graph comparing "
            "--copies-per-iter settings at equal total bytes per iteration."
        )
    )
    parser.add_argument(
        "--layout",
        default="src-discontinuous",
        choices=sorted(LAYOUT_ALIASES),
        help="Layout folder / benchmark layout to compare. Default: src-discontinuous.",
    )
    parser.add_argument(
        "--copy-mode",
        default="batch",
        help="Copy mode to compare. Default: batch.",
    )
    parser.add_argument(
        "--copies-per-iter",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="Copies-per-iteration values to compare. Default: 1 2 4 8.",
    )
    parser.add_argument(
        "--gpu-bus-id",
        default=analyzer.DEFAULT_GPU_BUS_ID,
        help=f"PCI bus ID of the collected/source GPU. Default: {analyzer.DEFAULT_GPU_BUS_ID}",
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
        "--skip-warmup-copies",
        type=int,
        default=None,
        help=(
            "Explicit number of earliest source-side Memcpy PtoP events to skip "
            "in each SQLite file. By default this is inferred from iteration counts."
        ),
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
        default=SCRIPT_DIR / "average_nvlink_copies_per_iter_comparison.png",
        help="Output PNG path. Default: ./average_nvlink_copies_per_iter_comparison.png.",
    )
    return parser.parse_args()


def message_size_bytes(label: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmg])", label.lower())
    if match is None:
        return float("inf")
    value = float(match.group(1))
    multiplier = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2)]
    return value * multiplier


def transfer_size_label(size_bytes: float) -> str:
    if not math.isfinite(size_bytes):
        return "unknown"
    size_int = int(size_bytes)
    if size_int % 1024 == 0:
        return f"{size_int // 1024}k"
    return f"{size_int}b"


def parse_report_path(path: Path) -> ReportInfo:
    match = REPORT_RE.fullmatch(path.name)
    if match is not None:
        copies_per_iter = int(match.group("copies"))
        size_label = match.group("size").lower()
        size_bytes = message_size_bytes(size_label)
        return ReportInfo(
            path=path,
            copies_per_iter=copies_per_iter,
            size_label=size_label,
            size_bytes=size_bytes,
            total_size_bytes_value=copies_per_iter * size_bytes,
            mode=match.group("mode"),
            series_id=f"uniform-{copies_per_iter}",
            series_label=f"{copies_per_iter} equal-size copies/iter",
            copy_size_description=size_label,
        )

    match = NON_UNIFORM_REPORT_RE.fullmatch(path.name)
    if match is not None:
        copies_per_iter = int(match.group("copies"))
        total_label = match.group("total").lower()
        small_label = match.group("small").lower()
        last_label = match.group("last").lower()
        return ReportInfo(
            path=path,
            copies_per_iter=copies_per_iter,
            size_label=f"{small_label}+{last_label}",
            size_bytes=float("nan"),
            total_size_bytes_value=message_size_bytes(total_label),
            mode=match.group("mode"),
            series_id=f"nonuniform-{copies_per_iter}",
            series_label=(
                f"{copies_per_iter} non-uniform copies/iter "
                f"({copies_per_iter - 1}x{small_label}+remainder)"
            ),
            copy_size_description=f"{copies_per_iter - 1}x{small_label}+{last_label}",
        )

    raise ValueError(
        f"Cannot parse {path.name}. Expected names like 2*256k_batch.sqlite "
        "or total_512k_batch_4x8k+488k.sqlite."
    )


def discover_sqlites(layout: str, copy_mode: str, copies_per_iter: set[int]) -> list[ReportInfo]:
    layout_dir = SCRIPT_DIR / LAYOUT_ALIASES[layout]
    if not layout_dir.exists():
        raise FileNotFoundError(layout_dir)

    reports: list[ReportInfo] = []
    for path in layout_dir.glob("*.sqlite"):
        info = parse_report_path(path)
        if info.mode == copy_mode and info.copies_per_iter in copies_per_iter:
            reports.append(info)

    if not reports:
        raise RuntimeError(
            f"No SQLite files found for layout={layout}, copy-mode={copy_mode}, "
            f"copies-per-iter={sorted(copies_per_iter)}"
        )
    return sorted(reports, key=lambda item: (item.total_size_bytes, item.copies_per_iter))


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


def require_number(value: Any, field_name: str) -> float:
    if value is None:
        raise RuntimeError(f"Missing value for {field_name}")
    return float(value)


def load_point(
    info: ReportInfo,
    gpu_bus_id: str,
    skip_warmup_copies: int | None,
    warmup_iters: int,
    timed_iters: int,
    nvlink_key: str,
) -> ReportPoint:
    with analyzer.connect(info.path) as conn:
        skipped_warmup_events = (
            skip_warmup_copies
            if skip_warmup_copies is not None
            else infer_warmup_skip(conn, gpu_bus_id, warmup_iters, timed_iters)
        )
        summary = analyzer.build_summary(
            conn,
            sqlite_path=info.path,
            bus_id=gpu_bus_id,
            peak_gb_s=None,
            skip_warmup_copies=skipped_warmup_events,
        )

    return ReportPoint(
        info=info,
        rx_percent=require_number(
            summary[nvlink_key]["rx"]["average_percent_of_peak"],
            "NVLink RX",
        ),
        tx_percent=require_number(
            summary[nvlink_key]["tx"]["average_percent_of_peak"],
            "NVLink TX",
        ),
        skipped_warmup_events=skipped_warmup_events,
        measured_copy_events=int(summary["counts"]["source_memcpy_ptop_count"]),
    )


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


def plot(points: list[ReportPoint], layout: str, copy_mode: str, output_path: Path, include_protocol: bool) -> None:
    total_sizes = sorted({point.info.total_size_bytes for point in points})
    labels = [transfer_size_label(total_size) for total_size in total_sizes]
    series_ids = sorted(
        {point.info.series_id for point in points},
        key=lambda series_id: (
            SERIES_ORDER.index(series_id)
            if series_id in SERIES_ORDER
            else len(SERIES_ORDER),
            series_id,
        ),
    )
    series_labels = {
        point.info.series_id: point.info.series_label
        for point in points
    }
    by_key = {
        (point.info.series_id, point.info.total_size_bytes): point
        for point in points
    }

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(max(8.2, len(labels) * 0.9), 6.6),
        dpi=160,
        sharex=True,
    )
    x_values = list(range(len(labels)))
    all_rx_values: list[float] = []
    all_tx_values: list[float] = []

    for series_id in series_ids:
        color = COLORS.get(series_id)
        marker = MARKERS.get(series_id, "o")
        rx_values = [
            by_key.get((series_id, total_size)).rx_percent
            if by_key.get((series_id, total_size)) is not None
            else math.nan
            for total_size in total_sizes
        ]
        tx_values = [
            by_key.get((series_id, total_size)).tx_percent
            if by_key.get((series_id, total_size)) is not None
            else math.nan
            for total_size in total_sizes
        ]
        all_rx_values.extend(rx_values)
        all_tx_values.extend(tx_values)
        label = series_labels[series_id]
        axes[0].plot(
            x_values,
            rx_values,
            marker=marker,
            linewidth=2.2,
            markersize=5,
            label=label,
            color=color,
        )
        axes[1].plot(
            x_values,
            tx_values,
            marker=marker,
            linewidth=2.2,
            markersize=5,
            label=label,
            color=color,
        )

    title = f"Average NVLink bandwidth utilization at equal total transfer size ({layout}, {copy_mode})"
    if include_protocol:
        title += " (user + protocol)"
    axes[0].set_title(title)
    axes[0].set_ylabel("RX (% of peak)")
    axes[1].set_ylabel("TX (% of peak)")
    axes[1].set_xlabel("Total transfer size per iteration (copies/iter * copy size)")
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(labels)
    for ax, values in ((axes[0], all_rx_values), (axes[1], all_tx_values)):
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        add_y_margin(ax, values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def print_values(points: list[ReportPoint]) -> None:
    print("Values:")
    for point in sorted(
        points,
        key=lambda item: (
            item.info.total_size_bytes,
            SERIES_ORDER.index(item.info.series_id)
            if item.info.series_id in SERIES_ORDER
            else len(SERIES_ORDER),
            item.info.series_id,
        ),
    ):
        print(
            f"  total={transfer_size_label(point.info.total_size_bytes)}, "
            f"{point.info.series_label}, copy_sizes={point.info.copy_size_description}: "
            f"events={point.measured_copy_events}, skipped={point.skipped_warmup_events}, "
            f"NVLink RX={point.rx_percent:.3f}%, TX={point.tx_percent:.3f}%"
        )


def main() -> int:
    args = parse_args()
    try:
        copies_per_iter = set(args.copies_per_iter)
        infos = discover_sqlites(args.layout, args.copy_mode, copies_per_iter)
        nvlink_key = (
            "nvlink_user_data_metrics_over_copy_window"
            if args.nvlink_mode == "user-data"
            else "nvlink_user_plus_protocol_metrics_over_copy_window"
        )
        points = [
            load_point(
                info,
                gpu_bus_id=args.gpu_bus_id,
                skip_warmup_copies=args.skip_warmup_copies,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                nvlink_key=nvlink_key,
            )
            for info in infos
        ]
        plot(
            points,
            layout=args.layout,
            copy_mode=args.copy_mode,
            output_path=args.output,
            include_protocol=args.nvlink_mode == "user-plus-protocol",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated graph: {args.output}")
    print_values(points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
