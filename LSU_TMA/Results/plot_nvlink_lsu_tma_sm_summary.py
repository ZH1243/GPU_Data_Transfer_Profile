#!/usr/bin/env python3
"""Plot LSU/TMA NVLink utilization as a function of SM count."""

from __future__ import annotations

import argparse
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

import analyze_nvlink_lsu_tma_report as analyzer


SCRIPT_DIR = Path(__file__).resolve().parent
COPY_TEST_DIR = SCRIPT_DIR / "nvlink_lsu_tma_copy_test"
ROUND_ROBIN_TEST_DIR = SCRIPT_DIR / "nvlink_lsu_tma_round_robin_copy_test"
SM_RE = re.compile(r"sm\*(\d+)", re.IGNORECASE)
MODE_RE = re.compile(r"^(?:round_robin_)?(lsu|tma)_persistent_", re.IGNORECASE)

SERIES_ORDER = (
    ("copy", "lsu"),
    ("copy", "tma"),
    ("round-robin", "lsu"),
    ("round-robin", "tma"),
)
SERIES_STYLE = {
    ("copy", "lsu"): {
        "label": "Non-round-robin LSU",
        "color": "#2f6fbb",
        "marker": "o",
    },
    ("copy", "tma"): {
        "label": "Non-round-robin TMA",
        "color": "#d17c29",
        "marker": "s",
    },
    ("round-robin", "lsu"): {
        "label": "Round-robin LSU",
        "color": "#4f8f3a",
        "marker": "^",
    },
    ("round-robin", "tma"): {
        "label": "Round-robin TMA",
        "color": "#8f4f9f",
        "marker": "D",
    },
}


@dataclass(frozen=True)
class PlotPoint:
    path: Path
    sm_count: int
    test_kind: str
    copy_mode: str
    rx_percent: float
    tx_percent: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RX/TX NVLink utilization plots for LSU/TMA copy and "
            "round-robin copy reports."
        )
    )
    parser.add_argument(
        "reports",
        nargs="*",
        type=Path,
        default=default_sqlites(),
        help=(
            "Input .sqlite or .nsys-rep files. Defaults to all LSU/TMA result "
            "SQLite files under this directory."
        ),
    )
    parser.add_argument(
        "--gpu-bus-id",
        default=analyzer.DEFAULT_GPU_BUS_ID,
        help=f"PCI bus ID of the collected GPU. Default: {analyzer.DEFAULT_GPU_BUS_ID}",
    )
    parser.add_argument(
        "--skip-warmup-kernels",
        type=int,
        default=analyzer.DEFAULT_SKIP_WARMUP_KERNELS,
        help=(
            "Ignore this many earliest matching kernels on the collected GPU. "
            f"Default: {analyzer.DEFAULT_SKIP_WARMUP_KERNELS}."
        ),
    )
    parser.add_argument(
        "--nvlink-mode",
        choices=("user-data", "user-plus-protocol"),
        default="user-data",
        help="Which NVLink metric group to plot. Default: user-data.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="For .nsys-rep input, regenerate the sibling .sqlite with nsys export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "nvlink_lsu_tma_sm_utilization.png",
        help="Output PNG path. Default: ./nvlink_lsu_tma_sm_utilization.png.",
    )
    return parser.parse_args()


def default_sqlites() -> list[Path]:
    return sorted(
        list(COPY_TEST_DIR.glob("*.sqlite"))
        + list(ROUND_ROBIN_TEST_DIR.glob("*.sqlite"))
    )


def classify_report(path: Path) -> tuple[int, str, str]:
    stem = path.stem
    sm_match = SM_RE.search(stem)
    if sm_match is None:
        raise ValueError(f"Could not parse SM count from {path.name!r}")

    mode_match = MODE_RE.search(stem)
    if mode_match is None:
        raise ValueError(f"Could not parse copy mode from {path.name!r}")

    test_kind = "round-robin" if stem.startswith("round_robin_") else "copy"
    copy_mode = mode_match.group(1).lower()
    return int(sm_match.group(1)), test_kind, copy_mode


def load_summary(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    sqlite_path = analyzer.resolve_sqlite(path, args.force_export)
    with analyzer.connect(sqlite_path) as conn:
        return analyzer.build_summary(
            conn,
            sqlite_path=sqlite_path,
            bus_id=args.gpu_bus_id,
            kernel_name=None,
            skip_warmup_kernels=args.skip_warmup_kernels,
            peak_gb_s=None,
        )


def require_number(value: Any, field_name: str, path: Path) -> float:
    if value is None:
        raise RuntimeError(f"Missing {field_name} in {path}")
    return float(value)


def make_point(path: Path, args: argparse.Namespace) -> PlotPoint:
    sm_count, test_kind, copy_mode = classify_report(path)
    summary = load_summary(path, args)
    key = (
        "nvlink_user_data_metrics_over_kernel_window"
        if args.nvlink_mode == "user-data"
        else "nvlink_user_plus_protocol_metrics_over_kernel_window"
    )
    metrics = summary[key]
    return PlotPoint(
        path=path,
        sm_count=sm_count,
        test_kind=test_kind,
        copy_mode=copy_mode,
        rx_percent=require_number(
            metrics.get("rx", {}).get("average_percent_of_peak"),
            "NVLink RX utilization",
            path,
        ),
        tx_percent=require_number(
            metrics.get("tx", {}).get("average_percent_of_peak"),
            "NVLink TX utilization",
            path,
        ),
    )


def collect_points(args: argparse.Namespace) -> list[PlotPoint]:
    points: list[PlotPoint] = []
    seen: set[tuple[str, str, int]] = set()
    for report in args.reports:
        point = make_point(report.expanduser().resolve(), args)
        key = (point.test_kind, point.copy_mode, point.sm_count)
        if key in seen:
            raise RuntimeError(
                "Duplicate result for "
                f"{point.test_kind} {point.copy_mode} sm={point.sm_count}"
            )
        seen.add(key)
        points.append(point)
    return sorted(points, key=lambda p: (p.test_kind, p.copy_mode, p.sm_count))


def add_y_margin(ax: plt.Axes, values: list[float]) -> None:
    if not values:
        return
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        span = max(abs(high), 1.0)
    ax.set_ylim(max(0.0, low - span * 0.14), high + span * 0.18)


def plot(points: list[PlotPoint], output_path: Path, nvlink_mode: str) -> None:
    if not points:
        raise RuntimeError("No points to plot")

    sm_counts = sorted({point.sm_count for point in points})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=160, sharex=True)

    for axis, direction, attr_name in (
        (axes[0], "RX", "rx_percent"),
        (axes[1], "TX", "tx_percent"),
    ):
        plotted_values: list[float] = []
        for series_key in SERIES_ORDER:
            series_points = [
                point
                for point in points
                if (point.test_kind, point.copy_mode) == series_key
            ]
            if not series_points:
                continue
            series_points.sort(key=lambda point: point.sm_count)
            style = SERIES_STYLE[series_key]
            xs = [point.sm_count for point in series_points]
            ys = [float(getattr(point, attr_name)) for point in series_points]
            plotted_values.extend(ys)
            axis.plot(
                xs,
                ys,
                marker=style["marker"],
                linewidth=2.2,
                markersize=5.5,
                label=style["label"],
                color=style["color"],
            )

        axis.set_title(f"NVLink {direction}")
        axis.set_xticks(sm_counts)
        axis.set_xlabel("Number of SMs")
        axis.grid(True, axis="y", alpha=0.3)
        add_y_margin(axis, plotted_values)

    axes[0].set_ylabel("Average utilization (% of peak)")
    mode_label = (
        "user data"
        if nvlink_mode == "user-data"
        else "user + protocol data"
    )
    fig.suptitle(f"NVLink utilization during persistent kernels ({mode_label})")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def print_values(points: list[PlotPoint]) -> None:
    print("Values plotted:")
    print("  test_kind,copy_mode,sm,rx_percent,tx_percent,path")
    for point in points:
        print(
            "  "
            f"{point.test_kind},"
            f"{point.copy_mode},"
            f"{point.sm_count},"
            f"{point.rx_percent:.3f},"
            f"{point.tx_percent:.3f},"
            f"{point.path}"
        )


def main() -> int:
    args = parse_args()
    if not args.reports:
        print("error: no input reports found", file=sys.stderr)
        return 1

    try:
        points = collect_points(args)
        plot(points, args.output.expanduser().resolve(), args.nvlink_mode)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated figure: {args.output.expanduser().resolve()}")
    print_values(points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
