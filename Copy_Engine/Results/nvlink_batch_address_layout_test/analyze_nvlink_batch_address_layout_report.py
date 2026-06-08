#!/usr/bin/env python3
"""Analyze one NVLink batch address-layout Nsight report.

This experiment uses the same source-side Memcpy PtoP and NVLink metric
queries as ``nvlink_copy_engine_test``.  Keep this entrypoint in the local
results folder so it mirrors the two-script workflow used by the earlier
experiment, while reusing the shared analyzer implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
COPY_ENGINE_RESULTS_DIR = SCRIPT_DIR.parent / "nvlink_copy_engine_test"
sys.path.insert(0, str(COPY_ENGINE_RESULTS_DIR))

import analyze_nvlink_copy_engine_report as _copy_analyzer  # noqa: E402
from analyze_nvlink_copy_engine_report import *  # noqa: F401,F403,E402


DEFAULT_REPORT = SCRIPT_DIR / "src_discontinuous" / "4*8k_batch.nsys-rep"
DEFAULT_WARMUP_ITERS = 10
DEFAULT_TIMED_ITERS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cudaMemcpyPeerAsync / Memcpy PtoP source statistics and "
            "NVLink RX/TX metrics for one batch address-layout report."
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
        "--gpu-bus-id",
        default=_copy_analyzer.DEFAULT_GPU_BUS_ID,
        help=(
            "PCI bus ID of the collected/source GPU. "
            f"Default: {_copy_analyzer.DEFAULT_GPU_BUS_ID}"
        ),
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
        "--skip-warmup-copies",
        type=int,
        default=None,
        help=(
            "Explicit number of earliest source-side Memcpy PtoP events to skip. "
            "By default this is inferred from --warmup-iters and --timed-iters."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help=f"Warmup iterations used by the benchmark. Default: {DEFAULT_WARMUP_ITERS}.",
    )
    parser.add_argument(
        "--timed-iters",
        type=int,
        default=DEFAULT_TIMED_ITERS,
        help=f"Timed iterations used by the benchmark. Default: {DEFAULT_TIMED_ITERS}.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser.parse_args()


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

    gpu = _copy_analyzer.find_gpu(conn, bus_id)
    ptop_copy_kind = _copy_analyzer.find_ptop_copy_kind(conn)
    all_events = _copy_analyzer.load_source_copy_events(
        conn, int(gpu["cuDevice"]), ptop_copy_kind
    )
    total_iters = warmup_iters + timed_iters
    return round(len(all_events) * warmup_iters / total_iters)


def main() -> int:
    args = parse_args()
    try:
        sqlite_path = _copy_analyzer.resolve_sqlite(args.report, args.force_export)
        with _copy_analyzer.connect(sqlite_path) as conn:
            skip_warmup_copies = (
                args.skip_warmup_copies
                if args.skip_warmup_copies is not None
                else infer_warmup_skip(
                    conn,
                    args.gpu_bus_id,
                    args.warmup_iters,
                    args.timed_iters,
                )
            )
            summary = _copy_analyzer.build_summary(
                conn,
                sqlite_path=sqlite_path,
                bus_id=args.gpu_bus_id,
                peak_gb_s=args.nvlink_peak_gb_s,
                skip_warmup_copies=skip_warmup_copies,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _copy_analyzer.print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
