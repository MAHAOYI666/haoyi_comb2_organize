#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd


ORGANIZE_ROOT = Path(__file__).resolve().parent
for local_path in (ORGANIZE_ROOT, ORGANIZE_ROOT / "evals", ORGANIZE_ROOT / "vendor" / "comb2-simbase"):
    text_path = str(local_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

from comb_eval.daily_eval import DEFAULT_LONG_RATIO, evaluate_daily, format_daily_evaluation, read_daily_evaluation
from comb_eval.run_eval_other import (
    _write_frame_if_requested,
    _write_text_if_requested,
    run_corr,
    run_exposure,
    run_overall,
    run_pnl,
    run_sim,
    run_va,
)

# Only the two-parquet run/read workflow is implemented here. All other
# evaluation modes are in run_eval_other.
DEFAULT_VA_WEIGHTS = "0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="runEval",
        description="Evaluate comb2 config outputs or local parquet/csv artifacts.",
        epilog="examples: runEval config.xml | runEval myposition.parquet target.parquet [run|read] | runEval --sim daily_ic.parquet",
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("target_path", nargs="?", default=None, help="Target alpha parquet for direct daily VA evaluation")
    parser.add_argument("daily_mode", nargs="?", choices=("run", "read"), default="run", help="Direct daily VA mode; run computes and saves results, read reuses saved results")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overall", action="store_true", help="Run the full config evaluation; default when no mode is given")
    mode.add_argument("--corr", nargs=2, metavar=("LEFT", "RIGHT"), help="Summarize daily matrix correlation for two parquet/csv matrices")
    mode.add_argument("--sim", metavar="PATH", help="Summarize an existing daily IC parquet/csv table")
    mode.add_argument("--pnl", metavar="PATH", help="Summarize an existing daily PnL parquet/csv table")
    mode.add_argument("--exposure", metavar="PATH", help="Summarize Barra style exposure for a signal matrix")
    mode.add_argument("--va", nargs=2, metavar=("BASE", "NEW"), help="Summarize value-add for two daily PnL parquet/csv tables")

    parser.add_argument("--report-dir", help="For overall mode, directory for eval artifacts; defaults to <output_root>/eval_report")
    parser.add_argument("--plot-output", help="For overall mode, path for the signal analysis long image")
    parser.add_argument("--pnlzz500", help="Optional benchmark pnl path")
    parser.add_argument("--booksize", type=float, help="For --va, booksize used to normalize PnL")
    parser.add_argument("--normalize-names", action="store_true", help="For --sim, map ic/5dic to 1d_IC/5d_IC names")
    parser.add_argument("--corr-days", type=int, default=240, help="For --corr, use only the most recent N overlapping dates; default 240")
    parser.add_argument("--min-valid", type=int, default=1000, help="For --corr, minimum nonzero overlapping instruments per day")
    parser.add_argument("--top-pct", type=float, default=10.0, help="For --corr, long-holding overlap top percentage; default 10")
    parser.add_argument("--cache-path", help="For --exposure, parent directory containing AshareCache")
    parser.add_argument("--eval-dir", help="For direct daily VA, persistent result directory used by run/read")
    parser.add_argument("--mosek", default="/root/mosek/mosek.lic", help="MOSEK license file for direct daily evaluation; default: /root/mosek/mosek.lic")
    parser.add_argument("--worker", type=int, default=10, help="For direct daily VA, concurrent weight backtests; default 10")
    parser.add_argument("--long-ratio", type=float, default=DEFAULT_LONG_RATIO, help="For direct daily VA, long bucket ratio used by long-short adjustment; default 0.5")
    parser.add_argument("--ti", type=int, help="For direct daily VA, execution and label time in HHMMSS; omitted when both inputs have one common intraday time")
    parser.add_argument("--simple", action="store_true", help="For direct daily VA, use the simplified optimizer profile; default uses the legacy optimizer profile")
    parser.add_argument("--exposure-mode", type=int, choices=(0, 1), default=0, help="For --exposure, 0=cross-sectional correlation, 1=beta")
    parser.add_argument("--column", default="longonly_pnl", help="For --va, PnL column to compare")
    parser.add_argument("--weights", default=DEFAULT_VA_WEIGHTS, help="For --va, comma-separated new-pnl blend weights")
    parser.add_argument("--output", help="Optional path for mode-specific generated table, such as exposure or VA csv")
    parser.add_argument("--summary-output", help="Optional path to write the formatted summary text")
    parser.add_argument("--blended-output", help="For --va, optional path to write blended and incremental daily series")
    parser.add_argument("--skip-deciles", action="store_true", help="For overall mode, skip 10-group backtests")
    parser.add_argument("--skip-exposure", action="store_true", help="For overall mode, skip Barra exposure analysis")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.target_path is not None:
            return run_daily_eval(args)
        if args.simple:
            raise ValueError("--simple requires direct daily evaluation inputs")
        if args.corr is not None:
            return run_corr(args)
        if args.sim is not None:
            return run_sim(args)
        if args.pnl is not None:
            return run_pnl(args)
        if args.exposure is not None:
            return run_exposure(args)
        if args.va is not None:
            return run_va(args)
        return run_overall(args)
    except Exception as exc:
        print(f"[EVAL] {exc}", file=sys.stderr)
        return 2


def run_daily_eval(args: argparse.Namespace) -> int:
    if args.config_flag:
        raise ValueError("direct daily evaluation does not accept --config")
    if args.overall or any(
        value is not None
        for value in (args.corr, args.sim, args.pnl, args.exposure, args.va)
    ):
        raise ValueError("direct daily evaluation cannot be combined with another mode")
    if args.worker <= 0:
        raise ValueError("--worker must be positive")
    if not 0.0 <= args.long_ratio <= 1.0:
        raise ValueError("--long-ratio must be between 0 and 1")
    if args.daily_mode == "run":
        mosek_path = Path(args.mosek).expanduser().resolve()
        if not mosek_path.is_file():
            raise FileNotFoundError(f"MOSEK license file not found: {mosek_path}")
        os.environ["MOSEKLM_LICENSE_FILE"] = str(mosek_path)
    if args.daily_mode == "run":
        print(
            "\n".join(
                [
                    f"[{time.strftime('%H:%M:%S', time.localtime())}]Start Evaluation",
                    str(Path(args.config).expanduser().resolve()),
                    str(Path(args.target_path).expanduser().resolve()),
                ]
            ),
            flush=True,
        )
        result = evaluate_daily(
            args.config,
            args.target_path,
            cache_path=args.cache_path,
            eval_dir=args.eval_dir,
            worker=args.worker,
            long_ratio=args.long_ratio,
            ti=args.ti,
            simple=args.simple,
            start=args.start,
            end=args.end,
        )
    else:
        result = read_daily_evaluation(
            args.config,
            args.target_path,
            cache_path=args.cache_path,
            eval_dir=args.eval_dir,
            long_ratio=args.long_ratio,
            ti=args.ti,
            simple=args.simple,
            start=args.start,
            end=args.end,
        )
    _write_frame_if_requested(result.va_table, args.output)
    text = format_daily_evaluation(result, include_header=False)
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
