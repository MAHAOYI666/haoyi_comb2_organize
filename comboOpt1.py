#!/usr/bin/env python3
"""Single-signal opt1 backtest: comboOpt1 SIGNAL.parquet [--config CONFIG.xml]."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from combo2.bootstrap import bootstrap_source_tree

bootstrap_source_tree()

from comb_eval.daily_eval import DEFAULT_LONG_RATIO, backtest_signal, format_signal_backtest
from combo2.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comboOpt1",
        description="Backtest one signal with the opt1 optimizer; without --config the default configuration is used",
    )
    parser.add_argument("signal", help="Signal parquet indexed by date (or (date, time))")
    parser.add_argument("--config", help="XML experiment config; its strategy/backtest/constants sections are used")
    parser.add_argument("--output-dir", help="Result directory; default <signal dir>/comboOpt1_<signal stem>")
    parser.add_argument("--cache-path", help="Parent directory containing AshareCache; default from config")
    parser.add_argument("--mosek", default=os.environ.get("MOSEKLM_LICENSE_FILE", "/root/mosek/mosek.lic"),
                        help="MOSEK license file; default $MOSEKLM_LICENSE_FILE or /root/mosek/mosek.lic")
    parser.add_argument("--start", help="Start date, e.g. 20200101")
    parser.add_argument("--end", help="End date, e.g. 20241231")
    parser.add_argument("--ti", type=int, help="Execution time HHMMSS; needed only when the signal has several")
    parser.add_argument("--long-ratio", type=float, default=DEFAULT_LONG_RATIO,
                        help="Long bucket ratio for the long-short adjustment; default 0.5")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mosek_path = Path(args.mosek).expanduser().resolve()
        if not mosek_path.is_file():
            raise FileNotFoundError(f"MOSEK license file not found: {mosek_path}")
        os.environ["MOSEKLM_LICENSE_FILE"] = str(mosek_path)
        config = load_config(args.config) if args.config else None
        result = backtest_signal(
            args.signal,
            config=config,
            cache_path=args.cache_path,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            ti=args.ti,
            long_ratio=args.long_ratio,
        )
    except Exception as exc:
        print(f"[comboOpt1] {exc}", file=sys.stderr)
        return 2
    print(format_signal_backtest(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
