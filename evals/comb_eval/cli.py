from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .constraints import equalize_short_side, trading_limit_exposure, universe_coverage
from .correlation import matrix_correlation, pnl_correlation, pnl_pool_correlation
from .evaluator import run_evaluation
from .formatting import output_dict_to_lines, output_frame_to_text
from .ic import summarize_cache_ic, summarize_ic
from .pnl import summarize_pnl, summarize_pnl_with_benchmark
from .report import config_eval_to_text, run_config_evaluation
from .value_add import netting_return, pool_value_added, value_added


def main() -> int:
    parser = argparse.ArgumentParser(prog="comb-eval", description="Local modular alpha evaluation.")
    subparsers = parser.add_subparsers(dest="command")

    pnl_parser = subparsers.add_parser("pnl", help="Summarize a local pnl file")
    pnl_parser.add_argument("path")
    pnl_parser.add_argument("--pnlzz500", help="Optional benchmark pnl for long/zz500 ret")
    _add_date_args(pnl_parser)

    ic_parser = subparsers.add_parser("ic", help="Summarize a local daily IC file")
    ic_parser.add_argument("path")
    ic_parser.add_argument("--normalize-names", action="store_true", help="Map ic/5dic to 1d_IC/5d_IC style names")
    _add_date_args(ic_parser)

    cache_ic_parser = subparsers.add_parser("cache-ic", help="Summarize an IC-like table from AshareCache via Memmaper2")
    cache_ic_parser.add_argument("path")
    cache_ic_parser.add_argument("start_ds")
    cache_ic_parser.add_argument("end_ds")
    cache_ic_parser.add_argument("df_type")

    corr_parser = subparsers.add_parser("corr", help="Compute candidate vs pool pnl correlation")
    corr_parser.add_argument("candidate")
    corr_parser.add_argument("pool", nargs="+")
    corr_parser.add_argument("--column", default="pnl")
    corr_parser.add_argument("--top", type=int, default=5)
    _add_date_args(corr_parser)

    matrix_corr_parser = subparsers.add_parser("matrix-corr", help="Compute position/trade matrix correlation")
    matrix_corr_parser.add_argument("candidate")
    matrix_corr_parser.add_argument("pool")
    matrix_corr_parser.add_argument("--corr-days", type=int, default=240, help="Use only the most recent N overlapping days; default 240")
    matrix_corr_parser.add_argument("--min-valid", type=int, default=2, help="Minimum nonzero overlapping instruments per day")
    _add_date_args(matrix_corr_parser)

    va_parser = subparsers.add_parser("value-add", help="Compute guidance-style value added IR")
    va_parser.add_argument("candidate")
    va_parser.add_argument("pool", nargs="+")
    va_parser.add_argument("--column", default="pnl")
    _add_date_args(va_parser)

    netting_parser = subparsers.add_parser("netting", help="Regress candidate pnl against pool pnl and summarize residual")
    netting_parser.add_argument("candidate")
    netting_parser.add_argument("pool")
    netting_parser.add_argument("--column", default="pnl")
    _add_date_args(netting_parser)

    equalize_parser = subparsers.add_parser("equalize-short", help="Apply local OpPercentile2-style short-side equalization")
    equalize_parser.add_argument("alpha")
    equalize_parser.add_argument("output")
    equalize_parser.add_argument("--percentile", type=float, default=0.5)
    equalize_parser.add_argument("--group")
    _add_date_args(equalize_parser)

    universe_parser = subparsers.add_parser("universe", help="Measure alpha coverage inside a universe mask")
    universe_parser.add_argument("alpha")
    universe_parser.add_argument("mask")
    _add_date_args(universe_parser)

    limit_parser = subparsers.add_parser("trade-limit", help="Measure trade exposure blocked by trading-limit mask")
    limit_parser.add_argument("trades")
    limit_parser.add_argument("mask")
    _add_date_args(limit_parser)

    eval_parser = subparsers.add_parser("eval", help="Run local checks over available files or a config output")
    eval_parser.add_argument("--config", help="Run a full evaluation report from a comb2-organize XML config")
    eval_parser.add_argument("--report-dir", help="Directory for config eval artifacts; defaults to <output_root>/eval_report")
    eval_parser.add_argument("--plot-output", help="Path for the signal analysis long image")
    eval_parser.add_argument("--pnl")
    eval_parser.add_argument("--pnlzz500")
    eval_parser.add_argument("--ic")
    eval_parser.add_argument("--skip-deciles", action="store_true", help="Skip 10-group backtests")
    eval_parser.add_argument("--skip-exposure", action="store_true", help="Skip Barra exposure analysis")
    _add_date_args(eval_parser)

    args = parser.parse_args()
    if args.command is None:
        parser.print_usage(sys.stderr)
        print("comb-eval: missing command; run comb-eval -h for available commands", file=sys.stderr)
        return 2

    if args.command == "pnl":
        result = summarize_pnl_with_benchmark(args.path, args.pnlzz500, start=args.start, end=args.end)
        print(output_frame_to_text(result.table))
    elif args.command == "ic":
        print(output_frame_to_text(summarize_ic(args.path, start=args.start, end=args.end, normalize_names=args.normalize_names).table))
    elif args.command == "cache-ic":
        print(output_frame_to_text(summarize_cache_ic(args.path, args.start_ds, args.end_ds, args.df_type).table))
    elif args.command == "corr":
        if len(args.pool) == 1:
            print(output_dict_to_lines(pnl_correlation(args.candidate, args.pool[0], column=args.column, start=args.start, end=args.end)))
        else:
            table = pnl_pool_correlation(args.candidate, args.pool, column=args.column, top=args.top, start=args.start, end=args.end)
            print(output_frame_to_text(table))
            if "summary" in table.attrs:
                print("\n[summary]")
                print(output_frame_to_text(table.attrs["summary"]))
    elif args.command == "matrix-corr":
        print(output_dict_to_lines(matrix_correlation(args.candidate, args.pool, start=args.start, end=args.end, corr_days=args.corr_days, min_valid=args.min_valid)))
    elif args.command == "value-add":
        if len(args.pool) == 1:
            print(output_dict_to_lines(value_added(args.candidate, args.pool[0], column=args.column, start=args.start, end=args.end)))
        else:
            table = pool_value_added(args.candidate, args.pool, column=args.column, start=args.start, end=args.end)
            print(output_frame_to_text(table))
            if "summary" in table.attrs:
                print("\n[summary]")
                print(output_frame_to_text(table.attrs["summary"]))
    elif args.command == "netting":
        print(output_dict_to_lines(netting_return(args.candidate, args.pool, column=args.column, start=args.start, end=args.end)))
    elif args.command == "equalize-short":
        equalize_short_side(args.alpha, percentile=args.percentile, group=args.group, start=args.start, end=args.end).to_parquet(args.output)
    elif args.command == "universe":
        print(output_frame_to_text(universe_coverage(args.alpha, args.mask, start=args.start, end=args.end)))
    elif args.command == "trade-limit":
        print(output_frame_to_text(trading_limit_exposure(args.trades, args.mask, start=args.start, end=args.end)))
    elif args.command == "eval":
        if args.config:
            result = run_config_evaluation(
                args.config,
                report_dir=args.report_dir,
                plot_path=args.plot_output,
                pnlzz500_path=args.pnlzz500,
                start=args.start,
                end=args.end,
                skip_deciles=args.skip_deciles,
                skip_exposure=args.skip_exposure,
            )
            print(config_eval_to_text(result))
            return 0
        if not args.pnl and not args.ic:
            print("comb-eval eval: missing input; pass --config or at least --pnl/--ic", file=sys.stderr)
            return 2
        outputs = run_evaluation(args.pnl, args.ic, pnlzz500_path=args.pnlzz500, start=args.start, end=args.end)
        for module, tables in outputs.items():
            print(f"\n[{module}.summary]")
            print(output_frame_to_text(tables["summary"]))
            print(f"\n[{module}.checks]")
            print(output_frame_to_text(tables["checks"]))
    return 0


def _add_date_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")


def _parse_df_type(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


if __name__ == "__main__":
    raise SystemExit(main())
