from __future__ import annotations

import argparse
from pathlib import Path

from .constraints import equalize_short_side, trading_limit_exposure, universe_coverage
from .correlation import matrix_correlation, pnl_correlation, pnl_pool_correlation
from .evaluator import run_evaluation
from .ic import summarize_cache_ic, summarize_ic
from .pnl import summarize_pnl, summarize_pnl_with_benchmark
from .value_add import netting_return, pool_value_added, value_added


def main() -> None:
    parser = argparse.ArgumentParser(prog="comb-eval", description="Local modular alpha evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    matrix_corr_parser.add_argument("--corr-days", type=int, help="Use only the most recent N overlapping days")
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

    eval_parser = subparsers.add_parser("eval", help="Run local checks over available files")
    eval_parser.add_argument("--pnl")
    eval_parser.add_argument("--pnlzz500")
    eval_parser.add_argument("--ic")
    _add_date_args(eval_parser)

    args = parser.parse_args()
    if args.command == "pnl":
        result = summarize_pnl_with_benchmark(args.path, args.pnlzz500, start=args.start, end=args.end)
        print(result.table.to_string())
    elif args.command == "ic":
        print(summarize_ic(args.path, start=args.start, end=args.end, normalize_names=args.normalize_names).table.to_string())
    elif args.command == "cache-ic":
        print(summarize_cache_ic(args.path, args.start_ds, args.end_ds, args.df_type).table.to_string())
    elif args.command == "corr":
        if len(args.pool) == 1:
            print(_dict_to_lines(pnl_correlation(args.candidate, args.pool[0], column=args.column, start=args.start, end=args.end)))
        else:
            table = pnl_pool_correlation(args.candidate, args.pool, column=args.column, top=args.top, start=args.start, end=args.end)
            print(table.to_string())
            if "summary" in table.attrs:
                print("\n[summary]")
                print(table.attrs["summary"].to_string())
    elif args.command == "matrix-corr":
        print(_dict_to_lines(matrix_correlation(args.candidate, args.pool, start=args.start, end=args.end, corr_days=args.corr_days, min_valid=args.min_valid)))
    elif args.command == "value-add":
        if len(args.pool) == 1:
            print(_dict_to_lines(value_added(args.candidate, args.pool[0], column=args.column, start=args.start, end=args.end)))
        else:
            table = pool_value_added(args.candidate, args.pool, column=args.column, start=args.start, end=args.end)
            print(table.to_string())
            if "summary" in table.attrs:
                print("\n[summary]")
                print(table.attrs["summary"].to_string())
    elif args.command == "netting":
        print(_dict_to_lines(netting_return(args.candidate, args.pool, column=args.column, start=args.start, end=args.end)))
    elif args.command == "equalize-short":
        equalize_short_side(args.alpha, percentile=args.percentile, group=args.group, start=args.start, end=args.end).to_parquet(args.output)
    elif args.command == "universe":
        print(universe_coverage(args.alpha, args.mask, start=args.start, end=args.end).to_string())
    elif args.command == "trade-limit":
        print(trading_limit_exposure(args.trades, args.mask, start=args.start, end=args.end).to_string())
    elif args.command == "eval":
        outputs = run_evaluation(args.pnl, args.ic, pnlzz500_path=args.pnlzz500, start=args.start, end=args.end)
        for module, tables in outputs.items():
            print(f"\n[{module}.summary]")
            print(tables["summary"].to_string())
            print(f"\n[{module}.checks]")
            print(tables["checks"].to_string())


def _add_date_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")


def _dict_to_lines(values: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in values.items())


if __name__ == "__main__":
    main()
