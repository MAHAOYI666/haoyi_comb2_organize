from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from comb_eval.io import read_table
from rundailypnl import DEFAULT_LABEL_PATH, _parse_df_type, calculate_daily_pnl

from _common import frame_to_text, write_frame, write_text

DEFAULT_WEIGHTS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]
TRADING_DAYS = 250
LONGONLY_COLUMN = "longonly_pnl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute value-add between base and new pnl files.")
    parser.add_argument("base", help="Base pnl path")
    parser.add_argument("new", help="New pnl path")
    parser.add_argument("--column", default="pnl", help="PnL column name")
    parser.add_argument("--weights", default=",".join(str(x) for x in DEFAULT_WEIGHTS), help="Comma-separated blend weights for new")
    parser.add_argument("--label", default=DEFAULT_LABEL_PATH, help="Forward-return label path when inputs are signal files")
    parser.add_argument("--label-df-type", default="true", help="df_type passed to Memmaper2.load for label paths")
    parser.add_argument("--label-is-table", action="store_true", help="Read --label as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--booksize", type=float, default=1e7, help="Booksize when converting signal inputs to pnl")
    parser.add_argument("--tradecost-ratio", type=float, default=0.0, help="Cost multiplier when converting signal inputs to pnl")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    parser.add_argument("--output", help="Optional path to write VA table")
    parser.add_argument("--blended-output", help="Optional path to write blended pnl series")
    args = parser.parse_args()

    base = _read_pnl_or_signal(
        args.base,
        args.column,
        label=args.label,
        label_is_table=args.label_is_table,
        label_df_type=_parse_df_type(args.label_df_type),
        booksize=args.booksize,
        tradecost_ratio=args.tradecost_ratio,
        start=args.start,
        end=args.end,
    )
    new = _read_pnl_or_signal(
        args.new,
        args.column,
        label=args.label,
        label_is_table=args.label_is_table,
        label_df_type=_parse_df_type(args.label_df_type),
        booksize=args.booksize,
        tradecost_ratio=args.tradecost_ratio,
        start=args.start,
        end=args.end,
    )
    aligned = _align_longonly_pnl(base, new)
    weights = [float(value) for value in args.weights.split(",") if value]
    table = _longonly_va_table(aligned, weights).round(2)
    text = frame_to_text(table)
    print(text)
    if args.output:
        write_text(text + "\n", args.output)
    if args.blended_output:
        blended_series = {}
        for weight in weights:
            blended_series[f"blend_{weight:g}"] = weight * aligned["new"] + (1 - weight) * aligned["base"]
            blended_series[f"incremental_{weight:g}"] = blended_series[f"blend_{weight:g}"] - aligned["base"]
        write_frame(pd.DataFrame(blended_series, index=aligned.index), args.blended_output)


def _read_pnl_or_signal(
    path: str,
    column: str,
    *,
    label: str,
    label_is_table: bool,
    label_df_type: object,
    booksize: float,
    tradecost_ratio: float,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    table = read_table(path, start=start, end=end)
    if column in table.columns:
        return table
    return calculate_daily_pnl(
        path,
        label,
        label_is_table=label_is_table,
        label_df_type=label_df_type,
        booksize=booksize,
        tradecost_ratio=tradecost_ratio,
        start=start,
        end=end,
    )


def _align_longonly_pnl(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if LONGONLY_COLUMN not in base.columns or LONGONLY_COLUMN not in new.columns:
        raise ValueError(f"Both inputs must contain {LONGONLY_COLUMN}; signal inputs are converted automatically by rundailypnl.")
    aligned = pd.concat([base[LONGONLY_COLUMN].rename("base"), new[LONGONLY_COLUMN].rename("new")], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN long-only pnl observations for value-add calculation.")
    return aligned



def _longonly_va_table(aligned: pd.DataFrame, weights: list[float]) -> pd.DataFrame:
    output_weights = [0.0, *weights, 1.0]
    rows = []
    for period, group in _period_groups(aligned):
        row = {"period": period}
        for weight in output_weights:
            if weight == 0.0:
                series = group["base"]
            elif weight == 1.0:
                series = group["new"]
            else:
                series = weight * group["new"] + (1 - weight) * group["base"] - group["base"]
            row[f"{weight:.2f}"] = _annualized_longonly_ret_pct(series)
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")



def _period_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [(str(year), group) for year, group in df.groupby(df.index.year)]
    groups.append(("full", df))
    return groups



def _annualized_longonly_ret_pct(series: pd.Series) -> float:
    return float(series.mean() / 1e7 * TRADING_DAYS * 100)


if __name__ == "__main__":
    main()
