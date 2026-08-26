from __future__ import annotations

import argparse
import sys
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALS_ROOT.parent
for local_path in (EVALS_ROOT, REPO_ROOT / "vendor" / "comb2-simbase"):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

import numpy as np
import pandas as pd

from comb2_simbase.cache_layout import daily_label_path
from comb2_simbase.snap_labels import load_snap_vwap_labels

from comb_eval.io import normalize_date_index, read_cache_array, read_matrix, read_table
from comb_eval.pnl import summarize_pnl_with_benchmark

from _common import frame_to_text, select_columns, write_frame, write_text

PNL_KEY_COLUMNS = [
    "pnl_m",
    "ret_pct",
    "tvr_pct",
    "ir",
    "sharpe",
    "dd_pct",
    "longonly_pnl_m",
    "longonly_ret_pct",
    "longonly_tvr_pct",
    "longonly_ir",
    "longonly_sharpe",
    "margin",
    "fitness",
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily pnl evaluation for one signal or daily pnl file.")
    parser.add_argument("path", help="Signal path by default, or daily pnl path with --input-is-pnl")
    parser.add_argument("--input-is-pnl", action="store_true", help="Treat input as an existing daily pnl dump")
    parser.add_argument("--cache-path", help="Parent directory containing AshareCache")
    parser.add_argument("--snap-ti", type=int, help="Use IntraVwap.Vwap30.HHMMSS to build the default label")
    parser.add_argument("--label", help="Explicit forward-return label path for signal -> pnl")
    parser.add_argument("--label-df-type", default="true", help="df_type passed to Memmaper2.load for label paths")
    parser.add_argument("--label-is-table", action="store_true", help="Read --label as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--booksize", type=float, default=1e7)
    parser.add_argument("--tradecost-ratio", type=float, default=1.0, help="Cost multiplier; cost = turnover * booksize * 2 * 0.003 * ratio")
    parser.add_argument("--pnlzz500", help="Optional benchmark pnl path")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    parser.add_argument("--output", help="Optional path to dump daily pnl")
    parser.add_argument("--summary-output", help="Optional path to write summary table")
    args = parser.parse_args()

    if args.input_is_pnl:
        daily_pnl = read_table(args.path, start=args.start, end=args.end)
    elif args.snap_ti is not None and not args.label:
        if not args.cache_path:
            raise ValueError("--snap-ti requires --cache-path")
        daily_pnl = calculate_daily_pnl_from_snap_ti(
            args.path,
            args.cache_path,
            args.snap_ti,
            booksize=args.booksize,
            tradecost_ratio=args.tradecost_ratio,
            start=args.start,
            end=args.end,
        )
    else:
        label_path = _resolve_label_path(args.label, args.cache_path, "vwap30_label1d")
        daily_pnl = calculate_daily_pnl(
            args.path,
            label_path,
            label_is_table=args.label_is_table,
            label_df_type=_parse_df_type(args.label_df_type),
            booksize=args.booksize,
            tradecost_ratio=args.tradecost_ratio,
            start=args.start,
            end=args.end,
        )

    if args.output:
        write_frame(daily_pnl, args.output)

    result = summarize_pnl_with_benchmark(daily_pnl, args.pnlzz500, start=args.start, end=args.end)
    text = frame_to_text(select_columns(result.table, PNL_KEY_COLUMNS))
    print(text)
    if args.summary_output:
        write_text(frame_to_text(result.table) + "\n", args.summary_output)


def _resolve_label_path(label: str | None, cache_path: str | None, label_name: str) -> Path:
    if label:
        return Path(label).expanduser()
    if not cache_path:
        raise ValueError("signal input requires --cache-path or an explicit --label")
    return daily_label_path(cache_path, label_name)


def calculate_daily_pnl(
    signal_path: str | Path,
    label_path: str | Path,
    *,
    label_is_table: bool = False,
    label_df_type: object = True,
    booksize: float = 1e7,
    tradecost_ratio: float = 0.0,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    signal = read_matrix(signal_path, start=start, end=end)
    signal.columns = signal.columns.astype(str).str.zfill(6)
    label = read_matrix(label_path, start=start, end=end) if label_is_table else _read_cache_label(label_path, signal, label_df_type)
    return _calculate_daily_pnl_from_frames(signal, label, booksize=booksize, tradecost_ratio=tradecost_ratio)


def calculate_daily_pnl_from_snap_ti(
    signal_path: str | Path,
    cache_path: str | Path,
    snap_ti: int,
    *,
    booksize: float = 1e7,
    tradecost_ratio: float = 0.0,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    signal = read_matrix(signal_path, start=start, end=end)
    signal.columns = signal.columns.astype(str).str.zfill(6)
    start_ds = signal.index.min().strftime("%Y%m%d")
    end_ds = signal.index.max().strftime("%Y%m%d")
    label = normalize_date_index(load_snap_vwap_labels(cache_path, snap_ti, start_ds, end_ds)[1])
    return _calculate_daily_pnl_from_frames(signal, label, booksize=booksize, tradecost_ratio=tradecost_ratio)


def _calculate_daily_pnl_from_frames(
    signal: pd.DataFrame,
    label: pd.DataFrame,
    *,
    booksize: float,
    tradecost_ratio: float,
) -> pd.DataFrame:
    label.columns = label.columns.astype(str).str.zfill(6)
    signal, label = signal.align(label, join="inner", axis=0)
    signal, label = signal.align(label, join="inner", axis=1)
    if signal.empty or label.empty:
        raise ValueError("No overlapping dates or instruments between signal and label.")

    x = signal.astype(float).to_numpy()
    y = label.astype(float).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    positions = _scale_to_book(np.where(valid, x, np.nan), booksize)
    long_positions = _scale_long_only(np.where(valid, x, np.nan), booksize)
    gross = np.nansum(positions * np.where(np.isfinite(y), y, np.nan), axis=1)
    long_gross = np.nansum(long_positions * np.where(np.isfinite(y), y, np.nan), axis=1)
    tradevalue = np.nansum(np.abs(np.diff(positions, axis=0, prepend=np.zeros_like(positions[:1]))), axis=1)
    long_tradevalue = np.nansum(np.abs(np.diff(long_positions, axis=0, prepend=np.zeros_like(long_positions[:1]))), axis=1)
    turnover = tradevalue / (booksize * 2)
    long_turnover = long_tradevalue / booksize
    tradecost = tradevalue * 0.003 * tradecost_ratio
    long_tradecost = long_tradevalue * 0.003 * tradecost_ratio
    pnl = gross - tradecost
    long_pnl = long_gross - long_tradecost
    long = np.nansum(np.where(positions > 0, positions, 0), axis=1)
    short = np.nansum(np.where(positions < 0, positions, 0), axis=1)
    long_count = np.sum(positions > 0, axis=1)
    short_count = np.sum(positions < 0, axis=1)
    label_count = np.sum(np.isfinite(y), axis=1).astype(float)
    label_count[label_count == 0] = np.nan
    coverage = np.sum(valid, axis=1) / label_count

    return pd.DataFrame(
        {
            "pnl": pnl,
            "pnl_gross": gross,
            "tradecost": tradecost,
            "longonly_pnl": long_pnl,
            "longonly_pnl_gross": long_gross,
            "longonly_tradecost": long_tradecost,
            "longonly_tvr_pct": long_turnover * 100,
            "tvr_pct": turnover * 100,
            "long": long,
            "short": short,
            "sh_hld": np.abs(long) + np.abs(short),
            "sh_trd": tradevalue,
            "n_long": long_count,
            "n_short": short_count,
            "coverage": coverage,
        },
        index=signal.index,
    )


def _read_cache_label(path: str | Path, signal: pd.DataFrame, df_type: object) -> pd.DataFrame:
    start_ds = signal.index.min().strftime("%Y%m%d")
    end_ds = signal.index.max().strftime("%Y%m%d")
    data = read_cache_array(path, start_ds, end_ds, df_type)
    if isinstance(data, pd.DataFrame):
        label = normalize_date_index(data).astype(float)
        label.columns = label.columns.astype(str).str.zfill(6)
        return label
    return pd.DataFrame(data, index=signal.index[: len(data)], columns=signal.columns[: data.shape[1]])


def _scale_to_book(values: np.ndarray, booksize: float) -> np.ndarray:
    positions = np.nan_to_num(values, nan=0.0).astype(float)
    long_sum = np.where(positions > 0, positions, 0).sum(axis=1)
    short_sum = -np.where(positions < 0, positions, 0).sum(axis=1)
    long_scale = np.divide(booksize, long_sum, out=np.zeros_like(long_sum), where=long_sum > 0)
    short_scale = np.divide(booksize, short_sum, out=np.zeros_like(short_sum), where=short_sum > 0)
    return np.where(positions > 0, positions * long_scale[:, None], np.where(positions < 0, positions * short_scale[:, None], 0.0))


def _scale_long_only(values: np.ndarray, booksize: float) -> np.ndarray:
    positions = np.nan_to_num(values, nan=0.0).astype(float)
    positions = np.where(positions > 0, positions, 0.0)
    long_sum = positions.sum(axis=1)
    long_scale = np.divide(booksize, long_sum, out=np.zeros_like(long_sum), where=long_sum > 0)
    return positions * long_scale[:, None]


def _parse_df_type(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


if __name__ == "__main__":
    main()
