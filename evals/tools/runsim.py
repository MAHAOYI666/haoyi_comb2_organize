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

from comb_eval.ic import summarize_ic
from comb_eval.io import normalize_date_index, read_cache_array, read_matrix, read_table

from _common import frame_to_text, write_frame, write_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Run local signal IC evaluation and summarize the result.")
    parser.add_argument("path", help="Signal path by default, or daily IC path with --input-is-ic")
    parser.add_argument("--input-is-ic", action="store_true", help="Treat input as an existing daily IC dump")
    parser.add_argument("--cache-path", help="Parent directory containing AshareCache")
    parser.add_argument("--label-1d", help="Explicit 1d forward-return label path")
    parser.add_argument("--label-5d", help="Explicit 5d forward-return label path")
    parser.add_argument("--label-df-type", default="true", help="df_type passed to Memmaper2.load for label paths")
    parser.add_argument("--label-1d-is-table", action="store_true", help="Read --label-1d as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--label-5d-is-table", action="store_true", help="Read --label-5d as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    parser.add_argument("--normalize-names", action="store_true", help="Map ic/5dic to 1d_IC/5d_IC style names for --input-is-ic")
    parser.add_argument("--output", help="Optional path to dump daily IC; defaults to <signal_dir>/daily_ic for signal input")
    parser.add_argument("--no-output", action="store_true", help="Do not dump daily IC for signal input")
    parser.add_argument("--summary-output", help="Optional path to write summary table")
    args = parser.parse_args()

    if args.input_is_ic:
        daily_ic = read_table(args.path, start=args.start, end=args.end)
        normalize_names = args.normalize_names
    else:
        label_1d_path = _resolve_label_path(args.label_1d, args.cache_path, "vwap30_label1d")
        label_5d_path = _resolve_label_path(args.label_5d, args.cache_path, "vwap30_label5d")
        daily_ic = calculate_daily_ic(
            args.path,
            label_1d_path,
            label_5d_path,
            label_1d_is_table=args.label_1d_is_table,
            label_5d_is_table=args.label_5d_is_table,
            label_df_type=_parse_df_type(args.label_df_type),
            start=args.start,
            end=args.end,
        )
        normalize_names = True

    if args.output:
        write_frame(daily_ic, args.output)
    elif not args.input_is_ic and not args.no_output:
        write_frame(daily_ic, Path(args.path).parent / "daily_ic")

    result = summarize_ic(daily_ic, start=args.start, end=args.end, normalize_names=normalize_names)
    text = frame_to_text(result.table)
    print(text)
    if args.summary_output:
        write_text(text + "\n", args.summary_output)


def _resolve_label_path(label: str | None, cache_path: str | None, label_name: str) -> Path:
    if label:
        return Path(label).expanduser()
    if not cache_path:
        raise ValueError(f"signal input requires --cache-path or an explicit --label-{label_name[-2:]}")
    return daily_label_path(cache_path, label_name)


def calculate_daily_ic(
    signal_path: str | Path,
    label_1d_path: str | Path,
    label_5d_path: str | Path,
    *,
    label_1d_is_table: bool = False,
    label_5d_is_table: bool = False,
    label_df_type: object = True,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    signal = read_matrix(signal_path, start=start, end=end)
    label_1d = _read_label(label_1d_path, signal, label_1d_is_table, label_df_type, start, end)
    label_5d = _read_label(label_5d_path, signal, label_5d_is_table, label_df_type, start, end)

    signal, label_1d = signal.align(label_1d, join="inner", axis=0)
    signal, label_1d = signal.align(label_1d, join="inner", axis=1)
    signal, label_5d = signal.align(label_5d, join="inner", axis=0)
    signal, label_5d = signal.align(label_5d, join="inner", axis=1)
    signal, label_1d = signal.align(label_1d, join="inner", axis=0)
    signal, label_1d = signal.align(label_1d, join="inner", axis=1)
    label_5d = label_5d.reindex(index=signal.index, columns=signal.columns)

    if signal.empty or label_1d.empty or label_5d.empty:
        raise ValueError("No overlapping dates or instruments between signal and labels.")

    x = signal.astype(float).to_numpy()
    y1 = label_1d.astype(float).to_numpy()
    y5 = label_5d.astype(float).to_numpy()
    valid_1d = np.isfinite(x) & np.isfinite(y1)
    label_count = np.sum(np.isfinite(y1), axis=1).astype(float)
    label_count[label_count == 0] = np.nan

    daily_ic = pd.DataFrame(
        {
            "ic": _row_corr(x, y1),
            "5dic": _row_corr(x, y5),
            "rankic": _row_corr(_row_rank(x), _row_rank(y1)),
            "percic": _row_corr(_row_percentile(x), _row_rank(y1)),
            "coverage": np.sum(valid_1d, axis=1) / label_count,
        },
        index=signal.index,
    )
    return daily_ic


def _read_label(
    path: str | Path,
    signal: pd.DataFrame,
    is_table: bool,
    df_type: object,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    if is_table:
        return read_matrix(path, start=start, end=end)

    start_ds = signal.index.min().strftime("%Y%m%d")
    end_ds = signal.index.max().strftime("%Y%m%d")
    data = read_cache_array(path, start_ds, end_ds, df_type)
    if isinstance(data, pd.DataFrame):
        return normalize_date_index(data).astype(float)
    return pd.DataFrame(data, index=signal.index[: len(data)], columns=signal.columns[: data.shape[1]]).astype(float)


def _row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1)
    left_masked = np.where(valid, left, np.nan)
    right_masked = np.where(valid, right, np.nan)
    left_mean = np.nanmean(left_masked, axis=1, keepdims=True)
    right_mean = np.nanmean(right_masked, axis=1, keepdims=True)
    left_centered = np.where(valid, left - left_mean, 0.0)
    right_centered = np.where(valid, right - right_mean, 0.0)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered * left_centered, axis=1) * np.sum(right_centered * right_centered, axis=1))
    corr = np.divide(numerator, denominator, out=np.full(left.shape[0], np.nan), where=(count >= 2) & (denominator > 0))
    return corr


def _row_rank(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(axis=1).to_numpy()


def _row_percentile(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(axis=1, pct=True).to_numpy()


def _parse_df_type(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


if __name__ == "__main__":
    main()
