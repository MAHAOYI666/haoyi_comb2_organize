#!/root/autodl/python310fs/bin/python3
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd


ORGANIZE_ROOT = Path(__file__).resolve().parent
for local_path in (ORGANIZE_ROOT, ORGANIZE_ROOT / "evals"):
    text_path = str(local_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

from comb_eval.correlation import matrix_correlation
from comb_eval.exposure import DEFAULT_ASHARE_CACHE_PATH, compute_barra_style_exposure
from comb_eval.formatting import output_dict_to_lines, output_frame_to_text
from comb_eval.ic import summarize_ic
from comb_eval.io import normalize_date_index, read_cache_array, read_matrix, read_table
from comb_eval.pnl import summarize_pnl_with_benchmark
from comb_eval.report import (
    PNL_KEY_COLUMNS,
    calculate_daily_ic_from_signal,
    calculate_daily_pnl_from_signal,
    check_config_outputs,
    config_eval_to_text,
    run_config_evaluation,
    summarize_exposure,
)

TRADING_DAYS = 250
DEFAULT_VA_WEIGHTS = "0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2"

organize_config_module = importlib.import_module("config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="runEval",
        description="Evaluate comb2 signals, PnL, correlation, exposure, and config outputs.",
        epilog="examples: runEval config.xml | runEval --sim alpha.parquet --config config.xml | runEval --corr pos_a.parquet pos_b.parquet",
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to XML experiment config")
    parser.add_argument("--config", dest="config_flag", type=str, default=None, help="Path to XML experiment config")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overall", action="store_true", help="Run the full config evaluation; default when no mode is given")
    mode.add_argument("--corr", nargs=2, metavar=("LEFT", "RIGHT"), help="Summarize daily matrix correlation for two parquet/csv matrices")
    mode.add_argument("--sim", metavar="PATH", help="Summarize signal IC; input is a signal matrix unless --input-is-ic is set")
    mode.add_argument("--pnl", metavar="PATH", help="Summarize PnL; input is a signal matrix unless --input-is-pnl is set")
    mode.add_argument("--exposure", metavar="PATH", help="Summarize Barra style exposure for a signal matrix")
    mode.add_argument("--va", nargs=2, metavar=("BASE", "NEW"), help="Summarize value-add for base/new PnL files or signal matrices")

    parser.add_argument("--report-dir", help="Directory for eval artifacts; defaults to <output_root>/eval_report")
    parser.add_argument("--plot-output", help="Path for the signal analysis long image")
    parser.add_argument("--pnlzz500", help="Optional benchmark pnl path")
    parser.add_argument("--label", "--label-1d", dest="label", help="1d forward-return label path for PNL/IC/decile calculation")
    parser.add_argument("--label-5d", help="5d forward-return label path for IC calculation")
    parser.add_argument("--label-is-table", "--label-1d-is-table", dest="label_is_table", action="store_true", help="Read --label as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--label-5d-is-table", action="store_true", help="Read --label-5d as csv/tsv/parquet instead of Memmaper2 cache")
    parser.add_argument("--label-df-type", default="true", help="df_type passed to Memmaper2.load for label paths")
    parser.add_argument("--booksize", type=float, help="Booksize for signal-to-PnL evaluation")
    parser.add_argument("--tradecost-ratio", type=float, help="Cost multiplier; cost = tradevalue * 0.003 * ratio")
    parser.add_argument("--input-is-ic", action="store_true", help="For --sim, treat PATH as an existing daily IC table")
    parser.add_argument("--input-is-pnl", action="store_true", help="For --pnl, treat PATH as an existing daily PnL table")
    parser.add_argument("--normalize-names", action="store_true", help="For --sim --input-is-ic, map ic/5dic to 1d_IC/5d_IC names")
    parser.add_argument("--corr-days", type=int, help="For --corr, use only the most recent N overlapping dates")
    parser.add_argument("--min-valid", type=int, default=1000, help="For --corr, minimum nonzero overlapping instruments per day")
    parser.add_argument("--top-pct", type=float, default=10.0, help="For --corr, long-holding overlap top percentage; default 10")
    parser.add_argument("--ashare-cache-path", help="For --exposure, AshareCache root; defaults to --config or built-in default")
    parser.add_argument("--exposure-mode", type=int, choices=(0, 1), default=0, help="For --exposure, 0=cross-sectional correlation, 1=beta")
    parser.add_argument("--column", default="longonly_pnl", help="For --va, PnL column to compare; signal inputs are converted to daily PnL first")
    parser.add_argument("--weights", default=DEFAULT_VA_WEIGHTS, help="For --va, comma-separated new-signal blend weights")
    parser.add_argument("--output", help="Optional path for mode-specific generated table, such as daily IC, daily PnL, exposure, or VA csv")
    parser.add_argument("--summary-output", help="Optional path to write the formatted summary text")
    parser.add_argument("--blended-output", help="For --va, optional path to write blended and incremental daily series")
    parser.add_argument("--skip-deciles", action="store_true", help="Skip 10-group backtests")
    parser.add_argument("--skip-exposure", action="store_true", help="Skip Barra exposure analysis")
    parser.add_argument("--start", help="Start date, e.g. 20160101")
    parser.add_argument("--end", help="End date, e.g. 20240101")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
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


def run_overall(args: argparse.Namespace) -> int:
    config_path = args.config_flag or args.config
    if not config_path:
        print("[EVAL] missing config.xml; pass a positional config or --config", file=sys.stderr)
        return 2
    if not Path(config_path).expanduser().is_file():
        print(f"[EVAL] config file not found: {config_path}", file=sys.stderr)
        return 2

    check = check_config_outputs(config_path)
    if not check.ok:
        print("[EVAL] 输出文件不齐")
        print(f"[config] {check.config_path}")
        print(f"[output_root] {check.output_root}")
        print("[missing]")
        for item in check.missing:
            print(f"- {item.name}: {item.path} ({item.reason})")
        return 2

    print("[EVAL] 输出文件齐全")
    for item in check.files:
        print(f"- {item.name}: {item.path}")

    result = run_config_evaluation(
        config_path,
        report_dir=args.report_dir,
        plot_path=args.plot_output,
        pnlzz500_path=args.pnlzz500,
        label_path=args.label,
        label_5d_path=args.label_5d,
        label_is_table=args.label_is_table,
        label_5d_is_table=args.label_5d_is_table,
        label_df_type=_parse_df_type(args.label_df_type),
        booksize=args.booksize,
        tradecost_ratio=args.tradecost_ratio,
        start=args.start,
        end=args.end,
        skip_deciles=args.skip_deciles,
        skip_exposure=args.skip_exposure,
    )
    print(config_eval_to_text(result))
    return 0


def run_corr(args: argparse.Namespace) -> int:
    result = matrix_correlation(
        args.corr[0],
        args.corr[1],
        start=args.start,
        end=args.end,
        corr_days=args.corr_days,
        min_valid=args.min_valid,
        top_pct=args.top_pct,
    )
    text = output_dict_to_lines(result)
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def run_sim(args: argparse.Namespace) -> int:
    if args.input_is_ic:
        daily_ic = read_table(args.sim, start=args.start, end=args.end)
        normalize_names = args.normalize_names
    else:
        config = _load_optional_config(args)
        signal = read_matrix(args.sim, start=args.start, end=args.end)
        label_1d_path = _resolve_label_path(args, config, label_5d=False)
        label_5d_path = _resolve_label_path(args, config, label_5d=True)
        label_1d = _read_label_for_signal(
            signal,
            label_1d_path,
            is_table=args.label_is_table,
            df_type=_parse_df_type(args.label_df_type),
        )
        label_5d = _read_label_for_signal(
            signal,
            label_5d_path,
            is_table=args.label_5d_is_table,
            df_type=_parse_df_type(args.label_df_type),
        )
        daily_ic = calculate_daily_ic_from_signal(signal, label_1d, label_5d)
        normalize_names = True
        _write_frame_if_requested(daily_ic, args.output)

    result = summarize_ic(daily_ic, start=args.start, end=args.end, normalize_names=normalize_names)
    text = output_frame_to_text(result.table)
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def run_pnl(args: argparse.Namespace) -> int:
    if args.input_is_pnl:
        daily_pnl = read_table(args.pnl, start=args.start, end=args.end)
    else:
        config = _load_optional_config(args)
        signal = read_matrix(args.pnl, start=args.start, end=args.end)
        label_path = _resolve_label_path(args, config, label_5d=False)
        label = _read_label_for_signal(
            signal,
            label_path,
            is_table=args.label_is_table,
            df_type=_parse_df_type(args.label_df_type),
        )
        daily_pnl = calculate_daily_pnl_from_signal(
            signal,
            label,
            booksize=_resolve_booksize(args, config),
            tradecost_ratio=_resolve_tradecost_ratio(args, config, default=1.0),
        )
        _write_frame_if_requested(daily_pnl, args.output)

    result = summarize_pnl_with_benchmark(daily_pnl, args.pnlzz500, start=args.start, end=args.end)
    text = output_frame_to_text(_select_columns(result.table, PNL_KEY_COLUMNS))
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def run_exposure(args: argparse.Namespace) -> int:
    config = _load_optional_config(args)
    signal = read_matrix(args.exposure, start=args.start, end=args.end)
    exposure = compute_barra_style_exposure(
        signal,
        start_ds=int(args.start) if args.start is not None else None,
        end_ds=int(args.end) if args.end is not None else None,
        mode=args.exposure_mode,
        ashare_cache_path=_resolve_ashare_cache_path(args, config),
    )
    _write_frame_if_requested(exposure, args.output)
    summary = summarize_exposure(exposure)
    text = output_frame_to_text(summary)
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def run_va(args: argparse.Namespace) -> int:
    config = _load_optional_config(args)
    booksize = _resolve_booksize(args, config)
    tradecost_ratio = _resolve_tradecost_ratio(args, config, default=0.0)
    base = _read_va_series(args.va[0], args, config, booksize=booksize, tradecost_ratio=tradecost_ratio)
    new = _read_va_series(args.va[1], args, config, booksize=booksize, tradecost_ratio=tradecost_ratio)
    aligned = pd.concat([base.rename("base"), new.rename("new")], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN observations for value-add calculation.")

    weights = _parse_weights(args.weights)
    table = _longonly_va_table(aligned, weights, booksize)
    _write_frame_if_requested(table, args.output)
    _write_blended_output(aligned, weights, args.blended_output)
    text = output_frame_to_text(table)
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def _load_optional_config(args: argparse.Namespace) -> dict | None:
    config_path = args.config_flag or args.config
    if not config_path:
        return None
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    return organize_config_module.load_config(str(path))


def _resolve_label_path(args: argparse.Namespace, config: dict | None, *, label_5d: bool) -> Path:
    raw = args.label_5d if label_5d else args.label
    if raw:
        return Path(raw).expanduser()
    if config is None:
        flag = "--label-5d" if label_5d else "--label"
        raise ValueError(f"{flag} is required unless --config supplies combo.loader.ashare_data_path")
    cache_root = Path(config["combo"]["loader"]["ashare_data_path"]).expanduser()
    suffix = "5d" if label_5d else "1d"
    return cache_root / "1d_DailyLabel" / f"DailyLabel.vwap30_label{suffix}"


def _read_label_for_signal(signal: pd.DataFrame, path: Path, *, is_table: bool, df_type: object) -> pd.DataFrame:
    if is_table:
        return read_matrix(path)
    normalized_signal = normalize_date_index(signal)
    start_ds = normalized_signal.index.min().strftime("%Y%m%d")
    end_ds = normalized_signal.index.max().strftime("%Y%m%d")
    data = read_cache_array(path, start_ds, end_ds, df_type)
    if isinstance(data, pd.DataFrame):
        label = normalize_date_index(data).astype(float)
        label.columns = label.columns.astype(str).str.zfill(6)
        return label
    return pd.DataFrame(data, index=normalized_signal.index[: len(data)], columns=normalized_signal.columns[: data.shape[1]]).astype(float)


def _resolve_booksize(args: argparse.Namespace, config: dict | None) -> float:
    if args.booksize is not None:
        return float(args.booksize)
    if config is not None:
        return float(config.get("backtest", {}).get("cash", 1e7))
    return 1e7


def _resolve_tradecost_ratio(args: argparse.Namespace, config: dict | None, *, default: float) -> float:
    if args.tradecost_ratio is not None:
        return float(args.tradecost_ratio)
    if config is not None:
        fee_rate = float(config.get("backtest", {}).get("fee_rate", 0.0))
        return fee_rate / 0.003 if fee_rate else 0.0
    return float(default)


def _resolve_ashare_cache_path(args: argparse.Namespace, config: dict | None) -> str | Path:
    if args.ashare_cache_path:
        return Path(args.ashare_cache_path).expanduser()
    if config is not None:
        return Path(config["combo"]["loader"]["ashare_data_path"]).expanduser()
    return DEFAULT_ASHARE_CACHE_PATH


def _read_va_series(
    path: str,
    args: argparse.Namespace,
    config: dict | None,
    *,
    booksize: float,
    tradecost_ratio: float,
) -> pd.Series:
    table = read_table(path, start=args.start, end=args.end)
    if args.column in table.columns:
        return table[args.column].astype(float)
    if "longonly_pnl" in table.columns:
        return table["longonly_pnl"].astype(float)
    if "pnl" in table.columns:
        return table["pnl"].astype(float)

    label_path = _resolve_label_path(args, config, label_5d=False)
    signal = read_matrix(path, start=args.start, end=args.end)
    label = _read_label_for_signal(
        signal,
        label_path,
        is_table=args.label_is_table,
        df_type=_parse_df_type(args.label_df_type),
    )
    daily_pnl = calculate_daily_pnl_from_signal(
        signal,
        label,
        booksize=booksize,
        tradecost_ratio=tradecost_ratio,
    )
    if args.column in daily_pnl.columns:
        return daily_pnl[args.column].astype(float)
    raise ValueError(f"column not found after signal-to-pnl conversion: {args.column}")


def _parse_weights(value: str) -> list[float]:
    weights = [float(item) for item in value.split(",") if item.strip()]
    if not weights:
        raise ValueError("--weights must contain at least one numeric value")
    return weights


def _longonly_va_table(aligned: pd.DataFrame, weights: list[float], booksize: float) -> pd.DataFrame:
    rows = []
    for period, group in _period_groups(aligned):
        row = {"period": period}
        row["0.00"] = _annualized_ret_pct(group["base"], booksize)
        for weight in weights:
            incremental = weight * group["new"] + (1 - weight) * group["base"] - group["base"]
            row[f"{weight:.2f}"] = _annualized_ret_pct(incremental, booksize)
        row["1.00"] = _annualized_ret_pct(group["new"], booksize)
        rows.append(row)
    return pd.DataFrame(rows).set_index("period")


def _period_groups(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [(str(year), group) for year, group in df.groupby(df.index.year)]
    groups.append(("full", df))
    return groups


def _annualized_ret_pct(series: pd.Series, booksize: float) -> float:
    return float(series.astype(float).mean() / booksize * TRADING_DAYS * 100)


def _write_blended_output(aligned: pd.DataFrame, weights: list[float], path: str | None) -> None:
    if not path:
        return
    output = {}
    for weight in weights:
        blend = weight * aligned["new"] + (1 - weight) * aligned["base"]
        output[f"blend_{weight:g}"] = blend
        output[f"incremental_{weight:g}"] = blend - aligned["base"]
    _write_frame(pd.DataFrame(output, index=aligned.index), path)


def _write_frame_if_requested(frame: pd.DataFrame, path: str | None) -> None:
    if path:
        _write_frame(frame, path)


def _write_frame(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path)
    else:
        frame.to_csv(output_path)


def _write_text_if_requested(text: str, path: str | None) -> None:
    if not path:
        return
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]]


def _parse_df_type(value: str) -> object:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


if __name__ == "__main__":
    raise SystemExit(main())
