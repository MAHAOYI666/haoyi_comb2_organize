"""All runEval modes other than the two-parquet daily run/read workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .correlation import matrix_correlation
from .exposure import compute_barra_style_exposure
from .formatting import output_dict_to_lines, output_frame_to_text
from .ic import summarize_ic
from .io import read_matrix, read_table
from .pnl import summarize_pnl_with_benchmark
from .report import PNL_KEY_COLUMNS, check_config_outputs, config_eval_to_text, run_config_evaluation, summarize_exposure


TRADING_DAYS = 250


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
        start=args.start,
        end=args.end,
        skip_deciles=args.skip_deciles,
        skip_exposure=args.skip_exposure,
    )
    print(config_eval_to_text(result))
    return 0


def run_corr(args: argparse.Namespace) -> int:
    _reject_config_for_single_mode(args, "--corr")
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
    _reject_config_for_single_mode(args, "--sim")
    daily_ic = read_table(args.sim, start=args.start, end=args.end)
    result = summarize_ic(daily_ic, start=args.start, end=args.end, normalize_names=args.normalize_names)
    text = output_frame_to_text(result.table)
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def run_pnl(args: argparse.Namespace) -> int:
    _reject_config_for_single_mode(args, "--pnl")
    daily_pnl = read_table(args.pnl, start=args.start, end=args.end)
    result = summarize_pnl_with_benchmark(daily_pnl, args.pnlzz500, start=args.start, end=args.end)
    text = output_frame_to_text(_select_columns(result.table, PNL_KEY_COLUMNS))
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def run_exposure(args: argparse.Namespace) -> int:
    _reject_config_for_single_mode(args, "--exposure")
    signal = read_matrix(args.exposure, start=args.start, end=args.end)
    results = []
    times = signal.index.hour * 10000 + signal.index.minute * 100 + signal.index.second
    for ti, frame in signal.groupby(times):
        frame = frame.copy()
        frame.index = frame.index.normalize()
        result = compute_barra_style_exposure(
            frame,
            start_ds=int(args.start) if args.start is not None else None,
            end_ds=int(args.end) if args.end is not None else None,
            mode=args.exposure_mode,
            cache_path=_resolve_cache_path(args),
        )
        seconds = (int(ti) // 10000) * 3600 + (int(ti) // 100 % 100) * 60 + int(ti) % 100
        result.index = pd.to_datetime(result.index.astype(str)) + pd.to_timedelta(seconds, unit="s")
        results.append(result)
    exposure = pd.concat(results).sort_index()
    _write_frame_if_requested(exposure, args.output)
    summary = summarize_exposure(exposure)
    text = output_frame_to_text(summary)
    print(text)
    _write_text_if_requested(text, args.summary_output)
    return 0


def run_va(args: argparse.Namespace) -> int:
    _reject_config_for_single_mode(args, "--va")
    booksize = _resolve_booksize(args)
    base = _read_va_series(args.va[0], args)
    new = _read_va_series(args.va[1], args)
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


def _reject_config_for_single_mode(args: argparse.Namespace, mode_flag: str) -> None:
    if args.config_flag or args.config:
        raise ValueError(f"{mode_flag} is a single-item mode; pass direct parquet/csv files instead of config.xml")


def _resolve_booksize(args: argparse.Namespace) -> float:
    if args.booksize is not None:
        return float(args.booksize)
    return 1e7


def _resolve_cache_path(args: argparse.Namespace) -> Path:
    if not args.cache_path:
        raise ValueError("--exposure requires --cache-path pointing to the parent directory of AshareCache")
    return Path(args.cache_path).expanduser()


def _read_va_series(path: str, args: argparse.Namespace) -> pd.Series:
    table = read_table(path, start=args.start, end=args.end)
    if args.column in table.columns:
        return table[args.column].astype(float)
    if "longonly_pnl" in table.columns:
        return table["longonly_pnl"].astype(float)
    if "pnl" in table.columns:
        return table["pnl"].astype(float)
    raise ValueError(f"column not found in daily pnl table: {args.column}")


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
    _write_frame_if_requested(pd.DataFrame(output, index=aligned.index), path)


def _write_frame_if_requested(frame: pd.DataFrame, path: str | None) -> None:
    if not path:
        return
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
