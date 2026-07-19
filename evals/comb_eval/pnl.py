from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_table
from .schemas import MetricResult, TRADING_DAYS


PERFORMANCE_COLUMNS = (
    "ret_pct",
    "longonly_ret_pct",
    "ir",
    "longonly_ir",
    "sharpe",
    "longonly_sharpe",
)


def sample_ir(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return float(values.mean() / std)


def drawdown_from_pnl(pnl: pd.Series, exposure: float) -> float:
    if exposure == 0 or np.isnan(exposure):
        return np.nan
    running = 0.0
    worst = 0.0
    for value in pnl.fillna(0).astype(float):
        running += value
        if running >= 0:
            running = 0.0
        worst = min(worst, running)
    return abs(worst) / exposure * 100


def period_groups(df: pd.DataFrame, include_all: bool = True) -> list[tuple[str, pd.DataFrame]]:
    groups: list[tuple[str, pd.DataFrame]] = []
    for _, group in df.groupby(df.index.year):
        groups.append((period_label(group), group))
    if include_all:
        groups.append(("ALL", df))
    return groups


def period_label(df: pd.DataFrame) -> str:
    return f"{df.index.min().strftime('%Y%m%d')}-{df.index.max().strftime('%Y%m%d')}"


def normalize_pnl_columns(df: pd.DataFrame) -> pd.DataFrame:
    original_columns = set(df.columns)
    rename = {
        "Date": "date",
        "PNL": "pnl",
        "Long": "long",
        "Short": "short",
        "Return": "ret",
        "return": "ret",
        "tvr": "tvr_pct",
        "long_num": "n_long",
        "longnum": "n_long",
        "shortnum": "n_short",
        "Holdvalue": "sh_hld",
        "holdvalue": "sh_hld",
        "Tradevalue": "sh_trd",
        "tradevalue": "sh_trd",
        "Longcount": "n_long",
        "longcount": "n_long",
        "Shortcount": "n_short",
        "shortcount": "n_short",
    }
    normalized = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return _complete_longonly_backtest_columns(normalized, original_columns=original_columns)


def _complete_longonly_backtest_columns(df: pd.DataFrame, *, original_columns: set[str]) -> pd.DataFrame:
    required = {"total_asset", "reserve_cash", "pnl", "trade_cost", "tvr_pct", "n_long"}
    if not required.issubset(df.columns):
        return df
    if "short" in df.columns or "n_short" in df.columns:
        return df

    completed = df.copy()
    long_value = (completed["total_asset"].astype(float) - completed["reserve_cash"].astype(float)).clip(lower=0.0)
    if "long" not in completed.columns:
        completed["long"] = long_value
    completed["short"] = 0.0
    completed["n_short"] = 0.0
    if "tvr" in original_columns:
        completed["tvr_pct"] = _combo_turnover_to_percent(completed["tvr_pct"])
    if "sh_hld" not in completed.columns:
        completed["sh_hld"] = completed["long"].abs() + completed["short"].abs()
    if "sh_trd" not in completed.columns:
        completed["sh_trd"] = completed["tvr_pct"].astype(float) / 100.0 * completed["sh_hld"].astype(float)
    if "longonly_pnl" not in completed.columns:
        completed["longonly_pnl"] = completed["pnl"]
    if "longonly_tradecost" not in completed.columns:
        completed["longonly_tradecost"] = completed["trade_cost"]
    if "longonly_tvr_pct" not in completed.columns:
        completed["longonly_tvr_pct"] = completed["tvr_pct"]
    return completed


def _combo_turnover_to_percent(values: pd.Series) -> pd.Series:
    numeric = values.astype(float)
    finite_abs = numeric.replace([np.inf, -np.inf], np.nan).abs().dropna()
    if finite_abs.empty:
        return numeric
    if float(finite_abs.quantile(0.95)) <= 5.0:
        return numeric * 100.0
    return numeric


def summarize_pnl(path: str | Path | pd.DataFrame, start: str | None = None, end: str | None = None) -> MetricResult:
    df = normalize_pnl_columns(read_table(path, start=start, end=end))
    rows = [_summarize_pnl_group(period, group) for period, group in period_groups(df, include_all=False)]
    table = pd.DataFrame(rows).set_index("period")
    table = _drop_empty_periods(table)
    table.loc["ALL"] = _average_all_row(table)
    return MetricResult("pnl", table, {"input": str(path), "start": start, "end": end})


def summarize_pnl_with_benchmark(path: str | Path | pd.DataFrame, benchmark_path: str | Path | pd.DataFrame | None = None, start: str | None = None, end: str | None = None) -> MetricResult:
    result = summarize_pnl(path, start=start, end=end)
    if benchmark_path is None:
        return result
    benchmark = summarize_pnl(benchmark_path, start=start, end=end).table
    table = result.table.copy()
    table["long_zz500_ret"] = np.nan
    common = table.index.intersection(benchmark.index)
    table.loc[common, "long_zz500_ret"] = benchmark.loc[common, "ret_pct"] / 100
    return MetricResult("pnl", table, {**result.metadata, "benchmark_input": str(benchmark_path)})


def pnl_quality_stats(table: pd.DataFrame) -> dict[str, float]:
    yearly = table.drop(index="ALL", errors="ignore")
    all_row = table.loc["ALL"] if "ALL" in table.index else table.iloc[-1]
    return {
        "ir": _value(all_row, "ir"),
        "margin": _value(all_row, "margin"),
        "tvr_pct": _value(all_row, "tvr_pct"),
        "lnum_ratio": _value(all_row, "lnum_ratio"),
        "long_zz500_ret": _value(all_row, "long_zz500_ret"),
        "min_ir_year": float(yearly["ir"].min()) if "ir" in yearly else np.nan,
        "min_ret_year": float(yearly["ret_pct"].min()) if "ret_pct" in yearly else np.nan,
    }


def _summarize_pnl_group(period: str, df: pd.DataFrame) -> dict[str, float | int | str]:
    pnl = _series(df, "pnl")
    long_side = _series(df, "long", df["total_asset"] - df.get("reserve_cash", 0) if "total_asset" in df.columns else None)
    short_side = _series(df, "short")
    ret = _series(df, "ret", pnl / long_side.replace(0, np.nan))
    turnover_input = _series(df, "tvr_pct") if "tvr_pct" in df.columns else None
    holdvalue = _series(df, "sh_hld", long_side.abs() + short_side.abs())
    tradevalue = _series(df, "sh_trd", turnover_input / 100 * holdvalue if turnover_input is not None else None)
    n_long = _series(df, "n_long")
    n_short = _series(df, "n_short")
    longonly_pnl = _series(df, "longonly_pnl")
    longonly_tradecost = _series(df, "longonly_tradecost")
    longonly_turnover_input = _series(df, "longonly_tvr_pct") if "longonly_tvr_pct" in df.columns else None

    active = (long_side.fillna(0) != 0) | (short_side.fillna(0) != 0)
    if not active.any():
        active = pnl.notna()
    days = int(active.sum())
    tdays = int(len(df))
    avg_long = float(long_side[active].sum() / days) if days else np.nan
    ir = sample_ir(ret)
    hold_sum = holdvalue.sum()
    trade_sum = tradevalue.sum()
    turnover = float(trade_sum / hold_sum) if hold_sum else np.nan
    if turnover_input is not None and not turnover_input.dropna().empty:
        turnover = float(turnover_input.mean() / 100)
    longonly_turnover = float(longonly_turnover_input.mean() / 100) if longonly_turnover_input is not None and not longonly_turnover_input.dropna().empty else np.nan
    annual_ret = float(ret.mean() * TRADING_DAYS * 100)
    longonly_ret = longonly_pnl / long_side.replace(0, np.nan)
    longonly_ir = sample_ir(longonly_ret)
    longonly_ret_pct = float(longonly_ret.mean() * TRADING_DAYS * 100)
    longonly_sharpe = float(longonly_ir * np.sqrt(TRADING_DAYS)) if not np.isnan(longonly_ir) else np.nan
    longonly_margin = float(longonly_pnl.sum() / longonly_tradecost.sum() * 0.003 * 10000) if longonly_tradecost.sum() else np.nan
    sharpe = float(ir * np.sqrt(TRADING_DAYS)) if not np.isnan(ir) else np.nan
    margin = float(pnl.sum() / trade_sum * 10000) if trade_sum else np.nan
    fitness = sharpe * np.sqrt(abs((annual_ret / 100) / turnover)) if turnover and not np.isnan(sharpe) else np.nan
    long_count_sum = n_long.sum()
    short_count_sum = n_short.sum()

    return {
        "period": period,
        "days": days,
        "tdays": tdays,
        "long_m": avg_long / 1e6 if not np.isnan(avg_long) else np.nan,
        "short_m": float(short_side[active].sum() / days / 1e6) if days else np.nan,
        "pnl_m": float(pnl.sum() / 1e6),
        "ret_pct": annual_ret,
        "longonly_pnl_m": float(longonly_pnl.sum() / 1e6) if longonly_pnl.notna().any() else np.nan,
        "longonly_ret_pct": longonly_ret_pct,
        "longonly_tvr_pct": longonly_turnover * 100 if not np.isnan(longonly_turnover) else np.nan,
        "longonly_ir": longonly_ir,
        "longonly_sharpe": longonly_sharpe,
        "longonly_margin": longonly_margin,
        "tvr_pct": turnover * 100 if not np.isnan(turnover) else np.nan,
        "ir": ir,
        "sharpe": sharpe,
        "dd_pct": drawdown_from_pnl(pnl, avg_long),
        "win_pct": float((pnl > 0).sum() / days * 100) if days else np.nan,
        "margin": margin,
        "fitness": fitness,
        "lnum": float(n_long[active].mean()) if days else np.nan,
        "snum": float(n_short[active].mean()) if days else np.nan,
        "tratio": days / tdays if tdays else np.nan,
        "lnum_ratio": float(long_count_sum / (long_count_sum + short_count_sum)) if (long_count_sum + short_count_sum) else np.nan,
    }


def _average_all_row(table: pd.DataFrame) -> pd.Series:
    if table.empty:
        return pd.Series({column: np.nan for column in table.columns}, dtype=float)
    row = table.mean(numeric_only=True)
    if "tdays" in table.columns:
        row["tdays"] = table["tdays"].sum()
    if "days" in table.columns:
        row["days"] = table["days"].sum()
    return row


def _drop_empty_periods(table: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in PERFORMANCE_COLUMNS if column in table.columns]
    if not columns:
        return table
    has_metric = np.isfinite(table[columns].astype(float)).any(axis=1)
    return table.loc[has_metric].copy()


def _series(df: pd.DataFrame, column: str, default: pd.Series | None = None) -> pd.Series:
    if column in df.columns:
        return df[column].astype(float)
    if default is not None:
        return default.astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _value(row: pd.Series, key: str) -> float:
    return float(row[key]) if key in row and pd.notna(row[key]) else np.nan
