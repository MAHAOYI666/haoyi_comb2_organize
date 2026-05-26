from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_cache_array, read_table
from .pnl import period_groups, sample_ir, _average_all_row
from .schemas import MetricResult, PNLSUPER_INTRADAY_INTERVALS, PNLSUPER_TRADING_DAYS


def normalize_ic_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "ic": "1d_IC",
        "5dic": "5d_IC",
        "10dic": "10d_IC",
        "rankic": "rankic",
        "percic": "percic",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def summarize_ic(path: str | Path | pd.DataFrame, start: str | None = None, end: str | None = None, normalize_names: bool = False) -> MetricResult:
    df = read_table(path, start=start, end=end)
    if normalize_names:
        df = normalize_ic_columns(df)
    coverage = df.pop("coverage") if "coverage" in df.columns else None
    rows = []
    for period, group in period_groups(df, include_all=False):
        row: dict[str, float | int | str] = {"period": period}
        active = _active_rows(group, coverage.loc[group.index] if coverage is not None else None)
        row["days"] = int(active.sum())
        for column in group.columns:
            values = group.loc[active, column].astype(float)
            row[f"{column}.avg"] = _scaled_average(column, values)
            row[f"{column}.ir"] = sample_ir(values)
            row[f"{column}.std"] = float(values.std(ddof=1)) if len(values.dropna()) >= 2 else np.nan
        if coverage is not None:
            row["coverage.avg"] = float(coverage.loc[group.index].astype(float).mean())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("period")
    table.loc["ALL"] = _average_all_row(table)
    return MetricResult("ic", table.round(2), {"input": str(path), "start": start, "end": end})


def summarize_cache_ic(path: str | Path, start_ds: str | int, end_ds: str | int, df_type: str, columns: list[str] | None = None) -> MetricResult:
    data = read_cache_array(path, start_ds, end_ds, df_type)
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
        if columns is not None:
            df.columns = columns
    return summarize_ic(df, start=str(start_ds), end=str(end_ds))


def ic_quality_stats(table: pd.DataFrame) -> dict[str, float]:
    yearly = table.drop(index="ALL", errors="ignore")
    all_row = table.loc["ALL"] if "ALL" in table.index else table.iloc[-1]
    stats: dict[str, float] = {}
    base_fields = sorted({column.rsplit(".", 1)[0] for column in table.columns if column.endswith((".avg", ".ir"))})
    for field in base_fields:
        avg_key = f"{field}.avg"
        ir_key = f"{field}.ir"
        if avg_key in all_row:
            stats[avg_key] = _row_value(all_row, avg_key)
            stats[f"{field}.min_each_year"] = float(yearly[avg_key].min()) if avg_key in yearly else np.nan
        if ir_key in all_row:
            stats[ir_key] = _row_value(all_row, ir_key)
            stats[f"{field}.ir_each_year"] = float(yearly[ir_key].min()) if ir_key in yearly else np.nan
    return stats


def universe_ratio(universe_table: pd.DataFrame, raw_ic_table: pd.DataFrame, universe: str, field: str, denominator_field: str = "AshareFiltered") -> float:
    numerator_key = f"{universe}.{field}.avg"
    denominator_key = f"{denominator_field}.{field}.avg"
    all_row = raw_ic_table.loc["ALL"] if "ALL" in raw_ic_table.index else raw_ic_table.iloc[-1]
    numerator = float(universe_table.loc["ALL", numerator_key]) if numerator_key in universe_table.columns else np.nan
    denominator = float(all_row[denominator_key]) if denominator_key in raw_ic_table.columns else np.nan
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return 0.0
    return numerator / denominator


def _active_rows(group: pd.DataFrame, coverage: pd.Series | None) -> pd.Series:
    if coverage is not None:
        return coverage.astype(float) > 0
    return group.fillna(0).ne(0).any(axis=1)


def _scaled_average(column: str, values: pd.Series) -> float:
    avg = float(values.mean())
    intraday = re.fullmatch(r"(\d+)t_pnl", column)
    daily = re.fullmatch(r"(\d+)d_pnl", column)
    barra_daily = re.fullmatch(r"barra(\d+)d_pnl", column)
    if intraday:
        intervals = int(intraday.group(1))
        return avg * PNLSUPER_TRADING_DAYS * 100 * PNLSUPER_INTRADAY_INTERVALS / intervals * 2
    if daily:
        days = int(daily.group(1))
        return avg * PNLSUPER_TRADING_DAYS * 100 / np.sqrt(days) * 2
    if column == "barra_pnl":
        return avg * PNLSUPER_TRADING_DAYS * 2
    if barra_daily:
        days = int(barra_daily.group(1))
        return avg * PNLSUPER_TRADING_DAYS / np.sqrt(days) * 2
    return avg


def _row_value(row: pd.Series, key: str) -> float:
    return float(row[key]) if key in row and pd.notna(row[key]) else np.nan
