from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_matrix


def equalize_short_side(alpha: str | Path | pd.DataFrame, percentile: float = 0.5, group: str | Path | pd.DataFrame | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    values = read_matrix(alpha, start=start, end=end)
    groups = read_matrix(group, start=start, end=end) if group is not None else None
    output = values.copy()
    for date in output.index:
        if groups is None:
            output.loc[date] = _equalize_row(output.loc[date], percentile)
        else:
            group_row = groups.loc[date, output.columns]
            for group_value in group_row.dropna().unique():
                columns = group_row[group_row == group_value].index
                output.loc[date, columns] = _equalize_row(output.loc[date, columns], percentile)
    return output


def universe_coverage(alpha: str | Path | pd.DataFrame, universe_mask: str | Path | pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    values = read_matrix(alpha, start=start, end=end)
    mask = read_matrix(universe_mask, start=start, end=end).astype(bool)
    dates = values.index.intersection(mask.index)
    columns = values.columns.intersection(mask.columns)
    rows = []
    for date in dates:
        active = values.loc[date, columns].notna()
        universe = mask.loc[date, columns]
        active_count = int(active.sum())
        in_universe = int((active & universe).sum())
        rows.append({
            "date": date,
            "active": active_count,
            "in_universe": in_universe,
            "ratio": in_universe / active_count if active_count else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


def trading_limit_exposure(trades: str | Path | pd.DataFrame, limit_mask: str | Path | pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    trade_df = read_matrix(trades, start=start, end=end)
    limit_df = read_matrix(limit_mask, start=start, end=end).astype(bool)
    dates = trade_df.index.intersection(limit_df.index)
    columns = trade_df.columns.intersection(limit_df.columns)
    rows = []
    for date in dates:
        trade_abs = trade_df.loc[date, columns].abs()
        valid = trade_abs.notna()
        total = float(trade_abs[valid].sum())
        limited = float(trade_abs[valid & limit_df.loc[date, columns]].sum())
        rows.append({
            "date": date,
            "trade_abs": total,
            "limited_trade_abs": limited,
            "limited_ratio": limited / total if total else np.nan,
        })
    return pd.DataFrame(rows).set_index("date")


def _equalize_row(row: pd.Series, percentile: float) -> pd.Series:
    result = row.copy()
    valid = row.dropna().astype(float)
    if valid.empty:
        return result
    cutoff = valid.quantile(percentile)
    selected = valid <= cutoff
    if selected.any():
        result.loc[selected.index[selected]] = float(valid[selected].mean())
    return result
