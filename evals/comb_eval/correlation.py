from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_matrix, read_table
from .pnl import sample_ir


def pnl_correlation(candidate: str | Path | pd.DataFrame, pool: str | Path | pd.DataFrame, column: str = "pnl", start: str | None = None, end: str | None = None) -> dict[str, float | int | str]:
    candidate_df = read_table(candidate, start=start, end=end)
    pool_df = read_table(pool, start=start, end=end)
    aligned = pd.concat([candidate_df[column].rename("candidate"), pool_df[column].rename("pool")], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN observations for pnl correlation.")
    return {
        "corr": float(aligned["candidate"].corr(aligned["pool"])),
        "n_obs": int(len(aligned)),
        "start": aligned.index.min().strftime("%Y%m%d"),
        "end": aligned.index.max().strftime("%Y%m%d"),
    }


def pnl_pool_correlation(candidate: str | Path | pd.DataFrame, pool_paths: list[str | Path | pd.DataFrame], column: str = "pnl", top: int = 5, ratio: float = 1.5, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    candidate_df = read_table(candidate, start=start, end=end)
    candidate_series = candidate_df[column].astype(float).rename("candidate")
    rows = []
    candidate_avg = float(candidate_series.mean())
    candidate_ir = sample_ir(candidate_series)
    for idx, pool_path in enumerate(pool_paths):
        pool_df = read_table(pool_path, start=start, end=end)
        pool_series = pool_df[column].astype(float).rename("pool")
        aligned = pd.concat([candidate_series, pool_series], axis=1, join="inner").dropna()
        corr = float(aligned["candidate"].corr(aligned["pool"])) if len(aligned) >= 2 else np.nan
        pool_avg = float(aligned["pool"].mean()) if not aligned.empty else np.nan
        pool_ir = sample_ir(aligned["pool"]) if not aligned.empty else np.nan
        rows.append({
            "name": _pool_name(pool_path, idx),
            "corr": corr,
            "n_obs": int(len(aligned)),
            "pool_avg": pool_avg,
            "pool_ir": pool_ir,
            "avg_ratio": _clipped_ratio(candidate_avg, pool_avg, corr),
            "ir_ratio": _clipped_ratio(candidate_ir, pool_ir, corr),
            "avg_beat": _beat(candidate_avg, pool_avg, corr, ratio),
            "ir_beat": _beat(candidate_ir, pool_ir, corr, ratio),
        })
    table = pd.DataFrame(rows).set_index("name")
    if table.empty:
        return table
    sorted_corr = table["corr"].sort_values(ascending=False)
    summary = pd.DataFrame({
        "value": {
            "maxcorr": float(table["corr"].max()),
            "avgcorr": float(table["corr"].mean()),
            f"topcorr{top}": float(sorted_corr.head(top).mean()),
            f"avgRatio{top}": float(table.loc[sorted_corr.head(top).index, "avg_ratio"].mean()),
            f"irRatio{top}": float(table.loc[sorted_corr.head(top).index, "ir_ratio"].mean()),
            "avgBeatByNum": int(table["avg_beat"].sum()),
            "irBeatByNum": int(table["ir_beat"].sum()),
        }
    })
    summary.index.name = "metric"
    table.attrs["summary"] = summary
    return table


def daily_matrix_correlation(
    candidate: str | Path | pd.DataFrame,
    pool: str | Path | pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    corr_days: int | None = None,
    min_valid: int = 2,
) -> pd.DataFrame:
    candidate_df = read_matrix(candidate, start=start, end=end)
    pool_df = read_matrix(pool, start=start, end=end)
    dates = candidate_df.index.intersection(pool_df.index).sort_values()
    if corr_days is not None:
        dates = dates[-corr_days:]
    columns = candidate_df.columns.intersection(pool_df.columns)
    rows = []
    for date in dates:
        left = candidate_df.loc[date, columns].astype(float)
        right = pool_df.loc[date, columns].astype(float)
        valid = left.notna() & right.notna() & (left != 0) & (right != 0)
        corr = float(left[valid].corr(right[valid])) if int(valid.sum()) >= min_valid else np.nan
        rows.append({"date": date, "corr": corr, "n_inst": int(valid.sum())})
    return pd.DataFrame(rows).set_index("date")


def position_correlation(
    candidate: str | Path | pd.DataFrame,
    pool: str | Path | pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    corr_days: int | None = None,
    min_valid: int = 1000,
) -> dict[str, float | int | str]:
    daily = daily_matrix_correlation(candidate, pool, start=start, end=end, corr_days=corr_days, min_valid=min_valid)
    valid = daily["corr"].dropna()
    if valid.empty:
        raise ValueError("No overlapping nonzero matrix rows with enough valid instruments.")
    return {
        "avg_corr": float(valid.mean()),
        "max_corr": float(valid.max()),
        "min_corr": float(valid.min()),
        "n_days": int(len(valid)),
        "corr_days": int(corr_days) if corr_days is not None else int(len(daily)),
        "min_valid": int(min_valid),
        "start": valid.index.min().strftime("%Y%m%d"),
        "end": valid.index.max().strftime("%Y%m%d"),
    }


def matrix_correlation(
    candidate: str | Path | pd.DataFrame,
    pool: str | Path | pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    corr_days: int | None = None,
    min_valid: int = 2,
) -> dict[str, float | int | str]:
    return position_correlation(candidate, pool, start=start, end=end, corr_days=corr_days, min_valid=min_valid)


def _pool_name(path: str | Path | pd.DataFrame, idx: int) -> str:
    if isinstance(path, pd.DataFrame):
        return f"pool_{idx}"
    return Path(path).stem


def _clipped_ratio(candidate_value: float, pool_value: float, corr: float) -> float:
    if pd.isna(candidate_value) or pd.isna(pool_value) or pd.isna(corr) or pool_value == 0 or corr == 0:
        return np.nan
    return float(np.clip(candidate_value / pool_value / corr, -5, 5))


def _beat(candidate_value: float, pool_value: float, corr: float, ratio: float) -> bool:
    if pd.isna(candidate_value) or pd.isna(pool_value) or pd.isna(corr) or pool_value == 0:
        return False
    return (candidate_value / pool_value) <= corr * ratio
