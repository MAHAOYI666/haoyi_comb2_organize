from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .correlation import pnl_correlation
from .io import read_table
from .pnl import sample_ir


def value_added(candidate: str | Path | pd.DataFrame, solid: str | Path | pd.DataFrame, column: str = "pnl", start: str | None = None, end: str | None = None) -> dict[str, float | int | str]:
    candidate_df = read_table(candidate, start=start, end=end)
    solid_df = read_table(solid, start=start, end=end)
    aligned = pd.concat([candidate_df[column].rename("candidate"), solid_df[column].rename("solid")], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN observations for value-added calculation.")
    candidate_ir = sample_ir(aligned["candidate"])
    solid_ir = sample_ir(aligned["solid"])
    corr = float(aligned["candidate"].corr(aligned["solid"])) if len(aligned) >= 2 else np.nan
    va = candidate_ir - corr * solid_ir if not any(pd.isna(x) for x in [candidate_ir, solid_ir, corr]) else np.nan
    return {
        "candidate_ir": float(candidate_ir),
        "solid_ir": float(solid_ir),
        "corr": corr,
        "value_added_ir": float(va),
        "n_obs": int(len(aligned)),
        "start": aligned.index.min().strftime("%Y%m%d"),
        "end": aligned.index.max().strftime("%Y%m%d"),
    }


def pool_value_added(candidate: str | Path | pd.DataFrame, pool_paths: list[str | Path | pd.DataFrame], column: str = "pnl", start: str | None = None, end: str | None = None) -> pd.DataFrame:
    rows = []
    for idx, pool_path in enumerate(pool_paths):
        result = value_added(candidate, pool_path, column=column, start=start, end=end)
        rows.append({"name": _name(pool_path, idx), **result})
    table = pd.DataFrame(rows).set_index("name")
    if not table.empty:
        table.attrs["summary"] = pd.DataFrame({"value": {
            "min_value_added_ir": float(table["value_added_ir"].min()),
            "avg_value_added_ir": float(table["value_added_ir"].mean()),
            "max_corr": float(table["corr"].max()),
        }})
    return table


def netting_return(candidate: str | Path | pd.DataFrame, pool: str | Path | pd.DataFrame, column: str = "pnl", start: str | None = None, end: str | None = None) -> dict[str, float | int | str]:
    candidate_df = read_table(candidate, start=start, end=end)
    pool_df = read_table(pool, start=start, end=end)
    aligned = pd.concat([candidate_df[column].rename("candidate"), pool_df[column].rename("pool")], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN observations for netting calculation.")
    variance = float(aligned["pool"].var(ddof=1))
    beta = float(aligned["candidate"].cov(aligned["pool"]) / variance) if variance else 0.0
    residual = aligned["candidate"] - beta * aligned["pool"]
    return {
        "beta": beta,
        "residual_mean": float(residual.mean()),
        "residual_ir": sample_ir(residual),
        "residual_sum": float(residual.sum()),
        "candidate_ir": sample_ir(aligned["candidate"]),
        "pool_ir": sample_ir(aligned["pool"]),
        "corr": pnl_correlation(aligned[["candidate"]].rename(columns={"candidate": column}), aligned[["pool"]].rename(columns={"pool": column}), column=column)["corr"],
        "n_obs": int(len(aligned)),
    }


def _name(path: str | Path | pd.DataFrame, idx: int) -> str:
    if isinstance(path, pd.DataFrame):
        return f"pool_{idx}"
    return Path(path).stem
