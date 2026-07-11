from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from .memmaper2 import Memmaper2
from .cache_layout import ashare_cache_path


def load_index_benchmark(
    cache_path: str | Path,
    start_ds: int,
    end_ds: int,
    ts_code: str = "000905.SH",
) -> pd.DataFrame:
    root = ashare_cache_path(cache_path)
    weight_path = root / "1d_IndexWeight" / f"IndexWeight.{ts_code}"
    close_path = root / "1d_DailyKline" / "DailyKline.close"
    prev_close_path = root / "1d_DailyKline" / "DailyKline.real_pre_close"
    for path in (weight_path, close_path, prev_close_path):
        if not path.exists():
            raise FileNotFoundError(f"benchmark cache path not found: {path}")

    weights = Memmaper2(str(weight_path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:].astype(float)
    close = Memmaper2(str(close_path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:].astype(float)
    prev_close = Memmaper2(str(prev_close_path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:].astype(float)
    common = weights.columns.intersection(close.columns).intersection(prev_close.columns)
    if common.empty:
        raise ValueError(f"benchmark cache has no common stock columns for {ts_code}")

    weights = weights.loc[:, common].reindex(close.index)
    # Use the raw previous close so the benchmark matches a price-index style series.
    stock_ret = close.loc[:, common] / prev_close.loc[:, common] - 1.0
    valid = np.isfinite(stock_ret.to_numpy(dtype=float)) & np.isfinite(weights.to_numpy(dtype=float)) & (weights.to_numpy(dtype=float) != 0)
    weighted = weights.to_numpy(dtype=float)
    returns = stock_ret.to_numpy(dtype=float)
    numerator = np.where(valid, weighted * returns, 0.0).sum(axis=1)
    denominator = np.where(valid, weighted, 0.0).sum(axis=1)
    index_ret = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    index_close = np.cumprod(1.0 + np.nan_to_num(index_ret, nan=0.0, posinf=0.0, neginf=0.0))
    return pd.DataFrame({"trade_date": close.index.astype(str), "close": index_close})


def benchmark_returns_from_cache(
    cache_path: str | Path,
    dates: list[int],
    ts_code: str = "000905.SH",
) -> pd.Series:
    if not dates:
        return pd.Series(dtype=float)
    bench = load_index_benchmark(cache_path, min(dates), max(dates), ts_code=ts_code)
    close = bench.sort_values("trade_date").set_index("trade_date")["close"].astype(float)
    close = close.reindex(pd.Index([str(int(date)) for date in dates], name="trade_date")).ffill()
    if close.isna().any():
        raise ValueError("local benchmark close series could not be aligned to requested dates")
    return close.pct_change().fillna(0.0).reset_index(drop=True)


def cache_path_from_rendered_config(config_path: str | Path) -> Path | None:
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    try:
        root = ET.parse(config_path).getroot()
    except Exception:
        return None
    constants = root.find("./constants")
    if constants is None:
        return None
    raw = constants.get("cache_path")
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()
