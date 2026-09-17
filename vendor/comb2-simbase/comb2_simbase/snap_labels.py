from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cache_layout import (
    BASE_UNIVERSE_MASK_NAME,
    FILTERED_MASK_NAME,
    ashare_cache_path,
    stock_mask_path,
)
from .memmaper2 import Memmaper2


def snap_vwap_price_name(snap_ti: int) -> str:
    return f"IntraVwap.Vwap30.{int(snap_ti):06d}"


def load_snap_vwap_labels(
    cache_path: str | Path,
    snap_ti: int,
    start_ds: int,
    end_ds: int,
    *,
    periods: tuple[int, ...] = (1, 5),
) -> dict[int, pd.DataFrame]:
    """Build forward labels from a snapshot VWAP cache field.

    Period one keeps the historical close-to-snapshot convention used by the
    existing label cache. Longer periods use the same next-day snapshot anchor
    and support arbitrary positive periods.
    """
    if not periods or any(int(period) <= 0 for period in periods):
        raise ValueError("periods must contain positive integers")
    periods = tuple(int(period) for period in periods)
    root = ashare_cache_path(cache_path)
    price_reader = Memmaper2(str(root / "1d_IntraVwap" / snap_vwap_price_name(snap_ti)))
    dates = np.asarray(price_reader._index, dtype=np.int64)
    end_idx = np.searchsorted(dates, int(end_ds), side="right") - 1
    extended_end_ds = int(dates[min(max(end_idx, 0) + max(periods) + 1, len(dates) - 1)])

    price = _load_frame(price_reader, start_ds, extended_end_ds).replace(0, np.nan)
    close = _load_frame(root / "1d_DailyKline" / "DailyKline.close_hfq", start_ds, extended_end_ds).replace(0, np.nan)
    adjustment = _load_frame(root / "1d_DailyKline" / "DailyKline.adj_factor", start_ds, extended_end_ds).replace(0, np.nan)
    base_mask = _load_frame(stock_mask_path(cache_path, BASE_UNIVERSE_MASK_NAME), start_ds, extended_end_ds)
    limit_mask = _load_frame(stock_mask_path(cache_path, FILTERED_MASK_NAME), start_ds, extended_end_ds)

    index = price.index
    columns = price.columns
    close = close.reindex(index=index, columns=columns)
    adjustment = adjustment.reindex(index=index, columns=columns)
    label_mask = (base_mask.reindex(index=index, columns=columns) * limit_mask.reindex(index=index, columns=columns)).shift(-1)
    price = price * adjustment

    requested_index = index[(index >= int(start_ds)) & (index <= int(end_ds))]
    labels = {}
    if 1 in periods:
        label = (close / price - 1).shift(-1) + (price / close.shift(1) - 1).shift(-2).fillna(0)
        labels[1] = (label * label_mask).reindex(index=requested_index)
    for period in periods:
        if period == 1:
            continue
        label = price.pct_change(period, fill_method=None).shift(-(period + 1))
        labels[period] = (label * label_mask).reindex(index=requested_index)
    return labels


def _load_frame(reader_or_path: Memmaper2 | str | Path, start_ds: int, end_ds: int) -> pd.DataFrame:
    reader = reader_or_path if isinstance(reader_or_path, Memmaper2) else Memmaper2(str(reader_or_path))
    return reader.load(start_ds=int(start_ds), end_ds=int(end_ds), df_type=True).dloc[:].astype(float)
