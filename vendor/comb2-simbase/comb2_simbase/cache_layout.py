from __future__ import annotations

from pathlib import Path


ASHARE_CACHE_DIRNAME = "AshareCache"
BARRA_STYLE_DIRNAME = "1d_BarraCNE5"
BARRA_STYLE_PREFIX = "BarraCNE5."
DAILY_LABEL_DIRNAME = "1d_DailyLabel"
DAILY_LABEL_PREFIX = "DailyLabel."
STOCK_MASK_DIRNAME = "1d_StockMask2"
VALID_MASK_NAME = "StockMask2.NoNewStockMask"
FILTERED_MASK_NAME = "StockMask2.LimitMask"
BASE_UNIVERSE_MASK_NAME = "StockMask2.BaseUnivMask"
SUSPEND_MASK_NAME = "StockMask2.SuspendStock"


def ashare_cache_path(cache_path: str | Path) -> Path:
    """Return the AshareCache directory below its configured parent."""
    return Path(cache_path).expanduser() / ASHARE_CACHE_DIRNAME


def daily_label_path(cache_path: str | Path, label_name: str) -> Path:
    return ashare_cache_path(cache_path) / DAILY_LABEL_DIRNAME / f"{DAILY_LABEL_PREFIX}{label_name}"


def stock_mask_path(cache_path: str | Path, mask_name: str) -> Path:
    return ashare_cache_path(cache_path) / STOCK_MASK_DIRNAME / mask_name


def barra_style_path(cache_path: str | Path, filename: str) -> Path:
    return ashare_cache_path(cache_path) / BARRA_STYLE_DIRNAME / f"{BARRA_STYLE_PREFIX}{filename}"
