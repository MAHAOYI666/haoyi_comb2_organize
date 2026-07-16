from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from comb2_simbase.cache_layout import BARRA_STYLE_DIRNAME, BARRA_STYLE_PREFIX, ashare_cache_path

from .io import normalize_date_index, read_cache_array
from .pnl import period_groups, sample_ir


MARKET_CAP_RELATIVE_PATH = Path("1d_DailyFdm") / "DailyFdm.mkt_cap"


def compute_style_factor_exposure(
    signal: pd.DataFrame,
    style_factors: Mapping[str, pd.DataFrame],
    mode: int = 0,
) -> pd.DataFrame:
    signal_df = _normalize_exposure_frame(signal)
    if not style_factors:
        raise ValueError("style_factors is empty.")
    if mode not in {0, 1}:
        raise ValueError("mode must be 0 (correlation exposure) or 1 (beta exposure).")

    common_dates = signal_df.index
    common_codes = signal_df.columns
    normalized_styles: dict[str, pd.DataFrame] = {}
    for style_name, frame in style_factors.items():
        normalized = _normalize_exposure_frame(frame)
        normalized_styles[style_name] = normalized
        common_dates = common_dates.intersection(normalized.index)
        common_codes = common_codes.intersection(normalized.columns)

    common_dates = common_dates.sort_values()
    common_codes = common_codes.sort_values()
    if common_dates.empty or common_codes.empty:
        raise ValueError("No overlapping dates/codes between signal and style factors.")

    signal_values = signal_df.reindex(index=common_dates, columns=common_codes).to_numpy(dtype=np.float32, copy=False)
    result = pd.DataFrame(index=common_dates)
    result.index.name = "date"

    for style_name, frame in normalized_styles.items():
        style_values = frame.reindex(index=common_dates, columns=common_codes).to_numpy(dtype=np.float32, copy=False)
        corr, beta = _compute_daily_exposure(signal_values, style_values)
        result[style_name] = corr if mode == 0 else beta
    return result


def compute_barra_style_exposure(
    signal: pd.DataFrame,
    cache_path: str | Path,
    start_ds: int | None = None,
    end_ds: int | None = None,
    mode: int = 0,
) -> pd.DataFrame:
    signal_df = _normalize_exposure_frame(signal)
    style_paths = _discover_barra_style_paths(cache_path)
    if mode not in {0, 1}:
        raise ValueError("mode must be 0 (correlation exposure) or 1 (beta exposure).")

    first_style = _load_cache_frame(style_paths[0][1], start_ds=None, end_ds=None)
    style_start = int(first_style.index.min())
    style_end = int(first_style.index.max())
    resolved_start = max(int(signal_df.index.min()), style_start) if start_ds is None else int(start_ds)
    resolved_end = min(int(signal_df.index.max()), style_end) if end_ds is None else int(end_ds)
    if resolved_start > resolved_end:
        raise ValueError("Resolved date range is empty.")

    filtered_signal = signal_df.loc[(signal_df.index >= resolved_start) & (signal_df.index <= resolved_end)]
    if filtered_signal.empty:
        raise ValueError("Signal has no observations in the requested date range.")

    style_factors = {
        style_name: _load_cache_frame(style_path, start_ds=resolved_start, end_ds=resolved_end)
        for style_name, style_path in style_paths
    }
    exposure = compute_style_factor_exposure(filtered_signal, style_factors, mode=mode)
    exposure.attrs["start_ds"] = resolved_start
    exposure.attrs["end_ds"] = resolved_end
    exposure.attrs["mode"] = mode
    return exposure


def compute_cap_corr(
    signal: pd.DataFrame,
    cache_path: str | Path,
    start_ds: int | None = None,
    end_ds: int | None = None,
) -> pd.Series:
    """Return daily CAP correlations using market cap available on the signal date.

    The signal is first transformed into a dollar-neutral long/short weight vector:
    cross-sectionally median-center it, then normalize the positive and negative
    legs to one gross unit each.  The resulting weights are correlated with that
    day's cross-sectionally ranked market caps.
    """
    signal_df = _normalize_cap_corr_frame(signal)
    resolved_start = int(signal_df.index.min()) if start_ds is None else int(start_ds)
    resolved_end = int(signal_df.index.max()) if end_ds is None else int(end_ds)
    filtered_signal = signal_df.loc[(signal_df.index >= resolved_start) & (signal_df.index <= resolved_end)]
    if filtered_signal.empty:
        raise ValueError("Signal has no observations in the requested date range.")

    market_cap_path = ashare_cache_path(cache_path) / MARKET_CAP_RELATIVE_PATH
    market_cap = _normalize_cap_corr_frame(_load_cache_frame(market_cap_path, resolved_start, resolved_end))
    common_dates = filtered_signal.index.intersection(market_cap.index).sort_values()
    common_codes = filtered_signal.columns.intersection(market_cap.columns).sort_values()
    if common_dates.empty or common_codes.empty:
        raise ValueError("No overlapping dates/codes between signal and market cap.")

    signal_values = filtered_signal.reindex(index=common_dates, columns=common_codes).to_numpy(dtype=np.float32, copy=False)
    market_cap_values = market_cap.reindex(index=common_dates, columns=common_codes).to_numpy(dtype=np.float32, copy=False)
    weights = _long_short_weights(signal_values, market_cap_values)
    ranked_market_cap = pd.DataFrame(market_cap_values).rank(axis=1).to_numpy(dtype=np.float32)
    corr, _ = _compute_daily_exposure(weights, ranked_market_cap)
    index = pd.to_datetime(common_dates.astype(str), format="%Y%m%d")
    return pd.Series(corr, index=index, name="cap_corr")


def summarize_cap_corr(cap_corr: pd.Series) -> pd.DataFrame:
    """Summarize daily CAP correlation by calendar year and over all valid days."""
    if not isinstance(cap_corr, pd.Series):
        raise TypeError("cap_corr must be a pandas Series.")
    daily = normalize_date_index(cap_corr.rename("cap_corr").to_frame())["cap_corr"].astype(float)
    rows = [_cap_corr_summary_row(period, group["cap_corr"]) for period, group in period_groups(daily.to_frame(), include_all=False)]
    rows.append(_cap_corr_summary_row("ALL", daily))
    return pd.DataFrame(rows).set_index("period")


def _compute_daily_exposure(signal: np.ndarray, style: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(signal) & np.isfinite(style)
    nobs = finite.sum(axis=1)

    safe_signal = np.where(finite, signal, 0.0)
    safe_style = np.where(finite, style, 0.0)
    denom_n = np.where(nobs > 0, nobs, 1).astype(np.float32)

    mean_signal = safe_signal.sum(axis=1) / denom_n
    mean_style = safe_style.sum(axis=1) / denom_n

    centered_signal = np.where(finite, signal - mean_signal[:, None], 0.0)
    centered_style = np.where(finite, style - mean_style[:, None], 0.0)

    ss_signal = np.sum(centered_signal * centered_signal, axis=1)
    ss_style = np.sum(centered_style * centered_style, axis=1)
    cross = np.sum(centered_signal * centered_style, axis=1)

    valid = (nobs >= 3) & (ss_signal > 0.0) & (ss_style > 0.0)
    corr = np.full(signal.shape[0], np.nan, dtype=np.float32)
    beta = np.full(signal.shape[0], np.nan, dtype=np.float32)
    corr[valid] = cross[valid] / np.sqrt(ss_signal[valid] * ss_style[valid])
    beta[valid] = cross[valid] / ss_style[valid]
    return corr, beta


def _long_short_weights(signal: np.ndarray, market_cap: np.ndarray) -> np.ndarray:
    weights = np.full(signal.shape, np.nan, dtype=np.float32)
    for row_idx in range(signal.shape[0]):
        valid = np.isfinite(signal[row_idx]) & np.isfinite(market_cap[row_idx])
        if valid.sum() < 3:
            continue
        alpha = signal[row_idx, valid].astype(np.float64, copy=False)
        centered = alpha - np.median(alpha)
        long_sum = centered[centered > 0].sum()
        short_sum = -centered[centered < 0].sum()
        if long_sum <= 0.0 or short_sum <= 0.0:
            continue

        row_weights = np.zeros_like(centered, dtype=np.float64)
        positive = centered > 0
        negative = centered < 0
        row_weights[positive] = centered[positive] / long_sum
        row_weights[negative] = centered[negative] / short_sum
        weights[row_idx, valid] = row_weights
    return weights


def _normalize_exposure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("signal/style factor input must be a pandas DataFrame.")
    normalized = frame.copy()
    if normalized.index.nlevels > 1 and "times" in normalized.index.names:
        normalized = normalized.reset_index("times", drop=True)
    coerced_dates = pd.Series(_coerce_int_dates(normalized.index), index=normalized.index)
    normalized = normalized.loc[coerced_dates.notna()].copy()
    normalized.index = pd.Index(coerced_dates.loc[coerced_dates.notna()].astype(int).to_numpy(), name="date")
    normalized.columns = pd.Index(normalized.columns.astype(str), name="code")
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    return normalized.sort_index()


def _normalize_cap_corr_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = _normalize_exposure_frame(frame)
    normalized.columns = pd.Index(normalized.columns.astype(str).str.zfill(6), name="code")
    return normalized


def _cap_corr_summary_row(period: str, values: pd.Series) -> dict[str, float | int | str]:
    valid = values.dropna().astype(float)
    return {
        "period": period,
        "days": int(len(valid)),
        "cap_corr.avg": float(valid.mean()) if not valid.empty else np.nan,
        "cap_corr.ir": sample_ir(valid),
        "cap_corr.std": float(valid.std(ddof=1)) if len(valid) >= 2 else np.nan,
    }


def _coerce_int_dates(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return index.strftime("%Y%m%d").astype(int).to_numpy()
    values = pd.Index(index).astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    dates = pd.to_numeric(values, errors="coerce")
    return dates.to_numpy(dtype="float64")


def _discover_barra_style_paths(cache_path: str | Path) -> list[tuple[str, Path]]:
    barra_root = ashare_cache_path(cache_path) / BARRA_STYLE_DIRNAME
    if not barra_root.exists():
        raise FileNotFoundError(f"Barra style directory not found: {barra_root}")
    paths = [
        (path.name.replace(BARRA_STYLE_PREFIX, "", 1), path)
        for path in sorted(barra_root.iterdir(), key=lambda item: item.name)
        if path.name.startswith(BARRA_STYLE_PREFIX)
    ]
    if not paths:
        raise FileNotFoundError(f"No Barra style files found under: {barra_root}")
    return paths


def _load_cache_frame(path: str | Path, start_ds: int | None, end_ds: int | None) -> pd.DataFrame:
    data = read_cache_array(path, start_ds, end_ds, True)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Cache path did not return a DataFrame: {path}")
    return _normalize_exposure_frame(data)
