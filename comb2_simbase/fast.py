from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def purify(x):
    if _is_torch(x):
        y = x.clone()
        y[torch.isinf(y)] = torch.nan
        return y
    y = np.array(x, copy=True)
    y[np.isinf(y)] = np.nan
    return y


def _axis_for(x, dim: int) -> int:
    ndim = x.ndim if hasattr(x, "ndim") else np.asarray(x).ndim
    return dim if dim >= 0 else ndim + dim


def rank(x, dim: int = 0, pct: bool = False):
    if _is_torch(x):
        arr = x.detach().cpu().numpy()
        ranked = pd.DataFrame(arr).rank(axis=_axis_for(arr, dim), pct=pct, method="first").to_numpy()
        return torch.as_tensor(ranked, dtype=x.dtype, device=x.device)
    axis = _axis_for(x, dim)
    if isinstance(x, pd.DataFrame):
        return x.rank(axis=axis, pct=pct, method="first")
    return pd.DataFrame(np.asarray(x)).rank(axis=axis, pct=pct, method="first").to_numpy()


def _perc_long_array(values: np.ndarray, percentile: float = 0.5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        threshold = np.nanquantile(arr, percentile, axis=1, keepdims=True)
    out = arr - threshold
    high = arr > threshold
    low = arr < threshold
    positive_sum = np.where(high, out, 0.0).sum(axis=1, keepdims=True)
    low_count = low.sum(axis=1, keepdims=True)
    replacement = np.divide(-positive_sum, low_count, out=np.zeros_like(positive_sum), where=low_count > 0)
    out = np.where(low & (low_count > 0), replacement, out)
    return np.where(np.isfinite(arr) & np.isfinite(threshold), out, np.nan)


def perc_long(x, percentile: float = 0.5):
    if _is_torch(x):
        arr = x.detach().cpu().numpy()
        return _perc_long_array(arr, percentile)
    result = _perc_long_array(np.asarray(x), percentile)
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(result, index=x.index, columns=x.columns)
    return result


def corr(left, right, dim: int = 1, keepdims: bool = False):
    if _is_torch(left) or _is_torch(right):
        left_t = left if _is_torch(left) else torch.as_tensor(left)
        right_t = right if _is_torch(right) else torch.as_tensor(right, dtype=left_t.dtype, device=left_t.device)
        left_t = left_t.to(torch.float32)
        right_t = right_t.to(torch.float32)
        valid = torch.isfinite(left_t) & torch.isfinite(right_t)
        count = valid.sum(dim=dim, keepdim=True)
        left_mean = torch.where(valid, left_t, torch.nan).nanmean(dim=dim, keepdim=True)
        right_mean = torch.where(valid, right_t, torch.nan).nanmean(dim=dim, keepdim=True)
        left_centered = torch.where(valid, left_t - left_mean, torch.zeros_like(left_t))
        right_centered = torch.where(valid, right_t - right_mean, torch.zeros_like(right_t))
        numerator = (left_centered * right_centered).sum(dim=dim, keepdim=True)
        denominator = torch.sqrt((left_centered * left_centered).sum(dim=dim, keepdim=True) * (right_centered * right_centered).sum(dim=dim, keepdim=True))
        out = numerator / denominator
        out = torch.where((count >= 2) & (denominator > 0), out, torch.full_like(out, torch.nan))
        return out if keepdims else out.squeeze(dim)

    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    valid = np.isfinite(left_arr) & np.isfinite(right_arr)
    count = valid.sum(axis=dim, keepdims=True)
    left_sum = np.where(valid, left_arr, 0.0).sum(axis=dim, keepdims=True)
    right_sum = np.where(valid, right_arr, 0.0).sum(axis=dim, keepdims=True)
    left_mean = np.divide(left_sum, count, out=np.full_like(left_sum, np.nan), where=count > 0)
    right_mean = np.divide(right_sum, count, out=np.full_like(right_sum, np.nan), where=count > 0)
    left_centered = np.where(valid, left_arr - left_mean, 0.0)
    right_centered = np.where(valid, right_arr - right_mean, 0.0)
    numerator = np.sum(left_centered * right_centered, axis=dim, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=dim, keepdims=True)
        * np.sum(right_centered * right_centered, axis=dim, keepdims=True)
    )
    out = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=(count >= 2) & (denominator > 0))
    return out if keepdims else np.squeeze(out, axis=dim)
