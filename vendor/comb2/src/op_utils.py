from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
import torch


def _default_eps(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-4
    return 1e-8


def nan_to_num(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)


def purify(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[torch.isinf(y)] = torch.nan
    return y


def rank(x: torch.Tensor, dim: int = 0, pct: bool = False) -> torch.Tensor:
    axis = dim if dim >= 0 else x.ndim + dim
    ranked = pd.DataFrame(x.detach().cpu().numpy()).rank(axis=axis, pct=pct, method="first").to_numpy()
    return torch.as_tensor(ranked, dtype=x.dtype, device=x.device)


def perc_long(x: torch.Tensor, percentile: float = 0.5) -> torch.Tensor:
    arr = x.detach().cpu().numpy()
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
    out = np.where(np.isfinite(arr) & np.isfinite(threshold), out, np.nan)
    return torch.as_tensor(out, dtype=x.dtype, device=x.device)


def corr(left: torch.Tensor, right: torch.Tensor, dim: int = -1, keepdims: bool = False) -> torch.Tensor:
    left = left.to(torch.float32)
    right = right.to(torch.float32)
    valid = torch.isfinite(left) & torch.isfinite(right)
    count = valid.sum(dim=dim, keepdim=True)
    left_mean = torch.where(valid, left, torch.nan).nanmean(dim=dim, keepdim=True)
    right_mean = torch.where(valid, right, torch.nan).nanmean(dim=dim, keepdim=True)
    left_centered = torch.where(valid, left - left_mean, torch.zeros_like(left))
    right_centered = torch.where(valid, right - right_mean, torch.zeros_like(right))
    numerator = (left_centered * right_centered).sum(dim=dim, keepdim=True)
    denominator = torch.sqrt(
        (left_centered * left_centered).sum(dim=dim, keepdim=True)
        * (right_centered * right_centered).sum(dim=dim, keepdim=True)
    )
    out = numerator / denominator
    out = torch.where((count >= 2) & (denominator > 0), out, torch.full_like(out, torch.nan))
    return out if keepdims else out.squeeze(dim)


def _nan_masked(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ~torch.isnan(x)
    safe_x = torch.where(mask, x, torch.zeros_like(x))
    return safe_x, mask


def nanmean(x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    safe_x, mask = _nan_masked(x)
    count = mask.sum(dim=dim, keepdim=keepdim)
    total = safe_x.sum(dim=dim, keepdim=keepdim)
    denom = torch.clamp(count, min=1).to(dtype=x.dtype)
    mean = total / denom
    nan_fill = torch.full_like(mean, torch.nan)
    return torch.where(count > 0, mean, nan_fill)


def nanstd(x: torch.Tensor, dim=None, keepdim: bool = False) -> torch.Tensor:
    mean = nanmean(x, dim=dim, keepdim=True)
    diff = x - mean
    diff = torch.where(torch.isnan(x), torch.zeros_like(diff), diff)
    count = (~torch.isnan(x)).sum(dim=dim, keepdim=True)
    denom = torch.clamp(count, min=1).to(dtype=x.dtype)
    var = diff.pow(2).sum(dim=dim, keepdim=True) / denom
    std = torch.sqrt(torch.clamp(var, min=0.0))
    if not keepdim and dim is not None:
        std = std.squeeze(dim)
        count = count.squeeze(dim)
    nan_fill = torch.full_like(std, torch.nan)
    return torch.where(count > 0, std, nan_fill)


def nanmedian(x: torch.Tensor) -> torch.Tensor:
    valid = x[~torch.isnan(x)]
    if valid.numel() == 0:
        return torch.tensor(torch.nan, device=x.device, dtype=x.dtype)
    return torch.median(valid)


def zscore(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    eps = _default_eps(x.dtype) if eps is None else eps
    mean = nanmean(x)
    std = nanstd(x)
    if (not torch.isfinite(std)) or std <= 0:
        std = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return (x - mean) / (std + eps)


def cs_zscore(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    eps = _default_eps(x.dtype) if eps is None else eps
    mean = nanmean(x, dim=-1, keepdim=True)
    std = nanstd(x, dim=-1, keepdim=True)
    std = torch.where(torch.isfinite(std) & (std > 0), std, torch.zeros_like(std))
    return (x - mean) / (std + eps)


def neut(
    y: torch.Tensor,
    xs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    intercept: bool = True,
    ratio: float | Sequence[float] = 1.0,
    eps: float | None = None,
) -> torch.Tensor:
    eps = _default_eps(y.dtype) if eps is None else eps
    if isinstance(xs, torch.Tensor):
        xs_list = [xs]
    else:
        xs_list = list(xs)
    if not xs_list:
        return y

    target_shape = y.shape
    cast_dtype = y.dtype
    device = y.device

    aligned_xs = []
    valid = torch.isfinite(y)
    for x in xs_list:
        aligned = torch.broadcast_to(x.to(device=device, dtype=cast_dtype), target_shape)
        aligned_xs.append(aligned)
        valid = valid & torch.isfinite(aligned)

    x_mat = torch.stack(aligned_xs, dim=-1)
    x_mat = torch.where(valid.unsqueeze(-1), x_mat, torch.zeros_like(x_mat))
    y_vec = torch.where(valid, y, torch.zeros_like(y)).unsqueeze(-1)

    if intercept:
        ones = valid.to(dtype=cast_dtype).unsqueeze(-1)
        x_mat = torch.cat([ones, x_mat], dim=-1)

    xt = x_mat.transpose(-1, -2)
    gram = xt @ x_mat
    rhs = xt @ y_vec
    eye = torch.eye(gram.shape[-1], device=device, dtype=cast_dtype)
    beta = torch.linalg.pinv(gram + eye * eps) @ rhs
    if isinstance(ratio, Sequence) and not isinstance(ratio, (str, bytes)):
        ratio_tensor = torch.as_tensor(list(ratio), device=device, dtype=cast_dtype)
        if ratio_tensor.numel() == len(xs_list):
            if intercept:
                ratio_tensor = torch.cat([torch.ones(1, device=device, dtype=cast_dtype), ratio_tensor])
        elif ratio_tensor.numel() != x_mat.shape[-1]:
            raise ValueError(
                f"neut ratio length must be {len(xs_list)}"
                f"{f' or {len(xs_list) + 1}' if intercept else ''}, got {ratio_tensor.numel()}"
            )
        fitted = ((x_mat * ratio_tensor) @ beta).squeeze(-1)
    else:
        fitted = (x_mat @ beta).squeeze(-1) * float(ratio)
    residual = y - fitted
    return torch.where(valid, residual, torch.full_like(y, torch.nan))


def truncate(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.clamp(x, lower, upper)


def winsorize_by_quantile(x: torch.Tensor, lower_q: float = 0.01, upper_q: float = 0.99) -> torch.Tensor:
    valid = x[~torch.isnan(x)]
    if valid.numel() == 0:
        return x
    low = torch.quantile(valid, lower_q)
    high = torch.quantile(valid, upper_q)
    return torch.clamp(x, low, high)


def normalize_by_max_abs(x: torch.Tensor, eps: float | None = None) -> torch.Tensor:
    eps = _default_eps(x.dtype) if eps is None else eps
    max_abs = torch.max(torch.abs(x))
    if (not torch.isfinite(max_abs)) or max_abs <= 0:
        return x
    return x / (max_abs + eps)


def to_bool_mask(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x
    return nan_to_num(x, 0.0) > 0
