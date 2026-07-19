from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np
import torch


def _resolve_axis(dim: int | None, axis: int | None, *, ndim: int | None = None, default: int | None = None) -> int | None:
    if axis is not None and dim is not None and int(axis) != int(dim):
        raise ValueError(f"conflicting dim/axis values: dim={dim}, axis={axis}")
    resolved = axis if axis is not None else dim
    if resolved is None:
        resolved = default
    if resolved is None:
        return None
    resolved = int(resolved)
    if ndim is not None:
        if resolved < 0:
            resolved += ndim
        if resolved < 0 or resolved >= ndim:
            raise IndexError(f"axis {resolved} is out of bounds for tensor with ndim={ndim}")
    return resolved


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


def rank(x: torch.Tensor, dim: int = 0, pct: bool = False, axis: int | None = None) -> torch.Tensor:
    rank_axis = _resolve_axis(dim, axis, ndim=x.ndim, default=0)
    moved = np.moveaxis(x.detach().cpu().numpy(), rank_axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    ranked = np.full_like(flat, np.nan, dtype=np.float64)
    for idx, row in enumerate(flat):
        valid = np.isfinite(row)
        count = int(valid.sum())
        if count == 0:
            continue
        order = np.argsort(row[valid], kind="mergesort")
        values = np.arange(1, count + 1, dtype=np.float64)
        if pct:
            values /= count
        row_rank = np.empty(count, dtype=np.float64)
        row_rank[order] = values
        ranked[idx, valid] = row_rank
    ranked = ranked.reshape(moved.shape)
    ranked = np.moveaxis(ranked, -1, rank_axis)
    return torch.as_tensor(ranked, dtype=x.dtype, device=x.device)


def perc_long(x: torch.Tensor, percentile: float = 0.5, axis: int = -1) -> torch.Tensor:
    axis = _resolve_axis(None, axis, ndim=x.ndim, default=-1)
    arr = x.detach().cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        threshold = np.nanquantile(arr, percentile, axis=axis, keepdims=True)
    out = arr - threshold
    high = arr > threshold
    low = arr < threshold
    positive_sum = np.where(high, out, 0.0).sum(axis=axis, keepdims=True)
    low_count = low.sum(axis=axis, keepdims=True)
    replacement = np.divide(-positive_sum, low_count, out=np.zeros_like(positive_sum), where=low_count > 0)
    out = np.where(low & (low_count > 0), replacement, out)
    out = np.where(np.isfinite(arr) & np.isfinite(threshold), out, np.nan)
    return torch.as_tensor(out, dtype=x.dtype, device=x.device)


def corr(
    left: torch.Tensor,
    right: torch.Tensor,
    dim: int = -1,
    keepdims: bool = False,
    axis: int | None = None,
) -> torch.Tensor:
    corr_axis = _resolve_axis(dim, axis, ndim=left.ndim, default=-1)
    left = left.to(torch.float32)
    right = right.to(torch.float32)
    valid = torch.isfinite(left) & torch.isfinite(right)
    count = valid.sum(dim=corr_axis, keepdim=True)
    left_mean = torch.where(valid, left, torch.nan).nanmean(dim=corr_axis, keepdim=True)
    right_mean = torch.where(valid, right, torch.nan).nanmean(dim=corr_axis, keepdim=True)
    left_centered = torch.where(valid, left - left_mean, torch.zeros_like(left))
    right_centered = torch.where(valid, right - right_mean, torch.zeros_like(right))
    numerator = (left_centered * right_centered).sum(dim=corr_axis, keepdim=True)
    denominator = torch.sqrt(
        (left_centered * left_centered).sum(dim=corr_axis, keepdim=True)
        * (right_centered * right_centered).sum(dim=corr_axis, keepdim=True)
    )
    out = numerator / denominator
    out = torch.where((count >= 2) & (denominator > 0), out, torch.full_like(out, torch.nan))
    return out if keepdims else out.squeeze(corr_axis)


def _nan_masked(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ~torch.isnan(x)
    safe_x = torch.where(mask, x, torch.zeros_like(x))
    return safe_x, mask


def nanmean(x: torch.Tensor, dim=None, keepdim: bool = False, axis: int | None = None) -> torch.Tensor:
    reduce_axis = _resolve_axis(dim, axis, ndim=x.ndim)
    safe_x, mask = _nan_masked(x)
    count = mask.sum(dim=reduce_axis, keepdim=keepdim)
    total = safe_x.sum(dim=reduce_axis, keepdim=keepdim)
    denom = torch.clamp(count, min=1).to(dtype=x.dtype)
    mean = total / denom
    nan_fill = torch.full_like(mean, torch.nan)
    return torch.where(count > 0, mean, nan_fill)


def nanstd(x: torch.Tensor, dim=None, keepdim: bool = False, axis: int | None = None) -> torch.Tensor:
    reduce_axis = _resolve_axis(dim, axis, ndim=x.ndim)
    mean = nanmean(x, dim=reduce_axis, keepdim=True)
    diff = x - mean
    diff = torch.where(torch.isnan(x), torch.zeros_like(diff), diff)
    count = (~torch.isnan(x)).sum(dim=reduce_axis, keepdim=True)
    denom = torch.clamp(count, min=1).to(dtype=x.dtype)
    var = diff.pow(2).sum(dim=reduce_axis, keepdim=True) / denom
    std = torch.sqrt(torch.clamp(var, min=0.0))
    if not keepdim and reduce_axis is not None:
        std = std.squeeze(reduce_axis)
        count = count.squeeze(reduce_axis)
    nan_fill = torch.full_like(std, torch.nan)
    return torch.where(count > 0, std, nan_fill)


def nanmedian(x: torch.Tensor, dim: int | None = None, keepdim: bool = False, axis: int | None = None) -> torch.Tensor:
    reduce_axis = _resolve_axis(dim, axis, ndim=x.ndim)
    if reduce_axis is None:
        valid = x[~torch.isnan(x)]
        if valid.numel() == 0:
            return torch.tensor(torch.nan, device=x.device, dtype=x.dtype)
        return torch.median(valid)
    return torch.nanmedian(x, dim=reduce_axis, keepdim=keepdim).values


def zscore(x: torch.Tensor, eps: float | None = None, dim: int | None = None, axis: int | None = None) -> torch.Tensor:
    eps = _default_eps(x.dtype) if eps is None else eps
    reduce_axis = _resolve_axis(dim, axis, ndim=x.ndim)
    keepdim = reduce_axis is not None
    mean = nanmean(x, dim=reduce_axis, keepdim=keepdim)
    std = nanstd(x, dim=reduce_axis, keepdim=keepdim)
    std = torch.where(torch.isfinite(std) & (std > 0), std, torch.zeros_like(std))
    return (x - mean) / (std + eps)


def cs_zscore(x: torch.Tensor, eps: float | None = None, axis: int = -1) -> torch.Tensor:
    return zscore(x, eps=eps, axis=axis)


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


def winsorize_by_quantile(
    x: torch.Tensor,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    axis: int | None = None,
) -> torch.Tensor:
    reduce_axis = _resolve_axis(None, axis, ndim=x.ndim)
    if reduce_axis is None:
        valid = x[~torch.isnan(x)]
        if valid.numel() == 0:
            return x
        low = torch.quantile(valid, lower_q)
        high = torch.quantile(valid, upper_q)
    else:
        arr = x.detach().cpu().numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            low = np.nanquantile(arr, lower_q, axis=reduce_axis, keepdims=True)
            high = np.nanquantile(arr, upper_q, axis=reduce_axis, keepdims=True)
        low = torch.as_tensor(low, dtype=x.dtype, device=x.device)
        high = torch.as_tensor(high, dtype=x.dtype, device=x.device)
    return torch.maximum(torch.minimum(x, high), low)


def normalize_by_max_abs(x: torch.Tensor, eps: float | None = None, axis: int | None = None) -> torch.Tensor:
    eps = _default_eps(x.dtype) if eps is None else eps
    reduce_axis = _resolve_axis(None, axis, ndim=x.ndim)
    finite = torch.isfinite(x)
    safe_abs = torch.where(finite, torch.abs(x), torch.zeros_like(x))
    max_abs = torch.max(safe_abs) if reduce_axis is None else safe_abs.amax(dim=reduce_axis, keepdim=True)
    has_finite = finite.any() if reduce_axis is None else finite.any(dim=reduce_axis, keepdim=True)
    if reduce_axis is None:
        if (not torch.isfinite(max_abs)) or max_abs <= 0 or (not bool(has_finite)):
            return x
        return x / (max_abs + eps)
    scale_valid = has_finite & torch.isfinite(max_abs) & (max_abs > 0)
    if not torch.any(scale_valid):
        return x
    scale = torch.where(scale_valid, max_abs + eps, torch.ones_like(max_abs))
    out = x / scale
    return torch.where(scale_valid, out, x)


def _rolling_reduce(x: torch.Tensor, window: int, axis: int, reducer, name: str) -> torch.Tensor:
    axis = _resolve_axis(None, axis, ndim=x.ndim, default=0)
    window = int(window)
    if window <= 0:
        raise ValueError(f"{name} window must be positive")
    moved = torch.movedim(x, axis, 0)
    out = torch.full_like(moved, torch.nan)
    for idx in range(moved.shape[0]):
        lo = max(0, idx - window + 1)
        out[idx] = reducer(moved[lo : idx + 1], dim=0)
    return torch.movedim(out, 0, axis)


def rolling_mean(x: torch.Tensor, window: int, axis: int = 0) -> torch.Tensor:
    return _rolling_reduce(x, window, axis, nanmean, "rolling_mean")


def rolling_std(x: torch.Tensor, window: int, axis: int = 0) -> torch.Tensor:
    return _rolling_reduce(x, window, axis, nanstd, "rolling_std")


def to_bool_mask(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x
    return nan_to_num(x, 0.0) > 0
