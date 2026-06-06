from __future__ import annotations

import torch


def _default_eps(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-4
    return 1e-8


def nan_to_num(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=value, posinf=value, neginf=value)


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
    fitted = (x_mat @ beta).squeeze(-1)
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
