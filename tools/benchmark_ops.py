"""Worker-only old/new framework rank and z-score microbenchmarks."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/comb2"))

import numpy as np
import torch
from comb2.op_utils import nanmean, nanstd, rank, zscore


def old_rank(x, dim=0, pct=False):
    moved = np.moveaxis(x.detach().cpu().numpy(), dim, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    ranked = np.full_like(flat, np.nan, dtype=np.float64)
    for index, row in enumerate(flat):
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
        ranked[index, valid] = row_rank
    return torch.as_tensor(
        np.moveaxis(ranked.reshape(moved.shape), -1, dim),
        dtype=x.dtype, device=x.device,
    )


def old_zscore(x, dim, eps):
    mean = nanmean(x, dim=dim, keepdim=True)
    std = nanstd(x, dim=dim, keepdim=True)
    std = torch.where(torch.isfinite(std) & (std > 0), std, torch.zeros_like(std))
    return (x - mean) / (std + eps)


def elapsed(fn, x, repeats=3):
    fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        began = time.perf_counter()
        fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    return statistics.median(times)


def compare(left, right, atol, rtol):
    close = torch.isclose(left, right, atol=atol, rtol=rtol, equal_nan=True)
    return {
        "parity": bool(close.all()),
        "different_elements": int((~close).sum()),
        "elements": left.numel(),
    }


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    results = []
    devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
    for device in devices:
        for dtype in (torch.float16, torch.float32):
            x = torch.randn((5642, 689), dtype=dtype, device=device)
            x[::97, ::17] = torch.nan
            x[::401, ::37] = torch.inf
            for axis in (0, 1):
                for operation, old, new, atol, rtol in (
                    ("rank", lambda value: old_rank(value, axis), lambda value: rank(value, dim=axis), 0, 0),
                    ("rank_pct", lambda value: old_rank(value, axis, True), lambda value: rank(value, dim=axis, pct=True), 0.001 if dtype == torch.float16 else 1e-6, 0),
                    ("zscore", lambda value: old_zscore(value, axis, 1e-4 if dtype == torch.float16 else 1e-8), lambda value: zscore(value, dim=axis), 0.002 if dtype == torch.float16 else 1e-5, 0.005 if dtype == torch.float16 else 1e-4),
                ):
                    expected = old(x)
                    actual = new(x)
                    parity = compare(expected, actual, atol, rtol)
                    old_seconds = elapsed(old, x)
                    new_seconds = elapsed(new, x)
                    results.append({
                        "device": device, "dtype": str(dtype), "axis": axis,
                        "operation": operation,
                        "old_median_seconds": round(old_seconds, 5),
                        "new_median_seconds": round(new_seconds, 5),
                        "speedup": round(old_seconds / new_seconds, 2),
                        **parity,
                    })
                    print(json.dumps(results[-1]), flush=True)
    assert all(row["parity"] for row in results), "operator parity failed"


if __name__ == "__main__":
    main()
