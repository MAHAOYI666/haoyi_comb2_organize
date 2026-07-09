from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from comb2_simbase import IndexMask, Memmaper2, fast
from comb2_simbase.benchmark import benchmark_returns_from_cache, load_index_benchmark


REAL_CACHE = Path("/root/ml-data1-pvc/factorsim_data/Cache")


def _write_memmaper2_fixture(base: Path, data: np.ndarray, index: np.ndarray, columns: np.ndarray, chunk_size: int = 2):
    base.mkdir(parents=True, exist_ok=True)
    chunks = 0
    for chunks, start in enumerate(range(0, data.shape[0], chunk_size)):
        block = np.memmap(base / f"{chunks}.ares", dtype=np.float64, mode="w+", shape=data[start : start + chunk_size].shape)
        block[:] = data[start : start + chunk_size]
        block.flush()
        del block
    meta = np.array([np.float64, 1, data.shape[0], data.shape[1], chunk_size, chunks + 1], dtype=object)
    np.save(base / "meta.npy", meta)
    np.save(base / "index.npy", index.astype(np.int64))
    np.save(base / "columns.npy", columns)


def test_memmaper2_loads_exact_2d_slices(tmp_path: Path):
    data = np.arange(15, dtype=np.float64).reshape(5, 3)
    index = np.array([20200101, 20200102, 20200106, 20200107, 20200108])
    columns = np.array(["000001", "000002", "000003"], dtype=object)
    _write_memmaper2_fixture(tmp_path, data, index, columns, chunk_size=2)

    mmap = Memmaper2(tmp_path)
    np.testing.assert_array_equal(mmap.load(20200102, 20200107)[:], data[1:4])

    frame = mmap.load(20200102, 20200107, df_type=True).dloc[:]
    expected = pd.DataFrame(data[1:4], index=index[1:4], columns=columns)
    pd.testing.assert_frame_equal(frame, expected)

    frame_from_cli_alias = mmap.load(20200102, 20200107, df_type="df")[:]
    pd.testing.assert_frame_equal(frame_from_cli_alias, expected)


def test_index_mask_matches_local_simbase_arrays():
    mask = IndexMask()
    assert mask.date[0] == 20100104
    assert mask.date[-1] == 20261230
    assert str(mask.code[0]).zfill(6) == "000001"
    assert mask.code2cidx("000001") == 0
    assert mask.didx2date(mask.date2didx(20100105)) == 20100105

    simbase_dir = Path("/root/autodl/simbase/IndexMask/index_mask/memmap_mask")
    if simbase_dir.exists():
        np.testing.assert_array_equal(mask.date, np.load(simbase_dir / "DateRange.FactorSim.npy"))
        np.testing.assert_array_equal(mask.time, np.load(simbase_dir / "TimeRange.FactorSim.npy"))
        np.testing.assert_array_equal(mask.code, np.load(simbase_dir / "CodeRange.FactorSim.npy", allow_pickle=True))


def test_fast_helpers_match_pandas_numpy_reference():
    left = np.array([[1.0, 2.0, np.nan, 4.0], [1.0, 1.0, 1.0, np.inf]])
    right = np.array([[1.0, 4.0, 3.0, 8.0], [2.0, 3.0, 4.0, 5.0]])

    purified = fast.purify(torch.tensor(left))
    assert torch.isnan(purified[1, 3])

    expected_rank = pd.DataFrame(left).rank(axis=1, method="first").to_numpy()
    np.testing.assert_allclose(fast.rank(left, dim=-1), expected_rank, equal_nan=True)

    expected_pct_rank = pd.DataFrame(left).rank(axis=1, pct=True, method="first").to_numpy()
    np.testing.assert_allclose(fast.rank(left, dim=-1, pct=True), expected_pct_rank, equal_nan=True)

    np.testing.assert_allclose(
        fast.perc_long(np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, np.nan]])),
        np.array([[-1.0, -1.0, 0.5, 1.5], [0.0, 0.0, 1.0, np.nan]]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        fast.perc_long(np.array([[1.0, 2.0, 3.0, 4.0]]), percentile=0.25),
        np.array([[-3.75, 0.25, 1.25, 2.25]]),
    )

    valid = np.isfinite(left[0]) & np.isfinite(right[0])
    expected_corr = np.corrcoef(left[0, valid], right[0, valid])[0, 1]
    actual = fast.corr(left, right, dim=-1, keepdims=True)
    np.testing.assert_allclose(actual[0, 0], expected_corr)
    assert np.isnan(actual[1, 0])


@pytest.mark.skipif(not (REAL_CACHE / "AshareCache").exists(), reason="local AshareCache is unavailable")
def test_memmaper2_reads_real_local_cache():
    path = REAL_CACHE / "AshareCache" / "1d_DailyKline" / "DailyKline.close"
    frame = Memmaper2(path).load(20200102, 20200106, df_type=True).dloc[:]
    assert list(frame.index.astype(int)) == [20200102, 20200103, 20200106]
    assert frame.shape[1] == len(IndexMask().code)
    assert "000001" in frame.columns

@pytest.mark.skipif(not (REAL_CACHE / "AshareCache").exists(), reason="local AshareCache is unavailable")
def test_local_benchmark_returns_match_manual_index_weight_formula():
    dates = [20200102, 20200103, 20200106]
    bench = load_index_benchmark(REAL_CACHE, dates[0], dates[-1], ts_code="000905.SH")
    returns = bench["close"].pct_change().fillna(0.0).to_numpy()

    weights = Memmaper2(REAL_CACHE / "AshareCache" / "1d_IndexWeight" / "IndexWeight.000905.SH").load(dates[0], dates[-1], df_type=True).dloc[:].astype(float)
    close = Memmaper2(REAL_CACHE / "AshareCache" / "1d_DailyKline" / "DailyKline.close").load(dates[0], dates[-1], df_type=True).dloc[:].astype(float)
    prev_close = Memmaper2(REAL_CACHE / "AshareCache" / "1d_DailyKline" / "DailyKline.real_pre_close").load(dates[0], dates[-1], df_type=True).dloc[:].astype(float)
    stock_ret = close / prev_close - 1.0

    for row_idx in (1, 2):
        w = weights.iloc[row_idx].to_numpy(dtype=float)
        r = stock_ret.iloc[row_idx].to_numpy(dtype=float)
        valid = np.isfinite(w) & np.isfinite(r) & (w != 0)
        expected = np.sum(w[valid] * r[valid]) / np.sum(w[valid])
        np.testing.assert_allclose(returns[row_idx], expected, rtol=1e-12, atol=1e-12)

    series = benchmark_returns_from_cache(REAL_CACHE, dates)
    np.testing.assert_allclose(series.to_numpy(), returns)


@pytest.mark.skipif(not (REAL_CACHE / "AshareCache").exists(), reason="local AshareCache is unavailable")
def test_local_benchmark_uses_raw_previous_close_instead_of_adjusted_previous_close():
    dates = [20150602, 20150603, 20150604]
    bench = load_index_benchmark(REAL_CACHE, dates[0], dates[-1], ts_code="000905.SH")
    actual = bench["close"].pct_change().fillna(0.0).to_numpy()

    weights = Memmaper2(REAL_CACHE / "AshareCache" / "1d_IndexWeight" / "IndexWeight.000905.SH").load(dates[0], dates[-1], df_type=True).dloc[:].astype(float)
    close = Memmaper2(REAL_CACHE / "AshareCache" / "1d_DailyKline" / "DailyKline.close").load(dates[0], dates[-1], df_type=True).dloc[:].astype(float)
    preclose = Memmaper2(REAL_CACHE / "AshareCache" / "1d_DailyKline" / "DailyKline.pre_close").load(dates[0], dates[-1], df_type=True).dloc[:].astype(float)

    adjusted = []
    for row_idx in range(len(dates)):
        w = weights.iloc[row_idx].to_numpy(dtype=float)
        r = (close.iloc[row_idx].to_numpy(dtype=float) / preclose.iloc[row_idx].to_numpy(dtype=float)) - 1.0
        valid = np.isfinite(w) & np.isfinite(r) & (w != 0)
        denom = np.sum(w[valid])
        adjusted.append(np.sum(w[valid] * r[valid]) / denom if denom != 0 else np.nan)
    adjusted = np.asarray(adjusted, dtype=float)

    assert not np.allclose(actual[1:], adjusted[1:])
