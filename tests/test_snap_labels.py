from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from comb2.DataRegistry import DataItem, DataRegistry, Universe
from comb2_simbase.snap_labels import load_snap_vwap_labels
from config import load_config
from evals.comb_eval.report import _resolve_artifacts


def _write_memmaper2(base: Path, data: np.ndarray, index: np.ndarray, columns: np.ndarray) -> None:
    base.mkdir(parents=True, exist_ok=True)
    block = np.memmap(base / "0.ares", dtype=np.float64, mode="w+", shape=data.shape)
    block[:] = data
    block.flush()
    del block
    np.save(base / "meta.npy", np.array([np.float64, 1, data.shape[0], data.shape[1], data.shape[0], 1], dtype=object))
    np.save(base / "index.npy", index)
    np.save(base / "columns.npy", columns)


def test_snapshot_labels_use_matching_price_field_formula_and_mask(tmp_path: Path):
    dates = np.array([20240102, 20240103, 20240104, 20240105, 20240108, 20240109, 20240110])
    columns = np.array(["000001", "000002"], dtype=object)
    root = tmp_path / "AshareCache"
    price = np.array([[10.0, 20.0], [11.0, 22.0], [12.0, 24.0], [13.0, 26.0], [14.0, 28.0], [15.0, 30.0], [16.0, 32.0]])
    close = np.tile(np.array([[10.0, 20.0]]), (len(dates), 1))
    ones = np.ones_like(price)

    _write_memmaper2(root / "1d_IntraVwap" / "IntraVwap.Vwap30.100000", price, dates, columns)
    _write_memmaper2(root / "1d_DailyKline" / "DailyKline.close_hfq", close, dates, columns)
    _write_memmaper2(root / "1d_DailyKline" / "DailyKline.adj_factor", ones, dates, columns)
    _write_memmaper2(root / "1d_StockMask2" / "StockMask2.BaseUnivMask", ones, dates, columns)
    limit = ones.copy()
    limit[1, 1] = 0.0
    _write_memmaper2(root / "1d_StockMask2" / "StockMask2.LimitMask", limit, dates, columns)

    labels = load_snap_vwap_labels(tmp_path, 100000, 20240102, 20240102)

    assert list(labels[1].index) == [20240102]
    np.testing.assert_allclose(labels[1].iloc[0, 0], 10.0 / 11.0 - 1.0 + 12.0 / 10.0 - 1.0)
    np.testing.assert_allclose(labels[5].iloc[0, 0], 16.0 / 11.0 - 1.0)
    assert labels[1].iloc[0, 1] == 0.0
    assert labels[5].iloc[0, 1] == 0.0


def test_registry_snapshot_label_is_an_ordinary_source(tmp_path):
    from comb2.DataRegistry import FactorsimReader
    dates = np.array([20240102,20240103,20240104,20240105,20240108,20240109,20240110])
    codes = np.array(["000001","000002"], dtype=object)
    root = tmp_path / "AshareCache"
    prices = np.arange(14).reshape(7,2) + 10.
    for relative, values in (
        ("1d_IntraVwap/IntraVwap.Vwap30.100000", prices),
        ("1d_DailyKline/DailyKline.close_hfq", prices + 1),
        ("1d_DailyKline/DailyKline.adj_factor", np.ones_like(prices)),
        ("1d_StockMask2/StockMask2.BaseUnivMask", np.ones_like(prices)),
        ("1d_StockMask2/StockMask2.LimitMask", np.ones_like(prices)),
    ):
        _write_memmaper2(root / relative, values, dates, codes)
    registry = DataRegistry(
        (DataItem("y", module="builtin.snap_label"),),
        universe=Universe(tuple(dates), tuple(codes), torch.float32),
        data_start_ds=int(dates[0]), ashare_cache_path=str(root),
    )
    registry.set_current_ti(100000)
    actual = registry.get_field("y","y",20240102,20240103)
    expected = load_snap_vwap_labels(tmp_path,100000,20240102,20240103)[1]
    np.testing.assert_allclose(actual, expected)


def test_evaluation_prefetches_snapshot_labels_and_reuses_inputs(tmp_path):
    import cProfile
    import pstats
    from types import SimpleNamespace
    from comb2 import ComboDataLoader, LoaderConfig
    from comb2_simbase import IndexMask
    from runCombo import calculate_alpha_ic

    dates = np.asarray(IndexMask().intv_trade_day(20230103, 20230531), dtype=int)[:90]
    codes = np.array(["000001", "000002"], dtype=object)
    prices = np.arange(len(dates) * 2).reshape(-1, 2) / 100 + 10
    root = tmp_path / "AshareCache"
    for relative, values in (
        ("1d_IntraVwap/IntraVwap.Vwap30.100000", prices),
        ("1d_IntraVwap/IntraVwap.Vwap30.110000", prices + .5),
        ("1d_DailyKline/DailyKline.close_hfq", prices + 1),
        ("1d_DailyKline/DailyKline.adj_factor", np.ones_like(prices)),
        ("1d_StockMask2/StockMask2.BaseUnivMask", np.ones_like(prices)),
        ("1d_StockMask2/StockMask2.LimitMask", np.ones_like(prices)),
        ("1d_StockMask2/StockMask2.NoNewStockMask", np.ones_like(prices)),
    ):
        _write_memmaper2(root / relative, values, dates, codes)

    class ResearchLoader(ComboDataLoader):
        def data_requirements(self):
            return (DataItem("label", module="builtin.snap_label", delay=1),
                    DataItem("price", path=str(root / "1d_IntraVwap" / "IntraVwap.Vwap30.{ti:06d}")))

        def model_input_sources(self):
            return ("price",)

        def model_target(self):
            return "label", "label"

    loader = ResearchLoader(LoaderConfig(cache_path=str(tmp_path), data_start_ds=int(dates[1]),
                                        sample_times=(100000, 110000), registry_cache_days=64))
    alpha = pd.DataFrame(np.tile(np.arange(len(loader.mask.code)), (144, 1)),
                         index=pd.MultiIndex.from_product([dates[1:73], loader.sample_times], names=["date", "time"]))
    sample_inputs = {}
    with cProfile.Profile() as profile:
        result = calculate_alpha_ic(alpha, SimpleNamespace(loader=loader), sample_inputs=sample_inputs)
    reads = sum(v[1] for (filename, _, name), v in pstats.Stats(profile).stats.items()
                if filename.endswith("snap_labels.py") and name == "_load_frame")
    assert reads == 20
    assert (result["count"] == 2).all() and len(sample_inputs) == 144
    for ti in loader.sample_times:
        expected = load_snap_vwap_labels(tmp_path, ti, int(dates[0]), int(dates[71]))[1]
        for offset, ds in enumerate(dates[1:73]):
            target, valid = sample_inputs[(int(ds), ti)]
            np.testing.assert_allclose(target[:2], expected.iloc[offset].to_numpy(dtype=np.float32), rtol=0, atol=0)
            assert valid[:2].all() and not valid[2:].any()
            np.testing.assert_allclose(result.loc[(ds, ti), "ic"], np.corrcoef([0., 1.], target[:2])[0, 1])
