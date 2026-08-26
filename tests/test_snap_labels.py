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


def test_config_routes_default_training_label_to_snapshot_source(tmp_path: Path):
    config_path = tmp_path / "config.xml"
    config_path.write_text(
        """
<config>
  <constants cache_path="cache" output_root="output" />
  <combo>
    <runtime snap_ti="100000" />
    <data>
      <item name="factor.close" module="builtin.factorsim" path="factor" role="factor" />
      <item name="label.default" module="builtin.factorsim" path="vwap30_label1d" role="label" />
    </data>
  </combo>
</config>
""",
        encoding="utf-8",
    )

    loaded = load_config(str(config_path))
    label = next(item for item in loaded["combo"]["loader"]["data_items"] if item["role"] == "label")

    assert label["module"] == "builtin.snap_label"
    assert label["path"] is None
    assert label["params"]["snap_ti"] == 100000


def test_registry_loads_snapshot_label_for_training(monkeypatch, tmp_path: Path):
    dates = (20240102, 20240103)
    codes = ("000001", "000002")
    expected = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=dates, columns=codes)
    data_registry = importlib.import_module("comb2.DataRegistry")
    calls = []

    def fake_load(cache_path, snap_ti, start_ds, end_ds):
        calls.append((Path(cache_path), snap_ti, start_ds, end_ds))
        return {1: expected, 5: expected}

    monkeypatch.setattr(data_registry, "load_snap_vwap_labels", fake_load)
    registry = DataRegistry(
        (DataItem(name="label.default", module="builtin.snap_label", role="label", params={"snap_ti": 100000}),),
        universe=Universe(dates=dates, codes=codes, dtype=torch.float32),
        data_start_ds=dates[0],
        ashare_cache_path=str(tmp_path / "cache" / "AshareCache"),
        config_path=None,
    )

    actual = registry.get_data("label.default", dates[0], dates[-1])

    assert calls == [(tmp_path / "cache", 100000, dates[0], dates[-1])]
    np.testing.assert_allclose(actual.numpy(), expected.to_numpy())


def test_config_evaluation_uses_snapshot_labels_without_explicit_overrides(tmp_path: Path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    pd.DataFrame([[1.0]], index=[20240102], columns=["000001"]).to_parquet(output_root / "alpha.parquet")
    config = {
        "constants": {"cache_path": str(tmp_path / "cache"), "output_root": str(output_root)},
        "combo": {"runtime": {"snap_ti": 100000}},
        "backtest": {"cash": 1e7, "fee_rate": 0.0},
    }

    artifacts = _resolve_artifacts(
        tmp_path / "config.xml",
        config,
        report_dir=None,
        plot_path=None,
        label_path=None,
        label_5d_path=None,
        label_is_table=False,
        label_5d_is_table=False,
        label_df_type=True,
        booksize=None,
        tradecost_ratio=None,
    )

    assert artifacts.snap_ti == 100000
    assert artifacts.label_path is None
    assert artifacts.label_5d_path is None
