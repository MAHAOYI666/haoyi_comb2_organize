from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import runCombo
from comb2_pcmaster import BacktestNode, DailyBacktest
from comb2_pcmaster import dataloader as dataloader_module
from comb2_pcmaster import default_strategy as default_strategy_module
from comb2_pcmaster.default_strategy import AlphaStrategy, _normalize_alpha
from config import DEFAULT_OPTIMIZER_CONFIG


def test_get_vwap_uses_snap_ti_cache_field(monkeypatch, tmp_path: Path):
    paths: list[Path] = []

    class FakeDloc:
        def __getitem__(self, _):
            return pd.DataFrame({"000001": [1.0]}, index=[20240102])

    class FakeMemmaper:
        def __init__(self, path):
            paths.append(Path(path))

        def load(self, **_):
            return SimpleNamespace(dloc=FakeDloc())

    monkeypatch.setattr(dataloader_module, "Memmaper2", FakeMemmaper)
    loader = dataloader_module.DataLoader.__new__(dataloader_module.DataLoader)
    loader.ashare_cache_path = tmp_path / "AshareCache"

    loader.get_vwap(20240102, 20240102, snap_ti=100000)

    assert paths == [tmp_path / "AshareCache" / "1d_IntraVwap" / "IntraVwap.Vwap30.100000"]


def test_get_vwap_keeps_legacy_field_when_snap_ti_is_empty(monkeypatch, tmp_path: Path):
    paths: list[Path] = []

    class FakeDloc:
        def __getitem__(self, _):
            return pd.DataFrame({"000001": [1.0]}, index=[20240102])

    class FakeMemmaper:
        def __init__(self, path):
            paths.append(Path(path))

        def load(self, **_):
            return SimpleNamespace(dloc=FakeDloc())

    monkeypatch.setattr(dataloader_module, "Memmaper2", FakeMemmaper)
    loader = dataloader_module.DataLoader.__new__(dataloader_module.DataLoader)
    loader.ashare_cache_path = tmp_path / "AshareCache"

    loader.get_vwap(20240102, 20240102)

    assert paths == [tmp_path / "AshareCache" / "1d_IntraVwap" / "IntraVwap.VwapBegin30"]


def test_backtest_receives_runtime_snap_ti(tmp_path: Path):
    config = {
        "strategy": {
            "start_ds": 20240102,
            "end_ds": 20240103,
            "path": str(tmp_path / "strategy.py"),
        },
        "backtest": {
            "output_path": str(tmp_path / "output" / "backtest"),
            "cash": 10_000_000.0,
            "fee_rate": 0.0015,
            "reserve_cash": 0.95,
            "daily_metrics_file": "daily_pnl.csv",
            "verbose": False,
            "universe": "base",
            "execution_price": "vwap30",
            "drawdown_stop": 0.0,
            "cooldown_days": 0,
        },
        "constants": {"cache_path": str(tmp_path / "cache")},
        "combo": {"runtime": {"snap_ti": 100000}},
    }

    node = runCombo.build_backtest_node(tmp_path / "strategy.py", config)

    assert node.snap_ti == 100000


def test_daily_backtest_passes_snap_ti_to_vwap_loader():
    calls: list[tuple[int, int, int | None]] = []
    frame = pd.DataFrame({"000001": [1.0]}, index=[20240102])

    class FakeLoader:
        def get_preclose(self, start_ds, end_ds):
            return frame

        def get_vwap(self, start_ds, end_ds, snap_ti=None):
            calls.append((start_ds, end_ds, snap_ti))
            return frame

        def get_close(self, start_ds, end_ds):
            return frame

        def get_market_cap(self, start_ds, end_ds):
            return frame

        def get_suspend(self, start_ds, end_ds):
            return frame

        def get_limit(self, start_ds, end_ds):
            return frame

    backtest = DailyBacktest.__new__(DailyBacktest)
    backtest.node = SimpleNamespace(start_ds=20240102, end_ds=20240102, execution_price="vwap30", snap_ti=100000)
    backtest.dataloader = FakeLoader()

    backtest._load_market_data()

    assert calls == [(20240102, 20240102, 100000)]


def test_default_optimizer_alpha_normalization_matches_reference():
    values = np.array([-4.0, -1.0, 0.0, 1.0e-6, 1.0, 3.0, np.nan, np.inf])

    normalized = _normalize_alpha(values, trim_threshold=1.0e-5)

    np.testing.assert_allclose(
        normalized,
        np.array([-0.8, -0.2, 0.0, 0.0, 0.25, 0.75, 0.0, 0.0]),
    )
    assert normalized[normalized > 0].sum() == pytest.approx(1.0)
    assert normalized[normalized < 0].sum() == pytest.approx(-1.0)


def test_default_optimizer_normalizes_actual_holdings_to_stock_book():
    strategy = AlphaStrategy.__new__(AlphaStrategy)
    strategy.columns = pd.Index(["000001", "000002", "000003"])
    actual_holdings = pd.Series({"000001": 0.095, "000002": 0.855, "000003": 0.0})

    previous, has_previous = strategy._actual_previous(actual_holdings)

    assert has_previous is True
    np.testing.assert_allclose(previous, np.array([0.1, 0.9, 0.0]))


def test_real_default_optimizer_uses_actual_holdings_for_two_days(tmp_path: Path):
    cache_value = os.environ.get("COMB2_TEST_CACHE_PATH")
    alpha_value = os.environ.get("COMB2_TEST_ALPHA_PATH")
    license_value = os.environ.get("MOSEKLM_LICENSE_FILE")
    if not cache_value or not alpha_value or not license_value:
        pytest.skip(
            "set COMB2_TEST_CACHE_PATH, COMB2_TEST_ALPHA_PATH, and "
            "MOSEKLM_LICENSE_FILE for the real optimizer integration test"
        )
    cache_path = Path(cache_value)
    alpha_path = Path(alpha_value)
    license_path = Path(license_value)
    if not cache_path.is_dir() or not alpha_path.is_file() or not license_path.is_file():
        pytest.skip("real optimizer cache, alpha, or MOSEK license path is unavailable")
    pytest.importorskip("mosek")

    dates = (20241028, 20241029)
    alpha = pd.read_parquet(alpha_path).reindex(index=dates)
    assert alpha.loc[list(dates)].notna().any(axis=1).all()
    strategy_path = Path(default_strategy_module.__file__).resolve()
    strategy_config = {
        "start_ds": dates[0],
        "end_ds": dates[1],
        "path": str(strategy_path),
        "optimizer": dict(DEFAULT_OPTIMIZER_CONFIG),
    }
    node = BacktestNode(
        start_ds=dates[0],
        end_ds=dates[1],
        output_path=str(tmp_path / "backtest"),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        strategy_config=strategy_config,
        cash=10_000_000.0,
        fee_rate=0.0015,
        reserve_cash=0.95,
        cache_path=str(cache_path),
    )
    backtest = DailyBacktest(node)

    first = backtest.step(dates[0], alpha.loc[dates[0]])
    second = backtest.step(dates[1], alpha.loc[dates[1]])

    assert first["holdings"].gt(0).any()
    assert node.last_hold is not None
    previous, has_previous = backtest.strategy._actual_previous(node.last_hold)
    assert has_previous is True
    assert previous.sum() == pytest.approx(1.0)

    target = second["target_weight"].reindex(backtest.strategy.columns).to_numpy(dtype=float)
    trim_allowance = max(0.0, 1.0 - float(target.sum()))
    assert np.abs(target - previous).sum() <= (
        DEFAULT_OPTIMIZER_CONFIG["maxtvr"] + trim_allowance + 1.0e-6
    )

    date = dates[1]
    signals = alpha.loc[date].reindex(backtest.strategy.columns).fillna(0.0)
    signals *= backtest.universe.loc[date].reindex(backtest.strategy.columns).fillna(0.0)
    normalized_alpha = _normalize_alpha(
        signals.to_numpy(dtype=float),
        float(DEFAULT_OPTIMIZER_CONFIG["trim_threshold"]),
    )
    benchmark = backtest.strategy._benchmark(date)
    base = backtest.strategy._row_at_delay(backtest.strategy.base, date, 0, "base universe")
    limit = backtest.strategy._row_at_delay(backtest.strategy.limit, date, 0, "limit mask")
    suspend = backtest.strategy._row_at_delay(
        backtest.strategy.suspend, date, 0, "suspend mask"
    )
    tradable = (
        np.isfinite(base)
        & (base != 0)
        & np.isfinite(limit)
        & (limit != 0)
        & np.isfinite(suspend)
        & (suspend != 0)
    )
    ret_pos = int(backtest.strategy.returns.index.searchsorted(date))
    history = backtest.strategy.returns.iloc[
        max(0, ret_pos - int(DEFAULT_OPTIMIZER_CONFIG["ret_days"])) : ret_pos
    ]
    return_valid = (
        np.isfinite(history.to_numpy(dtype=np.float32)).sum(axis=0)
        >= int(DEFAULT_OPTIMIZER_CONFIG["min_return_obs"])
    )
    buy_candidate = ((normalized_alpha > 0) | (benchmark > 0)) & tradable & return_valid
    sell_only = (previous > 0) & ~buy_candidate

    assert sell_only.any()
    assert np.all(target[sell_only] <= previous[sell_only] + 1.0e-7)
    assert np.isfinite(target).all()
