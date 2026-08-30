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
from comb2_pcmaster.default_strategy import AlphaStrategy, _normalize_alpha, _parse_limits
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


def test_default_optimizer_parses_and_transforms_cap_methods():
    hard = _parse_limits(
        "cap:-0.4:0.3,1,4|cap:-0.2:0.21,1,2",
        soft=False,
        label="risk_list",
    )
    soft = _parse_limits(
        "cap:-0.02:0.07:2,1,4",
        soft=True,
        label="soft_risk_list",
    )
    assert [(item.delay, item.method) for item in hard] == [(1, 4), (1, 2)]
    assert (soft[0].penalty, soft[0].delay, soft[0].method) == (2.0, 1, 4)

    strategy = AlphaStrategy.__new__(AlphaStrategy)
    strategy.market_cap = pd.DataFrame(
        [[1.0, 4.0, 16.0, 64.0]],
        index=[20240102],
        columns=["000001", "000002", "000003", "000004"],
    )
    strategy.styles = {}
    mask = np.array([True, True, True, False])

    method4 = strategy._factor(hard[0]._replace(delay=0), 20240102, mask)
    method2 = strategy._factor(hard[1]._replace(delay=0), 20240102, mask)

    np.testing.assert_allclose(
        method4[:3], np.array([-1.22474487, 0.0, 1.22474487])
    )
    np.testing.assert_allclose(method2, np.array([-0.5, 0.0, 0.5, 0.0]))

    with pytest.raises(ValueError, match="invalid strategy.optimizer.risk_list"):
        _parse_limits("cap:-0.4:0.3,1,3", soft=False, label="risk_list")


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

    dates = (20241028, 20241029, 20241030)
    alpha = pd.read_parquet(alpha_path).reindex(index=dates)
    assert alpha.loc[list(dates)].notna().any(axis=1).all()
    strategy_path = Path(default_strategy_module.__file__).resolve()
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["maxtrd"] = 0.00175
    optimizer["maxpos"] = 0.02
    strategy_config = {
        "start_ds": dates[0],
        "end_ds": dates[-1],
        "path": str(strategy_path),
        "optimizer": optimizer,
    }
    node = BacktestNode(
        start_ds=dates[0],
        end_ds=dates[-1],
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
        optimizer["maxtvr"] + trim_allowance + 1.0e-6
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

    liquidity = backtest.strategy._row_at_delay(
        backtest.strategy.amount,
        date,
        int(optimizer["liquidity_delay"]),
        "daily amount",
    )
    trade_upper = (
        float(optimizer["maxtrd"])
        * liquidity
        / float(optimizer["target_size"])
    )
    finite_trade = np.isfinite(trade_upper) & (trade_upper >= 0)
    assert np.all(
        np.abs(target[finite_trade] - previous[finite_trade])
        <= trade_upper[finite_trade] + trim_allowance + 1.0e-6
    )

    capacity_upper = (
        float(optimizer["maxpos"])
        * liquidity
        / float(optimizer["target_size"])
    )
    allowed_position = np.maximum(
        previous,
        np.minimum(float(optimizer["max_weight"]), capacity_upper),
    )
    finite_position = np.isfinite(allowed_position) & (allowed_position >= 0)
    assert np.all(
        target[finite_position]
        <= allowed_position[finite_position] + trim_allowance + 1.0e-6
    )
    assert target @ target <= (
        1.0
        / (
            float(DEFAULT_OPTIMIZER_CONFIG["min_participation_ratio"])
            * second["target_weight"].attrs["candidate_count"]
        )
        + trim_allowance
        + 1.0e-6
    )
    assert second["target_weight"].attrs["slippage_valid_candidates"] == second[
        "target_weight"
    ].attrs["candidate_count"]
    if float(DEFAULT_OPTIMIZER_CONFIG["lambda_slp"]) > 0:
        assert second["target_weight"].attrs["slippage_excluded_candidates"] > 0
    else:
        assert second["target_weight"].attrs["slippage_excluded_candidates"] == 0

    for limit_spec in backtest.strategy.hard_univ:
        membership = backtest.strategy._universe_membership(
            limit_spec.name, date, limit_spec.delay
        )
        exposure = float(np.dot(target, membership))
        assert exposure >= limit_spec.lo - trim_allowance - 1.0e-6
        assert exposure <= limit_spec.hi + 1.0e-6

    top3000 = backtest.strategy._normalize_frame(
        backtest.dataloader.get_trade_universe("TOP3000", dates[0], dates[1])
    )
    top3000_row = backtest.strategy._row_at_delay(top3000, date, 1, "TOP3000")
    non_top3000 = backtest.strategy._universe_membership("NONETOP3000", date, 1)
    np.testing.assert_array_equal(
        non_top3000.astype(bool),
        ~(np.isfinite(top3000_row) & (top3000_row > 0)),
    )

    assert set(backtest.strategy.groups) == {
        "WindIndustry.sw1",
        "WindIndustry.sw3",
    }
    for limit_spec in backtest.strategy.hard_group:
        group_values = backtest.strategy._row_at_delay(
            backtest.strategy.groups[limit_spec.name],
            date,
            limit_spec.delay,
            limit_spec.name,
        )
        for group_value in np.unique(group_values[np.isfinite(group_values)]):
            membership = (group_values == group_value).astype(float)
            active_exposure = float(np.dot(target - benchmark, membership))
            assert active_exposure >= limit_spec.lo - trim_allowance - 1.0e-6
            assert active_exposure <= limit_spec.hi + trim_allowance + 1.0e-6

    for limit_spec in backtest.strategy.hard_risk:
        if limit_spec.name != "cap":
            continue
        factor = backtest.strategy._factor(
            limit_spec, date, np.isfinite(base) & (base != 0)
        )
        active_exposure = float(np.dot(target - benchmark, factor))
        assert active_exposure >= limit_spec.lo - trim_allowance - 1.0e-6
        assert active_exposure <= limit_spec.hi + trim_allowance + 1.0e-6

    backtest.strategy.hard_univ = _parse_limits(
        "ZZ500:1.01:1.01,1",
        soft=False,
        label="univ_list",
    )
    backtest.dataloader.date = date
    with pytest.warns(RuntimeWarning, match="keeping previous holdings"):
        fallback = backtest.strategy.generate_positions(signals, node.last_hold)
    np.testing.assert_allclose(
        fallback.reindex(backtest.strategy.columns).to_numpy(dtype=float),
        previous,
    )
    assert fallback.attrs["solver_fallback"] is True
    assert fallback.attrs["solver_status"] != "Optimal"

    holdings_before_fallback = node.holdings.copy()
    adjustment = (
        backtest.preclose_data.loc[dates[2]] / backtest.close_data.loc[dates[1]]
    ).fillna(1.0)
    expected_holdings = np.floor(holdings_before_fallback / adjustment)
    with pytest.warns(RuntimeWarning, match="keeping previous holdings"):
        fallback_day = backtest.step(dates[2], alpha.loc[dates[2]])
    pd.testing.assert_series_equal(
        fallback_day["holdings"], expected_holdings, check_names=False
    )
    assert fallback_day["trade_cost"] == 0.0
    assert fallback_day["tvr"] == 0.0
    assert fallback_day["target_weight"].attrs["solver_fallback"] is True


def test_real_default_optimizer_forces_hard_zero_universe_exit(tmp_path: Path):
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

    start_ds, forced_exit_ds = 20230403, 20230428
    alpha = pd.read_parquet(alpha_path)
    strategy_path = Path(default_strategy_module.__file__).resolve()
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["lambda_slp"] = 0.058
    strategy_config = {
        "start_ds": start_ds,
        "end_ds": forced_exit_ds,
        "path": str(strategy_path),
        "optimizer": optimizer,
    }
    node = BacktestNode(
        start_ds=start_ds,
        end_ds=forced_exit_ds,
        output_path=str(tmp_path / "forced_exit_backtest"),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        strategy_config=strategy_config,
        cash=100_000_000.0,
        fee_rate=0.00075,
        reserve_cash=0.95,
        cache_path=str(cache_path),
        execution_price="vwap30",
        snap_ti=93000,
    )
    backtest = DailyBacktest(node)
    execution_dates = [
        int(date)
        for date in backtest.trade_date
        if start_ds <= int(date) <= forced_exit_ds
    ]
    for execution_ds in execution_dates:
        date_pos = backtest.trade_date.index(execution_ds)
        signal_ds = int(backtest.trade_date[date_pos - 1])
        result = backtest.step(execution_ds, alpha.loc[signal_ds])

    target = result["target_weight"].reindex(backtest.strategy.columns).fillna(0.0)
    st_membership = backtest.strategy._universe_membership(
        "AshareST", forced_exit_ds, 1
    ).astype(bool)
    assert result["target_weight"].attrs["forced_exit_count"] > 0
    assert result["target_weight"].attrs["forced_exit_weight"] > 0
    assert result["target_weight"].attrs["forced_exit_unpriced_count"] > 0
    assert result["target_weight"].attrs["forced_exit_unpriced_weight"] > 0
    assert target.to_numpy(dtype=float)[st_membership].sum() <= 1.0e-8
