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
    reduced, has_reduced = strategy._actual_previous(actual_holdings * 0.5)
    assert has_reduced
    np.testing.assert_allclose(reduced, previous)


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


def test_real_optimizer_solves_with_configured_benchmark_and_keeps_zz500_universe():
    cache_value = os.environ.get("COMB2_TEST_CACHE_PATH")
    alpha_value = os.environ.get("COMB2_TEST_ALPHA_PATH")
    license_value = os.environ.get("MOSEKLM_LICENSE_FILE")
    if not cache_value or not alpha_value or not license_value:
        pytest.skip(
            "set COMB2_TEST_CACHE_PATH, COMB2_TEST_ALPHA_PATH, and "
            "MOSEKLM_LICENSE_FILE for the real optimizer benchmark test"
        )
    cache_path = Path(cache_value)
    alpha_path = Path(alpha_value)
    license_path = Path(license_value)
    if not cache_path.is_dir() or not alpha_path.is_file() or not license_path.is_file():
        pytest.skip("real optimizer cache, alpha, or MOSEK license path is unavailable")
    pytest.importorskip("mosek")

    dates = (20241028, 20241029)
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["benchmark"] = "399300.SZ"
    loader = dataloader_module.DataLoader(cache_path=str(cache_path))
    strategy = AlphaStrategy(
        {
            "start_ds": dates[0],
            "end_ds": dates[-1],
            "optimizer": optimizer,
        },
        loader,
    )

    loaded_start = int(strategy.index_weight.index[0])
    loaded_end = int(strategy.index_weight.index[-1])
    expected_benchmark = strategy._normalize_frame(
        loader.get_index_weight(
            loaded_start,
            loaded_end,
            optimizer["benchmark"],
        )
    ).astype("float32", copy=False)
    pd.testing.assert_frame_equal(strategy.index_weight, expected_benchmark)

    date = dates[-1]
    benchmark = strategy._benchmark(date)
    assert benchmark.sum() == pytest.approx(1.0)
    expected_zz500 = strategy._normalize_frame(
        loader.get_index_weight(loaded_start, loaded_end, "000905.SH")
    ).astype("float32", copy=False)
    zz500_row = strategy._row_at_delay(
        expected_zz500,
        date,
        1,
        "ZZ500 weight",
    )
    expected_zz500_membership = (
        np.isfinite(zz500_row) & (zz500_row > 0)
    ).astype(float)
    np.testing.assert_array_equal(
        strategy._universe_membership("ZZ500", date, 1),
        expected_zz500_membership,
    )
    assert not np.array_equal(benchmark > 0, expected_zz500_membership > 0)

    weight_row = strategy.index_weight.loc[date].to_numpy(dtype=float)
    return_row = strategy.returns.loc[date].to_numpy(dtype=float)
    valid = np.isfinite(weight_row) & (weight_row > 0) & np.isfinite(return_row)
    expected_return = np.sum(weight_row[valid] * return_row[valid]) / np.sum(
        weight_row[valid]
    )
    assert strategy.index_return.loc[date] == pytest.approx(expected_return)

    signals = pd.read_parquet(alpha_path).loc[date].reindex(strategy.index_weight.columns)
    base = strategy._row_at_delay(strategy.base, date, 0, "base universe")
    signals *= np.isfinite(base) & (base != 0)
    zero_amount = pd.Series(0.0, index=signals.index)
    tradable = pd.Series(True, index=signals.index)
    loader.date = date
    orders = strategy.generate_orders(
        signals.fillna(0.0),
        zero_amount,
        zero_amount,
        tradable,
        tradable,
        9_500_000.0,
        0.0,
    )
    assert orders.attrs["solver_fallback"] is False
    target = orders.attrs["target_weight"].to_numpy(dtype=float)
    trim_allowance = max(0.0, 1.0 - float(target.sum()))
    factor_mask = np.isfinite(base) & (base != 0)
    for limit_spec in strategy.hard_risk:
        factor = strategy._factor(limit_spec, date, factor_mask)
        active_exposure = float(np.dot(target - benchmark, factor))
        assert active_exposure >= limit_spec.lo - trim_allowance - 1.0e-6
        assert active_exposure <= limit_spec.hi + trim_allowance + 1.0e-6
    for limit_spec in strategy.hard_group:
        groups = strategy._row_at_delay(
            strategy.groups[limit_spec.name],
            date,
            limit_spec.delay,
            limit_spec.name,
        )
        for group_value in np.unique(groups[np.isfinite(groups)]):
            membership = (groups == group_value).astype(float)
            active_exposure = float(np.dot(target - benchmark, membership))
            assert active_exposure >= limit_spec.lo - trim_allowance - 1.0e-6
            assert active_exposure <= limit_spec.hi + trim_allowance + 1.0e-6


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
    target = second["target_weight"].reindex(backtest.strategy.columns).to_numpy(dtype=float)
    second_orders = second["orders"].reindex(backtest.strategy.columns)
    previous_amount = target * float(node.target_stock_amount)
    previous_amount -= second_orders["buy_amount"].to_numpy(dtype=float)
    previous_amount += second_orders["sell_amount"].to_numpy(dtype=float)
    previous = previous_amount / float(node.target_stock_amount)
    assert previous.sum() > 0.0

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

    planned_turnover = second_orders[["buy_amount", "sell_amount"]].to_numpy(dtype=float).sum() / node.target_stock_amount
    normalization_allowance = abs(1.0 - float(previous.sum()))
    trim_allowance = max(0.0, 1.0 - float(target.sum()))
    assert planned_turnover <= (
        optimizer["maxtvr"]
        + float(second["target_weight"].attrs["forced_exit_weight"])
        + normalization_allowance
        + trim_allowance
        + 1.0e-6
    )

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
    current_amount = (node.holdings * backtest.vwap_data.loc[date]).fillna(0.0)
    locked_amount = (node.locked_holdings * backtest.vwap_data.loc[date]).fillna(0.0)
    sellable_amount = (current_amount - locked_amount).clip(lower=0.0)
    with pytest.warns(RuntimeWarning, match="returning zero orders"):
        buyable_mask, market_sellable_mask = backtest._trading_masks(
            date, backtest.vwap_data.loc[date]
        )
        fallback = backtest.strategy.generate_orders(
            signals,
            sellable_amount,
            locked_amount,
            buyable_mask,
            market_sellable_mask,
            node.target_stock_amount,
            node.executed_turnover_today,
        )
    np.testing.assert_allclose(
        fallback[["buy_amount", "sell_amount"]].to_numpy(dtype=float),
        0.0,
    )
    assert fallback.attrs["solver_fallback"] is True
    assert fallback.attrs["solver_status"] != "Optimal"

    holdings_before_fallback = node.holdings.copy()
    adjustment = (
        backtest.preclose_data.loc[dates[2]] / backtest.close_data.loc[dates[1]]
    ).fillna(1.0)
    expected_holdings = np.floor(holdings_before_fallback / adjustment)
    with pytest.warns(RuntimeWarning, match="returning zero orders"):
        fallback_day = backtest.step(dates[2], alpha.loc[dates[2]])
    pd.testing.assert_series_equal(
        fallback_day["holdings"], expected_holdings, check_names=False
    )
    assert fallback_day["trade_cost"] == 0.0
    assert fallback_day["tvr"] == 0.0
    assert fallback_day["target_weight"].attrs["solver_fallback"] is True


def test_real_opt2_enforces_t1_and_daily_turnover(tmp_path: Path):
    import warnings
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

    date, next_date, signal_date = 20200103, 20200106, 20200102
    alpha = pd.read_parquet(alpha_path).loc[signal_date]
    strategy_path = Path(default_strategy_module.__file__).resolve()
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["type"] = "opt2"
    optimizer["post_trim_renorm"] = True
    node = BacktestNode(
        start_ds=date,
        end_ds=next_date,
        output_path=str(tmp_path / "opt2_backtest"),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        strategy_config={
            "start_ds": date,
            "end_ds": next_date,
            "path": str(strategy_path),
            "optimizer": optimizer,
        },
        cash=10_000_000.0,
        fee_rate=0.00075,
        reserve_cash=0.95,
        cache_path=str(cache_path),
        execution_price="vwap30",
        snap_ti=93000,
    )
    backtest = DailyBacktest(node)
    first = backtest.step(date, alpha)

    assert first["target_weight"].attrs["optimizer_type"] == "opt2"
    assert first["target_weight"].attrs["solver_status"] == "SolutionStatus.Optimal"
    assert node.target_stock_amount == pytest.approx(9_500_000.0)
    pd.testing.assert_series_equal(
        first["locked_holdings"], first["holdings"], check_names=False
    )
    assert (first["holdings"] % 100 == 0).all()
    assert backtest.cash >= 0
    assert node.executed_turnover_today > 0
    assert node.executed_turnover_today <= first["orders"]["buy_amount"].sum()

    columns = backtest.strategy.columns
    assert columns is not None
    signals = -alpha.reindex(columns).fillna(0.0)
    prices = backtest.vwap_data.loc[date]
    buyable_mask, market_sellable_mask = backtest._trading_masks(date, prices)
    locked_amount = (node.locked_holdings * prices).fillna(0.0)
    zero_amount = locked_amount * 0.0
    backtest.dataloader.date = date
    locked_orders = backtest.strategy.generate_orders(
        signals,
        zero_amount,
        locked_amount,
        buyable_mask,
        market_sellable_mask,
        node.target_stock_amount,
        0.0,
    )
    assert locked_orders.attrs["solver_status"] == "SolutionStatus.Optimal"
    assert locked_orders.loc[locked_amount > 0, "sell_amount"].sum() == pytest.approx(
        0.0, abs=1.0e-5
    )

    planned_amount = first["target_weight"].reindex(columns).fillna(0.0)
    planned_amount *= node.target_stock_amount
    with warnings.catch_warnings(record=True) as quota_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        quota_orders = backtest.strategy.generate_orders(
            signals,
            planned_amount,
            zero_amount,
            buyable_mask,
            market_sellable_mask,
            node.target_stock_amount,
            float(optimizer["maxtvr"]) * node.target_stock_amount,
        )
    fallback_warnings = [w for w in quota_warnings
                         if issubclass(w.category, RuntimeWarning) and "returning zero orders" in str(w.message)]
    if quota_orders.attrs["solver_fallback"]:
        assert quota_orders.attrs["solver_status"] != "SolutionStatus.Optimal"
        assert fallback_warnings
    else:
        assert quota_orders.attrs["solver_status"] == "SolutionStatus.Optimal"
        assert not fallback_warnings
    assert quota_orders.to_numpy(dtype=float).sum() == pytest.approx(0.0, abs=1.0e-5)
    np.testing.assert_allclose(
        quota_orders.attrs["target_weight"].reindex(columns).to_numpy(dtype=float),
        planned_amount.to_numpy(dtype=float) / node.target_stock_amount,
        rtol=0, atol=1.0e-7,
    )

    backtest.strategy.hard_univ = _parse_limits(
        "AshareSH:0.00:0.00,0",
        soft=False,
        label="univ_list",
    )
    sellable_amount = planned_amount * 0.5
    forced_locked_amount = planned_amount * 0.5
    forced_orders = backtest.strategy.generate_orders(
        alpha.reindex(columns).fillna(0.0),
        sellable_amount,
        forced_locked_amount,
        buyable_mask,
        market_sellable_mask,
        node.target_stock_amount,
        0.0,
    )
    sh = np.asarray(backtest.strategy.static_universes["AshareSH"], dtype=bool)
    assert forced_orders.attrs["solver_status"] == "SolutionStatus.Optimal"
    assert forced_orders.attrs["forced_exit_count"] > 0
    assert forced_orders["buy_amount"].to_numpy(dtype=float)[sh].sum() == pytest.approx(
        0.0, abs=1.0e-5
    )
    assert forced_orders["sell_amount"].to_numpy(dtype=float)[sh].sum() == pytest.approx(
        sellable_amount.to_numpy(dtype=float)[sh].sum(), abs=1.0e-4
    )

    assert node.locked_holdings.sum() > 0
    backtest._advance_from_previous_close(next_date)
    assert node.locked_holdings.sum() == 0.0


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

    date, signal_date = 20200103, 20200102
    alpha = pd.read_parquet(alpha_path).loc[signal_date]
    strategy_path = Path(default_strategy_module.__file__).resolve()
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["lambda_slp"] = 0.058
    strategy_config = {
        "start_ds": date,
        "end_ds": date,
        "path": str(strategy_path),
        "optimizer": optimizer,
    }
    node = BacktestNode(
        start_ds=date,
        end_ds=date,
        output_path=str(tmp_path / "forced_exit_backtest"),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        strategy_config=strategy_config,
        cash=10_000_000.0,
        fee_rate=0.00075,
        reserve_cash=0.95,
        cache_path=str(cache_path),
        execution_price="vwap30",
        snap_ti=93000,
    )
    backtest = DailyBacktest(node)
    first = backtest.step(date, alpha)
    columns = backtest.strategy.columns
    assert columns is not None
    current_amount = first["target_weight"].reindex(columns).fillna(0.0)
    current_amount *= node.target_stock_amount
    zero_amount = current_amount * 0.0

    backtest.strategy.hard_univ = _parse_limits(
        "AshareCYB:0.00:0.00,0",
        soft=False,
        label="univ_list",
    )
    backtest.dataloader.date = date
    orders = backtest.strategy.generate_orders(
        alpha.reindex(columns).fillna(0.0),
        current_amount,
        zero_amount,
        *backtest._trading_masks(date, backtest.vwap_data.loc[date]),
        node.target_stock_amount,
        0.0,
    )
    membership = backtest.strategy._universe_membership(
        "AshareCYB", date, 0
    ).astype(bool)
    target = orders.attrs["target_weight"].reindex(columns).fillna(0.0)

    assert orders.attrs["solver_status"] == "SolutionStatus.Optimal"
    assert orders.attrs["forced_exit_count"] > 0
    assert orders.attrs["forced_exit_weight"] > 0
    assert orders["buy_amount"].to_numpy(dtype=float)[membership].sum() == pytest.approx(
        0.0, abs=1.0e-5
    )
    assert orders["sell_amount"].to_numpy(dtype=float)[membership].sum() == pytest.approx(
        current_amount.to_numpy(dtype=float)[membership].sum(), abs=1.0e-4
    )
    assert target.to_numpy(dtype=float)[membership].sum() <= 1.0e-8


def test_real_factorsim_delist_transition_writes_off_holding_at_zero(tmp_path: Path):
    cache_value = os.environ.get("COMB2_TEST_CACHE_PATH")
    alpha_value = os.environ.get("COMB2_TEST_ALPHA_PATH")
    license_value = os.environ.get("MOSEKLM_LICENSE_FILE")
    if not cache_value or not alpha_value or not license_value:
        pytest.skip(
            "set COMB2_TEST_CACHE_PATH, COMB2_TEST_ALPHA_PATH, and "
            "MOSEKLM_LICENSE_FILE for the real delist settlement test"
        )
    if not all(
        (
            Path(cache_value).is_dir(),
            Path(alpha_value).is_file(),
            Path(license_value).is_file(),
        )
    ):
        pytest.skip("real cache, alpha, or MOSEK license path is unavailable")
    pytest.importorskip("mosek")

    dates = (20220523, 20220524)
    strategy_path = Path(default_strategy_module.__file__).resolve()
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["maxtvr"] = 2.0
    optimizer["min_participation_ratio"] = 0.0
    for key in (
        "univ_list",
        "soft_univ_list",
        "risk_list",
        "soft_risk_list",
        "group_list",
        "soft_group_list",
    ):
        optimizer[key] = ""
    node = BacktestNode(
        start_ds=dates[0],
        end_ds=dates[-1],
        output_path=str(tmp_path / "delist_writeoff"),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        strategy_config={
            "start_ds": dates[0],
            "end_ds": dates[-1],
            "path": str(strategy_path),
            "optimizer": optimizer,
        },
        cash=10_000_000.0,
        fee_rate=0.0015,
        reserve_cash=0.95,
        cache_path=cache_value,
    )
    backtest = DailyBacktest(node)
    code = "603157"
    shares = 1_000.0
    last_mark = backtest.dataloader.get_close(20220517, 20220517).iloc[0]
    last_mark.index = pd.Index([str(value).zfill(6) for value in last_mark.index])
    assert np.isfinite(last_mark[code]) and last_mark[code] > 0
    assert np.isfinite(backtest.listed_days.loc[dates[0], code])
    assert np.isnan(backtest.listed_days.loc[dates[1], code])
    node.holdings.loc[code] = shares
    node.yesterday = dates[0]
    backtest.last_prices = last_mark.reindex(backtest.universe.columns)
    backtest.last_point = (dates[0], 150000)
    backtest.cooldown_left = 1
    node.prev_total_asset = backtest.cash + shares * float(last_mark[code])
    alpha = pd.read_parquet(alpha_value).loc[dates[0]]

    result = backtest.step(dates[1], alpha)
    writeoff = shares * float(last_mark[code])

    assert result["delist_count"] == 1
    assert result["delist_writeoff"] == pytest.approx(writeoff)
    assert node.holdings.loc[code] == 0.0
    assert node.locked_holdings.loc[code] == 0.0
    assert result["target_weight"].loc[code] == 0.0
    assert result["pnl"] == pytest.approx(-writeoff)
    assert result["trade_cost"] == 0.0
    assert result["tvr"] == 0.0
    settlements = pd.read_csv(backtest.settlement_path, dtype={"code": str})
    assert settlements.loc[0, "code"].zfill(6) == code
    assert settlements.loc[0, "settlement_price"] == 0.0
    assert settlements.loc[0, "writeoff_amount"] == pytest.approx(writeoff)
    assert settlements.loc[0, "policy"] == "factorsim_stock_listed_days_zero_writeoff"


def test_real_factorsim_missing_execution_price_cannot_enter_opt_or_execution(
    tmp_path: Path,
):
    cache_value = os.environ.get("COMB2_TEST_CACHE_PATH")
    alpha_value = os.environ.get("COMB2_TEST_ALPHA_PATH")
    license_value = os.environ.get("MOSEKLM_LICENSE_FILE")
    if not cache_value or not alpha_value or not license_value:
        pytest.skip(
            "set COMB2_TEST_CACHE_PATH, COMB2_TEST_ALPHA_PATH, and "
            "MOSEKLM_LICENSE_FILE for the real tradable-pool test"
        )
    if not all(
        (
            Path(cache_value).is_dir(),
            Path(alpha_value).is_file(),
            Path(license_value).is_file(),
        )
    ):
        pytest.skip("real cache, alpha, or MOSEK license path is unavailable")
    pytest.importorskip("mosek")

    date, signal_date = 20260413, 20260410
    strategy_path = Path(default_strategy_module.__file__).resolve()
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["maxtvr"] = 2.0
    optimizer["min_participation_ratio"] = 0.0
    for key in (
        "univ_list",
        "soft_univ_list",
        "risk_list",
        "soft_risk_list",
        "group_list",
        "soft_group_list",
    ):
        optimizer[key] = ""
    node = BacktestNode(
        start_ds=date,
        end_ds=date,
        output_path=str(tmp_path / "tradable_pool"),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        strategy_config={
            "start_ds": date,
            "end_ds": date,
            "path": str(strategy_path),
            "optimizer": optimizer,
        },
        cash=10_000_000.0,
        fee_rate=0.0015,
        reserve_cash=0.95,
        cache_path=cache_value,
        execution_price="vwap30",
    )
    backtest = DailyBacktest(node)
    alpha = pd.read_parquet(alpha_value).loc[signal_date]
    code = "300391"
    raw_prices = backtest.vwap_data.loc[date]
    buyable, market_sellable = backtest._trading_masks(date, raw_prices)
    assert np.isnan(raw_prices[code])
    assert not buyable[code]
    assert not market_sellable[code]

    result = backtest.step(date, alpha)

    assert result["orders"].loc[code].sum() == 0.0
    assert result["target_weight"].loc[code] == 0.0
    assert result["holdings"].loc[code] == 0.0
    active = result["orders"].sum(axis=1) > 0
    assert buyable[result["orders"]["buy_amount"] > 0].all()
    assert market_sellable[result["orders"]["sell_amount"] > 0].all()
    assert backtest._positive_finite(raw_prices[active]).all()

    current_amount = (
        result["holdings"] * backtest.close_data.loc[date]
    ).fillna(0.0)
    current_amount.loc[code] = 100_000.0
    locked_amount = current_amount * 0.0
    backtest.dataloader.date = date
    frozen_orders = backtest.strategy.generate_orders(
        alpha.reindex(backtest.universe.columns).fillna(0.0),
        current_amount,
        locked_amount,
        buyable,
        market_sellable,
        node.target_stock_amount,
        0.0,
    )
    assert frozen_orders.loc[code].sum() == 0.0
    assert frozen_orders.attrs["frozen_count"] == 1
    assert frozen_orders.attrs["target_weight"].loc[code] == pytest.approx(
        current_amount.loc[code] / node.target_stock_amount
    )


def test_real_array_pipeline_opt2_intraday_and_evaluation(tmp_path):
    """Run researcher code, Memmap, Torch, MOSEK, fills and reports without replacements."""
    import xml.etree.ElementTree as ET
    from comboHelloWorld import CONFIG_TEMPLATE, MODEL_TEMPLATE
    from config import load_config
    from vendor.perf_monitor import PerfMonitor
    from evals.comb_eval.report import run_config_evaluation

    cache = os.environ.get("COMB2_TEST_CACHE_PATH")
    license_path = os.environ.get("MOSEKLM_LICENSE_FILE")
    if not cache or not license_path:
        pytest.skip("real cache and MOSEK license are required")
    research = tmp_path / "Model.py"
    research.write_text(MODEL_TEMPLATE + '''
class ResearchLoader(ResearchLoader):
    def data_requirements(self):
        root = ashare_cache_path(self.config.cache_path)
        return (
            DataItem("factor", path=str(root / "1d_DailyKline" / "DailyKline.pct_chg"), delay=1),
            DataItem("returns", path=str(root / "1d_Returns" / "Returns.c2c_1d")),
            DataItem("execution", path=str(root / "1d_IntraVwap" / "IntraVwap.Vwap30.{ti:06d}")),
        )
    def model_target(self):
        return "returns", "returns"
    def gen_raw_target(self, ds, ti):
        self.set_current_ti(ti)
        endpoint = self.didx2date(self.date2didx(ds) + 1)
        return self.source_field(*self.model_target(), endpoint, endpoint)[0].float()
''')
    root = ET.fromstring(CONFIG_TEMPLATE)
    root.find("constants").set("cache_path", cache)
    root.find("constants").set("output_root", str(tmp_path / "output"))
    root.find("strategy").set("start_ds", "20241028")
    root.find("strategy").set("end_ds", "20241029")
    root.find("./strategy/optimizer").set("type", "opt2")
    root.find("./strategy/optimizer").set("maxtvr", "2.0")
    runtime = root.find("./combo/runtime")
    for key, value in dict(sample_times="100000,110000", trainDelay="2", tsDays="2",
                           max_train_days="5", torch_threads="2").items():
        runtime.set(key, value)
    root.find("./combo/loader").set("data_start_ds", "20241018")
    for key, value in dict(hidden_size="8", fc_size="8", epochs="1", batch_size="2",
                           dropout="0").items():
        root.find("./combo/model").set(key, value)
    config_path = tmp_path / "config.xml"
    ET.ElementTree(root).write(config_path)
    config = load_config(str(config_path))
    monitor = PerfMonitor.from_config(config)
    runner = runCombo.ExperimentRunner(config, monitor, str(config_path))
    runner.setup()
    runner.combo.Train(20241028)
    runner.combo.SaveCheckpointModel(runner.combo.modelDir, runner.combo._train_target_ds(20241028))
    first_day_budget = None
    turnover = 0.
    for ds in runner.dates():
        for ti in runner.target_times():
            before = runner.backtest.node.holdings.copy()
            locked_before = runner.backtest.node.locked_holdings.copy()
            runner.combo.Combine(ds, ti)
            metrics = runner.backtest_step(ds, runner.alpha_convert(ds), ti)
            assert metrics["target_weight"].attrs["solver_status"] == "SolutionStatus.Optimal"
            assert runner.backtest.cash >= 0
            held = runner.backtest.node.holdings
            locked = runner.backtest.node.locked_holdings
            assert (held >= locked).all() and (locked >= 0).all()
            if ti == 100000:
                first_day_budget = runner.backtest.node.target_stock_amount
                turnover = runner.backtest.node.executed_turnover_today
                if ds == 20241028:
                    pd.testing.assert_series_equal(held, locked, check_names=False)
                else:
                    pd.testing.assert_series_equal(
                        locked, (held - before).clip(lower=0), check_names=False
                    )
            else:
                assert runner.backtest.node.target_stock_amount == first_day_budget
                assert runner.backtest.node.executed_turnover_today >= turnover
                assert (held >= locked_before).all()
                assert len(runner.backtest.node.daily_metrics_history) == (1 if ds == 20241028 else 2)
    runner.backtest_finalize()
    runner.alpha_analysis()
    alpha = pd.read_parquet(tmp_path / "output" / "alpha.parquet")
    assert len(alpha) == 4 and alpha.index.names == ["date", "time"]
    daily = pd.read_csv(tmp_path / "output" / "backtest" / "daily_pnl.csv")
    assert len(daily) == 2
    assert daily.pnl.sum() == pytest.approx(daily.total_asset.iloc[-1] - 10_000_000.)
    result = run_config_evaluation(config_path, skip_exposure=True)
    assert len(result.ic_summary) == 2
    assert result.pnl_summary.loc["actual_execution", "days"] == 2
    assert result.artifacts.plot_path.is_file()
    executions = pd.read_csv(tmp_path / "output" / "backtest" / "executions.csv")
    assert set(executions.time) == {100000, 110000}
    assert executions.buy_shares.sum() > 0 and executions.sell_shares.sum() > 0
    import copy
    live_config = copy.deepcopy(config)
    live_config["combo"]["runtime"]["livetrading"] = True
    live = runCombo.ExperimentRunner(live_config, monitor, str(config_path))
    live.setup()
    for ti in live.target_times():
        live.combo.Combine(20241028, ti)
        values = live.alpha_convert(20241028)
        meta = live.live_step(20241028, values, ti)
        assert meta["pred_ds"] == 20241028 and meta["target_ti"] == ti
        np.testing.assert_allclose(values, alpha.loc[(20241028,ti)], equal_nan=True)
    quota_config = copy.deepcopy(config)
    quota_config["backtest"]["output_path"] = str(tmp_path / "quota")
    quota_config["strategy"]["optimizer"]["maxtvr"] = 0.4
    quota = DailyBacktest(runCombo.build_backtest_node(
        Path(quota_config["strategy"]["path"]), quota_config
    ))
    signal = alpha.loc[(20241028,100000)]
    runner.combo.loader.set_current_ti(100000)
    prices = pd.Series(runner.combo.loader.source_field("execution", "execution", 20241028,20241028)[0].numpy(), index=signal.index)
    quota.step(20241028, signal, ti=100000, prices=prices, last=False)
    assert quota.node.executed_turnover_today > 0.4 * quota.node.target_stock_amount
    held_before = quota.node.holdings.copy()
    with pytest.warns(RuntimeWarning):
        exhausted = quota.step(20241028, -signal, ti=110000, prices=prices, last=True)
    assert exhausted["target_weight"].attrs["solver_fallback"]
    assert exhausted["orders"].to_numpy().sum() == 0
    pd.testing.assert_series_equal(quota.node.holdings, held_before)
    monitor.close()
