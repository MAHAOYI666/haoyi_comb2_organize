from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


def load_alpha_strategy_class():
    class StrategyBase:
        def __init__(self, strategy_config: dict, dataloader):
            self.config = strategy_config
            self.dataloader = dataloader

    package = types.ModuleType("comb2_pcmaster")
    strategy_module = types.ModuleType("comb2_pcmaster.strategy")
    strategy_module.StrategyBase = StrategyBase

    original_modules = {
        "comb2_pcmaster": sys.modules.get("comb2_pcmaster"),
        "comb2_pcmaster.strategy": sys.modules.get("comb2_pcmaster.strategy"),
    }
    sys.modules["comb2_pcmaster"] = package
    sys.modules["comb2_pcmaster.strategy"] = strategy_module
    try:
        strategy_path = Path(__file__).resolve().parents[1] / "examples" / "alpha_strategy.py"
        spec = importlib.util.spec_from_file_location("test_alpha_strategy_module", strategy_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.AlphaStrategy
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


AlphaStrategy = load_alpha_strategy_class()


def make_strategy() -> AlphaStrategy:
    return AlphaStrategy(strategy_config={}, dataloader=None)


def generate_orders(signals: pd.Series) -> pd.DataFrame:
    zero = pd.Series(0.0, index=signals.index)
    tradable = pd.Series(True, index=signals.index)
    return make_strategy().generate_orders(
        signals, zero, zero, tradable, tradable, 100.0, 0.0
    )


def test_generate_orders_longs_above_cross_section_median():
    signals = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})

    orders = generate_orders(signals)

    expected = pd.Series({"a": 0.0, "b": 0.0, "c": 25.0, "d": 75.0})
    pd.testing.assert_series_equal(orders["buy_amount"], expected, check_names=False)
    assert orders["sell_amount"].sum() == 0.0


def test_generate_orders_median_ignores_missing_and_infinite_values():
    signals = pd.Series({"a": 1.0, "b": 2.0, "c": np.inf, "d": np.nan, "e": 3.0})

    orders = generate_orders(signals)

    expected = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 100.0})
    pd.testing.assert_series_equal(orders["buy_amount"], expected, check_names=False)


def test_generate_orders_returns_zero_when_no_signal_exceeds_median():
    signals = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})

    orders = generate_orders(signals)

    assert orders.to_numpy(dtype=float).sum() == 0.0
