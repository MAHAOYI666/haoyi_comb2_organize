from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import runCombo
from comb2_pcmaster import DailyBacktest
from comb2_pcmaster import dataloader as dataloader_module


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
