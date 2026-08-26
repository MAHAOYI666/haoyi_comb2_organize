from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SIMBASE_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "comb2-simbase"
if str(SIMBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBASE_ROOT))

from comb2_simbase import IndexMask
from comb2_simbase.benchmark import load_index_benchmark
from .dataloader import DataLoader
from .strategy import StrategyBase


def _load_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


@dataclass
class BacktestNode:
    start_ds: int
    end_ds: int
    output_path: str
    strategy_path: str
    strategy_class: str
    strategy_config: dict[str, Any]
    cash: float
    fee_rate: float
    reserve_cash: float
    daily_metrics_file: str = "daily_metrics.csv"
    cache_path: str = ""
    verbose: bool = False
    universe: str = "base"
    execution_price: str = "vwap30"
    snap_ti: int | None = None
    drawdown_stop: float = 0.0
    cooldown_days: int = 0
    holdings: pd.Series | None = None
    last_hold: pd.Series | None = None
    yesterday: int | None = None
    weight_index: pd.Index | None = None
    daily_metrics_history: list[dict] = field(default_factory=list)
    asset_history: list[list[float]] = field(default_factory=list)
    position_history: list[pd.DataFrame] = field(default_factory=list)
    hold_history: list[pd.Series] = field(default_factory=list)
    fig: Any | None = None
    ax: Any | None = None
    daily_metrics_written: bool = False
    prev_total_asset: float | None = None


class DailyBacktest:
    def __init__(self, node: BacktestNode):
        self.node = node
        os.makedirs(self.node.output_path, exist_ok=True)
        self.trade_date = sorted(IndexMask().date)
        self.dataloader = DataLoader(signal_path="", cache_path=self.node.cache_path)
        self.strategy = self._init_strategy()
        self._init_universe()
        self._load_market_data()
        self.initialize()

    def _init_strategy(self):
        file_path = self.node.strategy_path
        class_name = self.node.strategy_class
        module_name = f"comb2_pcmaster_strategy_{Path(file_path).stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        strategy_class = getattr(module, class_name, None)
        return strategy_class(
            strategy_config=dict(self.node.strategy_config),
            dataloader=self.dataloader,
        )

    def _init_universe(self):
        if self.node.universe == "base":
            self.universe = self.dataloader.get_base(self.node.start_ds, self.node.end_ds)
        else:
            raise ValueError(f"Unknown universe: {self.node.universe}")

    def _load_market_data(self):
        self.preclose_data = self.dataloader.get_preclose(self.node.start_ds, self.node.end_ds)
        if self.node.execution_price == "vwap30":
            self.vwap_data = self.dataloader.get_vwap(
                self.node.start_ds,
                self.node.end_ds,
                snap_ti=self.node.snap_ti,
            ).ffill()
        elif self.node.execution_price == "open":
            self.vwap_data = self.dataloader.get_open(self.node.start_ds, self.node.end_ds).ffill()
        else:
            raise ValueError(f"Unknown execution_price: {self.node.execution_price}")
        self.close_data = self.dataloader.get_close(self.node.start_ds, self.node.end_ds).ffill()
        self.market_cap = self.dataloader.get_market_cap(self.node.start_ds, self.node.end_ds).ffill()
        self.suspend = self.dataloader.get_suspend(self.node.start_ds, self.node.end_ds)
        self.limit = self.dataloader.get_limit(self.node.start_ds, self.node.end_ds)

    def initialize(self):
        self.node.holdings = pd.Series(0.0, index=self.vwap_data.columns)
        self.node.last_hold = None
        self.node.yesterday = None
        self.node.weight_index = None
        self.node.daily_metrics_history = []
        self.node.asset_history = []
        self.node.position_history = []
        self.node.hold_history = []
        self.node.fig, self.node.ax = None, None
        self.node.daily_metrics_written = False
        self.node.prev_total_asset = None
        self.equity_peak = float(self.node.cash)
        self.cooldown_left = 0
        self.cash = float(self.node.cash)
        self.daily_metrics_path = os.path.join(self.node.output_path, self.node.daily_metrics_file)
        self.pnl_summary_path = os.path.join(self.node.output_path, "pnl_summary.csv")
        for path in (self.daily_metrics_path, self.pnl_summary_path):
            if os.path.exists(path):
                os.remove(path)

    def _align_date(self, date: int) -> int:
        date = int(date)
        if date not in self.vwap_data.index:
            raise ValueError(f"date {date} is outside loaded backtest range")
        return date

    def _coerce_alpha(self, alpha: pd.Series | np.ndarray) -> pd.Series:
        if isinstance(alpha, pd.Series):
            series = alpha.astype(float)
        else:
            series = pd.Series(np.asarray(alpha, dtype=float), index=self.universe.columns)
        series = series.reindex(self.universe.columns)
        series.index.name = None
        return series

    def _advance_from_previous_close(self, date: int):
        if self.node.yesterday is None:
            return
        self.dataloader.date = self.node.yesterday
        close_yesterday = self.close_data.loc[self.node.yesterday]
        pre_close_today = self.preclose_data.loc[date]
        adj = (pre_close_today / close_yesterday).fillna(1.0)
        new_holdings = self.node.holdings / adj
        self.node.holdings = np.floor(new_holdings)
        self.cash += (pre_close_today * (new_holdings - self.node.holdings)).sum()
        total = self._total_asset(pre_close_today)
        self.node.last_hold = (self.node.holdings * pre_close_today / total).fillna(0.0)

    def _total_asset(self, prices_per_share: pd.Series) -> float:
        stock_value = (self.node.holdings * prices_per_share).sum()
        return float(stock_value + self.cash)

    def _append_daily_metrics(self, metrics: dict):
        pd.DataFrame([metrics]).to_csv(
            self.daily_metrics_path,
            mode="a",
            header=not self.node.daily_metrics_written,
            index=False,
        )
        self.node.daily_metrics_written = True

    def _log(self, message: str):
        if self.node.verbose:
            print(message)

    def _fetch_benchmark_data(self) -> pd.DataFrame:
        return load_index_benchmark(
            self.node.cache_path,
            start_ds=self.node.start_ds,
            end_ds=self.node.end_ds,
            ts_code="000905.SH",
        )

    def step(self, date: int, alpha: pd.Series | np.ndarray) -> dict:
        date = self._align_date(date)
        if self.node.yesterday is not None and date <= self.node.yesterday:
            raise ValueError(f"date {date} must be later than previous date {self.node.yesterday}")

        self._advance_from_previous_close(date)
        vwap_today = self.vwap_data.loc[date]
        pre_trade_total = self._total_asset(vwap_today)
        stop_triggered = False
        if self.node.drawdown_stop > 0 and pre_trade_total / self.equity_peak - 1.0 <= -self.node.drawdown_stop:
            stop_triggered = True
            self.cooldown_left = max(self.cooldown_left, int(self.node.cooldown_days))
            self.equity_peak = float(pre_trade_total)

        signals = self._coerce_alpha(alpha).fillna(0.0)
        if stop_triggered or self.cooldown_left > 0:
            target_weight = pd.Series(dtype=float)
            if not stop_triggered:
                self.cooldown_left -= 1
        else:
            signal_masked = signals * self.universe.loc[date].fillna(0.0)
            self.dataloader.date = date
            target_weight = self.strategy.generate_positions(signal_masked, self.node.last_hold)
        self.node.position_history.append(pd.DataFrame([target_weight], index=[date], columns=self.universe.columns))
        tvr_cost = 0.0

        if self.node.weight_index is not None:
            diff = self.node.weight_index.difference(target_weight.index)
            if len(diff) > 0:
                k_value = (self.node.holdings.loc[diff] * vwap_today.loc[diff]).sum()
                tvr_cost += k_value
                self.cash += k_value * (1 - self.node.fee_rate)
                self.node.holdings.loc[diff] = 0
        self.node.weight_index = target_weight.index

        total_asset = self._total_asset(vwap_today)
        target_value = target_weight * total_asset * self.node.reserve_cash
        current_value = self.node.holdings * vwap_today
        diff_value = target_value - current_value
        trade_cost = 0.0

        for stock in self.node.weight_index:
            if (stock not in vwap_today) or pd.isna(vwap_today[stock]) or pd.isna(self.suspend.loc[date, stock]) or pd.isna(self.limit.loc[date, stock]):
                continue

            price_per_share = vwap_today[stock]
            price_per_100_shares = price_per_share * 100
            value_diff = diff_value[stock]

            if value_diff > 0:
                max_lots = int(self.cash // (price_per_100_shares * (1 + self.node.fee_rate)))
                target_lots = int(value_diff // (price_per_100_shares * (1 + self.node.fee_rate)))
                buy_lots = min(target_lots, max_lots)
                if buy_lots > 0:
                    b_value = buy_lots * price_per_100_shares
                    cost = b_value * (1 + self.node.fee_rate)
                    trade_cost += buy_lots * price_per_100_shares * self.node.fee_rate
                    self.cash -= cost
                    self.node.holdings.loc[stock] = self.node.holdings.loc[stock] + buy_lots * 100
                    tvr_cost += b_value
            elif value_diff < 0:
                sell_value = -value_diff
                shares_to_sell = min(self.node.holdings.loc[stock], (sell_value // price_per_100_shares + 1) * 100)
                s_value = shares_to_sell * price_per_share
                proceeds = s_value * (1 - self.node.fee_rate)
                trade_cost += s_value * self.node.fee_rate
                self.cash += proceeds
                self.node.holdings.loc[stock] = self.node.holdings.loc[stock] - shares_to_sell
                tvr_cost += s_value

        close_today = self.close_data.loc[date]
        total = self._total_asset(close_today)
        self.equity_peak = max(self.equity_peak, float(total))
        pnl = 0.0 if self.node.prev_total_asset is None else float(total - self.node.prev_total_asset)
        self.node.prev_total_asset = float(total)
        tvr = float(tvr_cost / target_value.sum()) if target_value.sum() != 0 else 0.0
        long_num = int((self.node.holdings > 0).sum())

        metrics = {
            "date": int(date),
            "total_asset": float(total),
            "pnl": pnl,
            "trade_cost": float(trade_cost),
            "reserve_cash": float(self.cash),
            "tvr": tvr,
            "long_num": long_num,
        }
        self.node.daily_metrics_history.append(metrics)
        self.node.asset_history.append([date, total, trade_cost, self.cash, tvr, long_num])
        self.node.hold_history.append(self.node.holdings.rename(date))
        self._append_daily_metrics(metrics)
        self.node.yesterday = date
        self._log(
            f"date: {date}, total: {total:.2f}, trade_cost: {trade_cost:.2f}, "
            f"cash: {self.cash:.2f}, tvr: {tvr:.3f}, long_num: {long_num}"
        )
        return {
            **metrics,
            "target_weight": target_weight.copy(),
            "holdings": self.node.holdings.copy(),
        }

    def finalize(self):
        if self.node.position_history:
            self.position_data = pd.concat(self.node.position_history)
        else:
            self.position_data = pd.DataFrame(columns=self.universe.columns)
        if self.node.hold_history:
            self.hold_history = pd.concat(self.node.hold_history, axis=1).transpose()
        else:
            self.hold_history = pd.DataFrame(columns=self.universe.columns)
        self.asset_history = pd.DataFrame(
            self.node.asset_history,
            columns=["date", "total_asset", "trade_cost", "reserve_cash", "tvr", "long_num"],
        ).set_index("date")
        summary = self._pnl_summary()
        self.draw()
        prefix = f"{self.node.strategy_class}_{self.node.start_ds}_{self.node.end_ds}"
        self.asset_history.to_csv(os.path.join(self.node.output_path, f"{prefix}_yield.csv"))
        self.position_data.to_csv(os.path.join(self.node.output_path, f"{prefix}_position.csv"))
        self.hold_history.to_csv(os.path.join(self.node.output_path, f"{prefix}_holdings.csv"))
        return summary

    def setup_plot(self, title, xlabel, ylabel):
        plt = _load_pyplot()
        if plt is None:
            raise RuntimeError("matplotlib is required to draw backtest plots")
        self.node.fig, self.node.ax = plt.subplots(figsize=(10, 6))
        self.node.ax.set_title(title, fontsize=14)
        self.node.ax.set_xlabel(xlabel, fontsize=12)
        self.node.ax.set_ylabel(ylabel, fontsize=12)
        self.node.ax.grid(True)
        self.node.fig.tight_layout()

    def save_plot(self, name):
        plt = _load_pyplot()
        self.node.ax.legend()
        self.node.ax.tick_params(axis="x", rotation=45)
        self.node.fig.savefig(os.path.join(self.node.output_path, name))
        if plt is not None:
            plt.close(self.node.fig)

    def draw(self):
        if self.asset_history.empty:
            return
        if _load_pyplot() is None:
            warnings.warn("matplotlib is not installed; skipping backtest plots", RuntimeWarning)
            return
        x = [pd.to_datetime(str(date), format="%Y%m%d") for date in list(self.asset_history.index)]
        bench_data = self._fetch_benchmark_data()
        bench_data = bench_data.sort_values(by="trade_date").set_index("trade_date")
        bench_data = bench_data.reindex(self.vwap_data.index.astype(str)).ffill()

        self.setup_plot("Backtest Result", "date", "cash")
        y0 = self.asset_history.iloc[:, 0] / self.asset_history.iloc[0, 0]
        self.node.ax.plot(x, y0, label="backtest", color="blue")
        y1 = bench_data.loc[:, "close"].astype(float) / float(bench_data.loc[bench_data.index[0], "close"])
        self.node.ax.plot(x, y1, label="ZZ500", color="red")
        self.save_plot("return.jpg")

        self.setup_plot("Backtest Result", "date", "cash")
        y1.index = y0.index
        y = y0 - y1
        self.node.ax.plot(x, y, label="ex-ret for ZZ500", color="red")
        self.save_plot("ex_return.jpg")

        self.setup_plot("TradeCost", "date", "cost")
        y = self.asset_history.iloc[:, 1].cumsum()
        self.node.ax.plot(x, y, label="trade_cost", color="red")
        self.save_plot("trade_cost.jpg")

    def _pnl_summary(self, sdate=20160101, edate=20261231):
        if self.asset_history.empty:
            empty = pd.DataFrame()
            empty.to_csv(self.pnl_summary_path)
            return empty

        bench_data = self._fetch_benchmark_data()
        bench_data = bench_data.sort_values(by="trade_date")
        bench_data.loc[:, "date"] = pd.to_datetime(bench_data["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        bench_data = bench_data.set_index("date")

        df = self.asset_history.copy()
        if df.index.inferred_type == "integer":
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
        elif not np.issubdtype(df.index.dtype, np.datetime64):
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[(df.index >= pd.to_datetime(str(sdate))) & (df.index <= pd.to_datetime(str(edate)))].copy()
        if df.empty:
            empty = pd.DataFrame()
            empty.to_csv(self.pnl_summary_path)
            return empty

        def _max_drawdown(cum_ret: pd.Series) -> float:
            equity = 1 + cum_ret
            roll_max = equity.cummax()
            dd = (roll_max - equity) / roll_max
            return float(dd.max()) if len(dd) else np.nan

        def _sharpe(daily_ret: pd.Series) -> float:
            x = daily_ret.values
            x = x[~np.isnan(x)]
            if len(x) == 0:
                return np.nan
            std = np.std(x, ddof=1)
            if std == 0:
                return np.nan
            return float(np.mean(x) / std * np.sqrt(252))

        booksize = float(self.node.cash) if self.node.cash != 0 else 1.0
        bench_close = bench_data.reindex(df.index)["close"].astype(float).ffill()
        bench_ret = bench_close.pct_change().fillna(0.0)

        pnl = df["total_asset"].diff().fillna(0.0)
        ret = pnl / booksize
        df = df.assign(pnl=pnl, ret=ret, li_ret=ret - bench_ret)

        rows = []
        labels = []
        for year, g in df.groupby(df.index.year):
            start = g.index.min().strftime("%Y%m%d")
            end = g.index.max().strftime("%Y%m%d")
            labels.append(f"{start}-{end}")
            rows.append({
                "pnl": float(g["pnl"].sum()),
                "ret": float(g["ret"].sum()),
                "li_ret": float(g["li_ret"].sum()),
                "dd": _max_drawdown(g["ret"].cumsum()),
                "dd_li": _max_drawdown(g["li_ret"].cumsum()),
                "sharpe": _sharpe(g["ret"]),
                "sharpe_idx": _sharpe(g["li_ret"]),
                "days": int(len(g)),
            })

        summary = pd.DataFrame(rows, index=labels)
        global_label = f"{df.index.min().strftime('%Y%m%d')}-{df.index.max().strftime('%Y%m%d')}"
        if global_label not in summary.index:
            summary.loc[global_label] = {
                "pnl": float(df["pnl"].sum()),
                "ret": float(df["ret"].sum()),
                "li_ret": float(df["li_ret"].sum()),
                "dd": _max_drawdown(df["ret"].cumsum()),
                "dd_li": _max_drawdown(df["li_ret"].cumsum()),
                "sharpe": _sharpe(df["ret"]),
                "sharpe_idx": _sharpe(df["li_ret"]),
                "days": int(len(df)),
            }
        result = summary.round(4)
        result.to_csv(self.pnl_summary_path)
        return result
