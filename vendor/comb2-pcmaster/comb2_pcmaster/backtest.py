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
    locked_holdings: pd.Series | None = None
    yesterday: int | None = None
    target_stock_amount: float | None = None
    executed_turnover_today: float = 0.0
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
        if self.node.execution_price == "provided":
            self.vwap_data = self.dataloader.get_close(self.node.start_ds, self.node.end_ds)
        elif self.node.execution_price == "vwap30":
            self.vwap_data = self.dataloader.get_vwap(
                self.node.start_ds,
                self.node.end_ds,
                snap_ti=self.node.snap_ti,
            )
        elif self.node.execution_price == "open":
            self.vwap_data = self.dataloader.get_open(self.node.start_ds, self.node.end_ds)
        else:
            raise ValueError(f"Unknown execution_price: {self.node.execution_price}")
        self.close_data = self.dataloader.get_close(self.node.start_ds, self.node.end_ds).ffill()
        self.market_cap = self.dataloader.get_market_cap(self.node.start_ds, self.node.end_ds).ffill()
        self.suspend = self.dataloader.get_suspend(self.node.start_ds, self.node.end_ds)
        self.limit = self.dataloader.get_limit(self.node.start_ds, self.node.end_ds)
        self.listed_days = self.dataloader.get_stock_mask(
            "StockListedDays", self.node.start_ds, self.node.end_ds
        )

    def initialize(self):
        self.node.holdings = pd.Series(0.0, index=self.vwap_data.columns)
        self.node.locked_holdings = pd.Series(0.0, index=self.vwap_data.columns)
        self.node.yesterday = None
        self.node.target_stock_amount = None
        self.node.executed_turnover_today = 0.0
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
        self.last_point = None
        self.last_prices = None
        self.day_settled = True
        self.day_trade_cost = 0.0
        self.day_delist_writeoff = 0.0
        self.day_delist_count = 0
        self.node.prev_total_asset = self.cash
        self.daily_metrics_path = os.path.join(self.node.output_path, self.node.daily_metrics_file)
        self.execution_path = os.path.join(self.node.output_path, "executions.csv")
        self.executions_written = False
        self.settlement_path = os.path.join(self.node.output_path, "settlements.csv")
        self.settlements_written = False
        self.pnl_summary_path = os.path.join(self.node.output_path, "pnl_summary.csv")
        for path in (
            self.daily_metrics_path,
            self.execution_path,
            self.settlement_path,
            self.pnl_summary_path,
        ):
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
        self.node.locked_holdings = pd.Series(0.0, index=self.node.holdings.index)

    def _total_asset(self, prices_per_share: pd.Series) -> float:
        stock_value = (self.node.holdings * prices_per_share).sum()
        return float(stock_value + self.cash)

    @staticmethod
    def _positive_finite(values: pd.Series) -> pd.Series:
        return values.notna() & np.isfinite(values) & (values > 0)

    def _settle_delisted(self, date: int) -> tuple[float, int]:
        if self.node.yesterday is None:
            return 0.0, 0
        assert self.node.holdings is not None and self.node.locked_holdings is not None
        previous = self.listed_days.loc[self.node.yesterday].reindex(self.universe.columns)
        current = self.listed_days.loc[date].reindex(self.universe.columns)
        was_listed = self._positive_finite(previous)
        is_listed = self._positive_finite(current)
        delisted = (self.node.holdings > 0) & was_listed & ~is_listed
        if not delisted.any():
            return 0.0, 0

        assert self.last_prices is not None
        marks = self.last_prices.reindex(self.universe.columns)
        assert self._positive_finite(marks[delisted]).all(), (
            f"{date}: delisted holdings require a last observable mark"
        )
        shares = self.node.holdings[delisted].copy()
        writeoff = shares * marks[delisted]
        records = pd.DataFrame(
            {
                "date": int(date),
                "code": shares.index,
                "shares": shares.to_numpy(dtype=float),
                "last_mark_price": marks[delisted].to_numpy(dtype=float),
                "settlement_price": 0.0,
                "settlement_value": 0.0,
                "writeoff_amount": writeoff.to_numpy(dtype=float),
                "policy": "factorsim_stock_listed_days_zero_writeoff",
            }
        )
        records.to_csv(
            self.settlement_path,
            mode="a" if self.settlements_written else "w",
            header=not self.settlements_written,
            index=False,
        )
        self.settlements_written = True
        self.node.holdings.loc[delisted] = 0.0
        self.node.locked_holdings.loc[delisted] = 0.0
        return float(writeoff.sum()), int(delisted.sum())

    def _trading_masks(
        self, date: int, execution_prices: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        prices = execution_prices.reindex(self.universe.columns)
        limit = self.limit.loc[date].reindex(self.universe.columns)
        suspend = self.suspend.loc[date].reindex(self.universe.columns)
        base = self.universe.loc[date].reindex(self.universe.columns)
        market_sellable = (
            self._positive_finite(prices)
            & self._positive_finite(limit)
            & self._positive_finite(suspend)
        )
        buyable = market_sellable & self._positive_finite(base)
        return buyable.astype(bool), market_sellable.astype(bool)

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

    def step(self, date, alpha, *, ti=150000, prices=None, mark_prices=None, last=True):
        date = self._align_date(date)
        point = (date, int(ti))
        assert self.last_point is None or point > self.last_point, "execution points must increase"
        new_day = self.last_point is None or date != self.last_point[0]
        if not new_day:
            assert not self.day_settled, "day has already been settled"
            assert self.node.strategy_config["optimizer"]["type"] == "opt2", "intraday execution requires opt2"
        else:
            assert self.day_settled, "settle previous trading day before advancing"
            self.day_delist_writeoff, self.day_delist_count = self._settle_delisted(date)
            self._advance_from_previous_close(date)
            self.node.executed_turnover_today = 0.0
            self.day_trade_cost = 0.0
            self.day_settled = False
        day_delist_writeoff = self.day_delist_writeoff
        day_delist_count = self.day_delist_count
        assert self.node.holdings is not None and self.node.locked_holdings is not None
        vwap_today = self.vwap_data.loc[date] if prices is None else prices.reindex(self.universe.columns)
        if mark_prices is None:
            if new_day:
                mark_prices = self.preclose_data.loc[date].reindex(self.universe.columns)
                if self.last_prices is not None:
                    mark_prices = mark_prices.where(
                        self._positive_finite(mark_prices), self.last_prices
                    )
            else:
                mark_prices = self.last_prices
        mark_prices = mark_prices.reindex(self.universe.columns)
        held = self.node.holdings > 0
        assert self._positive_finite(mark_prices[held]).all(), "held stocks require observable marks"
        pre_trade_total = self._total_asset(mark_prices)
        if new_day:
            self.node.target_stock_amount = float(pre_trade_total * self.node.reserve_cash)
        assert np.isfinite(self.node.target_stock_amount) and self.node.target_stock_amount > 0
        current_value = (self.node.holdings * mark_prices).fillna(0.0)
        locked_value = (self.node.locked_holdings * mark_prices).fillna(0.0)
        sellable_value = (current_value - locked_value).clip(lower=0.0)
        buyable_mask, market_sellable_mask = self._trading_masks(date, vwap_today)
        stop_triggered = False
        if self.node.drawdown_stop > 0 and pre_trade_total / self.equity_peak - 1.0 <= -self.node.drawdown_stop:
            stop_triggered = True
            self.cooldown_left = max(self.cooldown_left, int(self.node.cooldown_days))
            self.equity_peak = float(pre_trade_total)

        signals = self._coerce_alpha(alpha).fillna(0.0)
        if stop_triggered or self.cooldown_left > 0:
            orders = pd.DataFrame(
                {
                    "buy_amount": np.zeros(len(self.universe.columns), dtype=float),
                    "sell_amount": sellable_value.where(
                        market_sellable_mask, 0.0
                    ).to_numpy(dtype=float),
                },
                index=self.universe.columns,
            )
            orders.attrs["optimizer_type"] = self.node.strategy_config.get(
                "optimizer", {}
            ).get("type", "custom")
            orders.attrs["solver_status"] = "drawdown_stop"
            orders.attrs["solver_fallback"] = False
            if new_day and not stop_triggered:
                self.cooldown_left -= 1
        else:
            signal_masked = signals * self.universe.loc[date].fillna(0.0)
            self.dataloader.date = date
            orders = self.strategy.generate_orders(
                signal_masked,
                sellable_value,
                locked_value,
                buyable_mask,
                market_sellable_mask,
                self.node.target_stock_amount,
                self.node.executed_turnover_today,
            )
        assert orders.index.equals(self.universe.columns)
        assert list(orders.columns) == ["buy_amount", "sell_amount"]
        order_values = orders.to_numpy(dtype=float)
        assert np.isfinite(order_values).all() and (order_values >= 0).all()
        assert not (
            (orders["buy_amount"].to_numpy(dtype=float) > 0)
            & (orders["sell_amount"].to_numpy(dtype=float) > 0)
        ).any()
        invalid_buy = (orders["buy_amount"] > 0) & ~buyable_mask
        invalid_sell = (orders["sell_amount"] > 0) & ~market_sellable_mask
        assert not invalid_buy.any(), (
            f"{point}: buy orders outside buyable pool: "
            f"{orders.index[invalid_buy].tolist()[:10]}"
        )
        assert not invalid_sell.any(), (
            f"{point}: sell orders outside market-sellable pool: "
            f"{orders.index[invalid_sell].tolist()[:10]}"
        )

        target_weight = orders.attrs.get("target_weight")
        if target_weight is None:
            target_amount = current_value + orders["buy_amount"] - orders["sell_amount"]
            target_weight = target_amount.clip(lower=0.0) / self.node.target_stock_amount
        else:
            target_weight = target_weight.copy()
        target_weight.name = point
        target_weight.attrs.update(
            {key: value for key, value in orders.attrs.items() if key != "target_weight"}
        )
        self.node.position_history.append(
            pd.DataFrame([target_weight], index=pd.MultiIndex.from_tuples([point], names=["date", "time"]), columns=self.universe.columns)
        )
        execution_index = orders.index[
            (orders["buy_amount"] > 0) | (orders["sell_amount"] > 0)
        ]
        tvr_cost = 0.0
        trade_cost = 0.0
        buy_shares = pd.Series(0.0, index=orders.index)
        sell_shares = pd.Series(0.0, index=orders.index)

        for stock in execution_index:
            price_per_share = vwap_today[stock]
            price_per_100_shares = price_per_share * 100
            buy_amount = orders.at[stock, "buy_amount"]
            sell_amount = orders.at[stock, "sell_amount"]

            if buy_amount > 0:
                max_lots = int(self.cash // (price_per_100_shares * (1 + self.node.fee_rate)))
                target_lots = int(buy_amount // (price_per_100_shares * (1 + self.node.fee_rate)))
                buy_lots = min(target_lots, max_lots)
                if buy_lots > 0:
                    b_value = buy_lots * price_per_100_shares
                    cost = b_value * (1 + self.node.fee_rate)
                    trade_cost += buy_lots * price_per_100_shares * self.node.fee_rate
                    self.cash -= cost
                    self.node.holdings.loc[stock] = self.node.holdings.loc[stock] + buy_lots * 100
                    self.node.locked_holdings.loc[stock] = (
                        self.node.locked_holdings.loc[stock] + buy_lots * 100
                    )
                    tvr_cost += b_value
                    buy_shares.loc[stock] = buy_lots * 100
            elif sell_amount > 0:
                sellable_shares = max(
                    self.node.holdings.loc[stock]
                    - self.node.locked_holdings.loc[stock],
                    0.0,
                )
                shares_to_sell = min(
                    sellable_shares,
                    (sell_amount // price_per_100_shares + 1) * 100,
                )
                s_value = shares_to_sell * price_per_share
                proceeds = s_value * (1 - self.node.fee_rate)
                trade_cost += s_value * self.node.fee_rate
                self.cash += proceeds
                self.node.holdings.loc[stock] = self.node.holdings.loc[stock] - shares_to_sell
                tvr_cost += s_value
                sell_shares.loc[stock] = shares_to_sell

        self.node.executed_turnover_today += float(tvr_cost)
        executed = orders.loc[execution_index].copy()
        executed["buy_shares"] = buy_shares.loc[execution_index]
        executed["sell_shares"] = sell_shares.loc[execution_index]
        executed["execution_price"] = vwap_today.loc[execution_index]
        executed["date"] = date
        executed["time"] = int(ti)
        executed.index.name = "code"
        executed.to_csv(self.execution_path, mode="a" if self.executions_written else "w",
                        header=not self.executions_written)
        self.executions_written = True

        self.day_trade_cost += float(trade_cost)
        self.last_prices = vwap_today.where(np.isfinite(vwap_today) & (vwap_today > 0), mark_prices)
        self.last_point = point
        total = self._total_asset(self.last_prices)
        metrics = {
            "date": date, "time": int(ti), "total_asset": total,
            "pnl": total - self.node.prev_total_asset, "trade_cost": float(trade_cost),
            "reserve_cash": float(self.cash),
            "tvr": float(tvr_cost / self.node.target_stock_amount),
            "long_num": int((self.node.holdings > 0).sum()),
            "delist_writeoff": day_delist_writeoff,
            "delist_count": day_delist_count,
        }
        if last:
            close_marks = self.close_data.loc[date].reindex(self.universe.columns)
            held = self.node.holdings > 0
            assert self._positive_finite(close_marks[held]).all(), (
                "held stocks require observable end-of-day marks"
            )
            total = self._total_asset(close_marks)
            self.last_prices = close_marks
            self.equity_peak = max(self.equity_peak, total)
            daily = {
                "date": date, "total_asset": total,
                "pnl": total - self.node.prev_total_asset,
                "trade_cost": self.day_trade_cost, "reserve_cash": float(self.cash),
                "tvr": self.node.executed_turnover_today / self.node.target_stock_amount,
                "long_num": metrics["long_num"],
                "delist_writeoff": day_delist_writeoff,
                "delist_count": day_delist_count,
            }
            self.node.prev_total_asset = total
            self.node.daily_metrics_history.append(daily)
            self.node.asset_history.append([
                date, total, self.day_trade_cost, self.cash, daily["tvr"],
                daily["long_num"], day_delist_writeoff, day_delist_count,
            ])
            self.node.hold_history.append(self.node.holdings.rename(date))
            self._append_daily_metrics(daily)
            self.node.yesterday = date
            self.day_settled = True
            metrics.update(daily)
        return {
            **metrics, "target_weight": target_weight.copy(), "orders": orders.copy(),
            "holdings": self.node.holdings.copy(), "locked_holdings": self.node.locked_holdings.copy(),
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
            columns=[
                "date", "total_asset", "trade_cost", "reserve_cash", "tvr",
                "long_num", "delist_writeoff", "delist_count",
            ],
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
        y0 = self.asset_history.iloc[:, 0] / self.node.cash
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

        pnl = df["total_asset"].diff()
        pnl.iloc[0] = df["total_asset"].iloc[0] - self.node.cash
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
