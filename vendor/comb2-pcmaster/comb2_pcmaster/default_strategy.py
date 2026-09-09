from __future__ import annotations

import math
import re
import time
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from comb2_pcmaster.strategy import StrategyBase


class RangeLimit(NamedTuple):
    name: str
    lo: float
    hi: float
    penalty: float
    delay: int
    method: int


def _parse_limits(raw: str, *, soft: bool, label: str) -> tuple[RangeLimit, ...]:
    limits: list[RangeLimit] = []
    is_risk = label in {"risk_list", "soft_risk_list"}
    for entry in (item.strip() for item in raw.split("|")):
        if not entry:
            continue
        pieces = [piece.strip() for piece in entry.split(",")]
        if len(pieces) > (3 if is_risk else 2):
            raise ValueError(f"strategy.optimizer.{label} has too many comma fields: {entry!r}")
        fields = pieces[0].split(":")
        expected = 4 if soft else 3
        if len(fields) != expected:
            shape = "name:lo:hi:penalty" if soft else "name:lo:hi"
            suffix = "[,delay[,method]]" if is_risk else "[,delay]"
            raise ValueError(
                f"strategy.optimizer.{label} entry must be {shape}{suffix}: {entry!r}"
            )
        name = fields[0]
        lo = float(fields[1])
        hi = float(fields[2])
        penalty = float(fields[3]) if soft else 1.0
        delay = int(pieces[1]) if len(pieces) >= 2 else 0
        method = int(pieces[2]) if len(pieces) == 3 else 2 if is_risk else 0
        if (
            not name
            or lo > hi
            or penalty < 0
            or delay < 0
            or (is_risk and method not in {2, 4})
        ):
            raise ValueError(f"invalid strategy.optimizer.{label} entry: {entry!r}")
        limits.append(RangeLimit(name, lo, hi, penalty, delay, method))
    return tuple(limits)


def _normalize_alpha(values: np.ndarray, trim_threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    alpha = np.zeros_like(values)
    finite = np.isfinite(values)
    positive = finite & (values > 0)
    negative = finite & (values < 0)
    if trim_threshold > 0 and positive.any():
        keep = positive.copy()
        for _ in range(50):
            threshold = trim_threshold * float(values[keep].sum())
            updated = positive & (values > threshold)
            if np.array_equal(updated, keep):
                break
            keep = updated
        positive = keep
    positive_sum = float(values[positive].sum())
    negative_sum = float(-values[negative].sum())
    if positive_sum > 0:
        alpha[positive] = values[positive] / positive_sum
    if negative_sum > 0:
        alpha[negative] = values[negative] / negative_sum
    return alpha


def _centered_rank(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    indexes = np.flatnonzero(mask & np.isfinite(values))
    if len(indexes) <= 1:
        return result
    ranks = pd.Series(values[indexes]).rank(method="average").to_numpy(dtype=np.float64) - 1.0
    ranks /= len(indexes) - 1
    ranks -= float(ranks.mean())
    result[indexes] = ranks
    return result


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


class AlphaStrategy(StrategyBase):
    def __init__(self, strategy_config: dict, dataloader):
        super().__init__(strategy_config, dataloader)
        assert "optimizer" in strategy_config
        self.optimizer = dict(strategy_config["optimizer"])
        assert self.optimizer["type"] in {"opt1", "opt2"}
        self.hard_univ = _parse_limits(
            self.optimizer["univ_list"], soft=False, label="univ_list"
        )
        self.soft_univ = _parse_limits(
            self.optimizer["soft_univ_list"], soft=True, label="soft_univ_list"
        )
        self.hard_risk = _parse_limits(
            self.optimizer["risk_list"], soft=False, label="risk_list"
        )
        self.soft_risk = _parse_limits(
            self.optimizer["soft_risk_list"], soft=True, label="soft_risk_list"
        )
        self.hard_group = _parse_limits(
            self.optimizer["group_list"], soft=False, label="group_list"
        )
        self.soft_group = _parse_limits(
            self.optimizer["soft_group_list"], soft=False, label="soft_group_list"
        )
        self.columns: pd.Index | None = None
        self._load_inputs(int(strategy_config["start_ds"]), int(strategy_config["end_ds"]))

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result.index = pd.Index(result.index.to_numpy(dtype=np.int64), name=result.index.name)
        result.columns = pd.Index([str(value).zfill(6) for value in result.columns])
        return result.sort_index()

    def _load_inputs(self, start_ds: int, end_ds: int) -> None:
        trade_dates = np.asarray(self.dataloader.trade_date, dtype=np.int64)
        start_pos = int(np.searchsorted(trade_dates, start_ds))
        lookback = max(130, int(self.optimizer["ret_days"]) + int(self.optimizer["ret_delay"]) + 5)
        history_start = int(trade_dates[max(0, start_pos - lookback)])

        self.returns = self._normalize_frame(
            self.dataloader.get_returns(history_start, end_ds)
        ).astype("float32", copy=False)
        self.close = self._normalize_frame(
            self.dataloader.get_close(history_start, end_ds)
        ).astype("float32", copy=False)
        self.amount = None
        if float(self.optimizer["maxtrd"]) > 0 or float(self.optimizer["maxpos"]) > 0:
            self.amount = self._normalize_frame(
                self.dataloader.get_amount(history_start, end_ds)
            ).astype("float32", copy=False)
        self.slippage = None
        if float(self.optimizer["lambda_slp"]) > 0:
            self.slippage = self._normalize_frame(
                self.dataloader.get_slippage(history_start, end_ds)
            ).astype("float32", copy=False)
        benchmark = self.optimizer["benchmark"]
        self.index_weight = self._normalize_frame(
            self.dataloader.get_index_weight(history_start, end_ds, benchmark)
        ).astype("float32", copy=False)
        self.base = self._normalize_frame(
            self.dataloader.get_base(start_ds, end_ds)
        ).astype("float32", copy=False)
        self.limit = self._normalize_frame(
            self.dataloader.get_limit(start_ds, end_ds)
        ).astype("float32", copy=False)
        self.suspend = self._normalize_frame(
            self.dataloader.get_suspend(start_ds, end_ds)
        ).astype("float32", copy=False)

        universe_names = {limit.name for limit in (*self.hard_univ, *self.soft_univ)}
        supported_universes = {
            "ZZ500",
            "ZZ1800",
            "HS300",
            "AshareST",
            "AshareSH",
            "AshareSZ",
            "AshareCYB",
            "NONETOP3000",
        }
        unknown_universes = sorted(universe_names - supported_universes)
        if unknown_universes:
            raise ValueError(
                "unsupported optimizer universes: " + ", ".join(unknown_universes)
            )

        self.universes = {}
        if "ZZ500" in universe_names:
            self.universes["ZZ500"] = (
                self.index_weight
                if benchmark == "000905.SH"
                else self._normalize_frame(
                    self.dataloader.get_index_weight(
                        history_start, end_ds, "000905.SH"
                    )
                ).astype("float32", copy=False)
            )
        if "HS300" in universe_names:
            self.universes["HS300"] = self._normalize_frame(
                self.dataloader.get_index_weight(history_start, end_ds, "399300.SZ")
            ).astype("float32", copy=False)
        if "ZZ1800" in universe_names:
            zz800 = self._normalize_frame(
                self.dataloader.get_index_weight(history_start, end_ds, "000906.SH")
            )
            zz1000 = self._normalize_frame(
                self.dataloader.get_index_weight(history_start, end_ds, "000852.SH")
            )
            if not zz800.index.equals(zz1000.index) or not zz800.columns.equals(zz1000.columns):
                raise ValueError("ZZ800 and ZZ1000 axes do not match")
            self.universes["ZZ1800"] = (
                (np.isfinite(zz800) & (zz800 > 0))
                | (np.isfinite(zz1000) & (zz1000 > 0))
            )
        if "AshareST" in universe_names:
            non_st = self._normalize_frame(
                self.dataloader.get_stock_mask("STStock", history_start, end_ds)
            )
            self.universes["AshareST"] = ~(np.isfinite(non_st) & (non_st > 0))
        if "NONETOP3000" in universe_names:
            top3000 = self._normalize_frame(
                self.dataloader.get_trade_universe("TOP3000", history_start, end_ds)
            )
            self.universes["NONETOP3000"] = ~(
                np.isfinite(top3000) & (top3000 > 0)
            )

        codes = self.index_weight.columns.to_numpy(dtype=str)
        static_universes = {
            "AshareSH": np.char.startswith(codes, "6"),
            "AshareSZ": np.char.startswith(codes, "0") | np.char.startswith(codes, "3"),
            "AshareCYB": np.char.startswith(codes, "300")
            | np.char.startswith(codes, "301"),
        }
        self.static_universes = {
            name: values for name, values in static_universes.items() if name in universe_names
        }

        group_names = {limit.name for limit in (*self.hard_group, *self.soft_group)}
        group_fields = {
            "WindIndustry.sw1": "SW2021_L1",
            "WindIndustry.sw3": "SW2021_L3",
        }
        unknown_groups = sorted(group_names - group_fields.keys())
        if unknown_groups:
            raise ValueError(
                "unsupported optimizer groups: " + ", ".join(unknown_groups)
            )
        self.groups = {
            name: self._normalize_frame(
                self.dataloader.get_industry(group_fields[name], history_start, end_ds)
            ).astype("float32", copy=False)
            for name in sorted(group_names)
        }

        factor_names = {limit.name for limit in (*self.hard_risk, *self.soft_risk)}
        barra_names = sorted(
            name.split(".", 1)[1]
            for name in factor_names
            if name.startswith("BarraCNE5.")
        )
        supported = {"cap", "returns120", "vola_30", "vola_5", "close"}
        unknown = sorted(
            name for name in factor_names if not name.startswith("BarraCNE5.") and name not in supported
        )
        if unknown:
            raise ValueError("unsupported optimizer risk factors: " + ", ".join(unknown))
        self.styles = {
            f"BarraCNE5.{name}": self._normalize_frame(
                self.dataloader.get_barra_style(name, history_start, end_ds)
            ).astype("float32", copy=False)
            for name in barra_names
        }
        self.market_cap = None
        if "cap" in factor_names:
            self.market_cap = self._normalize_frame(
                self.dataloader.get_market_cap(history_start, end_ds)
            ).astype("float32", copy=False)
        self.index_return = self._build_index_return()

    def _build_index_return(self) -> pd.Series:
        weights = self.index_weight.reindex(index=self.returns.index)
        weight_values = weights.to_numpy(dtype=np.float64)
        return_values = self.returns.to_numpy(dtype=np.float64)
        valid = np.isfinite(weight_values) & (weight_values > 0) & np.isfinite(return_values)
        numerator = np.where(valid, weight_values * return_values, 0.0).sum(axis=1)
        denominator = np.where(valid, weight_values, 0.0).sum(axis=1)
        values = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
        return pd.Series(
            values,
            index=self.returns.index,
            name=f"{self.optimizer['benchmark']}_return",
        )

    def _bind_columns(self, columns: pd.Index) -> None:
        normalized = pd.Index([str(value).zfill(6) for value in columns])
        frames = {
            "returns": self.returns,
            "close": self.close,
            "index weight": self.index_weight,
            "base universe": self.base,
            "limit mask": self.limit,
            "suspend mask": self.suspend,
            **{f"universe {name}": frame for name, frame in self.universes.items()},
            **{f"group {name}": frame for name, frame in self.groups.items()},
            **self.styles,
        }
        if self.market_cap is not None:
            frames["market cap"] = self.market_cap
        if self.amount is not None:
            frames["amount"] = self.amount
        if self.slippage is not None:
            frames["slippage"] = self.slippage
        for label, frame in frames.items():
            if not frame.columns.equals(normalized):
                raise ValueError(f"{label} columns do not match the backtest instrument axis")
        self.columns = normalized

    def _universe_membership(self, name: str, date: int, delay: int) -> np.ndarray:
        if name in self.static_universes:
            values = self.static_universes[name]
        else:
            values = self._row_at_delay(self.universes[name], date, delay, name)
        return (np.isfinite(values) & (values > 0)).astype(np.float64)

    @staticmethod
    def _row_at_delay(frame: pd.DataFrame, date: int, delay: int, label: str) -> np.ndarray:
        pos = int(frame.index.searchsorted(date))
        if pos >= len(frame.index) or int(frame.index[pos]) != date:
            raise ValueError(f"{label}: date {date} not found")
        row_pos = pos - delay
        if row_pos < 0:
            raise ValueError(f"{label}: date {date} lacks delay={delay} history")
        return frame.iloc[row_pos].to_numpy(dtype=np.float64)

    def _actual_previous(
        self, current_amount: pd.Series | None
    ) -> tuple[np.ndarray, bool]:
        assert self.columns is not None
        if current_amount is None:
            return np.zeros(len(self.columns), dtype=np.float64), False
        values = pd.to_numeric(
            current_amount.reindex(self.columns), errors="coerce"
        ).to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
        total = float(values.sum())
        if total <= 0:
            return np.zeros(len(self.columns), dtype=np.float64), False
        return values / total, True

    def _benchmark(self, date: int) -> np.ndarray:
        benchmark = self.optimizer["benchmark"]
        values = self._row_at_delay(
            self.index_weight,
            date,
            int(self.optimizer["benchmark_delay"]),
            f"benchmark {benchmark} weight",
        )
        valid = np.isfinite(values) & (values > 0)
        total = float(values[valid].sum())
        if total <= 0:
            raise ValueError(
                f"benchmark {benchmark} weight {date} has no positive values"
            )
        result = np.zeros_like(values)
        result[valid] = values[valid] / total
        return result

    def _derived_factor(self, name: str, date: int, delay: int) -> np.ndarray:
        date_pos = int(self.returns.index.searchsorted(date))
        if date_pos >= len(self.returns.index) or int(self.returns.index[date_pos]) != date:
            raise ValueError(f"returns: date {date} not found")
        row_pos = date_pos - delay
        if row_pos < 0:
            raise ValueError(f"{name}: date {date} lacks delay={delay} history")
        if name == "close":
            return self.close.iloc[row_pos].to_numpy(dtype=np.float64)

        window_size = 120 if name == "returns120" else 30 if name == "vola_30" else 5
        minimum = 60 if name == "returns120" else 15 if name == "vola_30" else 3
        start = max(0, row_pos + 1 - window_size)
        values = self.returns.iloc[start : row_pos + 1].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        counts = finite.sum(axis=0)
        if name == "returns120":
            logs = np.where(finite, np.log1p(np.clip(values, -0.999999, None)), 0.0)
            result = np.expm1(logs.sum(axis=0))
        else:
            sums = np.where(finite, values, 0.0).sum(axis=0)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
            centered = np.where(finite, values - means, 0.0)
            variance = np.divide(
                (centered * centered).sum(axis=0),
                counts - 1,
                out=np.full_like(sums, np.nan),
                where=counts > 1,
            )
            result = np.sqrt(np.maximum(variance, 0.0))
        result[counts < minimum] = np.nan
        return result

    def _factor(self, limit: RangeLimit, date: int, factor_mask: np.ndarray) -> np.ndarray:
        if limit.name == "cap":
            assert self.market_cap is not None
            values = self._row_at_delay(
                self.market_cap, date, limit.delay, "market cap"
            )
        elif limit.name in self.styles:
            values = self._row_at_delay(self.styles[limit.name], date, limit.delay, limit.name)
            valid = factor_mask & np.isfinite(values)
            if valid.any():
                values = values.copy()
                values[valid] -= float(values[valid].mean())
        else:
            values = self._derived_factor(limit.name, date, limit.delay)
        if limit.method == 2:
            return _centered_rank(values, factor_mask)

        valid = factor_mask & np.isfinite(values) & (values > 0)
        result = np.zeros_like(values, dtype=np.float64)
        logs = np.log(values[valid])
        if logs.size:
            scale = float(logs.std(ddof=0))
            if np.isfinite(scale) and scale > 0:
                result[valid] = (logs - float(logs.mean())) / scale
        return result

    def _variance_matrix(self, date: int, candidate_idx: np.ndarray):
        from mosek.fusion import Matrix

        ret_days = int(self.optimizer["ret_days"])
        ret_delay = int(self.optimizer["ret_delay"])
        shrinkage = float(self.optimizer["shrinkage"])
        ret_method = int(self.optimizer["ret_method"])
        pos = int(self.returns.index.searchsorted(date))
        if pos >= len(self.returns.index) or int(self.returns.index[pos]) != date:
            raise ValueError(f"returns: date {date} not found")
        end = pos + 1 - ret_delay
        start = end - ret_days
        if start < 0 or end <= start:
            raise ValueError(f"returns: date {date} lacks {ret_days} days with delay={ret_delay}")

        history_dates = self.returns.index[start:end]
        history = self.returns.iloc[start:end, candidate_idx].to_numpy(
            dtype=np.float64, copy=True
        )
        history[~np.isfinite(history)] = 0.0
        if ret_method == 2:
            index_history = self.index_return.reindex(history_dates).to_numpy(
                dtype=np.float64, copy=True
            )
            index_history[~np.isfinite(index_history)] = 0.0
            history -= index_history[:, None]

        mean = history.sum(axis=0) / ret_days
        centered = history - mean
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        row_offset = 0
        if shrinkage < 1:
            factor = centered[::-1] * math.sqrt((1.0 - shrinkage) / ret_days)
            flat = factor.ravel()
            nonzero = np.flatnonzero(flat)
            row_parts.append(nonzero // len(candidate_idx))
            col_parts.append(nonzero % len(candidate_idx))
            value_parts.append(flat[nonzero])
            row_offset = ret_days
        if shrinkage > 0:
            second = (history * history).sum(axis=0) / ret_days
            diagonal = np.sqrt(shrinkage * np.maximum(second - mean * mean, 0.0))
            indexes = np.flatnonzero(diagonal)
            row_parts.append(row_offset + indexes)
            col_parts.append(indexes)
            value_parts.append(diagonal[indexes])
            row_offset += len(candidate_idx)
        if not row_parts:
            raise ValueError("variance matrix is empty")
        rows = np.concatenate(row_parts).astype(np.int32)
        cols = np.concatenate(col_parts).astype(np.int32)
        data = np.concatenate(value_parts).astype(np.float64)
        return Matrix.sparse(
            row_offset,
            len(candidate_idx),
            rows.tolist(),
            cols.tolist(),
            data.tolist(),
        )

    @staticmethod
    def _add_soft_range(model, expression, limit: RangeLimit, scale: float, objective, label: str):
        from mosek.fusion import Domain, Expr

        lower = model.variable(f"{label}_lo", 1, Domain.greaterThan(0.0))
        upper = model.variable(f"{label}_hi", 1, Domain.greaterThan(0.0))
        model.constraint(f"{label}_lower", Expr.add(expression, lower), Domain.greaterThan(limit.lo))
        model.constraint(f"{label}_upper", Expr.sub(expression, upper), Domain.lessThan(limit.hi))
        penalty = scale * limit.penalty
        if penalty > 0:
            objective = Expr.sub(objective, Expr.mul(penalty, Expr.add(lower, upper)))
        return objective

    def generate_orders(
        self,
        signals,
        sellable_amount,
        locked_amount,
        buyable_mask,
        market_sellable_mask,
        target_stock_amount,
        executed_turnover_today,
    ):
        from mosek.fusion import Domain, Expr, Model, ObjectiveSense, SolutionStatus

        if self.columns is None:
            self._bind_columns(signals.index)
        assert self.columns is not None
        optimizer_type = self.optimizer["type"]
        target_stock_amount = float(target_stock_amount)
        executed_turnover_today = float(executed_turnover_today)
        assert np.isfinite(target_stock_amount) and target_stock_amount > 0
        assert np.isfinite(executed_turnover_today) and executed_turnover_today >= 0
        sellable_values = pd.to_numeric(
            sellable_amount.reindex(self.columns), errors="coerce"
        ).to_numpy(dtype=np.float64)
        locked_values = pd.to_numeric(
            locked_amount.reindex(self.columns), errors="coerce"
        ).to_numpy(dtype=np.float64)
        buyable_series = buyable_mask.reindex(self.columns)
        market_sellable_series = market_sellable_mask.reindex(self.columns)
        assert np.isfinite(sellable_values).all() and (sellable_values >= 0).all()
        assert np.isfinite(locked_values).all() and (locked_values >= 0).all()
        assert buyable_series.notna().all() and market_sellable_series.notna().all()
        buyable_values = buyable_series.to_numpy(dtype=bool)
        market_sellable_values = market_sellable_series.to_numpy(dtype=bool)
        assert not (buyable_values & ~market_sellable_values).any()
        current_values = sellable_values + locked_values
        if optimizer_type == "opt1":
            previous, has_previous = self._actual_previous(
                pd.Series(current_values, index=self.columns)
            )
        else:
            previous = current_values / target_stock_amount
            has_previous = float(current_values.sum()) > 0
        locked_weight = locked_values / target_stock_amount
        date = int(self.dataloader.date)
        signal_values = pd.to_numeric(signals.reindex(self.columns), errors="coerce").to_numpy(dtype=np.float64)
        alpha = _normalize_alpha(signal_values, float(self.optimizer["trim_threshold"]))
        benchmark = self._benchmark(date)
        hard_univ_rows = [
            (
                limit_spec,
                self._universe_membership(
                    limit_spec.name, date, limit_spec.delay
                ),
            )
            for limit_spec in self.hard_univ
        ]
        forced_exit = np.zeros(len(self.columns), dtype=bool)
        for limit_spec, membership in hard_univ_rows:
            if limit_spec.hi == 0:
                forced_exit |= (previous > 0) & (membership > 0)
        forced_exit &= market_sellable_values
        forced_exit_weight = float(previous[forced_exit].sum())
        if optimizer_type == "opt2":
            forced_turnover_weight = float(
                sellable_values[forced_exit].sum() / target_stock_amount
            )
        else:
            forced_turnover_weight = forced_exit_weight

        liquidity_enabled = (
            float(self.optimizer["maxtrd"]) > 0
            or float(self.optimizer["maxpos"]) > 0
        )
        if liquidity_enabled:
            assert self.amount is not None
            liquidity = self._row_at_delay(
                self.amount,
                date,
                int(self.optimizer["liquidity_delay"]),
                "daily amount",
            )
            liquidity_valid = np.isfinite(liquidity) & (liquidity > 0)
        else:
            liquidity = np.full(len(self.columns), np.inf, dtype=np.float64)
            liquidity_valid = np.ones(len(self.columns), dtype=bool)
        lambda_slp = float(self.optimizer["lambda_slp"])
        if lambda_slp > 0:
            assert self.slippage is not None
            slippage = self._row_at_delay(
                self.slippage,
                date,
                int(self.optimizer["slippage_delay"]),
                "spread slippage",
            )
            slippage_price = self._row_at_delay(
                self.close,
                date,
                int(self.optimizer["slippage_delay"]),
                "slippage close",
            )
            slippage_valid = (
                np.isfinite(slippage)
                & (slippage >= 0)
                & np.isfinite(slippage_price)
                & (slippage_price > 0)
            )
            if not slippage_valid.any():
                raise ValueError(
                    f"{date}: spread slippage has no valid values at "
                    f"delay={int(self.optimizer['slippage_delay'])}"
                )
        else:
            slippage = np.zeros(len(self.columns), dtype=np.float64)
            slippage_price = np.ones(len(self.columns), dtype=np.float64)
            slippage_valid = np.ones(len(self.columns), dtype=bool)

        base = self._row_at_delay(self.base, date, 0, "base universe")

        ret_pos = int(self.returns.index.searchsorted(date))
        history = self.returns.iloc[max(0, ret_pos - int(self.optimizer["ret_days"])) : ret_pos]
        return_valid = (
            np.isfinite(history.to_numpy(dtype=np.float32)).sum(axis=0)
            >= int(self.optimizer["min_return_obs"])
        )
        buy_candidate = (
            ((alpha > 0) | (benchmark > 0))
            & buyable_values
            & return_valid
            & liquidity_valid
        )
        slippage_excluded_candidates = 0
        if lambda_slp > 0:
            slippage_excluded_candidates = int((buy_candidate & ~slippage_valid).sum())
            buy_candidate &= slippage_valid
        held = current_values > 0
        decision_mask = buy_candidate | (held & market_sellable_values)
        frozen = held & ~market_sellable_values
        tradable_candidate_count = int(decision_mask.sum())
        candidate_idx = np.flatnonzero(decision_mask | frozen)
        modeled_instrument_count = int(len(candidate_idx))
        if modeled_instrument_count < int(self.optimizer["min_valid_instruments"]):
            raise ValueError(
                f"{date}: only {modeled_instrument_count} modeled optimizer instruments; "
                f"minimum is {int(self.optimizer['min_valid_instruments'])}"
            )

        local_alpha = alpha[candidate_idx]
        local_benchmark = benchmark[candidate_idx]
        local_previous = previous[candidate_idx]
        local_buy = buy_candidate[candidate_idx]
        local_frozen = frozen[candidate_idx]
        local_actual_weight = current_values[candidate_idx] / target_stock_amount
        max_weight = float(self.optimizer["max_weight"])
        upper = np.full(len(candidate_idx), max_weight, dtype=np.float64)
        target_size = float(self.optimizer["target_size"])
        maxpos = float(self.optimizer["maxpos"])
        if maxpos > 0:
            capacity_scale = target_size if optimizer_type == "opt1" else target_stock_amount
            capacity_upper = maxpos * liquidity[candidate_idx] / capacity_scale
            capacity_upper = np.where(
                np.isfinite(capacity_upper) & (capacity_upper >= 0),
                capacity_upper,
                0.0,
            )
            upper = np.minimum(upper, capacity_upper)
        upper = np.maximum(upper, np.where(local_previous > max_weight, local_previous, 0.0))
        if maxpos > 0:
            upper = np.maximum(upper, local_previous)
        upper[~local_buy] = local_actual_weight[~local_buy]
        local_forced_exit = forced_exit[candidate_idx]
        if optimizer_type == "opt2":
            upper[local_forced_exit] = locked_weight[candidate_idx][local_forced_exit]
        else:
            upper[local_forced_exit] = 0.0
        upper[local_frozen] = local_actual_weight[local_frozen]

        maxtrd = float(self.optimizer["maxtrd"])
        if maxtrd > 0:
            trade_scale = target_size if optimizer_type == "opt1" else target_stock_amount
            trade_upper = maxtrd * liquidity[candidate_idx] / trade_scale
            trade_upper = np.where(
                np.isfinite(trade_upper) & (trade_upper >= 0), trade_upper, 0.0
            )
        else:
            trade_upper = np.full(len(candidate_idx), np.inf, dtype=np.float64)
        if lambda_slp > 0:
            trade_upper[~slippage_valid[candidate_idx]] = 0.0
        forced_trade_weight = (
            sellable_values[candidate_idx] / target_stock_amount
            if optimizer_type == "opt2"
            else local_previous
        )
        trade_upper[local_forced_exit] = np.maximum(
            trade_upper[local_forced_exit], forced_trade_weight[local_forced_exit]
        )
        slippage_rate = np.divide(
            slippage[candidate_idx],
            slippage_price[candidate_idx],
            out=np.zeros(len(candidate_idx), dtype=np.float64),
            where=slippage_valid[candidate_idx],
        )

        factor_mask = np.isfinite(base) & (base != 0)
        hard_risk_rows = [
            (limit_spec, self._factor(limit_spec, date, factor_mask))
            for limit_spec in self.hard_risk
        ]
        soft_risk_rows = [
            (limit_spec, self._factor(limit_spec, date, factor_mask))
            for limit_spec in self.soft_risk
        ]
        hard_group_rows = [
            (
                limit_spec,
                self._row_at_delay(
                    self.groups[limit_spec.name],
                    date,
                    limit_spec.delay,
                    limit_spec.name,
                ),
            )
            for limit_spec in self.hard_group
        ]
        soft_group_rows = [
            (
                limit_spec,
                self._row_at_delay(
                    self.groups[limit_spec.name],
                    date,
                    limit_spec.delay,
                    limit_spec.name,
                ),
            )
            for limit_spec in self.soft_group
        ]
        risk_matrix = None
        if float(self.optimizer["lambda0"]) > 0:
            risk_matrix = self._variance_matrix(date, candidate_idx)

        with Model(f"default_optimizer_{date}") as model:
            model.setSolverParam("numThreads", int(self.optimizer["num_mosek_threads"]))
            if float(self.optimizer["max_time"]) > 0:
                model.setSolverParam("optimizerMaxTime", float(self.optimizer["max_time"]))

            if optimizer_type == "opt1":
                lower = np.zeros(len(candidate_idx), dtype=np.float64)
                lower[local_frozen] = local_actual_weight[local_frozen]
                position = model.variable(
                    "weights", len(candidate_idx), Domain.greaterThan(lower.tolist())
                )
                weights = position
                current_decision = local_previous.copy()
                current_decision[~local_buy] = local_actual_weight[~local_buy]
                normalized_trade_scale = 1.0
            else:
                local_locked_amount = locked_values[candidate_idx]
                lower = local_locked_amount.copy()
                lower[local_frozen] = current_values[candidate_idx][local_frozen]
                position = model.variable(
                    "amounts",
                    len(candidate_idx),
                    Domain.greaterThan(lower.tolist()),
                )
                weights = Expr.mul(1.0 / target_stock_amount, position)
                current_decision = current_values[candidate_idx]
                normalized_trade_scale = 1.0 / target_stock_amount
            model.constraint("position_upper", weights, Domain.lessThan(upper.tolist()))
            model.constraint("stock_budget", Expr.sum(weights), Domain.equalsTo(1.0))
            objective = Expr.dot(local_alpha.tolist(), weights)

            absolute_trade = model.variable(
                "absolute_trade", len(candidate_idx), Domain.greaterThan(0.0)
            )
            model.constraint(
                "trade_positive",
                Expr.sub(Expr.sub(position, current_decision.tolist()), absolute_trade),
                Domain.lessThan(0.0),
            )
            model.constraint(
                "trade_negative",
                Expr.sub(Expr.sub(current_decision.tolist(), position), absolute_trade),
                Domain.lessThan(0.0),
            )
            normalized_trade = Expr.mul(normalized_trade_scale, absolute_trade)
            if maxtrd > 0:
                model.constraint(
                    "maxtrd",
                    normalized_trade,
                    Domain.lessThan(trade_upper.tolist()),
                )
            if has_previous or executed_turnover_today > 0:
                remaining_turnover = max(
                    float(self.optimizer["maxtvr"])
                    - executed_turnover_today / target_stock_amount,
                    0.0,
                )
                model.constraint(
                    "maxtvr",
                    Expr.sum(normalized_trade),
                    Domain.lessThan(
                        remaining_turnover + forced_turnover_weight
                    ),
                )
            if lambda_slp > 0:
                objective = Expr.sub(
                    objective,
                    Expr.mul(
                        lambda_slp,
                        Expr.dot(slippage_rate.tolist(), normalized_trade),
                    ),
                )

            for index, (limit_spec, membership) in enumerate(hard_univ_rows):
                coefficient = membership[candidate_idx]
                exposure_weights = weights
                if limit_spec.hi == 0:
                    fixed_weight = np.zeros(len(candidate_idx), dtype=np.float64)
                    fixed_weight[local_frozen] = local_actual_weight[local_frozen]
                    if optimizer_type == "opt2":
                        fixed_weight = np.maximum(
                            fixed_weight, locked_weight[candidate_idx]
                        )
                    exposure_weights = Expr.sub(
                        weights, fixed_weight.tolist()
                    )
                model.constraint(
                    f"univ_{index}_{_safe_name(limit_spec.name)}",
                    Expr.dot(coefficient.tolist(), exposure_weights),
                    Domain.inRange(limit_spec.lo, limit_spec.hi),
                )

            for index, limit_spec in enumerate(self.soft_univ):
                coefficient = self._universe_membership(
                    limit_spec.name, date, limit_spec.delay
                )[candidate_idx]
                objective = self._add_soft_range(
                    model,
                    Expr.dot(coefficient.tolist(), weights),
                    limit_spec,
                    float(self.optimizer["soft_univ_penalty"]),
                    objective,
                    f"soft_univ_{index}_{_safe_name(limit_spec.name)}",
                )

            for index, (limit_spec, full_factor) in enumerate(hard_risk_rows):
                coefficient = full_factor[candidate_idx]
                benchmark_exposure = float(np.dot(benchmark, full_factor))
                active = Expr.sub(
                    Expr.dot(coefficient.tolist(), weights), benchmark_exposure
                )
                model.constraint(
                    f"risk_{index}_{_safe_name(limit_spec.name)}",
                    active,
                    Domain.inRange(limit_spec.lo, limit_spec.hi),
                )

            for index, (limit_spec, full_factor) in enumerate(soft_risk_rows):
                coefficient = full_factor[candidate_idx]
                benchmark_exposure = float(np.dot(benchmark, full_factor))
                active = Expr.sub(
                    Expr.dot(coefficient.tolist(), weights), benchmark_exposure
                )
                objective = self._add_soft_range(
                    model,
                    active,
                    limit_spec,
                    float(self.optimizer["soft_risk_penalty"]),
                    objective,
                    f"soft_risk_{index}_{_safe_name(limit_spec.name)}",
                )

            for index, (limit_spec, group_values) in enumerate(hard_group_rows):
                for group_value in np.unique(group_values[np.isfinite(group_values)]):
                    full_membership = (group_values == group_value).astype(np.float64)
                    coefficient = full_membership[candidate_idx]
                    benchmark_exposure = float(np.dot(benchmark, full_membership))
                    active = Expr.sub(
                        Expr.dot(coefficient.tolist(), weights), benchmark_exposure
                    )
                    label = _safe_name(
                        f"group_{index}_{limit_spec.name}_{group_value:g}"
                    )
                    model.constraint(
                        label,
                        active,
                        Domain.inRange(limit_spec.lo, limit_spec.hi),
                    )

            for index, (limit_spec, group_values) in enumerate(soft_group_rows):
                for group_value in np.unique(group_values[np.isfinite(group_values)]):
                    full_membership = (group_values == group_value).astype(np.float64)
                    coefficient = full_membership[candidate_idx]
                    benchmark_exposure = float(np.dot(benchmark, full_membership))
                    active = Expr.sub(
                        Expr.dot(coefficient.tolist(), weights), benchmark_exposure
                    )
                    label = _safe_name(
                        f"soft_group_{index}_{limit_spec.name}_{group_value:g}"
                    )
                    objective = self._add_soft_range(
                        model,
                        active,
                        limit_spec,
                        float(self.optimizer["soft_group_penalty"]),
                        objective,
                        label,
                    )

            participation = float(self.optimizer["min_participation_ratio"])
            parti_penalty = float(self.optimizer["parti_penalty"])
            if participation > 0 or parti_penalty > 0:
                concentration = model.variable("participation", 1, Domain.greaterThan(0.0))
                model.constraint(
                    "participation_cone",
                    Expr.vstack(concentration, 0.5, weights),
                    Domain.inRotatedQCone(),
                )
                if participation > 0:
                    maximum_sum_squares = 1.0 / (
                        participation * tradable_candidate_count
                    )
                    model.constraint(
                        "participation_limit",
                        concentration,
                        Domain.lessThan(maximum_sum_squares),
                    )
                if parti_penalty > 0:
                    objective = Expr.sub(
                        objective, Expr.mul(parti_penalty, concentration)
                    )

            if risk_matrix is not None:
                active_weight = Expr.sub(weights, local_benchmark.tolist())
                active_return = Expr.mul(risk_matrix, active_weight)
                variance = model.variable("variance", 1, Domain.greaterThan(0.0))
                model.constraint(
                    "variance_cone",
                    Expr.vstack(variance, 0.5, active_return),
                    Domain.inRotatedQCone(),
                )
                objective = Expr.sub(
                    objective,
                    Expr.mul(float(self.optimizer["lambda0"]), variance),
                )

            model.objective(ObjectiveSense.Maximize, objective)
            started = time.monotonic()
            try:
                model.solve()
            except Exception as exc:
                raise RuntimeError(f"{date}: MOSEK optimizer failed: {exc}") from exc
            solve_seconds = time.monotonic() - started
            solution_status = model.getPrimalSolutionStatus()
            solver_fallback = solution_status != SolutionStatus.Optimal
            if solver_fallback:
                warnings.warn(
                    f"{date}: MOSEK solution status is {solution_status}; "
                    "returning zero orders",
                    RuntimeWarning,
                    stacklevel=2,
                )
                solution = local_previous.copy()
                solution[local_frozen] = local_actual_weight[local_frozen]
            else:
                solution = np.asarray(position.level(), dtype=np.float64)
                if optimizer_type == "opt2":
                    solution /= target_stock_amount

        if not np.isfinite(solution).all() or float(solution.min()) < -1.0e-7:
            raise RuntimeError(f"{date}: MOSEK returned invalid target weights")
        solution = np.maximum(solution, 0.0)
        if not solver_fallback:
            trim_threshold = float(self.optimizer["trim_threshold"])
            if trim_threshold > 0:
                can_trim = (locked_weight[candidate_idx] <= 0) & ~local_frozen
                solution[(solution < trim_threshold) & can_trim] = 0.0
            if bool(self.optimizer["post_trim_renorm"]):
                if optimizer_type == "opt1":
                    free = ~local_frozen
                    free_total = float(solution[free].sum())
                    target_free = max(1.0 - float(solution[local_frozen].sum()), 0.0)
                    if free_total > 0:
                        solution[free] *= target_free / free_total
                else:
                    fixed_weight = locked_weight[candidate_idx].copy()
                    fixed_weight[local_frozen] = local_actual_weight[local_frozen]
                    excess = np.maximum(solution - fixed_weight, 0.0)
                    excess_total = float(excess.sum())
                    target_excess = max(1.0 - float(fixed_weight.sum()), 0.0)
                    if excess_total > 0:
                        solution = fixed_weight + excess * (
                            target_excess / excess_total
                        )

        result = np.zeros(len(self.columns), dtype=np.float64)
        result[candidate_idx] = solution
        target_weight = pd.Series(result, index=self.columns, name=date, dtype=float)
        target_weight.attrs["optimizer_type"] = optimizer_type
        target_weight.attrs["solve_seconds"] = solve_seconds
        target_weight.attrs["solver_status"] = str(solution_status)
        target_weight.attrs["solver_fallback"] = solver_fallback
        target_weight.attrs["slippage_valid_candidates"] = int(
            (slippage_valid & decision_mask).sum()
        )
        target_weight.attrs["candidate_count"] = tradable_candidate_count
        target_weight.attrs["tradable_candidate_count"] = tradable_candidate_count
        target_weight.attrs["modeled_instrument_count"] = modeled_instrument_count
        target_weight.attrs["frozen_count"] = int(frozen.sum())
        target_weight.attrs["slippage_excluded_candidates"] = slippage_excluded_candidates
        target_weight.attrs["forced_exit_count"] = int(forced_exit.sum())
        target_weight.attrs["forced_exit_weight"] = forced_exit_weight
        target_weight.attrs["forced_exit_sellable_amount"] = float(
            sellable_values[forced_exit].sum()
        )
        target_weight.attrs["forced_exit_unpriced_count"] = int(
            (forced_exit & ~slippage_valid).sum()
        )
        target_weight.attrs["forced_exit_unpriced_weight"] = float(
            previous[forced_exit & ~slippage_valid].sum()
        )
        if solver_fallback:
            delta_amount = np.zeros(len(self.columns), dtype=np.float64)
        else:
            delta_amount = target_weight.to_numpy(dtype=np.float64) * target_stock_amount
            delta_amount -= current_values
            tolerance = 1.0e-5
            invalid_buy = (delta_amount > tolerance) & ~buyable_values
            invalid_sell = (delta_amount < -tolerance) & ~market_sellable_values
            if invalid_buy.any() or invalid_sell.any():
                bad = self.columns[invalid_buy | invalid_sell].tolist()
                raise RuntimeError(
                    f"{date}: optimizer generated orders outside the tradable pool: {bad[:10]}"
                )
            delta_amount[(delta_amount > 0) & ~buyable_values] = 0.0
            delta_amount[(delta_amount < 0) & ~market_sellable_values] = 0.0
        orders = pd.DataFrame(
            {
                "buy_amount": np.maximum(delta_amount, 0.0),
                "sell_amount": np.maximum(-delta_amount, 0.0),
            },
            index=self.columns,
        )
        orders.attrs.update(target_weight.attrs)
        orders.attrs["target_weight"] = target_weight
        return orders
