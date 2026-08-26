from __future__ import annotations

import math
import re
import time
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


def _parse_limits(raw: str, *, soft: bool, label: str) -> tuple[RangeLimit, ...]:
    limits: list[RangeLimit] = []
    for entry in (item.strip() for item in raw.split("|")):
        if not entry:
            continue
        pieces = [piece.strip() for piece in entry.split(",")]
        if len(pieces) > 2:
            raise ValueError(f"strategy.optimizer.{label} has too many comma fields: {entry!r}")
        fields = pieces[0].split(":")
        expected = 4 if soft else 3
        if len(fields) != expected:
            shape = "name:lo:hi:penalty" if soft else "name:lo:hi"
            raise ValueError(f"strategy.optimizer.{label} entry must be {shape}[,delay]: {entry!r}")
        name = fields[0]
        lo = float(fields[1])
        hi = float(fields[2])
        penalty = float(fields[3]) if soft else 1.0
        delay = int(pieces[1]) if len(pieces) == 2 else 0
        if not name or lo > hi or penalty < 0 or delay < 0:
            raise ValueError(f"invalid strategy.optimizer.{label} entry: {entry!r}")
        limits.append(RangeLimit(name, lo, hi, penalty, delay))
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
        self.index_weight = self._normalize_frame(
            self.dataloader.get_index_weight(history_start, end_ds)
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

        factor_names = {limit.name for limit in (*self.hard_risk, *self.soft_risk)}
        barra_names = sorted(
            name.split(".", 1)[1]
            for name in factor_names
            if name.startswith("BarraCNE5.")
        )
        supported = {"returns120", "vola_30", "vola_5", "close"}
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
        return pd.Series(values, index=self.returns.index, name="ZZ500_return")

    def _bind_columns(self, columns: pd.Index) -> None:
        normalized = pd.Index([str(value).zfill(6) for value in columns])
        frames = {
            "returns": self.returns,
            "close": self.close,
            "index weight": self.index_weight,
            "base universe": self.base,
            "limit mask": self.limit,
            "suspend mask": self.suspend,
            **self.styles,
        }
        for label, frame in frames.items():
            if not frame.columns.equals(normalized):
                raise ValueError(f"{label} columns do not match the backtest instrument axis")
        self.columns = normalized

    @staticmethod
    def _row_at_delay(frame: pd.DataFrame, date: int, delay: int, label: str) -> np.ndarray:
        pos = int(frame.index.searchsorted(date))
        if pos >= len(frame.index) or int(frame.index[pos]) != date:
            raise ValueError(f"{label}: date {date} not found")
        row_pos = pos - delay
        if row_pos < 0:
            raise ValueError(f"{label}: date {date} lacks delay={delay} history")
        return frame.iloc[row_pos].to_numpy(dtype=np.float64)

    def _actual_previous(self, last_hold: pd.Series | None) -> tuple[np.ndarray, bool]:
        assert self.columns is not None
        if last_hold is None:
            return np.zeros(len(self.columns), dtype=np.float64), False
        values = pd.to_numeric(last_hold.reindex(self.columns), errors="coerce").to_numpy(dtype=np.float64)
        values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
        total = float(values.sum())
        if total <= 0:
            return np.zeros(len(self.columns), dtype=np.float64), False
        return values / total, True

    def _benchmark(self, date: int) -> np.ndarray:
        values = self._row_at_delay(
            self.index_weight,
            date,
            int(self.optimizer["benchmark_delay"]),
            "ZZ500 weight",
        )
        valid = np.isfinite(values) & (values > 0)
        total = float(values[valid].sum())
        if total <= 0:
            raise ValueError(f"ZZ500 weight {date} has no positive values")
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
        if limit.name in self.styles:
            values = self._row_at_delay(self.styles[limit.name], date, limit.delay, limit.name)
            valid = factor_mask & np.isfinite(values)
            if valid.any():
                values = values.copy()
                values[valid] -= float(values[valid].mean())
        else:
            values = self._derived_factor(limit.name, date, limit.delay)
        return _centered_rank(values, factor_mask)

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

    def generate_positions(self, signals, last_hold):
        from mosek.fusion import Domain, Expr, Model, ObjectiveSense

        if self.columns is None:
            self._bind_columns(signals.index)
        assert self.columns is not None
        date = int(self.dataloader.date)
        signal_values = pd.to_numeric(signals.reindex(self.columns), errors="coerce").to_numpy(dtype=np.float64)
        alpha = _normalize_alpha(signal_values, float(self.optimizer["trim_threshold"]))
        benchmark = self._benchmark(date)
        previous, has_previous = self._actual_previous(last_hold)

        base = self._row_at_delay(self.base, date, 0, "base universe")
        limit = self._row_at_delay(self.limit, date, 0, "limit mask")
        suspend = self._row_at_delay(self.suspend, date, 0, "suspend mask")
        tradable = (
            np.isfinite(base)
            & (base != 0)
            & np.isfinite(limit)
            & (limit != 0)
            & np.isfinite(suspend)
            & (suspend != 0)
        )

        ret_pos = int(self.returns.index.searchsorted(date))
        history = self.returns.iloc[max(0, ret_pos - int(self.optimizer["ret_days"])) : ret_pos]
        return_valid = (
            np.isfinite(history.to_numpy(dtype=np.float32)).sum(axis=0)
            >= int(self.optimizer["min_return_obs"])
        )
        buy_candidate = ((alpha > 0) | (benchmark > 0)) & tradable & return_valid
        held = previous > 0
        candidate_idx = np.flatnonzero(buy_candidate | held)
        if len(candidate_idx) < int(self.optimizer["min_valid_instruments"]):
            raise ValueError(
                f"{date}: only {len(candidate_idx)} optimizer candidates; "
                f"minimum is {int(self.optimizer['min_valid_instruments'])}"
            )

        local_alpha = alpha[candidate_idx]
        local_benchmark = benchmark[candidate_idx]
        local_previous = previous[candidate_idx]
        local_buy = buy_candidate[candidate_idx]
        max_weight = float(self.optimizer["max_weight"])
        upper = np.full(len(candidate_idx), max_weight, dtype=np.float64)
        upper[~local_buy] = local_previous[~local_buy]
        upper = np.maximum(upper, np.where(local_previous > max_weight, local_previous, 0.0))

        factor_mask = np.isfinite(base) & (base != 0)
        hard_risk_rows = [
            (limit_spec, self._factor(limit_spec, date, factor_mask))
            for limit_spec in self.hard_risk
        ]
        soft_risk_rows = [
            (limit_spec, self._factor(limit_spec, date, factor_mask))
            for limit_spec in self.soft_risk
        ]
        risk_matrix = None
        if float(self.optimizer["lambda0"]) > 0:
            risk_matrix = self._variance_matrix(date, candidate_idx)

        with Model(f"default_optimizer_{date}") as model:
            model.setSolverParam("numThreads", int(self.optimizer["num_mosek_threads"]))
            if float(self.optimizer["max_time"]) > 0:
                model.setSolverParam("optimizerMaxTime", float(self.optimizer["max_time"]))

            weights = model.variable("weights", len(candidate_idx), Domain.greaterThan(0.0))
            model.constraint("position_upper", weights, Domain.lessThan(upper.tolist()))
            model.constraint("booksize", Expr.sum(weights), Domain.equalsTo(1.0))
            objective = Expr.dot(local_alpha.tolist(), weights)

            if has_previous:
                absolute_trade = model.variable(
                    "absolute_trade", len(candidate_idx), Domain.greaterThan(0.0)
                )
                model.constraint(
                    "trade_positive",
                    Expr.sub(Expr.sub(weights, local_previous.tolist()), absolute_trade),
                    Domain.lessThan(0.0),
                )
                model.constraint(
                    "trade_negative",
                    Expr.sub(Expr.sub(local_previous.tolist(), weights), absolute_trade),
                    Domain.lessThan(0.0),
                )
                model.constraint(
                    "maxtvr",
                    Expr.sum(absolute_trade),
                    Domain.lessThan(float(self.optimizer["maxtvr"])),
                )

            for index, limit_spec in enumerate(self.hard_univ):
                if limit_spec.name != "ZZ500":
                    raise ValueError(f"unsupported optimizer universe: {limit_spec.name}")
                membership = self._row_at_delay(
                    self.index_weight, date, limit_spec.delay, limit_spec.name
                )
                coefficient = (
                    np.isfinite(membership[candidate_idx]) & (membership[candidate_idx] > 0)
                ).astype(np.float64)
                model.constraint(
                    f"univ_{index}_ZZ500",
                    Expr.dot(coefficient.tolist(), weights),
                    Domain.inRange(limit_spec.lo, limit_spec.hi),
                )

            for index, limit_spec in enumerate(self.soft_univ):
                if limit_spec.name != "ZZ500":
                    raise ValueError(f"unsupported optimizer universe: {limit_spec.name}")
                membership = self._row_at_delay(
                    self.index_weight, date, limit_spec.delay, limit_spec.name
                )
                coefficient = (
                    np.isfinite(membership[candidate_idx]) & (membership[candidate_idx] > 0)
                ).astype(np.float64)
                objective = self._add_soft_range(
                    model,
                    Expr.dot(coefficient.tolist(), weights),
                    limit_spec,
                    float(self.optimizer["soft_univ_penalty"]),
                    objective,
                    f"soft_univ_{index}_ZZ500",
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

            participation = float(self.optimizer["min_participation_ratio"])
            if participation > 0:
                maximum_sum_squares = 1.0 / (participation * len(candidate_idx))
                concentration = model.variable("participation", 1, Domain.greaterThan(0.0))
                model.constraint(
                    "participation_cone",
                    Expr.vstack(concentration, 0.5, weights),
                    Domain.inRotatedQCone(),
                )
                model.constraint(
                    "participation_limit",
                    concentration,
                    Domain.lessThan(maximum_sum_squares),
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
                solution = np.asarray(weights.level(), dtype=np.float64)
            except Exception as exc:
                raise RuntimeError(f"{date}: MOSEK optimizer failed: {exc}") from exc
            solve_seconds = time.monotonic() - started

        if not np.isfinite(solution).all() or float(solution.min()) < -1.0e-7:
            raise RuntimeError(f"{date}: MOSEK returned invalid target weights")
        solution = np.maximum(solution, 0.0)
        trim_threshold = float(self.optimizer["trim_threshold"])
        if trim_threshold > 0:
            solution[solution < trim_threshold] = 0.0
        if bool(self.optimizer["post_trim_renorm"]):
            total = float(solution.sum())
            if total > 0:
                solution /= total

        result = np.zeros(len(self.columns), dtype=np.float64)
        result[candidate_idx] = solution
        output = pd.Series(result, index=self.columns, name=date, dtype=float)
        output.attrs["solve_seconds"] = solve_seconds
        return output
