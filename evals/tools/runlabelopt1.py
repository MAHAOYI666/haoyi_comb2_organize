from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

EVALS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALS_ROOT.parent
for local_path in (
    REPO_ROOT,
    REPO_ROOT / "vendor" / "comb2",
    REPO_ROOT / "vendor" / "comb2-simbase",
    REPO_ROOT / "vendor" / "comb2-pcmaster",
):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

import numpy as np
import pandas as pd
import torch

from comb2.op_utils import (
    nan_to_num,
    nanmedian,
    nanstd,
    normalize_by_max_abs,
    truncate,
    winsorize_by_quantile,
)
from comb2_pcmaster import BacktestNode, DailyBacktest
from comb2_simbase import IndexMask, Memmaper2
from comb2_simbase.cache_layout import (
    BASE_UNIVERSE_MASK_NAME,
    FILTERED_MASK_NAME,
    VALID_MASK_NAME,
    ashare_cache_path,
    daily_label_path,
    stock_mask_path,
)
from config import DEFAULT_OPTIMIZER_CONFIG


RET_DAYS = 5
DECAY_WEIGHTS = torch.arange(RET_DAYS, 0, -1, dtype=torch.float32)
TRADING_DAYS = 250.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the legacy retDays=5 oracle label signal and pass it directly "
            "to the default opt1 daily backtest."
        )
    )
    parser.add_argument(
        "--cache-path",
        required=True,
        help="Parent directory containing AshareCache",
    )
    parser.add_argument("--signal-start", type=int, default=20160101)
    parser.add_argument(
        "--signal-end",
        type=int,
        help="Last signal date; defaults to the latest date with a complete five-label window",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cash", type=float, default=10_000_000.0)
    parser.add_argument("--fee-rate", type=float, default=0.00075)
    parser.add_argument("--reserve-cash", type=float, default=0.95)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.Index(result.index.to_numpy(dtype=np.int64), name="date")
    result.columns = pd.Index(
        [str(value).zfill(6) for value in result.columns], name="code"
    )
    return result.sort_index()


def resolve_dates(
    label_path: Path,
    signal_start: int,
    signal_end: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    label_dates = np.load(label_path / "index.npy", allow_pickle=True).astype(np.int64)
    assert len(label_dates) >= RET_DAYS
    first_pos = int(np.searchsorted(label_dates, int(signal_start), side="left"))
    last_complete_pos = len(label_dates) - RET_DAYS
    if signal_end is None:
        last_pos = last_complete_pos
    else:
        last_pos = int(np.searchsorted(label_dates, int(signal_end), side="right")) - 1
        assert last_pos <= last_complete_pos, (
            f"signal_end={signal_end} has no complete {RET_DAYS}-label window; "
            f"latest valid signal date is {int(label_dates[last_complete_pos])}"
        )
    assert 0 <= first_pos <= last_pos, "requested signal range has no eligible dates"

    signal_dates = label_dates[first_pos : last_pos + 1]
    trade_dates = np.asarray(IndexMask().date, dtype=np.int64)
    trade_positions = np.searchsorted(trade_dates, signal_dates)
    assert np.array_equal(trade_dates[trade_positions], signal_dates), (
        "all signal dates must be trading dates"
    )
    assert np.all(trade_positions + 1 < len(trade_dates)), (
        "every signal date must have a following execution date"
    )
    execution_dates = trade_dates[trade_positions + 1]
    label_window_end = int(label_dates[last_pos + RET_DAYS - 1])
    return signal_dates, execution_dates, label_window_end


def load_inputs(
    cache_path: Path,
    signal_dates: np.ndarray,
    label_window_end: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    label = normalize_frame(
        Memmaper2(str(daily_label_path(cache_path, "vwap30_label1d")))
        .load(
            start_ds=int(signal_dates[0]),
            end_ds=int(label_window_end),
            df_type=True,
        )
        .dloc[:]
    )
    required_label_dates = len(signal_dates) + RET_DAYS - 1
    assert len(label) == required_label_dates, (
        f"loaded {len(label)} label dates, expected {required_label_dates}"
    )
    assert np.array_equal(
        label.index.to_numpy(dtype=np.int64)[: len(signal_dates)], signal_dates
    )

    valid = np.ones((len(signal_dates), len(label.columns)), dtype=bool)
    for name in (VALID_MASK_NAME, FILTERED_MASK_NAME, BASE_UNIVERSE_MASK_NAME):
        mask = normalize_frame(
            Memmaper2(str(stock_mask_path(cache_path, name)))
            .load(
                start_ds=int(signal_dates[0]),
                end_ds=int(signal_dates[-1]),
                df_type=True,
            )
            .dloc[:]
        ).reindex(index=signal_dates, columns=label.columns)
        values = mask.to_numpy(dtype=np.float32, copy=False)
        valid &= np.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        ) > 0
    return label, valid


def build_signal(
    label_values: np.ndarray,
    valid_mask: np.ndarray,
    offset: int,
) -> np.ndarray:
    returns = torch.tensor(
        label_values[offset : offset + RET_DAYS], dtype=torch.float32
    )
    returns = nan_to_num(returns, 0.0)
    signal = torch.tensordot(DECAY_WEIGHTS, returns, dims=([0], [0]))
    valid = torch.as_tensor(valid_mask[offset], dtype=torch.bool)
    valid &= torch.isfinite(signal)
    valid_values = signal[valid]
    assert valid_values.numel() > 0, f"signal row {offset} has no valid instruments"
    valid_values = winsorize_by_quantile(valid_values, 0.01, 0.99)
    valid_values = valid_values - nanmedian(valid_values)
    valid_values = valid_values / (nanstd(valid_values) + 1.0e-8)
    valid_values = truncate(valid_values, -3.0, 3.0)
    valid_values = normalize_by_max_abs(valid_values)
    signal[valid] = valid_values
    signal[~valid] = 0.0
    return signal.to(torch.float16).cpu().numpy()


def prepare_output(output_dir: Path) -> None:
    if output_dir.exists():
        assert not any(output_dir.iterdir()), (
            f"output directory is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def backtest_node(
    cache_path: Path,
    output_dir: Path,
    execution_dates: np.ndarray,
    args: argparse.Namespace,
) -> BacktestNode:
    optimizer = dict(DEFAULT_OPTIMIZER_CONFIG)
    optimizer["type"] = "opt1"
    strategy_path = (
        REPO_ROOT
        / "vendor"
        / "comb2-pcmaster"
        / "comb2_pcmaster"
        / "default_strategy.py"
    ).resolve()
    strategy_config = {
        "start_ds": int(execution_dates[0]),
        "end_ds": int(execution_dates[-1]),
        "path": str(strategy_path),
        "optimizer": optimizer,
    }
    return BacktestNode(
        start_ds=int(execution_dates[0]),
        end_ds=int(execution_dates[-1]),
        output_path=str(output_dir),
        strategy_path=str(strategy_path),
        strategy_class="AlphaStrategy",
        strategy_config=strategy_config,
        cash=float(args.cash),
        fee_rate=float(args.fee_rate),
        reserve_cash=float(args.reserve_cash),
        daily_metrics_file="daily_metrics.csv",
        cache_path=str(cache_path),
        verbose=False,
        universe="base",
        execution_price="vwap30",
    )


def daily_row(
    result: dict,
    signal_date: int,
    previous_asset: float,
    wall_seconds: float,
) -> dict[str, object]:
    attrs = result["target_weight"].attrs
    total_asset = float(result["total_asset"])
    trade_cost = float(result["trade_cost"])
    return {
        "date": int(result["date"]),
        "signal_date": int(signal_date),
        "total_asset": total_asset,
        "net_return": total_asset / previous_asset - 1.0,
        "gross_return": (total_asset + trade_cost) / previous_asset - 1.0,
        "trade_cost": trade_cost,
        "tvr": float(result["tvr"]),
        "long_num": int(result["long_num"]),
        "solver_status": str(attrs.get("solver_status")),
        "solver_fallback": bool(attrs.get("solver_fallback", False)),
        "solve_seconds": float(attrs.get("solve_seconds", math.nan)),
        "candidate_count": int(attrs.get("candidate_count", 0)),
        "forced_exit_count": int(attrs.get("forced_exit_count", 0)),
        "wall_seconds": float(wall_seconds),
    }


def metric_summary(frame: pd.DataFrame, initial_cash: float) -> dict[str, object]:
    net = frame["net_return"].to_numpy(dtype=float)
    gross = frame["gross_return"].to_numpy(dtype=float)
    recurring_tvr = frame["tvr"].iloc[1:].to_numpy(dtype=float)
    net_std = float(np.std(net, ddof=1))
    gross_std = float(np.std(gross, ddof=1))
    equity = np.concatenate(([float(initial_cash)], frame["total_asset"].to_numpy(dtype=float)))
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    years = len(frame) / TRADING_DAYS
    final_asset = float(frame["total_asset"].iloc[-1])
    return {
        "days": int(len(frame)),
        "execution_start": int(frame["date"].iloc[0]),
        "execution_end": int(frame["date"].iloc[-1]),
        "signal_start": int(frame["signal_date"].iloc[0]),
        "signal_end": int(frame["signal_date"].iloc[-1]),
        "initial_cash": float(initial_cash),
        "final_asset": final_asset,
        "total_net_return_pct": (final_asset / float(initial_cash) - 1.0) * 100.0,
        "annualized_net_return_pct": float(np.mean(net) * TRADING_DAYS * 100.0),
        "annualized_gross_return_pct": float(np.mean(gross) * TRADING_DAYS * 100.0),
        "geometric_annualized_net_return_pct": (
            math.exp(math.log(final_asset / float(initial_cash)) / years) - 1.0
        ) * 100.0,
        "net_sharpe": float(np.mean(net) / net_std * math.sqrt(TRADING_DAYS)),
        "gross_sharpe": float(np.mean(gross) / gross_std * math.sqrt(TRADING_DAYS)),
        "max_drawdown_pct": float(np.min(drawdown) * 100.0),
        "avg_tvr_pct_including_entry": float(frame["tvr"].mean() * 100.0),
        "avg_tvr_pct_recurring": float(np.mean(recurring_tvr) * 100.0),
        "median_tvr_pct_recurring": float(np.median(recurring_tvr) * 100.0),
        "p95_tvr_pct_recurring": float(np.quantile(recurring_tvr, 0.95) * 100.0),
        "avg_long_count": float(frame["long_num"].mean()),
        "total_trade_cost": float(frame["trade_cost"].sum()),
        "fallback_days": int(frame["solver_fallback"].sum()),
        "solver_status_counts": dict(Counter(frame["solver_status"])),
        "total_solve_seconds": float(frame["solve_seconds"].sum()),
        "avg_solve_seconds": float(frame["solve_seconds"].mean()),
        "total_wall_seconds": float(frame["wall_seconds"].sum()),
    }


def yearly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["year"] = work["date"].astype(str).str[:4]
    rows = []
    for year, group in work.groupby("year", sort=True):
        net = group["net_return"]
        gross = group["gross_return"]
        rows.append(
            {
                "year": year,
                "days": len(group),
                "annualized_net_return_pct": net.mean() * TRADING_DAYS * 100.0,
                "annualized_gross_return_pct": gross.mean() * TRADING_DAYS * 100.0,
                "net_sharpe": net.mean() / net.std(ddof=1) * math.sqrt(TRADING_DAYS),
                "avg_tvr_pct": group["tvr"].mean() * 100.0,
                "avg_long_count": group["long_num"].mean(),
                "fallback_days": int(group["solver_fallback"].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def main() -> None:
    args = parse_args()
    assert args.cash > 0
    assert args.fee_rate >= 0
    assert 0 < args.reserve_cash <= 1
    assert args.progress_every > 0
    cache_path = Path(args.cache_path).expanduser().resolve()
    ashare_path = ashare_cache_path(cache_path)
    assert ashare_path.is_dir(), f"AshareCache not found: {ashare_path}"
    label_path = daily_label_path(cache_path, "vwap30_label1d")
    assert label_path.is_dir(), f"label cache not found: {label_path}"
    output_dir = Path(args.output_dir).expanduser().resolve()
    prepare_output(output_dir)

    signal_dates, execution_dates, label_window_end = resolve_dates(
        label_path, args.signal_start, args.signal_end
    )
    print(
        f"[SETUP] signals={int(signal_dates[0])}-{int(signal_dates[-1])} "
        f"executions={int(execution_dates[0])}-{int(execution_dates[-1])} "
        f"days={len(signal_dates)}"
    )
    label, valid = load_inputs(cache_path, signal_dates, label_window_end)
    label_values = label.to_numpy(dtype=np.float32, copy=False)

    init_start = time.perf_counter()
    node = backtest_node(cache_path, output_dir, execution_dates, args)
    backtest = DailyBacktest(node)
    print(f"[SETUP] backtest_initialized seconds={time.perf_counter() - init_start:.2f}")

    detail_path = output_dir / "opt1_daily.csv"
    position_path = output_dir / (
        f"{node.strategy_class}_{node.start_ds}_{node.end_ds}_position.csv"
    )
    rows: list[dict[str, object]] = []
    previous_asset = float(args.cash)
    loop_start = time.perf_counter()
    for offset, (signal_date, execution_date) in enumerate(
        zip(signal_dates, execution_dates), start=1
    ):
        signal = pd.Series(
            build_signal(label_values, valid, offset - 1),
            index=label.columns,
            dtype=float,
        )
        step_start = time.perf_counter()
        result = backtest.step(int(execution_date), signal)
        target_weight = result["target_weight"].reindex(backtest.universe.columns)
        target_values = target_weight.to_numpy(dtype=float)
        assert np.isfinite(target_values).all() and (target_values >= 0).all()
        target_weight.name = int(execution_date)
        target_weight.to_frame().T.to_csv(
            position_path,
            mode="a",
            header=offset == 1,
            index=True,
        )
        row = daily_row(
            result,
            int(signal_date),
            previous_asset,
            time.perf_counter() - step_start,
        )
        previous_asset = float(result["total_asset"])
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            detail_path,
            mode="a",
            header=offset == 1,
            index=False,
        )
        if offset % args.progress_every == 0 or offset == len(signal_dates):
            elapsed = time.perf_counter() - loop_start
            rate = elapsed / offset
            eta = rate * (len(signal_dates) - offset)
            print(
                f"[PROGRESS] {offset}/{len(signal_dates)} "
                f"date={int(execution_date)} elapsed={elapsed:.1f}s eta={eta:.1f}s "
                f"asset={previous_asset:.2f} tvr={float(result['tvr']):.4f}"
            )

    daily = pd.DataFrame(rows)
    summary = metric_summary(daily, float(args.cash))
    summary.update(
        {
            "signal": "legacy retDays=5 weighted vwap30_label1d oracle",
            "decay_weights": [5, 4, 3, 2, 1],
            "optimizer_type": "opt1",
            "fee_rate": float(args.fee_rate),
            "reserve_cash": float(args.reserve_cash),
            "cache_path": str(cache_path),
            "position_path": str(position_path),
            "position_rows": int(len(daily)),
            "position_columns": int(len(backtest.universe.columns)),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    yearly_summary(daily).to_csv(output_dir / "yearly_summary.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
