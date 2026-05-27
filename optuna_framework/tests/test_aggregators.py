"""Aggregator tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optuna_framework.aggregators import (
    build_baseline_thresholds,
    ensure_factor_audit_template,
    final_objective,
    load_baseline_thresholds,
    require_tuning_period_baseline,
    running_score,
    single_objective,
)
from optuna_framework.metrics_parser import WindowMetrics, parse_full_period, parse_window_metrics


FIXTURES = Path(__file__).parent / "fixtures"


def metric(sharpe: float, dd: float) -> WindowMetrics:
    return WindowMetrics(sharpe, dd, 0.1, 0.1, 1.0, 240, "row", ["row"])


def test_running_score() -> None:
    assert running_score([1.2]) == 1.2
    expected = float(np.mean([1.0, 2.0, 3.0]) - 0.3 * np.std([1.0, 2.0, 3.0]))
    assert running_score([1.0, 2.0, 3.0]) == expected


def test_single_objective_hard_filter() -> None:
    thresholds = load_baseline_thresholds(FIXTURES / "baseline_thresholds.json")
    score, fired = single_objective(metric(1.0, 0.08), thresholds["hard_filter"])
    assert fired is False
    assert score == 1.0
    assert final_objective([metric(1.0, 0.08)], thresholds["hard_filter"]) == (1.0, False)
    assert single_objective(metric(0.1, 0.08), thresholds["hard_filter"]) == (-10.0, True)
    assert single_objective(metric(1.0, 0.5), thresholds["hard_filter"]) == (-10.0, True)
    with pytest.raises(ValueError, match="exactly one"):
        final_objective([metric(1.0, 0.08), metric(1.1, 0.09)], thresholds["hard_filter"])


def test_missing_thresholds_message(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run_baseline.py"):
        load_baseline_thresholds(tmp_path / "missing.json")


def test_factor_audit_template(tmp_path) -> None:
    path = ensure_factor_audit_template(tmp_path)
    assert path.exists()
    first = path.read_text(encoding="utf-8")
    ensure_factor_audit_template(tmp_path)
    assert path.read_text(encoding="utf-8") == first
    assert "2024/2025" in first


def test_build_baseline_thresholds_from_single_full_run(tmp_path) -> None:
    summary = tmp_path / "pnl_summary.csv"
    summary.write_text((FIXTURES / "pnl_summary_full_period_sample.csv").read_text(encoding="utf-8"), encoding="utf-8")
    expected = _write_daily_pnl(tmp_path / "daily_pnl.csv")

    full_by_year = parse_full_period(summary)
    tuning_period = parse_window_metrics(
        summary,
        run_start_ds=20200102,
        run_end_ds=20240628,
        score_start_ds=20210104,
        score_end_ds=20231229,
    )
    thresholds = build_baseline_thresholds(full_by_year, tuning_period)

    assert set(thresholds) == {"tuning_period", "by_segment", "full_period", "hard_filter"}
    assert set(thresholds["by_segment"]) == {"holdout_2020", "holdout_2024h1"}
    assert set(thresholds["full_period"]["by_year"]) == {"2020", "2021", "2022", "2023", "2024"}
    assert thresholds["by_segment"]["holdout_2020"]["sharpe_idx"] == pytest.approx(0.6115)
    assert thresholds["by_segment"]["holdout_2024h1"]["sharpe_idx"] == pytest.approx(-0.7774)
    assert thresholds["full_period"]["sharpe_idx"] == pytest.approx(0.7456)
    assert thresholds["tuning_period"]["days"] == expected["days"]
    assert thresholds["tuning_period"]["sharpe_idx"] == pytest.approx(expected["sharpe_idx"])
    assert thresholds["hard_filter"]["min_sharpe_threshold"] == pytest.approx(expected["sharpe_idx"] - 0.3)
    assert thresholds["hard_filter"]["max_dd_threshold"] == pytest.approx(expected["dd_li"] * 1.3)
    assert "tuning_min_sharpe" not in thresholds
    assert "hard_filter_by_segment" not in thresholds


def test_require_tuning_period_baseline() -> None:
    thresholds = load_baseline_thresholds(FIXTURES / "baseline_thresholds.json")
    assert require_tuning_period_baseline(thresholds)["sharpe_idx"] == 1.2
    with pytest.raises(ValueError, match="tuning_period"):
        require_tuning_period_baseline({"hard_filter": {}})


def _write_daily_pnl(path: Path) -> dict[str, float | int]:
    dates = pd.bdate_range("2020-01-02", "2024-06-28")
    idx = np.arange(len(dates), dtype=float)
    li_ret = np.where((idx % 5) == 0, 0.0020, -0.00035)
    ret = li_ret + 0.0002
    date_int = np.asarray([int(date.strftime("%Y%m%d")) for date in dates])
    df = pd.DataFrame(
        {
            "date": date_int,
            "pnl": ret * 10_000_000.0,
            "ret": ret,
            "li_ret": li_ret,
        }
    )
    df.to_csv(path, index=False)
    score = df[(df["date"] >= 20210104) & (df["date"] <= 20231229)].copy()
    x = score["li_ret"].to_numpy(dtype=float)
    sharpe_idx = float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(252))
    cum = score["li_ret"].cumsum()
    equity = 1 + cum
    dd_li = float(((equity.cummax() - equity) / equity.cummax()).max())
    return {"days": int(len(score)), "sharpe_idx": sharpe_idx, "dd_li": dd_li}
