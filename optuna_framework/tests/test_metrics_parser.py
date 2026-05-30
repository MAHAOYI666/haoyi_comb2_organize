"""Metrics parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from optuna_framework.metrics_parser import parse_full_period, parse_segment_metrics


FIXTURES = Path(__file__).parent / "fixtures"


def test_exact_row_match() -> None:
    metrics = parse_segment_metrics(FIXTURES / "pnl_summary_sample.csv", 20210104, 20211231, role="tuning")
    assert metrics.row_label == "20210104-20211231"
    assert metrics.sharpe_idx == 1.1
    assert metrics.dd_li == 0.08
    assert metrics.days == 243
    assert "fallback-row" in metrics.all_rows


def test_days_max_fallback(tmp_path) -> None:
    path = tmp_path / "pnl_summary.csv"
    path.write_text(
        ",pnl,ret,li_ret,dd,dd_li,sharpe,sharpe_idx,days\n"
        "small,1,0.1,0.1,0.1,0.1,1,1,10\n"
        "large,2,0.2,0.2,0.1,0.2,2,2,20\n",
        encoding="utf-8",
    )
    with pytest.warns(RuntimeWarning):
        metrics = parse_segment_metrics(path, 20210104, 20211231)
    assert metrics.row_label == "large"
    assert metrics.sharpe_idx == 2.0


def test_nan_validation_raises(tmp_path) -> None:
    path = tmp_path / "pnl_summary.csv"
    path.write_text(
        ",pnl,ret,li_ret,dd,dd_li,sharpe,sharpe_idx,days\n"
        "bad,1,0.1,NAN,0.1,0.1,1,NAN,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="NaN sharpe_idx"):
        parse_segment_metrics(path, 20210104, 20211231)


def test_parse_full_period_roles() -> None:
    parsed = parse_full_period(FIXTURES / "pnl_summary_full_period_sample.csv")
    assert set(parsed) == {"2020", "2021", "2022", "2023", "2024", "full"}
    assert parsed["2024"].role == "holdout_2024h1"
    assert parsed["2021"].role == "tuning"
    assert parsed["full"].role == "full_period"
    assert parsed["full"].days == 1090

