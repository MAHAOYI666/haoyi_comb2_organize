"""Metrics parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

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


def test_scoring_window_prefers_exact_summary_row(tmp_path) -> None:
    summary = tmp_path / "pnl_summary.csv"
    summary.write_text(
        ",pnl,ret,li_ret,dd,dd_li,sharpe,sharpe_idx,days\n"
        "20200102-20231229,999,0.9,0.8,0.7,0.6,0.5,0.4,900\n"
        "20210104-20231229,123,0.12,0.10,0.04,0.05,1.2,1.5,727\n",
        encoding="utf-8",
    )
    metrics = parse_segment_metrics(
        summary,
        20200102,
        20231229,
        role="tuning_period",
        score_start_ds=20210104,
        score_end_ds=20231229,
    )
    assert metrics.row_label == "20210104-20231229"
    assert metrics.sharpe_idx == 1.5
    assert metrics.run_start_ds == 20200102
    assert metrics.score_start_ds == 20210104


def test_scoring_window_computes_from_daily_when_exact_row_missing(tmp_path) -> None:
    summary = tmp_path / "pnl_summary.csv"
    summary.write_text(
        ",pnl,ret,li_ret,dd,dd_li,sharpe,sharpe_idx,days\n"
        "20200102-20201231,100,0.1,0.1,0.1,0.1,1,1,2\n"
        "20200102-20231229,999,0.9,0.8,0.7,0.6,0.5,9.9,5\n",
        encoding="utf-8",
    )
    (tmp_path / "daily_pnl.csv").write_text(
        "date,pnl,ret,li_ret\n"
        "20200102,10,0.10,0.10\n"
        "20201231,20,0.20,0.20\n"
        "20210104,100,0.01,0.01\n"
        "20210105,-200,-0.02,-0.02\n"
        "20210106,300,0.03,0.03\n",
        encoding="utf-8",
    )

    metrics = parse_segment_metrics(
        summary,
        20200102,
        20231229,
        role="tuning_period",
        score_start_ds=20210104,
        score_end_ds=20210106,
    )

    li_ret = np.asarray([0.01, -0.02, 0.03], dtype=float)
    expected_sharpe = float(li_ret.mean() / li_ret.std(ddof=1) * np.sqrt(252))
    expected_dd = float((1.01 - 0.99) / 1.01)
    assert metrics.row_label == "20210104-20210106"
    assert metrics.sharpe_idx == pytest.approx(expected_sharpe)
    assert metrics.dd_li == pytest.approx(expected_dd)
    assert metrics.li_ret == pytest.approx(0.02)
    assert metrics.pnl == pytest.approx(200.0)
    assert metrics.days == 3


def test_daily_scoring_uses_rendered_config_cash_when_ret_missing(tmp_path) -> None:
    segment_dir = tmp_path / "trial" / "tuning_2020_2023"
    backtest_dir = segment_dir / "output" / "backtest"
    backtest_dir.mkdir(parents=True)
    (segment_dir / "config.xml").write_text(
        '<config><backtest cash="1000" /></config>',
        encoding="utf-8",
    )
    summary = backtest_dir / "pnl_summary.csv"
    summary.write_text(
        ",pnl,ret,li_ret,dd,dd_li,sharpe,sharpe_idx,days\n"
        "20200102-20231229,999,0.9,0.8,0.7,0.6,0.5,9.9,5\n",
        encoding="utf-8",
    )
    (backtest_dir / "daily_pnl.csv").write_text(
        "date,pnl,benchmark_ret\n"
        "20200102,10,0.0\n"
        "20210104,10,0.0\n"
        "20210105,20,0.0\n"
        "20210106,-10,0.0\n",
        encoding="utf-8",
    )

    metrics = parse_segment_metrics(
        summary,
        20200102,
        20231229,
        role="tuning_period",
        score_start_ds=20210104,
        score_end_ds=20210106,
    )

    ret = np.asarray([0.01, 0.02, -0.01], dtype=float)
    expected_sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(252))
    assert metrics.ret == pytest.approx(0.02)
    assert metrics.sharpe_idx == pytest.approx(expected_sharpe)

