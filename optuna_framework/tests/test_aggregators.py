"""Aggregator tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

from optuna_framework.aggregators import (
    build_baseline_thresholds,
    ensure_factor_audit_template,
    final_objective,
    load_baseline_thresholds,
    require_tuning_period_baseline,
    running_score,
)
from optuna_framework.metrics_parser import SegmentMetrics


FIXTURES = Path(__file__).parent / "fixtures"


def metric(sharpe: float, dd: float) -> SegmentMetrics:
    return SegmentMetrics(sharpe, dd, 0.1, 0.1, 1.0, 240, "row", ["row"])


def metric_with_role(sharpe: float, dd: float, role: str) -> SegmentMetrics:
    return SegmentMetrics(sharpe, dd, 0.1, 0.1, 1.0, 240, "row", ["row"], role=role)


def test_running_score() -> None:
    assert running_score([1.2]) == 1.2
    expected = float(np.mean([1.0, 2.0, 3.0]) - 0.3 * np.std([1.0, 2.0, 3.0]))
    assert running_score([1.0, 2.0, 3.0]) == expected


def test_final_objective_hard_filter() -> None:
    thresholds = load_baseline_thresholds(FIXTURES / "baseline_thresholds.json")
    score, fired = final_objective([metric(1.0, 0.08), metric(1.1, 0.09)], thresholds["hard_filter"])
    assert fired is False
    assert score > 0
    score, fired = final_objective([metric(0.1, 0.08)], thresholds["hard_filter"])
    assert (score, fired) == (-10.0, True)
    score, fired = final_objective([metric(1.0, 0.5)], thresholds["hard_filter"])
    assert (score, fired) == (-10.0, True)


def test_missing_thresholds_message(tmp_path) -> None:
    try:
        load_baseline_thresholds(tmp_path / "missing.json")
    except FileNotFoundError as exc:
        assert "请先人工运行 run_baseline.py" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_factor_audit_template(tmp_path) -> None:
    path = ensure_factor_audit_template(tmp_path)
    assert path.exists()
    first = path.read_text(encoding="utf-8")
    ensure_factor_audit_template(tmp_path)
    assert path.read_text(encoding="utf-8") == first
    assert "2024/2025" in first


def test_build_thresholds_prefers_continuous_tuning_period_for_hard_filter() -> None:
    tuning = metric_with_role(1.4, 0.08, "tuning_period")
    thresholds = build_baseline_thresholds(
        {"tuning_2020_2023": tuning, "holdout_2020": metric_with_role(0.7, 0.1, "holdout")},
        metric_with_role(1.0, 0.12, "full_period"),
        {"full": metric_with_role(1.0, 0.12, "full_period")},
    )
    assert thresholds["tuning_period"]["sharpe_idx"] == 1.4
    assert thresholds["hard_filter"]["min_sharpe_threshold"] == pytest.approx(1.1)
    assert thresholds["hard_filter"]["max_dd_threshold"] == pytest.approx(0.104)
    assert require_tuning_period_baseline(thresholds)["dd_li"] == 0.08


def test_require_tuning_period_baseline_rejects_legacy_thresholds() -> None:
    legacy = load_baseline_thresholds(FIXTURES / "baseline_thresholds.json")
    with pytest.raises(ValueError, match="continuous tuning_period"):
        require_tuning_period_baseline(legacy)

