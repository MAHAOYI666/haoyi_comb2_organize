"""Aggregator tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from optuna_framework.aggregators import ensure_factor_audit_template, final_objective, load_baseline_thresholds, running_score
from optuna_framework.metrics_parser import SegmentMetrics


FIXTURES = Path(__file__).parent / "fixtures"


def metric(sharpe: float, dd: float) -> SegmentMetrics:
    return SegmentMetrics(sharpe, dd, 0.1, 0.1, 1.0, 240, "row", ["row"])


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

