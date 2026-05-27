"""Phase B tuning-period stability tests."""

from __future__ import annotations

from optuna_framework.metrics_parser import WindowMetrics
from optuna_framework.scripts.run_phase_b import _phase_b_rejection_reasons


def metric(sharpe: float, dd: float) -> WindowMetrics:
    return WindowMetrics(sharpe, dd, 0.1, 0.1, 1.0, 240, "row", ["row"])


def test_phase_b_rejects_seed_below_continuous_baseline_thresholds() -> None:
    baseline = {"sharpe_idx": 1.0, "dd_li": 0.10}
    reasons = _phase_b_rejection_reasons(
        [
            (42, metric(0.84, 0.10)),
            (43, metric(1.00, 0.10)),
            (44, metric(1.00, 0.10)),
        ],
        baseline,
    )
    assert reasons == ["seed 42 scoring-window sharpe_idx below baseline-0.15"]


def test_phase_b_rejects_dd_and_seed_std() -> None:
    baseline = {"sharpe_idx": 1.0, "dd_li": 0.10}
    reasons = _phase_b_rejection_reasons(
        [
            (42, metric(1.35, 0.10)),
            (43, metric(0.95, 0.12)),
            (44, metric(1.05, 0.10)),
        ],
        baseline,
    )
    assert "seed 43 scoring-window dd_li above baseline*1.15" in reasons
    assert "three-seed scoring-window sharpe_idx std > 0.15" in reasons
