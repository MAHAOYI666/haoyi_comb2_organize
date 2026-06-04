from __future__ import annotations

import json
from pathlib import Path

from optuna_framework.metrics_parser import WindowMetrics
from optuna_framework.scripts.run_phase_b import _phase_b_rejection_reasons
from optuna_framework.scripts.run_phase_c import _accept_candidate


FIXTURES = Path(__file__).parent / "fixtures"


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


def test_phase_c_rejects_bad_holdout() -> None:
    thresholds = json.loads((FIXTURES / "baseline_thresholds.json").read_text(encoding="utf-8"))
    full = thresholds["full_period"]
    metrics = {
        "full": {"sharpe_idx": full["sharpe_idx"], "dd_li": full["dd_li"]},
        "2020": {"sharpe_idx": thresholds["by_segment"]["holdout_2020"]["sharpe_idx"] - 0.5, "dd_li": 0.1},
        "2021": {"sharpe_idx": full["by_year"]["2021"]["sharpe_idx"] + 0.1, "dd_li": 0.1},
        "2022": {"sharpe_idx": full["by_year"]["2022"]["sharpe_idx"] + 0.1, "dd_li": 0.1},
        "2023": {"sharpe_idx": full["by_year"]["2023"]["sharpe_idx"] + 0.1, "dd_li": 0.1},
        "2024": {"sharpe_idx": thresholds["by_segment"]["holdout_2024h1"]["sharpe_idx"], "dd_li": 0.1},
    }
    seed_results = [{"seed": seed, "metrics": metrics} for seed in (42, 43, 44)]

    accepted, reasons = _accept_candidate(seed_results, thresholds)

    assert accepted is False
    assert "seed 42 holdout_2020 sharpe below threshold" in reasons
