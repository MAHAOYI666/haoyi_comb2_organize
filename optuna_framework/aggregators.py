"""Objective scoring, hard filters, and baseline threshold helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optuna_framework.metrics_parser import SegmentMetrics


HARD_FILTER_MISSING_MESSAGE = (
    "baseline_thresholds.json not found. 请先人工运行 run_baseline.py 生成 baseline_thresholds.json"
)

def running_score(seg_sharpes: list[float]) -> float:
    """Compute the intermediate Optuna score after one or more segments."""

    if not seg_sharpes:
        raise ValueError("seg_sharpes must not be empty")
    values = np.asarray(seg_sharpes, dtype=float)
    if len(values) == 1:
        return float(values[0])
    return float(values.mean() - 0.3 * values.std())


def final_objective(metrics: list[SegmentMetrics], hard_filter: dict[str, float]) -> tuple[float, bool]:
    """Compute the final objective and whether the hard filter fired."""

    if not metrics:
        raise ValueError("metrics must not be empty")
    sharpes = np.asarray([item.sharpe_idx for item in metrics], dtype=float)
    dd_li = np.asarray([item.dd_li for item in metrics], dtype=float)
    if sharpes.min() < float(hard_filter["min_sharpe_threshold"]):
        return -10.0, True
    if dd_li.max() > float(hard_filter["max_dd_threshold"]):
        return -10.0, True
    return running_score(list(sharpes)), False


def load_baseline_thresholds(path: str | Path) -> dict:
    """Load baseline thresholds, raising a clear action-oriented error if missing."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(HARD_FILTER_MISSING_MESSAGE + f": {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_baseline_thresholds(
    by_segment: dict[str, SegmentMetrics],
    full_period: SegmentMetrics,
    full_by_year: dict[str, SegmentMetrics],
) -> dict:
    """Build the JSON payload emitted after manual baseline evaluation."""

    tuning = {name: metric for name, metric in by_segment.items() if name.startswith("seg")}
    tuning_sharpes = [metric.sharpe_idx for metric in tuning.values()]
    tuning_dd = [metric.dd_li for metric in tuning.values()]
    tuning_min_sharpe = min(tuning_sharpes) if tuning_sharpes else 0.0
    tuning_max_dd = max(tuning_dd) if tuning_dd else 0.0
    return {
        "by_segment": {name: metric.to_dict() for name, metric in by_segment.items()},
        "full_period": {
            **full_period.to_dict(),
            "by_year": {
                key: metric.to_dict()
                for key, metric in full_by_year.items()
                if key != "full"
            },
        },
        "tuning_min_sharpe": tuning_min_sharpe,
        "tuning_max_dd": tuning_max_dd,
        "hard_filter": {
            "min_sharpe_threshold": tuning_min_sharpe - 0.3,
            "max_dd_threshold": tuning_max_dd * 1.3,
        },
        "hard_filter_by_segment": {
            name: {
                "min_sharpe": metric.sharpe_idx - 0.3,
                "max_dd": metric.dd_li * 1.3,
            }
            for name, metric in tuning.items()
        },
    }

