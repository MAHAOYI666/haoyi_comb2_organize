"""Objective scoring, hard filters, and audit helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from optuna_framework.metrics_parser import WindowMetrics


HARD_FILTER_MISSING_MESSAGE = (
    "baseline_thresholds.json not found. 请先手动运行 run_baseline.py 生成 baseline_thresholds.json"
)

TUNING_PERIOD_MISSING_MESSAGE = (
    "baseline_thresholds.json does not contain tuning_period metrics. "
    "Please rerun run_baseline.py after the window refactor."
)

FACTOR_AUDIT_TEMPLATE = (
    "# Factor Audit\n\n"
    "因子时间合规性已由用户确认：2024/2025 命名仅为版本代号，不代表使用了未来信息。\n"
)


def running_score(sharpes: list[float]) -> float:
    """Compute an intermediate Optuna score from one or more sharpe values."""

    if not sharpes:
        raise ValueError("sharpes must not be empty")
    values = np.asarray(sharpes, dtype=float)
    if len(values) == 1:
        return float(values[0])
    return float(values.mean() - 0.3 * values.std())


def single_objective(metric: WindowMetrics, hard_filter: dict[str, float]) -> tuple[float, bool]:
    """Score the single Phase A scoring window and apply baseline hard filters."""

    if metric.sharpe_idx < float(hard_filter["min_sharpe_threshold"]):
        return -10.0, True
    if metric.dd_li > float(hard_filter["max_dd_threshold"]):
        return -10.0, True
    return float(metric.sharpe_idx), False


def final_objective(metrics: list[WindowMetrics], hard_filter: dict[str, float]) -> tuple[float, bool]:
    """Compute the final Phase A objective from its single scoring-window metric."""

    if len(metrics) != 1:
        raise ValueError("Phase A objective expects exactly one scoring-window metric")
    return single_objective(metrics[0], hard_filter)


def load_baseline_thresholds(path: str | Path) -> dict:
    """Load baseline thresholds, raising a clear action-oriented error if missing."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(HARD_FILTER_MISSING_MESSAGE + f": {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_tuning_period_baseline(thresholds: dict) -> dict:
    """Return tuning-period baseline metrics or raise a clear rerun error."""

    tuning_period = thresholds.get("tuning_period")
    if isinstance(tuning_period, dict):
        return tuning_period
    raise ValueError(TUNING_PERIOD_MISSING_MESSAGE)


def build_baseline_thresholds(
    full_by_year: dict[str, WindowMetrics],
    tuning_period: WindowMetrics,
) -> dict:
    """Build the JSON payload emitted after the single full-window baseline run."""

    required = {"full", "2020", "2021", "2022", "2023", "2024"}
    missing = sorted(required - set(full_by_year))
    if missing:
        raise ValueError(f"full-period baseline metrics are missing keys: {missing}")
    return {
        "tuning_period": tuning_period.to_dict(),
        "by_segment": {
            "holdout_2020": full_by_year["2020"].to_dict(),
            "holdout_2024h1": full_by_year["2024"].to_dict(),
        },
        "full_period": {
            **full_by_year["full"].to_dict(),
            "by_year": {
                key: full_by_year[key].to_dict()
                for key in ("2020", "2021", "2022", "2023", "2024")
            },
        },
        "hard_filter": {
            "min_sharpe_threshold": tuning_period.sharpe_idx - 0.3,
            "max_dd_threshold": tuning_period.dd_li * 1.3,
        },
    }


def ensure_factor_audit_template(study_root: Path) -> Path:
    """Create the factor audit template under ``study_root`` if it is absent."""

    path = Path(study_root) / "audit" / "factor_audit.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(FACTOR_AUDIT_TEMPLATE, encoding="utf-8")
    return path
