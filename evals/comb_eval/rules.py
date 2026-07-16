from __future__ import annotations

import numpy as np
import pandas as pd

from .ic import ic_quality_stats
from .pnl import pnl_quality_stats
from .schemas import MetricResult

RuleSet = dict[str, tuple[str, float]]

PNL_L1: RuleSet = {
    "ir": (">=", 0.35),
    "margin": (">=", 8.0),
    "min_ir_year": (">=", 0.06),
    "min_ret_year": (">=", 5.0),
    "tvr_pct": ("<=", 70.0),
    "lnum_ratio": ("<=", 0.55),
    "long_zz500_ret": (">=", 0.08),
}

PNL_L2: RuleSet = {
    "ir": (">=", 0.25),
    "margin": (">=", 5.0),
    "min_ir_year": (">=", 0.02),
    "min_ret_year": (">=", 2.0),
    "tvr_pct": ("<=", 80.0),
    "lnum_ratio": ("<=", 0.60),
    "long_zz500_ret": (">=", 0.06),
}

IC_L1: RuleSet = {
    "1d_IC.avg": (">=", 0.012),
    "1d_IC.min_each_year": (">=", 0.003),
    "1d_IC.ir": (">=", 0.40),
    "5d_IC.avg": (">=", 0.016),
    "5d_IC.ir": (">=", 0.47),
    "10d_IC.avg": (">=", 0.018),
    "10d_IC.ir": (">=", 0.50),
    "barra1d_IC.avg": (">=", 0.009),
    "barra1d_IC.ir": (">=", 0.32),
    "barra5d_IC.avg": (">=", 0.013),
    "barra5d_IC.ir": (">=", 0.37),
    "barra10d_IC.avg": (">=", 0.015),
    "barra10d_IC.ir": (">=", 0.42),
    "rankic.avg": (">=", 0.011),
    "lIC.avg": (">=", 0.009),
}

IC_L2: RuleSet = {
    "1d_IC.avg": (">=", 0.010),
    "1d_IC.min_each_year": (">=", 0.002),
    "1d_IC.ir": (">=", 0.25),
    "5d_IC.avg": (">=", 0.013),
    "5d_IC.ir": (">=", 0.33),
    "10d_IC.avg": (">=", 0.015),
    "10d_IC.ir": (">=", 0.37),
    "barra1d_IC.avg": (">=", 0.007),
    "barra1d_IC.ir": (">=", 0.20),
    "barra5d_IC.avg": (">=", 0.011),
    "barra5d_IC.ir": (">=", 0.25),
    "barra10d_IC.avg": (">=", 0.013),
    "barra10d_IC.ir": (">=", 0.30),
    "rankic.avg": (">=", 0.009),
    "lIC.avg": (">=", 0.007),
}


def evaluate_result(result: MetricResult) -> pd.DataFrame:
    if result.module == "pnl":
        return evaluate_values(pnl_quality_stats(result.table), PNL_L1, PNL_L2)
    if result.module == "ic":
        return evaluate_values(ic_quality_stats(result.table), IC_L1, IC_L2)
    raise ValueError(f"unsupported module: {result.module}")


def evaluate_values(values: dict[str, float], l1: RuleSet, l2: RuleSet) -> pd.DataFrame:
    rows = []
    keys = sorted((set(l1) | set(l2)) & set(values))
    for key in keys:
        actual = float(values[key]) if key in values and pd.notna(values[key]) else np.nan
        rows.append({
            "metric": key,
            "actual": actual,
            "L1": _check(actual, *l1[key]) if key in l1 else None,
            "L1_threshold": _format_rule(*l1[key]) if key in l1 else None,
            "L2": _check(actual, *l2[key]) if key in l2 else None,
            "L2_threshold": _format_rule(*l2[key]) if key in l2 else None,
        })
    return pd.DataFrame(rows).set_index("metric")


def _check(actual: float, cmp: str, threshold: float) -> bool:
    if pd.isna(actual):
        return False
    if cmp == ">=":
        return actual >= threshold
    if cmp == "<=":
        return actual <= threshold
    raise ValueError(f"unsupported comparison: {cmp}")


def _format_rule(cmp: str, threshold: float) -> str:
    return f"{cmp} {threshold:g}"
