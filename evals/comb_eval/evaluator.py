from __future__ import annotations

from pathlib import Path

import pandas as pd

from .ic import summarize_ic
from .pnl import summarize_pnl_with_benchmark
from .rules import evaluate_result


def run_evaluation(pnl_path: str | Path | None = None, ic_path: str | Path | None = None, pnlzz500_path: str | Path | None = None, start: str | None = None, end: str | None = None) -> dict[str, dict[str, pd.DataFrame]]:
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    if pnl_path:
        pnl_result = summarize_pnl_with_benchmark(pnl_path, benchmark_path=pnlzz500_path, start=start, end=end)
        outputs["pnl"] = {"summary": pnl_result.table, "checks": evaluate_result(pnl_result)}
    if ic_path:
        ic_result = summarize_ic(ic_path, start=start, end=end, normalize_names=True)
        outputs["ic"] = {"summary": ic_result.table, "checks": evaluate_result(ic_result)}
    return outputs


__all__ = ["evaluate_result", "run_evaluation"]
