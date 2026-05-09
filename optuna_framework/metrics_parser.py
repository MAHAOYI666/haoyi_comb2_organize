"""Parse comb2-pcmaster ``pnl_summary.csv`` outputs."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SegmentMetrics:
    """Metrics extracted from one pnl summary row."""

    sharpe_idx: float
    dd_li: float
    li_ret: float
    ret: float
    pnl: float
    days: int
    row_label: str
    all_rows: list[str]
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics for JSON/CSV outputs."""

        return {
            "sharpe_idx": self.sharpe_idx,
            "dd_li": self.dd_li,
            "li_ret": self.li_ret,
            "ret": self.ret,
            "pnl": self.pnl,
            "days": self.days,
            "row_label": self.row_label,
            "all_rows": self.all_rows,
            "role": self.role,
        }


def parse_segment_metrics(
    pnl_summary_path: str | Path,
    start_ds: int,
    end_ds: int,
    role: str | None = None,
) -> SegmentMetrics:
    """Parse one segment summary, preferring the exact date-range row."""

    df = _read_summary(pnl_summary_path)
    all_rows = [str(idx) for idx in df.index]
    exact_label = f"{int(start_ds)}-{int(end_ds)}"
    if exact_label in df.index.astype(str):
        row_label = exact_label
        row = df.loc[df.index.astype(str) == exact_label].iloc[0]
    else:
        if "days" not in df.columns:
            raise ValueError(f"{pnl_summary_path} is missing required column: days")
        row_idx = pd.to_numeric(df["days"], errors="coerce").idxmax()
        row_label = str(row_idx)
        row = df.loc[row_idx]
    metrics = _row_to_metrics(row, row_label=row_label, all_rows=all_rows, role=role)
    _warn_if_days_suspicious(metrics.days, int(start_ds), int(end_ds), role)
    return metrics


def parse_full_period(pnl_summary_path: str | Path) -> dict[str, SegmentMetrics]:
    """Split a full-period summary into yearly rows plus the global row."""

    df = _read_summary(pnl_summary_path)
    all_rows = [str(idx) for idx in df.index]
    result: dict[str, SegmentMetrics] = {}
    date_rows: list[tuple[str, int, int]] = []
    for raw_label in all_rows:
        match = re.fullmatch(r"(\d{8})-(\d{8})", raw_label)
        if match:
            date_rows.append((raw_label, int(match.group(1)), int(match.group(2))))

    if not date_rows:
        raise ValueError(f"{pnl_summary_path} does not contain date-range rows")

    full_label, full_start, full_end = max(date_rows, key=lambda item: int(df.loc[item[0], "days"]))
    for label, start_ds, end_ds in date_rows:
        row = df.loc[label]
        if label == full_label or str(start_ds)[:4] != str(end_ds)[:4]:
            result["full"] = _row_to_metrics(row, row_label=label, all_rows=all_rows, role="full_period")
            continue
        year = str(start_ds)[:4]
        if year == "2020":
            role = "holdout_2020"
        elif year in {"2021", "2022", "2023"}:
            role = "tuning"
        elif year == "2024":
            role = "holdout_2024h1"
        else:
            role = "yearly"
        result[year] = _row_to_metrics(row, row_label=label, all_rows=all_rows, role=role)

    if "full" not in result:
        row = df.loc[full_label]
        result["full"] = _row_to_metrics(row, row_label=full_label, all_rows=all_rows, role="full_period")
    return result


def _read_summary(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pnl_summary.csv not found: {path}")
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        raise ValueError(f"pnl_summary.csv is empty: {path}")
    df.index = df.index.astype(str)
    return df


def _row_to_metrics(row: pd.Series, row_label: str, all_rows: list[str], role: str | None) -> SegmentMetrics:
    required = ("sharpe_idx", "dd_li", "li_ret", "ret", "pnl", "days")
    missing = [column for column in required if column not in row.index]
    if missing:
        raise ValueError(f"pnl_summary row {row_label} is missing columns: {missing}")
    values = {column: pd.to_numeric(row[column], errors="coerce") for column in required}
    for column in ("sharpe_idx", "dd_li", "li_ret"):
        if pd.isna(values[column]):
            raise ValueError(f"pnl_summary row {row_label} has NaN {column}")
    return SegmentMetrics(
        sharpe_idx=float(values["sharpe_idx"]),
        dd_li=float(values["dd_li"]),
        li_ret=float(values["li_ret"]),
        ret=float(values["ret"]) if not pd.isna(values["ret"]) else float("nan"),
        pnl=float(values["pnl"]) if not pd.isna(values["pnl"]) else float("nan"),
        days=int(values["days"]),
        row_label=str(row_label),
        all_rows=all_rows,
        role=role,
    )


def _warn_if_days_suspicious(days: int, start_ds: int, end_ds: int, role: str | None) -> None:
    expected = _expected_trading_days(start_ds, end_ds)
    if expected is not None:
        if abs(days - expected) > 5:
            warnings.warn(
                f"pnl_summary days={days} differs from expected trading days={expected} for {start_ds}-{end_ds}",
                RuntimeWarning,
                stacklevel=2,
            )
        return

    if role == "full_period" or end_ds - start_ds > 10000:
        minimum = 1000
    elif str(start_ds)[:4] == str(end_ds)[:4] and str(end_ds)[4:6] <= "06":
        minimum = 80
    else:
        minimum = 200
    if days < minimum:
        warnings.warn(
            f"pnl_summary days={days} is below fallback minimum {minimum} for {start_ds}-{end_ds}",
            RuntimeWarning,
            stacklevel=2,
        )


def _expected_trading_days(start_ds: int, end_ds: int) -> int | None:
    try:
        from factorsim import IndexMask
    except Exception:
        return None
    try:
        dates = [int(date) for date in IndexMask().date]
    except Exception:
        return None
    return sum(1 for date in dates if int(start_ds) <= date <= int(end_ds))


def is_valid_summary(path: str | Path) -> bool:
    """Return whether a summary contains at least one parseable non-NaN row."""

    try:
        df = _read_summary(path)
    except Exception:
        return False
    for label, row in df.iterrows():
        try:
            metrics = _row_to_metrics(row, row_label=str(label), all_rows=list(df.index.astype(str)), role=None)
        except Exception:
            continue
        if metrics.days > 0 and not np.isnan(metrics.sharpe_idx):
            return True
    return False

