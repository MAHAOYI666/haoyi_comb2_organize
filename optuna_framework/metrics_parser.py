"""Parse comb2-pcmaster ``pnl_summary.csv`` and ``daily_pnl.csv`` outputs."""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

SIMBASE_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "comb2-simbase"
if str(SIMBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBASE_ROOT))

from comb2_simbase.benchmark import benchmark_returns_from_cache, cache_path_from_rendered_config


@dataclass(frozen=True)
class WindowMetrics:
    """Metrics extracted for one run or scoring window."""

    sharpe_idx: float
    dd_li: float
    li_ret: float
    ret: float
    pnl: float
    days: int
    row_label: str
    all_rows: list[str]
    run_start_ds: int | None = None
    run_end_ds: int | None = None
    score_start_ds: int | None = None
    score_end_ds: int | None = None

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
            "run_start_ds": self.run_start_ds,
            "run_end_ds": self.run_end_ds,
            "score_start_ds": self.score_start_ds,
            "score_end_ds": self.score_end_ds,
        }


def parse_window_metrics(
    pnl_summary_path: str | Path,
    run_start_ds: int,
    run_end_ds: int,
    score_start_ds: int | None = None,
    score_end_ds: int | None = None,
    daily_metrics_path: str | Path | None = None,
) -> WindowMetrics:
    """Parse metrics for a run window or compute a separate scoring window from daily PnL."""

    run_start_ds = int(run_start_ds)
    run_end_ds = int(run_end_ds)
    score_start_ds = run_start_ds if score_start_ds is None else int(score_start_ds)
    score_end_ds = run_end_ds if score_end_ds is None else int(score_end_ds)
    df = _read_summary(pnl_summary_path)
    all_rows = [str(idx) for idx in df.index]

    if (score_start_ds, score_end_ds) != (run_start_ds, run_end_ds):
        metrics = _metrics_from_daily_output(
            pnl_summary_path=Path(pnl_summary_path),
            summary_df=df,
            all_rows=all_rows,
            run_start_ds=run_start_ds,
            run_end_ds=run_end_ds,
            score_start_ds=score_start_ds,
            score_end_ds=score_end_ds,
            daily_metrics_path=daily_metrics_path,
        )
        _warn_if_days_suspicious(metrics.days, score_start_ds, score_end_ds)
        return metrics

    exact_label = f"{score_start_ds}-{score_end_ds}"
    if exact_label in df.index.astype(str):
        row_label = exact_label
        row = df.loc[df.index.astype(str) == exact_label].iloc[0]
    else:
        if "days" not in df.columns:
            raise ValueError(f"{pnl_summary_path} is missing required column: days")
        row_label = str(pd.to_numeric(df["days"], errors="coerce").idxmax())
        row = df.loc[row_label]
    metrics = _row_to_metrics(
        row,
        row_label=row_label,
        all_rows=all_rows,
        run_start_ds=run_start_ds,
        run_end_ds=run_end_ds,
        score_start_ds=score_start_ds,
        score_end_ds=score_end_ds,
    )
    _warn_if_days_suspicious(metrics.days, score_start_ds, score_end_ds)
    return metrics


def parse_full_period(pnl_summary_path: str | Path) -> dict[str, WindowMetrics]:
    """Split a full-period summary into yearly rows plus the global row."""

    df = _read_summary(pnl_summary_path)
    all_rows = [str(idx) for idx in df.index]
    result: dict[str, WindowMetrics] = {}
    date_rows: list[tuple[str, int, int]] = []
    for raw_label in all_rows:
        match = re.fullmatch(r"(\d{8})-(\d{8})", raw_label)
        if match:
            date_rows.append((raw_label, int(match.group(1)), int(match.group(2))))

    if not date_rows:
        raise ValueError(f"{pnl_summary_path} does not contain date-range rows")

    full_label, _, _ = max(date_rows, key=lambda item: int(df.loc[item[0], "days"]))
    for label, start_ds, end_ds in date_rows:
        row = df.loc[label]
        key = "full" if label == full_label or str(start_ds)[:4] != str(end_ds)[:4] else str(start_ds)[:4]
        result[key] = _row_to_metrics(
            row,
            row_label=label,
            all_rows=all_rows,
            run_start_ds=start_ds,
            run_end_ds=end_ds,
            score_start_ds=start_ds,
            score_end_ds=end_ds,
        )

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


def _row_to_metrics(
    row: pd.Series,
    row_label: str,
    all_rows: list[str],
    run_start_ds: int | None = None,
    run_end_ds: int | None = None,
    score_start_ds: int | None = None,
    score_end_ds: int | None = None,
) -> WindowMetrics:
    required = ("sharpe_idx", "dd_li", "li_ret", "ret", "pnl", "days")
    missing = [column for column in required if column not in row.index]
    if missing:
        raise ValueError(f"pnl_summary row {row_label} is missing columns: {missing}")
    values = {column: pd.to_numeric(row[column], errors="coerce") for column in required}
    for column in ("sharpe_idx", "dd_li", "li_ret"):
        if pd.isna(values[column]):
            raise ValueError(f"pnl_summary row {row_label} has NaN {column}")
    return WindowMetrics(
        sharpe_idx=float(values["sharpe_idx"]),
        dd_li=float(values["dd_li"]),
        li_ret=float(values["li_ret"]),
        ret=float(values["ret"]) if not pd.isna(values["ret"]) else float("nan"),
        pnl=float(values["pnl"]) if not pd.isna(values["pnl"]) else float("nan"),
        days=int(values["days"]),
        row_label=str(row_label),
        all_rows=all_rows,
        run_start_ds=run_start_ds,
        run_end_ds=run_end_ds,
        score_start_ds=score_start_ds,
        score_end_ds=score_end_ds,
    )


def _metrics_from_daily_output(
    pnl_summary_path: Path,
    summary_df: pd.DataFrame,
    all_rows: list[str],
    run_start_ds: int,
    run_end_ds: int,
    score_start_ds: int,
    score_end_ds: int,
    daily_metrics_path: str | Path | None,
) -> WindowMetrics:
    """Compute scoring-window metrics from daily output."""

    daily_path = Path(daily_metrics_path) if daily_metrics_path is not None else pnl_summary_path.with_name("daily_pnl.csv")
    daily = _read_daily_metrics(daily_path)
    daily = daily[(daily["date"] >= int(run_start_ds)) & (daily["date"] <= int(run_end_ds))].copy()
    if daily.empty:
        raise ValueError(f"daily metrics has no rows in run window {run_start_ds}-{run_end_ds}: {daily_path}")

    daily_ret = _daily_return_series(daily, summary_df, daily_path, pnl_summary_path)
    daily_li_ret = _daily_li_return_series(daily, daily_ret, daily_path)
    score_mask = (daily["date"] >= int(score_start_ds)) & (daily["date"] <= int(score_end_ds))
    score_daily = daily.loc[score_mask].copy()
    if score_daily.empty:
        raise ValueError(f"daily metrics has no rows in scoring window {score_start_ds}-{score_end_ds}: {daily_path}")

    ret = daily_ret.loc[score_daily.index]
    li_ret = daily_li_ret.loc[score_daily.index]
    sharpe_idx = _sharpe(li_ret)
    if np.isnan(sharpe_idx):
        raise ValueError(f"computed scoring window {score_start_ds}-{score_end_ds} has NaN sharpe_idx")

    return WindowMetrics(
        sharpe_idx=float(sharpe_idx),
        dd_li=_max_drawdown(li_ret.cumsum()),
        li_ret=float(li_ret.sum()),
        ret=float(ret.sum()),
        pnl=float(pd.to_numeric(score_daily["pnl"], errors="coerce").sum()) if "pnl" in score_daily else float("nan"),
        days=int(len(score_daily)),
        row_label=f"{score_start_ds}-{score_end_ds}",
        all_rows=all_rows,
        run_start_ds=int(run_start_ds),
        run_end_ds=int(run_end_ds),
        score_start_ds=int(score_start_ds),
        score_end_ds=int(score_end_ds),
    )


def _read_daily_metrics(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"daily metrics file not found for scoring-window metrics: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"daily metrics file is empty: {path}")
    date_col = _find_column(df, ("date", "trade_date", "ds"))
    if date_col is None:
        raise ValueError(f"daily metrics file is missing a date column: {path}")
    df = df.copy()
    df["date"] = pd.to_numeric(df[date_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def _find_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _daily_return_series(
    daily: pd.DataFrame,
    summary_df: pd.DataFrame,
    daily_path: Path,
    pnl_summary_path: Path,
) -> pd.Series:
    ret_col = _find_column(daily, ("ret", "daily_ret"))
    if ret_col is not None:
        return pd.to_numeric(daily[ret_col], errors="coerce")
    if "pnl" not in daily.columns:
        raise ValueError(f"daily metrics needs either ret or pnl to compute scoring-window metrics: {daily_path}")
    booksize = _read_booksize_from_rendered_config(pnl_summary_path)
    if booksize is None:
        booksize = _infer_booksize(summary_df)
    pnl = pd.to_numeric(daily["pnl"], errors="coerce")
    return pnl / booksize


def _read_booksize_from_rendered_config(pnl_summary_path: Path) -> float | None:
    try:
        config_path = pnl_summary_path.parents[2] / "config.xml"
    except IndexError:
        return None
    if not config_path.exists():
        return None
    try:
        root = ET.parse(config_path).getroot()
    except Exception:
        return None
    backtest = root.find("./backtest")
    if backtest is None:
        return None
    raw_cash = backtest.get("cash")
    if raw_cash is None:
        return None
    cash = float(raw_cash)
    if not np.isfinite(cash) or cash == 0:
        return None
    return abs(cash)


def _daily_li_return_series(daily: pd.DataFrame, daily_ret: pd.Series, daily_path: Path) -> pd.Series:
    li_col = _find_column(daily, ("li_ret", "excess_ret", "idx_ret_excess"))
    if li_col is not None:
        return pd.to_numeric(daily[li_col], errors="coerce")

    benchmark_col = _find_column(daily, ("benchmark_ret", "bench_ret", "index_ret", "idx_ret"))
    if benchmark_col is not None:
        benchmark_ret = pd.to_numeric(daily[benchmark_col], errors="coerce")
    else:
        benchmark_ret = _fetch_benchmark_returns(daily["date"].tolist(), daily_path)
    return daily_ret - benchmark_ret


def _infer_booksize(summary_df: pd.DataFrame) -> float:
    for _, row in summary_df.iterrows():
        pnl = pd.to_numeric(row.get("pnl"), errors="coerce")
        ret = pd.to_numeric(row.get("ret"), errors="coerce")
        if pd.notna(pnl) and pd.notna(ret) and abs(float(ret)) > 1e-12:
            booksize = float(pnl) / float(ret)
            if np.isfinite(booksize) and abs(booksize) > 0:
                return abs(booksize)
    raise ValueError("unable to infer booksize from pnl_summary.csv; daily metrics must include ret")


def _fetch_benchmark_returns(dates: list[int], daily_path: Path) -> pd.Series:
    cache_path = _cache_path_for_daily_metrics(daily_path)
    if cache_path is None:
        raise ValueError(
            "daily metrics lacks li_ret/benchmark_ret columns and no rendered config with constants.cache_path "
            f"was found; cannot compute scoring-window sharpe_idx from {daily_path}"
        )
    return benchmark_returns_from_cache(cache_path, dates, ts_code="000905.SH")


def _cache_path_for_daily_metrics(daily_path: Path) -> Path | None:
    candidates: list[Path] = []
    for parent in daily_path.parents:
        candidates.append(parent / "config.xml")
    for config_path in candidates:
        cache_path = cache_path_from_rendered_config(config_path)
        if cache_path is not None:
            return cache_path
    return None


def _max_drawdown(cum_ret: pd.Series) -> float:
    equity = 1 + cum_ret
    roll_max = equity.cummax()
    dd = (roll_max - equity) / roll_max
    return float(dd.max()) if len(dd) else float("nan")


def _sharpe(daily_ret: pd.Series) -> float:
    x = pd.to_numeric(daily_ret, errors="coerce").to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan")
    std = np.std(x, ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(x) / std * np.sqrt(252))


def _warn_if_days_suspicious(days: int, start_ds: int, end_ds: int) -> None:
    expected = _expected_trading_days(start_ds, end_ds)
    if expected is not None:
        if abs(days - expected) > 5:
            warnings.warn(
                f"pnl_summary days={days} differs from expected trading days={expected} for {start_ds}-{end_ds}",
                RuntimeWarning,
                stacklevel=2,
            )
        return

    start_year = int(str(start_ds)[:4])
    end_year = int(str(end_ds)[:4])
    span_years = end_year - start_year
    if span_years >= 4:
        minimum = 1000
    elif span_years >= 2:
        minimum = 600
    elif start_year == end_year and str(end_ds)[4:6] <= "06":
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
        from comb2_simbase import IndexMask
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
            metrics = _row_to_metrics(row, row_label=str(label), all_rows=list(df.index.astype(str)))
        except Exception:
            continue
        if metrics.days > 0 and not np.isnan(metrics.sharpe_idx):
            return True
    return False
