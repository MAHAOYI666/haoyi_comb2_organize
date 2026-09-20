from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ORGANIZE_ROOT = Path(__file__).resolve().parents[2]
for local_path in (
    ORGANIZE_ROOT,
    ORGANIZE_ROOT / "vendor" / "comb2-pcmaster",
    ORGANIZE_ROOT / "vendor" / "comb2-simbase",
):
    text_path = str(local_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)

from comb2_pcmaster import BacktestNode, DataLoader, DailyBacktest
from comb2_pcmaster.backtest import _adjust_alpha_by_long_ratio
from comb2_simbase import load_snap_vwap_labels
from comb2_simbase.benchmark import benchmark_returns_from_cache
from config import DEFAULT_CONFIG, SIMPLE_OPTIMIZER_CONFIG

from .exposure import compute_cap_corr


SNAP_TI = 93000
DEFAULT_WORKERS = 10
DEFAULT_LONG_RATIO = 0.5
TRADING_DAYS = 250
LABEL_PERIODS = (1, 2, 5, 10, 20)
SIGNAL_BLEND_PROFILE = "long_short_l1_v1"
ZZ500_TS_CODE = "000905.SH"
PNL_MODE = "excess_zz500"
VA_WEIGHTS = (0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 1.00)
SPEC_RET_RELATIVE_PATH = Path("BaseCache") / "BarraCNE5_RET" / "CNE5SpecRet.parquet"

# Backtests invoke the optimizer once per trading day.  Running several of
# those optimizers in threads makes them share the same Python/native runtime
# and can be substantially slower than running them in isolated processes.
_PROCESS_CONTEXT = None
_PROCESS_THREADPOOL_LIMIT = None
_WORKER_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class WeightResult:
    weight: float
    daily_returns: pd.Series


@dataclass(frozen=True)
class DailyEvaluationResult:
    myposition_path: Path
    target_path: Path
    cache_path: Path
    start_ds: int
    end_ds: int
    workers: int
    va_table: pd.DataFrame
    ic_table: pd.DataFrame
    long_ratio: float = DEFAULT_LONG_RATIO
    ti: int = SNAP_TI
    simple: bool = False


def resolve_daily_eval_dir(
    myposition_path: str | Path,
    target_path: str | Path,
    eval_dir: str | Path | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    long_ratio: float = DEFAULT_LONG_RATIO,
    ti: int = SNAP_TI,
    simple: bool = False,
) -> Path:
    if eval_dir is not None:
        return Path(eval_dir).expanduser().resolve()
    ti = _coerce_ti(ti)
    simple = bool(simple)
    myposition_path = Path(myposition_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    profile_tag = "simple" if simple else "old"
    key = "|".join(
        (
            str(myposition_path),
            str(target_path),
            profile_tag,
            f"{float(long_ratio):.8f}",
            f"{ti:06d}",
        )
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    ratio_tag = f"{float(long_ratio):.6f}"
    return myposition_path.parent / "comb2_eval" / f"{myposition_path.stem}__{target_path.stem}_{digest}_{profile_tag}_{SIGNAL_BLEND_PROFILE}_lr{ratio_tag}"


def _date_int(value) -> int:
    text = str(value).strip().replace("-", "").replace("/", "")
    return int(text[:8])


def _coerce_date_index(index: pd.Index) -> pd.Index:
    if isinstance(index, pd.DatetimeIndex):
        return pd.Index(index.strftime("%Y%m%d").astype(int), name="date")
    values = pd.Index(index).astype(str).str.replace("-", "", regex=False)
    values = values.str.replace("/", "", regex=False).str.slice(0, 8)
    parsed = pd.to_numeric(values, errors="coerce")
    if pd.isna(parsed).any():
        raise ValueError("daily alpha contains an invalid date index")
    return pd.Index(parsed.astype(np.int64), name="date")


def _time_values(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return (
            index.hour.to_numpy(dtype=np.int64) * 10000
            + index.minute.to_numpy(dtype=np.int64) * 100
            + index.second.to_numpy(dtype=np.int64)
        )
    values = pd.to_numeric(pd.Index(index).astype(str), errors="coerce")
    return values.to_numpy(dtype=np.float64)


def _coerce_ti(value: int | str) -> int:
    try:
        ti = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ti must be an HHMMSS integer, got {value!r}") from exc
    hours = ti // 10000
    minutes = ti // 100 % 100
    seconds = ti % 100
    if ti < 0 or hours >= 24 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"ti must be a valid HHMMSS time, got {value!r}")
    return ti


def _frame_times(frame: pd.DataFrame) -> tuple[int, ...]:
    if "date" in frame.columns and "time" in frame.columns:
        raw_index = pd.Index(frame["time"])
    elif isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels > 1:
        raw_index = frame.index.get_level_values(1)
    elif isinstance(frame.index, pd.DatetimeIndex):
        raw_index = frame.index
    else:
        return ()
    times = _time_values(raw_index)
    finite_times = times[np.isfinite(times)]
    if finite_times.size == 0 or np.all(finite_times == 0):
        return ()
    return tuple(sorted({int(value) for value in finite_times}))


def _resolve_daily_ti(
    frames: tuple[tuple[Path, pd.DataFrame], ...],
    requested_ti: int | None,
) -> int:
    selected_ti = _coerce_ti(requested_ti) if requested_ti is not None else None
    frame_times = [(path, set(_frame_times(frame))) for path, frame in frames]
    timed_sets = [times for _, times in frame_times if times]
    if selected_ti is None:
        available = sorted(set().union(*(times for times in timed_sets)))
        if len(available) > 1:
            raise ValueError(
                "daily inputs contain multiple intraday times; pass --ti explicitly"
            )
        selected_ti = available[0] if available else SNAP_TI
    for path, times in frame_times:
        if times and selected_ti not in times:
            raise ValueError(
                f"{path} does not contain the requested {selected_ti:06d} sample"
            )
    return selected_ti


def _drop_intraday_level(frame: pd.DataFrame, *, ti: int = SNAP_TI) -> pd.DataFrame:
    result = frame
    if isinstance(result.index, pd.MultiIndex):
        dates = result.index.get_level_values(0)
        if result.index.nlevels > 1:
            times = _time_values(result.index.get_level_values(1))
            finite_times = times[np.isfinite(times)]
            if ti in finite_times:
                result = result.loc[times == ti].copy()
                dates = result.index.get_level_values(0)
            elif len(np.unique(finite_times)) > 1:
                raise ValueError(
                    "daily alpha contains multiple intraday times; "
                    f"the {ti:06d} sample is required"
                )
        result.index = pd.Index(dates, name="date")
        return result

    if isinstance(result.index, pd.DatetimeIndex):
        times = _time_values(result.index)
        finite_times = times[np.isfinite(times)]
        if np.any(finite_times != 0):
            if ti in finite_times:
                result = result.loc[times == ti].copy()
            elif len(np.unique(finite_times)) > 1:
                raise ValueError(
                    "daily alpha contains multiple intraday times; "
                    f"the {ti:06d} sample is required"
                )
    return result


def _normalize_daily_frame(
    frame: pd.DataFrame, *, label: str, ti: int = SNAP_TI
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    ti = _coerce_ti(ti)
    result = frame.copy()
    if "date" in result.columns:
        if "time" in result.columns:
            times = _time_values(pd.Index(result["time"]))
            finite_times = times[np.isfinite(times)]
            if ti in finite_times:
                result = result.loc[times == ti].copy()
            elif len(np.unique(finite_times)) > 1:
                raise ValueError(
                    f"{label} contains multiple intraday times; "
                    f"the {ti:06d} sample is required"
                )
            result = result.set_index("date")
            result = result.drop(columns=["time"], errors="ignore")
        else:
            result = result.set_index("date")
    elif "Date" in result.columns:
        result = result.set_index("Date")

    result = _drop_intraday_level(result, ti=ti)
    result.index = _coerce_date_index(result.index)
    result.columns = pd.Index(
        [str(value).split(".", 1)[0].zfill(6) for value in result.columns],
        name="code",
    )
    if result.columns.duplicated().any():
        raise ValueError(f"{label} contains duplicate instrument columns")
    result = result.apply(pd.to_numeric, errors="coerce")
    if result.index.duplicated().any():
        raise ValueError(f"{label} contains duplicate daily dates")
    return result.sort_index()


def _read_position_frame(path: str | Path) -> tuple[Path, pd.DataFrame]:
    position_path = Path(path).expanduser().resolve()
    if not position_path.is_file():
        raise FileNotFoundError(f"position parquet not found: {position_path}")
    return position_path, pd.read_parquet(position_path)


def _filter_position_frame(
    frame: pd.DataFrame,
    position_path: Path,
    *,
    start: str | None,
    end: str | None,
    ti: int,
) -> pd.DataFrame:
    frame = _normalize_daily_frame(frame, label=str(position_path), ti=ti)
    if start is not None:
        frame = frame.loc[frame.index >= _date_int(start)]
    if end is not None:
        frame = frame.loc[frame.index <= _date_int(end)]
    if frame.empty:
        raise ValueError(f"position has no dates in the requested range: {position_path}")
    return frame


def load_position(
    path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    ti: int = SNAP_TI,
) -> pd.DataFrame:
    position_path, raw_frame = _read_position_frame(path)
    return _filter_position_frame(
        raw_frame,
        position_path,
        start=start,
        end=end,
        ti=_coerce_ti(ti),
    )


def _load_position_pair(
    myposition_path: str | Path,
    target_path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    ti: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    myposition_file, myposition_raw = _read_position_frame(myposition_path)
    target_file, target_raw = _read_position_frame(target_path)
    selected_ti = _resolve_daily_ti(
        ((myposition_file, myposition_raw), (target_file, target_raw)), ti
    )
    myposition = _filter_position_frame(
        myposition_raw,
        myposition_file,
        start=start,
        end=end,
        ti=selected_ti,
    )
    target = _filter_position_frame(
        target_raw,
        target_file,
        start=start,
        end=end,
        ti=selected_ti,
    )
    return myposition, target, selected_ti


def _benchmark_returns_for_index(cache_path: Path, index: pd.Index) -> pd.Series:
    """Return ZZ500 returns with the same first-day convention as backtest.li_ret."""
    dates = pd.DatetimeIndex(pd.to_datetime(index)).normalize()
    if dates.empty:
        return pd.Series(dtype=float, index=dates, name="zz500_return")
    date_ints = [int(date.strftime("%Y%m%d")) for date in dates]
    values = benchmark_returns_from_cache(
        cache_path,
        date_ints,
        ts_code=ZZ500_TS_CODE,
    )
    return pd.Series(
        np.asarray(values, dtype=float),
        index=dates,
        name="zz500_return",
    )


def _apply_excess_returns(
    results: dict[float, WeightResult],
    benchmark_returns: pd.Series,
) -> dict[float, WeightResult]:
    """Convert raw strategy returns to backtest's li_ret convention."""
    converted: dict[float, WeightResult] = {}
    for weight, result in results.items():
        returns = result.daily_returns.astype(float)
        benchmark = benchmark_returns.reindex(pd.DatetimeIndex(returns.index))
        if benchmark.isna().any():
            raise ValueError(f"ZZ500 benchmark could not be aligned for weight {weight:.2f}")
        excess = returns - benchmark
        excess.name = returns.name
        converted[weight] = WeightResult(weight, excess)
    return converted


def _write_daily_eval_artifacts(
    eval_dir: Path,
    results: dict[float, WeightResult],
    *,
    myposition_path: Path,
    target_path: Path,
    cache_path: Path,
    start_ds: int,
    end_ds: int,
    workers: int,
    long_ratio: float = DEFAULT_LONG_RATIO,
    ti: int = SNAP_TI,
    simple: bool = False,
    raw_results: dict[float, WeightResult] | None = None,
    benchmark_returns: pd.Series | None = None,
) -> None:
    ti = _coerce_ti(ti)
    eval_dir.mkdir(parents=True, exist_ok=True)
    weights = sorted(float(weight) for weight in results)
    for weight in weights:
        frame = results[weight].daily_returns.rename("return").rename_axis("date").reset_index()
        frame.to_csv(eval_dir / f"daily_returns_{weight:.2f}.csv", index=False, date_format="%Y-%m-%d")
        if raw_results is not None and benchmark_returns is not None:
            raw_path = eval_dir / "backtest" / f"weight_{weight:.2f}" / "daily_pnl.csv"
            if raw_path.is_file():
                raw_frame = pd.read_csv(raw_path)
                if "date" not in raw_frame.columns or "pnl" not in raw_frame.columns:
                    raise ValueError(f"invalid raw backtest PNL artifact: {raw_path}")
                raw_dates = pd.to_datetime(
                    raw_frame["date"].astype(str), format="%Y%m%d", errors="coerce"
                )
                if raw_dates.isna().any():
                    raise ValueError(f"invalid dates in raw backtest PNL artifact: {raw_path}")
                benchmark = benchmark_returns.reindex(pd.DatetimeIndex(raw_dates))
                if benchmark.isna().any():
                    raise ValueError(f"ZZ500 benchmark could not be aligned: {raw_path}")
                cash = float(DEFAULT_CONFIG["backtest"]["cash"])
                raw_frame["zz500_return"] = benchmark.to_numpy(dtype=float)
                raw_frame["zz500_pnl"] = raw_frame["zz500_return"] * cash
                raw_frame["excess_pnl"] = (
                    pd.to_numeric(raw_frame["pnl"], errors="raise") - raw_frame["zz500_pnl"]
                )
                raw_frame["excess_return"] = raw_frame["excess_pnl"] / cash
                raw_frame.to_csv(raw_path.with_name("daily_pnl_excess.csv"), index=False)
    manifest = {
        "version": 3,
        "signal_blend": SIGNAL_BLEND_PROFILE,
        "pnl_mode": PNL_MODE,
        "benchmark_ts_code": ZZ500_TS_CODE,
        "myposition_path": str(myposition_path),
        "target_path": str(target_path),
        "cache_path": str(cache_path),
        "start_ds": int(start_ds),
        "end_ds": int(end_ds),
        "workers": int(workers),
        "long_ratio": float(long_ratio),
        "ti": ti,
        "simple": bool(simple),
        "weights": weights,
    }
    temporary_manifest = eval_dir / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(eval_dir / "manifest.json")


def _read_daily_eval_artifacts(
    eval_dir: Path,
    *,
    myposition_path: Path,
    target_path: Path,
    ti: int = SNAP_TI,
    simple: bool = False,
) -> tuple[dict[float, WeightResult], dict]:
    manifest_path = eval_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"daily evaluation artifacts not found: {manifest_path}; run the evaluation first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["myposition_path"]).resolve() != myposition_path.resolve():
        raise ValueError("read artifacts belong to a different myposition parquet")
    if Path(manifest["target_path"]).resolve() != target_path.resolve():
        raise ValueError("read artifacts belong to a different target parquet")
    artifact_ti = _coerce_ti(manifest.get("ti", SNAP_TI))
    if artifact_ti != _coerce_ti(ti):
        raise ValueError(
            f"read artifacts use ti={artifact_ti:06d}, but requested ti={int(ti):06d}"
        )
    artifact_simple = bool(manifest.get("simple", False))
    if artifact_simple != bool(simple):
        requested_profile = "simple" if simple else "old"
        artifact_profile = "simple" if artifact_simple else "old"
        raise ValueError(
            f"read artifacts use config profile {artifact_profile}, "
            f"but requested {requested_profile}"
        )
    artifact_signal_blend = manifest.get("signal_blend")
    if artifact_signal_blend != SIGNAL_BLEND_PROFILE:
        raise ValueError(
            f"read artifacts use signal blend {artifact_signal_blend!r}, "
            f"but current runEval requires {SIGNAL_BLEND_PROFILE!r}; rerun the evaluation"
        )
    results: dict[float, WeightResult] = {}
    for raw_weight in manifest.get("weights", []):
        weight = float(raw_weight)
        result_path = eval_dir / f"daily_returns_{weight:.2f}.csv"
        if not result_path.is_file():
            raise FileNotFoundError(f"daily evaluation artifact is missing: {result_path}")
        frame = pd.read_csv(result_path)
        if "date" not in frame.columns or "return" not in frame.columns:
            raise ValueError(f"invalid daily evaluation artifact: {result_path}")
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"invalid dates in daily evaluation artifact: {result_path}")
        returns = pd.Series(
            pd.to_numeric(frame["return"], errors="coerce").to_numpy(dtype=float),
            index=pd.DatetimeIndex(dates),
            name=f"{weight:.2f}",
        )
        results[weight] = WeightResult(weight, returns)
    if 0.0 not in results or 1.0 not in results:
        raise ValueError("daily evaluation artifacts must include 0.00 and 1.00 weights")
    return results, manifest


def align_positions(myposition: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = myposition.index.intersection(target.index).sort_values()
    codes = myposition.columns.intersection(target.columns).sort_values()
    if dates.empty or codes.empty:
        raise ValueError("myposition and target have no overlapping dates/instruments")
    left = myposition.reindex(index=dates, columns=codes)
    right = target.reindex(index=dates, columns=codes)
    if not (np.isfinite(left.to_numpy(dtype=float)).any(axis=1) | np.isfinite(right.to_numpy(dtype=float)).any(axis=1)).all():
        raise ValueError("some overlapping dates have no finite alpha in either input")
    return left, right


def resolve_cache_path(cache_path: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if cache_path:
        candidates.append(Path(cache_path).expanduser())
    for env_name in ("COMB2_CACHE_PATH", "FACTORSIM_CACHE_PATH"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    configured = Path(DEFAULT_CONFIG["constants"]["cache_path"]).expanduser()
    candidates.append(configured if configured.is_absolute() else ORGANIZE_ROOT / configured)
    candidates.extend(
        [
            Path("/root/ml-data1-pvc/factorsim_data/Cache"),
            Path("/root/autodl/data/Cache"),
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.name == "AshareCache" and candidate.is_dir():
            candidate = candidate.parent
        if (candidate / "AshareCache").is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"AshareCache not found; searched: {searched}")


def _load_base_and_limit(cache_path: Path, start_ds: int, end_ds: int, columns: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(cache_path=str(cache_path))
    base = _normalize_daily_frame(
        loader.get_base(start_ds, end_ds), label="AshareCache.BaseUnivMask"
    ).reindex(columns=columns)
    limit = _normalize_daily_frame(
        loader.get_limit(start_ds, end_ds), label="AshareCache.LimitMask"
    ).reindex(columns=columns)
    dates = base.index.intersection(limit.index).sort_values()
    if dates.empty:
        raise ValueError("BaseUnivMask and LimitMask have no overlapping dates")
    return base.reindex(dates), limit.reindex(dates)


def adjust_position(frame: pd.DataFrame, base_mask: pd.DataFrame, long_ratio: float = 0.5) -> pd.DataFrame:
    output = np.full(frame.shape, np.nan, dtype=np.float64)
    base_mask = base_mask.reindex(index=frame.index, columns=frame.columns)
    for row_idx, date in enumerate(frame.index):
        output[row_idx] = _adjust_alpha_by_long_ratio(
            frame.iloc[row_idx],
            base_mask.loc[date].gt(0),
            long_ratio,
        ).to_numpy(dtype=np.float64)
    return pd.DataFrame(output, index=frame.index, columns=frame.columns)


def normalize_position_sides(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize positive and negative alpha mass independently for each day.

    Finite positive values are divided by their positive-side L1 sum, and
    finite negative values are divided by the absolute negative-side L1 sum.
    Missing values remain missing. If one side has no finite nonzero mass,
    that side is left at zero rather than being artificially filled.
    """
    values = frame.to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(values)
    values[~finite] = np.nan
    positive = finite & (values > 0.0)
    negative = finite & (values < 0.0)
    positive_sum = np.where(positive, values, 0.0).sum(axis=1)
    negative_sum = np.where(negative, -values, 0.0).sum(axis=1)
    normalized = values.copy()
    positive_where = positive & (positive_sum[:, None] > 0.0)
    negative_where = negative & (negative_sum[:, None] > 0.0)
    np.divide(
        values,
        positive_sum[:, None],
        out=normalized,
        where=positive_where,
    )
    np.divide(
        normalized,
        negative_sum[:, None],
        out=normalized,
        where=negative_where,
    )
    return pd.DataFrame(normalized, index=frame.index, columns=frame.columns)


def blend_positions(target: pd.DataFrame, myposition: pd.DataFrame, weight: float) -> pd.DataFrame:
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"VA weight must be in [0, 1], got {weight}")
    if weight == 0.0:
        return target.copy()
    if weight == 1.0:
        return myposition.copy()
    target_values = target.to_numpy(dtype=np.float64)
    myposition_values = myposition.to_numpy(dtype=np.float64)
    target_valid = np.isfinite(target_values)
    myposition_valid = np.isfinite(myposition_values)
    target_safe = np.where(target_valid, target_values, 0.0)
    myposition_safe = np.where(myposition_valid, myposition_values, 0.0)
    values = (1.0 - weight) * target_safe + weight * myposition_safe
    values[~(target_valid | myposition_valid)] = np.nan
    return pd.DataFrame(values, index=target.index, columns=target.columns)


def _strategy_config(
    start_ds: int, end_ds: int, *, simple: bool = False
) -> dict:
    strategy = deepcopy(DEFAULT_CONFIG["strategy"])
    strategy["start_ds"] = int(start_ds)
    strategy["end_ds"] = int(end_ds)
    optimizer = SIMPLE_OPTIMIZER_CONFIG if simple else DEFAULT_CONFIG["strategy"]["optimizer"]
    strategy["optimizer"] = deepcopy(optimizer)
    strategy["optimizer"]["type"] = "opt1"
    return strategy


def _build_backtest_node(
    cache_path: Path,
    output_path: Path,
    start_ds: int,
    end_ds: int,
    ti: int = SNAP_TI,
    *,
    simple: bool = False,
) -> BacktestNode:
    ti = _coerce_ti(ti)
    strategy = _strategy_config(start_ds, end_ds, simple=simple)
    backtest = deepcopy(DEFAULT_CONFIG["backtest"])
    output_path.mkdir(parents=True, exist_ok=True)
    return BacktestNode(
        start_ds=int(start_ds),
        end_ds=int(end_ds),
        output_path=str(output_path),
        strategy_path=str(strategy["path"]),
        strategy_class="AlphaStrategy",
        strategy_config=strategy,
        cash=float(backtest["cash"]),
        fee_rate=float(backtest["fee_rate"]),
        reserve_cash=float(backtest["reserve_cash"]),
        daily_metrics_file=backtest.get("daily_metrics_file", "daily_metrics.csv"),
        cache_path=str(cache_path),
        verbose=False,
        universe=backtest.get("universe", "base"),
        execution_price="vwap30",
        snap_ti=ti,
        drawdown_stop=float(backtest.get("drawdown_stop", 0.0)),
        cooldown_days=int(backtest.get("cooldown_days", 0)),
        draw_output=False,
    )


def _run_weight(
    weight: float,
    target: pd.DataFrame,
    myposition: pd.DataFrame,
    dates: tuple[int, ...],
    cache_path: Path,
    output_root: Path,
    ti: int,
    simple: bool,
) -> WeightResult:
    ti = _coerce_ti(ti)
    signal = blend_positions(target, myposition, weight)
    node = _build_backtest_node(
        cache_path,
        output_root / f"weight_{weight:.2f}",
        dates[0],
        dates[-1],
        ti,
        simple=simple,
    )
    backtest = DailyBacktest(node)
    available_dates = set(int(date) for date in backtest.vwap_data.index)
    missing = [date for date in dates if date not in available_dates]
    if missing:
        raise ValueError(
            f"{ti:06d} execution data is missing dates, first missing={missing[0]}"
        )
    for date in dates:
        backtest.step(
            date,
            signal.loc[date],
            ti=ti,
            last=True,
            alpha_already_adjusted=True,
        )
    backtest.finalize()
    assets = backtest.asset_history["total_asset"].astype(float)
    pnl = assets.diff()
    if not pnl.empty:
        pnl.iloc[0] = assets.iloc[0] - float(node.cash)
    returns = pnl / float(node.cash)
    returns.index = pd.to_datetime(returns.index.astype(str), format="%Y%m%d")
    returns.name = f"{weight:.2f}"
    return WeightResult(weight, returns)


def _configure_process_worker() -> None:
    """Keep native numerical libraries from oversubscribing each worker."""
    for variable in _WORKER_THREAD_ENV:
        os.environ[variable] = "1"

    # torch is imported by the optimizer/configuration path.  These setters
    # are process-local and complement the environment variables above.
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # The inter-op pool may already have been initialized by an import.
            pass
    except (ImportError, RuntimeError):
        pass

    # OpenBLAS/OpenMP libraries may already be loaded before the child
    # initializer runs, so apply a runtime limit as well when available.
    try:
        from threadpoolctl import threadpool_limits

        global _PROCESS_THREADPOOL_LIMIT
        _PROCESS_THREADPOOL_LIMIT = threadpool_limits(limits=1)
        _PROCESS_THREADPOOL_LIMIT.__enter__()
    except ImportError:
        pass


def _init_process_worker(
    target: pd.DataFrame,
    myposition: pd.DataFrame,
    dates: tuple[int, ...],
    cache_path: Path,
    output_root: Path,
    ti: int,
    simple: bool,
) -> None:
    global _PROCESS_CONTEXT
    _configure_process_worker()
    _PROCESS_CONTEXT = (target, myposition, dates, cache_path, output_root, ti, simple)


def _run_weight_in_process(weight: float) -> WeightResult:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("backtest process worker was not initialized")
    target, myposition, dates, cache_path, output_root, ti, simple = _PROCESS_CONTEXT
    return _run_weight(
        weight, target, myposition, dates, cache_path, output_root, ti, simple
    )


def run_weight_backtests(
    target: pd.DataFrame,
    myposition: pd.DataFrame,
    cache_path: Path,
    start_ds: int,
    end_ds: int,
    workers: int = DEFAULT_WORKERS,
    weights: tuple[float, ...] = VA_WEIGHTS,
    output_root: Path | None = None,
    ti: int = SNAP_TI,
    simple: bool = False,
) -> dict[float, WeightResult]:
    ti = _coerce_ti(ti)
    simple = bool(simple)
    # evaluate_daily applies long_ratio adjustment first. Normalize each
    # source by positive/negative L1 mass before any weight is blended so that
    # the VA weight has a stable signal-space meaning across the two inputs.
    target = normalize_position_sides(target)
    myposition = normalize_position_sides(myposition)
    dates = tuple(int(date) for date in target.index if start_ds <= int(date) <= end_ds)
    if not dates:
        raise ValueError("no common alpha dates are available for backtest")
    workers = max(1, min(int(workers), len(weights)))
    def run_parallel(root: Path) -> dict[float, WeightResult]:
        root.mkdir(parents=True, exist_ok=True)

        results: dict[float, WeightResult] = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_process_worker,
            initargs=(target, myposition, dates, cache_path, root, ti, simple),
        ) as executor:
            futures = {
                executor.submit(_run_weight_in_process, weight): weight
                for weight in weights
            }
            try:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results[result.weight] = result
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return results

    if output_root is not None:
        results = run_parallel(Path(output_root))
    else:
        with tempfile.TemporaryDirectory(prefix="comb2_daily_va_") as temp_dir:
            results = run_parallel(Path(temp_dir))
    return {weight: results[weight] for weight in weights}


def _annualized_percent(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if values.empty:
        return np.nan
    return float(values.mean() * TRADING_DAYS * 100.0)


def build_va_table(results: dict[float, WeightResult], weights: tuple[float, ...] = VA_WEIGHTS) -> pd.DataFrame:
    if 0.0 not in results or 1.0 not in results:
        raise ValueError("VA results must include 0.00 benchmark and 1.00 test weights")
    benchmark = results[0.0].daily_returns
    rows: list[dict] = []
    year_values = sorted(benchmark.index.year.unique())
    groups: list[tuple[str, pd.Series]] = [
        (str(year), benchmark.loc[benchmark.index.year == year]) for year in year_values
    ]
    groups.append(("full", benchmark))
    for period, base_group in groups:
        row: dict[str, float | str] = {"period": period}
        for weight in weights:
            current = results[weight].daily_returns.reindex(base_group.index)
            if weight in (0.0, 1.0):
                value = current
            else:
                value = current - base_group
            row[f"{weight:.2f}"] = _annualized_percent(value)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("period")
    if year_values and int((benchmark.index.year == year_values[0]).sum()) < 20:
        table = table.iloc[1:]
    return table


def _format_percent(value: float) -> str:
    return "nan%" if not np.isfinite(value) else f"{value * 100.0:.2f}%"


def _format_va_table(table: pd.DataFrame) -> str:
    return table.to_string(float_format=lambda value: f"{value:.2f}")


def _summarize_metric(series: pd.Series) -> dict[str, str]:
    values = series.dropna().astype(float)
    result: dict[str, str] = {}
    for year, group in values.groupby(values.index.year):
        result[str(year)] = _format_percent(float(group.mean()))
    result["all"] = _format_percent(float(values.mean())) if not values.empty else "nan"
    return result


def _row_correlation(signal: pd.DataFrame, label: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.Series:
    signal, label = signal.align(label, join="inner", axis=0)
    signal, label = signal.align(label, join="inner", axis=1)
    valid_mask = valid_mask.reindex(index=signal.index, columns=signal.columns).fillna(False)
    x = signal.to_numpy(dtype=np.float64)
    y = label.to_numpy(dtype=np.float64)
    valid = valid_mask.to_numpy(dtype=bool) & np.isfinite(y)
    x = np.where(np.isfinite(x), x, 0.0)
    result = np.full(len(signal), np.nan, dtype=np.float64)
    for row_idx in range(len(signal)):
        selected = valid[row_idx]
        if selected.sum() < 2:
            continue
        x_row = x[row_idx, selected]
        y_row = y[row_idx, selected]
        if np.std(x_row) > 0.0 and np.std(y_row) > 0.0:
            result[row_idx] = float(np.corrcoef(x_row, y_row)[0, 1])
    return pd.Series(result, index=pd.to_datetime(signal.index.astype(str), format="%Y%m%d"))


def _risk_correlation(signal: pd.DataFrame, risk: pd.DataFrame) -> pd.Series:
    signal, risk = signal.align(risk, join="inner", axis=0)
    signal, risk = signal.align(risk, join="inner", axis=1)
    x = signal.to_numpy(dtype=np.float64)
    ranked_risk = risk.rank(axis=1, method="average").to_numpy(dtype=np.float64)
    result = np.full(len(signal), np.nan, dtype=np.float64)
    for row_idx in range(len(signal)):
        risk_valid = np.isfinite(ranked_risk[row_idx])
        if risk_valid.sum() < 2:
            continue
        alpha = np.where(risk_valid, x[row_idx], np.nan)
        finite_alpha = np.isfinite(alpha)
        if not finite_alpha.any():
            continue
        centered = np.where(finite_alpha, alpha - np.nanmedian(alpha), 0.0)
        long_sum = centered[centered > 0.0].sum()
        short_sum = -centered[centered < 0.0].sum()
        if long_sum <= 0.0 or short_sum <= 0.0:
            continue
        weights = np.zeros_like(centered)
        positive = centered > 0.0
        negative = centered < 0.0
        weights[positive] = centered[positive] / long_sum
        weights[negative] = centered[negative] / short_sum
        risk_values = ranked_risk[row_idx, risk_valid]
        weight_values = weights[risk_valid]
        if np.std(risk_values) > 0.0 and np.std(weight_values) > 0.0:
            result[row_idx] = float(np.corrcoef(risk_values, weight_values)[0, 1])
    return pd.Series(result, index=pd.to_datetime(signal.index.astype(str), format="%Y%m%d"))


def build_ic_table(
    signal: pd.DataFrame,
    cache_path: Path,
    start_ds: int,
    end_ds: int,
    ti: int = SNAP_TI,
) -> pd.DataFrame:
    ti = _coerce_ti(ti)
    base, limit = _load_base_and_limit(cache_path, start_ds, end_ds, signal.columns)
    evaluation_mask = (base * limit).shift(-1).gt(0)
    labels = load_snap_vwap_labels(
        cache_path,
        ti,
        start_ds,
        end_ds,
        periods=LABEL_PERIODS,
    )
    rows: dict[str, dict[str, str]] = {}
    for period in LABEL_PERIODS:
        label = _normalize_daily_frame(
            labels[period], label=f"{period}d {ti:06d} label", ti=ti
        )
        ic = _row_correlation(signal, label, evaluation_mask)
        rows[f"IC_{period}d_Filter"] = _summarize_metric(ic)

    spec_path = cache_path / SPEC_RET_RELATIVE_PATH
    if not spec_path.is_file():
        raise FileNotFoundError(f"Barra specific-return parquet not found: {spec_path}")
    spec_ret = _normalize_daily_frame(pd.read_parquet(spec_path), label="CNE5SpecRet")
    barra_ic = _risk_correlation(signal, spec_ret)
    rows["barra IC"] = _summarize_metric(barra_ic)
    cap_corr = compute_cap_corr(signal, cache_path, start_ds=start_ds, end_ds=end_ds)
    rows["cap corr"] = _summarize_metric(cap_corr)

    columns = sorted({column for row in rows.values() for column in row if column != "all"})
    columns.append("all")
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns)


def evaluate_daily(
    myposition_path: str | Path,
    target_path: str | Path,
    *,
    cache_path: str | Path | None = None,
    eval_dir: str | Path | None = None,
    worker: int = DEFAULT_WORKERS,
    start: str | None = None,
    end: str | None = None,
    long_ratio: float = DEFAULT_LONG_RATIO,
    ti: int | None = None,
    simple: bool = False,
) -> DailyEvaluationResult:
    long_ratio = float(long_ratio)
    if not np.isfinite(long_ratio) or not 0.0 <= long_ratio <= 1.0:
        raise ValueError("long_ratio must be between 0 and 1")
    simple = bool(simple)
    myposition_path = Path(myposition_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    myposition, target, selected_ti = _load_position_pair(
        myposition_path,
        target_path,
        start=start,
        end=end,
        ti=ti,
    )
    myposition, target = align_positions(myposition, target)
    start_ds = max(int(myposition.index.min()), int(target.index.min()))
    end_ds = min(int(myposition.index.max()), int(target.index.max()))
    cache = resolve_cache_path(cache_path)

    base, _ = _load_base_and_limit(cache, start_ds, end_ds, myposition.columns)
    common_dates = myposition.index.intersection(base.index).sort_values()
    if common_dates.empty:
        raise ValueError("alpha and BaseUnivMask have no overlapping dates")
    myposition = myposition.reindex(common_dates)
    target = target.reindex(common_dates)
    base = base.reindex(common_dates, columns=myposition.columns)
    adjusted_myposition = adjust_position(myposition, base, long_ratio=long_ratio)
    adjusted_target = adjust_position(target, base, long_ratio=long_ratio)
    resolved_eval_dir = resolve_daily_eval_dir(
        myposition_path,
        target_path,
        eval_dir,
        start=start,
        end=end,
        long_ratio=long_ratio,
        ti=selected_ti,
        simple=simple,
    )
    raw_results = run_weight_backtests(
        adjusted_target,
        adjusted_myposition,
        cache,
        int(common_dates.min()),
        int(common_dates.max()),
        workers=worker,
        output_root=resolved_eval_dir / "backtest",
        ti=selected_ti,
        simple=simple,
    )
    benchmark_returns = _benchmark_returns_for_index(cache, raw_results[0.0].daily_returns.index)
    results = _apply_excess_returns(raw_results, benchmark_returns)
    va_table = build_va_table(results)
    ic_table = build_ic_table(
        myposition,
        cache,
        int(common_dates.min()),
        int(common_dates.max()),
        ti=selected_ti,
    )
    _write_daily_eval_artifacts(
        resolved_eval_dir,
        results,
        myposition_path=myposition_path,
        target_path=target_path,
        cache_path=cache,
        start_ds=int(common_dates.min()),
        end_ds=int(common_dates.max()),
        workers=max(1, min(int(worker), len(VA_WEIGHTS))),
        long_ratio=long_ratio,
        ti=selected_ti,
        simple=simple,
        raw_results=raw_results,
        benchmark_returns=benchmark_returns,
    )
    return DailyEvaluationResult(
        myposition_path=myposition_path,
        target_path=target_path,
        cache_path=cache,
        start_ds=int(common_dates.min()),
        end_ds=int(common_dates.max()),
        workers=max(1, min(int(worker), len(VA_WEIGHTS))),
        va_table=va_table,
        ic_table=ic_table,
        long_ratio=long_ratio,
        ti=selected_ti,
        simple=simple,
    )


def read_daily_evaluation(
    myposition_path: str | Path,
    target_path: str | Path,
    *,
    cache_path: str | Path | None = None,
    eval_dir: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    long_ratio: float = DEFAULT_LONG_RATIO,
    ti: int | None = None,
    simple: bool = False,
) -> DailyEvaluationResult:
    long_ratio = float(long_ratio)
    if not np.isfinite(long_ratio) or not 0.0 <= long_ratio <= 1.0:
        raise ValueError("long_ratio must be between 0 and 1")
    simple = bool(simple)
    myposition_path = Path(myposition_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    myposition, target, selected_ti = _load_position_pair(
        myposition_path,
        target_path,
        start=start,
        end=end,
        ti=ti,
    )
    resolved_eval_dir = resolve_daily_eval_dir(
        myposition_path,
        target_path,
        eval_dir,
        start=start,
        end=end,
        long_ratio=long_ratio,
        ti=selected_ti,
        simple=simple,
    )
    results, manifest = _read_daily_eval_artifacts(
        resolved_eval_dir,
        myposition_path=myposition_path,
        target_path=target_path,
        ti=selected_ti,
        simple=simple,
    )
    artifact_long_ratio = float(manifest.get("long_ratio", DEFAULT_LONG_RATIO))
    if not np.isclose(artifact_long_ratio, long_ratio):
        raise ValueError(
            f"read artifacts use long_ratio={artifact_long_ratio:g}, "
            f"but requested long_ratio={long_ratio:g}"
        )
    cache = resolve_cache_path(cache_path or manifest.get("cache_path"))
    if manifest.get("pnl_mode") != PNL_MODE:
        benchmark_returns = _benchmark_returns_for_index(
            cache, results[0.0].daily_returns.index
        )
        results = _apply_excess_returns(results, benchmark_returns)
    artifact_start = int(manifest["start_ds"])
    artifact_end = int(manifest["end_ds"])
    requested_start = max(artifact_start, _date_int(start)) if start is not None else artifact_start
    requested_end = min(artifact_end, _date_int(end)) if end is not None else artifact_end
    if requested_start > requested_end:
        raise ValueError("requested read range does not overlap the saved evaluation")
    myposition = myposition.loc[
        (myposition.index >= requested_start) & (myposition.index <= requested_end)
    ]
    target = target.loc[
        (target.index >= requested_start) & (target.index <= requested_end)
    ]
    myposition, target = align_positions(myposition, target)
    start_ds = max(int(myposition.index.min()), int(target.index.min()))
    end_ds = min(int(myposition.index.max()), int(target.index.max()))
    base, _ = _load_base_and_limit(cache, start_ds, end_ds, myposition.columns)
    common_dates = myposition.index.intersection(base.index).sort_values()
    if common_dates.empty:
        raise ValueError("alpha and BaseUnivMask have no overlapping dates")
    myposition = myposition.reindex(common_dates)
    start_ds = int(common_dates.min())
    end_ds = int(common_dates.max())
    for weight, result in list(results.items()):
        returns = result.daily_returns
        returns = returns[(returns.index >= pd.Timestamp(str(start_ds))) & (returns.index <= pd.Timestamp(str(end_ds)))]
        results[weight] = WeightResult(weight, returns)
    va_table = build_va_table(results)
    ic_table = build_ic_table(myposition, cache, start_ds, end_ds, ti=selected_ti)
    return DailyEvaluationResult(
        myposition_path=myposition_path,
        target_path=target_path,
        cache_path=cache,
        start_ds=start_ds,
        end_ds=end_ds,
        workers=int(manifest.get("workers", DEFAULT_WORKERS)),
        va_table=va_table,
        ic_table=ic_table,
        long_ratio=artifact_long_ratio,
        ti=selected_ti,
        simple=simple,
    )


def format_daily_evaluation(
    result: DailyEvaluationResult, *, include_header: bool = True
) -> str:
    lines = []
    profile = "simple" if result.simple else "old"
    if include_header:
        lines.extend(
            [
                f"[{time.strftime('%H:%M:%S', time.localtime())}]Start Evaluation",
                str(result.myposition_path),
                str(result.target_path),
            ]
        )
    lines.extend(
        [
            f"[VA] workers={result.workers} execution={result.ti:06d} config={profile} signal_blend={SIGNAL_BLEND_PROFILE}",
            _format_va_table(result.va_table),
            f"IC FROM {result.start_ds} to {result.end_ds}",
            result.ic_table.to_string(),
        ]
    )
    return "\n".join(lines)
