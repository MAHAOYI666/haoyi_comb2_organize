from __future__ import annotations

import importlib
import json
import math
import warnings
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from comb2_simbase.cache_layout import (
    BASE_UNIVERSE_MASK_NAME,
    FILTERED_MASK_NAME,
    daily_label_path,
    stock_mask_path,
)
from comb2_simbase.snap_labels import load_snap_vwap_labels, snap_vwap_price_name

warnings.filterwarnings("ignore", category=pd.errors.ChainedAssignmentError)

from .exposure import compute_barra_style_exposure, compute_cap_corr, summarize_cap_corr
from .formatting import output_frame_to_text
from .ic import summarize_ic
from .io import normalize_date_index, read_cache_array, read_matrix
from .pnl import summarize_pnl_with_benchmark
from .rules import evaluate_result

DEFAULT_LABEL_DF_TYPE = True
PNL_KEY_COLUMNS = [
    "pnl_m",
    "ret_pct",
    "tvr_pct",
    "ir",
    "sharpe",
    "dd_pct",
    "longonly_pnl_m",
    "longonly_ret_pct",
    "longonly_tvr_pct",
    "longonly_ir",
    "longonly_sharpe",
    "margin",
    "fitness",
]
IC_KEY_COLUMNS = [
    "1d_IC.avg",
    "1d_IC.ir",
    "5d_IC.avg",
    "5d_IC.ir",
    "rankic.avg",
    "rankic.ir",
    "lIC.avg",
    "lIC.ir",
    "layerSpread.avg",
    "coverage.avg",
]


@dataclass(frozen=True)
class ConfigEvalArtifacts:
    config_path: Path
    output_root: Path
    report_dir: Path
    alpha_path: Path
    plot_path: Path
    label_path: Path | None
    label_5d_path: Path | None
    snap_ti: int | None
    label_is_table: bool
    label_5d_is_table: bool
    label_df_type: object
    booksize: float
    tradecost_ratio: float
    cache_path: Path


@dataclass
class ConfigEvalResult:
    artifacts: ConfigEvalArtifacts
    ic_summary: pd.DataFrame | None
    pnl_summary: pd.DataFrame | None
    ic_checks: pd.DataFrame | None
    pnl_checks: pd.DataFrame | None
    decile_summary: pd.DataFrame | None
    exposure_summary: pd.DataFrame | None
    cap_corr_summary: pd.DataFrame | None
    top10_excess: pd.DataFrame | None
    messages: list[str]


@dataclass(frozen=True)
class OutputFileStatus:
    name: str
    path: Path
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class ConfigOutputCheck:
    config_path: Path
    output_root: Path
    files: tuple[OutputFileStatus, ...]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.files)

    @property
    def missing(self) -> tuple[OutputFileStatus, ...]:
        return tuple(item for item in self.files if not item.ok)


def run_config_evaluation(
    config_path: str | Path,
    *,
    report_dir: str | Path | None = None,
    plot_path: str | Path | None = None,
    pnlzz500_path: str | Path | None = None,
    label_path: str | Path | None = None,
    label_5d_path: str | Path | None = None,
    label_is_table: bool = False,
    label_5d_is_table: bool = False,
    label_df_type: object = DEFAULT_LABEL_DF_TYPE,
    booksize: float | None = None,
    tradecost_ratio: float | None = None,
    start: str | None = None,
    end: str | None = None,
    skip_deciles: bool = False,
    skip_exposure: bool = False,
) -> ConfigEvalResult:
    config = _load_organize_config(config_path)
    eval_start = start or _strategy_date(config, "start_ds")
    eval_end = end or _strategy_date(config, "end_ds")
    artifacts = _resolve_artifacts(
        Path(config_path).expanduser().resolve(),
        config,
        report_dir=report_dir,
        plot_path=plot_path,
        label_path=label_path,
        label_5d_path=label_5d_path,
        label_is_table=label_is_table,
        label_5d_is_table=label_5d_is_table,
        label_df_type=label_df_type,
        booksize=booksize,
        tradecost_ratio=tradecost_ratio,
    )
    artifacts.report_dir.mkdir(parents=True, exist_ok=True)

    messages: list[str] = []
    alpha = _read_alpha(artifacts.alpha_path, start=eval_start, end=eval_end)
    if alpha.empty:
        raise ValueError(f"alpha has no rows after date filtering: {artifacts.alpha_path}")

    snap_labels = None
    if artifacts.snap_ti is not None and (artifacts.label_path is None or artifacts.label_5d_path is None):
        start_ds = int(normalize_date_index(alpha).index.min().strftime("%Y%m%d"))
        end_ds = int(normalize_date_index(alpha).index.max().strftime("%Y%m%d"))
        snap_labels = load_snap_vwap_labels(artifacts.cache_path, artifacts.snap_ti, start_ds, end_ds)
    label_1d = (
        snap_labels[1]
        if artifacts.label_path is None
        else _read_label_for_signal(alpha, artifacts, path=artifacts.label_path, is_table=artifacts.label_is_table)
    )
    label_5d = (
        snap_labels[5]
        if artifacts.label_5d_path is None
        else _read_label_for_signal(alpha, artifacts, path=artifacts.label_5d_path, is_table=artifacts.label_5d_is_table)
    )
    evaluation_mask = load_evaluation_mask(alpha, artifacts.cache_path)
    alpha, label_1d, label_5d = align_and_mask_evaluation_inputs(alpha, label_1d, label_5d, evaluation_mask)
    daily_ic = calculate_daily_ic_from_signal(alpha, label_1d, label_5d)
    daily_pnl = calculate_daily_pnl_from_signal(
        alpha,
        label_1d,
        booksize=artifacts.booksize,
        tradecost_ratio=artifacts.tradecost_ratio,
    )

    ic_result = summarize_ic(daily_ic, start=eval_start, end=eval_end, normalize_names=True)
    pnl_result = summarize_pnl_with_benchmark(daily_pnl, pnlzz500_path, start=eval_start, end=eval_end)
    ic_summary = ic_result.table if ic_result is not None else None
    pnl_summary = pnl_result.table if pnl_result is not None else None
    ic_checks = evaluate_result(ic_result) if ic_result is not None else None
    pnl_checks = evaluate_result(pnl_result) if pnl_result is not None else None

    decile_daily: dict[str, pd.DataFrame] = {}
    decile_summary = None
    top10_excess = None
    if skip_deciles:
        messages.append("decile backtest skipped by --skip-deciles")
    else:
        try:
            decile_daily = calculate_decile_daily_pnls(
                alpha,
                label_1d,
                booksize=artifacts.booksize,
                tradecost_ratio=artifacts.tradecost_ratio,
            )
            decile_summary = summarize_decile_daily_pnls(decile_daily, start=eval_start, end=eval_end)
            top10_excess = compute_top10_excess(decile_daily, daily_pnl, artifacts.booksize)
        except Exception as exc:  # pragma: no cover - depends on local data/cache availability
            messages.append(f"decile backtest unavailable: {exc}")

    exposure_summary = None
    cap_corr_summary = None
    if skip_exposure:
        messages.append("Barra exposure and CAP correlation skipped by --skip-exposure")
    else:
        try:
            exposure = compute_barra_style_exposure(
                alpha,
                start_ds=int(eval_start) if eval_start is not None else None,
                end_ds=int(eval_end) if eval_end is not None else None,
                mode=0,
                cache_path=artifacts.cache_path,
            )
            exposure_summary = summarize_exposure(exposure)
        except Exception as exc:  # pragma: no cover - depends on local AshareCache availability
            messages.append(f"Barra exposure unavailable: {exc}")
        try:
            cap_corr = compute_cap_corr(
                alpha,
                start_ds=int(eval_start) if eval_start is not None else None,
                end_ds=int(eval_end) if eval_end is not None else None,
                cache_path=artifacts.cache_path,
            )
            cap_corr_summary = summarize_cap_corr(cap_corr)
        except Exception as exc:  # pragma: no cover - depends on local AshareCache availability
            messages.append(f"CAP correlation unavailable: {exc}")

    _write_outputs(
        artifacts,
        ic_summary=ic_summary,
        pnl_summary=pnl_summary,
        ic_checks=ic_checks,
        pnl_checks=pnl_checks,
        decile_summary=decile_summary,
        exposure_summary=exposure_summary,
        cap_corr_summary=cap_corr_summary,
        top10_excess=top10_excess,
        messages=messages,
        start=eval_start,
        end=eval_end,
    )
    plot_signal_analysis(
        artifacts.plot_path,
        alpha=alpha,
        daily_ic=daily_ic,
        daily_pnl=daily_pnl,
        ic_summary=ic_summary,
        pnl_summary=pnl_summary,
        decile_daily=decile_daily,
        decile_summary=decile_summary,
        exposure_summary=exposure_summary,
        cap_corr_summary=cap_corr_summary,
        top10_excess=top10_excess,
        messages=messages,
        booksize=artifacts.booksize,
    )

    return ConfigEvalResult(
        artifacts=artifacts,
        ic_summary=ic_summary,
        pnl_summary=pnl_summary,
        ic_checks=ic_checks,
        pnl_checks=pnl_checks,
        decile_summary=decile_summary,
        exposure_summary=exposure_summary,
        cap_corr_summary=cap_corr_summary,
        top10_excess=top10_excess,
        messages=messages,
    )


def check_config_outputs(config_path: str | Path) -> ConfigOutputCheck:
    resolved_config_path = Path(config_path).expanduser().resolve()
    config = _load_organize_config(resolved_config_path)
    output_root = Path(config["constants"]["output_root"]).expanduser().resolve()
    alpha_path = _find_existing_alpha_path(output_root) or output_root / "alpha.parquet"
    files = (
        _file_status("alpha.parquet", alpha_path),
    )
    return ConfigOutputCheck(
        config_path=resolved_config_path,
        output_root=output_root,
        files=files,
    )


def _strategy_date(config: dict, key: str) -> str | None:
    value = config.get("strategy", {}).get(key)
    if value in (None, ""):
        return None
    return str(value)


def calculate_daily_pnl_from_signal(
    signal: pd.DataFrame,
    label: pd.DataFrame,
    *,
    booksize: float = 1e7,
    tradecost_ratio: float = 0.0,
) -> pd.DataFrame:
    signal = normalize_date_index(signal)
    label = normalize_date_index(label)
    signal.columns = signal.columns.astype(str).str.zfill(6)
    label.columns = label.columns.astype(str).str.zfill(6)
    signal, label = signal.align(label, join="inner", axis=0)
    signal, label = signal.align(label, join="inner", axis=1)
    if signal.empty or label.empty:
        raise ValueError("No overlapping dates or instruments between signal and label.")

    x = signal.astype(float).to_numpy()
    y = label.astype(float).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    positions = _scale_to_book(np.where(valid, x, np.nan), booksize)
    long_positions = _scale_long_only(np.where(valid, x, np.nan), booksize)
    gross = np.nansum(positions * np.where(np.isfinite(y), y, np.nan), axis=1)
    long_gross = np.nansum(long_positions * np.where(np.isfinite(y), y, np.nan), axis=1)
    tradevalue = np.nansum(np.abs(np.diff(positions, axis=0, prepend=np.zeros_like(positions[:1]))), axis=1)
    long_tradevalue = np.nansum(np.abs(np.diff(long_positions, axis=0, prepend=np.zeros_like(long_positions[:1]))), axis=1)
    turnover = tradevalue / (booksize * 2)
    long_turnover = long_tradevalue / booksize
    tradecost = tradevalue * 0.003 * tradecost_ratio
    long_tradecost = long_tradevalue * 0.003 * tradecost_ratio
    pnl = gross - tradecost
    long_pnl = long_gross - long_tradecost
    long = np.nansum(np.where(positions > 0, positions, 0), axis=1)
    short = np.nansum(np.where(positions < 0, positions, 0), axis=1)
    long_count = np.sum(positions > 0, axis=1)
    short_count = np.sum(positions < 0, axis=1)
    label_count = np.sum(np.isfinite(y), axis=1).astype(float)
    label_count[label_count == 0] = np.nan
    coverage = np.sum(valid, axis=1) / label_count

    return pd.DataFrame(
        {
            "pnl": pnl,
            "pnl_gross": gross,
            "tradecost": tradecost,
            "longonly_pnl": long_pnl,
            "longonly_pnl_gross": long_gross,
            "longonly_tradecost": long_tradecost,
            "longonly_tvr_pct": long_turnover * 100,
            "tvr_pct": turnover * 100,
            "long": long,
            "short": short,
            "sh_hld": np.abs(long) + np.abs(short),
            "sh_trd": tradevalue,
            "n_long": long_count,
            "n_short": short_count,
            "coverage": coverage,
        },
        index=signal.index,
    )


def calculate_decile_daily_pnls(
    signal: pd.DataFrame,
    label: pd.DataFrame,
    *,
    booksize: float = 1e7,
    tradecost_ratio: float = 0.0,
) -> dict[str, pd.DataFrame]:
    signal = normalize_date_index(signal)
    signal_values = signal.astype(float)
    percentile = signal.rank(axis=1, pct=True, method="first").to_numpy(dtype=float)
    finite_count = np.isfinite(signal_values.to_numpy()).sum(axis=1)
    row_min = signal_values.min(axis=1, skipna=True).to_numpy(dtype=float)
    row_max = signal_values.max(axis=1, skipna=True).to_numpy(dtype=float)
    has_cross_sectional_signal = (finite_count >= 2) & np.isfinite(row_min) & np.isfinite(row_max) & (row_min < row_max)
    buckets = np.full(percentile.shape, -1, dtype=np.int16)
    valid = np.isfinite(percentile) & has_cross_sectional_signal[:, None]
    buckets[valid] = np.minimum((percentile[valid] * 10).astype(np.int16), 9)

    output: dict[str, pd.DataFrame] = {}
    for bucket in range(10):
        group_signal = pd.DataFrame(
            np.where(buckets == bucket, 1.0, np.nan),
            index=signal.index,
            columns=signal.columns,
        )
        output[f"Q{bucket + 1}"] = calculate_daily_pnl_from_signal(
            group_signal,
            label,
            booksize=booksize,
            tradecost_ratio=tradecost_ratio,
        )
    return output


def summarize_decile_daily_pnls(
    decile_daily: dict[str, pd.DataFrame],
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    rows = []
    for name, daily in decile_daily.items():
        summary = summarize_pnl_with_benchmark(daily, start=start, end=end).table
        all_row = summary.loc["ALL"] if "ALL" in summary.index else summary.iloc[-1]
        rows.append(
            {
                "bucket": name,
                "longonly_ret_pct": _value(all_row, "longonly_ret_pct"),
                "longonly_ir": _value(all_row, "longonly_ir"),
                "longonly_sharpe": _value(all_row, "longonly_sharpe"),
                "longonly_tvr_pct": _value(all_row, "longonly_tvr_pct"),
                "pnl_ret_pct": _value(all_row, "ret_pct"),
                "pnl_ir": _value(all_row, "ir"),
                "coverage": _value(all_row, "tratio"),
            }
        )
    return pd.DataFrame(rows).set_index("bucket")


def compute_top10_excess(
    decile_daily: dict[str, pd.DataFrame],
    base_daily_pnl: pd.DataFrame | None,
    booksize: float,
) -> pd.DataFrame | None:
    if "Q10" not in decile_daily:
        return None
    top = _daily_return(decile_daily["Q10"], booksize).rename("top10_ret")
    if base_daily_pnl is not None:
        base = _daily_return(base_daily_pnl, booksize).rename("base_ret")
    else:
        base = pd.concat([_daily_return(frame, booksize) for frame in decile_daily.values()], axis=1).mean(axis=1).rename("base_ret")
    aligned = pd.concat([top, base], axis=1, join="inner").dropna()
    if aligned.empty:
        return None
    aligned["excess_ret"] = aligned["top10_ret"] - aligned["base_ret"]
    aligned["cum_top10_ret"] = aligned["top10_ret"].cumsum()
    aligned["cum_base_ret"] = aligned["base_ret"].cumsum()
    aligned["cum_excess_ret"] = aligned["excess_ret"].cumsum()
    return aligned


def summarize_exposure(exposure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in exposure.columns:
        values = exposure[column].dropna().astype(float)
        rows.append(
            {
                "style": column,
                "min": float(values.min()) if not values.empty else np.nan,
                "mean": float(values.mean()) if not values.empty else np.nan,
                "max": float(values.max()) if not values.empty else np.nan,
                "std": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("style")


def plot_signal_analysis(
    path: str | Path,
    *,
    alpha: pd.DataFrame,
    daily_ic: pd.DataFrame | None,
    daily_pnl: pd.DataFrame | None,
    ic_summary: pd.DataFrame | None,
    pnl_summary: pd.DataFrame | None,
    decile_daily: dict[str, pd.DataFrame],
    decile_summary: pd.DataFrame | None,
    exposure_summary: pd.DataFrame | None,
    cap_corr_summary: pd.DataFrame | None,
    top10_excess: pd.DataFrame | None,
    messages: list[str],
    booksize: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    alpha = normalize_date_index(alpha)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 32), constrained_layout=True)
    grid = fig.add_gridspec(8, 1, height_ratios=[1.0, 1.2, 1.3, 1.9, 1.8, 1.5, 1.9, 0.8])
    fig.suptitle("Signal Evaluation Report", fontsize=20, fontweight="bold")

    ax_summary = fig.add_subplot(grid[0])
    _plot_summary_text(ax_summary, alpha, ic_summary, pnl_summary, decile_summary, exposure_summary, cap_corr_summary, messages)

    ax_ic = fig.add_subplot(grid[1])
    _plot_ic_stats(ax_ic, daily_ic)

    ax_pnl = fig.add_subplot(grid[2])
    _plot_pnl_stats(ax_pnl, daily_pnl, booksize)

    ax_quantile = fig.add_subplot(grid[3])
    _plot_signal_quantiles(ax_quantile, alpha)

    ax_decile = fig.add_subplot(grid[4])
    _plot_decile_backtest(ax_decile, decile_daily, booksize)

    ax_decile_bar = fig.add_subplot(grid[5])
    _plot_decile_summary(ax_decile_bar, decile_summary)

    ax_exposure = fig.add_subplot(grid[6])
    _plot_exposure(ax_exposure, exposure_summary)

    ax_top = fig.add_subplot(grid[7])
    _plot_top10_excess(ax_top, top10_excess)

    date_axes = [ax_ic, ax_pnl, ax_quantile, ax_decile, ax_top]
    for ax in fig.axes:
        if ax.has_data():
            ax.grid(True, alpha=0.25, linewidth=0.7)
    for ax in date_axes:
        if ax.has_data():
            try:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            except Exception:
                pass
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_organize_config(config_path: str | Path) -> dict[str, Any]:
    try:
        config_module_path = _find_organize_config_module()
    except FileNotFoundError:
        config_module = importlib.import_module("config")
        return config_module.load_config(str(config_path))

    config_spec = importlib.util.spec_from_file_location("comb2_organize_config", config_module_path)
    if config_spec is None or config_spec.loader is None:
        raise ImportError(f"unable to load config module: {config_module_path}")
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    load_config = config_module.load_config

    return load_config(str(config_path))


def _find_organize_config_module() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("unable to find comb2-organize config.py")


def _resolve_artifacts(
    config_path: Path,
    config: dict[str, Any],
    *,
    report_dir: str | Path | None,
    plot_path: str | Path | None,
    label_path: str | Path | None,
    label_5d_path: str | Path | None,
    label_is_table: bool,
    label_5d_is_table: bool,
    label_df_type: object,
    booksize: float | None,
    tradecost_ratio: float | None,
) -> ConfigEvalArtifacts:
    output_root = Path(config["constants"]["output_root"]).expanduser().resolve()
    alpha_path = _find_alpha_path(output_root)
    resolved_report_dir = Path(report_dir).expanduser().resolve() if report_dir else output_root / "eval_report"
    resolved_plot_path = Path(plot_path).expanduser().resolve() if plot_path else resolved_report_dir / "signal_analysis.png"
    snap_ti = config["combo"]["runtime"].get("snap_ti")
    resolved_label_path = Path(label_path).expanduser().resolve() if label_path else None
    resolved_label_5d_path = Path(label_5d_path).expanduser().resolve() if label_5d_path else None
    if snap_ti is None:
        resolved_label_path = resolved_label_path or _default_label_path(config)
        resolved_label_5d_path = resolved_label_5d_path or _default_label_5d_path(config)
    resolved_booksize = float(booksize if booksize is not None else config["backtest"].get("cash", 1e7))
    resolved_tradecost_ratio = float(
        tradecost_ratio
        if tradecost_ratio is not None
        else _tradecost_ratio_from_fee(config["backtest"].get("fee_rate", 0.0))
    )
    cache_path = Path(config["constants"]["cache_path"]).expanduser().resolve()
    return ConfigEvalArtifacts(
        config_path=config_path,
        output_root=output_root,
        report_dir=resolved_report_dir,
        alpha_path=alpha_path,
        plot_path=resolved_plot_path,
        label_path=resolved_label_path,
        label_5d_path=resolved_label_5d_path,
        snap_ti=int(snap_ti) if snap_ti is not None else None,
        label_is_table=label_is_table,
        label_5d_is_table=label_5d_is_table,
        label_df_type=label_df_type,
        booksize=resolved_booksize,
        tradecost_ratio=resolved_tradecost_ratio,
        cache_path=cache_path,
    )


def _find_alpha_path(output_root: Path) -> Path:
    path = _find_existing_alpha_path(output_root)
    if path is None:
        raise FileNotFoundError(f"config output_root must contain at least one alpha.parquet: {output_root}")
    return path


def _find_existing_alpha_path(output_root: Path) -> Path | None:
    direct = output_root / "alpha.parquet"
    if direct.exists() and direct.is_file():
        return direct
    if not output_root.exists():
        return None
    candidates = sorted(output_root.rglob("alpha.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _file_status(name: str, path: Path) -> OutputFileStatus:
    if not path.exists():
        return OutputFileStatus(name, path, False, "missing")
    if not path.is_file():
        return OutputFileStatus(name, path, False, "not a file")
    if path.stat().st_size <= 0:
        return OutputFileStatus(name, path, False, "empty")
    return OutputFileStatus(name, path, True)


def _default_label_path(config: dict[str, Any]) -> Path:
    return daily_label_path(config["constants"]["cache_path"], "vwap30_label1d")


def _default_label_5d_path(config: dict[str, Any]) -> Path:
    return daily_label_path(config["constants"]["cache_path"], "vwap30_label5d")


def _tradecost_ratio_from_fee(fee_rate: float) -> float:
    # rundailypnl.py uses cost = tradevalue * 0.003 * tradecost_ratio.
    return float(fee_rate) / 0.003 if fee_rate else 0.0


def _read_alpha(path: Path, *, start: str | None, end: str | None) -> pd.DataFrame:
    return read_matrix(path, start=start, end=end)


def calculate_daily_ic_from_signal(signal: pd.DataFrame, label_1d: pd.DataFrame, label_5d: pd.DataFrame) -> pd.DataFrame:
    signal, label_1d, label_5d = align_and_mask_evaluation_inputs(signal, label_1d, label_5d)

    x = signal.astype(float).to_numpy()
    y1 = label_1d.astype(float).to_numpy()
    y5 = label_5d.astype(float).to_numpy()
    valid_1d = np.isfinite(x) & np.isfinite(y1)
    layer_ic, layer_spread = _row_layer_metrics(x, y1)
    label_count = np.sum(np.isfinite(y1), axis=1).astype(float)
    label_count[label_count == 0] = np.nan
    return pd.DataFrame(
        {
            "ic": _row_corr(x, y1),
            "5dic": _row_corr(x, y5),
            "rankic": _row_corr(_row_rank(x), _row_rank(y1)),
            "lic": layer_ic,
            "layerspread": layer_spread,
            "coverage": np.sum(valid_1d, axis=1) / label_count,
        },
        index=signal.index,
    )


def load_evaluation_mask(signal: pd.DataFrame, cache_path: str | Path) -> pd.DataFrame:
    normalized = _normalize_matrix(signal)
    if normalized.empty:
        raise ValueError("signal is empty")
    start_ds = normalized.index.min().strftime("%Y%m%d")
    end_ds = normalized.index.max().strftime("%Y%m%d")
    base = _normalize_matrix(read_cache_array(stock_mask_path(cache_path, BASE_UNIVERSE_MASK_NAME), start_ds, end_ds, True))
    trading = _normalize_matrix(read_cache_array(stock_mask_path(cache_path, FILTERED_MASK_NAME), start_ds, end_ds, True))
    dates = base.index.intersection(trading.index)
    codes = base.columns.intersection(trading.columns)
    base = base.reindex(index=dates, columns=codes)
    trading = trading.reindex(index=dates, columns=codes)
    current = base.notna() & base.ne(0) & trading.notna() & trading.ne(0)
    return current.shift(-1).fillna(False)


def align_and_mask_evaluation_inputs(
    signal: pd.DataFrame,
    label_1d: pd.DataFrame,
    label_5d: pd.DataFrame,
    evaluation_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [_normalize_matrix(signal), _normalize_matrix(label_1d), _normalize_matrix(label_5d)]
    dates = frames[0].index
    codes = frames[0].columns
    for frame in frames[1:]:
        dates = dates.intersection(frame.index)
        codes = codes.intersection(frame.columns)
    if evaluation_mask is not None:
        evaluation_mask = _normalize_matrix(evaluation_mask)
        dates = dates.intersection(evaluation_mask.index)
        codes = codes.intersection(evaluation_mask.columns)
    if dates.empty or codes.empty:
        raise ValueError("No overlapping dates or instruments between signal, labels, and masks.")
    aligned = [frame.reindex(index=dates, columns=codes) for frame in frames]
    if evaluation_mask is not None:
        valid = evaluation_mask.reindex(index=dates, columns=codes).fillna(False).astype(bool)
        aligned = [frame.where(valid) for frame in aligned]
    return aligned[0], aligned[1], aligned[2]


def _normalize_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_date_index(frame).copy()
    normalized.columns = normalized.columns.astype(str).str.zfill(6)
    return normalized.sort_index()


def _read_label_for_signal(
    signal: pd.DataFrame,
    artifacts: ConfigEvalArtifacts,
    *,
    path: str | Path,
    is_table: bool,
) -> pd.DataFrame:
    label_path = Path(path)
    if is_table:
        return read_matrix(label_path)
    start_ds = normalize_date_index(signal).index.min().strftime("%Y%m%d")
    end_ds = normalize_date_index(signal).index.max().strftime("%Y%m%d")
    data = read_cache_array(label_path, start_ds, end_ds, artifacts.label_df_type)
    if isinstance(data, pd.DataFrame):
        label = normalize_date_index(data).astype(float)
        label.columns = label.columns.astype(str).str.zfill(6)
        return label
    signal = normalize_date_index(signal)
    return pd.DataFrame(data, index=signal.index[: len(data)], columns=signal.columns[: data.shape[1]]).astype(float)


def _write_outputs(
    artifacts: ConfigEvalArtifacts,
    *,
    ic_summary: pd.DataFrame | None,
    pnl_summary: pd.DataFrame | None,
    ic_checks: pd.DataFrame | None,
    pnl_checks: pd.DataFrame | None,
    decile_summary: pd.DataFrame | None,
    exposure_summary: pd.DataFrame | None,
    cap_corr_summary: pd.DataFrame | None,
    top10_excess: pd.DataFrame | None,
    messages: list[str],
    start: str | None,
    end: str | None,
) -> None:
    frames = {
        "ic_summary.csv": ic_summary,
        "pnl_summary.csv": pnl_summary,
        "ic_checks.csv": ic_checks,
        "pnl_checks.csv": pnl_checks,
        "decile_summary.csv": decile_summary,
        "barra_exposure_summary.csv": exposure_summary,
        "cap_corr_summary.csv": cap_corr_summary,
        "top10_excess.csv": top10_excess,
    }
    for filename, frame in frames.items():
        if frame is not None:
            frame.to_csv(artifacts.report_dir / filename)
    manifest = {
        "config_path": str(artifacts.config_path),
        "output_root": str(artifacts.output_root),
        "alpha_path": str(artifacts.alpha_path),
        "plot_path": str(artifacts.plot_path),
        "label_path": _label_source(artifacts.label_path, artifacts.snap_ti, 1),
        "label_5d_path": _label_source(artifacts.label_5d_path, artifacts.snap_ti, 5),
        "start": start,
        "end": end,
        "messages": messages,
    }
    (artifacts.report_dir / "report.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def _label_source(path: Path | None, snap_ti: int | None, period: int) -> str:
    if path is not None:
        return str(path)
    return f"dynamic:{snap_vwap_price_name(snap_ti)}:label{period}d"


def config_eval_to_text(result: ConfigEvalResult) -> str:
    lines = [
        f"[config] {result.artifacts.config_path}",
        f"[output_root] {result.artifacts.output_root}",
        f"[alpha] {result.artifacts.alpha_path}",
        f"[report_dir] {result.artifacts.report_dir}",
        f"[plot] {result.artifacts.plot_path}",
    ]
    if result.messages:
        lines.append("\n[messages]")
        lines.extend(f"- {message}" for message in result.messages)
    if result.ic_summary is not None:
        lines.append("\n[ic.summary]")
        lines.append(output_frame_to_text(_select_columns(result.ic_summary, IC_KEY_COLUMNS)))
    if result.ic_checks is not None:
        lines.append("\n[ic.checks]")
        lines.append(output_frame_to_text(result.ic_checks))
    if result.pnl_summary is not None:
        lines.append("\n[pnl.summary]")
        lines.append(output_frame_to_text(_select_columns(result.pnl_summary, PNL_KEY_COLUMNS)))
    if result.pnl_checks is not None:
        lines.append("\n[pnl.checks]")
        lines.append(output_frame_to_text(result.pnl_checks))
    if result.decile_summary is not None:
        lines.append("\n[decile.summary]")
        lines.append(output_frame_to_text(result.decile_summary))
    if result.exposure_summary is not None:
        lines.append("\n[barra.exposure.summary]")
        lines.append(output_frame_to_text(result.exposure_summary))
    if result.cap_corr_summary is not None:
        lines.append("\n[cap_corr.summary]")
        lines.append(output_frame_to_text(result.cap_corr_summary))
    return "\n".join(lines)


def _plot_summary_text(
    ax,
    alpha: pd.DataFrame,
    ic_summary: pd.DataFrame | None,
    pnl_summary: pd.DataFrame | None,
    decile_summary: pd.DataFrame | None,
    exposure_summary: pd.DataFrame | None,
    cap_corr_summary: pd.DataFrame | None,
    messages: list[str],
) -> None:
    ax.axis("off")
    alpha = normalize_date_index(alpha)
    lines = [
        f"alpha dates: {alpha.index.min().date()} to {alpha.index.max().date()}",
        f"alpha shape: {alpha.shape[0]} dates x {alpha.shape[1]} instruments",
        f"finite coverage: {np.isfinite(alpha.to_numpy(dtype=float)).mean():.2%}",
    ]
    if ic_summary is not None and "ALL" in ic_summary.index:
        row = ic_summary.loc["ALL"]
        lines.append(f"IC ALL: 1d avg={_value(row, '1d_IC.avg'):.4f}, 1d ir={_value(row, '1d_IC.ir'):.4f}, rank avg={_value(row, 'rankic.avg'):.4f}")
        lines.append(
            f"Layer ALL: lIC={_value(row, 'lIC.avg'):.4f}, lIR={_value(row, 'lIC.ir'):.4f}, "
            f"Q10-Q1={_value(row, 'layerSpread.avg'):.4f}"
        )
    if pnl_summary is not None and "ALL" in pnl_summary.index:
        row = pnl_summary.loc["ALL"]
        lines.append(f"PNL ALL: ret={_value(row, 'ret_pct'):.2f}%, ir={_value(row, 'ir'):.4f}, sharpe={_value(row, 'sharpe'):.2f}, tvr={_value(row, 'tvr_pct'):.2f}%")
    if decile_summary is not None and "Q10" in decile_summary.index:
        row = decile_summary.loc["Q10"]
        lines.append(f"Top10 backtest: long-only ret={_value(row, 'longonly_ret_pct'):.2f}%, ir={_value(row, 'longonly_ir'):.4f}")
    if exposure_summary is not None:
        strongest = exposure_summary["mean"].abs().sort_values(ascending=False).head(3)
        lines.append("Largest mean Barra exposure: " + ", ".join(f"{idx}={value:.3f}" for idx, value in strongest.items()))
    if cap_corr_summary is not None and "ALL" in cap_corr_summary.index:
        row = cap_corr_summary.loc["ALL"]
        lines.append(f"CAP corr ALL: avg={_value(row, 'cap_corr.avg'):.4f}, ir={_value(row, 'cap_corr.ir'):.4f}")
    if messages:
        lines.append("Notes: " + " | ".join(messages[:4]))
    ax.text(0.01, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12, family="monospace")


def _plot_ic_stats(ax, daily_ic: pd.DataFrame | None) -> None:
    ax.set_title("IC Statistics")
    if daily_ic is None or daily_ic.empty:
        _plot_unavailable(ax, "IC data unavailable")
        return
    daily_ic = normalize_date_index(daily_ic)
    for column in ["ic", "5dic", "rankic", "lic", "1d_IC", "5d_IC", "lIC"]:
        if column in daily_ic.columns:
            ax.plot(daily_ic.index, daily_ic[column].astype(float).rolling(20, min_periods=1).mean(), label=f"{column} 20d")
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("rolling IC")
    ax.legend(loc="upper left", ncols=3, fontsize=9)


def _plot_pnl_stats(ax, daily_pnl: pd.DataFrame | None, booksize: float) -> None:
    ax.set_title("PNL Statistics")
    if daily_pnl is None or daily_pnl.empty:
        _plot_unavailable(ax, "PNL data unavailable")
        return
    daily_pnl = normalize_date_index(daily_pnl)
    ret = _daily_return(daily_pnl, booksize)
    ax.plot(ret.index, ret.cumsum() * 100, label="cumulative return")
    if "pnl" in daily_pnl.columns:
        ax2 = ax.twinx()
        ax2.plot(daily_pnl.index, daily_pnl["pnl"].astype(float).rolling(20, min_periods=1).mean() / 1e6, color="#b45f06", alpha=0.75, label="20d pnl m")
        ax2.set_ylabel("20d pnl (m)")
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("cum ret (%)")
    ax.legend(loc="upper left", fontsize=9)


def _plot_signal_quantiles(ax, alpha: pd.DataFrame) -> None:
    ax.set_title("Signal Value Distribution by Time")
    quantiles = alpha.quantile([i / 10 for i in range(1, 10)], axis=1).T
    quantiles.columns = [f"q{int(q * 100)}" for q in quantiles.columns]
    palette = ["#355c7d", "#457b9d", "#2a9d8f", "#6a994e", "#a7c957", "#f4a261", "#e76f51", "#c1121f", "#780000"]
    for idx, column in enumerate(quantiles.columns):
        ax.plot(quantiles.index, quantiles[column], label=column, linewidth=1.2, color=palette[idx % len(palette)])
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("signal value")
    ax.legend(loc="upper left", ncols=5, fontsize=8)


def _plot_decile_backtest(ax, decile_daily: dict[str, pd.DataFrame], booksize: float) -> None:
    ax.set_title("Decile Backtest Cumulative Long-only Return")
    if not decile_daily:
        _plot_unavailable(ax, "decile backtest unavailable")
        return
    colors = plt_colors(len(decile_daily))
    for (name, daily), color in zip(decile_daily.items(), colors):
        ret = _daily_return(daily, booksize)
        ax.plot(ret.index, ret.cumsum() * 100, label=name, linewidth=1.15 if name not in {"Q1", "Q10"} else 2.0, color=color)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("cum ret (%)")
    ax.legend(loc="upper left", ncols=5, fontsize=8)


def _plot_decile_summary(ax, decile_summary: pd.DataFrame | None) -> None:
    ax.set_title("Decile Backtest Annualized Return")
    if decile_summary is None or decile_summary.empty:
        _plot_unavailable(ax, "decile summary unavailable")
        return
    values = decile_summary["longonly_ret_pct"].astype(float)
    colors = ["#6c757d"] * len(values)
    if len(colors) >= 10:
        colors[0] = "#457b9d"
        colors[-1] = "#c1121f"
    ax.bar(values.index, values.values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("annualized ret (%)")
    for idx, value in enumerate(values.values):
        if np.isfinite(value):
            ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)


def _plot_exposure(ax, exposure_summary: pd.DataFrame | None) -> None:
    ax.set_title("Barra Style Exposure Range")
    if exposure_summary is None or exposure_summary.empty:
        _plot_unavailable(ax, "Barra exposure unavailable")
        return
    table = exposure_summary.sort_values("mean")
    y = np.arange(len(table))
    left = table["min"].astype(float).to_numpy()
    right = table["max"].astype(float).to_numpy()
    mean = table["mean"].astype(float).to_numpy()
    ax.hlines(y, left, right, color="#457b9d", linewidth=3)
    ax.scatter(mean, y, color="#c1121f", zorder=3, label="mean")
    ax.axvline(0.0, color="black", linewidth=0.9)
    ax.set_yticks(y, table.index)
    ax.set_xlabel("daily cross-sectional exposure")
    ax.legend(loc="lower right", fontsize=9)
    for yi, min_value, max_value in zip(y, left, right):
        if np.isfinite(min_value):
            ax.text(min_value, yi, f"{min_value:.2f}", va="center", ha="right", fontsize=7)
        if np.isfinite(max_value):
            ax.text(max_value, yi, f"{max_value:.2f}", va="center", ha="left", fontsize=7)


def _plot_top10_excess(ax, top10_excess: pd.DataFrame | None) -> None:
    ax.set_title("Top10% Backtest Excess Return")
    if top10_excess is None or top10_excess.empty:
        _plot_unavailable(ax, "top10 excess unavailable")
        return
    top10_excess = normalize_date_index(top10_excess)
    ax.plot(top10_excess.index, top10_excess["cum_top10_ret"] * 100, label="top10")
    ax.plot(top10_excess.index, top10_excess["cum_base_ret"] * 100, label="base")
    ax.plot(top10_excess.index, top10_excess["cum_excess_ret"] * 100, label="excess", linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel("cum ret (%)")
    ax.legend(loc="upper left", ncols=3, fontsize=9)


def _plot_unavailable(ax, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)


def _daily_return(daily: pd.DataFrame, booksize: float) -> pd.Series:
    daily = normalize_date_index(daily)
    if "longonly_pnl" in daily.columns:
        return daily["longonly_pnl"].astype(float) / booksize
    if "ret" in daily.columns:
        return daily["ret"].astype(float)
    if "Return" in daily.columns:
        return daily["Return"].astype(float)
    if "pnl" in daily.columns:
        denominator = None
        for column in ("long", "total_asset", "sh_hld"):
            if column in daily.columns:
                denominator = daily[column].astype(float).abs().replace(0, np.nan)
                break
        if denominator is None:
            denominator = pd.Series(booksize, index=daily.index, dtype=float)
        return daily["pnl"].astype(float) / denominator
    raise ValueError("daily pnl needs one of longonly_pnl, ret, Return, or pnl")


def _scale_to_book(values: np.ndarray, booksize: float) -> np.ndarray:
    positions = np.nan_to_num(values, nan=0.0).astype(float)
    long_sum = np.where(positions > 0, positions, 0).sum(axis=1)
    short_sum = -np.where(positions < 0, positions, 0).sum(axis=1)
    long_scale = np.divide(booksize, long_sum, out=np.zeros_like(long_sum), where=long_sum > 0)
    short_scale = np.divide(booksize, short_sum, out=np.zeros_like(short_sum), where=short_sum > 0)
    return np.where(positions > 0, positions * long_scale[:, None], np.where(positions < 0, positions * short_scale[:, None], 0.0))


def _scale_long_only(values: np.ndarray, booksize: float) -> np.ndarray:
    positions = np.nan_to_num(values, nan=0.0).astype(float)
    positions = np.where(positions > 0, positions, 0.0)
    long_sum = positions.sum(axis=1)
    long_scale = np.divide(booksize, long_sum, out=np.zeros_like(long_sum), where=long_sum > 0)
    return positions * long_scale[:, None]


def _row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1)
    count_column = count[:, None]
    left_sum = np.where(valid, left, 0.0).sum(axis=1, keepdims=True)
    right_sum = np.where(valid, right, 0.0).sum(axis=1, keepdims=True)
    left_mean = np.divide(left_sum, count_column, out=np.full_like(left_sum, np.nan, dtype=float), where=count_column > 0)
    right_mean = np.divide(right_sum, count_column, out=np.full_like(right_sum, np.nan, dtype=float), where=count_column > 0)
    left_centered = np.where(valid, left - left_mean, 0.0)
    right_centered = np.where(valid, right - right_mean, 0.0)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered * left_centered, axis=1) * np.sum(right_centered * right_centered, axis=1))
    return np.divide(numerator, denominator, out=np.full(left.shape[0], np.nan), where=(count >= 2) & (denominator > 0))


def _row_rank(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(axis=1).to_numpy()


def _row_layer_metrics(
    signal: np.ndarray,
    forward_return: np.ndarray,
    *,
    layer_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return daily monotonicity IC and Q10-Q1 spread for equal-frequency alpha layers.

    Ties are never split across layers. If ties leave fewer than ``layer_count``
    non-empty buckets, the date is excluded so the spread remains literal Q10-Q1.
    """
    layer_ic = np.full(signal.shape[0], np.nan, dtype=float)
    layer_spread = np.full(signal.shape[0], np.nan, dtype=float)
    layer_numbers = np.arange(1, layer_count + 1, dtype=float)

    for row_idx in range(signal.shape[0]):
        valid = np.isfinite(signal[row_idx]) & np.isfinite(forward_return[row_idx])
        if valid.sum() < layer_count:
            continue
        alpha = signal[row_idx, valid]
        returns = forward_return[row_idx, valid]
        buckets = np.asarray(pd.qcut(alpha, q=layer_count, labels=False, duplicates="drop"))
        if not np.isfinite(buckets).all():
            continue
        buckets = buckets.astype(np.intp, copy=False)
        if not np.array_equal(np.unique(buckets), np.arange(layer_count)):
            continue

        layer_returns = np.array([returns[buckets == bucket].mean() for bucket in range(layer_count)], dtype=float)
        layer_ic[row_idx] = _row_corr(layer_numbers[None, :], layer_returns[None, :])[0]
        layer_spread[row_idx] = layer_returns[-1] - layer_returns[0]

    return layer_ic, layer_spread


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]]


def _value(row: pd.Series, key: str) -> float:
    return float(row[key]) if key in row and pd.notna(row[key]) else math.nan


def plt_colors(n: int) -> list[str]:
    base = ["#457b9d", "#2a9d8f", "#6a994e", "#a7c957", "#f4a261", "#e76f51", "#c1121f", "#9d4edd", "#5a189a", "#212529"]
    if n <= len(base):
        return base[:n]
    return [base[idx % len(base)] for idx in range(n)]


__all__ = [
    "ConfigEvalArtifacts",
    "ConfigEvalResult",
    "ConfigOutputCheck",
    "OutputFileStatus",
    "calculate_daily_ic_from_signal",
    "calculate_daily_pnl_from_signal",
    "calculate_decile_daily_pnls",
    "check_config_outputs",
    "config_eval_to_text",
    "run_config_evaluation",
    "summarize_decile_daily_pnls",
    "summarize_exposure",
]
