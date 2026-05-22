#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ORGANIZE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ORGANIZE_ROOT / "config.py"


@dataclass(frozen=True)
class FactorSpec:
    index: int
    name: str
    raw_path: str
    resolved_path: str


@dataclass
class CorrAccumulator:
    names: list[str]
    sum_corr: np.ndarray
    sum_abs_corr: np.ndarray
    max_abs_corr: np.ndarray
    count_days: np.ndarray
    high_abs_count: np.ndarray
    coverage_sum: np.ndarray
    coverage_count: np.ndarray
    used_dates: list[int]
    skipped_dates: list[tuple[int, str]]

    @classmethod
    def create(cls, names: list[str]) -> "CorrAccumulator":
        n = len(names)
        return cls(
            names=names,
            sum_corr=np.zeros((n, n), dtype=np.float64),
            sum_abs_corr=np.zeros((n, n), dtype=np.float64),
            max_abs_corr=np.full((n, n), np.nan, dtype=np.float64),
            count_days=np.zeros((n, n), dtype=np.int32),
            high_abs_count=np.zeros((n, n), dtype=np.int32),
            coverage_sum=np.zeros(n, dtype=np.float64),
            coverage_count=np.zeros(n, dtype=np.int32),
            used_dates=[],
            skipped_dates=[],
        )

    def add_day(
        self,
        ds: int,
        corr: np.ndarray,
        pair_counts: np.ndarray,
        coverage: np.ndarray,
        min_valid: int,
        high_corr_threshold: float,
    ) -> None:
        good = np.isfinite(corr) & (pair_counts >= min_valid)
        abs_corr = np.abs(corr)
        self.sum_corr[good] += corr[good]
        self.sum_abs_corr[good] += abs_corr[good]
        self.count_days[good] += 1
        current_max = self.max_abs_corr[good]
        self.max_abs_corr[good] = np.where(np.isnan(current_max), abs_corr[good], np.maximum(current_max, abs_corr[good]))
        self.high_abs_count[good] += (abs_corr[good] >= high_corr_threshold).astype(np.int32)

        factor_good = np.isfinite(coverage)
        self.coverage_sum[factor_good] += coverage[factor_good]
        self.coverage_count[factor_good] += 1
        self.used_dates.append(ds)

    def add_skip(self, ds: int, reason: str) -> None:
        self.skipped_dates.append((int(ds), reason))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pairwise cross-sectional correlations for all factors listed in an XML config. "
            "Outputs CSV matrices, pair/factor summaries, and optional Plotly heatmaps."
        )
    )
    parser.add_argument("config", nargs="?", default=str(ORGANIZE_ROOT / "eg-torch" / "config.xml"), help="XML config path")
    parser.add_argument("--config", dest="config_flag", default=None, help="XML config path, equivalent to positional config")
    parser.add_argument("--start-ds", type=int, default=None, help="Override analysis start date, e.g. 20200101")
    parser.add_argument("--end-ds", type=int, default=None, help="Override analysis end date, e.g. 20240630")
    parser.add_argument("--dates", default=None, help="Comma-separated explicit dates; bypasses IndexMask date selection")
    parser.add_argument("--date-step", type=int, default=1, help="Use every Nth selected trading day")
    parser.add_argument("--max-days", type=int, default=None, help="Evenly downsample to at most this many dates")
    parser.add_argument("--method", choices=("pearson", "spearman"), default="pearson", help="Correlation method")
    parser.add_argument(
        "--preprocess",
        choices=("raw", "model"),
        default="raw",
        help="raw keeps NaNs and computes pairwise finite correlations; model applies cs-zscore, truncate [-4, 4], then fill NaN with 0",
    )
    parser.add_argument(
        "--universe",
        choices=("none", "valid", "filtered", "base", "tradable"),
        default="none",
        help="Universe mask applied before correlation; default none only loads XML factors; tradable = valid & filtered & base",
    )
    parser.add_argument("--factor-root", default=None, help="Override constants.factor_root for relative factor paths")
    parser.add_argument("--cache-path", default=None, help="Override constants.cache_path for valid/filtered/base masks")
    parser.add_argument("--min-valid", type=int, default=1000, help="Minimum pairwise instruments required per day")
    parser.add_argument("--high-corr-threshold", type=float, default=0.8, help="Threshold used for high-correlation day ratio")
    parser.add_argument("--interpret-abs-corr-threshold", type=float, default=0.8, help="Mean absolute correlation threshold for duplicate-factor interpretation")
    parser.add_argument("--interpret-high-day-ratio-threshold", type=float, default=0.5, help="High-correlation day-ratio threshold for stable duplicate interpretation")
    parser.add_argument("--interpret-episodic-max-threshold", type=float, default=0.8, help="Max absolute correlation threshold for episodic-high-correlation interpretation")
    parser.add_argument("--interpret-episodic-mean-abs-max", type=float, default=0.5, help="Upper mean absolute correlation bound for episodic-high-correlation interpretation")
    parser.add_argument("--interpret-low-coverage-threshold", type=float, default=0.8, help="Average coverage threshold below which a factor is flagged")
    parser.add_argument("--interpret-top-n", type=int, default=30, help="Rows shown in interpretation markdown/json sections")
    parser.add_argument("--top-n", type=int, default=100, help="Rows to keep in top_pairs.csv")
    parser.add_argument("--output-dir", default=None, help="Report directory; defaults to <combo output_dir>/factor_corr")
    parser.add_argument("--skip-bad-days", action="store_true", help="Skip dates that fail to load instead of stopping")
    parser.add_argument("--no-html", action="store_true", help="Do not write Plotly HTML heatmaps")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved factors/dates/output path without loading factor data")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N processed dates")
    return parser.parse_args()


def import_organize_config():
    spec = importlib.util.spec_from_file_location("comb2_organize_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load config module: {CONFIG_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_factorsim():
    try:
        from factorsim import IndexMask, Memmaper2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "factorsim is required to load factor memmaps. Run this script in the project environment "
            "where runCombo.py can import factorsim."
        ) from exc
    return IndexMask, Memmaper2


def parse_xml_factor_paths(xml_path: Path) -> tuple[list[str], str | None]:
    root = ET.parse(xml_path).getroot()
    constants = root.find("constants")
    factor_root = constants.attrib.get("factor_root") if constants is not None else None
    loader = root.find("./combo/loader")
    factor_paths_element = loader.find("factor_paths") if loader is not None else None
    paths: list[str] = []
    if factor_paths_element is not None:
        for path_element in factor_paths_element.findall("path"):
            value = (path_element.text or "").strip()
            if value:
                paths.append(value)
    return paths, factor_root


def parse_xml_constant(xml_path: Path, name: str) -> str | None:
    root = ET.parse(xml_path).getroot()
    constants = root.find("constants")
    if constants is None:
        return None
    return constants.attrib.get(name)


def is_absolute_like(path_text: str) -> bool:
    path = Path(path_text).expanduser()
    return path.is_absolute() or path_text.startswith("/")


def join_data_path(base_text: str, child_text: str, xml_dir: Path) -> str:
    if is_absolute_like(child_text):
        return child_text

    normalized_child = child_text.replace("\\", "/")
    if base_text.startswith("/") and not Path(base_text).drive:
        return f"{base_text.rstrip('/')}/{normalized_child}"

    base_path = Path(base_text).expanduser()
    if not base_path.is_absolute():
        base_path = (xml_dir / base_path).resolve()
    return str((base_path / child_text).resolve())


def make_factor_name(raw_path: str, used: set[str]) -> str:
    name = Path(raw_path).name or raw_path.strip().replace("\\", "/").rstrip("/").split("/")[-1]
    if not name:
        name = "factor"
    base = name
    suffix = 2
    while name in used:
        name = f"{base}#{suffix}"
        suffix += 1
    used.add(name)
    return name


def build_factor_specs(xml_path: Path, config: dict, factor_root_override: str | None) -> list[FactorSpec]:
    raw_paths, xml_factor_root = parse_xml_factor_paths(xml_path)
    loaded_paths = list(config["combo"]["loader"].get("factor_paths") or [])
    if not raw_paths:
        raw_paths = [str(path) for path in loaded_paths]
    if not raw_paths:
        return []

    used_names: set[str] = set()
    specs: list[FactorSpec] = []
    factor_root = factor_root_override or xml_factor_root or str(xml_path.parent)
    for idx, raw in enumerate(raw_paths):
        resolved = join_data_path(factor_root, raw, xml_path.parent)
        specs.append(FactorSpec(idx, make_factor_name(raw, used_names), raw, resolved))
    return specs


def apply_cache_paths(config: dict, xml_path: Path, cache_path_override: str | None) -> None:
    cache_path = cache_path_override or parse_xml_constant(xml_path, "cache_path")
    if cache_path is None:
        return
    ashare_cache = join_data_path(cache_path, "AshareCache", xml_path.parent)
    loader = config["combo"]["loader"]
    loader["valid_path"] = join_data_path(ashare_cache, "Ashare", xml_path.parent)
    loader["filtered_path"] = join_data_path(ashare_cache, "AshareFiltered", xml_path.parent)
    loader["base_universe_path"] = join_data_path(ashare_cache, "1d_StockMask2/StockMask2.BaseUnivMask", xml_path.parent)


def parse_explicit_dates(value: str) -> list[int]:
    dates = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dates:
        raise ValueError("--dates did not contain any dates")
    return dates


def choose_dates(args: argparse.Namespace, config: dict, IndexMask) -> list[int]:
    if args.dates:
        dates = parse_explicit_dates(args.dates)
    else:
        start_ds = int(args.start_ds if args.start_ds is not None else config["strategy"]["start_ds"])
        end_ds = int(args.end_ds if args.end_ds is not None else config["strategy"]["end_ds"])
        all_dates = [int(ds) for ds in np.asarray(IndexMask().date).tolist()]
        dates = [ds for ds in all_dates if start_ds <= ds <= end_ds]
        if not dates:
            raise ValueError(f"no IndexMask dates found in range [{start_ds}, {end_ds}]")

    step = max(1, int(args.date_step))
    dates = dates[::step]
    if args.max_days is not None and len(dates) > args.max_days:
        positions = np.linspace(0, len(dates) - 1, int(args.max_days))
        dates = [dates[int(round(pos))] for pos in positions]
    return sorted(dict.fromkeys(dates))


class MemmapCache:
    def __init__(self, Memmaper2):
        self.Memmaper2 = Memmaper2
        self._items = {}

    def load_1d(self, path: str, ds: int) -> np.ndarray:
        if path not in self._items:
            self._items[path] = self.Memmaper2(path)
        data = self._items[path].load(start_ds=int(ds), end_ds=int(ds))[:]
        arr = np.asarray(data)
        arr = np.squeeze(arr)
        if arr.ndim != 1:
            raise ValueError(f"expected 1d data from {path} at {ds}, got shape {arr.shape}")
        return arr


def load_factor_matrix(cache: MemmapCache, specs: Sequence[FactorSpec], ds: int) -> np.ndarray:
    columns = []
    expected_len: int | None = None
    for spec in specs:
        values = cache.load_1d(spec.resolved_path, ds).astype(np.float64, copy=False)
        if expected_len is None:
            expected_len = len(values)
        elif len(values) != expected_len:
            raise ValueError(
                f"factor length mismatch at {ds}: {spec.name} has {len(values)}, expected {expected_len}"
            )
        columns.append(values)
    matrix = np.column_stack(columns)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def values_to_bool_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values != 0)


_MISSING_MASK_WARNED: set[str] = set()


def load_universe_mask(cache: MemmapCache, config: dict, universe: str, ds: int, length: int) -> np.ndarray:
    if universe == "none":
        return np.ones(length, dtype=bool)

    loader = config["combo"]["loader"]
    path_by_name = {
        "valid": loader.get("valid_path"),
        "filtered": loader.get("filtered_path"),
        "base": loader.get("base_universe_path"),
    }
    if universe == "tradable":
        names = ("valid", "filtered", "base")
    else:
        names = (universe,)

    mask = np.ones(length, dtype=bool)
    for name in names:
        path = path_by_name.get(name)
        if not path:
            raise ValueError(f"universe '{universe}' requires missing loader path: {name}")
        if not os.path.exists(str(path)):
            if str(path) not in _MISSING_MASK_WARNED:
                print(f"[FACTOR-CORR] mask path missing, using all-true mask like training loader: {path}")
                _MISSING_MASK_WARNED.add(str(path))
            continue
        values = cache.load_1d(str(path), ds)
        if len(values) != length:
            raise ValueError(f"mask length mismatch at {ds}: {name} has {len(values)}, expected {length}")
        mask &= values_to_bool_mask(values)
    return mask


def rank_columns(matrix: np.ndarray) -> np.ndarray:
    return pd.DataFrame(matrix).rank(axis=0, method="average", na_option="keep").to_numpy(dtype=np.float64)


def dense_corr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    ss = np.sum(centered * centered, axis=0)
    denom = np.sqrt(np.outer(ss, ss))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = centered.T @ centered / denom
    corr[~np.isfinite(corr)] = np.nan
    return corr, np.full_like(corr, n, dtype=np.float64)


def pairwise_corr_with_nans(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(matrix)
    x = np.where(valid, matrix, 0.0)
    m = valid.astype(np.float64)
    counts = m.T @ m
    sums = x.T @ m
    sums_t = sums.T
    sums_sq = (x * x).T @ m
    cross = x.T @ x

    with np.errstate(invalid="ignore", divide="ignore"):
        cov_num = cross - (sums * sums_t / counts)
        var_left = sums_sq - (sums * sums / counts)
        var_right = sums_sq.T - (sums_t * sums_t / counts)
        denom = np.sqrt(var_left * var_right)
        corr = cov_num / denom
    corr[(counts < 2) | ~np.isfinite(corr)] = np.nan
    return corr, counts


def model_preprocess(matrix: np.ndarray) -> np.ndarray:
    valid = np.isfinite(matrix)
    counts = valid.sum(axis=0)
    safe = np.where(valid, matrix, 0.0)
    sums = safe.sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centered = np.where(valid, matrix - means, np.nan)
    var = np.nansum(centered * centered, axis=0)
    std = np.sqrt(np.divide(var, counts, out=np.zeros_like(var), where=counts > 0))
    z = np.divide(centered, std + 1e-8, out=np.full_like(centered, np.nan), where=std > 0)
    z = np.clip(z, -4.0, 4.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def compute_day_corr(matrix: np.ndarray, method: str, preprocess: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverage = np.isfinite(matrix).mean(axis=0)
    if method == "spearman":
        matrix = rank_columns(matrix)

    if preprocess == "model":
        processed = model_preprocess(matrix)
        return (*dense_corr(processed), coverage)

    return (*pairwise_corr_with_nans(matrix), coverage)


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def matrix_df(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=names, columns=names)


PAIR_COLUMNS = ["factor_i", "factor_j", "mean_corr", "mean_abs_corr", "max_abs_corr", "days", "high_abs_day_ratio"]


def build_all_pair_table(
    names: list[str],
    mean_corr: np.ndarray,
    mean_abs_corr: np.ndarray,
    max_abs_corr: np.ndarray,
    day_counts: np.ndarray,
    high_abs_count: np.ndarray,
) -> pd.DataFrame:
    rows = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            days = int(day_counts[i, j])
            if days <= 0:
                continue
            rows.append(
                {
                    "factor_i": names[i],
                    "factor_j": names[j],
                    "mean_corr": mean_corr[i, j],
                    "mean_abs_corr": mean_abs_corr[i, j],
                    "max_abs_corr": max_abs_corr[i, j],
                    "days": days,
                    "high_abs_day_ratio": float(high_abs_count[i, j]) / days,
                }
            )
    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    summary = pd.DataFrame(rows)
    summary["abs_mean_corr"] = summary["mean_corr"].abs()
    summary = summary.sort_values(
        ["mean_abs_corr", "abs_mean_corr", "max_abs_corr"],
        ascending=[False, False, False],
    ).drop(columns=["abs_mean_corr"])
    return summary


def build_pair_summary(
    names: list[str],
    mean_corr: np.ndarray,
    mean_abs_corr: np.ndarray,
    max_abs_corr: np.ndarray,
    day_counts: np.ndarray,
    high_abs_count: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    summary = build_all_pair_table(names, mean_corr, mean_abs_corr, max_abs_corr, day_counts, high_abs_count)
    return summary.head(max(1, int(top_n)))


def build_factor_summary(
    specs: Sequence[FactorSpec],
    mean_abs_corr: np.ndarray,
    day_counts: np.ndarray,
    coverage_sum: np.ndarray,
    coverage_count: np.ndarray,
) -> pd.DataFrame:
    names = [spec.name for spec in specs]
    off_diag_abs = mean_abs_corr.copy()
    np.fill_diagonal(off_diag_abs, np.nan)
    off_diag_days = day_counts.copy().astype(float)
    np.fill_diagonal(off_diag_days, np.nan)
    rows = []
    coverage = safe_divide(coverage_sum, coverage_count)
    for idx, spec in enumerate(specs):
        row_abs = off_diag_abs[idx]
        if np.all(np.isnan(row_abs)):
            nearest_name = None
            max_abs = np.nan
        else:
            nearest_idx = int(np.nanargmax(row_abs))
            nearest_name = names[nearest_idx]
            max_abs = row_abs[nearest_idx]
        rows.append(
            {
                "index": spec.index,
                "factor": spec.name,
                "raw_path": spec.raw_path,
                "resolved_path": spec.resolved_path,
                "avg_coverage": coverage[idx],
                "mean_abs_corr_to_others": np.nanmean(row_abs),
                "max_abs_corr_to_other": max_abs,
                "nearest_factor": nearest_name,
                "correlated_pairs_observed": int(np.nansum(off_diag_days[idx] > 0)),
            }
        )
    return pd.DataFrame(rows)


def build_interpretation_tables(all_pairs: pd.DataFrame, factor_summary: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    stable = all_pairs[
        (all_pairs["mean_abs_corr"] >= args.interpret_abs_corr_threshold)
        & (all_pairs["high_abs_day_ratio"] >= args.interpret_high_day_ratio_threshold)
    ].sort_values(["high_abs_day_ratio", "mean_abs_corr", "days"], ascending=[False, False, False])

    inverse = all_pairs[
        (all_pairs["mean_corr"] <= -args.interpret_abs_corr_threshold)
        & (all_pairs["mean_abs_corr"] >= args.interpret_abs_corr_threshold)
    ].sort_values(["mean_corr", "mean_abs_corr"], ascending=[True, False])

    episodic = all_pairs[
        (all_pairs["max_abs_corr"] >= args.interpret_episodic_max_threshold)
        & (all_pairs["mean_abs_corr"] < args.interpret_episodic_mean_abs_max)
    ].sort_values(["max_abs_corr", "mean_abs_corr"], ascending=[False, True])

    low_coverage = factor_summary[
        factor_summary["avg_coverage"] < args.interpret_low_coverage_threshold
    ].sort_values(["avg_coverage", "max_abs_corr_to_other"], ascending=[True, False])

    return {
        "stable_duplicates": stable,
        "inverse_duplicates": inverse,
        "episodic_high_corr_pairs": episodic,
        "low_coverage_factors": low_coverage,
    }


def format_markdown_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.4f}"
    if isinstance(value, (np.floating,)):
        value = float(value)
        if not np.isfinite(value):
            return ""
        return f"{value:.4f}"
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str], max_rows: int) -> str:
    shown = df.loc[:, [column for column in columns if column in df.columns]].head(max_rows)
    if shown.empty:
        return "No rows.\n"
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join("---" for _ in shown.columns) + " |"
    rows = []
    for _, row in shown.iterrows():
        rows.append("| " + " | ".join(format_markdown_value(row[column]) for column in shown.columns) + " |")
    return "\n".join([header, separator, *rows]) + "\n"


def dataframe_records(df: pd.DataFrame, max_rows: int) -> list[dict]:
    return json.loads(df.head(max_rows).to_json(orient="records"))


def write_interpretation_outputs(
    output_dir: Path,
    all_pairs: pd.DataFrame,
    factor_summary: pd.DataFrame,
    args: argparse.Namespace,
    metadata: dict,
) -> None:
    tables = build_interpretation_tables(all_pairs, factor_summary, args)
    for name, table in tables.items():
        table.to_csv(output_dir / f"interpretation_{name}.csv", index=False)

    top_n = max(1, int(args.interpret_top_n))
    counts = {name: int(len(table)) for name, table in tables.items()}
    summary = {
        "metadata": metadata,
        "thresholds": {
            "stable_mean_abs_corr_min": args.interpret_abs_corr_threshold,
            "stable_high_abs_day_ratio_min": args.interpret_high_day_ratio_threshold,
            "inverse_mean_corr_max": -args.interpret_abs_corr_threshold,
            "inverse_mean_abs_corr_min": args.interpret_abs_corr_threshold,
            "episodic_max_abs_corr_min": args.interpret_episodic_max_threshold,
            "episodic_mean_abs_corr_max": args.interpret_episodic_mean_abs_max,
            "low_coverage_max": args.interpret_low_coverage_threshold,
        },
        "counts": counts,
        "top_rows": {name: dataframe_records(table, top_n) for name, table in tables.items()},
    }
    (output_dir / "interpretation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Factor Correlation Interpretation",
        "",
        "## Run Context",
        "",
        f"- Config: `{metadata['config']}`",
        f"- Factors: {metadata['factor_count']}",
        f"- Dates used: {metadata['used_date_count']} ({metadata['first_used_date']}..{metadata['last_used_date']})",
        f"- Method: `{metadata['method']}`",
        f"- Preprocess: `{metadata['preprocess']}`",
        f"- Universe: `{metadata['universe']}`",
        f"- Min valid instruments per pair/day: {metadata['min_valid']}",
        "",
        "## Signal Counts",
        "",
        f"- Stable duplicate-like pairs: {counts['stable_duplicates']}",
        f"- Reverse duplicate-like pairs: {counts['inverse_duplicates']}",
        f"- Episodic high-correlation pairs: {counts['episodic_high_corr_pairs']}",
        f"- Low-coverage factors: {counts['low_coverage_factors']}",
        "",
        "## Stable Duplicate-Like Pairs",
        "",
        (
            f"Criteria: `mean_abs_corr >= {args.interpret_abs_corr_threshold}` and "
            f"`high_abs_day_ratio >= {args.interpret_high_day_ratio_threshold}`. "
            "These pairs are consistently similar across time and are candidates for removal, downweighting, or orthogonalization."
        ),
        "",
        dataframe_to_markdown(
            tables["stable_duplicates"],
            ["factor_i", "factor_j", "mean_corr", "mean_abs_corr", "high_abs_day_ratio", "max_abs_corr", "days"],
            top_n,
        ),
        "## Reverse Duplicate-Like Pairs",
        "",
        (
            f"Criteria: `mean_corr <= {-args.interpret_abs_corr_threshold}` and "
            f"`mean_abs_corr >= {args.interpret_abs_corr_threshold}`. "
            "These pairs carry highly overlapping information with opposite sign."
        ),
        "",
        dataframe_to_markdown(
            tables["inverse_duplicates"],
            ["factor_i", "factor_j", "mean_corr", "mean_abs_corr", "high_abs_day_ratio", "max_abs_corr", "days"],
            top_n,
        ),
        "## Episodic High-Correlation Pairs",
        "",
        (
            f"Criteria: `max_abs_corr >= {args.interpret_episodic_max_threshold}` and "
            f"`mean_abs_corr < {args.interpret_episodic_mean_abs_max}`. "
            "These pairs are not usually redundant, but they become very similar in some periods; check for regime effects or data issues."
        ),
        "",
        dataframe_to_markdown(
            tables["episodic_high_corr_pairs"],
            ["factor_i", "factor_j", "mean_corr", "mean_abs_corr", "max_abs_corr", "high_abs_day_ratio", "days"],
            top_n,
        ),
        "## Low-Coverage Factors",
        "",
        (
            f"Criteria: `avg_coverage < {args.interpret_low_coverage_threshold}`. "
            "Correlation conclusions for these factors are less stable because fewer instruments contribute to their daily cross-sections."
        ),
        "",
        dataframe_to_markdown(
            tables["low_coverage_factors"],
            ["index", "factor", "avg_coverage", "mean_abs_corr_to_others", "max_abs_corr_to_other", "nearest_factor"],
            top_n,
        ),
    ]
    (output_dir / "interpretation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_heatmap(
    values: np.ndarray,
    names: list[str],
    path: Path,
    title: str,
    zmin: float,
    zmax: float,
    colorscale: str,
    colorbar_title: str = "value",
) -> None:
    import plotly.graph_objects as go

    size = max(800, min(2400, 20 * len(names)))
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=names,
            y=names,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            colorbar={"title": colorbar_title},
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, width=size, height=size, margin={"l": 160, "r": 40, "t": 70, "b": 160})
    fig.write_html(path, include_plotlyjs="cdn")


def write_pairwise_heatmaps(
    output_dir: Path,
    names: list[str],
    mean_corr: np.ndarray,
    mean_abs_corr: np.ndarray,
    max_abs_corr: np.ndarray,
    observed_days: np.ndarray,
    observed_day_ratio: np.ndarray,
    high_abs_ratio: np.ndarray,
    used_date_count: int,
) -> None:
    write_heatmap(mean_corr, names, output_dir / "mean_corr_heatmap.html", "Mean Daily Factor Correlation", -1.0, 1.0, "RdBu", "corr")
    write_heatmap(
        mean_abs_corr,
        names,
        output_dir / "mean_abs_corr_heatmap.html",
        "Mean Daily Absolute Factor Correlation",
        0.0,
        1.0,
        "Viridis",
        "abs corr",
    )
    write_heatmap(
        max_abs_corr,
        names,
        output_dir / "max_abs_corr_heatmap.html",
        "Maximum Daily Absolute Factor Correlation",
        0.0,
        1.0,
        "Viridis",
        "abs corr",
    )
    write_heatmap(
        high_abs_ratio,
        names,
        output_dir / "high_abs_day_ratio_heatmap.html",
        "High Absolute Correlation Day Ratio",
        0.0,
        1.0,
        "Magma",
        "ratio",
    )
    write_heatmap(
        observed_days.astype(float),
        names,
        output_dir / "observed_days_heatmap.html",
        "Observed Valid Days Per Factor Pair",
        0.0,
        float(max(used_date_count, 1)),
        "Blues",
        "days",
    )
    write_heatmap(
        observed_day_ratio,
        names,
        output_dir / "observed_day_ratio_heatmap.html",
        "Observed Valid Day Ratio Per Factor Pair",
        0.0,
        1.0,
        "Blues",
        "ratio",
    )


def write_outputs(
    output_dir: Path,
    specs: Sequence[FactorSpec],
    acc: CorrAccumulator,
    args: argparse.Namespace,
    config_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [spec.name for spec in specs]
    mean_corr = safe_divide(acc.sum_corr, acc.count_days)
    mean_abs_corr = safe_divide(acc.sum_abs_corr, acc.count_days)
    high_abs_ratio = safe_divide(acc.high_abs_count.astype(np.float64), acc.count_days)
    observed_day_ratio = safe_divide(
        acc.count_days.astype(np.float64),
        np.full_like(acc.count_days, max(len(acc.used_dates), 1), dtype=np.float64),
    )

    matrix_df(mean_corr, names).to_csv(output_dir / "mean_corr.csv")
    matrix_df(mean_abs_corr, names).to_csv(output_dir / "mean_abs_corr.csv")
    matrix_df(acc.max_abs_corr, names).to_csv(output_dir / "max_abs_corr.csv")
    matrix_df(acc.count_days.astype(float), names).to_csv(output_dir / "observed_days.csv")
    matrix_df(observed_day_ratio, names).to_csv(output_dir / "observed_day_ratio.csv")
    matrix_df(high_abs_ratio, names).to_csv(output_dir / "high_abs_day_ratio.csv")

    all_pairs = build_all_pair_table(
        names,
        mean_corr,
        mean_abs_corr,
        acc.max_abs_corr,
        acc.count_days,
        acc.high_abs_count,
    )
    all_pairs.to_csv(output_dir / "all_pairs.csv", index=False)

    pair_summary = build_pair_summary(
        names,
        mean_corr,
        mean_abs_corr,
        acc.max_abs_corr,
        acc.count_days,
        acc.high_abs_count,
        args.top_n,
    )
    pair_summary.to_csv(output_dir / "top_pairs.csv", index=False)

    factor_summary = build_factor_summary(specs, mean_abs_corr, acc.count_days, acc.coverage_sum, acc.coverage_count)
    factor_summary.to_csv(output_dir / "factor_summary.csv", index=False)

    pd.DataFrame(acc.skipped_dates, columns=["date", "reason"]).to_csv(output_dir / "skipped_dates.csv", index=False)

    metadata = {
        "config": str(config_path),
        "factor_count": len(specs),
        "used_date_count": len(acc.used_dates),
        "skipped_date_count": len(acc.skipped_dates),
        "first_used_date": min(acc.used_dates) if acc.used_dates else None,
        "last_used_date": max(acc.used_dates) if acc.used_dates else None,
        "method": args.method,
        "preprocess": args.preprocess,
        "universe": args.universe,
        "min_valid": args.min_valid,
        "high_corr_threshold": args.high_corr_threshold,
        "interpret_abs_corr_threshold": args.interpret_abs_corr_threshold,
        "interpret_high_day_ratio_threshold": args.interpret_high_day_ratio_threshold,
        "interpret_episodic_max_threshold": args.interpret_episodic_max_threshold,
        "interpret_episodic_mean_abs_max": args.interpret_episodic_mean_abs_max,
        "interpret_low_coverage_threshold": args.interpret_low_coverage_threshold,
        "date_step": args.date_step,
        "max_days": args.max_days,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_interpretation_outputs(output_dir, all_pairs, factor_summary, args, metadata)

    if not args.no_html:
        write_pairwise_heatmaps(
            output_dir,
            names,
            mean_corr,
            mean_abs_corr,
            acc.max_abs_corr,
            acc.count_days,
            observed_day_ratio,
            high_abs_ratio,
            len(acc.used_dates),
        )


def default_output_dir(config: dict) -> Path:
    return Path(config["combo"]["paths"]["output_dir"]) / "factor_corr"


def print_dry_run(output_dir: Path, specs: Sequence[FactorSpec], dates: Sequence[int], args: argparse.Namespace) -> None:
    print(f"[DRY-RUN] factors={len(specs)} dates={len(dates)} output_dir={output_dir}")
    print(f"[DRY-RUN] method={args.method} preprocess={args.preprocess} universe={args.universe}")
    if dates:
        print(f"[DRY-RUN] date_range={dates[0]}..{dates[-1]}")
    for spec in specs[:10]:
        print(f"[DRY-RUN] factor[{spec.index}] {spec.name} -> {spec.resolved_path}")
    if len(specs) > 10:
        print(f"[DRY-RUN] ... {len(specs) - 10} more factors")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_flag or args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (ORGANIZE_ROOT / config_path).resolve()

    config_module = import_organize_config()
    config = config_module.load_config(str(config_path))
    apply_cache_paths(config, config_path, args.cache_path)
    specs = build_factor_specs(config_path, config, args.factor_root)
    if len(specs) < 2:
        raise ValueError("at least two factors are required for correlation analysis")

    IndexMask = None
    Memmaper2 = None
    if args.dates is None or not args.dry_run:
        IndexMask, Memmaper2 = import_factorsim()
    dates = choose_dates(args, config, IndexMask) if args.dates is None else parse_explicit_dates(args.dates)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(config)
    if not output_dir.is_absolute():
        output_dir = (ORGANIZE_ROOT / output_dir).resolve()

    if args.dry_run:
        print_dry_run(output_dir, specs, dates, args)
        return

    assert Memmaper2 is not None
    cache = MemmapCache(Memmaper2)
    acc = CorrAccumulator.create([spec.name for spec in specs])
    progress_every = max(1, int(args.progress_every))
    start_time = time.time()
    print(
        f"[FACTOR-CORR] factors={len(specs)} dates={len(dates)} method={args.method} "
        f"preprocess={args.preprocess} universe={args.universe}"
    )

    for pos, ds in enumerate(dates, start=1):
        try:
            matrix = load_factor_matrix(cache, specs, ds)
            universe_mask = load_universe_mask(cache, config, args.universe, ds, matrix.shape[0])
            matrix = matrix.copy()
            matrix[~universe_mask, :] = np.nan
            corr, pair_counts, coverage = compute_day_corr(matrix, args.method, args.preprocess)
            acc.add_day(ds, corr, pair_counts, coverage, args.min_valid, args.high_corr_threshold)
        except Exception as exc:
            if args.skip_bad_days:
                acc.add_skip(ds, f"{type(exc).__name__}: {exc}")
                continue
            raise

        if pos % progress_every == 0 or pos == len(dates):
            elapsed = time.time() - start_time
            rate = pos / elapsed if elapsed > 0 else math.nan
            print(f"[FACTOR-CORR] processed {pos}/{len(dates)} dates, used={len(acc.used_dates)}, rate={rate:.2f} dates/s")

    if not acc.used_dates:
        raise RuntimeError("no dates were successfully processed")

    write_outputs(output_dir, specs, acc, args, config_path)
    print(f"[FACTOR-CORR] report written to {output_dir}")


if __name__ == "__main__":
    main()
