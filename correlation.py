from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


try:
    from .io import read_matrix, read_table
    from .pnl import sample_ir
except ImportError:

    def read_table(
        path_or_df: str | Path | pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        df = _read_dataframe(path_or_df)
        return _filter_date_range(df, start=start, end=end)

    def read_matrix(
        path_or_df: str | Path | pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        if isinstance(path_or_df, (str, Path)) and Path(path_or_df).suffix.lower() in {".pt", ".pth"}:
            df = _read_alpha_history(path_or_df)
        else:
            df = _read_dataframe(path_or_df)
        return _filter_date_range(df, start=start, end=end)

    def sample_ir(series: pd.Series) -> float:
        values = series.astype(float).dropna()
        std = float(values.std(ddof=1))
        if values.empty or std == 0 or pd.isna(std):
            return np.nan
        return float(values.mean() / std)


def pnl_correlation(candidate: str | Path | pd.DataFrame, pool: str | Path | pd.DataFrame, column: str = "pnl", start: str | None = None, end: str | None = None) -> dict[str, float | int | str]:
    candidate_df = read_table(candidate, start=start, end=end)
    pool_df = read_table(pool, start=start, end=end)
    aligned = pd.concat([candidate_df[column].rename("candidate"), pool_df[column].rename("pool")], axis=1, join="inner").dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN observations for pnl correlation.")
    return {
        "corr": float(aligned["candidate"].corr(aligned["pool"])),
        "n_obs": int(len(aligned)),
        "start": _format_date(aligned.index.min()),
        "end": _format_date(aligned.index.max()),
    }


def pnl_pool_correlation(candidate: str | Path | pd.DataFrame, pool_paths: list[str | Path | pd.DataFrame], column: str = "pnl", top: int = 5, ratio: float = 1.5, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    candidate_df = read_table(candidate, start=start, end=end)
    candidate_series = candidate_df[column].astype(float).rename("candidate")
    rows = []
    candidate_avg = float(candidate_series.mean())
    candidate_ir = sample_ir(candidate_series)
    for idx, pool_path in enumerate(pool_paths):
        pool_df = read_table(pool_path, start=start, end=end)
        pool_series = pool_df[column].astype(float).rename("pool")
        aligned = pd.concat([candidate_series, pool_series], axis=1, join="inner").dropna()
        corr = float(aligned["candidate"].corr(aligned["pool"])) if len(aligned) >= 2 else np.nan
        pool_avg = float(aligned["pool"].mean()) if not aligned.empty else np.nan
        pool_ir = sample_ir(aligned["pool"]) if not aligned.empty else np.nan
        rows.append({
            "name": _pool_name(pool_path, idx),
            "corr": corr,
            "n_obs": int(len(aligned)),
            "pool_avg": pool_avg,
            "pool_ir": pool_ir,
            "avg_ratio": _clipped_ratio(candidate_avg, pool_avg, corr),
            "ir_ratio": _clipped_ratio(candidate_ir, pool_ir, corr),
            "avg_beat": _beat(candidate_avg, pool_avg, corr, ratio),
            "ir_beat": _beat(candidate_ir, pool_ir, corr, ratio),
        })
    table = pd.DataFrame(rows).set_index("name")
    if table.empty:
        return table
    sorted_corr = table["corr"].sort_values(ascending=False)
    summary = pd.DataFrame({
        "value": {
            "maxcorr": float(table["corr"].max()),
            "avgcorr": float(table["corr"].mean()),
            f"topcorr{top}": float(sorted_corr.head(top).mean()),
            f"avgRatio{top}": float(table.loc[sorted_corr.head(top).index, "avg_ratio"].mean()),
            f"irRatio{top}": float(table.loc[sorted_corr.head(top).index, "ir_ratio"].mean()),
            "avgBeatByNum": int(table["avg_beat"].sum()),
            "irBeatByNum": int(table["ir_beat"].sum()),
        }
    })
    summary.index.name = "metric"
    table.attrs["summary"] = summary
    return table.round(6)


def daily_matrix_correlation(
    candidate: str | Path | pd.DataFrame,
    pool: str | Path | pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    corr_days: int | None = None,
    min_valid: int = 2,
) -> pd.DataFrame:
    candidate_df = read_matrix(candidate, start=start, end=end)
    pool_df = read_matrix(pool, start=start, end=end)
    dates = candidate_df.index.intersection(pool_df.index).sort_values()
    if corr_days is not None:
        dates = dates[-corr_days:]
    columns = candidate_df.columns.intersection(pool_df.columns)
    rows = []
    for date in dates:
        left = candidate_df.loc[date, columns].astype(float)
        right = pool_df.loc[date, columns].astype(float)
        valid = left.notna() & right.notna() & (left != 0) & (right != 0)
        corr = float(left[valid].corr(right[valid])) if int(valid.sum()) >= min_valid else np.nan
        rows.append({"date": date, "corr": corr, "n_inst": int(valid.sum())})
    return pd.DataFrame(rows).set_index("date")


def position_correlation(
    candidate: str | Path | pd.DataFrame,
    pool: str | Path | pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    corr_days: int | None = None,
    min_valid: int = 1000,
) -> dict[str, float | int | str]:
    daily = daily_matrix_correlation(candidate, pool, start=start, end=end, corr_days=corr_days, min_valid=min_valid)
    valid = daily["corr"].dropna()
    if valid.empty:
        raise ValueError("No overlapping nonzero matrix rows with enough valid instruments.")
    return {
        "avg_corr": float(valid.mean()),
        "max_corr": float(valid.max()),
        "min_corr": float(valid.min()),
        "n_days": int(len(valid)),
        "corr_days": int(corr_days) if corr_days is not None else int(len(daily)),
        "min_valid": int(min_valid),
        "start": _format_date(valid.index.min()),
        "end": _format_date(valid.index.max()),
    }


def matrix_correlation(
    candidate: str | Path | pd.DataFrame,
    pool: str | Path | pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    corr_days: int | None = None,
    min_valid: int = 2,
) -> dict[str, float | int | str]:
    return position_correlation(candidate, pool, start=start, end=end, corr_days=corr_days, min_valid=min_valid)


def _pool_name(path: str | Path | pd.DataFrame, idx: int) -> str:
    if isinstance(path, pd.DataFrame):
        return f"pool_{idx}"
    return Path(path).stem


def _clipped_ratio(candidate_value: float, pool_value: float, corr: float) -> float:
    if pd.isna(candidate_value) or pd.isna(pool_value) or pd.isna(corr) or pool_value == 0 or corr == 0:
        return np.nan
    return float(np.clip(candidate_value / pool_value / corr, -5, 5))


def _beat(candidate_value: float, pool_value: float, corr: float, ratio: float) -> bool:
    if pd.isna(candidate_value) or pd.isna(pool_value) or pd.isna(corr) or pool_value == 0:
        return False
    return (candidate_value / pool_value) <= corr * ratio


def _read_alpha_history(path: str | Path) -> pd.DataFrame:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "alpha_history" in payload and isinstance(payload["alpha_history"], dict):
        payload = payload["alpha_history"]
    if not isinstance(payload, dict):
        raise TypeError("alpha_history.pt must contain a dict: date -> alpha vector.")

    rows = []
    dates = []
    expected_size = None
    for date, value in payload.items():
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        row = np.asarray(value, dtype=np.float64).reshape(-1)
        if expected_size is None:
            expected_size = row.size
        elif row.size != expected_size:
            raise ValueError(f"Inconsistent alpha vector length at {date}: expected {expected_size}, got {row.size}")
        dates.append(_coerce_date(date))
        rows.append(row)
    if not rows:
        raise ValueError(f"No alpha rows found in {path}")
    return pd.DataFrame(np.vstack(rows), index=pd.Index(dates, name="date")).sort_index()


def _read_dataframe(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df.copy()
    path = Path(path_or_df)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt", ".tsv", ""}:
        sep = "\t" if suffix in {".txt", ".tsv"} else None
        df = pd.read_csv(path, sep=sep, engine="python")
        if len(df.columns) and str(df.columns[0]).lower() in {"date", "ds", "datetime", "index", "unnamed: 0"}:
            df = df.set_index(df.columns[0])
        return df
    raise ValueError(f"Unsupported file suffix for {path}")


def _filter_date_range(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if start is None and end is None:
        return df
    date_values = pd.Series([_date_int(value) for value in df.index], index=df.index, dtype="float64")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= date_values >= int(start)
    if end is not None:
        mask &= date_values <= int(end)
    return df.loc[mask.to_numpy()]


def _date_int(value: Any) -> int:
    if isinstance(value, pd.Timestamp):
        return int(value.strftime("%Y%m%d"))
    if hasattr(value, "strftime"):
        return int(value.strftime("%Y%m%d"))
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) >= 8 and text[:8].isdigit():
        return int(text[:8])
    return int(pd.to_datetime(value).strftime("%Y%m%d"))


def _coerce_date(value: Any) -> Any:
    try:
        return _date_int(value)
    except Exception:
        return value


def _format_date(value: Any) -> str:
    try:
        return f"{_date_int(value):08d}"
    except Exception:
        return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate pos corr for two alpha_history.pt or matrix files.")
    parser.add_argument("candidate")
    parser.add_argument("pool")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--corr-days", type=int, default=None)
    parser.add_argument("--min-valid", type=int, default=1000)
    parser.add_argument("--daily-output", default=None)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--print-daily", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = position_correlation(
        args.candidate,
        args.pool,
        start=args.start,
        end=args.end,
        corr_days=args.corr_days,
        min_valid=args.min_valid,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.print_daily or args.daily_output:
        daily = daily_matrix_correlation(
            args.candidate,
            args.pool,
            start=args.start,
            end=args.end,
            corr_days=args.corr_days,
            min_valid=args.min_valid,
        )
        if args.print_daily:
            print(daily.to_string())
        if args.daily_output:
            daily_output = Path(args.daily_output)
            daily_output.parent.mkdir(parents=True, exist_ok=True)
            daily.to_csv(daily_output)
            print(f"daily_output={daily_output}")

    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"json_output={json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
