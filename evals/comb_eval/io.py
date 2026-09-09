from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DateLike = str | int | pd.Timestamp


def coerce_date_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        assert index.nlevels == 2, "sample index must contain date and time"
        dates = coerce_date_index(index.get_level_values(0))
        times = index.get_level_values(1).astype(int)
        seconds = (times // 10000) * 3600 + (times // 100 % 100) * 60 + times % 100
        return dates + pd.to_timedelta(seconds, unit="s")
    if isinstance(index, pd.DatetimeIndex):
        return index
    values = index.astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    return pd.to_datetime(values, format="%Y%m%d", errors="coerce")


def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df = df.set_index(["date", "time"] if "time" in df.columns else "date")
    elif "Date" in df.columns:
        df = df.set_index("Date")
    elif "Unnamed: 0" in df.columns:
        df = df.set_index("Unnamed: 0")
    df.index = coerce_date_index(df.index)
    df = df[df.index.notna()]
    return df.sort_index()


def filter_dates(df: pd.DataFrame, start: DateLike | None = None, end: DateLike | None = None) -> pd.DataFrame:
    if start is not None:
        df = df[df.index.normalize() >= pd.to_datetime(str(start)).normalize()]
    if end is not None:
        df = df[df.index.normalize() <= pd.to_datetime(str(end)).normalize()]
    return df


def read_table(path: str | Path | pd.DataFrame, start: DateLike | None = None, end: DateLike | None = None) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        file_path = Path(path)
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, sep=None, engine="python")
    return filter_dates(normalize_date_index(df), start, end)


def read_cache_array(path: str | Path, start_ds: str | int, end_ds: str | int, df_type: object):
    simbase_root = Path(__file__).resolve().parents[2] / "vendor" / "comb2-simbase"
    if str(simbase_root) not in sys.path:
        sys.path.insert(0, str(simbase_root))
    from comb2_simbase import Memmaper2

    return Memmaper2(str(path)).load(start_ds, end_ds, df_type)[:]


def read_matrix(path: str | Path | pd.DataFrame, start: DateLike | None = None, end: DateLike | None = None) -> pd.DataFrame:
    df = read_table(path, start=start, end=end)
    return df.astype(float)
