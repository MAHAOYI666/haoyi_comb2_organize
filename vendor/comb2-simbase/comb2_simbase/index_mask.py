from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .np_utils import search_sorted_left_idx, search_sorted_right_idx


class IndexMask:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        base = Path(__file__).resolve().parent / "index_mask" / "memmap_mask"
        self._date = np.load(base / "DateRange.FactorSim.npy")
        self._time = np.load(base / "TimeRange.FactorSim.npy")
        self._code = np.load(base / "CodeRange.FactorSim.npy", allow_pickle=True)
        self._code_dict = {str(code).zfill(6): idx for idx, code in enumerate(self._code)}
        self._initialized = True

    @property
    def date(self):
        return self._date

    @property
    def time(self):
        return self._time

    @property
    def code(self):
        return self._code

    @property
    def didx(self):
        return np.arange(self.date.shape[0])

    @property
    def tidx(self):
        return np.arange(self.time.shape[0])

    @property
    def cidx(self):
        return np.arange(self.code.shape[0])

    def last_n_day(self, date, last_n):
        return self.didx2date(max(search_sorted_left_idx(self.date, int(date)) - int(last_n), 0))

    def last_n_day_left_side(self, date, last_n):
        return self.last_n_day(date, last_n)

    def last_n_day_right_side(self, date, last_n):
        return self.didx2date(max(search_sorted_right_idx(self.date, int(date)) - int(last_n), 0))

    def intv_tidx(self, n):
        n = int(n)
        if n < -1:
            raise ValueError("n should be positive integer or -1")
        if n == 1:
            return self.tidx
        if n == -1:
            return self.tidx[[-1]]
        return np.append(self.tidx[::n], 238)

    def intv_time(self, n):
        return self.time[self.intv_tidx(n)]

    def intv_trade_day(self, start_date, end_date):
        l_idx = search_sorted_right_idx(self.date, int(start_date))
        r_idx = search_sorted_left_idx(self.date, int(end_date))
        return self.date[l_idx : r_idx + 1]

    def date2didx(self, date):
        return search_sorted_left_idx(self.date, int(date))

    def time2tidx(self, time):
        return search_sorted_left_idx(self.time, int(time))

    def code2cidx(self, code):
        normalized = str(code).zfill(6)
        try:
            return self._code_dict[normalized]
        except KeyError as exc:
            raise ValueError(f"Do not find {code} in IndexMask!") from exc

    def didx2date(self, didx):
        try:
            return self.date[int(didx)]
        except Exception as exc:
            raise ValueError(f"Do not find {didx} in IndexMask!") from exc

    def tidx2time(self, tidx):
        try:
            return self.time[int(tidx)]
        except Exception as exc:
            raise ValueError(f"Do not find {tidx} in IndexMask!") from exc

    def cidx2code(self, cidx):
        try:
            return self.code[int(cidx)]
        except Exception as exc:
            raise ValueError(f"Do not find {cidx} in IndexMask!") from exc

    def datetime_to_min_flag(self, datetime_series):
        time_idx = datetime_series.values % 1000000000
        date = pd.to_datetime(str(datetime_series.iloc[0])[:8], format="%Y%m%d")
        rounded = pd.to_datetime(datetime_series, format="%Y%m%d%H%M%S%f").dt.ceil("1min").values
        rounded[time_idx <= 93000000] = date.replace(hour=9, minute=30, second=0, microsecond=0)
        rounded[(112900000 < time_idx) & (time_idx <= 130000000)] = date.replace(hour=11, minute=30, second=0, microsecond=0)
        rounded[time_idx > 145700000] = date.replace(hour=15, minute=0, second=0, microsecond=0)
        return pd.Series(rounded)
