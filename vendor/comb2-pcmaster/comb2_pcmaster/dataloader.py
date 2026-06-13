from pathlib import Path

import pandas as pd
from comb2_simbase import IndexMask, Memmaper2


class DataLoader:
    def __init__(self, signal_path: str = "", cache_path: str = ""):
        self.signal_path = signal_path
        self.cache_path = Path(cache_path)
        self.ashare_cache_path = self.cache_path / "AshareCache"
        self.trade_date = sorted(IndexMask().date)
        self.date = None

    def get_signals(self) -> pd.DataFrame:
        if not self.signal_path:
            raise ValueError("signal_path is empty")
        return pd.read_parquet(self.signal_path)

    def get_preclose(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.pre_close")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_vwap(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_IntraVwap" / "IntraVwap.VwapBegin30")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_open(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.open")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_close(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.close")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_market_cap(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyFdm" / "DailyFdm.mkt_cap")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_suspend(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_StockMask2" / "StockMask2.SuspendStock")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_limit(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_StockMask2" / "StockMask2.LimitMask")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_base(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_StockMask2" / "StockMask2.BaseUnivMask")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]
