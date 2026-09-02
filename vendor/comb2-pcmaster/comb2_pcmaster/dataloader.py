from pathlib import Path
import sys

import pandas as pd

SIMBASE_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "comb2-simbase"
if str(SIMBASE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMBASE_ROOT))

from comb2_simbase import IndexMask, Memmaper2
from comb2_simbase.cache_layout import (
    BASE_UNIVERSE_MASK_NAME,
    FILTERED_MASK_NAME,
    SUSPEND_MASK_NAME,
    ashare_cache_path,
    stock_mask_path,
)


class DataLoader:
    def __init__(self, signal_path: str = "", cache_path: str = ""):
        self.signal_path = signal_path
        self.cache_path = Path(cache_path)
        self.ashare_cache_path = ashare_cache_path(self.cache_path)
        self.trade_date = sorted(IndexMask().date)
        self.date = None

    def get_signals(self) -> pd.DataFrame:
        if not self.signal_path:
            raise ValueError("signal_path is empty")
        return pd.read_parquet(self.signal_path)

    def get_preclose(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.pre_close")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_vwap(self, start_ds, end_ds, snap_ti=None) -> pd.DataFrame:
        name = "IntraVwap.VwapBegin30" if snap_ti is None else f"IntraVwap.Vwap30.{int(snap_ti):06d}"
        return Memmaper2(str(self.ashare_cache_path / "1d_IntraVwap" / name)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_open(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.open")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_close(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.close")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_returns(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.pct_chg")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_index_weight(self, start_ds, end_ds, ts_code="000905.SH") -> pd.DataFrame:
        path = self.ashare_cache_path / "1d_IndexWeight" / f"IndexWeight.{ts_code}"
        return Memmaper2(str(path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_trade_universe(self, name, start_ds, end_ds) -> pd.DataFrame:
        path = self.ashare_cache_path / "1d_TradeUniverse" / f"TradeUniverse.{name}"
        return Memmaper2(str(path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_stock_mask(self, name, start_ds, end_ds) -> pd.DataFrame:
        path = self.ashare_cache_path / "1d_StockMask2" / f"StockMask2.{name}"
        return Memmaper2(str(path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_industry(self, name, start_ds, end_ds) -> pd.DataFrame:
        path = self.ashare_cache_path / "1d_SwIndMask" / f"SwIndMask.{name}"
        return Memmaper2(str(path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_barra_style(self, name, start_ds, end_ds) -> pd.DataFrame:
        filename = str(name).upper()
        path = self.ashare_cache_path / "1d_BarraCNE5" / f"BarraCNE5.{filename}"
        return Memmaper2(str(path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_market_cap(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyFdm" / "DailyFdm.mkt_cap")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_amount(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(self.ashare_cache_path / "1d_DailyKline" / "DailyKline.amount")).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_slippage(self, start_ds, end_ds) -> pd.DataFrame:
        path = self.ashare_cache_path / "1d_DailySpread" / "DailySpread.spread_slippage_new"
        return Memmaper2(str(path)).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_suspend(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(stock_mask_path(self.cache_path, SUSPEND_MASK_NAME))).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_limit(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(stock_mask_path(self.cache_path, FILTERED_MASK_NAME))).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]

    def get_base(self, start_ds, end_ds) -> pd.DataFrame:
        return Memmaper2(str(stock_mask_path(self.cache_path, BASE_UNIVERSE_MASK_NAME))).load(start_ds=start_ds, end_ds=end_ds, df_type=True).dloc[:]
