from __future__ import annotations

import pandas as pd

from comb2_pcmaster.strategy import StrategyBase


class AlphaStrategy(StrategyBase):
    def generate_positions(self, signals, last_hold):
        weights = signals.replace([float("inf"), float("-inf")], pd.NA).dropna()
        weights = weights[weights > 0]
        if weights.empty:
            return pd.Series(dtype=float)
        weights = weights / weights.sum()
        weights.index.name = None
        return weights.astype(float)
