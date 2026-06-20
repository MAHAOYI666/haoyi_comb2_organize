from __future__ import annotations

import numpy as np
import pandas as pd

from comb2_pcmaster.strategy import StrategyBase


class AlphaStrategy(StrategyBase):
    def generate_positions(self, signals, last_hold):
        weights = pd.to_numeric(signals, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if weights.empty:
            return pd.Series(dtype=float)

        weights = weights - weights.median()
        weights = weights[weights > 0]
        if weights.empty:
            return pd.Series(dtype=float)
        weights = weights / weights.sum()
        weights.index.name = None
        return weights.astype(float)
