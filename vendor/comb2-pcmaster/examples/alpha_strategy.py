from __future__ import annotations

import numpy as np
import pandas as pd

from comb2_pcmaster.strategy import StrategyBase


class AlphaStrategy(StrategyBase):
    def generate_orders(
        self,
        signals,
        sellable_amount,
        locked_amount,
        target_stock_amount,
        executed_turnover_today,
    ):
        weights = pd.to_numeric(signals, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        target = pd.Series(0.0, index=signals.index, dtype=float)
        if not weights.empty:
            weights = weights - weights.median()
            weights = weights[weights > 0]
            if not weights.empty:
                target.loc[weights.index] = weights / weights.sum()
        current = sellable_amount.reindex(signals.index) + locked_amount.reindex(signals.index)
        delta = target * float(target_stock_amount) - current
        return pd.DataFrame(
            {
                "buy_amount": delta.clip(lower=0.0),
                "sell_amount": (-delta).clip(lower=0.0),
            },
            index=signals.index,
        )
