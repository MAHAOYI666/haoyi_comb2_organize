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
        buyable_mask,
        market_sellable_mask,
        target_stock_amount,
        executed_turnover_today,
    ):
        buyable = buyable_mask.reindex(signals.index)
        market_sellable = market_sellable_mask.reindex(signals.index)
        assert buyable.notna().all() and market_sellable.notna().all()
        buyable = buyable.astype(bool)
        market_sellable = market_sellable.astype(bool)
        current = sellable_amount.reindex(signals.index) + locked_amount.reindex(signals.index)
        frozen_amount = float(current[~market_sellable].sum())
        available_target = max(float(target_stock_amount) - frozen_amount, 0.0)
        weights = pd.to_numeric(signals[buyable], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        target = pd.Series(0.0, index=signals.index, dtype=float)
        if not weights.empty:
            weights = weights - weights.median()
            weights = weights[weights > 0]
            if not weights.empty:
                target.loc[weights.index] = weights / weights.sum()
        delta = target * available_target - current
        return pd.DataFrame(
            {
                "buy_amount": delta.clip(lower=0.0).where(buyable, 0.0),
                "sell_amount": (-delta).clip(lower=0.0).where(market_sellable, 0.0),
            },
            index=signals.index,
        )
