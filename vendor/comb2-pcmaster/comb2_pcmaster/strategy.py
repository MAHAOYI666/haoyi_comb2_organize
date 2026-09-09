from abc import ABCMeta, abstractmethod


class StrategyBase(metaclass=ABCMeta):
    def __init__(self, strategy_config: dict, dataloader):
        self.config = strategy_config
        self.dataloader = dataloader

    @abstractmethod
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
        raise NotImplementedError
