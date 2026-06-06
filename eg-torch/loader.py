from __future__ import annotations

import torch

from comb2 import ComboDataLoader
from src.op_utils import cs_zscore, nan_to_num, truncate


class ResearchLoader(ComboDataLoader):
    """eg-torch data loader scaffold.

    Default flow:
    gen_feature(ds) -> build_raw_feature(ds) -> preprocess_feature(feature, ds)
    gen_label(ds, ret_days) -> preprocess_label(label_values, valid_mask, ds, ret_days)
    load_feature_window(end_ds, ts_days) -> process_feature_window(feature_window)
    """

    def gen_feature(self, ds: int) -> torch.Tensor:
        return super().gen_feature(ds)

    def preprocess_feature(self, feature: torch.Tensor, ds: int) -> torch.Tensor:
        feature = cs_zscore(feature.transpose(0, 1)).transpose(0, 1)
        feature = truncate(feature, -4.0, 4.0)
        return nan_to_num(feature, 0.0).to(self.dtype)

    def gen_label(self, ds: int, ret_days: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return super().gen_label(ds, ret_days=ret_days)

    def preprocess_label(
        self,
        label_values: torch.Tensor,
        valid_mask: torch.Tensor,
        ds: int,
        ret_days: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().preprocess_label(label_values, valid_mask, ds, ret_days=ret_days)

    def _feature_available_mask(self, feature_window: torch.Tensor) -> torch.Tensor:
        return super()._feature_available_mask(feature_window)

    def process_feature_window(self, feature_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().process_feature_window(feature_window)
