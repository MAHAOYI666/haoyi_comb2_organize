from __future__ import annotations

import torch

from comb2 import ComboTrainDataset


class ResearchDataset(ComboTrainDataset):
    """eg-torch train dataset scaffold.

    Default sample format is kept as (idx, x, y, w), which is what model.py expects.
    """

    def _build_validinsts(self) -> torch.Tensor:
        return super()._build_validinsts()

    def __getitem__(self, idx: int):
        return super().__getitem__(idx)
