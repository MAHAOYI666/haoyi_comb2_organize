from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

TRADING_DAYS = 250
PNLSUPER_TRADING_DAYS = 242
PNLSUPER_INTRADAY_INTERVALS = 54


@dataclass(frozen=True)
class MetricResult:
    module: str
    table: pd.DataFrame
    metadata: dict[str, Any]
