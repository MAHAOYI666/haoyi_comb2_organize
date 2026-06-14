from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import pandas as pd


def output_frame_to_text(frame: pd.DataFrame) -> str:
    return frame.to_string(float_format=_format_float)


def output_dict_to_lines(values: Mapping[str, Any]) -> str:
    return "\n".join(f"{key}: {_format_value(value)}" for key, value in values.items())


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return _format_float(value)
    return value


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    abs_value = abs(float(value))
    if abs_value == 0:
        return "0.00"
    if abs_value < 1:
        decimal_places = max(2, -math.floor(math.log10(abs_value)) + 1)
        return f"{value:.{decimal_places}f}"
    return f"{value:.2f}"
