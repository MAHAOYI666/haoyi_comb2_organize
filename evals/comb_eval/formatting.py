from __future__ import annotations

from collections.abc import Mapping
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
    return f"{value:.2f}"
