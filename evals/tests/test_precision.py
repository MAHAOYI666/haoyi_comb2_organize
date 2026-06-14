from __future__ import annotations

import numpy as np
import pandas as pd

from comb_eval.formatting import output_frame_to_text
from comb_eval.pnl import summarize_pnl


def test_summarize_pnl_keeps_full_precision_until_output() -> None:
    pnl = pd.DataFrame(
        {
            "pnl": [1.0, 2.0],
            "long": [100.0, 100.0],
            "short": [-100.0, -100.0],
            "sh_hld": [200.0, 200.0],
            "sh_trd": [10.0, 10.0],
            "n_long": [3.0, 3.0],
            "n_short": [2.0, 2.0],
        },
        index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
    )

    result = summarize_pnl(pnl)

    ir = result.table.loc["20200101-20200102", "ir"]
    assert np.isclose(ir, 2.1213203435596424)
    assert ir != 2.12
    assert "2.12" in output_frame_to_text(result.table)


def test_output_frame_uses_effective_decimal_places_for_small_values() -> None:
    frame = pd.DataFrame(
        {
            "small": [0.006789],
            "negative_small": [-0.0004567],
            "normal": [12.345],
        },
        index=["row"],
    )

    text = output_frame_to_text(frame)

    assert "0.0068" in text
    assert "-0.00046" in text
    assert "12.35" in text
    assert "0.01" not in text
