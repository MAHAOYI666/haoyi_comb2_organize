from __future__ import annotations

import numpy as np
import pandas as pd

from comb_eval.formatting import output_frame_to_text
from comb_eval.pnl import summarize_pnl
from comb_eval.report import calculate_decile_daily_pnls


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


def test_summarize_pnl_omits_periods_without_finite_performance_metrics() -> None:
    pnl = pd.DataFrame(
        {
            "pnl": [0.0, 1.0, 2.0],
            "long": [0.0, 100.0, 100.0],
            "short": [0.0, -100.0, -100.0],
            "sh_hld": [0.0, 200.0, 200.0],
            "sh_trd": [0.0, 10.0, 10.0],
            "n_long": [0.0, 3.0, 3.0],
            "n_short": [0.0, 2.0, 2.0],
        },
        index=pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"]),
    )

    result = summarize_pnl(pnl)

    assert "20191231-20191231" not in result.table.index
    assert result.table.loc["ALL", "days"] == 2
    assert np.isfinite(result.table.loc["ALL", "ret_pct"])


def test_decile_backtest_ignores_constant_signal_rows() -> None:
    signal = pd.DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        index=pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"]),
        columns=["000001", "000002", "000003", "000004"],
    )
    label = pd.DataFrame(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.01, 0.02, 0.03, 0.04],
            [0.02, 0.03, 0.04, 0.05],
        ],
        index=signal.index,
        columns=signal.columns,
    )

    decile_daily = calculate_decile_daily_pnls(signal, label, booksize=100.0)
    q10_summary = summarize_pnl(decile_daily["Q10"]).table

    assert decile_daily["Q10"].loc[pd.Timestamp("2019-12-31"), "long"] == 0.0
    assert "20191231-20191231" not in q10_summary.index
    assert "20200102-20200103" in q10_summary.index
