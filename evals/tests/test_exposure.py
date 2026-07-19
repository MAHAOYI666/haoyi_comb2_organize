from __future__ import annotations

import numpy as np
import pandas as pd

from comb_eval import exposure


def test_cap_corr_uses_same_day_ranked_market_cap(monkeypatch) -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03"])
    codes = [f"{code:06d}" for code in range(1, 7)]
    signal = pd.DataFrame(
        [[-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]] * 2,
        index=dates,
        columns=codes,
    )
    market_cap = pd.DataFrame(
        [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
        index=[20200102, 20200103],
        columns=codes,
    )
    read_calls = []

    def fake_read(path, start_ds, end_ds, df_type):
        read_calls.append((str(path), start_ds, end_ds, df_type))
        return market_cap

    monkeypatch.setattr(exposure, "read_cache_array", fake_read)

    result = exposure.compute_cap_corr(signal, "/cache")

    np.testing.assert_allclose(result.to_numpy(), [1.0, -1.0])
    assert read_calls == [
        ("/cache/AshareCache/1d_DailyFdm/DailyFdm.mkt_cap", 20200102, 20200103, True),
    ]


def test_cap_corr_summary_all_row_uses_all_valid_days() -> None:
    cap_corr = pd.Series(
        [1.0, -1.0, -1.0],
        index=pd.to_datetime(["2020-01-02", "2021-01-04", "2021-01-05"]),
        name="cap_corr",
    )

    summary = exposure.summarize_cap_corr(cap_corr)

    assert summary.loc["ALL", "days"] == 3
    assert np.isclose(summary.loc["ALL", "cap_corr.avg"], -1.0 / 3.0)
