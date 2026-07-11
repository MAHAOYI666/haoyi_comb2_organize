from __future__ import annotations

import numpy as np
import pandas as pd

from comb_eval.correlation import daily_top_overlap, matrix_correlation


def test_matrix_correlation_reports_top_long_overlap() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    columns = ["000001", "000002", "000003", "000004"]
    left = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 3.0, 5.0, 7.0],
            [8.0, 6.0, 4.0, 2.0],
        ],
        index=dates,
        columns=columns,
    )
    right = pd.DataFrame(
        [
            [1.0, 2.0, 4.0, 3.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 3.0, 5.0, 7.0],
        ],
        index=dates,
        columns=columns,
    )

    daily = daily_top_overlap(left, right, min_valid=2, top_pct=50.0)
    result = matrix_correlation(left, right, min_valid=2, top_pct=50.0)

    assert daily["long_overlap"].tolist() == [1.0, 1.0, 0.0]
    assert result["long_top_pct"] == 50.0
    assert np.isclose(result["avg_long_overlap"], 2.0 / 3.0)
    assert result["max_long_overlap"] == 1.0
    assert result["min_long_overlap"] == 0.0
    assert result["n_overlap_days"] == 3
