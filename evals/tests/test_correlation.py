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


def test_sample_index_survives_parquet_csv_and_day_filters(tmp_path):
    from comb_eval.io import read_matrix
    points = pd.MultiIndex.from_product(
        [[20240102,20240103,20240104], [100000,110000]], names=["date","time"]
    )
    values = pd.DataFrame(np.arange(24).reshape(6,4) + 1., index=points, columns=["000001","000002","000003","000004"])
    parquet = tmp_path / "alpha.parquet"
    csv = tmp_path / "positions.csv"
    values.to_parquet(parquet)
    values.to_csv(csv)
    left = read_matrix(parquet, start="20240103", end="20240104")
    right = read_matrix(csv, start="20240103", end="20240104")
    pd.testing.assert_frame_equal(left, right)
    assert len(left) == 4 and left.index[0].hour == 10 and left.index[-1].hour == 11
    result = matrix_correlation(parquet, csv, corr_days=1, min_valid=2)
    assert result["n_days"] == 1 and result["n_samples"] == 2
    assert np.isclose(result["avg_corr"], 1.)
