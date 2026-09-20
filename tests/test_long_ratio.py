from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from comb2_pcmaster.backtest import _adjust_alpha_by_long_ratio


def test_long_ratio_shifts_by_cross_sectional_threshold():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    adjusted = _adjust_alpha_by_long_ratio(
        values,
        pd.Series(np.ones(len(values), dtype=bool)),
        long_ratio=0.5,
    )

    np.testing.assert_allclose(adjusted, [-1.5, -0.5, 0.5, 1.5])
    assert (adjusted > 0).sum() == 2


def test_long_ratio_ignores_ineligible_values_and_preserves_missing():
    values = pd.Series([1.0, 100.0, 3.0, np.nan])
    eligible = pd.Series([True, False, True, True])
    adjusted = _adjust_alpha_by_long_ratio(values, eligible, long_ratio=0.5)

    np.testing.assert_allclose(adjusted[[0, 1, 2]], [-1.0, 100.0, 1.0])
    assert np.isnan(adjusted[3])


@pytest.mark.parametrize("long_ratio", [-0.01, 1.01])
def test_long_ratio_rejects_out_of_range_values(long_ratio):
    with pytest.raises(ValueError, match="between 0 and 1"):
        _adjust_alpha_by_long_ratio(
            pd.Series([1.0, 2.0]),
            pd.Series(np.ones(2, dtype=bool)),
            long_ratio=long_ratio,
        )
